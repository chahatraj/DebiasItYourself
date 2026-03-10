#!/usr/bin/env python3
"""
Bias-Aware PEFT: Parameter-Efficient Fine-Tuning with Interchange Ablation Layer Selection

Paper : "Bias-Aware Parameter-Efficient Fine-Tuning for Debiasing Large Language Models"
        ACL 2025 Long (aclanthology.org/2025.acl-long.717)

Method (Sections 3–4):
  Step 1 – Interchange ablation layer selection (Section 4.2.1 / Eq. 2):
    For each layer l and each (gold, biased) sample pair:
      Replace the biased sample's hidden state at layer l with the gold sample's.
      Run remaining layers and measure KL(bias_dist || intervened_dist).
    Select the layer with the maximum average KL divergence.
    Apply LoRA only to that layer's o_proj and mlp.down_proj.

  Step 2 – Three-term training objective (Section 4.2.2):
    L = α·L_LM + β·L_bal1 + γ·L_bal2
    where:
      L_LM   = standard language-modeling NLL loss on the unbiased (anti) response
      L_bal1 = ||h_correct − mean(h_correct)||₂ + ||h_biased − mean(h_biased)||₂
               (within-class compactness of option logits)
      L_bal2 = −||mean(h_correct) − mean(h_biased)||₂
               (repulsion between correct and biased logit centroids)
    Default weights: α=β=γ=1.0

Official hyperparameters:
  learning_rate : 5e-5  (Adam)
  epochs        : 3
  batch_size    : 16
  LoRA r/alpha  : 8 / 16
  target_modules: [o_proj, mlp.down_proj] of the selected layer

Dataset-agnostic adaptations:
  - BBQ      : gold = label, biased = target_loc; BBQDataset + compute_bbq_bias_losses
  - CrowS    : gold = anti sentence, biased = stereo sentence; PairPeftDataset
  - StereoSet: same as CrowS
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
AVAILABLE_MODELS: Dict[str, str] = {
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
}
CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None):
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return AVAILABLE_MODELS["llama_8b"]
    raise ValueError(f"Unsupported --model alias: {model_alias}")


def _safe_str(x) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def get_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def prepare_kbit_model_for_training(model):
    """
    Enable k-bit training with non-reentrant checkpointing when available.
    This avoids known instability around default reentrant checkpoint backward.
    """
    try:
        return prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model)
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()
        return model


def disable_gradient_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_disable"):
        model.model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        model.config.use_cache = True


def format_bbq_prompt(row):
    return f"{row['context']}\n{row['question']}\nA. {row['ans0']}\nB. {row['ans1']}\nC. {row['ans2']}\nAnswer:"


class BBQDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = format_bbq_prompt(row)
        answer = ["A", "B", "C"][int(row["label"])]
        full_text = prompt + " " + answer

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"])),
        }


def compute_bbq_bias_losses(logits, labels, option_tokens):
    option_logits = logits[:, option_tokens]

    h_c = []
    h_n = []
    for i in range(logits.shape[0]):
        true_label = int(labels[i].item())
        for opt_idx in range(3):
            if opt_idx == true_label:
                h_c.append(option_logits[i, opt_idx])
            else:
                h_n.append(option_logits[i, opt_idx])

    if not h_c or not h_n:
        z = torch.tensor(0.0, device=logits.device)
        return z, z

    h_c = torch.stack(h_c)
    h_n = torch.stack(h_n)

    l_bal1 = torch.norm(h_c - h_c.mean(), p=2) + torch.norm(h_n - h_n.mean(), p=2)
    l_bal2 = -torch.norm(h_c.mean() - h_n.mean(), p=2)
    return l_bal1, l_bal2


def compute_pair_bias_losses(model, anti_ids, anti_mask, stereo_ids, stereo_mask):
    anti_out = model(input_ids=anti_ids, attention_mask=anti_mask)
    stereo_out = model(input_ids=stereo_ids, attention_mask=stereo_mask)

    anti_logits = anti_out.logits[:, -1, :]
    stereo_logits = stereo_out.logits[:, -1, :]

    anti_scores = anti_logits.max(dim=-1).values
    stereo_scores = stereo_logits.max(dim=-1).values

    l_bal1 = torch.norm(anti_scores - anti_scores.mean(), p=2) + torch.norm(stereo_scores - stereo_scores.mean(), p=2)
    l_bal2 = -torch.norm(anti_scores.mean() - stereo_scores.mean(), p=2)
    return l_bal1, l_bal2


def find_target_layer(model, tokenizer, df: pd.DataFrame, text_col: Optional[str] = None, num_samples=10):
    """
    Interchange ablation layer selection (official method, Section 4.2.1 / Eq. 2).

    For each sample pair (gold, biased), at each layer l we replace the biased sample's
    hidden state at layer l with the gold sample's hidden state, run the remaining layers,
    and measure the KL divergence between the original and intervened final output distributions.
    The layer with the maximum average KL divergence is selected as the target.
    """
    num_layers = model.config.num_hidden_layers

    # Build (gold_text, biased_text) pairs from the dataframe
    if text_col is not None:
        # Pair dataset (CrowS-Pairs / StereoSet): anti = gold, stereo = biased
        anti_col = "anti_sentence" if "anti_sentence" in df.columns else text_col
        stereo_col = "stereo_sentence" if "stereo_sentence" in df.columns else text_col
        sample_df = df.sample(n=min(num_samples, len(df)), random_state=SEED)
        pairs = [(row[anti_col], row[stereo_col]) for _, row in sample_df.iterrows()]
    else:
        # BBQ: gold prompt = prompt + correct answer token, biased prompt = prompt + bias label token
        sample_df = df.sample(n=min(num_samples, len(df)), random_state=SEED)
        pairs = []
        for _, row in sample_df.iterrows():
            prompt = format_bbq_prompt(row)
            gold_ans = ["A", "B", "C"][int(row["label"])]
            # Use target_loc if available for biased answer, else pick a different option
            tgt = row.get("target_loc")
            if tgt is not None and int(tgt) != int(row["label"]):
                bias_ans = ["A", "B", "C"][int(tgt)]
            else:
                opts = [0, 1, 2]
                opts.remove(int(row["label"]))
                bias_ans = ["A", "B", "C"][opts[0]]
            pairs.append((prompt + " " + gold_ans, prompt + " " + bias_ans))

    kl_scores = [0.0] * num_layers
    n_valid = 0

    for gold_text, biased_text in tqdm(pairs, desc="Finding target layer (interchange ablation)"):
        gold_enc = tokenizer(gold_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        bias_enc = tokenizer(biased_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)

        with torch.no_grad():
            gold_out = model(**gold_enc, output_hidden_states=True)
            bias_out = model(**bias_enc, output_hidden_states=True)

        # Original biased final logits distribution
        bias_logits = bias_out.logits[0, -1, :]
        bias_probs = torch.softmax(bias_logits.float(), dim=-1)

        gold_hidden = gold_out.hidden_states  # tuple of (num_layers+1) tensors

        for layer_idx in range(num_layers):
            # Replace biased hidden state at layer_idx with gold hidden state (if seq lengths match)
            gold_h = gold_hidden[layer_idx + 1]  # [1, seq_gold, d]
            bias_h = bias_out.hidden_states[layer_idx + 1]  # [1, seq_bias, d]

            # Use last-token replacement (most informative position)
            if gold_h.shape[1] == 0 or bias_h.shape[1] == 0:
                continue

            # Build intervened hidden states: replace last token of bias at this layer with gold's last token
            intervened_h = bias_h.clone()
            intervened_h[0, -1, :] = gold_h[0, -1, :]

            # Run remaining layers from layer_idx+1 onward
            h = intervened_h
            try:
                for subsequent_layer in model.model.layers[layer_idx + 1:]:
                    h = subsequent_layer(h)[0]
                h = model.model.norm(h)
                intervened_logits = model.lm_head(h)[0, -1, :]
                intervened_probs = torch.softmax(intervened_logits.float(), dim=-1)

                # KL(bias_probs || intervened_probs)
                kl = F.kl_div(intervened_probs.log(), bias_probs, reduction="sum").item()
                kl_scores[layer_idx] += abs(kl)
            except Exception:
                continue

        n_valid += 1

    if n_valid == 0:
        # Fallback: pick middle layer
        return num_layers // 2

    avg_kl = [kl_scores[i] / n_valid for i in range(num_layers)]
    return int(np.argmax(avg_kl))


def identify_crows_stereotype(row: pd.Series) -> Tuple[str, str, str]:
    sent_more = _safe_str(row.get("sent_more", "")).strip()
    sent_less = _safe_str(row.get("sent_less", "")).strip()
    direction = _safe_str(row.get("stereo_antistereo", "")).strip().lower()

    if direction == "stereo":
        return sent_more, sent_less, "stereo"
    if direction == "antistereo":
        return sent_less, sent_more, "antistereo"
    return sent_more, sent_less, "unknown"


def create_crows_training_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        stereo, anti, direction = identify_crows_stereotype(row)
        if not stereo or not anti:
            continue
        rows.append(
            {
                "pair_id": int(idx),
                "stereo_sentence": stereo,
                "anti_sentence": anti,
                "direction_label": direction,
                "bias_type": _safe_str(row.get("bias_type", "")),
            }
        )
    return pd.DataFrame(rows)


def create_stereoset_training_df(examples: List[Dict]) -> pd.DataFrame:
    rows = []
    for i, ex in enumerate(examples):
        rows.append(
            {
                "pair_id": i,
                "stereo_sentence": ex["stereotype"],
                "anti_sentence": ex["anti"],
                "unrelated_sentence": ex.get("unrelated", ""),
                "split": ex.get("split", ""),
                "bias_type": ex.get("bias_type", ""),
            }
        )
    return pd.DataFrame(rows)


class PairPeftDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        anti_enc = self.tokenizer(
            row["anti_sentence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        stereo_enc = self.tokenizer(
            row["stereo_sentence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "anti_input_ids": anti_enc["input_ids"].squeeze(0),
            "anti_attention_mask": anti_enc["attention_mask"].squeeze(0),
            "stereo_input_ids": stereo_enc["input_ids"].squeeze(0),
            "stereo_attention_mask": stereo_enc["attention_mask"].squeeze(0),
        }


def warmup_cuda(device):
    if not torch.cuda.is_available():
        return
    torch.cuda.init()
    with torch.no_grad():
        a = torch.randn((128, 128), device=device, dtype=torch.float16)
        b = torch.randn((128, 128), device=device, dtype=torch.float16)
        _ = a @ b
    torch.cuda.synchronize()


def run_step_with_retry(model, optimizer, anti_ids, anti_mask, stereo_ids, stereo_mask, alpha, beta, gamma):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            anti_outputs = model(input_ids=anti_ids, attention_mask=anti_mask, labels=anti_ids)
            lm_loss = anti_outputs.loss
            l_bal1, l_bal2 = compute_pair_bias_losses(model, anti_ids, anti_mask, stereo_ids, stereo_mask)
            loss = alpha * lm_loss + beta * l_bal1 + gamma * l_bal2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss
        except RuntimeError as e:
            err_msg = str(e).lower()
            transient = any(
                needle in err_msg
                for needle in (
                    "cublas_status_not_initialized",
                    "cuda error",
                    "out of memory",
                    "outofmemory",
                    "cudnn",
                )
            )
            if (not transient) or attempt == max_attempts - 1:
                raise
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(0.2)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model": "llama_8b",
            "category": "Age",
            "data_dir": "/scratch/craj/diy/outputs/2_base_models/bbq/llama_8b",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models",
            "epochs": 3,
            "batch_size": 4,
            "lr": 5e-5,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "test_size": 0.9,
            "max_length": 256,
        },
        "crowspairs": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models_crowspairs",
            "model_tag": "crowspairs_all",
            "epochs": 3,
            "batch_size": 4,
            "lr": 5e-5,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "test_size": 0.9,
            "max_length": 256,
        },
        "stereoset": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models_stereoset",
            "model_tag": "stereoset_all",
            "epochs": 3,
            "batch_size": 4,
            "lr": 5e-5,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 1.0,
            "test_size": 0.1,
            "max_length": 256,
        },
    }

    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bias-aware PEFT training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)

    parser.add_argument("--category", type=str, default=None, choices=CATEGORIES)
    parser.add_argument("--data_dir", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    if args.model_tag is None:
        args.model_tag = args.category if args.dataset == "bbq" else f"{args.dataset}_all"

    model_name = _resolve_model_name(args.model_name, args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_quantization_config(args.use_4bit),
        device_map="auto",
        torch_dtype=torch.float16,
        token=args.hf_token,
    )

    if args.dataset == "bbq":
        if not args.model_tag:
            args.model_tag = args.category

        df = pd.read_csv(f"{args.data_dir}/bbq_preds_llama_8b_{args.category}.csv")
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=SEED)

        target_layer = find_target_layer(model, tokenizer, train_df) if args.target_layer is None else int(args.target_layer)
        print(f"Target layer: {target_layer}")

        model = prepare_kbit_model_for_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                f"model.layers.{target_layer}.self_attn.o_proj",
                f"model.layers.{target_layer}.mlp.down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        disable_gradient_checkpointing(model)
        model.print_trainable_parameters()
        warmup_cuda(model.device)

        train_loader = DataLoader(BBQDataset(train_df, tokenizer, max_length=args.max_length), batch_size=args.batch_size, shuffle=True)
        option_tokens = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in ["A", "B", "C"]]

        optimizer = Adam(model.parameters(), lr=args.lr)
        model.train()

        for epoch in range(args.epochs):
            total = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["label"].to(model.device)

                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                        lm_loss = outputs.loss
                        logits = outputs.logits[:, -2, :]

                        l_bal1, l_bal2 = compute_bbq_bias_losses(logits, labels, option_tokens)
                        loss = args.alpha * lm_loss + args.beta * l_bal1 + args.gamma * l_bal2

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        break
                    except RuntimeError as e:
                        err_msg = str(e).lower()
                        transient = any(
                            needle in err_msg
                            for needle in (
                                "cublas_status_not_initialized",
                                "cuda error",
                                "out of memory",
                                "outofmemory",
                                "cudnn",
                            )
                        )
                        if (not transient) or attempt == max_attempts - 1:
                            raise
                        optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        time.sleep(0.2)

                total += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch + 1}: Avg Loss = {total / max(1, len(train_loader)):.4f}")

    elif args.dataset == "crowspairs":
        raw_df = pd.read_csv(args.data_path)
        if args.bias_type is not None:
            if "bias_type" not in raw_df.columns:
                raise ValueError("--bias_type provided but dataset has no 'bias_type' column")
            raw_df = raw_df[raw_df["bias_type"].astype(str) == args.bias_type].copy()

        df = create_crows_training_df(raw_df)
        if df.empty:
            raise ValueError("No valid CrowS-Pairs rows after preprocessing")
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=SEED)

        target_layer = find_target_layer(model, tokenizer, train_df, text_col="anti_sentence") if args.target_layer is None else int(args.target_layer)
        print(f"Target layer: {target_layer}")

        model = prepare_kbit_model_for_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                f"model.layers.{target_layer}.self_attn.o_proj",
                f"model.layers.{target_layer}.mlp.down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        disable_gradient_checkpointing(model)
        model.print_trainable_parameters()
        warmup_cuda(model.device)

        train_loader = DataLoader(PairPeftDataset(train_df, tokenizer, max_length=args.max_length), batch_size=args.batch_size, shuffle=True)
        optimizer = Adam(model.parameters(), lr=args.lr)
        model.train()

        for epoch in range(args.epochs):
            total = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                anti_ids = batch["anti_input_ids"].to(model.device)
                anti_mask = batch["anti_attention_mask"].to(model.device)
                stereo_ids = batch["stereo_input_ids"].to(model.device)
                stereo_mask = batch["stereo_attention_mask"].to(model.device)

                loss = run_step_with_retry(
                    model,
                    optimizer,
                    anti_ids,
                    anti_mask,
                    stereo_ids,
                    stereo_mask,
                    args.alpha,
                    args.beta,
                    args.gamma,
                )

                total += float(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch + 1}: Avg Loss = {total / max(1, len(train_loader)):.4f}")

    else:
        shared_mod = _load_module(
            "paper_baselines_shared_for_peft_train_stereo",
            BASE_DIR / "paper_baselines_shared.py",
            add_to_syspath=BASE_DIR,
        )
        stereo_common = shared_mod.get_dataset_common("stereoset")
        data = stereo_common.load_stereoset_data(args.data_path)
        examples = stereo_common.flatten_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
        if not examples:
            raise ValueError("No StereoSet examples after filtering")

        df = create_stereoset_training_df(examples)
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=SEED)

        target_layer = find_target_layer(model, tokenizer, train_df, text_col="anti_sentence") if args.target_layer is None else int(args.target_layer)
        print(f"Target layer: {target_layer}")

        model = prepare_kbit_model_for_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                f"model.layers.{target_layer}.self_attn.o_proj",
                f"model.layers.{target_layer}.mlp.down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        disable_gradient_checkpointing(model)
        model.print_trainable_parameters()
        warmup_cuda(model.device)

        train_loader = DataLoader(PairPeftDataset(train_df, tokenizer, max_length=args.max_length), batch_size=args.batch_size, shuffle=True)
        optimizer = Adam(model.parameters(), lr=args.lr)
        model.train()

        for epoch in range(args.epochs):
            total = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                anti_ids = batch["anti_input_ids"].to(model.device)
                anti_mask = batch["anti_attention_mask"].to(model.device)
                stereo_ids = batch["stereo_input_ids"].to(model.device)
                stereo_mask = batch["stereo_attention_mask"].to(model.device)

                loss = run_step_with_retry(
                    model,
                    optimizer,
                    anti_ids,
                    anti_mask,
                    stereo_ids,
                    stereo_mask,
                    args.alpha,
                    args.beta,
                    args.gamma,
                )

                total += float(loss.item())
                pbar.set_postfix(loss=total / max(1, (pbar.n + 1)))

    model_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_path = os.path.join(args.output_dir, f"test_{args.model_tag}.csv")
    test_df.to_csv(test_path, index=False)

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "model_name": model_name,
        "model_tag": args.model_tag,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "target_layer": int(target_layer),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "max_length": args.max_length,
    }
    if args.dataset == "bbq":
        config.update({"category": args.category, "data_dir": args.data_dir})
    else:
        config.update({"data_path": args.data_path, "bias_type": args.bias_type})
        if args.dataset == "stereoset":
            config.update({"split": args.split, "limit": args.limit})

    with open(os.path.join(args.output_dir, f"config_{args.model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {model_dir}")
    print(f"Test data saved: {test_path}")


if __name__ == "__main__":
    main()
