#!/usr/bin/env python3
"""
Preference-Aware DPO for Social Bias Mitigation (SBM)

Paper : "SBM: Social Bias Mitigation for LLMs via Preference-Aware DPO"
        Proceedings of WWW 2025 (arXiv:2412.16155)
GitHub: https://github.com/KID-22/LLM-SBM

Method (Section 3):
  Standard DPO re-weighted by per-pair preference intensity delta:
    L = -mean( delta^alpha * log_sigmoid(beta*(r_chosen - r_rejected)) )
  where r = beta*(log π_policy - log π_ref) are implicit rewards.
  Delta is computed from the base model's predicted probability gap between the
  biased and unbiased answer choices, normalised to [0.01, 1].
  LR scheduler: cosine with linear warmup (warmup_ratio=0.1).

Official hyperparameters (yamls/llama3/lora/msmarco_dpo/msmarco_dpo.yaml):
  learning_rate : 5e-6
  beta          : 0.1
  weight_alpha  : 2.0   (our `alpha`)
  num_epochs    : 3
  batch_size    : 4 per device (grad_accum=8 → effective=32)
  LoRA targets  : "all" (we use q/v/k/o/gate/up/down_proj)

Dataset-agnostic adaptations:
  - BBQ    : delta from base-model option-probability gap
  - CrowS  : delta from log-prob margin between stereo / anti sentences
  - StereoSet: same as CrowS
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("bbq", "crowspairs", "stereoset")

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


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_base_model(model_arg: str) -> str:
    if "/" in model_arg:
        return model_arg
    if model_arg not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model '{model_arg}'. Supported aliases: {sorted(AVAILABLE_MODELS)}")
    return AVAILABLE_MODELS[model_arg]


def _default_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _safe_str(x) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _identify_crows_stereo_pair(row: pd.Series) -> Tuple[str, str, str]:
    sent_more = _safe_str(row.get("sent_more", "")).strip()
    sent_less = _safe_str(row.get("sent_less", "")).strip()
    direction = _safe_str(row.get("stereo_antistereo", "")).strip().lower()

    if direction == "stereo":
        return sent_more, sent_less, "stereo"
    if direction == "antistereo":
        return sent_less, sent_more, "antistereo"
    return sent_more, sent_less, "unknown"


class PairDPODataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256, use_prompt_lengths: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_prompt_lengths = use_prompt_lengths

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chosen_text = str(row["chosen_text"])
        rejected_text = str(row["rejected_text"])

        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_len = 0
        if self.use_prompt_lengths:
            prompt_text = str(row.get("prompt_text", ""))
            prompt_enc = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_len = int(prompt_enc["input_ids"].shape[1])

        out = {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "delta": torch.tensor(float(row.get("delta", 1.0)), dtype=torch.float32),
            "prompt_length": torch.tensor(prompt_len, dtype=torch.long),
        }

        if "ref_chosen_logps" in row.index:
            out["ref_chosen_logps"] = torch.tensor(float(row["ref_chosen_logps"]), dtype=torch.float32)
            out["ref_rejected_logps"] = torch.tensor(float(row["ref_rejected_logps"]), dtype=torch.float32)

        return out


def compute_sequence_logps(model, input_ids, attention_mask, prompt_lengths: Optional[torch.Tensor] = None):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    if prompt_lengths is not None:
        seq_len = shift_mask.shape[1]
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        prefix = (prompt_lengths.to(input_ids.device) - 1).clamp(min=0, max=seq_len).unsqueeze(1)
        shift_mask = shift_mask * (pos >= prefix).float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * shift_mask).sum(dim=-1)


def _is_transient_cuda_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    needles = (
        "cuda error",
        "cublas_status_not_initialized",
        "out of memory",
        "outofmemory",
        "cudnn",
    )
    return any(x in msg for x in needles)


def _recover_cuda(optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def preference_aware_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    delta,
    beta=0.1,
    alpha=2.0,
):
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    logits = chosen_rewards - rejected_rewards
    dpo_loss = -F.logsigmoid(logits)
    weighted_loss = torch.pow(delta, alpha) * dpo_loss

    return weighted_loss


def compute_reference_logps(
    model,
    pref_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    use_prompt_lengths: bool = False,
):
    model.eval()
    loader = DataLoader(
        PairDPODataset(pref_df, tokenizer, max_length=max_length, use_prompt_lengths=use_prompt_lengths),
        batch_size=batch_size,
        shuffle=False,
    )

    ref_chosen, ref_rejected = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing reference logps"):
            chosen_ids = batch["chosen_input_ids"].to(model.device)
            chosen_mask = batch["chosen_attention_mask"].to(model.device)
            rejected_ids = batch["rejected_input_ids"].to(model.device)
            rejected_mask = batch["rejected_attention_mask"].to(model.device)
            prompt_lens = batch["prompt_length"].to(model.device) if use_prompt_lengths else None

            chosen_lp = compute_sequence_logps(model, chosen_ids, chosen_mask, prompt_lens)
            rejected_lp = compute_sequence_logps(model, rejected_ids, rejected_mask, prompt_lens)

            ref_chosen.extend(chosen_lp.detach().cpu().tolist())
            ref_rejected.extend(rejected_lp.detach().cpu().tolist())

    return ref_chosen, ref_rejected


def train_preference_aware_dpo(
    model,
    train_df: pd.DataFrame,
    tokenizer,
    epochs: int,
    batch_size: int,
    lr: float,
    beta: float,
    alpha: float,
    max_length: int,
    use_prompt_lengths: bool,
    warmup_ratio: float = 0.1,
):
    loader = DataLoader(
        PairDPODataset(train_df, tokenizer, max_length=max_length, use_prompt_lengths=use_prompt_lengths),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = AdamW(model.parameters(), lr=lr)

    # Cosine LR schedule with linear warmup — matches official: lr_scheduler_type: cosine, warmup_ratio: 0.1
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        skipped = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            try:
                chosen_ids = batch["chosen_input_ids"].to(model.device)
                chosen_mask = batch["chosen_attention_mask"].to(model.device)
                rejected_ids = batch["rejected_input_ids"].to(model.device)
                rejected_mask = batch["rejected_attention_mask"].to(model.device)
                prompt_lens = batch["prompt_length"].to(model.device) if use_prompt_lengths else None

                delta = batch["delta"].to(model.device)
                ref_chosen = batch["ref_chosen_logps"].to(model.device)
                ref_rejected = batch["ref_rejected_logps"].to(model.device)

                policy_chosen = compute_sequence_logps(model, chosen_ids, chosen_mask, prompt_lens)
                policy_rejected = compute_sequence_logps(model, rejected_ids, rejected_mask, prompt_lens)

                loss = preference_aware_dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    delta,
                    beta=beta,
                    alpha=alpha,
                ).mean()

                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                total_loss += float(loss.item())
                n_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", skipped=skipped)
            except RuntimeError as err:
                if not _is_transient_cuda_error(err):
                    raise
                skipped += 1
                _recover_cuda(optimizer)
                pbar.set_postfix(loss=f"{(total_loss / max(1, n_batches)):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", skipped=skipped)
                continue

        avg = total_loss / max(1, n_batches)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg:.4f}")
        if skipped:
            print(f"[warn] epoch {epoch + 1}: skipped {skipped} CUDA-fragile batches")


def _score_texts_logprob(model, tokenizer, texts: List[str], batch_size: int = 4) -> List[float]:
    out: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            ids = enc["input_ids"].to(model.device)
            mask = enc["attention_mask"].to(model.device)
            lp = compute_sequence_logps(model, ids, mask, prompt_lengths=None)
            out.extend([float(x) for x in lp.detach().cpu().tolist()])
    return out


def _build_bbq_preference_df(data_dir: str, bbq_dir: str, meta_file: str, category: str) -> pd.DataFrame:
    pred_path = os.path.join(data_dir, f"bbq_preds_llama_8b_{category}.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Base prediction CSV not found: {pred_path}")
    df = pd.read_csv(pred_path)

    bbq_path = os.path.join(bbq_dir, f"{category}.jsonl")
    if not os.path.exists(bbq_path):
        raise FileNotFoundError(f"BBQ category file not found: {bbq_path}")

    with open(bbq_path, "r", encoding="utf-8") as f:
        bbq_rows = [json.loads(line) for line in f if line.strip()]
    bbq_df = pd.DataFrame(bbq_rows)

    meta_df = pd.read_csv(meta_file)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]

    merged = df.merge(
        bbq_df[["example_id", "question_polarity"]].drop_duplicates("example_id"),
        on="example_id",
        how="left",
    )
    merged = merged.merge(
        meta_df[["example_id", "target_loc"]].drop_duplicates("example_id"),
        on="example_id",
        how="left",
    )
    merged = merged.dropna(subset=["target_loc"]).copy()
    merged["target_loc"] = merged["target_loc"].astype(int)

    rows = []
    for _, row in merged.iterrows():
        probs = row.get("option_probs")
        if isinstance(probs, str):
            probs = ast.literal_eval(probs)
        if not isinstance(probs, dict):
            continue

        prob_list = [float(probs.get("A", 0.0)), float(probs.get("B", 0.0)), float(probs.get("C", 0.0))]
        correct_idx = int(row["label"])
        target_idx = int(row["target_loc"])

        if correct_idx == target_idx:
            continue

        delta_raw = prob_list[target_idx] - prob_list[correct_idx]
        if delta_raw <= -0.5:
            continue

        prompt = (
            f"{row['context']}\n"
            f"{row['question']}\n"
            f"A. {row['ans0']}\n"
            f"B. {row['ans1']}\n"
            f"C. {row['ans2']}\n"
            "Answer:"
        )
        chosen_letter = ["A", "B", "C"][correct_idx]
        rejected_letter = ["A", "B", "C"][target_idx]
        delta = max(0.01, (delta_raw + 1.0) / 2.0)

        rows.append(
            {
                "example_id": row["example_id"],
                "category": category,
                "prompt_text": prompt,
                "chosen_text": f"{prompt} {chosen_letter}",
                "rejected_text": f"{prompt} {rejected_letter}",
                "delta": float(delta),
            }
        )

    pref_df = pd.DataFrame(rows)
    if pref_df.empty:
        raise ValueError(f"No DPO preference pairs generated for BBQ category={category}")
    return pref_df


def _build_crows_preference_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        stereo, anti, direction = _identify_crows_stereo_pair(row)
        if not stereo or not anti:
            continue

        delta = 1.0
        if "stereo_logprob" in row.index and "anti_logprob" in row.index:
            try:
                margin = float(row["stereo_logprob"]) - float(row["anti_logprob"])
                delta = max(0.01, min(1.0, (margin + 10.0) / 20.0))
            except Exception:
                delta = 1.0

        rows.append(
            {
                "pair_id": int(idx),
                "direction_label": direction,
                "bias_type": _safe_str(row.get("bias_type", "")),
                "chosen_text": anti,
                "rejected_text": stereo,
                "delta": float(delta),
            }
        )

    pref_df = pd.DataFrame(rows)
    if pref_df.empty:
        raise ValueError("No valid preference pairs created from CrowS-Pairs data")
    return pref_df


def _build_stereoset_preference_df(examples: List[Dict], model, tokenizer, batch_size: int) -> pd.DataFrame:
    stereo = [str(x["stereotype"]) for x in examples]
    anti = [str(x["anti"]) for x in examples]

    stereo_lp = _score_texts_logprob(model, tokenizer, stereo, batch_size=batch_size)
    anti_lp = _score_texts_logprob(model, tokenizer, anti, batch_size=batch_size)

    rows = []
    for i, ex in enumerate(examples):
        margin = stereo_lp[i] - anti_lp[i]
        delta = max(0.01, min(1.0, (margin + 10.0) / 20.0))
        rows.append(
            {
                "pair_id": i,
                "example_id": ex.get("example_id"),
                "split": ex.get("split", ""),
                "bias_type": ex.get("bias_type", ""),
                "chosen_text": str(ex["anti"]),
                "rejected_text": str(ex["stereotype"]),
                "delta": float(delta),
            }
        )

    pref_df = pd.DataFrame(rows)
    if pref_df.empty:
        raise ValueError("No valid preference pairs created from StereoSet examples")
    return pref_df


def _build_train_test(pref_df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0:
        return pref_df.reset_index(drop=True), pref_df.iloc[0:0].copy()
    train_df, test_df = train_test_split(pref_df, test_size=test_size, random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "data_dir": "/scratch/craj/diy/outputs/2_base_models/bbq/llama_8b",
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "category": "Age",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models",
            "batch_size": 2,
            "max_length": 256,
            "test_size": 0.9,
        },
        "crowspairs": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models_crowspairs",
            "model_tag": "crowspairs_all",
            "batch_size": 2,
            "max_length": 256,
            "test_size": 0.9,
        },
        "stereoset": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models_stereoset",
            "model_tag": "stereoset_all",
            "batch_size": 2,
            "max_length": 256,
            "test_size": 0.1,
        },
    }

    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.model_tag is None:
        if args.dataset == "bbq":
            args.model_tag = args.category
        elif args.dataset == "crowspairs":
            args.model_tag = "crowspairs_all"
        else:
            args.model_tag = "stereoset_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def _prepare_model_and_tokenizer(base_model: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_default_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    return model, tokenizer


def main() -> None:
    args = parse_args()
    _apply_defaults(args)

    base_model_name = args.base_model or _resolve_base_model(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = _prepare_model_and_tokenizer(base_model_name, args.hf_token)

    if args.dataset == "bbq":
        pref_df = _build_bbq_preference_df(args.data_dir, args.bbq_dir, args.meta_file, args.category)
        use_prompt_lengths = True
    elif args.dataset == "crowspairs":
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"CrowS-Pairs file not found: {args.data_path}")
        crows_df = pd.read_csv(args.data_path)
        if args.bias_type is not None:
            if "bias_type" not in crows_df.columns:
                raise ValueError("--bias_type provided but dataset has no 'bias_type' column")
            crows_df = crows_df[crows_df["bias_type"].astype(str) == args.bias_type].copy()
        if args.limit:
            crows_df = crows_df.iloc[: args.limit].copy()
        pref_df = _build_crows_preference_df(crows_df)
        use_prompt_lengths = False
    else:
        shared_mod = _load_module(
            "paper_baselines_shared_for_dpo_train_stereo",
            BASE_DIR / "paper_baselines_shared.py",
            add_to_syspath=BASE_DIR,
        )
        stereo_common = shared_mod.get_dataset_common("stereoset")
        data = stereo_common.load_stereoset_data(args.data_path)
        examples = stereo_common.flatten_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
        if not examples:
            raise ValueError("No StereoSet examples after filtering")
        pref_df = _build_stereoset_preference_df(examples, model, tokenizer, batch_size=args.batch_size)
        use_prompt_lengths = False

    train_df, test_df = _build_train_test(pref_df, args.test_size)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Compute reference log-probs from the frozen base model BEFORE applying LoRA.
    # DPO requires π_ref to be the initial (unmodified) model; using the LoRA-wrapped
    # model here would make chosen/rejected reference log-probs identical to policy
    # log-probs at step 0, collapsing the DPO objective.
    ref_chosen, ref_rejected = compute_reference_logps(
        model,
        train_df,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_prompt_lengths=use_prompt_lengths,
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    train_df = train_df.reset_index(drop=True)
    train_df["ref_chosen_logps"] = ref_chosen
    train_df["ref_rejected_logps"] = ref_rejected

    train_preference_aware_dpo(
        model,
        train_df,
        tokenizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        alpha=args.alpha,
        max_length=args.max_length,
        use_prompt_lengths=use_prompt_lengths,
    )

    model_path = os.path.join(args.output_dir, f"model_{args.model_tag}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    test_df.to_csv(os.path.join(args.output_dir, f"test_{args.model_tag}.csv"), index=False)

    cfg = {
        "dataset": args.dataset,
        "model": args.model,
        "base_model": base_model_name,
        "model_tag": args.model_tag,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "beta": args.beta,
        "alpha": args.alpha,
        "test_size": args.test_size,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_pairs": int(len(train_df)),
        "test_pairs": int(len(test_df)),
    }
    if args.dataset == "bbq":
        cfg.update({"category": args.category, "data_dir": args.data_dir, "bbq_dir": args.bbq_dir})
    elif args.dataset == "crowspairs":
        cfg.update({"data_path": args.data_path, "bias_type": args.bias_type})
    else:
        cfg.update({"data_path": args.data_path, "split": args.split, "bias_type": args.bias_type, "limit": args.limit})

    with open(os.path.join(args.output_dir, f"config_{args.model_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
