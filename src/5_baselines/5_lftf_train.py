#!/usr/bin/env python3
"""Dataset-agnostic LFTF training entrypoint."""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import Adam
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


def _safe_str(x) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    return str(x)


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return AVAILABLE_MODELS["llama_8b"]
    raise ValueError(f"Unsupported --model alias: {model_alias}")


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def calculate_bmi(model, tokenizer, num_layers: int, prompts: List[str], max_length: int = 256):
    layer_diffs = {i: [] for i in range(num_layers)}

    for prompt in tqdm(prompts, desc="Calculating BMI"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hs = outputs.hidden_states
        for i in range(num_layers):
            h_i = hs[i][0, -1, :].float()
            h_ip1 = hs[i + 1][0, -1, :].float()
            cos_sim = F.cosine_similarity(h_i.unsqueeze(0), h_ip1.unsqueeze(0)).item()
            layer_diffs[i].append(1 - cos_sim)

    bmi_scores = {i: float(np.mean(layer_diffs[i])) if layer_diffs[i] else 0.0 for i in range(num_layers)}
    filtered = {k: v for k, v in bmi_scores.items() if k > 0}
    optimal_layer = int(max(filtered, key=filtered.get)) if filtered else 0
    return bmi_scores, optimal_layer


def create_bbq_prompts_for_bmi(num_samples: int = 100) -> List[str]:
    professions = [
        "nurse", "doctor", "teacher", "engineer", "secretary", "programmer",
        "chef", "lawyer", "accountant", "manager", "receptionist", "mechanic",
        "pilot", "firefighter", "librarian", "dentist", "therapist", "surgeon",
    ]
    templates = ["The {} smiled because", "The {} went home because", "The {} was tired because"]
    return [t.format(p) for p in professions for t in templates][:num_samples]


def create_bbq_training_prompts() -> List[str]:
    professions = [
        "nurse", "doctor", "teacher", "engineer", "secretary", "programmer",
        "chef", "lawyer", "accountant", "manager", "receptionist", "mechanic",
        "pilot", "firefighter", "librarian", "dentist", "therapist", "surgeon",
        "cashier", "janitor", "plumber", "electrician", "pharmacist", "veterinarian",
    ]
    templates = [
        "The {} smiled because",
        "The {} went home because",
        "The {} was tired because",
        "The {} got promoted because",
    ]
    return [t.format(p) for p in professions for t in templates]


def identify_crows_stereotype(row: pd.Series) -> Tuple[str, str, str]:
    sent_more = _safe_str(row.get("sent_more", "")).strip()
    sent_less = _safe_str(row.get("sent_less", "")).strip()
    direction = _safe_str(row.get("stereo_antistereo", "")).strip().lower()

    if direction == "stereo":
        return sent_more, sent_less, "stereo"
    if direction == "antistereo":
        return sent_less, sent_more, "antistereo"
    return sent_more, sent_less, "unknown"


def load_crows_pairs(data_path: str, bias_type: Optional[str], limit: Optional[int]) -> List[Dict[str, str]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CrowS-Pairs file not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"sent_more", "sent_less"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CrowS-Pairs file: {sorted(missing)}")

    if bias_type is not None:
        if "bias_type" not in df.columns:
            raise ValueError("--bias_type provided but dataset has no 'bias_type' column")
        df = df[df["bias_type"].astype(str) == bias_type].copy()

    rows = []
    for _, row in df.iterrows():
        stereo, anti, direction = identify_crows_stereotype(row)
        if not stereo or not anti:
            continue
        rows.append(
            {
                "stereotype": stereo,
                "anti": anti,
                "direction": direction,
                "bias_type": _safe_str(row.get("bias_type", "")),
            }
        )

    if limit:
        rows = rows[:limit]
    return rows


def load_stereoset_pairs(data_path: str, split: str, bias_type: Optional[str], limit: Optional[int]) -> List[Dict[str, str]]:
    mod_path = BASE_DIR / "paper_baselines_shared.py"
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    spec = importlib.util.spec_from_file_location("paper_baselines_shared_lftf_train", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stereo_common = mod.get_dataset_common("stereoset")
    data = stereo_common.load_stereoset_data(data_path)
    examples = stereo_common.flatten_examples(data, split=split, bias_type=bias_type, limit=limit)
    if not examples:
        raise ValueError("No StereoSet examples after filtering")

    return [
        {
            "stereotype": x["stereotype"],
            "anti": x["anti"],
            "split": x.get("split", ""),
            "bias_type": x.get("bias_type", ""),
        }
        for x in examples
    ]


def sequence_logprob(model, tokenizer, text: str, max_length: int = 256) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    out = model(**enc)
    logits = out.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = enc["input_ids"][:, 1:]
    shift_mask = enc["attention_mask"][:, 1:].float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_log_probs * shift_mask).sum(dim=1)
    return seq_logp.mean()


def lftf_train_bbq(model, tokenizer, prompts: List[str], he_id: int, she_id: int, args):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    best_diff = 1.0
    best_state = None

    for epoch in range(args.num_epochs):
        sum_p_he = 0.0
        sum_p_she = 0.0
        n = 0

        progress = tqdm(range(0, len(prompts), args.batch_size), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for i in progress:
            batch = prompts[i : i + args.batch_size]
            optimizer.zero_grad()

            batch_loss = 0.0
            batch_he = 0.0
            batch_she = 0.0

            for prompt in batch:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logits = model(**inputs).logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)

                p_he = probs[he_id]
                p_she = probs[she_id]

                loss = p_he + p_she
                batch_loss += loss

                batch_he += float(p_he.item())
                batch_she += float(p_she.item())
                sum_p_he += float(p_he.item())
                sum_p_she += float(p_she.item())
                n += 1

            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            optimizer.step()

            progress.set_postfix({"P(he)": f"{batch_he / max(1, len(batch)):.4f}", "P(she)": f"{batch_she / max(1, len(batch)):.4f}"})

        avg_he = sum_p_he / max(1, n)
        avg_she = sum_p_she / max(1, n)
        diff = abs(avg_he - avg_she)

        if diff < best_diff and (avg_he + avg_she) > 0.01:
            best_diff = diff
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if avg_he + avg_she < 0.001:
            if best_state:
                model.load_state_dict(best_state)
            break

    model.eval()
    return model, float(best_diff)


def lftf_train_crows(model, tokenizer, pairs: List[Dict[str, str]], args):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    best_gap = 1.0
    best_state = None

    for epoch in range(args.num_epochs):
        random.shuffle(pairs)
        sum_st = 0.0
        sum_an = 0.0
        n = 0

        progress = tqdm(range(0, len(pairs), args.batch_size), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for i in progress:
            batch = pairs[i : i + args.batch_size]
            optimizer.zero_grad()

            batch_loss = 0.0
            batch_st = 0.0
            batch_an = 0.0

            for item in batch:
                st_inputs = tokenizer(item["stereotype"], return_tensors="pt", truncation=True, max_length=256).to(model.device)
                an_inputs = tokenizer(item["anti"], return_tensors="pt", truncation=True, max_length=256).to(model.device)

                st_logits = model(**st_inputs).logits[0, -1, :]
                an_logits = model(**an_inputs).logits[0, -1, :]

                st_score = torch.max(torch.softmax(st_logits, dim=0))
                an_score = torch.max(torch.softmax(an_logits, dim=0))

                loss = st_score + an_score
                batch_loss += loss

                batch_st += float(st_score.item())
                batch_an += float(an_score.item())
                sum_st += float(st_score.item())
                sum_an += float(an_score.item())
                n += 1

            batch_loss = batch_loss / max(1, len(batch))
            batch_loss.backward()
            optimizer.step()

            progress.set_postfix({"S": f"{batch_st / max(1, len(batch)):.4f}", "A": f"{batch_an / max(1, len(batch)):.4f}"})

        avg_st = sum_st / max(1, n)
        avg_an = sum_an / max(1, n)
        gap = abs(avg_st - avg_an)

        if gap < best_gap and (avg_st + avg_an) > 0.01:
            best_gap = gap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if avg_st + avg_an < 0.001:
            if best_state:
                model.load_state_dict(best_state)
            break

    model.eval()
    return model, float(best_gap)


def lftf_train_stereoset(model, tokenizer, pairs: List[Dict[str, str]], args):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    best_state = None
    best_gap = float("inf")

    for epoch in range(args.num_epochs):
        random.shuffle(pairs)
        pbar = tqdm(range(0, len(pairs), args.batch_size), desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for i in pbar:
            batch = pairs[i : i + args.batch_size]
            optimizer.zero_grad()

            losses = []
            gaps = []
            for item in batch:
                stereo_lp = sequence_logprob(model, tokenizer, item["stereotype"], max_length=256)
                anti_lp = sequence_logprob(model, tokenizer, item["anti"], max_length=256)
                loss = F.softplus(stereo_lp - anti_lp)
                losses.append(loss)
                gaps.append((stereo_lp - anti_lp).detach())

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()

            avg_gap = torch.stack(gaps).mean().item()
            pbar.set_postfix(loss=float(batch_loss.item()), gap=avg_gap)

            if abs(avg_gap) < best_gap:
                best_gap = abs(avg_gap)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        try:
            model.load_state_dict(best_state, strict=False)
        except RuntimeError as e:
            print(f"[warn] Could not restore best checkpoint: {e}")

    model.eval()
    return model, float(best_gap)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/lftf/models",
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "lora_r": 8,
            "lora_alpha": 16,
            "bmi_samples": 100,
        },
        "crowspairs": {
            "model": "llama_8b",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/lftf/models_crowspairs",
            "model_tag": "crowspairs_all",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "lora_r": 8,
            "lora_alpha": 16,
            "bmi_samples": 100,
        },
        "stereoset": {
            "model": "llama_8b",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/lftf/models_stereoset",
            "model_tag": "stereoset_all",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "num_epochs": 2,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "lora_r": 8,
            "lora_alpha": 16,
            "bmi_samples": 100,
        },
    }

    for key, value in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFTF training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--target_layer", type=int, default=None)
    parser.add_argument("--bmi_samples", type=int, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--train_limit", type=int, default=None)

    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    model_name = _resolve_model_name(args.model_name, args.model)
    quant = get_quantization_config() if args.use_4bit else None

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
        token=args.hf_token,
    )

    num_layers = model.config.num_hidden_layers

    if args.dataset == "bbq":
        if args.target_layer is None:
            bmi_prompts = create_bbq_prompts_for_bmi(num_samples=args.bmi_samples)
            bmi_scores, optimal_layer = calculate_bmi(model, tokenizer, num_layers, bmi_prompts, max_length=256)
        else:
            bmi_scores = {}
            optimal_layer = int(args.target_layer)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                f"model.layers.{optimal_layer}.self_attn.q_proj",
                f"model.layers.{optimal_layer}.self_attn.v_proj",
                f"model.layers.{optimal_layer}.mlp.down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        prompts = create_bbq_training_prompts()
        he_id = tokenizer.encode(" he", add_special_tokens=False)[0]
        she_id = tokenizer.encode(" she", add_special_tokens=False)[0]

        model, best_gap = lftf_train_bbq(model, tokenizer, prompts, he_id, she_id, args)
        model_dir = args.output_dir
        config_path = os.path.join(args.output_dir, "config.json")

    elif args.dataset == "crowspairs":
        pairs = load_crows_pairs(args.data_path, bias_type=args.bias_type, limit=args.train_limit)
        if not pairs:
            raise ValueError("No CrowS-Pairs training pairs found after filtering")

        bmi_prompts = [x["stereotype"] for x in pairs[: args.bmi_samples]] + [x["anti"] for x in pairs[: args.bmi_samples]]
        if args.target_layer is None:
            bmi_scores, optimal_layer = calculate_bmi(model, tokenizer, num_layers, bmi_prompts, max_length=256)
        else:
            bmi_scores = {}
            optimal_layer = int(args.target_layer)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                f"model.layers.{optimal_layer}.self_attn.q_proj",
                f"model.layers.{optimal_layer}.self_attn.v_proj",
                f"model.layers.{optimal_layer}.mlp.down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        model, best_gap = lftf_train_crows(model, tokenizer, pairs, args)
        model_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
        config_path = os.path.join(args.output_dir, f"config_{args.model_tag}.json")

    else:
        pairs = load_stereoset_pairs(args.data_path, split=args.split, bias_type=args.bias_type, limit=args.train_limit)
        if not pairs:
            raise ValueError("No StereoSet training pairs found after filtering")

        bmi_prompts = [x["anti"] for x in pairs[: args.bmi_samples]]
        if args.target_layer is None:
            bmi_scores, optimal_layer = calculate_bmi(model, tokenizer, num_layers, bmi_prompts, max_length=256)
        else:
            bmi_scores = {}
            optimal_layer = int(args.target_layer)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=[
                f"model.layers.{optimal_layer}.self_attn.q_proj",
                f"model.layers.{optimal_layer}.self_attn.v_proj",
                f"model.layers.{optimal_layer}.mlp.down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        model, best_gap = lftf_train_stereoset(model, tokenizer, pairs, args)
        model_dir = os.path.join(args.output_dir, f"model_{args.model_tag}")
        config_path = os.path.join(args.output_dir, f"config_{args.model_tag}.json")

    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "model_name": model_name,
        "model_tag": args.model_tag,
        "optimal_layer": int(optimal_layer),
        "bmi_scores": bmi_scores,
        "best_gap": float(best_gap),
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "seed": int(args.seed),
    }
    if args.dataset != "bbq":
        config.update({"data_path": args.data_path, "bias_type": args.bias_type, "train_limit": args.train_limit})
    if args.dataset == "stereoset":
        config["split"] = args.split

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
