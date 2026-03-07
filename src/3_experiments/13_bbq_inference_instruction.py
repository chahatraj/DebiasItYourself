#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


SEED = 42

AVAILABLE_MODELS: Dict[str, Dict[str, str]] = {
    "llama_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.1-8b-instruct",
    },
    "llama_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.3-70b-instruct",
    },
}

DEFAULT_BBQ_PROCESSED = "/scratch/craj/diy/data/processed_bbq_all.csv"
VALID_BBQ_SOURCE_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Race_x_gender.jsonl",
    "Race_x_SES.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def format_bbq_prompt(context: str, question: str, ans0: str, ans1: str, ans2: str) -> str:
    return f"{context}\n{question}\nA. {ans0}\nB. {ans1}\nC. {ans2}\nAnswer:"


def load_model_and_tokenizer(model_key: str, model_path: Optional[str]) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key `{model_key}`")

    info = AVAILABLE_MODELS[model_key]
    model_source = model_path or info["model"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=info["cache_dir"])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(info["model"], cache_dir=info["cache_dir"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=info["cache_dir"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, cache_dir=info["cache_dir"])
        model = model.to("cpu")

    model.eval()
    return model, tokenizer, model_source


def completion_logprobs_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    options: Sequence[str],
    max_length: int,
) -> List[float]:
    full_texts = [f"{prompt} {opt}" for opt in options]
    enc = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * shift_mask
    seq_log_probs = token_log_probs.sum(dim=1)
    return [float(v) for v in seq_log_probs.detach().cpu().tolist()]


def resolve_instruction(mode: str, strategy: Optional[str], model_tag: Optional[str], model_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    helper = _load_module(
        "inference_instruction_helper",
        Path(__file__).resolve().parent / "4_inference_time_instruction.py",
    )
    return helper.resolve_inference_instruction(mode, strategy, model_tag, model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic BBQ inference with optional inference-time instruction.")
    parser.add_argument("--model", choices=list(AVAILABLE_MODELS.keys()), default="llama_8b")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=DEFAULT_BBQ_PROCESSED)
    parser.add_argument("--source_file", choices=VALID_BBQ_SOURCE_FILES, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--inference_instruction_mode", choices=["off", "strategy"], default="off")
    parser.add_argument("--inference_strategy", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    strategy_key, instruction = resolve_instruction(
        mode=args.inference_instruction_mode,
        strategy=args.inference_strategy,
        model_tag=args.model_tag,
        model_path=args.model_path,
    )

    model, tokenizer, model_source = load_model_and_tokenizer(args.model, args.model_path)
    print(f"[INFO] model_source={model_source}")
    print(f"[INFO] instruction_mode={args.inference_instruction_mode}")
    print(f"[INFO] instruction_strategy={strategy_key}")

    df = pd.read_csv(args.data_path)
    df["answer_info"] = df["answer_info"].apply(ast.literal_eval)
    df = df[df["source_file"] == args.source_file].reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[: int(args.limit)].copy()

    rows: List[Dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"BBQ {args.source_file}"):
        options = [row["ans0"], row["ans1"], row["ans2"]]
        base_prompt = format_bbq_prompt(row["context"], row["question"], *options)
        prompt = base_prompt if instruction is None else f"{instruction}\n\nContent:\n{base_prompt}"

        try:
            logps = completion_logprobs_for_prompt(model, tokenizer, prompt, options, max_length=args.max_length)
            arr = np.array(logps, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                probs_arr = np.ones(len(options), dtype=np.float64) / float(len(options))
            else:
                m = float(np.max(finite))
                exps = np.where(np.isfinite(arr), np.exp(arr - m), 0.0)
                denom = float(exps.sum()) or 1.0
                probs_arr = exps / denom
            pred_idx = int(np.argmax(probs_arr))
            pred_letter = chr(65 + pred_idx)

            rows.append(
                {
                    "example_id": row.example_id,
                    "source_file": row.source_file,
                    "context_condition": row.context_condition,
                    "label": row.label,
                    "context": row.context,
                    "question": row.question,
                    "ans0": row.ans0,
                    "ans1": row.ans1,
                    "ans2": row.ans2,
                    "model_output": options[pred_idx],
                    "pred_letter": pred_letter,
                    "pred_index": pred_idx,
                    "option_probs": json.dumps({chr(65 + i): float(p) for i, p in enumerate(probs_arr)}),
                    "option_logprobs": json.dumps({chr(65 + i): float(lp) for i, lp in enumerate(logps)}),
                    "inference_instruction_mode": args.inference_instruction_mode,
                    "inference_instruction_strategy": strategy_key or "",
                }
            )
        except Exception as exc:
            print(f"[WARN] row={idx} failed: {exc}")

    if args.model_tag:
        base_tag = slug(args.model_tag)
    elif args.model_path:
        base_tag = slug(os.path.basename(os.path.normpath(args.model_path)))
    else:
        base_tag = slug(args.model)
    suffix = "instr_off" if strategy_key is None else f"instr_{strategy_key}"
    tag = slug(f"{base_tag}_{suffix}")

    out_csv = os.path.join(args.output_dir, f"bbq_preds_{tag}_{args.source_file.replace('.jsonl', '')}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] saved={out_csv}")


if __name__ == "__main__":
    main()
