#!/usr/bin/env python3
import os, json, ast, re, random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

# NumPy>=2 moved VisibleDeprecationWarning to np.exceptions; cappr expects np.VisibleDeprecationWarning.
if not hasattr(np, "VisibleDeprecationWarning") and hasattr(np, "exceptions"):
    if hasattr(np.exceptions, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = np.exceptions.VisibleDeprecationWarning

from cappr.huggingface.classify import predict_proba
from inference_instruction import (
    apply_instruction_to_content,
    resolve_inference_instruction,
)

# ============================================================
# Globals
# ============================================================
INPUT_CSV = "/scratch/craj/diy/data/processed_bbq_all.csv"
DEFAULT_OUTPUT_DIR = "/scratch/craj/diy/outputs/2_base_models/bbq"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

AVAILABLE_MODELS = {
    "llama_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.1-8b-instruct"
    },
    "llama_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.3-70b-instruct"
    },
    "aya_8b": {
        "model": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/cache/model_cache/aya-expanse-8b"
    },
    "qwen_32b": {
        "model": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/cache/model_cache/qwen-32b"
    },
}

VALID_SOURCE_FILES = [
    "Age.jsonl", "Disability_status.jsonl", "Gender_identity.jsonl",
    "Nationality.jsonl", "Physical_appearance.jsonl", "Race_ethnicity.jsonl",
    "Race_x_gender.jsonl", "Race_x_SES.jsonl", "Religion.jsonl",
    "SES.jsonl", "Sexual_orientation.jsonl"
]

# ============================================================
# CLI Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama_8b")
parser.add_argument("--source_file", type=str, choices=VALID_SOURCE_FILES, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_path", type=str, default=None, help="Local/HF path of merged finetuned model.")
parser.add_argument("--model_tag", type=str, default=None, help="Tag for output filename.")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to write prediction CSVs.")
parser.add_argument(
    "--inference_instruction_mode",
    type=str,
    choices=["off", "strategy"],
    default="off",
    help="Inference-time prompt mode: off or strategy-conditioned instruction prefix.",
)
parser.add_argument(
    "--inference_strategy",
    type=str,
    default=None,
    help="Optional explicit strategy for inference instruction (sr/ci/ind/pt/pc/all or full name).",
)
args = parser.parse_args()

if args.output_dir is None:
    model_output_dir = os.path.join(DEFAULT_OUTPUT_DIR, args.model)
else:
    model_output_dir = args.output_dir
os.makedirs(model_output_dir, exist_ok=True)

if args.model_tag:
    model_tag = args.model_tag
elif args.model_path:
    model_tag = os.path.basename(os.path.normpath(args.model_path))
else:
    model_tag = args.model
model_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_tag)

inference_strategy_key, inference_instruction = resolve_inference_instruction(
    mode=args.inference_instruction_mode,
    strategy=args.inference_strategy,
    model_tag=model_tag,
    model_path=args.model_path,
)
if inference_instruction:
    print(
        f"✅ Inference instruction enabled: mode={args.inference_instruction_mode}, "
        f"strategy={inference_strategy_key}"
    )
    print(f"   Instruction: {inference_instruction}")

# ============================================================
# Load Model + Tokenizer
# ============================================================
model_info = AVAILABLE_MODELS[args.model]
model_source = args.model_path if args.model_path else model_info["model"]
try:
    tokenizer = AutoTokenizer.from_pretrained(model_source, cache_dir=model_info["cache_dir"])
except Exception as e:
    if args.model_path:
        print(f"⚠️ Tokenizer load from merged model failed ({e}). Falling back to base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_info["model"], cache_dir=model_info["cache_dir"])
    else:
        raise
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=quant_config,
        device_map="auto",
        cache_dir=model_info["cache_dir"]
    )
except Exception as e:
    print(f"⚠️ Quantized load failed ({e}). Falling back to float16.")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=model_info["cache_dir"]
    )

model.eval()
print(f"✅ Loaded model: {model_source} (tag={model_tag})")

# ============================================================
# Load Data
# ============================================================
df = pd.read_csv(INPUT_CSV)
df["answer_info"] = df["answer_info"].apply(ast.literal_eval)
df = df[df["source_file"] == args.source_file].reset_index(drop=True)

LIMIT = None  # e.g., 100 for quick test
if LIMIT:
    df = df.iloc[:LIMIT]

print(f"✅ Loaded {len(df)} examples from source file: {args.source_file}")

# ============================================================
# Prompt Formatter
# ============================================================
def format_prompt(context, question, ans0, ans1, ans2):
    """Removes explicit 'Context:' and 'Question:' prefixes."""
    return f"{context}\n{question}\nA. {ans0}\nB. {ans1}\nC. {ans2}\nAnswer:"

# ============================================================
# Inference (batched)
# ============================================================
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
    options = [row["ans0"], row["ans1"], row["ans2"]]
    prompt = format_prompt(row["context"], row["question"], *options)
    prompt = apply_instruction_to_content(prompt, inference_instruction)

    try:
        probs = predict_proba(
            prompt,
            completions=options,
            model_and_tokenizer=(model, tokenizer),
            batch_size=1
        )
        # take the first (and only) result

        pred_idx = int(np.argmax(probs))
        pred_letter = chr(65 + pred_idx)

        results.append({
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
            "option_probs": {chr(65 + k): float(p) for k, p in enumerate(probs)},
            "inference_instruction_mode": args.inference_instruction_mode,
            "inference_instruction_strategy": inference_strategy_key or "",
        })
    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        continue


output_csv = os.path.join(
    model_output_dir,
    f"bbq_preds_{model_tag}_{args.source_file.replace('.jsonl', '')}.csv"
)
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n✅ Inference complete. Saved to {output_csv}")
