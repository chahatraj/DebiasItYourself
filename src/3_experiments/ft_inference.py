#!/usr/bin/env python3
import os, json, ast, random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from cappr.huggingface.classify import predict_proba

# ============================================================
# Globals
# ============================================================
INPUT_CSV = "/scratch/craj/diy/data/processed_bbq_all.csv"
OUTPUT_DIR = "/scratch/craj/diy/outputs/5_finetuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hugging Face repo of your fine-tuned model
# FINETUNED_REPO = "chahatraj/cognitive_collaborative"
FINETUNED_REPO = "chahatraj/cognitive_stereotypereplacement"
# FINETUNED_REPO = "chahatraj/cognitive_counterimaging"
# FINETUNED_REPO = "chahatraj/cognitive_individuating"
# FINETUNED_REPO = "chahatraj/cognitive_perspectivetaking"
# FINETUNED_REPO = "chahatraj/cognitive_positivecontact"

random.seed(42)
torch.manual_seed(42)

AVAILABLE_MODELS = {
    "llama_70b": {
        "base": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"
    },
    "aya_8b": {
        "base": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"
    },
    "qwen_32b": {
        "base": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/model_cache/qwen-32b"
    },
    "llama_8b": {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"
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
parser.add_argument("--source_file", type=str, choices=VALID_SOURCE_FILES, default="Religion.jsonl",
                    help="Select one of the BBQ source files (e.g., Gender_identity.jsonl)")
args = parser.parse_args()

# ============================================================
# Load tokenizer + model with LoRA adapter
# ============================================================
model_info = AVAILABLE_MODELS[args.model]

# Load tokenizer from fine-tuned repo
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_REPO, cache_dir=model_info["cache_dir"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base model with 4-bit quantization
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(
    model_info["base"],
    quantization_config=quant_config,
    device_map="auto",
    cache_dir=model_info["cache_dir"]
)

# Attach fine-tuned LoRA adapter from HF repo
model = PeftModel.from_pretrained(base_model, FINETUNED_REPO)

# # Debug: check adapter parameters
# model.print_trainable_parameters()

# # Confirm adapter is attached
# print("Active adapters:", model.peft_config)

# # Peek at LoRA layers
# for name, param in model.named_parameters():
#     if "lora" in name:
#         print("Found LoRA param:", name, param.shape)
#         break


# inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
# with torch.no_grad():
#     out = model.generate(**inputs, max_new_tokens=10)
# print("Sample output:", tokenizer.decode(out[0]))

# ============================================================
# Load and filter BBQ data
# ============================================================
df = pd.read_csv(INPUT_CSV)
df["answer_info"] = df["answer_info"].apply(ast.literal_eval)
df = df[df["source_file"] == args.source_file].reset_index(drop=True)

LIMIT = None  # e.g. set to 100 for quick debugging
if LIMIT:
    df = df.iloc[:LIMIT]

print(f"✅ Loaded {len(df)} examples from source file: {args.source_file}")

# ============================================================
# Prompt formatter
# ============================================================
def format_prompt(context, question, ans0, ans1, ans2):
    """Removes explicit 'Context:' and 'Question:' prefixes."""
    return f"{context}\n{question}\nA. {ans0}\nB. {ans1}\nC. {ans2}\nAnswer:"

# ============================================================
# Inference loop
# ============================================================
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
    options = [row["ans0"], row["ans1"], row["ans2"]]
    prompt = format_prompt(row["context"], row["question"], *options)

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
            "option_probs": {chr(65 + k): float(p) for k, p in enumerate(probs)}
        })
    except Exception as e:
        print(f"❌ Error at row {idx}: {e}")
        continue

# ============================================================
# Save output
# ============================================================
output_csv = os.path.join(
    OUTPUT_DIR, f"bbq_preds_{args.model}_ft_stereotypereplacement_{args.source_file.replace('.jsonl', '')}.csv"
)
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n✅ Inference complete. Saved to {output_csv}")
