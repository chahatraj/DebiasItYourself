import os
import argparse

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# Config
# =========================
AVAILABLE_MODELS = {
    "llama": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/cache/model_cache/llama-3.3-70b-instruct"
    },
    "aya": {
        "model": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/cache/model_cache/aya-expanse-8b"
    },
    "qwen": {
        "model": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/cache/model_cache/qwen-32b"
    }
}

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Bucket contrastive concepts into thematic categories.")
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama")
parser.add_argument("--input_path", type=str, default="/scratch/craj/diy/data/description_based_concepts.csv")
parser.add_argument("--output_path", type=str, default="/scratch/craj/diy/outputs/1_generations/concepts/concept_buckets.csv")
parser.add_argument("--max_new_tokens", type=int, default=80)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick tests.")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

# =========================
# Load model
# =========================
model_info = AVAILABLE_MODELS[args.model]
tokenizer = AutoTokenizer.from_pretrained(model_info["model"], cache_dir=model_info["cache_dir"])
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    model_info["model"],
    quantization_config=quant_config,
    device_map="auto",
    cache_dir=model_info["cache_dir"]
)

# =========================
# Prompting
# =========================
def build_prompt(concept1: str, concept2: str, explanation: str) -> list:
    system = "You are a concise classifier. Output only the bucket name."
    user = (
        "Classify the contrastive concept pair into ONE short theme label "
        "(1-3 words, lowercase, no punctuation). "
        "If possible, reuse common themes across items (e.g., technology, education, health, family, "
        "work, crime, politics, religion, sports, finances, personality, communication, substance_use). "
        "Return ONLY the bucket label and nothing else.\n\n"
        f"Concept1: {concept1}\n"
        f"Concept2: {concept2}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def generate(messages) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    gen_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def parse_theme(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        return ""
    first_line = raw_text.strip().splitlines()[0].strip()
    return first_line.lower()

# =========================
# Main
# =========================
df = pd.read_csv(args.input_path)
if args.limit:
    df = df.head(args.limit)

concept_buckets = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Bucketing concepts"):
    concept1 = str(row.get("Concept1", "")).strip()
    concept2 = str(row.get("Concept2", "")).strip()
    explanation = str(row.get("Explanation", "")).strip()

    prompt = build_prompt(concept1, concept2, explanation)
    raw = generate(prompt)
    theme = parse_theme(raw)
    concept_buckets.append(theme)

out_df = df.copy()
out_df["concept_bucket"] = concept_buckets
out_df.to_csv(args.output_path, index=False)

print(f"Saved:\n- {args.output_path}")
