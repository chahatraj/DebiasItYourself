import os
import json
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

BIAS_DIMENSIONS = [
    "physical_disability",
    "age",
    "gender",
    "race_ethnicity",
    "nationality",
    "religion",
    "sexual_orientation",
    "physical_appearance",
    "socioeconomic_status",
    # "education_level",
    # "accent",
    # "language_ability",
    # "marital_parenting_status",
    # "mental_health_status",
    # "occupation",
    # "caste",
]

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Generate identity lists for each bias dimension.")
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), default="llama")
parser.add_argument("--output_prefix", type=str, default="bias_identities")
parser.add_argument("--identities_per_dim", type=int, default=20)
parser.add_argument("--dimensions", type=str, default=None)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.95)
args = parser.parse_args()

# Resolve dimensions
if args.dimensions:
    SELECTED_DIMS = [d.strip() for d in args.dimensions.split(",") if d.strip()]
else:
    SELECTED_DIMS = BIAS_DIMENSIONS[:]

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
# Prompt builder
# =========================

def build_prompt(dimension: str, k: int):
    return [
        {"role": "system", "content": "You are a JSON-only generator. Output strictly valid JSON and nothing else."},
        {"role": "user", "content": f'List {k} real-world identities for "{dimension}". Each identity must be a person-denoting noun (e.g., "amputee", "blind person", "blonde"), not conditions or abstract traits (e.g., "amputation", "hair color", "poverty"). Return ONLY JSON: {{dimension:{dimension},identities:[...]}} // {k} unique items'}
    ]

# =========================
# Inference
# =========================

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

    # decode ONLY the newly generated tokens (not the prompt)
    gen_tokens = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

# =========================
# Parse JSON safely
# =========================
def parse_identities(raw_text: str, expected_dimension: str, k: int):
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        data = json.loads(raw_text[start:end])
    except Exception:
        return []
    if data.get("dimension") != expected_dimension:
        return []
    identities = data.get("identities", [])
    if not isinstance(identities, list):
        return []
    # clean & dedup
    cleaned = []
    for item in identities:
        if isinstance(item, str):
            val = item.strip()
            if val and val not in cleaned:
                cleaned.append(val)
    return cleaned[:k]

# =========================
# Main loop
# =========================
rows = []
for dim in tqdm(SELECTED_DIMS, desc="Generating identities"):
    prompt = build_prompt(dim, args.identities_per_dim)
    raw = generate(prompt)
    identities = parse_identities(raw, dim, args.identities_per_dim)
    rows.append({
        "dimension": dim,
        "identities": identities,
        "raw_output": raw
    })

# =========================
# Save
# =========================
df = pd.DataFrame(rows)
csv_path = f"/scratch/craj/diy/outputs/1_generations/identities/{args.output_prefix}_{args.model}.csv"
jsonl_path = f"/scratch/craj/diy/outputs/1_generations/identities/{args.output_prefix}_{args.model}.jsonl"

df_csv = df.copy()
df_csv["identities"] = df_csv["identities"].apply(lambda x: json.dumps(x, ensure_ascii=False))
df_csv.to_csv(csv_path, index=False)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Saved identities for {len(rows)} dimensions to:\n- {csv_path}\n- {jsonl_path}")
