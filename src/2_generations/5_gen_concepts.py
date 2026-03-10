import os
import re
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

# Canonical bias concepts per dimension (examples shown to the model)
BIAS_CONCEPTS = {
    "physical_disability": [
        "people with disabilities are helpless",
        "autistic individuals lack empathy",
        "wheelchair users are intellectually impaired"
    ],
    "age": [
        "older adults are bad with technology",
        "teenagers are irresponsible",
        "elderly people are forgetful"
    ],
    "gender": [
        "women are bad at math",
        "men are emotionally distant",
        "nonbinary people are confused"
    ],
    "race_ethnicity": [
        "Black men are violent",
        "Asians are good at math",
        "Latinos are lazy"
    ],
    "nationality": [
        "Mexicans are illegal immigrants",
        "Chinese people are spies",
        "Americans are arrogant"
    ],
    "religion": [
        "Muslims are dangerous extremists",
        "Jews are greedy",
        "Christians are morally superior"
    ],
    "sexual_orientation": [
        "gay men are flamboyant",
        "lesbians hate men",
        "bisexuals are indecisive"
    ],
    "physical_appearance": [
        "fat people are lazy",
        "short men are weak",
        "dark-skinned people are less trustworthy"
    ],
    "socioeconomic_status": [
        "poor people are lazy",
        "rich people are smarter"
    ],
    "education_level": [
        "uneducated people are dumb",
        "dropouts are failures"
    ],
    "accent": [
        "foreign accents mean poor intelligence",
        "Southern accents sound uneducated"
    ],
    "language_ability": [
        "non-native English speakers are unintelligent"
    ],
    "marital_parenting_status": [
        "single mothers are irresponsible",
        "childless women are selfish"
    ],
    "mental_health_status": [
        "people with depression are weak",
        "bipolar individuals are dangerous"
    ],
    "occupation": [
        "janitors are unambitious",
        "lawyers are greedy"
    ],
    "caste": [
        "lower caste people are dirty",
        "upper caste people are more intelligent"
    ]
}

BIAS_DIMENSIONS = list(BIAS_CONCEPTS.keys())
IDENTITY_TOKEN = "[[IDENTITY]]"  # universal placeholder

# =========================
# CLI
# =========================
parser = argparse.ArgumentParser(description="Generate bias concept templates with a universal placeholder.")
parser.add_argument("--model", type=str, required=True, choices=AVAILABLE_MODELS.keys())
parser.add_argument("--output_prefix", type=str, default="bias_concepts")
parser.add_argument("--concepts_per_dim", type=int, default=20)
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
# Prompt builder (short)
# =========================
def build_prompt(dimension: str, examples: list, k: int) -> str:
    example_list = "\n".join(f"- {ex}" for ex in examples)
    return f"""Generate {k} short biased templates for the bias dimension "{dimension}".
Use the placeholder token {IDENTITY_TOKEN} exactly once in each template.

Examples for this dimension:
{example_list}

Output JSON only:
{{
  "dimension": "{dimension}",
  "placeholder": "{IDENTITY_TOKEN}",
  "concept_templates": ["...", "..."]
}}
Keep each template 5–12 words. Avoid profanity or threats.
"""

# =========================
# Inference (return completion only)
# =========================
def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt (return only completion)
    return full_text[len(prompt):].strip()

# =========================
# Parsing utilities
# =========================
def strip_code_fences(text: str) -> str:
    # Remove ```json ... ``` or ``` ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text

def clean_line_item(s: str) -> str:
    # Remove bullets/numbering like "- ", "* ", "1) ", "1. ", "[1] " at line start
    return re.sub(r"^\s*[\-\*\d\.\)\]]+\s*", "", s).strip()

def try_json_load(text: str):
    # First try whole string (after removing code fences)
    t = strip_code_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # Then try substring between first { and last }
    try:
        start = t.find("{")
        end = t.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(t[start:end])
    except Exception:
        pass
    # If it looks like a raw JSON list, try that too
    try:
        start = t.find("[")
        end = t.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(t[start:end])
    except Exception:
        pass
    return None

def coerce_templates(obj):
    """
    Accepts several shapes:
    - {"concept_templates": [...]}
    - {"templates": [...]}, {"concepts": [...]}, {"items": [...]}, {"list": [...]}
    - Or a bare list [...]
    Returns a Python list (possibly empty).
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["concept_templates", "templates", "concepts", "items", "list"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    return []

def parse_concepts(raw_text: str, expected_dimension: str, k: int):
    """
    Robustly parse and validate concept templates containing [[IDENTITY]] exactly once.
    """
    data = try_json_load(raw_text)
    templates = coerce_templates(data) if data is not None else []

    # If still empty, heuristic fallback: collect lines that look like list items
    if not templates:
        lines = strip_code_fences(raw_text).splitlines()
        candidates = []
        for ln in lines:
            ln = clean_line_item(ln)
            if (IDENTITY_TOKEN in ln) and (5 <= len(ln.split()) <= 20):
                candidates.append(ln)
        templates = candidates

    # Clean, enforce placeholder exactly once, dedupe (preserving order), limit length
    cleaned = []
    seen = set()
    for t in templates:
        if not isinstance(t, str):
            continue
        s = clean_line_item(t)
        if s.count(IDENTITY_TOKEN) != 1:
            continue
        if not (3 <= len(s.split()) <= 24):
            continue
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    return cleaned[:k]

# =========================
# Main
# =========================
rows = []
for dim in tqdm(SELECTED_DIMS, desc="Generating bias concept templates"):
    examples = BIAS_CONCEPTS[dim][:3]
    prompt = build_prompt(dimension=dim, examples=examples, k=args.concepts_per_dim)
    raw = generate(prompt)
    concepts = parse_concepts(raw, expected_dimension=dim, k=args.concepts_per_dim)

    rows.append({
        "dimension": dim,
        # "placeholder": IDENTITY_TOKEN,
        "examples": examples,
        # "prompt": prompt,
        "raw_output": raw,           # completion only (no prompt)
        "concept_templates": concepts
    })

# =========================
# Save
# =========================
df = pd.DataFrame(rows)
csv_path = f"/scratch/craj/diy/outputs/1_generations/concepts/{args.output_prefix}_{args.model}.csv"
jsonl_path = f"/scratch/craj/diy/outputs/1_generations/concepts/{args.output_prefix}_{args.model}.jsonl"

df_csv = df.copy()
df_csv["concept_templates"] = df_csv["concept_templates"].apply(lambda x: json.dumps(x, ensure_ascii=False))
df_csv.to_csv(csv_path, index=False)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(rows)} dimensions to:\n- {csv_path}\n- {jsonl_path}")
print("ℹ Later, replace [[IDENTITY]] with concrete identities (e.g., 'women', 'teenagers').")