#!/usr/bin/env python3
import argparse
import json
import re
import pandas as pd
from itertools import product
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# Defaults
# ============================================================
SCENARIOS = [
    "education", "environment", "healthcare", "workplace", "media",
    "law_and_policy", "sports", "technology", "economics", "art_and_leisure"
]
INSTANCE_TYPES = [
    "biased_statement", "biased_story", "biased_dialogue",
    "biased_news_snippet", "biased_question"
]
SOCIAL_SETUPS = ["individual", "pair", "individual_vs_group", "group_vs_group"]
FRAMING_TYPES = ["explicit", "implicit"]

AVAILABLE_MODELS = {
    "llama_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.1-8b-instruct"
    },
    "llama_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "cache_dir": "/scratch/craj/model_cache/llama-3.3-70b-instruct"
    },
    "aya_8b": {
        "model": "CohereForAI/aya-expanse-8b",
        "cache_dir": "/scratch/craj/model_cache/aya-expanse-8b"
    },
    "qwen_32b": {
        "model": "Qwen/QwQ-32B",
        "cache_dir": "/scratch/craj/model_cache/qwen-32b"
    },
}

# ============================================================
# Prompt Builder
# ============================================================
def build_prompt(concept_filled, scenario, instance_type, social_setup, framing_type):
    return f"""Generate THREE short variations for the following setup:

Concept: "{concept_filled}"
Scenario: {scenario}
Instance type: {instance_type}
Social setup: {social_setup}
Framing type: {framing_type}

Versions:
1. Pure opinion → One short biased statement. Must be direct, like "All [IDENTITY] are dishonest."
2. Biased action → One sentence where that opinion causes a discriminatory action.
3. Biased event → One sentence describing a situation or incident that could cause someone to form or reinforce that biased opinion. 
   It must describe an observable event, not an action done by the biased person. 
   Example: "After a store robbery where a [[IDENTITY]] was accused without evidence, many started assuming [[IDENTITY]] are dishonest."

Rules:
- Keep realistic and coherent.
- Ensure the bias is clear but avoid profanity or slurs.
- Each version must be exactly one sentence.
- Output ONLY the three versions, each on a new line prefixed with "1.", "2.", "3.".
- After completing the third version, end immediately and output "<END>" on a new line.
"""

# ============================================================
# Model Loader
# ============================================================
def load_model(model_key):
    model_info = AVAILABLE_MODELS[model_key]
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
    return tokenizer, model

# ============================================================
# Generation Function
# ============================================================
def generate_three(prompt, tokenizer, model, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=min(max_new_tokens, 120),  # Limit output length
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt text if included
    if generated.startswith(prompt):
        generated = generated[len(prompt):]

    # Stop at <END> marker if present
    generated = generated.split("<END>")[0].strip()

    # Cut off if model accidentally continues numbering
    generated = re.split(r"(?:\n|^)\s*4\.", generated)[0]

    # Extract numbered lines robustly
    matches = re.findall(r"(?:^|\n)\s*[123]\.\s*(.*)", generated)
    lines = [m.strip() for m in matches if m.strip()]

    versions = {"opinion": "", "action": "", "event": ""}
    for l in lines:
        if l.startswith("1") or re.match(r"^1[\.\)]", l):
            versions["opinion"] = re.sub(r"^[123\.\)\s]+", "", l).strip()
        elif l.startswith("2") or re.match(r"^2[\.\)]", l):
            versions["action"] = re.sub(r"^[123\.\)\s]+", "", l).strip()
        elif l.startswith("3") or re.match(r"^3[\.\)]", l):
            versions["event"] = re.sub(r"^[123\.\)\s]+", "", l).strip()

    # Fallback if event missing — pick likely causal sentence
    if not versions["event"]:
        candidates = [l for l in lines if re.search(r"\b(after|because|when|due to)\b", l, re.I)]
        if candidates:
            versions["event"] = candidates[-1]

    # Enforce only one sentence for event
    if versions["event"]:
        m = re.match(r"(.+?[.!?])(\s|$)", versions["event"])
        if m:
            versions["event"] = m.group(1).strip()

    return versions

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seed instances from bias concepts and identities")
    parser.add_argument("--concepts_file", default="/scratch/craj/diy/outputs/1_generations/bias_concepts_llama.csv")
    parser.add_argument("--identities_file", default="/scratch/craj/cognitive_debiasing/outputs/generations/bias_identities/bias_identities_flat.csv")
    parser.add_argument("--model", choices=AVAILABLE_MODELS.keys(), default="llama_8b")
    parser.add_argument("--bias_dimensions", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=["workplace"])
    parser.add_argument("--instance_types", nargs="+", default=["biased_statement"])
    parser.add_argument("--social_setups", nargs="+", default=["individual"])
    parser.add_argument("--framing_types", nargs="+", default=["explicit"])
    parser.add_argument("--max_concepts_per_identity", type=int, default=1)
    parser.add_argument("--max_identities_per_dimension", type=int, default=1)
    parser.add_argument("--itemcount", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_prefix", default="seed_instances")
    args = parser.parse_args()

    # Load concept templates
    concepts_df = pd.read_csv(args.concepts_file)
    concepts_df["concept_templates"] = concepts_df["concept_templates"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Load identities
    identities_df = pd.read_csv(args.identities_file)

    # Optional bias dimension filter
    if args.bias_dimensions:
        concepts_df = concepts_df[concepts_df["dimension"].isin(args.bias_dimensions)]
        identities_df = identities_df[identities_df["bias_dimension"].isin(args.bias_dimensions)]

    # Limit identities per dimension
    if args.max_identities_per_dimension is not None:
        identities_df = (
            identities_df.groupby("bias_dimension", group_keys=False)
            .apply(lambda g: g.head(args.max_identities_per_dimension))
            .reset_index(drop=True)
        )

    # Merge
    merged = concepts_df.merge(
        identities_df,
        left_on="dimension",
        right_on="bias_dimension",
        how="inner"
    )
    merged = merged.head(10)

    # Load model
    tokenizer, model = load_model(args.model)

    combos = list(product(args.scenarios, args.instance_types, args.social_setups, args.framing_types))

    instances = []
    print(f"ℹ️ Merged rows after filters: {len(merged)}")
    count = 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Generating"):
        templates = row["concept_templates"][:args.max_concepts_per_identity]
        for concept_template in templates:
            concept_filled = concept_template.replace("[[IDENTITY]]", row["identity"])
            for scenario, inst_type, setup, framing in combos:
                if args.itemcount and count >= args.itemcount:
                    break
                prompt = build_prompt(concept_filled, scenario, inst_type, setup, framing)
                versions = generate_three(
                    prompt, tokenizer, model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                instances.append({
                    "bias_dimension": row["dimension"],
                    "identity": row["identity"],
                    "scenario": scenario,
                    "instance_type": inst_type,
                    "social_setup": setup,
                    "framing_type": framing,
                    "concept_template": concept_template,
                    "opinion_version": versions["opinion"],
                    "action_version": versions["action"],
                    "event_version": versions["event"]
                })
                count += 1
            if args.itemcount and count >= args.itemcount:
                break
        if args.itemcount and count >= args.itemcount:
            break

    # Save
    out_df = pd.DataFrame(instances)
    csv_path = f"/scratch/craj/diy/outputs/1_generations/{args.output_prefix}.csv"
    jsonl_path = f"/scratch/craj/diy/outputs/1_generations/{args.output_prefix}.jsonl"
    out_df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(instances)} instances.")
    print(f"💾 CSV saved to {csv_path}")
    print(f"💾 JSONL saved to {jsonl_path}")
