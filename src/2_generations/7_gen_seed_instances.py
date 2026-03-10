#!/usr/bin/env python3
import os
import time
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
SCENARIOS = ["education", "environment", "healthcare", "workplace", "media", "law_and_policy", "sports", "technology", "economics", "art_and_leisure"]
INSTANCE_TYPES = ["biased_statement", "biased_story", "biased_dialogue", "biased_news_snippet", "biased_question"]
SOCIAL_SETUPS = ["individual", "pair", "individual_vs_group", "group_vs_group"]
FRAMING_TYPES = ["explicit", "implicit"]

BIAS_DIMENSIONS = ["physical_disability", "age", "gender", "race_ethnicity", "nationality", "religion", "sexual_orientation", "physical_appearance", 
"socioeconomic_status", "education_level", "accent", "language_ability", "marital_parenting_status", "mental_health_status", "occupation", "caste"]

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
3. Biased event → One sentence describing an event that leads to forming that opinion.

Rules:
- Keep realistic and coherent.
- Ensure the bias is clear but avoid profanity or slurs.
- Use 1 sentence per version only.
- Output ONLY the three versions, each on a new line prefixed with "1.", "2.", "3.". END###
"""


# ============================================================
# Model Loader
# ============================================================
def load_model(model_key):
    model_info = AVAILABLE_MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_info["model"], cache_dir=model_info["cache_dir"])
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    stop_token_id = tokenizer.encode("###")[0]

    # quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_info["model"],
        quantization_config=quant_config,
        device_map="auto",
        cache_dir=model_info["cache_dir"]
    )
    return tokenizer, model, stop_token_id

# ============================================================
# Generation Function
# ============================================================
def generate_three(prompt, tokenizer, model, stop_token_id, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        # eos_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True, 
        output_hidden_states=False
    )

    # generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


    # # Remove prompt text if included
    # if generated.startswith(prompt):
    #     generated = generated[len(prompt):]

    # Extract lines starting with 1., 2., 3.
    lines = [l.strip() for l in generated.split("\n") if l.strip()]
    versions = {"opinion": "", "action": "", "event": ""}
    for l in lines:
        if l.startswith("1"):
            versions["opinion"] = l.lstrip("123. ").strip()
        elif l.startswith("2"):
            versions["action"] = l.lstrip("123. ").strip()
        elif l.startswith("3"):
            versions["event"] = l.lstrip("123. ").strip()
    return versions

# ============================================================
# Progress Loader
# ============================================================
def load_progress(csv_path):
    if not os.path.exists(csv_path):
        return set(), 0
    try:
        df = pd.read_csv(csv_path)
        used = set(
            zip(
                df["bias_dimension"],
                df["identity"],
                df["scenario"],
                df["instance_type"],
                df["social_setup"],
                df["framing_type"],
                df["concept_template"]
            )
        )
        return used, len(df)
    except:
        return set(), 0

# ============================================================
# ETA Estimator
# ============================================================
def estimate_eta(merged, args, tokenizer, model, stop_token_id):
    """
    Measure average generation time and estimate total runtime.
    """

    # Calculate total rows
    total_rows = (
        len(merged)
        * args.max_concepts_per_identity
        * len(args.scenarios)
        * len(args.instance_types)
        * len(args.social_setups)
        * len(args.framing_types)
    )

    # Build a quick test prompt
    sample_prompt = build_prompt(
        concept_filled="[[IDENTITY]]",
        scenario=args.scenarios[0],
        instance_type=args.instance_types[0],
        social_setup=args.social_setups[0],
        framing_type=args.framing_types[0],
    )

    # Measure time for 3 samples
    start = time.time()
    for _ in range(3):
        _ = generate_three(
            sample_prompt,
            tokenizer, model, stop_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    end = time.time()

    avg_time = (end - start) / 3
    total_hours = (total_rows * avg_time) / 3600

    # Print results in your exact desired format
    print(f"\nAvg time per generation: {avg_time:.2f} seconds")
    print(f"Total rows: {total_rows:,}")
    print(f"Estimated runtime: {total_hours:.2f} hours\n")

    return avg_time, total_hours


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seed instances from bias concepts and identities")
    parser.add_argument("--concepts_file", default="/scratch/craj/diy/outputs/1_generations/concepts/bias_concepts_llama.csv")
    parser.add_argument("--identities_file", default="/scratch/craj/diy/outputs/1_generations/identities/bias_identities_flat.csv")
    parser.add_argument("--model", choices=AVAILABLE_MODELS.keys(), default="llama_70b")
    parser.add_argument("--bias_dimensions", nargs="+", default=["physical_disability", "age", "gender", "race_ethnicity", "nationality", "religion", "sexual_orientation", "physical_appearance", "socioeconomic_status"])
    parser.add_argument("--scenarios", nargs="+", default=["education", "environment", "healthcare", "workplace", "media", "law_and_policy", "sports", "technology", "economics", "art_and_leisure"])
    parser.add_argument("--instance_types", nargs="+", default=["biased_statement"])
    parser.add_argument("--social_setups", nargs="+", default=["individual"])
    parser.add_argument("--framing_types", nargs="+", default=["explicit"])
    parser.add_argument("--max_concepts_per_identity", type=int, default=5)
    parser.add_argument("--max_identities_per_dimension", type=int, default=10)
    parser.add_argument("--itemcount", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_prefix", default="seed_instances")
    args = parser.parse_args()

    scenario_name = args.scenarios[0] if len(args.scenarios) == 1 else "_".join(args.scenarios)

    # Output paths
    csv_path = f"/scratch/craj/diy/outputs/1_generations/seed_instances/{args.output_prefix}_{scenario_name}_all_dim_new.csv"
    jsonl_path = f"/scratch/craj/diy/outputs/1_generations/seed_instances/{args.output_prefix}_{scenario_name}_all_dim_new.jsonl"

    # Load previous progress
    completed_keys, completed_count = load_progress(csv_path)
    if completed_count > 0:
        print(f"🔄 Resuming from previous progress: {completed_count} rows already generated.")


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
    # merged = merged.head(10)

    # Load model
    tokenizer, model, stop_token_id = load_model(args.model)
    estimate_eta(merged, args, tokenizer, model, stop_token_id)

    SAVE_EVERY = 100

    combos = list(product(args.scenarios, args.instance_types, args.social_setups, args.framing_types))

    instances = []
    print(f"ℹ️ Merged rows after filters: {len(merged)}")
    count = 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Generating"):
        identity = row["identity"]
        dimension = row["dimension"]
        templates = row["concept_templates"][:args.max_concepts_per_identity]

        for concept_template in templates:
            concept_filled = concept_template.replace("[[IDENTITY]]", identity)

            for scenario, inst_type, setup, framing in combos:

                # Unique key for resume
                key = (dimension, identity, scenario, inst_type, setup, framing, concept_template)

                # Skip if already completed
                if key in completed_keys:
                    continue

                if args.itemcount and count >= args.itemcount:
                    break

                # Build prompt & generate
                prompt = build_prompt(concept_filled, scenario, inst_type, setup, framing)
                versions = generate_three(
                    prompt, tokenizer, model, stop_token_id,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )


                instances.append({
                    "bias_dimension": dimension,
                    "identity": identity,
                    "scenario": scenario,
                    "instance_type": inst_type,
                    "social_setup": setup,
                    "framing_type": framing,
                    "concept_template": concept_template,
                    "opinion_version": versions["opinion"],
                    "action_version": versions["action"],
                    "event_version": versions["event"]
                })

                # Periodic save
                if len(instances) % SAVE_EVERY == 0:
                    print(f"💾 Saving progress at {len(instances)} buffered rows...")
                    tmp_df = pd.DataFrame(instances)
                    if os.path.exists(csv_path):
                        tmp_df.to_csv(csv_path, mode="a", index=False, header=False)
                    else:
                        tmp_df.to_csv(csv_path, index=False)
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        for inst in instances:
                            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                    instances = []  # clear buffer

                count += 1

            if args.itemcount and count >= args.itemcount:
                break

        if args.itemcount and count >= args.itemcount:
            break


    # Final save
    if instances:
        print(f"💾 Final save of {len(instances)} remaining rows...")
        if os.path.exists(csv_path):
            pd.DataFrame(instances).to_csv(csv_path, mode="a", index=False, header=False)
        else:
            pd.DataFrame(instances).to_csv(csv_path, index=False)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for inst in instances:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
