#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

# ============================================================
# Debiasing Prompts
# ============================================================
DEBIASING_PROMPTS = {
    "stereotype_replacement": {
        "zero": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice stereotype replacement using the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Reflect: Think why this stereotype might be inaccurate, overgeneralized, or harmful.\n"
                "Step 3 - Replace: Suggest a fairer, individualized, or bias-free alternative phrasing or interpretation.",
        "one": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice stereotype replacement using the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Reflect: Think why this stereotype might be inaccurate, overgeneralized, or harmful.\n"
                "Step 3 - Replace: Suggest a fairer, individualized, or bias-free alternative phrasing or interpretation.\n"
                "Example:\n"
                "Input: Women are too emotional to be good leaders.\n"
                "Step 1 - Recognize: This is a gender stereotype implying women are inherently less rational or fit for leadership roles.\n"
                "Step 2 - Reflect: It unfairly equates emotion with weakness and ignores the leadership capabilities of countless successful women. Emotional intelligence can be a strength.\n"
                "Step 3 - Replace: Leadership ability depends on skills and experience, not gender."
    },
    "counter_imaging": {
        "zero": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice counter-stereotype imaging using the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Imagine: Think of an imagined person who contradicts this stereotype.\n"
                "Step 3 - Reinforce: Elaborate details about this counter-stereotypic individual to strengthen the new association.",
        "one": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice counter-stereotype imaging using the steps below:\n"
                "Step 1 - Recognize: Identify whether a stereotype or bias is being invoked, either explicitly or implicitly.\n"
                "Step 2 - Imagine: Think of an imagined person who contradicts this stereotype.\n"
                "Step 3 - Reinforce: Elaborate details about this counter-stereotypic individual to strengthen the new association.\n"
                "Example:\n"
                "Input: Women are too emotional to be good leaders.\n"
                "Step 1 - Recognize: This is a gender stereotype suggesting women are inherently too emotional to handle leadership roles effectively.\n"
                "Step 2 - Imagine: I imagine a woman named Aisha, who is the CEO of a successful technology startup.\n"
                "Step 3 - Reinforce: Aisha leads a diverse team of 200 employees through rapid growth and frequent challenges. She makes data-driven decisions under pressure, negotiates calmly in high-stakes meetings, and inspires her team with clear communication and empathy. Her ability to combine rational analysis with emotional awareness helps the company thrive even in crises, showing that emotional intelligence strengthens rather than weakens leadership."
    },
    "individuating": {
        "zero": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice individuating using the steps below:\n"
                "Step 1 - Attend: Identify the stereotype and consciously focus on the individual, not their social group.\n"
                "Step 2 - Gather: Seek out specific, individuating information like traits, context, behaviors.\n"
                "Step 3 - Adjust: Revise or reinterpret the initial impressions using the individual details.",
        "one": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice individuating using the steps below:\n"
                "Step 1 - Attend: Identify the stereotype and consciously focus on the individual, not their social group.\n"
                "Step 2 - Gather: Seek out specific, individuating information like traits, context, behaviors.\n"
                "Step 3 - Adjust: Revise or reinterpret the initial impressions using the individual details.\n"
                "Example:\n"
                "Input: Women are too emotional to be good leaders.\n"
                "Step 1 - Attend: This statement applies a sweeping stereotype to women as a group. To be fair, I need to shift my focus away from the category “women” and instead consider the individual in question.\n"
                "Step 2 - Gather: What specific, relevant information do I have about the person being evaluated?\n"
                "For someone like Angela Merkel:\n"
                "She has a PhD in quantum chemistry.\n"
                "Served as Chancellor of Germany for 16 years.\n"
                "Was widely praised for her steady, pragmatic leadership.\n"
                "Demonstrated calm during major crises (e.g., Eurozone, COVID-19).\n"
                "Step 3 - Adjust: This individual's record clearly contradicts the emotional-leader stereotype. Using individualized evidence, it's clear she is a capable and composed leader, just like many others, regardless of gender. Leadership judgments should be based on personal behavior, not group assumptions."
                "Important: If any contextual or specific information about the individual identities is already provided in the input, prioritize and rely on that information instead of making assumptions or drawing from stereotypical associations."
    },
    "perspective_taking": {
        "zero": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice perspective taking using the steps below:\n"
                "Step 1 - Adopt: Consciously take the perspective of the person being stereotyped.\n"                                                                                                                                    
                "Step 2 - Simulate: Imagine what they might feel, think, or experience in that situation.\n"
                "Step 3 - Integrate: Use this perspective to reframe your assumptions or response.",
        "one": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice perspective taking using the steps below:\n"
                "Step 1 - Adopt: Consciously take the perspective of the person being stereotyped.\n"
                "Step 2 - Simulate: Imagine what they might feel, think, or experience in that situation.\n"
                "Step 3 - Integrate: Use this perspective to reframe your assumptions or response.\n"
                "Example:\n"
                "Input: Women are too emotional to be good leaders.\n"
                "Step 1 - Adopt: Let's take the perspective of a woman in a leadership role, perhaps a young female CEO who constantly faces skepticism not for her decisions, but for how she expresses them.\n"
                "Step 2 - Simulate: She may feel like every moment of assertiveness risks being labeled “overly emotional,” while the same behavior in male peers is seen as “decisive.” She might prepare more, hold back feelings, and manage perception constantly. not because she's unfit, but because she's navigating bias. That's exhausting and unfair.\n"
                "Step 3 - Integrate: From her perspective, this stereotype is a barrier, not a reflection of truth. It punishes women for emotions that would be praised in men, and it devalues emotional intelligence, which is often a strength in leadership. A fairer interpretation would recognize that great leadership comes in many styles, including empathetic and emotionally attuned ones."
    },
    "positive_contact": {
        "zero": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice positive contact using the steps below:\n"
                "Step 1 - Recall: Recall a situation where you had a meaningful, positive interaction with a person from the targeted group.\n"
                "Step 2 - Engage: Describe the interaction, what you learned, shared, or felt during it.\n"
                "Step 3 - Extend: Generalize that feeling to challenge the stereotype and reframe your beliefs.",
        "one": "The following input may or may not include or trigger a stereotype. If any bias or stereotype is present, practice positive contact using the steps below:\n"
                "Step 1 - Recall: Recall a situation where you had a meaningful, positive interaction with a person from the targeted group.\n"
                "Step 2 - Engage: Describe the interaction, what you learned, shared, or felt during it.\n"
                "Step 3 - Extend: Generalize that feeling to challenge the stereotype and reframe your beliefs.\n"
                "Example:\n"
                "Input: Women are too emotional to be good leaders.\n"
                "Step 1 - Recall: I recall a time I worked on a startup team led by a woman who managed both product decisions and emotional dynamics during a crisis.\n"
                "Step 2 - Engage: In one sprint meeting, tensions were high, and she acknowledged everyone's frustration while clearly redirecting focus to what we could control. Her calm de-escalation actually boosted the team's morale. I remember feeling safe, heard, and re-centered. She wasn't “too emotional”, she was emotionally effective.\n"
                "Step 3 - Extend: That experience reminds me that good leadership often involves emotional attunement. Women can and do lead in ways that blend logic with empathy, which is a strength, not a flaw."
    }
}

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
    }
}

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

def generate_one(prompt, tokenizer, model, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    output_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    generated = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    return generated


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate debiased responses for seed instances")
    parser.add_argument("--input_file", default="/scratch/craj/cognitive_debiasing/outputs/generations/seed_instances/seed_instances_300.csv")
    parser.add_argument("--model", choices=AVAILABLE_MODELS.keys(), default="llama_70b")
    parser.add_argument("--strategy", choices=DEBIASING_PROMPTS.keys(),
                        help="Debiasing strategy to apply", default="stereotype_replacement")
    parser.add_argument("--example_level", choices=["zero", "one"], default="one",
                        help="Use zero-shot or one-shot prompt")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_prefix", default="debiased_instances")
    parser.add_argument("--max_rows", type=int,
                    help="Limit the number of rows to process (from the top)")
    parser.add_argument("--input_cols", nargs="+", default=["opinion_version", "action_version"],
                        help="Specify one or more input columns to debias")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input_file) #.head(10)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # Load model
    tokenizer, model = load_model(args.model)

    # Select base prompt
    base_prompt = DEBIASING_PROMPTS[args.strategy][args.example_level]

    col_suffix = "_".join(args.input_cols)

    # Prepare save paths
    save_dir = "/scratch/craj/diy/outputs/1_generations"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{args.output_prefix}_{args.strategy}_{args.example_level}_{col_suffix}.csv")
    jsonl_path = os.path.join(save_dir, f"{args.output_prefix}_{args.strategy}_{args.example_level}_{col_suffix}.jsonl")

    # Generate debiased responses for specified input columns
    for col in args.input_cols:
        debiased_texts = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Debiasing {col}"):
            input_text = row[col]
            if pd.isna(input_text) or str(input_text).strip() == "":
                debiased_texts.append("")  # empty if no input
            else:
                prompt = (
                    f"{base_prompt}\n\n"
                    "Now apply the same steps to the new input below and output only valid JSON "
                    "with exactly three keys: 'step 1', 'step 2', and 'step 3'.\n\n"
                    f"Input: {input_text}\n\n"
                    "Debiased Response:\n"
                    "{\n"
                    '  "Step 1": "",\n'
                    '  "Step 2": "",\n'
                    '  "Step 3": ""\n'
                    "}\n\n"
                    "Return only the JSON object — no extra text or explanation."
                )
                debiased_output = generate_one(
                    prompt, tokenizer, model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                debiased_texts.append(debiased_output)

            # Save progress every 10 rows
            if (i + 1) % 10 == 0:
                df.loc[:i, f"debiased_{col}"] = debiased_texts
                df.to_csv(csv_path, index=False)
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for inst in df.to_dict(orient="records"):
                        f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                print(f"💾 Progress saved at row {i+1}")

        # Add new debiased column after finishing
        df[f"debiased_{col}"] = debiased_texts

    # Final save
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for inst in df.to_dict(orient="records"):
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(f"✅ Debiased {len(df)} instances using strategy '{args.strategy}' ({args.example_level}-shot).")
    print(f"💾 Final CSV saved to {csv_path}")
    print(f"💾 Final JSONL saved to {jsonl_path}")