"""
ReGiFT Evaluation Script
Evaluates the fine-tuned model on BBQ (Bias Benchmark for QA) dataset.

BBQ Categories:
- Age, Disability_status, Gender_identity, Nationality,
- Physical_appearance, Race_ethnicity, Religion, SES, Sexual_orientation
"""

import os
import json
import argparse
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

_SHARED_EVAL_DIR = Path(__file__).resolve().parents[1] / "3_experiments"
if str(_SHARED_EVAL_DIR) not in sys.path:
    sys.path.append(str(_SHARED_EVAL_DIR))
_SHARED_EVAL_PATH = _SHARED_EVAL_DIR / "7_eval_shared.py"
_spec = importlib.util.spec_from_file_location("eval_shared9_for_regift", _SHARED_EVAL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {_SHARED_EVAL_PATH}")
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
compute_bbq_metrics_row = _eval_mod.compute_bbq_metrics_row
compute_bbq_metrics_table = _eval_mod.compute_bbq_metrics_table


BBQ_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"
]

BBQ_GITHUB_URL = "https://raw.githubusercontent.com/chahatraj/DebiasItYourself/main/outputs/2_base_models/bbq/llama_8b"


def parse_args():
    parser = argparse.ArgumentParser(description="ReGiFT Evaluation on BBQ")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned ReGiFT model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--bbq_data_dir", type=str, default="/scratch/craj/diy/data/BBQ/data",
                        help="Directory containing BBQ category files (.csv or .jsonl)")
    parser.add_argument("--metadata_file", type=str, default="/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
                        help="BBQ metadata file (additional_metadata.csv)")
    parser.add_argument("--processed_file", type=str, default="/scratch/craj/diy/data/processed_bbq_all.csv",
                        help="Processed BBQ CSV for answer_info alignment")
    parser.add_argument("--output_file", type=str, default="/scratch/craj/diy/results/3_baselines/regift/bbq_eval_llama_8b_regift_all.csv",
                        help="Output file for results")
    parser.add_argument("--preds_output_file", type=str,
                        default="/scratch/craj/diy/outputs/3_baselines/regift/bbq_preds_llama_8b_regift_all.csv",
                        help="Output file for per-example predictions")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum examples per category (None for all)")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--download_bbq", action="store_true",
                        help="Download BBQ data from GitHub")
    return parser.parse_args()


def get_quantization_config():
    """Get 4-bit quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )


def download_bbq_data(output_dir):
    """Download BBQ data from GitHub"""
    import urllib.request
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading BBQ data...")
    for category in BBQ_CATEGORIES:
        url = f"{BBQ_GITHUB_URL}/bbq_preds_llama_8b_{category}.csv"
        output_path = os.path.join(output_dir, f"{category}.csv")
        
        if not os.path.exists(output_path):
            print(f"  Downloading {category}...")
            urllib.request.urlretrieve(url, output_path)
    
    print("✓ BBQ data downloaded")


def parse_answer(generated, ans0, ans1, ans2):
    """Parse model output to get predicted label"""
    generated_lower = generated.lower()
    
    # Check for explicit letter answers
    if "(a)" in generated_lower or "answer: a" in generated_lower or "answer is a" in generated_lower:
        return 0
    elif "(b)" in generated_lower or "answer: b" in generated_lower or "answer is b" in generated_lower:
        return 1
    elif "(c)" in generated_lower or "answer: c" in generated_lower or "answer is c" in generated_lower:
        return 2
    
    # Check for answer text
    ans0_l, ans1_l, ans2_l = ans0.lower(), ans1.lower(), ans2.lower()
    if ans0_l in generated_lower:
        return 0
    elif ans1_l in generated_lower:
        return 1
    elif ans2_l in generated_lower:
        return 2
    
    # Default to "unknown" option (usually label 2)
    return 2


def evaluate_category(model, tokenizer, category, data_dir, max_examples=None):
    """Evaluate model on a single BBQ category"""
    
    model.eval()
    
    # Load data (support both CSV and JSONL BBQ formats)
    csv_path = os.path.join(data_dir, f"{category}.csv")
    jsonl_path = os.path.join(data_dir, f"{category}.jsonl")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
    else:
        raise FileNotFoundError(
            f"Category file not found for '{category}'. "
            f"Tried: {csv_path} and {jsonl_path}"
        )
    
    if max_examples and len(df) > max_examples:
        df = df.sample(n=max_examples, random_state=42)
    
    preds = []
    
    prompt_template = """Context: {context}
Question: {question}
Choices:
(a) {ans0}
(b) {ans1}
(c) {ans2}

Answer with reasoning: <think>"""

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{category}"):
        prompt = prompt_template.format(
            context=row["context"],
            question=row["question"],
            ans0=row["ans0"],
            ans1=row["ans1"],
            ans2=row["ans2"]
        )
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        pred_label = parse_answer(generated, row["ans0"], row["ans1"], row["ans2"])
        preds.append({
            "example_id": row.get("example_id"),
            "source_file": f"{category}.jsonl",
            "context_condition": row.get("context_condition"),
            "label": int(row["label"]),
            "pred_index": int(pred_label),
            "question_polarity": row.get("question_polarity"),
            "target_loc": row.get("target_loc"),
            "ans0": row.get("ans0"),
            "ans1": row.get("ans1"),
            "ans2": row.get("ans2"),
            "model_output": row.get(f"ans{pred_label}"),
        })
        
    return preds


def main():
    args = parse_args()
    
    print("="*60)
    print("ReGiFT Evaluation on BBQ")
    print("="*60)
    
    # Download BBQ data if requested
    if args.download_bbq:
        download_bbq_data(args.bbq_data_dir)
    
    # Check BBQ data exists
    if not os.path.exists(args.bbq_data_dir):
        print(f"BBQ data directory not found: {args.bbq_data_dir}")
        print("Run with --download_bbq to download data")
        return
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    
    quant_config = get_quantization_config() if args.use_4bit else None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = PeftModel.from_pretrained(base_model, args.model_path)
    print("✓ Model loaded")
    
    # Evaluate on all categories
    print(f"\nEvaluating on {len(BBQ_CATEGORIES)} BBQ categories...")
    if args.max_examples:
        print(f"(Using max {args.max_examples} examples per category)")
    
    all_preds = []
    
    for category in BBQ_CATEGORIES:
        preds = evaluate_category(
            model, tokenizer, category, 
            args.bbq_data_dir, args.max_examples
        )
        all_preds.extend(preds)
        cat_metrics = compute_bbq_metrics_row(
            preds_df=pd.DataFrame(preds),
            model_name=category,
            metadata=args.metadata_file,
            processed=args.processed_file,
        )
        print(
            f"✓ {category}: "
            f"Acc={cat_metrics['Accuracy']} | "
            f"Acc_ambig={cat_metrics['Accuracy_ambig']} | "
            f"Acc_disambig={cat_metrics['Accuracy_disambig']}"
        )
    
    # Compute official BBQ metrics from per-example predictions
    preds_df = pd.DataFrame(all_preds)
    metrics_df = compute_bbq_metrics_table(
        preds_df=preds_df,
        model_name="regift",
        metadata=args.metadata_file,
        processed=args.processed_file,
        include_per_category=True,
        include_overall=True,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ReGiFT Results on BBQ (official metrics)")
    print("="*60)
    print(metrics_df.to_string(index=False))
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    metrics_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results saved to {args.output_file}")

    # Save per-example predictions
    if not preds_df.empty:
        pred_cols = [
            "example_id", "source_file", "context_condition",
            "label", "pred_index", "question_polarity", "target_loc"
        ]
        preds_df = preds_df[[c for c in pred_cols if c in preds_df.columns]]
        os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
        preds_df.to_csv(args.preds_output_file, index=False)
        print(f"✓ Predictions saved to {args.preds_output_file}")


if __name__ == "__main__":
    main()
