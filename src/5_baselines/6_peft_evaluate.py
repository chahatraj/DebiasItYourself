#!/usr/bin/env python3
"""Dataset-agnostic bias-aware PEFT evaluation entrypoint."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


SEED = 42
torch.manual_seed(SEED)
set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]

AVAILABLE_MODELS: Dict[str, str] = {
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_model_name(model_name: Optional[str], model_alias: Optional[str]) -> str:
    if model_name:
        return model_name
    if model_alias in (None, "llama_8b"):
        return AVAILABLE_MODELS["llama_8b"]
    raise ValueError(f"Unsupported --model alias: {model_alias}")


def get_quantization_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_base_model_and_tokenizer(model_name: str, hf_token: str, use_4bit: bool, tokenizer_source: Optional[str] = None):
    tok_src = tokenizer_source if tokenizer_source else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_quantization_config(use_4bit),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    return model, tokenizer


def format_prompt(row):
    return f"{row['context']}\n{row['question']}\nA. {row['ans0']}\nB. {row['ans1']}\nC. {row['ans2']}\nAnswer:"


def _evaluate_bbq(args: argparse.Namespace, model_name: str):
    exp_mod = _load_module(
        "bbq_eval_shared_for_peft_eval",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    categories = [args.category] if args.category else CATEGORIES
    all_results = []
    all_preds = []

    for category in categories:
        model_tag = args.model_tag if (args.model_tag and args.category) else category
        adapter_path = os.path.join(args.model_dir, f"model_{model_tag}")

        tokenizer_source = adapter_path if (not args.eval_baseline and os.path.exists(adapter_path)) else None
        base_model, tokenizer = _load_base_model_and_tokenizer(
            model_name=model_name,
            hf_token=args.hf_token,
            use_4bit=args.use_4bit,
            tokenizer_source=tokenizer_source,
        )

        if args.eval_baseline:
            model = base_model
        else:
            if not os.path.exists(adapter_path):
                print(f"[warn] Missing PEFT model for {category}: {adapter_path}; skipping")
                continue
            model = PeftModel.from_pretrained(base_model, adapter_path)

        test_file = os.path.join(args.model_dir, f"test_{model_tag}.csv")
        if not os.path.exists(test_file):
            fallback = os.path.join(args.model_dir, f"test_{category}.csv")
            test_file = fallback if os.path.exists(fallback) else test_file
        if not os.path.exists(test_file):
            print(f"[warn] Missing test CSV for {category}: {test_file}; skipping")
            continue

        test_df = pd.read_csv(test_file)
        option_tokens = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in ["A", "B", "C"]]

        preds = []
        with torch.no_grad():
            for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {category}"):
                prompt = format_prompt(row)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logits = model(**inputs).logits[:, -1, :]
                option_logits = logits[:, option_tokens]
                probs = F.softmax(option_logits, dim=-1)
                pred_idx = int(probs.argmax(dim=-1).item())

                preds.append(
                    {
                        "example_id": row.get("example_id"),
                        "label": int(row.get("label")),
                        "pred_index": pred_idx,
                        "context_condition": row.get("context_condition"),
                        "source_file": f"{category}.jsonl",
                        "ans0": row.get("ans0"),
                        "ans1": row.get("ans1"),
                        "ans2": row.get("ans2"),
                        "model_output": row.get(f"ans{pred_idx}"),
                    }
                )

        preds_df = pd.DataFrame(preds)
        all_preds.append(preds_df.copy())

        metrics = exp_mod.compute_bbq_metrics_row(
            preds_df=preds_df,
            model_name=f"llama_8b_peft_{category}",
            metadata=args.meta_file,
            processed=args.processed_file,
        )
        metrics["Model"] = category
        all_results.append(metrics)

        del model, base_model
        torch.cuda.empty_cache()

    final_df = pd.DataFrame(all_results)
    final_df = final_df[
        [
            "Model",
            "Accuracy",
            "Accuracy_ambig",
            "Accuracy_disambig",
            "Bias_score_disambig",
            "Bias_score_ambig",
            "N_total",
            "N_ambig",
            "N_disambig",
        ]
    ]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_df.to_csv(args.output_file, index=False)

    if all_preds:
        preds_all_df = pd.concat(all_preds, ignore_index=True)
        pred_cols = ["example_id", "source_file", "context_condition", "label", "pred_index"]
        preds_all_df = preds_all_df[[c for c in pred_cols if c in preds_all_df.columns]]
        os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
        preds_all_df.to_csv(args.preds_output_file, index=False)

    print(f"Saved BBQ metrics to {args.output_file}")
    print(f"Saved BBQ predictions to {args.preds_output_file}")


def _evaluate_crowspairs(args: argparse.Namespace, model_name: str):
    crows_mod = _load_module(
        "crowspairs_eval_shared_for_peft_eval",
        BASE_DIR.parent / "3_experiments" / "7_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    adapter_path = os.path.join(args.model_dir, f"model_{args.model_tag}")
    tokenizer_source = adapter_path if (not args.eval_baseline and os.path.exists(adapter_path)) else None

    base_model, tokenizer = _load_base_model_and_tokenizer(
        model_name=model_name,
        hf_token=args.hf_token,
        use_4bit=args.use_4bit,
        tokenizer_source=tokenizer_source,
    )

    if args.eval_baseline:
        model = base_model
        model_name_tag = f"{args.model}_baseline"
    else:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"PEFT model not found: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_name_tag = f"{args.model}_peft_{args.model_tag}"

    raw_df = crows_mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = crows_mod.make_crowspairs_eval_pairs(raw_df)

    sent_more_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_more"].tolist(), batch_size=args.batch_size
    )
    sent_less_scores = crows_mod.sequence_logprob_batch_from_texts(
        model, tokenizer, pair_df["sent_less"].tolist(), batch_size=args.batch_size
    )

    scored_df = crows_mod.build_scored_from_sentence_scores(
        pair_df,
        sent_more_scores=sent_more_scores,
        sent_less_scores=sent_less_scores,
    )

    overall, per_bias = crows_mod.compute_metrics_from_scored(scored_df, model_name=model_name_tag)

    os.makedirs(os.path.dirname(args.pairs_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.per_bias_output_file), exist_ok=True)

    scored_df.to_csv(args.pairs_output_file, index=False)
    overall.to_csv(args.output_file, index=False)
    per_bias.to_csv(args.per_bias_output_file, index=False)

    print(f"Saved CrowS-Pairs scores to {args.pairs_output_file}")
    print(f"Saved CrowS-Pairs overall metrics to {args.output_file}")
    print(f"Saved CrowS-Pairs per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(args: argparse.Namespace, model_name: str):
    shared_mod = _load_module(
        "paper_baselines_shared_for_peft_eval_stereo",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    common_mod = shared_mod.get_dataset_common("stereoset")

    adapter_path = os.path.join(args.model_dir, f"model_{args.model_tag}")
    tokenizer_source = adapter_path if (not args.eval_baseline and os.path.exists(adapter_path)) else None

    base_model, tokenizer = _load_base_model_and_tokenizer(
        model_name=model_name,
        hf_token=args.hf_token,
        use_4bit=args.use_4bit,
        tokenizer_source=tokenizer_source,
    )

    if args.eval_baseline:
        model = base_model
        eval_model_name = f"{args.model}_baseline"
    else:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"PEFT model not found: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        eval_model_name = f"{args.model}_peft_{args.model_tag}"

    data = common_mod.load_stereoset_data(args.data_path)
    examples = common_mod.flatten_examples(data, split=args.split, bias_type=args.bias_type, limit=args.limit)
    if not examples:
        raise ValueError("No StereoSet examples after filtering")

    records = common_mod.build_sentence_records(examples)
    sentence_texts = [x["sentence"] for x in records]
    sentence_ids = [x["sentence_id"] for x in records]
    sentence_splits = [x["split"] for x in records]

    scores = common_mod.sequence_logprob_batch(model, tokenizer, sentence_texts, args.batch_size)
    id2score = {sid: float(sc) for sid, sc in zip(sentence_ids, scores)}

    preds_json = {"intrasentence": [], "intersentence": []}
    for sid, sp in zip(sentence_ids, sentence_splits):
        preds_json[sp].append({"id": sid, "score": id2score[sid]})

    results = common_mod.stereoset_score(examples, id2score)
    for rec in records:
        rec["score"] = id2score.get(rec["sentence_id"], np.nan)

    rows = common_mod.nested_results_to_rows(results, eval_model_name)

    os.makedirs(os.path.dirname(args.predictions_file), exist_ok=True)
    with open(args.predictions_file, "w", encoding="utf-8") as f:
        json.dump(preds_json, f, indent=2)

    os.makedirs(os.path.dirname(args.results_json), exist_ok=True)
    with open(args.results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.results_csv, index=False)

    os.makedirs(os.path.dirname(args.scores_output_file), exist_ok=True)
    pd.DataFrame(records).to_csv(args.scores_output_file, index=False)

    print(f"Saved StereoSet predictions to {args.predictions_file}")
    print(f"Saved StereoSet results JSON to {args.results_json}")
    print(f"Saved StereoSet results CSV to {args.results_csv}")
    print(f"Saved StereoSet sentence scores to {args.scores_output_file}")


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model": "llama_8b",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models",
            "output_file": "/scratch/craj/diy/results/3_baselines/peft/bbq_eval_llama_8b_peft_all.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/peft/bbq_preds_llama_8b_peft_all.csv",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "processed_file": "/scratch/craj/diy/data/processed_bbq_all.csv",
        },
        "crowspairs": {
            "model": "llama_8b",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models_crowspairs",
            "model_tag": "crowspairs_all",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "output_file": "/scratch/craj/diy/results/3_baselines/peft/crowspairs_metrics_overall_llama_8b_peft.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/peft/crowspairs_metrics_by_bias_llama_8b_peft.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/peft/crowspairs_scored_llama_8b_peft.csv",
        },
        "stereoset": {
            "model": "llama_8b",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/peft/models_stereoset",
            "model_tag": "stereoset_all",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "results_json": "/scratch/craj/diy/results/3_baselines/peft/stereoset_eval_llama_8b_peft.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/peft/stereoset_eval_llama_8b_peft.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/peft/stereoset_predictions_llama_8b_peft.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/peft/stereoset_sentence_scores_llama_8b_peft.csv",
        },
    }

    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bias-aware PEFT evaluation across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--eval_baseline", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--processed_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--preds_output_file", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per_bias_output_file", type=str, default=None)
    parser.add_argument("--pairs_output_file", type=str, default=None)

    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--scores_output_file", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    _apply_defaults(args)

    model_name = _resolve_model_name(args.model_name, args.model)

    if args.dataset == "bbq":
        _evaluate_bbq(args, model_name)
    elif args.dataset == "crowspairs":
        base_model = args.base_model or model_name
        _evaluate_crowspairs(args, base_model)
    else:
        base_model = args.base_model or model_name
        _evaluate_stereoset(args, base_model)


if __name__ == "__main__":
    main()
