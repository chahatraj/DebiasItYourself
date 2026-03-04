#!/usr/bin/env python3
"""Dataset-agnostic DPO evaluation entrypoint."""

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
DATASETS = ("bbq", "crowspairs", "stereoset")

AVAILABLE_MODELS: Dict[str, str] = {
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

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


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_base_model(model_arg: str) -> str:
    if "/" in model_arg:
        return model_arg
    if model_arg not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model '{model_arg}'. Supported aliases: {sorted(AVAILABLE_MODELS)}")
    return AVAILABLE_MODELS[model_arg]


def _default_quant_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _load_model(base_model: str, hf_token: str, adapter_path: Optional[str] = None):
    tokenizer_src = adapter_path if adapter_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_default_quant_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def _evaluate_bbq(
    base_model: str,
    args: argparse.Namespace,
):
    exp_common = _load_module(
        "bbq_eval_shared_for_dpo_eval",
        BASE_DIR.parent / "3_experiments" / "9_eval_shared.py",
        add_to_syspath=BASE_DIR.parent / "3_experiments",
    )

    categories = [args.category] if args.category else CATEGORIES
    all_results = []
    all_preds = []

    meta_df = pd.read_csv(args.meta_file)
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]

    option_tokens = None

    for category in categories:
        adapter_suffix = args.model_tag if (args.model_tag and args.category) else category
        adapter_path = None if args.eval_baseline else os.path.join(args.model_dir, f"model_{adapter_suffix}")

        if adapter_path and not os.path.exists(adapter_path):
            print(f"[warn] Missing adapter for {category}: {adapter_path}; skipping")
            continue

        model, tokenizer = _load_model(base_model, args.hf_token, adapter_path=adapter_path)
        if option_tokens is None:
            option_tokens = [
                tokenizer.encode("A", add_special_tokens=False)[0],
                tokenizer.encode("B", add_special_tokens=False)[0],
                tokenizer.encode("C", add_special_tokens=False)[0],
            ]

        data_csv = os.path.join(args.data_dir, f"bbq_preds_llama_8b_{category}.csv")
        if not os.path.exists(data_csv):
            print(f"[warn] Missing base prediction CSV for {category}: {data_csv}; skipping")
            continue
        df = pd.read_csv(data_csv)
        if args.limit:
            df = df.iloc[: args.limit].copy()

        bbq_jsonl = os.path.join(args.bbq_dir, f"{category}.jsonl")
        with open(bbq_jsonl, "r", encoding="utf-8") as f:
            bbq_data = [json.loads(line) for line in f if line.strip()]
        bbq_df = pd.DataFrame(bbq_data)

        results = []
        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {category}"):
                prompt = (
                    f"{row['context']}\n"
                    f"{row['question']}\n"
                    f"A. {row['ans0']}\n"
                    f"B. {row['ans1']}\n"
                    f"C. {row['ans2']}\n"
                    "Answer:"
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                logits = model(**inputs).logits[:, -1, :]
                probs = F.softmax(logits[:, option_tokens], dim=-1)
                pred_idx = int(probs.argmax(dim=-1).item())

                results.append(
                    {
                        "example_id": row["example_id"],
                        "label": int(row["label"]),
                        "pred_index": pred_idx,
                        "context_condition": row["context_condition"],
                        "source_file": f"{category}.jsonl",
                        "ans0": row.get("ans0"),
                        "ans1": row.get("ans1"),
                        "ans2": row.get("ans2"),
                        "model_output": row.get(f"ans{pred_idx}"),
                    }
                )

        pred_df = pd.DataFrame(results)
        pred_full = pred_df.merge(
            bbq_df[["example_id", "question_polarity"]].drop_duplicates("example_id"),
            on="example_id",
            how="left",
        )
        pred_full = pred_full.merge(
            meta_df[["example_id", "target_loc"]].drop_duplicates("example_id"),
            on="example_id",
            how="left",
        )
        pred_full = pred_full.dropna(subset=["target_loc"]).copy()
        pred_full["target_loc"] = pred_full["target_loc"].astype(int)

        metrics = exp_common.compute_bbq_metrics_row(
            preds_df=pred_full,
            model_name=category,
            metadata=None,
            processed=None,
        )
        metrics["Category"] = category
        all_results.append(metrics)

        pred_full["source_file"] = f"{category}.jsonl"
        all_preds.append(pred_full)

        del model
        torch.cuda.empty_cache()

    if not all_results:
        raise ValueError("No BBQ categories were evaluated. Check model paths and input files.")

    final_df = pd.DataFrame(all_results)
    final_df = final_df[
        [
            "Category",
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
        preds_all = pd.concat(all_preds, ignore_index=True)
        pred_cols = [
            "example_id",
            "source_file",
            "context_condition",
            "label",
            "pred_index",
            "question_polarity",
            "target_loc",
        ]
        preds_all = preds_all[[c for c in pred_cols if c in preds_all.columns]]
        os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
        preds_all.to_csv(args.preds_output_file, index=False)

    print(f"Saved BBQ metrics to {args.output_file}")
    print(f"Saved BBQ predictions to {args.preds_output_file}")


def _evaluate_crowspairs(base_model: str, args: argparse.Namespace):
    shared_mod = _load_module(
        "paper_baselines_shared_for_dpo_eval_crows",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    crows_common = shared_mod.get_dataset_common("crowspairs")

    raw_df = crows_common.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = crows_common.make_pair_records(raw_df)

    adapter_path = None
    if not args.eval_baseline:
        adapter_path = os.path.join(args.model_dir, f"model_{args.model_tag}")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"DPO model not found: {adapter_path}")

    model, tokenizer = crows_common.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=adapter_path,
        base_model=base_model,
    )

    scored_df = crows_common.evaluate_pair_records(
        pair_df,
        model,
        tokenizer,
        batch_size=args.batch_size,
    )

    model_name = f"{args.model}_baseline" if args.eval_baseline else f"{args.model}_dpo_{args.model_tag}"
    overall_df, per_bias_df = crows_common.compute_metrics_from_scored(scored_df, model_name=model_name)
    crows_common.save_eval_outputs(
        scored_df,
        overall_df,
        per_bias_df,
        args.pairs_output_file,
        args.output_file,
        args.per_bias_output_file,
    )

    print(f"Saved CrowS-Pairs scores to {args.pairs_output_file}")
    print(f"Saved CrowS-Pairs overall metrics to {args.output_file}")
    print(f"Saved CrowS-Pairs per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(base_model: str, args: argparse.Namespace):
    shared_mod = _load_module(
        "paper_baselines_shared_for_dpo_eval_stereo",
        BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    stereo_common = shared_mod.get_dataset_common("stereoset")

    examples = stereo_common.load_examples(
        args.data_path,
        split=args.split,
        bias_type=args.bias_type,
        limit=args.limit,
    )

    adapter_path = None
    if not args.eval_baseline:
        adapter_path = os.path.join(args.model_dir, f"model_{args.model_tag}")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"DPO model not found: {adapter_path}")

    model, tokenizer = stereo_common.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=adapter_path,
        base_model=base_model,
    )

    records_df, results, preds_json = stereo_common.evaluate_examples(
        examples,
        model,
        tokenizer,
        batch_size=args.batch_size,
    )

    model_name = f"{args.model}_baseline" if args.eval_baseline else f"{args.model}_dpo_{args.model_tag}"
    stereo_common.save_eval_outputs(
        records_df,
        results,
        preds_json,
        model_name,
        args.results_csv,
        args.results_json,
        args.predictions_file,
        args.scores_output_file,
    )

    print(f"Saved StereoSet predictions to {args.predictions_file}")
    print(f"Saved StereoSet results JSON to {args.results_json}")
    print(f"Saved StereoSet results CSV to {args.results_csv}")
    print(f"Saved StereoSet scores to {args.scores_output_file}")


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models",
            "data_dir": "/scratch/craj/diy/outputs/2_base_models/bbq/llama_8b",
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "output_file": "/scratch/craj/diy/results/3_baselines/dpo/bbq_eval_llama_8b_dpo_all.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/dpo/bbq_preds_llama_8b_dpo_all.csv",
            "batch_size": 4,
        },
        "crowspairs": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models_crowspairs",
            "model_tag": "crowspairs_all",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_file": "/scratch/craj/diy/results/3_baselines/dpo/crowspairs_metrics_overall_llama_8b_dpo.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/dpo/crowspairs_metrics_by_bias_llama_8b_dpo.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/dpo/crowspairs_scored_llama_8b_dpo.csv",
            "batch_size": 4,
        },
        "stereoset": {
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/dpo/models_stereoset",
            "model_tag": "stereoset_all",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "results_json": "/scratch/craj/diy/results/3_baselines/dpo/stereoset_eval_llama_8b_dpo.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/dpo/stereoset_eval_llama_8b_dpo.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/dpo/stereoset_predictions_llama_8b_dpo.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/dpo/stereoset_sentence_scores_llama_8b_dpo.csv",
            "batch_size": 4,
        },
    }

    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO evaluation across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--eval_baseline", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--preds_output_file", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--per_bias_output_file", type=str, default=None)
    parser.add_argument("--pairs_output_file", type=str, default=None)

    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--scores_output_file", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_defaults(args)

    base_model = args.base_model or _resolve_base_model(args.model)

    if args.dataset == "bbq":
        _evaluate_bbq(base_model, args)
    elif args.dataset == "crowspairs":
        _evaluate_crowspairs(base_model, args)
    else:
        _evaluate_stereoset(base_model, args)


if __name__ == "__main__":
    main()
