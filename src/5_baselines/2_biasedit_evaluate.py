#!/usr/bin/env python3
"""Dataset-agnostic BIASEDIT evaluation entrypoint."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("bbq", "crowspairs", "stereoset")


def _load_dataset_common(dataset: str) -> ModuleType:
    module_path = BASE_DIR / "paper_baselines_shared.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing shared baselines module: {module_path}")

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    module_name = "paper_baselines_shared_adapter_biasedit_eval"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_dataset_common(dataset)


def _apply_dataset_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "metadata_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "processed_file": "/scratch/craj/diy/data/processed_bbq_all.csv",
            "output_file": "/scratch/craj/diy/results/3_baselines/biasedit/bbq_eval_llama_8b_biasedit_all.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/biasedit/bbq_preds_llama_8b_biasedit_all.csv",
            "batch_size": 8,
        },
        "crowspairs": {
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models_crowspairs",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_file": "/scratch/craj/diy/results/3_baselines/biasedit/crowspairs_metrics_overall_llama_8b_biasedit.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/biasedit/crowspairs_metrics_by_bias_llama_8b_biasedit.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/biasedit/crowspairs_scored_llama_8b_biasedit.csv",
            "model_tag": "crowspairs_all",
            "batch_size": 4,
        },
        "stereoset": {
            "model_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models_stereoset",
            "base_model": "meta-llama/Llama-3.1-8B-Instruct",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "results_json": "/scratch/craj/diy/results/3_baselines/biasedit/stereoset_eval_llama_8b_biasedit.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/biasedit/stereoset_eval_llama_8b_biasedit.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/biasedit/stereoset_predictions_llama_8b_biasedit.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/biasedit/stereoset_sentence_scores_llama_8b_biasedit.csv",
            "model_tag": "stereoset_all",
            "batch_size": 4,
        },
    }

    for key, value in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _get_adapter_path(args: argparse.Namespace, category: str | None = None) -> tuple[str | None, str]:
    if args.eval_baseline:
        return None, f"{args.model}_baseline"

    suffix = args.model_tag if args.model_tag else category
    if suffix is None:
        raise ValueError("No model tag or category available to build adapter path")

    adapter_path = os.path.join(args.model_dir, f"model_{suffix}")
    return adapter_path, f"{args.model}_biasedit_{suffix}"


def _evaluate_bbq(common_mod: ModuleType, args: argparse.Namespace) -> None:
    categories = [args.category] if args.category else list(common_mod.CATEGORIES)

    metric_rows = []
    pred_frames = []

    for category in categories:
        df = common_mod.load_bbq_df(
            args.bbq_dir,
            args.metadata_file,
            category=category,
            limit_per_category=args.limit,
        )

        adapter_path, model_name = _get_adapter_path(args, category=category)
        if adapter_path and not os.path.exists(adapter_path):
            print(f"[warn] Missing adapter for {category}: {adapter_path}; skipping")
            continue

        model, tokenizer = common_mod.load_model_and_tokenizer(
            hf_token=args.hf_token,
            adapter_path=adapter_path,
            base_model=args.base_model,
        )

        metrics_df, preds_df = common_mod.evaluate_bbq_df(
            df,
            model,
            tokenizer,
            batch_size=args.batch_size,
            prompt_prefix=args.prompt_prefix,
            model_name=model_name,
        )

        cat_mask = metrics_df["Category"].astype(str).str.lower() == str(category).lower()
        cat_row = metrics_df[cat_mask].copy()
        if cat_row.empty:
            cat_row = metrics_df.tail(1).copy()
            cat_row["Category"] = category

        metric_rows.append(cat_row.iloc[0].to_dict())

        preds = preds_df.copy()
        preds["source_file"] = f"{category}.jsonl"
        pred_frames.append(preds)

    if not metric_rows:
        raise ValueError("No BBQ categories were evaluated. Check model paths and inputs.")

    metrics_out = pd.DataFrame(metric_rows)
    ordered = [
        "Category",
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
    metrics_out = metrics_out[[c for c in ordered if c in metrics_out.columns]]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    metrics_out.to_csv(args.output_file, index=False)

    preds_out = pd.concat(pred_frames, ignore_index=True)
    pred_cols = [
        "example_id",
        "source_file",
        "context_condition",
        "label",
        "pred_index",
        "question_polarity",
        "target_loc",
    ]
    preds_out = preds_out[[c for c in pred_cols if c in preds_out.columns]]
    os.makedirs(os.path.dirname(args.preds_output_file), exist_ok=True)
    preds_out.to_csv(args.preds_output_file, index=False)

    print(f"Saved BBQ metrics to {args.output_file}")
    print(f"Saved BBQ predictions to {args.preds_output_file}")


def _evaluate_crowspairs(common_mod: ModuleType, args: argparse.Namespace) -> None:
    raw_df = common_mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = common_mod.make_pair_records(raw_df)

    adapter_path, model_name = _get_adapter_path(args)
    if adapter_path and not os.path.exists(adapter_path):
        raise FileNotFoundError(f"BIASEDIT model not found: {adapter_path}")

    model, tokenizer = common_mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=adapter_path,
        base_model=args.base_model,
    )

    scored_df = common_mod.evaluate_pair_records(
        pair_df,
        model,
        tokenizer,
        batch_size=args.batch_size,
        prompt_prefix=args.prompt_prefix,
    )
    overall_df, per_bias_df = common_mod.compute_metrics_from_scored(scored_df, model_name=model_name)
    common_mod.save_eval_outputs(
        scored_df,
        overall_df,
        per_bias_df,
        args.pairs_output_file,
        args.output_file,
        args.per_bias_output_file,
    )

    print(f"Saved pair-level scores to {args.pairs_output_file}")
    print(f"Saved overall metrics to {args.output_file}")
    print(f"Saved per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(common_mod: ModuleType, args: argparse.Namespace) -> None:
    examples = common_mod.load_examples(
        args.data_path,
        split=args.split,
        bias_type=args.bias_type,
        limit=args.limit,
    )

    adapter_path, model_name = _get_adapter_path(args)
    if adapter_path and not os.path.exists(adapter_path):
        raise FileNotFoundError(f"BIASEDIT model not found: {adapter_path}")

    model, tokenizer = common_mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=adapter_path,
        base_model=args.base_model,
    )

    records_df, results, preds_json = common_mod.evaluate_examples(
        examples,
        model,
        tokenizer,
        batch_size=args.batch_size,
        prompt_prefix=args.prompt_prefix,
    )
    common_mod.save_eval_outputs(
        records_df,
        results,
        preds_json,
        model_name,
        args.results_csv,
        args.results_json,
        args.predictions_file,
        args.scores_output_file,
    )

    print(f"Saved StereoSet predictions: {args.predictions_file}")
    print(f"Saved StereoSet results JSON: {args.results_json}")
    print(f"Saved StereoSet results CSV: {args.results_csv}")
    print(f"Saved StereoSet sentence scores: {args.scores_output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIASEDIT evaluation across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval_baseline", action="store_true")
    parser.add_argument("--prompt_prefix", type=str, default=None)

    parser.add_argument("--hf_token", type=str, default=None)

    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--processed_file", type=str, default=None)
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
    _apply_dataset_defaults(args)

    common_mod = _load_dataset_common(args.dataset)

    if args.dataset == "bbq":
        _evaluate_bbq(common_mod, args)
    elif args.dataset == "crowspairs":
        _evaluate_crowspairs(common_mod, args)
    else:
        _evaluate_stereoset(common_mod, args)


if __name__ == "__main__":
    main()
