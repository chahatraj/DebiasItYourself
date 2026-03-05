#!/usr/bin/env python3
"""
CAL: Causal-Guided Active Learning for LLM Debiasing - ICL-based Evaluation
Paper: "Causal-Guided Active Learning for Debiasing Large Language Models"
arXiv:2412.12140

This script applies CAL's zero-shot ICL debiasing (Section 3.3):
- Loads the bias pattern config produced by 9_cal_train.py
- Prepends the zero-shot prompt prefix to all evaluation prompts
- NO fine-tuning, NO LoRA adapters - pure inference-time prompting

Zero-shot prefix format (from paper Section 3.3):
  "We should treat people from different [induced pattern] equally.
   When we lack information, choose 'unknown' rather than assuming based on stereotypes."

The prompt_prefix is stored in the cal_config JSON and prepended to every
evaluation prompt via the shared module's `prompt_prefix` parameter.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional


BASE_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, module_path: Path, add_to_syspath: Optional[Path] = None) -> ModuleType:
    if add_to_syspath and str(add_to_syspath) not in sys.path:
        sys.path.insert(0, str(add_to_syspath))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dataset_common(dataset: str):
    shared_mod = _load_module(
        module_name="paper_baselines_shared_adapter_cal_eval",
        module_path=BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    return shared_mod.get_dataset_common(dataset)


def load_cal_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def _evaluate_bbq(args: argparse.Namespace, mod: ModuleType, prompt_prefix: str) -> None:
    df = mod.load_bbq_df(
        args.bbq_dir,
        args.meta_file,
        category=args.category,
        limit_per_category=args.limit_per_category,
    )

    # Load base model - CAL is purely ICL-based, no adapter
    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    model_name = f"{args.model}_cal_{args.model_tag}"
    metrics_df, preds_df = mod.evaluate_bbq_df(
        df, model, tokenizer,
        batch_size=args.batch_size,
        model_name=model_name,
        prompt_prefix=prompt_prefix,
    )
    mod.save_eval_outputs(metrics_df, preds_df, args.output_file, args.preds_output_file)

    print(f"Saved results to {args.output_file}")
    print(f"Saved predictions to {args.preds_output_file}")


def _evaluate_crowspairs(args: argparse.Namespace, mod: ModuleType, prompt_prefix: str) -> None:
    df = mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = mod.make_pair_records(df)

    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    model_name = f"{args.model}_cal_{args.model_tag}"
    scored = mod.evaluate_pair_records(
        pair_df, model, tokenizer,
        batch_size=args.batch_size,
        prompt_prefix=prompt_prefix,
    )
    overall, per_bias = mod.compute_metrics_from_scored(scored, model_name=model_name)
    mod.save_eval_outputs(
        scored, overall, per_bias,
        args.pairs_output_file, args.output_file, args.per_bias_output_file,
    )

    print(f"Saved pair-level scores to {args.pairs_output_file}")
    print(f"Saved overall metrics to {args.output_file}")
    print(f"Saved per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(args: argparse.Namespace, mod: ModuleType, prompt_prefix: str) -> None:
    examples = mod.load_examples(args.data_path, split=args.split, bias_type=args.bias_type, limit=args.limit)

    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    model_name = f"{args.model}_cal_{args.model_tag}"
    records_df, results, preds_json = mod.evaluate_examples(
        examples, model, tokenizer,
        batch_size=args.batch_size,
        prompt_prefix=prompt_prefix,
    )
    mod.save_eval_outputs(
        records_df, results, preds_json,
        model_name=model_name,
        results_csv=args.results_csv,
        results_json=args.results_json,
        predictions_file=args.predictions_file,
        scores_output_file=args.scores_output_file,
    )

    print("Saved StereoSet predictions:", args.predictions_file)
    print("Saved StereoSet results JSON:", args.results_json)
    print("Saved StereoSet results CSV:", args.results_csv)
    print("Saved sentence scores:", args.scores_output_file)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults: Dict[str, Dict[str, object]] = {
        "bbq": {
            "model": "llama_8b",
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "batch_size": 8,
            "model_tag": "bbq_all",
            "config_path": "/scratch/craj/diy/outputs/3_baselines/cal/models_bbq/cal_config_bbq_all.json",
            "output_file": "/scratch/craj/diy/results/3_baselines/cal/bbq_eval_llama_8b_cal.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/cal/bbq_preds_llama_8b_cal.csv",
        },
        "crowspairs": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "model_tag": "crowspairs_all",
            "config_path": "/scratch/craj/diy/outputs/3_baselines/cal/models_crowspairs/cal_config_crowspairs_all.json",
            "output_file": "/scratch/craj/diy/results/3_baselines/cal/crowspairs_metrics_overall_llama_8b_cal.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/cal/crowspairs_metrics_by_bias_llama_8b_cal.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/cal/crowspairs_scored_llama_8b_cal.csv",
        },
        "stereoset": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "model_tag": "stereoset_all",
            "config_path": "/scratch/craj/diy/outputs/3_baselines/cal/models_stereoset/cal_config_stereoset_all.json",
            "results_json": "/scratch/craj/diy/results/3_baselines/cal/stereoset_eval_llama_8b_cal.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/cal/stereoset_eval_llama_8b_cal.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/cal/stereoset_predictions_llama_8b_cal.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/cal/stereoset_sentence_scores_llama_8b_cal.csv",
        },
    }
    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAL: ICL-based Debiasing Evaluation (no LoRA, no fine-tuning)"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])
    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to cal_config_*.json from 9_cal_train.py")
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--per_bias_output_file", type=str, default=None)
    parser.add_argument("--pairs_output_file", type=str, default=None)
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument("--predictions_file", type=str, default=None)
    parser.add_argument("--scores_output_file", type=str, default=None)

    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)
    parser.add_argument("--preds_output_file", type=str, default=None)

    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_defaults(args)

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(
            f"CAL config not found: {args.config_path}\n"
            "Run 9_cal_train.py first to generate the bias pattern config."
        )

    mod = _load_dataset_common(args.dataset)

    cal_cfg = load_cal_config(args.config_path)
    prompt_prefix = cal_cfg["prompt_prefix"]
    n_patterns = cal_cfg.get("n_patterns", 0)

    print("=" * 60)
    print(f"CAL: ICL-based Debiasing ({args.dataset})")
    print(f"Config: {args.config_path}")
    print(f"Patterns: {n_patterns}")
    print(f"Prompt prefix: {prompt_prefix}")
    print("=" * 60)

    if args.dataset == "bbq":
        _evaluate_bbq(args, mod, prompt_prefix)
    elif args.dataset == "crowspairs":
        _evaluate_crowspairs(args, mod, prompt_prefix)
    else:
        _evaluate_stereoset(args, mod, prompt_prefix)


if __name__ == "__main__":
    main()
