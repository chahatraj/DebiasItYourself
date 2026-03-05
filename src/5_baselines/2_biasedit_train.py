#!/usr/bin/env python3
"""Dataset-agnostic BIASEDIT training entrypoint."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


BASE_DIR = Path(__file__).resolve().parent
DATASETS = ("bbq", "crowspairs", "stereoset")


def _load_dataset_common(dataset: str) -> ModuleType:
    module_path = BASE_DIR / "paper_baselines_shared.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing shared baselines module: {module_path}")

    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    module_name = "paper_baselines_shared_adapter_biasedit_train"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_dataset_common(dataset)


def _apply_dataset_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "bbq": {
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "category": "Age",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models",
            "batch_size": 8,
            "max_length": 320,
        },
        "crowspairs": {
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models_crowspairs",
            "model_tag": "crowspairs_all",
            "batch_size": 8,
            "max_length": 256,
        },
        "stereoset": {
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/biasedit/models_stereoset",
            "model_tag": "stereoset_all",
            "batch_size": 8,
            "max_length": 256,
        },
    }

    for key, value in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.model_tag is None and args.dataset == "bbq":
        args.model_tag = args.category if args.category else "bbq_all"


def _build_strategy(common_mod: ModuleType, args: argparse.Namespace):
    kwargs = {
        "name": "biasedit",
        "pair_pref_weight": args.pair_pref_weight,
        "gap_mse_weight": args.gap_mse_weight,
        "margin": args.margin,
        "cda_weight": 0.0,
    }
    if args.dataset == "bbq":
        kwargs["chosen_lm_weight"] = args.lambda_r
    else:
        kwargs["anti_lm_weight"] = args.lambda_r
    return common_mod.TrainStrategy(**kwargs)


def _train_bbq(common_mod: ModuleType, args: argparse.Namespace) -> str:
    df = common_mod.load_bbq_df(
        args.bbq_dir,
        args.meta_file,
        category=args.category,
        limit_per_category=args.limit_per_category,
    )
    pair_df = common_mod.build_preference_pairs(df)

    if args.train_limit:
        pair_df = pair_df.iloc[: args.train_limit].copy()

    strategy = _build_strategy(common_mod, args)
    return common_mod.train_lora_pairwise(
        pair_df,
        strategy,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


def _train_crowspairs(common_mod: ModuleType, args: argparse.Namespace) -> str:
    df = common_mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = common_mod.make_pair_records(df)

    if args.train_limit:
        pair_df = pair_df.iloc[: args.train_limit].copy()

    strategy = _build_strategy(common_mod, args)
    return common_mod.train_lora_pairwise(
        pair_df,
        strategy,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


def _train_stereoset(common_mod: ModuleType, args: argparse.Namespace) -> str:
    examples = common_mod.load_examples(
        args.data_path,
        split=args.split,
        bias_type=args.bias_type,
        limit=args.limit,
    )

    if args.train_limit:
        examples = examples[: args.train_limit]

    strategy = _build_strategy(common_mod, args)
    return common_mod.train_lora_pairwise(
        examples,
        strategy,
        output_dir=args.output_dir,
        model_tag=args.model_tag,
        hf_token=args.hf_token,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BIASEDIT training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))

    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_r", type=float, default=0.0)
    parser.add_argument("--pair_pref_weight", type=float, default=0.0)
    parser.add_argument("--gap_mse_weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_layers", type=str, default=None)

    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))

    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_dataset_defaults(args)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_size is not None:
        print("[warn] --test_size is ignored in the shared BIASEDIT trainer.")
    if args.target_layers is not None:
        print("[warn] --target_layers is ignored in the shared BIASEDIT trainer.")
    if args.model != "llama_8b":
        print("[warn] shared trainer currently uses llama_8b configs in dataset common modules.")

    common_mod = _load_dataset_common(args.dataset)

    if args.dataset == "bbq":
        model_dir = _train_bbq(common_mod, args)
    elif args.dataset == "crowspairs":
        model_dir = _train_crowspairs(common_mod, args)
    else:
        model_dir = _train_stereoset(common_mod, args)

    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
