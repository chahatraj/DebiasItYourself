#!/usr/bin/env python3
"""Dataset-agnostic Debias-LLMs-inspired LoRA training entrypoint."""

import argparse
import importlib.util
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
        module_name="paper_baselines_shared_adapter",
        module_path=BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    return shared_mod.get_dataset_common(dataset)


def _build_strategy(dataset: str, mod: ModuleType):
    kwargs = dict(
        name="debias_llms",
        pair_pref_weight=1.0,
        gap_mse_weight=0.0,
        margin=0.0,
        cda_weight=0.0,
    )
    if dataset == "bbq":
        kwargs["chosen_lm_weight"] = 1.0
    else:
        kwargs["anti_lm_weight"] = 1.0
    return mod.TrainStrategy(**kwargs)


def _prepare_training_data(args: argparse.Namespace, mod: ModuleType):
    if args.dataset == "bbq":
        df = mod.load_bbq_df(
            args.bbq_dir,
            args.meta_file,
            category=args.category,
            limit_per_category=args.limit_per_category,
        )
        return mod.build_preference_pairs(df)
    if args.dataset == "crowspairs":
        df = mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
        return mod.make_pair_records(df)
    return mod.load_examples(args.data_path, split=args.split, bias_type=args.bias_type, limit=args.limit)


def _apply_defaults(args: argparse.Namespace) -> None:
    defaults: Dict[str, Dict[str, object]] = {
        "bbq": {
            "bbq_dir": "/scratch/craj/diy/data/BBQ/data",
            "meta_file": "/scratch/craj/diy/data/BBQ/analysis_scripts/additional_metadata.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_llms/models_bbq",
            "model_tag": "bbq_all",
            "epochs": 3,
            "batch_size": 4,
            "lr": 5e-5,
            "max_length": 320,
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "crowspairs": {
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_llms/models_crowspairs",
            "model_tag": "crowspairs_all",
            "epochs": 3,
            "batch_size": 8,
            "lr": 5e-5,
            "max_length": 256,
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "stereoset": {
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_llms/models_stereoset",
            "model_tag": "stereoset_all",
            "epochs": 3,
            "batch_size": 4,
            "lr": 5e-5,
            "max_length": 256,
            "lora_r": 8,
            "lora_alpha": 16,
        },
    }

    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debias-LLMs-inspired training across BBQ/CrowS-Pairs/StereoSet")
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["all", "intrasentence", "intersentence"])
    parser.add_argument("--bias_type", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--bbq_dir", type=str, default=None)
    parser.add_argument("--meta_file", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_defaults(args)
    mod = _load_dataset_common(args.dataset)

    train_data = _prepare_training_data(args, mod)
    strategy = _build_strategy(args.dataset, mod)

    model_dir = mod.train_lora_pairwise(
        train_data,
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
    print(f"Model saved to {model_dir}")


if __name__ == "__main__":
    main()
