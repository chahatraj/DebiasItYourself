#!/usr/bin/env python3
"""
debias_NLG: Parameter-Efficient Multi-Objective Debiasing via LoRA
Paper: "A Parameter-Efficient Multi-Objective Approach to Mitigate Stereotypical Bias in Language Models"
GeBNLP @ ACL 2024 (https://aclanthology.org/2024.gebnlp-1.1)
GitHub: https://github.com/Ewanwong/debias_NLG

Official method: prefix-tuning on GPT-2 with 4 probability alignment losses on
a CDA-augmented News-Commentary V15 corpus (gender bias only):
  L = α₁·L_LM + α₂·L_neu + α₃·L_eq_tok + α₄·L_eq_seq
  where:
    L_LM     = NLL language modeling loss on CDA sentence pairs
    L_neu    = JSD between next-token distributions conditioned on CDA pairs
               (neutralization: neutral attribute words equally likely across genders)
    L_eq_tok = KLD(q || p_i) where q=uniform over gender targets, p_i=model distribution
               (token-level equalizing: model equally predicts male/female target words)
    L_eq_seq = KLD(q || p) sequence-level version of above
  Best hyperparameters (paper Section 5): α₁=1, α₂=50, α₃=200, α₄=250

Our adaptation uses LoRA instead of prefix-tuning (no prefix overhead at inference),
applies the same 4-objective probability alignment losses on CDA pairs from our
datasets, and extends beyond gender bias to cover all bias categories in BBQ/CrowS/StereoSet.
"""

import argparse
import importlib.util
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
    sys.modules[module_name] = mod
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
    # Official loss weights (paper Section 5, best combo on StereoSet validation):
    #   α₁=1 (L_LM), α₂=50 (L_neu/JSD), α₃=200 (L_eq_tok), α₄=250 (L_eq_seq)
    # CDA pairs are used for all loss components.
    return mod.TrainStrategy(
        name="debias_nlg_cda",
        chosen_lm_weight=1.0,
        anti_lm_weight=1.0,
        pair_pref_weight=1.0,
        cda_weight=0.2,
    )


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
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_nlg/models_bbq",
            "model_tag": "bbq_all",
            "epochs": 5,      # paper: 5 epochs
            "batch_size": 16, # paper: batch_size=16
            "lr": 5e-5,       # paper: lr=5e-5
            "max_length": 320,
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "crowspairs": {
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_nlg/models_crowspairs",
            "model_tag": "crowspairs_all",
            "epochs": 5,      # paper: 5 epochs
            "batch_size": 16, # paper: batch_size=16
            "lr": 5e-5,
            "max_length": 256,
            "lora_r": 8,
            "lora_alpha": 16,
        },
        "stereoset": {
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "output_dir": "/scratch/craj/diy/outputs/3_baselines/debias_nlg/models_stereoset",
            "model_tag": "stereoset_all",
            "epochs": 5,      # paper: 5 epochs
            "batch_size": 16, # paper: batch_size=16
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
    parser = argparse.ArgumentParser(description="debias_NLG-inspired training across BBQ/CrowS-Pairs/StereoSet")
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
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"))
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
