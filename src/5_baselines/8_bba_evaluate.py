#!/usr/bin/env python3
"""
BBA: Bi-directional Bias Attribution - Inference-Time Neuron Intervention
Paper: "Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts"
arXiv:2602.04398
GitHub: https://github.com/XMUDeepLIT/Bi-directional-Bias-Attribution

This script applies the inference-time neuron intervention from BBA (Section 3.4):
- Loads the neuron attribution config produced by 8_bba_train.py
- Registers a forward hook on model.model.norm (final norm before lm_head)
- At inference, sets selected neuron activations to constant C (h_j := C)
- NO fine-tuning, NO LoRA adapters - pure inference-time intervention

The intervention: h̃_j = { C  if j in top-N neurons; h_j  otherwise }
where the projection layer (lm_head) receives the modified h̃ instead of h.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


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
        module_name="paper_baselines_shared_adapter_bba_eval",
        module_path=BASE_DIR / "paper_baselines_shared.py",
        add_to_syspath=BASE_DIR,
    )
    return shared_mod.get_dataset_common(dataset)


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


class BBANeuronHook:
    """
    Inference-time neuron intervention (BBA Section 3.4).

    Registers a forward hook on model.model.norm that sets the selected
    neuron activations to constant C before the lm_head linear projection.

    Intervention: h̃_j = C for all j in neuron_ids, all positions, all batches.
    """

    def __init__(self, model, neuron_ids: List[int], constant_c: float = 0.0):
        self.model = model
        self.neuron_ids = neuron_ids
        self.constant_c = constant_c
        self.hook_handle = None

    def _hook_fn(self, module, inp, output):
        # output shape: [batch_size, seq_len, hidden_dim]
        output = output.clone()
        output[:, :, self.neuron_ids] = self.constant_c
        return output

    def register(self):
        """Hook into the final norm layer (model.model.norm) output."""
        self.hook_handle = self.model.model.norm.register_forward_hook(self._hook_fn)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def load_bba_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def _evaluate_bbq(args: argparse.Namespace, mod: ModuleType) -> None:
    df = mod.load_bbq_df(
        args.bbq_dir,
        args.meta_file,
        category=args.category,
        limit_per_category=args.limit_per_category,
    )

    # Load base model (no LoRA adapter - BBA is an inference-time method)
    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    # Apply BBA neuron intervention hook
    bba_cfg = load_bba_config(args.config_path)
    neuron_ids = bba_cfg["neuron_ids"]
    constant_c = bba_cfg.get("constant_c", args.constant_c)
    hook = BBANeuronHook(model, neuron_ids, constant_c)
    hook.register()
    print(f"BBA: Intervening on {len(neuron_ids)} neurons (C={constant_c})")

    model_name = f"{args.model}_bba_{args.model_tag}"
    try:
        metrics_df, preds_df = mod.evaluate_bbq_df(
            df, model, tokenizer,
            batch_size=args.batch_size,
            model_name=model_name,
        )
        mod.save_eval_outputs(metrics_df, preds_df, args.output_file, args.preds_output_file)
    finally:
        hook.remove()

    print(f"Saved results to {args.output_file}")
    print(f"Saved predictions to {args.preds_output_file}")


def _evaluate_crowspairs(args: argparse.Namespace, mod: ModuleType) -> None:
    df = mod.load_crowspairs_df(args.data_path, bias_type=args.bias_type, limit=args.limit)
    pair_df = mod.make_pair_records(df)

    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    bba_cfg = load_bba_config(args.config_path)
    neuron_ids = bba_cfg["neuron_ids"]
    constant_c = bba_cfg.get("constant_c", args.constant_c)
    hook = BBANeuronHook(model, neuron_ids, constant_c)
    hook.register()
    print(f"BBA: Intervening on {len(neuron_ids)} neurons (C={constant_c})")

    model_name = f"{args.model}_bba_{args.model_tag}"
    try:
        scored = mod.evaluate_pair_records(pair_df, model, tokenizer, batch_size=args.batch_size)
        overall, per_bias = mod.compute_metrics_from_scored(scored, model_name=model_name)
        mod.save_eval_outputs(
            scored, overall, per_bias,
            args.pairs_output_file, args.output_file, args.per_bias_output_file,
        )
    finally:
        hook.remove()

    print(f"Saved pair-level scores to {args.pairs_output_file}")
    print(f"Saved overall metrics to {args.output_file}")
    print(f"Saved per-bias metrics to {args.per_bias_output_file}")


def _evaluate_stereoset(args: argparse.Namespace, mod: ModuleType) -> None:
    examples = mod.load_examples(args.data_path, split=args.split, bias_type=args.bias_type, limit=args.limit)

    model, tokenizer = mod.load_model_and_tokenizer(
        hf_token=args.hf_token,
        adapter_path=None,
        base_model=mod.BASE_MODEL,
    )

    bba_cfg = load_bba_config(args.config_path)
    neuron_ids = bba_cfg["neuron_ids"]
    constant_c = bba_cfg.get("constant_c", args.constant_c)
    hook = BBANeuronHook(model, neuron_ids, constant_c)
    hook.register()
    print(f"BBA: Intervening on {len(neuron_ids)} neurons (C={constant_c})")

    model_name = f"{args.model}_bba_{args.model_tag}"
    try:
        records_df, results, preds_json = mod.evaluate_examples(
            examples, model, tokenizer, batch_size=args.batch_size
        )
        mod.save_eval_outputs(
            records_df, results, preds_json,
            model_name=model_name,
            results_csv=args.results_csv,
            results_json=args.results_json,
            predictions_file=args.predictions_file,
            scores_output_file=args.scores_output_file,
        )
    finally:
        hook.remove()

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
            "config_path": "/scratch/craj/diy/outputs/3_baselines/bba/models_bbq/bba_config_bbq_all.json",
            "output_file": "/scratch/craj/diy/results/3_baselines/bba/bbq_eval_llama_8b_bba.csv",
            "preds_output_file": "/scratch/craj/diy/outputs/3_baselines/bba/bbq_preds_llama_8b_bba.csv",
        },
        "crowspairs": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
            "batch_size": 4,
            "model_tag": "crowspairs_all",
            "config_path": "/scratch/craj/diy/outputs/3_baselines/bba/models_crowspairs/bba_config_crowspairs_all.json",
            "output_file": "/scratch/craj/diy/results/3_baselines/bba/crowspairs_metrics_overall_llama_8b_bba.csv",
            "per_bias_output_file": "/scratch/craj/diy/results/3_baselines/bba/crowspairs_metrics_by_bias_llama_8b_bba.csv",
            "pairs_output_file": "/scratch/craj/diy/outputs/3_baselines/bba/crowspairs_scored_llama_8b_bba.csv",
        },
        "stereoset": {
            "model": "llama_8b",
            "data_path": "/scratch/craj/diy/data/stereoset/dev.json",
            "split": "all",
            "batch_size": 4,
            "model_tag": "stereoset_all",
            "config_path": "/scratch/craj/diy/outputs/3_baselines/bba/models_stereoset/bba_config_stereoset_all.json",
            "results_json": "/scratch/craj/diy/results/3_baselines/bba/stereoset_eval_llama_8b_bba.json",
            "results_csv": "/scratch/craj/diy/results/3_baselines/bba/stereoset_eval_llama_8b_bba.csv",
            "predictions_file": "/scratch/craj/diy/outputs/3_baselines/bba/stereoset_predictions_llama_8b_bba.json",
            "scores_output_file": "/scratch/craj/diy/outputs/3_baselines/bba/stereoset_sentence_scores_llama_8b_bba.csv",
        },
    }
    for key, val in defaults[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBA: Inference-Time Neuron Intervention (no LoRA, no fine-tuning)"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["bbq", "crowspairs", "stereoset"])
    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to bba_config_*.json from 8_bba_train.py")
    parser.add_argument("--constant_c", type=float, default=0.0,
                        help="Override constant C (if not in config)")
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
            f"BBA config not found: {args.config_path}\n"
            "Run 8_bba_train.py first to generate the neuron attribution config."
        )

    mod = _load_dataset_common(args.dataset)

    print("=" * 60)
    print(f"BBA: Inference-Time Neuron Intervention ({args.dataset})")
    print(f"Config: {args.config_path}")
    print("=" * 60)

    if args.dataset == "bbq":
        _evaluate_bbq(args, mod)
    elif args.dataset == "crowspairs":
        _evaluate_crowspairs(args, mod)
    else:
        _evaluate_stereoset(args, mod)


if __name__ == "__main__":
    main()
