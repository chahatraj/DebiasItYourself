#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

ROOT = Path("/scratch/craj/diy")
EXP_DIR = ROOT / "src" / "3_experiments"

VALID_BBQ_SOURCE_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Race_x_gender.jsonl",
    "Race_x_SES.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]

STRATEGY_ALIASES = {
    "sr": "stereotype_replacement",
    "stereotype_replacement": "stereotype_replacement",
    "stereotype-replacement": "stereotype_replacement",
    "stereotype replacement": "stereotype_replacement",
    "ci": "counter_imaging",
    "counter_imaging": "counter_imaging",
    "counter-imaging": "counter_imaging",
    "counter imaging": "counter_imaging",
    "ind": "individuating",
    "individuating": "individuating",
    "pt": "perspective_taking",
    "perspective_taking": "perspective_taking",
    "perspective-taking": "perspective_taking",
    "perspective taking": "perspective_taking",
    "pc": "positive_contact",
    "positive_contact": "positive_contact",
    "positive-contact": "positive_contact",
    "positive contact": "positive_contact",
    "all": "all_strategies",
    "all_strategies": "all_strategies",
    "all-strategies": "all_strategies",
    "all strategies": "all_strategies",
    "allstrategy": "all_strategies",
    "allstrategies": "all_strategies",
}

STRATEGY_ORDER = [
    "stereotype_replacement",
    "counter_imaging",
    "individuating",
    "perspective_taking",
    "positive_contact",
]


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_exact_strategy_prompts() -> Dict[str, str]:
    """Parse DEBIASING_PROMPTS directly from 3_finetune_llama.py without importing heavy deps."""
    src = (EXP_DIR / "3_finetune_llama.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEBIASING_PROMPTS":
                    raw = ast.literal_eval(node.value)
                    out: Dict[str, str] = {}
                    for strategy, shot_map in raw.items():
                        if isinstance(shot_map, dict) and "zero" in shot_map:
                            out[str(strategy)] = str(shot_map["zero"])  # exact text from file
                    return out
    raise ValueError("Could not locate DEBIASING_PROMPTS in 3_finetune_llama.py")


_EXACT_PROMPTS = _load_exact_strategy_prompts()


def _normalize_strategy(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return STRATEGY_ALIASES.get(key)


def _infer_strategy_from_identifiers(*identifiers: Optional[str]) -> Optional[str]:
    hay = " ".join(str(x).lower() for x in identifiers if x)
    if not hay:
        return None
    # Prefer explicit strategy mentions; then shorthand tags used in model names.
    probes = [
        ("all_strategies", ["allstrateg", "all_strateg", "all-strateg"]),
        ("stereotype_replacement", ["stereotype_replacement", "stereotype-replacement", "_sr_", "sr_"]),
        ("counter_imaging", ["counter_imaging", "counter-imaging", "_ci_", "ci_"]),
        ("individuating", ["individuating", "_ind_", "ind_"]),
        ("perspective_taking", ["perspective_taking", "perspective-taking", "_pt_", "pt_"]),
        ("positive_contact", ["positive_contact", "positive-contact", "_pc_", "pc_"]),
    ]
    for strategy, pats in probes:
        for pat in pats:
            if pat in hay:
                return strategy
    return None


def resolve_exact_instruction(
    mode: str,
    strategy: Optional[str] = None,
    model_tag: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "off":
        return None, None
    if normalized_mode != "strategy":
        raise ValueError(f"Unsupported inference mode: {mode}")

    strategy_key = _normalize_strategy(strategy)
    if strategy and strategy_key is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy_key is None:
        strategy_key = _infer_strategy_from_identifiers(model_tag, model_path)

    if strategy_key is None:
        raise ValueError(
            "Could not infer strategy from identifiers. Pass --inference_strategy explicitly."
        )

    if strategy_key == "all_strategies":
        # Keep each instruction text exactly as defined in 3_finetune_llama.py.
        instruction = "\n\n".join(_EXACT_PROMPTS[k] for k in STRATEGY_ORDER)
        return strategy_key, instruction

    if strategy_key not in _EXACT_PROMPTS:
        raise ValueError(f"No exact prompt found for strategy {strategy_key}")
    return strategy_key, _EXACT_PROMPTS[strategy_key]


def run_eval_shared(eval_args: Sequence[str], use_exact_prompt_patch: bool = True) -> None:
    module = _load_module("eval_shared_runtime", EXP_DIR / "7_eval_shared.py")
    if use_exact_prompt_patch:
        module.resolve_inference_instruction = resolve_exact_instruction

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(EXP_DIR / "7_eval_shared.py")] + list(eval_args)
        module.main()
    finally:
        sys.argv = old_argv


def run_bbq_infer(eval_args: Sequence[str], use_exact_prompt_patch: bool = True) -> None:
    module = _load_module("bbq_infer_runtime", EXP_DIR / "13_bbq_inference_instruction.py")
    if use_exact_prompt_patch:
        module.resolve_instruction = resolve_exact_instruction

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(EXP_DIR / "13_bbq_inference_instruction.py")] + list(eval_args)
        module.main()
    finally:
        sys.argv = old_argv


def method_entrypoint(
    *,
    method_name: str,
    default_model_path: Optional[str],
    default_inference_mode: str,
    default_inference_strategy: Optional[str],
) -> None:
    parser = argparse.ArgumentParser(description=f"Run {method_name} on shared datasets.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "bbq",
            "bbq_eval",
            "crowspairs",
            "stereoset",
            "bold",
            "honest",
            "winobias",
            "winogender",
            "unqover",
            "bias_in_bios",
        ],
    )
    parser.add_argument("--model", default="llama_8b")
    parser.add_argument("--model_path", default=default_model_path)
    parser.add_argument("--model_tag", default=_slug(method_name))

    parser.add_argument("--output_root", default="/scratch/craj/diy/outputs/new_outputs")
    parser.add_argument("--results_root", default="/scratch/craj/diy/results/new_results")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--results_dir", default=None)

    parser.add_argument("--source_file", choices=VALID_BBQ_SOURCE_FILES, default=None)
    parser.add_argument("--unqover_dim", choices=["gender", "race", "religion", "nationality"], default="gender")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--inference_instruction_mode", choices=["off", "strategy"], default=default_inference_mode)
    parser.add_argument("--inference_strategy", default=default_inference_strategy)

    args = parser.parse_args()

    method_slug = _slug(method_name)
    group = "bbq" if args.dataset in ("bbq", "bbq_eval") else "evalshared"
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / method_slug / group
    results_dir = Path(args.results_dir) if args.results_dir else Path(args.results_root) / method_slug / group
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "bbq":
        if not args.source_file:
            raise ValueError("--source_file is required when --dataset bbq")
        bbq_args = [
            "--model", args.model,
            "--source_file", args.source_file,
            "--output_dir", str(output_dir),
            "--model_tag", args.model_tag,
            "--max_length", str(args.max_length),
            "--seed", str(args.seed),
            "--inference_instruction_mode", args.inference_instruction_mode,
        ]
        if args.model_path:
            bbq_args += ["--model_path", args.model_path]
        if args.inference_strategy:
            bbq_args += ["--inference_strategy", args.inference_strategy]
        if args.limit is not None:
            bbq_args += ["--limit", str(args.limit)]
        run_bbq_infer(bbq_args, use_exact_prompt_patch=True)
        return

    if args.dataset == "bbq_eval":
        output_file = results_dir / f"bbq_metrics_{_slug(args.model_tag)}.csv"
        eval_args = [
            "--dataset", "bbq",
            "--model_dir", str(output_dir),
            "--output_file", str(output_file),
            "--model_name", args.model_tag,
        ]
        run_eval_shared(eval_args, use_exact_prompt_patch=False)
        return

    eval_args = [
        "--dataset", args.dataset,
        "--model", args.model,
        "--output_dir", str(output_dir),
        "--results_dir", str(results_dir),
        "--model_tag", args.model_tag,
        "--seed", str(args.seed),
        "--max_length", str(args.max_length),
        "--inference_instruction_mode", args.inference_instruction_mode,
    ]

    if args.model_path:
        eval_args += ["--model_path", args.model_path]
    if args.inference_strategy:
        eval_args += ["--inference_strategy", args.inference_strategy]
    if args.batch_size is not None:
        eval_args += ["--batch_size", str(args.batch_size)]
    if args.max_samples is not None:
        eval_args += ["--max_samples", str(args.max_samples)]
    if args.limit is not None:
        eval_args += ["--limit", str(args.limit)]
    if args.dataset == "unqover":
        eval_args += ["--unqover_dim", args.unqover_dim]

    run_eval_shared(eval_args, use_exact_prompt_patch=True)
