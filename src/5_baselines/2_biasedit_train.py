#!/usr/bin/env python3
"""
BiasEdit training entrypoint.

Faithfully runs the official BiasEdit codebase (https://github.com/zjunlp/BiasEdit)
by invoking their main.py via subprocess from the cloned repo directory.

Official method: MALMEN hypernetwork-based model editing.
  - Trains a small MALMENNet that predicts parameter shifts (ΔW) for target MLP layers.
  - Debiasing loss: increase P(anti-stereotype) relative to P(stereotype).
  - Locality loss: KL divergence on unrelated sentences to preserve LM ability.
  - No LoRA — edits are direct temporary weight shifts applied to down_proj layers.

Datasets supported: stereoset, crowspairs
(BBQ is not used for training in the original paper.)

Usage:
    python 2_biasedit_train.py --dataset stereoset
    python 2_biasedit_train.py --dataset crowspairs
    python 2_biasedit_train.py --dataset stereoset --model_config llama3_last123 --n_edits 64
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent / "biasedit_repo"

# Paths to our local data
DATA_PATHS = {
    "stereoset": {
        "train": "/scratch/craj/diy/data/stereoset/dev.json",   # official repo uses dev.json for both train and test
        "valid": "/scratch/craj/diy/data/stereoset/dev.json",
    },
    "crowspairs": {
        # official repo splits by bias_type; we use the full file as train
        # The repo's Crows data loader reads sent_more/sent_less columns
        "train": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
        "valid": "/scratch/craj/diy/data/crows_pairs_anonymized.csv",
    },
}

# Checkpoint save directory (where main.py writes checkpoints/)
CHECKPOINT_BASE = "/scratch/craj/diy/outputs/3_baselines/biasedit"

# Default model config — last 3 layers of LLaMA-3-8B (matches official llama3.sh)
DEFAULT_MODEL_CONFIG = "llama3_last123"

# Maps model config name → HF model path (for main.py model.name_or_path override)
MODEL_PATHS = {
    "llama3_last1":   "meta-llama/Meta-Llama-3-8B",
    "llama3_last12":  "meta-llama/Meta-Llama-3-8B",
    "llama3_last123": "meta-llama/Meta-Llama-3-8B",
}

# Llama 3.1 Instruct cache dir
CACHE_DIR = "/scratch/craj/cache/model_cache/llama-3.1-8b-instruct"


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the subprocess command that mirrors the official llama3.sh script."""
    data_paths = DATA_PATHS[args.dataset]

    cmd = [
        sys.executable, "main.py",
        f"data={args.dataset}",
        f"model={args.model_config}",
        "editor=malmen",
        f"data.n_edits={args.n_edits}",
        f"data.batch_size={args.batch_size}",
        f"data.train_path={data_paths['train']}",
        f"data.valid_path={data_paths['valid']}",
        f"model_device={args.model_device}",
        f"editor_device={args.editor_device}",
        f"editor.n_epochs={args.n_epochs}",
        f"early_stop_patience={args.early_stop_patience}",
        "eval_only=False",
        "use_wandb=False",
    ]

    # Override model name_or_path if using instruct variant from cache
    if args.model_name_or_path:
        cmd.append(f"model.name_or_path={args.model_name_or_path}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="BiasEdit training — runs official main.py via subprocess"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["stereoset", "crowspairs"],
        help="Training dataset (BBQ not used for BiasEdit training)"
    )
    parser.add_argument(
        "--model_config", default=DEFAULT_MODEL_CONFIG,
        choices=["llama3_last1", "llama3_last12", "llama3_last123"],
        help="Which layer config to edit (default: llama3_last123, i.e. last 3 layers)"
    )
    parser.add_argument(
        "--model_name_or_path", default=None,
        help="Override model name_or_path (e.g. use instruct variant). "
             "Defaults to meta-llama/Meta-Llama-3-8B as in the paper."
    )
    parser.add_argument(
        "--n_edits", type=int, default=64,
        help="Number of edits per batch (n_edits in official script: 64)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Data batch_size within each edit batch (official: 16)"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100,
        help="Max training epochs (editor.n_epochs, official: 100)"
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=3,
        help="Early stopping patience (official llama3.sh: 3)"
    )
    parser.add_argument(
        "--model_device", default="cuda:0",
        help="Device for the LLM (e.g. cuda:0, cuda:1)"
    )
    parser.add_argument(
        "--editor_device", default="cuda:0",
        help="Device for the MALMEN editor network"
    )
    args = parser.parse_args()

    if not REPO_DIR.exists():
        print(
            f"ERROR: BiasEdit repo not found at {REPO_DIR}\n"
            "Clone it with:\n"
            f"  git clone https://github.com/zjunlp/BiasEdit {REPO_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Checkpoints are written to <cwd>/checkpoints/ by default in main.py
    # We run main.py from CHECKPOINT_BASE so checkpoints land there
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)

    cmd = build_command(args)

    print("Running BiasEdit training (official MALMEN method):")
    print(f"  repo: {REPO_DIR}")
    print(f"  cwd:  {CHECKPOINT_BASE}")
    print(f"  cmd:  {' '.join(cmd)}\n")

    env = os.environ.copy()
    # Ensure the repo's modules (nets, editor, data, util) are importable
    repo_str = str(REPO_DIR)
    env["PYTHONPATH"] = repo_str + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # Hydra writes outputs relative to cwd; run from CHECKPOINT_BASE
    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),   # main.py uses relative paths for config/ and cache/
        env=env,
    )

    if result.returncode != 0:
        print(f"\nBiasEdit training failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(
        f"\nTraining complete. Checkpoints saved under:\n"
        f"  {REPO_DIR}/checkpoints/\n"
        f"Pass --checkpoint_dir to 2_biasedit_evaluate.py to run evaluation."
    )


if __name__ == "__main__":
    main()
