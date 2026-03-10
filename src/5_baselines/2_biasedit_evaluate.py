#!/usr/bin/env python3
"""
BiasEdit evaluation entrypoint.

Faithfully runs the official BiasEdit codebase (https://github.com/zjunlp/BiasEdit)
by invoking their main.py with eval_only=True via subprocess.

The official code reports:
  - SS Score  (stereotype score): fraction of edits where P(anti) > P(stereo); closer to 50 is unbiased
  - LMS       (language model score): perplexity-based language quality metric; higher is better
  - ICAT      (ideal CAT score): computed as LMS * min(SS, 100-SS) / 50; higher is better

These are computed by the official editor.valid() loop and logged to stdout.
This script captures and parses that output, then saves it to our CSV format.

Datasets supported: stereoset, crowspairs
(BBQ evaluation is not part of the original BiasEdit paper.)

Usage:
    python 2_biasedit_evaluate.py --dataset stereoset
    python 2_biasedit_evaluate.py --dataset stereoset --valid_path dataset/stereoset/gender_test.json
    python 2_biasedit_evaluate.py --dataset crowspairs
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent / "biasedit_repo"

# Our results output directory
RESULTS_DIR = "/scratch/craj/diy/results/3_baselines/biasedit"

# Default eval data paths (official repo test splits)
DEFAULT_VALID_PATHS = {
    "stereoset": {
        # Official: evaluate on gender/race/religion test splits
        "gender":   "dataset/stereoset/gender_test.json",
        "race":     "dataset/stereoset/race_test.json",
        "religion": "dataset/stereoset/religion_test.json",
    },
    "crowspairs": {
        "gender":   "dataset/crows/gender.csv",
        "race":     "dataset/crows/race.csv",
        "religion": "dataset/crows/religion.csv",
    },
}

DEFAULT_MODEL_CONFIG = "llama3_last123"


def build_eval_command(args: argparse.Namespace, valid_path: str) -> list[str]:
    """Build the eval subprocess command — mirrors official llama3.sh eval block."""
    cmd = [
        sys.executable, "main.py",
        f"data={args.dataset}",
        f"model={args.model_config}",
        "editor=malmen",
        f"data.n_edits={args.n_edits}",
        f"data.batch_size={args.batch_size}",
        f"data.valid_path={valid_path}",
        f"model_device={args.model_device}",
        f"editor_device={args.editor_device}",
        "editor.load_checkpoint=True",
        "eval_only=True",
        "use_wandb=False",
    ]

    if args.model_name_or_path:
        cmd.append(f"model.name_or_path={args.model_name_or_path}")

    return cmd


def parse_official_output(stdout: str) -> dict:
    """
    Parse the SS score, LMS, and delta_LMS from official editor.valid() log output.

    Official format (from base.py LOG.info calls):
        Overall results:
         Test -------- pre_ss: 0.45, edit_ss: 0.51, pre_lms: 0.82, edit_lms: 0.85, delta_lms: 0.03
    """
    metrics = {}

    # Match the final "Overall results" line
    overall_pattern = re.compile(
        r"Test\s*-+\s*"
        r"pre_ss:\s*([\d.]+),\s*"
        r"edit_ss:\s*([\d.]+)"
        r"(?:,\s*pre_lms:\s*([\d.]+))?"
        r"(?:,\s*edit_lms:\s*([\d.]+))?"
        r"(?:,\s*delta_lms:\s*([-\d.]+))?",
        re.IGNORECASE,
    )

    # Find the last match (Overall results block)
    matches = list(overall_pattern.finditer(stdout))
    if matches:
        m = matches[-1]
        metrics["pre_ss_score"]  = float(m.group(1))
        metrics["edit_ss_score"] = float(m.group(2))
        metrics["pre_lms"]       = float(m.group(3)) if m.group(3) else float("nan")
        metrics["edit_lms"]      = float(m.group(4)) if m.group(4) else float("nan")
        metrics["delta_lms"]     = float(m.group(5)) if m.group(5) else float("nan")

        # ICAT = LMS * min(SS, 100-SS) / 50  (standard StereoSet formula)
        # SS score from official code is a fraction [0,1]; convert to percent for ICAT
        ss_pct = metrics["edit_ss_score"] * 100
        lms    = metrics["edit_lms"] * 100
        metrics["icat_score"] = lms * min(ss_pct, 100 - ss_pct) / 50.0

    return metrics


def run_eval_split(args: argparse.Namespace, split_name: str, valid_path: str) -> dict:
    """Run official main.py for one eval split, capture output, parse metrics."""
    cmd = build_eval_command(args, valid_path)
    print(f"\nEvaluating split '{split_name}': {valid_path}")
    print(f"  cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    repo_str = str(REPO_DIR)
    env["PYTHONPATH"] = repo_str + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        capture_output=True,
        text=True,
    )

    # Always print output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"WARNING: eval for split '{split_name}' exited with code {result.returncode}", file=sys.stderr)

    combined = result.stdout + "\n" + result.stderr
    metrics = parse_official_output(combined)
    metrics["split"] = split_name
    metrics["valid_path"] = valid_path
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="BiasEdit evaluation — runs official main.py eval_only=True via subprocess"
    )
    parser.add_argument(
        "--dataset", required=True, choices=["stereoset", "crowspairs"],
        help="Evaluation dataset"
    )
    parser.add_argument(
        "--model_config", default=DEFAULT_MODEL_CONFIG,
        choices=["llama3_last1", "llama3_last12", "llama3_last123"],
    )
    parser.add_argument(
        "--model_name_or_path", default=None,
        help="Override model name_or_path"
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Which test splits to evaluate (default: all). "
             "For stereoset: gender, race, religion. For crowspairs: gender, race, religion."
    )
    parser.add_argument(
        "--valid_path", default=None,
        help="Override valid_path for a single evaluation (ignores --splits)"
    )
    parser.add_argument(
        "--n_edits", type=int, default=64,
        help="Must match the value used during training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
    )
    parser.add_argument(
        "--model_device", default="cuda:0",
    )
    parser.add_argument(
        "--editor_device", default="cuda:0",
    )
    parser.add_argument(
        "--output_file", default=None,
        help="Path to save CSV results (default: auto-generated in results_dir)"
    )
    parser.add_argument(
        "--results_dir", default=RESULTS_DIR,
    )
    args = parser.parse_args()

    if not REPO_DIR.exists():
        print(
            f"ERROR: BiasEdit repo not found at {REPO_DIR}\n"
            "Run training first (which clones the repo) or clone manually:\n"
            f"  git clone https://github.com/zjunlp/BiasEdit {REPO_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.results_dir, exist_ok=True)

    # Determine splits to evaluate
    if args.valid_path:
        # Single explicit path
        split_name = Path(args.valid_path).stem
        eval_splits = {split_name: args.valid_path}
    else:
        available = DEFAULT_VALID_PATHS[args.dataset]
        if args.splits:
            eval_splits = {s: available[s] for s in args.splits if s in available}
            missing = set(args.splits) - set(available)
            if missing:
                print(f"WARNING: unknown splits {missing}; available: {list(available)}", file=sys.stderr)
        else:
            eval_splits = available  # all splits

    if not eval_splits:
        print("ERROR: no valid evaluation splits found", file=sys.stderr)
        sys.exit(1)

    # Run evaluation for each split
    rows = []
    for split_name, valid_path in eval_splits.items():
        metrics = run_eval_split(args, split_name, valid_path)
        metrics["dataset"] = args.dataset
        metrics["model_config"] = args.model_config
        metrics["model"] = f"biasedit_{args.model_config}"
        rows.append(metrics)

    results_df = pd.DataFrame(rows)

    # Reorder columns for readability
    col_order = [
        "model", "dataset", "model_config", "split",
        "pre_ss_score", "edit_ss_score",
        "pre_lms", "edit_lms", "delta_lms",
        "icat_score",
        "valid_path",
    ]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    # Save
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
            args.results_dir,
            f"biasedit_eval_{args.dataset}_{args.model_config}.csv",
        )

    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
