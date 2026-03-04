#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


ROOT = Path("/scratch/craj/diy")
EXP_DIR = ROOT / "src/3_experiments"
SLURM_RUNNER = ROOT / "src/4_slurmjobs/run_job_array.slurm"
ARRAYS_ROOT = ROOT / "src/4_slurmjobs/arrays"

LOG_OUT_DIR = Path("/scratch/craj/logs/diy/out")
LOG_ERR_DIR = Path("/scratch/craj/logs/diy/err")
DEFAULT_VENV = "/home/craj/nanotron-env/bin/activate"


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit baseline jobs for additional bias benchmarks")
    parser.add_argument("--run_tag", type=str, default=f"baseline_additional_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--bold_max_samples", type=int, default=3000)
    parser.add_argument("--honest_max_samples", type=int, default=None)
    parser.add_argument("--winobias_max_samples", type=int, default=None)
    parser.add_argument("--winogender_max_samples", type=int, default=None)
    parser.add_argument("--unqover_max_samples", type=int, default=2500)
    parser.add_argument("--bios_max_samples", type=int, default=500)

    parser.add_argument("--partition", type=str, default="gpuq")
    parser.add_argument("--qos", type=str, default="gpu")
    parser.add_argument("--gres", type=str, default="gpu:A100.80gb:1")
    parser.add_argument("--mem", type=str, default="40G")
    parser.add_argument("--time", type=str, default="0-08:00:00")

    parser.add_argument("--venv_path", type=str, default=DEFAULT_VENV)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    run_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.run_tag)

    output_root = ROOT / "outputs/10_additional_benchmarks" / "baselines" / run_tag
    results_root = ROOT / "results/10_additional_benchmarks" / "baselines" / run_tag
    array_dir = ARRAYS_ROOT / run_tag

    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)
    LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ERR_DIR.mkdir(parents=True, exist_ok=True)

    model_arg = f"--model {args.model}"
    model_path_arg = f" --model_path {args.model_path}" if args.model_path else ""

    def opt_max(x: int | None) -> str:
        return "" if x is None else f" --max_samples {x}"

    commands: List[str] = []

    commands.append(
        "python 11_additional_benchmarks_eval.py "
        f"--benchmark bold {model_arg}{model_path_arg} "
        f"--output_dir {output_root} --results_dir {results_root} "
        f"--batch_size 8 --max_length 1024{opt_max(args.bold_max_samples)}"
    )

    commands.append(
        "python 11_additional_benchmarks_eval.py "
        f"--benchmark honest {model_arg}{model_path_arg} "
        f"--honest_config en_binary "
        f"--output_dir {output_root} --results_dir {results_root} "
        f"--batch_size 8 --max_length 512{opt_max(args.honest_max_samples)}"
    )

    commands.append(
        "python 11_additional_benchmarks_eval.py "
        f"--benchmark winobias {model_arg}{model_path_arg} "
        f"--output_dir {output_root} --results_dir {results_root} "
        f"--batch_size 8 --max_length 512{opt_max(args.winobias_max_samples)}"
    )

    commands.append(
        "python 11_additional_benchmarks_eval.py "
        f"--benchmark winogender {model_arg}{model_path_arg} "
        f"--output_dir {output_root} --results_dir {results_root} "
        f"--batch_size 8 --max_length 512 --option_batch_size 2{opt_max(args.winogender_max_samples)}"
    )

    for dim in ["gender", "race", "religion", "nationality"]:
        commands.append(
            "python 11_additional_benchmarks_eval.py "
            f"--benchmark unqover --unqover_dim {dim} {model_arg}{model_path_arg} "
            f"--output_dir {output_root} --results_dir {results_root} "
            f"--batch_size 8 --max_length 512 --option_batch_size 2{opt_max(args.unqover_max_samples)}"
        )

    commands.append(
        "python 11_additional_benchmarks_eval.py "
        f"--benchmark bias_in_bios --bios_split test {model_arg}{model_path_arg} "
        f"--output_dir {output_root} --results_dir {results_root} "
        f"--batch_size 4 --max_length 768 --option_batch_size 8{opt_max(args.bios_max_samples)}"
    )

    array_file = array_dir / "baseline_additional_array.txt"
    write_lines(array_file, commands)

    meta = {
        "run_tag": run_tag,
        "model": args.model,
        "model_path": args.model_path,
        "output_root": str(output_root),
        "results_root": str(results_root),
        "array_file": str(array_file),
        "num_jobs": len(commands),
        "partition": args.partition,
        "qos": args.qos,
        "gres": args.gres,
        "mem": args.mem,
        "time": args.time,
    }

    job_id = ""
    if args.submit:
        array_spec = f"1-{len(commands)}"
        out_file = LOG_OUT_DIR / f"baseline_additional.{run_tag}.%A_%a.out.txt"
        err_file = LOG_ERR_DIR / f"baseline_additional.{run_tag}.%A_%a.err.txt"

        cmd = [
            "sbatch",
            "--parsable",
            "--job-name",
            "baseline_additional",
            "--partition",
            args.partition,
            "--qos",
            args.qos,
            "--gres",
            args.gres,
            "--mem",
            args.mem,
            "--time",
            args.time,
            "--array",
            array_spec,
            "--output",
            str(out_file),
            "--error",
            str(err_file),
            "--export",
            f"ALL,ARRAY_FILE={array_file},WORKDIR={EXP_DIR},VENV_PATH={args.venv_path}",
            str(SLURM_RUNNER),
        ]

        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = res.stdout.strip()
        meta["job_id"] = job_id

        submitted_file = array_dir / "submitted_jobs.txt"
        with submitted_file.open("w", encoding="utf-8") as f:
            f.write(f"run_tag: {run_tag}\n")
            f.write(f"job_id: {job_id}\n")
            f.write(f"array_file: {array_file}\n")
        meta["submitted_file"] = str(submitted_file)

    meta_file = array_dir / "meta.json"
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print("Monitor: squeue -u craj -o '%.18i %.15P %.30j %.10T %.12M %.6D %R'")


if __name__ == "__main__":
    main()
