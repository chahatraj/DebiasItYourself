#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path("/scratch/craj/diy")
EXP_DIR = ROOT / "src" / "3_experiments"
RUNNER = ROOT / "src" / "4_slurmjobs" / "run_single_baseline_anynode.slurm"
VENV = "/home/craj/nanotron-env/bin/activate"

OUT_ROOT = ROOT / "outputs" / "10_additional_benchmarks" / "baseline_methods"
RES_ROOT = ROOT / "results" / "10_additional_benchmarks" / "baseline_methods"
TRACK_ROOT = ROOT / "tracking" / "baseline_methods"


def methods_to_adapters() -> Dict[str, str]:
    return {
        "biasedit": str(ROOT / "outputs/3_baselines/old/biasedit/models_stereoset/model_stereoset_all"),
        "bba": str(ROOT / "outputs/3_baselines/old/bba/models_stereoset/model_stereoset_all"),
        "cal": str(ROOT / "outputs/3_baselines/old/cal/models_stereoset/model_stereoset_all"),
        "dpo": str(ROOT / "outputs/3_baselines/dpo/models_stereoset/model_stereoset_all"),
        "lftf": str(ROOT / "outputs/3_baselines/lftf/models_stereoset/model_stereoset_all"),
        "peft": str(ROOT / "outputs/3_baselines/peft/models_stereoset/model_stereoset_all"),
        "debias_nlg": str(ROOT / "outputs/3_baselines/debias_nlg/models_stereoset/model_stereoset_all"),
        "debias_llms": str(ROOT / "outputs/3_baselines/debias_llms/models_stereoset/model_stereoset_all"),
        "mbias": str(ROOT / "outputs/3_baselines/mbias/models_stereoset/model_stereoset_all"),
    }


def build_commands(run_tag: str) -> List[Tuple[str, List[str]]]:
    cmds: List[Tuple[str, List[str]]] = []
    model = "llama_8b"

    for method, adapter in methods_to_adapters().items():
        tag = f"{method}_addl_{run_tag}"
        out_dir = OUT_ROOT / run_tag / method
        res_dir = RES_ROOT / run_tag / method
        common = [
            "python",
            "7_eval_shared.py",
            "--model",
            model,
            "--adapter_path",
            adapter,
            "--model_tag",
            tag,
            "--output_dir",
            str(out_dir),
            "--results_dir",
            str(res_dir),
        ]

        # Remaining datasets (beyond BBQ/CrowS/StereoSet).
        cmds.append((f"{method}_bold", common + ["--dataset", "bold", "--batch_size", "8", "--max_length", "1024", "--max_samples", "3000"]))
        cmds.append((f"{method}_honest", common + ["--dataset", "honest", "--honest_config", "en_binary", "--batch_size", "8", "--max_length", "512"]))
        cmds.append((f"{method}_winobias", common + ["--dataset", "winobias", "--batch_size", "8", "--max_length", "512"]))
        cmds.append((f"{method}_winogender", common + ["--dataset", "winogender", "--batch_size", "8", "--max_length", "512", "--option_batch_size", "2"]))
        cmds.append((f"{method}_bias_in_bios", common + ["--dataset", "bias_in_bios", "--bios_split", "test", "--batch_size", "4", "--max_length", "768", "--option_batch_size", "8", "--max_samples", "500"]))
        for dim in ("gender", "race", "religion", "nationality"):
            cmds.append(
                (
                    f"{method}_unqover_{dim}",
                    common + ["--dataset", "unqover", "--unqover_dim", dim, "--batch_size", "8", "--max_length", "512", "--option_batch_size", "2", "--max_samples", "2500"],
                )
            )

    return cmds


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit additional-dataset eval jobs for baseline adapters.")
    parser.add_argument("--run_tag", type=str, default=f"baseline_remaining_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--partition", type=str, default="contrib-gpuq")
    parser.add_argument("--qos", type=str, default="cs_dept")
    parser.add_argument("--gres", type=str, default="gpu:A100.80gb:1")
    parser.add_argument("--cpus_per_task", type=int, default=10)
    parser.add_argument("--mem", type=str, default="70G")
    parser.add_argument("--time", type=str, default="0-10:00:00")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    run_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.run_tag)
    track_dir = TRACK_ROOT / run_tag
    track_dir.mkdir(parents=True, exist_ok=True)

    cmds = build_commands(run_tag)
    manifest = {
        "run_tag": run_tag,
        "partition": args.partition,
        "qos": args.qos,
        "gres": args.gres,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
        "time": args.time,
        "num_jobs": len(cmds),
        "jobs": [],
    }

    for name, cmd in cmds:
        rec = {"name": name, "cmd": cmd, "job_id": ""}
        if args.submit:
            sb = [
                "sbatch",
                "--parsable",
                "--job-name",
                f"brem_{name[:40]}",
                "--partition",
                args.partition,
                "--qos",
                args.qos,
                "--gres",
                args.gres,
                "--cpus-per-task",
                str(args.cpus_per_task),
                "--mem",
                args.mem,
                "--time",
                args.time,
                "--export",
                f"ALL,WORKDIR={EXP_DIR},VENV_PATH={VENV}",
                str(RUNNER),
            ] + cmd
            res = subprocess.run(sb, check=True, capture_output=True, text=True)
            rec["job_id"] = res.stdout.strip()
        manifest["jobs"].append(rec)

    out_json = track_dir / "submitted_jobs.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"run_tag": run_tag, "num_jobs": len(cmds), "manifest": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
