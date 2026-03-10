#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path("/scratch/craj/diy")
EXP_DIR = ROOT / "src/3_experiments"
SLURM_DIR = ROOT / "src/4_slurmjobs"
RUNNER_SLURM = SLURM_DIR / "run_job_array.slurm"

LOG_OUT_DIR = Path("/scratch/craj/logs/diy/out")
LOG_ERR_DIR = Path("/scratch/craj/logs/diy/err")
DEFAULT_VENV = "/home/craj/nanotron-env/bin/activate"

BBQ_SOURCE_FILES = [
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

METHODS = {"cot", "token_insertion", "post_inference_correction"}

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
}

ALL_STRATEGIES = [
    "stereotype_replacement",
    "counter_imaging",
    "individuating",
    "perspective_taking",
    "positive_contact",
    "all_strategies",
]


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def normalize_strategy(raw: str) -> str:
    key = str(raw).strip().lower()
    if key not in STRATEGY_ALIASES:
        raise ValueError(f"Unknown strategy `{raw}`")
    return STRATEGY_ALIASES[key]


def parse_methods(raw: str) -> List[str]:
    vals = [v.strip() for v in parse_csv(raw)]
    if not vals:
        raise ValueError("At least one method is required")
    for v in vals:
        if v not in METHODS:
            raise ValueError(f"Unknown method `{v}`")
    return list(dict.fromkeys(vals))


def parse_strategies(raw: str) -> List[str]:
    if str(raw).strip().lower() == "all":
        return list(ALL_STRATEGIES)
    vals = [normalize_strategy(v) for v in parse_csv(raw)]
    if not vals:
        raise ValueError("At least one strategy is required")
    return list(dict.fromkeys(vals))


def maybe_arg(flag: str, value: Optional[object]) -> str:
    if value is None:
        return ""
    return f" {flag} {value}"


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip() + "\n")


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def submit_array_stage(
    *,
    stage_name: str,
    array_file: Path,
    run_tag: str,
    partition: str,
    qos: str,
    gres: str,
    mem: str,
    time_limit: str,
    max_parallel: int,
    venv_path: str,
    dependency_job_id: Optional[str] = None,
) -> str:
    n = count_nonempty_lines(array_file)
    if n == 0:
        print(f"[SKIP] {stage_name}: empty array file {array_file}")
        return ""

    if max_parallel > 0:
        array_spec = f"1-{n}%{max_parallel}"
    else:
        array_spec = f"1-{n}"

    out_file = LOG_OUT_DIR / f"{stage_name}.{run_tag}.%A_%a.out.txt"
    err_file = LOG_ERR_DIR / f"{stage_name}.{run_tag}.%A_%a.err.txt"

    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        stage_name,
        "--partition",
        partition,
        "--qos",
        qos,
        "--gres",
        gres,
        "--mem",
        mem,
        "--time",
        time_limit,
        "--array",
        array_spec,
        "--output",
        str(out_file),
        "--error",
        str(err_file),
        "--export",
        f"ALL,ARRAY_FILE={array_file},WORKDIR={EXP_DIR},VENV_PATH={venv_path}",
        str(RUNNER_SLURM),
    ]

    if dependency_job_id:
        cmd[2:2] = ["--dependency", f"afterok:{dependency_job_id}"]

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    job_id = res.stdout.strip()
    print(
        f"[SUBMIT] {stage_name}: job={job_id} tasks={n} partition={partition} qos={qos} gres={gres}"
    )
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and submit jobs for reasoning/token/post debias experiments")
    parser.add_argument("--run_tag", type=str, default=f"reasoning_token_post_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    parser.add_argument("--model", type=str, default="llama_8b")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--methods", type=str, default="cot,token_insertion,post_inference_correction")
    parser.add_argument("--strategies", type=str, default="sr,ci,ind,pt,pc")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--partition", type=str, default="contrib-gpuq")
    parser.add_argument("--qos", type=str, default="cs_dept")
    parser.add_argument("--gres", type=str, default="gpu:3g.40gb:1")
    parser.add_argument("--mem", type=str, default="40G")

    parser.add_argument("--crows_time", type=str, default="0-03:00:00")
    parser.add_argument("--stereo_time", type=str, default="0-04:00:00")
    parser.add_argument("--bbq_infer_time", type=str, default="0-06:00:00")
    parser.add_argument("--bbq_eval_time", type=str, default="0-01:00:00")

    parser.add_argument("--crows_max_parallel", type=int, default=0)
    parser.add_argument("--stereo_max_parallel", type=int, default=0)
    parser.add_argument("--bbq_infer_max_parallel", type=int, default=0)
    parser.add_argument("--bbq_eval_max_parallel", type=int, default=0)

    parser.add_argument("--venv_path", type=str, default=DEFAULT_VENV)
    parser.add_argument("--submit", action="store_true", help="If set, submit sbatch jobs after writing arrays.")
    parser.add_argument(
        "--submit_bbq_infer",
        action="store_true",
        help="Submit BBQ inference array (disabled by default).",
    )
    parser.add_argument(
        "--submit_bbq_eval",
        action="store_true",
        help="Submit BBQ eval array (disabled by default).",
    )
    args = parser.parse_args()

    run_tag = slug(args.run_tag)
    run_root = ROOT / "outputs/9_reasoning_token_post" / run_tag
    results_root = ROOT / "results/9_reasoning_token_post" / run_tag
    array_dir = SLURM_DIR / "arrays" / run_tag

    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    array_dir.mkdir(parents=True, exist_ok=True)
    LOG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ERR_DIR.mkdir(parents=True, exist_ok=True)

    methods = parse_methods(args.methods)
    strategies = parse_strategies(args.strategies)

    crows_lines: List[str] = []
    stereo_lines: List[str] = []
    bbq_infer_lines: List[str] = []
    bbq_eval_lines: List[str] = []
    registry_rows: List[Dict[str, str]] = []

    for method in methods:
        for strategy in strategies:
            tag = slug(f"dbm_{method}_{strategy}")

            crows_out = run_root / "crowspairs" / tag
            crows_res = results_root / "crowspairs" / tag
            stereo_out = run_root / "stereoset" / tag
            stereo_res = results_root / "stereoset" / tag
            bbq_out = run_root / "bbq" / tag
            bbq_res_file = results_root / "bbq" / f"{tag}.csv"

            crows_cmd = (
                "python 6_reasoning_token_post_experiment.py eval_crows "
                f"--model {args.model} "
                f"--debias_method {method} "
                f"--strategy {strategy} "
                f"--model_tag {tag} "
                f"--output_dir {crows_out} "
                f"--results_dir {crows_res} "
                "--batch_size 4"
                f"{maybe_arg('--limit', args.limit)}"
                f"{maybe_arg('--model_path', args.model_path)}"
            )
            crows_lines.append(crows_cmd)

            stereo_cmd = (
                "python 6_reasoning_token_post_experiment.py eval_stereoset "
                f"--model {args.model} "
                f"--debias_method {method} "
                f"--strategy {strategy} "
                f"--model_tag {tag} "
                f"--output_dir {stereo_out} "
                f"--results_dir {stereo_res} "
                "--batch_size 4"
                f"{maybe_arg('--limit', args.limit)}"
                f"{maybe_arg('--model_path', args.model_path)}"
            )
            stereo_lines.append(stereo_cmd)

            for source_file in BBQ_SOURCE_FILES:
                bbq_infer_cmd = (
                    "python 6_reasoning_token_post_experiment.py infer_bbq "
                    f"--model {args.model} "
                    f"--debias_method {method} "
                    f"--strategy {strategy} "
                    f"--model_tag {tag} "
                    f"--source_file {source_file} "
                    f"--output_dir {bbq_out}"
                    f"{maybe_arg('--limit', args.limit)}"
                    f"{maybe_arg('--model_path', args.model_path)}"
                )
                bbq_infer_lines.append(bbq_infer_cmd)

            bbq_eval_cmd = (
                "python 6_reasoning_token_post_experiment.py eval_bbq "
                f"--model_dir {bbq_out} "
                f"--output_file {bbq_res_file} "
                f"--model_name {tag}"
            )
            bbq_eval_lines.append(bbq_eval_cmd)

            registry_rows.append(
                {
                    "tag": tag,
                    "method": method,
                    "strategy": strategy,
                    "crows_out": str(crows_out),
                    "stereo_out": str(stereo_out),
                    "bbq_out": str(bbq_out),
                    "bbq_metrics": str(bbq_res_file),
                }
            )

    crows_array = array_dir / "crows_array.txt"
    stereo_array = array_dir / "stereoset_array.txt"
    bbq_infer_array = array_dir / "bbq_infer_array.txt"
    bbq_eval_array = array_dir / "bbq_eval_array.txt"
    registry_json = array_dir / "registry.json"

    write_lines(crows_array, crows_lines)
    write_lines(stereo_array, stereo_lines)
    write_lines(bbq_infer_array, bbq_infer_lines)
    write_lines(bbq_eval_array, bbq_eval_lines)

    with registry_json.open("w", encoding="utf-8") as f:
        json.dump(registry_rows, f, indent=2)

    meta = {
        "run_tag": run_tag,
        "run_root": str(run_root),
        "results_root": str(results_root),
        "array_dir": str(array_dir),
        "model": args.model,
        "model_path": args.model_path,
        "methods": methods,
        "strategies": strategies,
        "counts": {
            "crows_jobs": len(crows_lines),
            "stereoset_jobs": len(stereo_lines),
            "bbq_infer_jobs": len(bbq_infer_lines),
            "bbq_eval_jobs": len(bbq_eval_lines),
        },
        "files": {
            "crows_array": str(crows_array),
            "stereoset_array": str(stereo_array),
            "bbq_infer_array": str(bbq_infer_array),
            "bbq_eval_array": str(bbq_eval_array),
            "registry": str(registry_json),
        },
    }

    job_ids = {}
    if args.submit:
        crows_job = submit_array_stage(
            stage_name="dbm_crows",
            array_file=crows_array,
            run_tag=run_tag,
            partition=args.partition,
            qos=args.qos,
            gres=args.gres,
            mem=args.mem,
            time_limit=args.crows_time,
            max_parallel=args.crows_max_parallel,
            venv_path=args.venv_path,
        )
        stereo_job = submit_array_stage(
            stage_name="dbm_stereo",
            array_file=stereo_array,
            run_tag=run_tag,
            partition=args.partition,
            qos=args.qos,
            gres=args.gres,
            mem=args.mem,
            time_limit=args.stereo_time,
            max_parallel=args.stereo_max_parallel,
            venv_path=args.venv_path,
        )
        bbq_infer_job = ""
        if args.submit_bbq_infer:
            bbq_infer_job = submit_array_stage(
                stage_name="dbm_bbqinf",
                array_file=bbq_infer_array,
                run_tag=run_tag,
                partition=args.partition,
                qos=args.qos,
                gres=args.gres,
                mem=args.mem,
                time_limit=args.bbq_infer_time,
                max_parallel=args.bbq_infer_max_parallel,
                venv_path=args.venv_path,
            )

        bbq_eval_job = ""
        if args.submit_bbq_eval:
            bbq_eval_job = submit_array_stage(
                stage_name="dbm_bbqeval",
                array_file=bbq_eval_array,
                run_tag=run_tag,
                partition=args.partition,
                qos=args.qos,
                gres=args.gres,
                mem=args.mem,
                time_limit=args.bbq_eval_time,
                max_parallel=args.bbq_eval_max_parallel,
                venv_path=args.venv_path,
                dependency_job_id=bbq_infer_job if bbq_infer_job else None,
            )

        job_ids = {"crows_job": crows_job, "stereoset_job": stereo_job}
        if bbq_infer_job:
            job_ids["bbq_infer_job"] = bbq_infer_job
        if bbq_eval_job:
            job_ids["bbq_eval_job"] = bbq_eval_job

        submitted_file = array_dir / "submitted_jobs.txt"
        with submitted_file.open("w", encoding="utf-8") as f:
            f.write(f"run_tag: {run_tag}\n")
            f.write(f"run_root: {run_root}\n")
            f.write(f"results_root: {results_root}\n")
            for key, val in job_ids.items():
                f.write(f"{key}: {val}\n")
        meta["submitted_jobs"] = str(submitted_file)

    meta["job_ids"] = job_ids
    meta["submit_bbq_infer"] = args.submit_bbq_infer
    meta["submit_bbq_eval"] = args.submit_bbq_eval

    meta_json = array_dir / "array_meta.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print("Monitor: squeue -u craj -o '%.18i %.15P %.30j %.10T %.12M %.6D %R'")


if __name__ == "__main__":
    main()
