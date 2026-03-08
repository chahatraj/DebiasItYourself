#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path("/scratch/craj/diy")
EXP_DIR = ROOT / "src" / "3_experiments"
SLURM_DIR = ROOT / "src" / "4_slurmjobs"
RESV_RUNNER = SLURM_DIR / "run_single_baseline.slurm"
ANY_RUNNER = SLURM_DIR / "run_single_baseline_anynode.slurm"
VENV = "/home/craj/nanotron-env/bin/activate"

OUTPUT_ROOT = ROOT / "outputs" / "new_outputs"
RESULTS_ROOT = ROOT / "results" / "new_results"
LOG_ROOT = Path("/scratch/craj/logs/diy/cognitive_methods")
TRACK_ROOT = ROOT / "tracking" / "cognitive_methods"

RESERVATION = "craj_278"
PARTITION = "contrib-gpuq"
RESV_NODE = "gpu029"
RESV_QOS = "cs_dept"
OFFLOAD_QOS = "gpu"

MODEL_KEY = "llama_8b"
FT_MODEL_PATH = "/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-allstrategies-opinion-action-event-allversions"

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

ADDL_DATASETS = ["bold", "honest", "winobias", "winogender", "bias_in_bios"]
UNQOVER_DIMS = ["gender", "race", "religion", "nationality"]

METHODS = [
    {"key": "m14_strategy_conditioned_dpo", "script": "14_strategy_conditioned_dpo.py", "tag": "m14_dpo_exact", "model_path": FT_MODEL_PATH},
    {"key": "m15_multi_objective_sft", "script": "15_multi_objective_sft.py", "tag": "m15_mosft_exact", "model_path": FT_MODEL_PATH},
    {"key": "m16_strategy_router_adapter_fusion", "script": "16_strategy_router_adapter_fusion.py", "tag": "m16_router_exact", "model_path": FT_MODEL_PATH},
    {"key": "m17_counterfactual_consistency", "script": "17_counterfactual_consistency.py", "tag": "m17_counterfactual_exact", "model_path": FT_MODEL_PATH},
    {"key": "m18_bias_reward_reranking", "script": "18_bias_reward_reranking.py", "tag": "m18_rerank_exact", "model_path": FT_MODEL_PATH},
    {"key": "m19_contrastive_decoding", "script": "19_contrastive_decoding.py", "tag": "m19_contrastive_exact", "model_path": FT_MODEL_PATH},
    {"key": "m20_self_refine_strategy_critique", "script": "20_self_refine_strategy_critique.py", "tag": "m20_selfrefine_exact", "model_path": FT_MODEL_PATH},
    {"key": "m21_hard_negative_curriculum", "script": "21_hard_negative_curriculum.py", "tag": "m21_hardneg_exact", "model_path": FT_MODEL_PATH},
]


@dataclass
class JobSpec:
    name: str
    method: str
    dataset: str
    cmd: List[str]
    queue: str  # reservation|offload
    time: str
    mem: str
    cpus: int
    workdir: str = str(EXP_DIR)
    dependency_names: List[str] = field(default_factory=list)
    expected_files: List[str] = field(default_factory=list)
    notes: str = ""
    job_id: Optional[str] = None


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def ensure_dirs(ts: str) -> Dict[str, Path]:
    run_dir = TRACK_ROOT / f"run_{ts}"
    log_out = LOG_ROOT / "out"
    log_err = LOG_ROOT / "err"
    for p in [OUTPUT_ROOT, RESULTS_ROOT, run_dir, log_out, log_err]:
        p.mkdir(parents=True, exist_ok=True)
    return {"run_dir": run_dir, "log_out": log_out, "log_err": log_err}


def _result_file_for(dataset: str, results_dir: Path, model_tag: str, unqover_dim: Optional[str] = None) -> Optional[str]:
    if dataset == "crowspairs":
        return str(results_dir / f"crows_pairs_metrics_overall_{model_tag}.csv")
    if dataset == "stereoset":
        return str(results_dir / f"stereoset_metrics_{model_tag}.csv")
    if dataset == "bold":
        return str(results_dir / "bold" / model_tag / f"bold_metrics_overall_{model_tag}.csv")
    if dataset == "honest":
        return str(results_dir / "honest" / model_tag / f"honest_metrics_overall_{model_tag}.csv")
    if dataset == "winobias":
        return str(results_dir / "winobias" / model_tag / f"winobias_metrics_overall_{model_tag}.csv")
    if dataset == "winogender":
        return str(results_dir / "winogender" / model_tag / f"winogender_metrics_overall_{model_tag}.csv")
    if dataset == "bias_in_bios":
        return str(results_dir / "bias_in_bios" / model_tag / f"bias_in_bios_metrics_overall_{model_tag}.csv")
    if dataset == "unqover" and unqover_dim:
        return str(results_dir / "unqover" / unqover_dim / model_tag / f"unqover_{unqover_dim}_metrics_overall_{model_tag}.csv")
    return None


def build_jobs() -> List[JobSpec]:
    jobs: List[JobSpec] = []

    for method in METHODS:
        mkey = method["key"]
        script = method["script"]
        tag = method["tag"]
        model_path = method["model_path"]

        out_eval = OUTPUT_ROOT / mkey / "evalshared"
        res_eval = RESULTS_ROOT / mkey / "evalshared"

        for ds in ["crowspairs", "stereoset"] + ADDL_DATASETS:
            model_tag = f"{tag}_{ds}" if ds in ADDL_DATASETS else tag
            cmd = [
                "python", script,
                "--dataset", ds,
                "--model", MODEL_KEY,
                "--model_path", model_path,
                "--output_dir", str(out_eval),
                "--results_dir", str(res_eval),
                "--model_tag", model_tag,
            ]
            expected = _result_file_for(ds, res_eval, model_tag)
            jobs.append(
                JobSpec(
                    name=f"{mkey}_{ds}",
                    method=mkey,
                    dataset=ds,
                    cmd=cmd,
                    queue="offload",
                    time="0-10:00:00",
                    mem="70G",
                    cpus=10,
                    expected_files=[expected] if expected else [],
                )
            )

        for dim in UNQOVER_DIMS:
            model_tag = f"{tag}_unqover_{dim}"
            cmd = [
                "python", script,
                "--dataset", "unqover",
                "--unqover_dim", dim,
                "--model", MODEL_KEY,
                "--model_path", model_path,
                "--output_dir", str(out_eval),
                "--results_dir", str(res_eval),
                "--model_tag", model_tag,
            ]
            expected = _result_file_for("unqover", res_eval, model_tag, unqover_dim=dim)
            jobs.append(
                JobSpec(
                    name=f"{mkey}_unqover_{dim}",
                    method=mkey,
                    dataset=f"unqover:{dim}",
                    cmd=cmd,
                    queue="offload",
                    time="0-12:00:00",
                    mem="70G",
                    cpus=10,
                    expected_files=[expected] if expected else [],
                )
            )

        out_bbq = OUTPUT_ROOT / mkey / "bbq"
        res_bbq = RESULTS_ROOT / mkey / "bbq"
        infer_names: List[str] = []
        for src in BBQ_SOURCE_FILES:
            cat = src.replace(".jsonl", "")
            jn = f"{mkey}_bbq_{cat}"
            infer_names.append(jn)
            cmd = [
                "python", script,
                "--dataset", "bbq",
                "--source_file", src,
                "--model", MODEL_KEY,
                "--model_path", model_path,
                "--output_dir", str(out_bbq),
                "--results_dir", str(res_bbq),
                "--model_tag", tag,
            ]
            jobs.append(
                JobSpec(
                    name=jn,
                    method=mkey,
                    dataset=f"bbq:{cat}",
                    cmd=cmd,
                    queue="reservation",
                    time="0-08:00:00",
                    mem="70G",
                    cpus=10,
                )
            )

        eval_cmd = [
            "python", script,
            "--dataset", "bbq_eval",
            "--output_dir", str(out_bbq),
            "--results_dir", str(res_bbq),
            "--model_tag", tag,
        ]
        jobs.append(
            JobSpec(
                name=f"{mkey}_bbq_eval",
                method=mkey,
                dataset="bbq",
                cmd=eval_cmd,
                queue="reservation",
                time="0-02:00:00",
                mem="40G",
                cpus=6,
                dependency_names=infer_names,
                expected_files=[str(res_bbq / f"bbq_metrics_{slug(tag)}.csv")],
            )
        )

    return jobs


def submit_job(job: JobSpec, dep_job_ids: List[str], log_out_dir: Path, log_err_dir: Path) -> str:
    runner = str(RESV_RUNNER if job.queue == "reservation" else ANY_RUNNER)
    submit = [
        "sbatch",
        "--parsable",
        "--job-name",
        job.name[:120],
        "--partition",
        PARTITION,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--gres",
        "gpu:A100.80gb:1",
        "--cpus-per-task",
        str(job.cpus),
        "--mem",
        job.mem,
        "--time",
        job.time,
        "--output",
        str(log_out_dir / f"{job.name}.%j.out.txt"),
        "--error",
        str(log_err_dir / f"{job.name}.%j.err.txt"),
        "--export",
        f"ALL,WORKDIR={job.workdir},VENV_PATH={VENV}",
    ]
    if job.queue == "reservation":
        submit += ["--reservation", RESERVATION, "--qos", RESV_QOS, "--nodelist", RESV_NODE]
    else:
        submit += ["--qos", OFFLOAD_QOS]

    if dep_job_ids:
        submit += ["--dependency", "afterok:" + ":".join(dep_job_ids), "--kill-on-invalid-dep=yes"]

    submit += [runner] + job.cmd
    res = subprocess.run(submit, text=True, capture_output=True)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, submit, output=res.stdout, stderr=res.stderr)
    out = (res.stdout or "").strip()
    return out.split(";")[0].strip()


def existing_jobs_by_name() -> Dict[str, str]:
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "squeue -u craj -h -o '%i|%j'"],
            text=True,
        )
    except Exception:
        return {}

    mapping: Dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) != 2:
            continue
        jid, name = parts
        if jid and name and name not in mapping:
            mapping[name] = jid
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit cognitive-method matrix jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Build manifest but do not call sbatch.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = ensure_dirs(ts)
    run_dir = dirs["run_dir"]
    log_out_dir = dirs["log_out"]
    log_err_dir = dirs["log_err"]

    jobs = build_jobs()
    name_to_id: Dict[str, str] = {}
    submit_log = []
    existing = existing_jobs_by_name()

    for job in jobs:
        if job.name in existing:
            jid = existing[job.name]
            job.job_id = jid
            name_to_id[job.name] = jid
            submit_log.append(
                {
                    "job_name": job.name,
                    "job_id": jid,
                    "method": job.method,
                    "dataset": job.dataset,
                    "queue": job.queue,
                    "deps": [],
                    "cmd": " ".join(shlex.quote(x) for x in job.cmd),
                    "expected_files": job.expected_files,
                    "reused_existing_job": True,
                }
            )
            print(f"[REUSE] {jid} {job.name}")
            continue

        dep_ids = [name_to_id[d] for d in job.dependency_names if d in name_to_id]
        try:
            if args.dry_run:
                jid = ""
            else:
                jid = submit_job(job, dep_ids, log_out_dir, log_err_dir)
            job.job_id = jid
            if jid:
                name_to_id[job.name] = jid
            submit_log.append(
                {
                    "job_name": job.name,
                    "job_id": jid,
                    "method": job.method,
                    "dataset": job.dataset,
                    "queue": job.queue,
                    "deps": dep_ids,
                    "cmd": " ".join(shlex.quote(x) for x in job.cmd),
                    "expected_files": job.expected_files,
                    "dry_run": bool(args.dry_run),
                }
            )
            if args.dry_run:
                print(f"[DRYRUN] {job.name} queue={job.queue} deps={dep_ids}")
            else:
                print(f"[SUBMIT] {jid} {job.name} queue={job.queue} deps={dep_ids}")
        except subprocess.CalledProcessError as exc:
            submit_log.append(
                {
                    "job_name": job.name,
                    "job_id": None,
                    "method": job.method,
                    "dataset": job.dataset,
                    "queue": job.queue,
                    "deps": dep_ids,
                    "cmd": " ".join(shlex.quote(x) for x in job.cmd),
                    "expected_files": job.expected_files,
                    "error_stdout": exc.output,
                    "error_stderr": exc.stderr,
                }
            )
            serr = (exc.stderr or "").strip()
            print(f"[ERROR] submit failed for {job.name}: rc={exc.returncode} stderr={serr}")

    manifest = {
        "timestamp": ts,
        "root": str(ROOT),
        "output_root": str(OUTPUT_ROOT),
        "results_root": str(RESULTS_ROOT),
        "log_out_dir": str(log_out_dir),
        "log_err_dir": str(log_err_dir),
        "jobs_total": len(jobs),
        "jobs_submitted": sum(1 for j in jobs if j.job_id),
        "jobs": [asdict(j) for j in jobs],
        "submit_log": submit_log,
        "dry_run": bool(args.dry_run),
    }

    manifest_path = run_dir / "jobs_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] manifest={manifest_path}")


if __name__ == "__main__":
    main()
