#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, List


PARTITION = "contrib-gpuq"
RESERVATION = "craj_278"
RESV_NODE = "gpu029"
RESV_QOS = "cs_dept"
OFFLOAD_QOS = "gpu"
VENV = "/home/craj/nanotron-env/bin/activate"
RESV_RUNNER = "/scratch/craj/diy/src/4_slurmjobs/run_single_baseline.slurm"
ANY_RUNNER = "/scratch/craj/diy/src/4_slurmjobs/run_single_baseline_anynode.slurm"


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def current_queue_count() -> int:
    out = subprocess.check_output(["bash", "-lc", "squeue -u craj -h | wc -l"], text=True).strip()
    try:
        return int(out)
    except Exception:
        return 999


def submit_job(job: dict, dep_ids: List[str], log_out_dir: str, log_err_dir: str) -> str:
    queue = job["queue"]
    runner = RESV_RUNNER if queue == "reservation" else ANY_RUNNER
    name = job["name"][:120]
    submit = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--partition",
        PARTITION,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--gres",
        "gpu:A100.80gb:1",
        "--cpus-per-task",
        str(job["cpus"]),
        "--mem",
        job["mem"],
        "--time",
        job["time"],
        "--output",
        f"{log_out_dir}/{name}.%j.out.txt",
        "--error",
        f"{log_err_dir}/{name}.%j.err.txt",
        "--export",
        f"ALL,WORKDIR={job['workdir']},VENV_PATH={VENV}",
    ]
    if queue == "reservation":
        submit += ["--reservation", RESERVATION, "--qos", RESV_QOS, "--nodelist", RESV_NODE]
    else:
        submit += ["--qos", OFFLOAD_QOS]
    if dep_ids:
        submit += ["--dependency", "afterok:" + ":".join(dep_ids), "--kill-on-invalid-dep=yes"]
    submit += [runner] + list(job["cmd"])
    out = subprocess.check_output(submit, text=True).strip()
    return out.split(";")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refill unsubmitted jobs from manifest as queue slots become available.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--target-queued", type=int, default=95)
    parser.add_argument("--poll-secs", type=int, default=180)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    while True:
        manifest = read_json(manifest_path)
        jobs = manifest["jobs"]
        name_to_id: Dict[str, str] = {j["name"]: str(j["job_id"]) for j in jobs if j.get("job_id")}

        queued = current_queue_count()
        budget = max(0, args.target_queued - queued)
        if budget <= 0:
            time.sleep(max(30, args.poll_secs))
            continue

        submitted_now = 0
        for j in jobs:
            if submitted_now >= budget:
                break
            if j.get("job_id"):
                continue
            dep_names = j.get("dependency_names", [])
            if any(dn not in name_to_id for dn in dep_names):
                continue
            dep_ids = [name_to_id[dn] for dn in dep_names]
            try:
                jid = submit_job(j, dep_ids, manifest["log_out_dir"], manifest["log_err_dir"])
                j["job_id"] = jid
                name_to_id[j["name"]] = jid
                manifest["submit_log"].append(
                    {
                        "job_name": j["name"],
                        "job_id": jid,
                        "method": j["method"],
                        "dataset": j["dataset"],
                        "queue": j["queue"],
                        "deps": dep_ids,
                        "cmd": " ".join(shlex.quote(x) for x in j["cmd"]),
                        "expected_files": j.get("expected_files", []),
                        "resubmitted_by_refill": True,
                    }
                )
                submitted_now += 1
                print(f"[REFILL-SUBMIT] {jid} {j['name']}")
            except subprocess.CalledProcessError:
                # Still blocked by qos limits; try next cycle.
                break

        if submitted_now > 0:
            manifest["jobs_submitted"] = sum(1 for j in jobs if j.get("job_id"))
            write_json(manifest_path, manifest)

        time.sleep(max(30, args.poll_secs))


if __name__ == "__main__":
    main()
