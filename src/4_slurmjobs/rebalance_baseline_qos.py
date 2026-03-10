#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from typing import List, Tuple


USER = "craj"


def run(cmd: List[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return res.stdout


def gpu_qos_counts() -> Tuple[int, int]:
    """
    Returns (running, total) counts for jobs under qos=gpu for USER.
    """
    out = run(["bash", "-lc", f"squeue -h -u {USER} -o '%q %T'"])
    running = 0
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        qos, state = parts[0], parts[1]
        if qos != "gpu":
            continue
        total += 1
        if state == "RUNNING":
            running += 1
    return running, total


def blocked_baseline_jobs() -> List[str]:
    """
    Pending baseline jobs blocked by cs_dept A100 quota.
    """
    out = run(
        [
            "bash",
            "-lc",
            (
                f"squeue -h -u {USER} -t PENDING -o '%i %j %q %R' "
                r"| awk '$2 ~ /^brem_/ && $3==\"cs_dept\" && $4 ~ /QOSGrpGRES/ {print $1}'"
            ),
        ]
    )
    return [x.strip() for x in out.splitlines() if x.strip()]


def set_gpu_qos(job_id: str) -> bool:
    res = subprocess.run(
        ["scontrol", "update", f"JobId={job_id}", "QOS=gpu"],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously rebalance blocked baseline jobs to qos=gpu.")
    parser.add_argument("--gpu-max-submit", type=int, default=40)
    parser.add_argument("--poll-secs", type=int, default=90)
    args = parser.parse_args()

    while True:
        try:
            gpu_running, gpu_total = gpu_qos_counts()
            room = max(0, args.gpu_max_submit - gpu_total)
            if room > 0:
                blocked = blocked_baseline_jobs()
                moved = 0
                for jid in blocked[:room]:
                    ok = set_gpu_qos(jid)
                    if ok:
                        moved += 1
                        print(f"[REBALANCE] moved JobId={jid} to qos=gpu", flush=True)
                    else:
                        # Stop when scheduler rejects policy changes.
                        break
                if moved > 0:
                    print(
                        f"[REBALANCE] gpu_running={gpu_running} gpu_total={gpu_total} room={room} moved={moved}",
                        flush=True,
                    )
            time.sleep(max(30, args.poll_secs))
        except Exception as exc:
            print(f"[REBALANCE-ERROR] {exc}", flush=True)
            time.sleep(max(30, args.poll_secs))


if __name__ == "__main__":
    main()
