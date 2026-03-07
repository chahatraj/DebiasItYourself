#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


BASELINE_ROOT = Path("/scratch/craj/diy/results/10_additional_benchmarks/baselines/baseline_evalshared_all_20260306_1249")
GIT_ENV = {
    "GIT_AUTHOR_NAME": "craj",
    "GIT_AUTHOR_EMAIL": "craj@local",
    "GIT_COMMITTER_NAME": "craj",
    "GIT_COMMITTER_EMAIL": "craj@local",
}


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def dataset_key(raw: str) -> str:
    s = str(raw)
    if s.startswith("unqover:"):
        return s
    if s.startswith("bbq:"):
        return "bbq"
    return s


def baseline_file_for(dataset: str) -> Optional[Path]:
    if dataset == "crowspairs":
        return BASELINE_ROOT / "crows_pairs_metrics_overall_llama_8b.csv"
    if dataset == "stereoset":
        return BASELINE_ROOT / "stereoset_metrics_llama_8b.csv"
    if dataset == "bbq":
        return BASELINE_ROOT / "bbq" / "bbq_metrics_baseline_evalshared_all_20260306_1249.csv"
    if dataset == "bold":
        return BASELINE_ROOT / "bold" / "llama_8b" / "bold_metrics_overall_llama_8b.csv"
    if dataset == "honest":
        return BASELINE_ROOT / "honest" / "llama_8b" / "honest_metrics_overall_llama_8b.csv"
    if dataset == "winobias":
        return BASELINE_ROOT / "winobias" / "llama_8b" / "winobias_metrics_overall_llama_8b.csv"
    if dataset == "winogender":
        return BASELINE_ROOT / "winogender" / "llama_8b" / "winogender_metrics_overall_llama_8b.csv"
    if dataset.startswith("unqover:"):
        dim = dataset.split(":", 1)[1]
        return BASELINE_ROOT / "unqover" / dim / "llama_8b" / f"unqover_{dim}_metrics_overall_llama_8b.csv"
    if dataset == "bias_in_bios":
        return BASELINE_ROOT / "bias_in_bios" / "llama_8b" / "bias_in_bios_metrics_overall_llama_8b.csv"
    return None


def to_float(v) -> Optional[float]:
    try:
        fv = float(v)
        if pd.isna(fv):
            return None
        return fv
    except Exception:
        return None


def score_from_file(path: Path, dataset: str) -> Optional[float]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    dkey = dataset_key(dataset)
    if dkey == "crowspairs":
        v = to_float(df.iloc[0].get("metric_score"))
        return None if v is None else abs(v - 50.0)
    if dkey == "stereoset":
        if "ICAT Score" not in df.columns:
            return None
        v = to_float(df["ICAT Score"].mean())
        return None if v is None else -v
    if dkey == "bbq":
        row = df
        if "input_file" in df.columns:
            overall = df[df["input_file"].astype(str) == "__overall__"]
            if not overall.empty:
                row = overall
        b1 = to_float(row.iloc[0].get("Bias_score_disambig"))
        b2 = to_float(row.iloc[0].get("Bias_score_ambig"))
        acc = to_float(row.iloc[0].get("Accuracy")) or 0.0
        if b1 is None or b2 is None:
            return None
        return abs(b1) + abs(b2) - 0.1 * acc
    if dkey == "bold":
        tox = to_float(df.iloc[0].get("toxicity_prop"))
        if tox is not None:
            return tox
        neg = to_float(df.iloc[0].get("sentiment_negative_prop")) or 0.0
        m = to_float(df.iloc[0].get("gender_unigram_male_prop")) or 0.0
        f = to_float(df.iloc[0].get("gender_unigram_female_prop")) or 0.0
        return neg + abs(m - f)
    if dkey == "honest":
        v = to_float(df.iloc[0].get("honest_score"))
        return v
    if dkey == "winobias":
        v = to_float(df.iloc[0].get("abs_pro_anti_gap"))
        return v
    if dkey == "winogender":
        v = to_float(df.iloc[0].get("mean_abs_occupation_gender_bias_score"))
        return v
    if dkey.startswith("unqover:"):
        d = to_float(df.iloc[0].get("delta_positional_error")) or 0.0
        e = to_float(df.iloc[0].get("epsilon_attributive_error")) or 0.0
        m = to_float(df.iloc[0].get("mu_bias_intensity")) or 0.0
        return abs(d) + abs(e) + abs(m)
    if dkey == "bias_in_bios":
        tpr = to_float(df.iloc[0].get("mean_abs_tpr_gap")) or 0.0
        gap = to_float(df.iloc[0].get("gender_accuracy_gap_abs")) or 0.0
        return tpr + gap
    return None


def fetch_states(job_ids: List[str]) -> Dict[str, str]:
    if not job_ids:
        return {}
    cmd = [
        "sacct",
        "-X",
        "-j",
        ",".join(job_ids),
        "--format=JobIDRaw,State",
        "--parsable2",
        "-n",
    ]
    out = subprocess.check_output(cmd, text=True)
    states: Dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        jid, state = parts[0].strip(), parts[1].strip()
        if jid and state:
            states[jid] = state
    return states


def git_commit_files(repo_root: Path, files: List[Path], message: str) -> bool:
    rels = [str(p.relative_to(repo_root)) for p in files if p.exists()]
    if not rels:
        return False
    subprocess.run(["git", "-C", str(repo_root), "add", "--"] + rels, check=True)
    # No-op commit guard
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
        check=False,
    )
    if diff.returncode == 0:
        return False
    env = os.environ.copy()
    env.update(GIT_ENV)
    subprocess.run(["git", "-C", str(repo_root), "commit", "-m", message], check=True, env=env)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor new-method jobs and commit improvements.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--poll-secs", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    repo_root = Path(manifest["root"])
    run_dir = manifest_path.parent
    state_path = run_dir / "monitor_state.json"

    if state_path.exists():
        state = read_json(state_path)
    else:
        state = {"committed_jobs": {}, "checked_jobs": {}, "history": []}

    jobs = [j for j in manifest["jobs"] if j.get("job_id")]
    job_ids = [str(j["job_id"]) for j in jobs]
    job_by_id = {str(j["job_id"]): j for j in jobs}

    while True:
        states = fetch_states(job_ids)
        changed = False

        for jid, slurm_state in states.items():
            job = job_by_id.get(jid)
            if job is None:
                continue

            jname = job["name"]
            prev = state["checked_jobs"].get(jid, "")
            if prev != slurm_state:
                state["history"].append({"job_id": jid, "job_name": jname, "state": slurm_state, "ts": int(time.time())})
                state["checked_jobs"][jid] = slurm_state
                changed = True

            if not slurm_state.startswith("COMPLETED"):
                continue
            if jid in state["committed_jobs"]:
                continue

            expected = [Path(p) for p in job.get("expected_files", [])]
            ready = [p for p in expected if p.exists()]
            if not ready:
                continue

            dkey = dataset_key(job.get("dataset", ""))
            baseline_path = baseline_file_for(dkey)
            baseline_score = score_from_file(baseline_path, dkey) if baseline_path and baseline_path.exists() else None
            cand_score = score_from_file(ready[0], dkey)
            improved = False
            if cand_score is not None:
                if baseline_score is None:
                    improved = True
                else:
                    improved = cand_score < baseline_score

            if improved:
                msg = f"improve: {job['method']} {job['dataset']} ({job['name']}, job {jid})"
                try:
                    committed = git_commit_files(repo_root, ready, msg)
                    if committed:
                        state["committed_jobs"][jid] = {
                            "job_name": jname,
                            "files": [str(p) for p in ready],
                            "baseline_score": baseline_score,
                            "candidate_score": cand_score,
                            "commit_message": msg,
                        }
                        state["history"].append({"job_id": jid, "event": "commit", "msg": msg, "ts": int(time.time())})
                        changed = True
                except Exception as exc:
                    state["history"].append({"job_id": jid, "event": "commit_error", "error": str(exc), "ts": int(time.time())})
                    changed = True

        if changed:
            write_json(state_path, state)

        if args.once:
            break
        time.sleep(max(30, int(args.poll_secs)))


if __name__ == "__main__":
    main()
