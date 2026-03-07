#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
LOG_ROOT = Path("/scratch/craj/logs/diy/new_methods")
TRACK_ROOT = ROOT / "tracking" / "new_methods"

RESERVATION = "craj_278"
PARTITION = "contrib-gpuq"
RESV_NODE = "gpu029"
RESV_QOS = "cs_dept"
OFFLOAD_QOS = "gpu"

MODEL_KEY = "llama_8b"
STRATEGY = "stereotype_replacement"
FT_MODEL_PATH = "/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-allstrategies-opinion-action-event-allversions"
SOFT_PROMPT_DIR = "/scratch/craj/diy/outputs/8_soft_prompt/sp_embed_20260228_1/models/learnable"

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


def add_evalshared_job(
    jobs: List[JobSpec],
    *,
    method: str,
    dataset: str,
    model_path: Optional[str],
    model_tag: str,
    inference_mode: str = "off",
    inference_strategy: Optional[str] = None,
    queue: str = "offload",
    extra_args: Optional[List[str]] = None,
) -> None:
    out_dir = OUTPUT_ROOT / method / "evalshared"
    res_dir = RESULTS_ROOT / method / "evalshared"
    cmd = [
        "python",
        "7_eval_shared.py",
        "--dataset",
        dataset,
        "--model",
        MODEL_KEY,
        "--output_dir",
        str(out_dir),
        "--results_dir",
        str(res_dir),
        "--model_tag",
        model_tag,
        "--inference_instruction_mode",
        inference_mode,
    ]
    if inference_strategy:
        cmd += ["--inference_strategy", inference_strategy]
    if model_path:
        cmd += ["--model_path", model_path]
    if extra_args:
        cmd += extra_args

    exp_files: List[str] = []
    if dataset == "crowspairs":
        exp_files.append(str(res_dir / f"crows_pairs_metrics_overall_{model_tag}.csv"))
    elif dataset == "stereoset":
        exp_files.append(str(res_dir / f"stereoset_metrics_{model_tag}.csv"))
    elif dataset in ADDL_DATASETS:
        if dataset == "bold":
            exp_files.append(str(res_dir / "bold" / model_tag / f"bold_metrics_overall_{model_tag}.csv"))
        elif dataset == "honest":
            exp_files.append(str(res_dir / "honest" / model_tag / f"honest_metrics_overall_{model_tag}.csv"))
        elif dataset == "winobias":
            exp_files.append(str(res_dir / "winobias" / model_tag / f"winobias_metrics_overall_{model_tag}.csv"))
        elif dataset == "winogender":
            exp_files.append(str(res_dir / "winogender" / model_tag / f"winogender_metrics_overall_{model_tag}.csv"))
        elif dataset == "bias_in_bios":
            exp_files.append(str(res_dir / "bias_in_bios" / model_tag / f"bias_in_bios_metrics_overall_{model_tag}.csv"))
    jobs.append(
        JobSpec(
            name=f"{method}_{dataset}",
            method=method,
            dataset=dataset,
            cmd=cmd,
            queue=queue,
            time="0-10:00:00",
            mem="70G",
            cpus=10,
            expected_files=exp_files,
        )
    )


def add_unqover_jobs(
    jobs: List[JobSpec],
    *,
    method: str,
    model_path: Optional[str],
    model_tag_prefix: str,
    inference_mode: str,
    inference_strategy: Optional[str],
    queue: str = "offload",
) -> None:
    for dim in UNQOVER_DIMS:
        tag = f"{model_tag_prefix}_unqover_{dim}"
        out_dir = OUTPUT_ROOT / method / "evalshared"
        res_dir = RESULTS_ROOT / method / "evalshared"
        cmd = [
            "python",
            "7_eval_shared.py",
            "--dataset",
            "unqover",
            "--unqover_dim",
            dim,
            "--model",
            MODEL_KEY,
            "--output_dir",
            str(out_dir),
            "--results_dir",
            str(res_dir),
            "--model_tag",
            tag,
            "--inference_instruction_mode",
            inference_mode,
        ]
        if inference_strategy:
            cmd += ["--inference_strategy", inference_strategy]
        if model_path:
            cmd += ["--model_path", model_path]
        expected = str(res_dir / "unqover" / dim / tag / f"unqover_{dim}_metrics_overall_{tag}.csv")
        jobs.append(
            JobSpec(
                name=f"{method}_unqover_{dim}",
                method=method,
                dataset=f"unqover:{dim}",
                cmd=cmd,
                queue=queue,
                time="0-12:00:00",
                mem="70G",
                cpus=10,
                expected_files=[expected],
            )
        )


def add_generic_bbq_jobs(
    jobs: List[JobSpec],
    *,
    method: str,
    model_path: Optional[str],
    model_tag: str,
    inference_mode: str,
    inference_strategy: Optional[str],
    queue: str = "reservation",
) -> None:
    bbq_out = OUTPUT_ROOT / method / "bbq"
    bbq_res = RESULTS_ROOT / method / "bbq"
    infer_names: List[str] = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace(".jsonl", "")
        jn = f"{method}_bbq_{s}"
        infer_names.append(jn)
        cmd = [
            "python",
            "13_bbq_inference_instruction.py",
            "--model",
            MODEL_KEY,
            "--source_file",
            src,
            "--output_dir",
            str(bbq_out),
            "--model_tag",
            model_tag,
            "--inference_instruction_mode",
            inference_mode,
        ]
        if inference_strategy:
            cmd += ["--inference_strategy", inference_strategy]
        if model_path:
            cmd += ["--model_path", model_path]
        expected = str(bbq_out / f"bbq_preds_{model_tag}_{'instr_off' if inference_mode == 'off' else 'instr_' + STRATEGY}_{s}.csv")
        jobs.append(
            JobSpec(
                name=jn,
                method=method,
                dataset=f"bbq:{s}",
                cmd=cmd,
                queue=queue,
                time="0-08:00:00",
                mem="70G",
                cpus=10,
                expected_files=[expected],
            )
        )

    eval_out = bbq_res / f"bbq_metrics_{model_tag}.csv"
    eval_cmd = [
        "python",
        "8_bbq_eval_shared.py",
        "--model_dir",
        str(bbq_out),
        "--output_file",
        str(eval_out),
        "--model_name",
        model_tag,
    ]
    jobs.append(
        JobSpec(
            name=f"{method}_bbq_eval",
            method=method,
            dataset="bbq",
            cmd=eval_cmd,
            queue=queue,
            time="0-02:00:00",
            mem="40G",
            cpus=6,
            dependency_names=infer_names,
            expected_files=[str(eval_out)],
        )
    )


def add_native_method_jobs(jobs: List[JobSpec]) -> None:
    # method5: soft prompting (native on crows/stereo/bbq)
    m = "m5_soft_prompting"
    out = OUTPUT_ROOT / m
    res = RESULTS_ROOT / m
    tag = "m5_softprompt_sr"
    jobs.append(
        JobSpec(
            name=f"{m}_crowspairs",
            method=m,
            dataset="crowspairs",
            cmd=[
                "python",
                "5_soft_prompting.py",
                "eval_crows",
                "--soft_prompt_dir",
                SOFT_PROMPT_DIR,
                "--strategy",
                STRATEGY,
                "--model",
                MODEL_KEY,
                "--output_dir",
                str(out / "crowspairs"),
                "--results_dir",
                str(res / "crowspairs"),
                "--model_tag",
                tag,
            ],
            queue="reservation",
            time="0-08:00:00",
            mem="70G",
            cpus=10,
            expected_files=[str(res / "crowspairs" / f"crows_pairs_metrics_overall_{tag}.csv")],
        )
    )
    jobs.append(
        JobSpec(
            name=f"{m}_stereoset",
            method=m,
            dataset="stereoset",
            cmd=[
                "python",
                "5_soft_prompting.py",
                "eval_stereoset",
                "--soft_prompt_dir",
                SOFT_PROMPT_DIR,
                "--strategy",
                STRATEGY,
                "--model",
                MODEL_KEY,
                "--output_dir",
                str(out / "stereoset"),
                "--results_dir",
                str(res / "stereoset"),
                "--model_tag",
                tag,
            ],
            queue="reservation",
            time="0-10:00:00",
            mem="70G",
            cpus=10,
            expected_files=[str(res / "stereoset" / f"stereoset_metrics_{tag}.csv")],
        )
    )
    infer_names = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace(".jsonl", "")
        jn = f"{m}_bbq_{s}"
        infer_names.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m,
                dataset=f"bbq:{s}",
                cmd=[
                    "python",
                    "5_soft_prompting.py",
                    "infer_bbq",
                    "--soft_prompt_dir",
                    SOFT_PROMPT_DIR,
                    "--strategy",
                    STRATEGY,
                    "--model",
                    MODEL_KEY,
                    "--source_file",
                    src,
                    "--output_dir",
                    str(out / "bbq"),
                    "--model_tag",
                    tag,
                ],
                queue="reservation",
                time="0-08:00:00",
                mem="70G",
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f"{m}_bbq_eval",
            method=m,
            dataset="bbq",
            cmd=[
                "python",
                "8_bbq_eval_shared.py",
                "--model_dir",
                str(out / "bbq"),
                "--output_file",
                str(res / "bbq" / f"bbq_metrics_{tag}.csv"),
                "--model_name",
                tag,
            ],
            queue="reservation",
            time="0-02:00:00",
            mem="40G",
            cpus=6,
            dependency_names=infer_names,
            expected_files=[str(res / "bbq" / f"bbq_metrics_{tag}.csv")],
        )
    )

    # method6: reasoning token/post (native on crows/stereo/bbq)
    m6 = "m6_reasoning_post"
    tag6 = "m6_postcorr_sr"
    out6 = OUTPUT_ROOT / m6
    res6 = RESULTS_ROOT / m6
    for ds, subcmd in [("crowspairs", "eval_crows"), ("stereoset", "eval_stereoset")]:
        cmd = [
            "python",
            "6_reasoning_token_post_experiment.py",
            subcmd,
            "--model",
            MODEL_KEY,
            "--debias_method",
            "post_inference_correction",
            "--strategy",
            STRATEGY,
            "--model_tag",
            tag6,
            "--output_dir",
            str(out6 / ds),
            "--results_dir",
            str(res6 / ds),
        ]
        expected = str(res6 / ds / (f"crows_pairs_metrics_overall_{tag6}.csv" if ds == "crowspairs" else f"stereoset_metrics_{tag6}.csv"))
        jobs.append(
            JobSpec(
                name=f"{m6}_{ds}",
                method=m6,
                dataset=ds,
                cmd=cmd,
                queue="reservation",
                time="0-10:00:00",
                mem="70G",
                cpus=10,
                expected_files=[expected],
            )
        )
    infer_names6 = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace(".jsonl", "")
        jn = f"{m6}_bbq_{s}"
        infer_names6.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m6,
                dataset=f"bbq:{s}",
                cmd=[
                    "python",
                    "6_reasoning_token_post_experiment.py",
                    "infer_bbq",
                    "--model",
                    MODEL_KEY,
                    "--debias_method",
                    "post_inference_correction",
                    "--strategy",
                    STRATEGY,
                    "--model_tag",
                    tag6,
                    "--source_file",
                    src,
                    "--output_dir",
                    str(out6 / "bbq"),
                ],
                queue="reservation",
                time="0-08:00:00",
                mem="70G",
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f"{m6}_bbq_eval",
            method=m6,
            dataset="bbq",
            cmd=[
                "python",
                "6_reasoning_token_post_experiment.py",
                "eval_bbq",
                "--model_dir",
                str(out6 / "bbq"),
                "--output_file",
                str(res6 / "bbq" / f"bbq_metrics_{tag6}.csv"),
                "--model_name",
                tag6,
            ],
            queue="reservation",
            time="0-02:00:00",
            mem="40G",
            cpus=6,
            dependency_names=infer_names6,
            expected_files=[str(res6 / "bbq" / f"bbq_metrics_{tag6}.csv")],
        )
    )

    # method11: chain of thought (native on crows/stereo/bbq)
    m11 = "m11_chain_of_thought"
    tag11 = "m11_cot_sr"
    out11 = OUTPUT_ROOT / m11
    res11 = RESULTS_ROOT / m11
    for ds, subcmd in [("crowspairs", "eval_crows"), ("stereoset", "eval_stereoset")]:
        cmd = [
            "python",
            "11_chain_of_thought.py",
            subcmd,
            "--model",
            MODEL_KEY,
            "--strategy",
            STRATEGY,
            "--model_tag",
            tag11,
            "--output_dir",
            str(out11 / ds),
            "--results_dir",
            str(res11 / ds),
        ]
        expected = str(res11 / ds / (f"crows_pairs_metrics_overall_{tag11}.csv" if ds == "crowspairs" else f"stereoset_metrics_{tag11}.csv"))
        jobs.append(
            JobSpec(
                name=f"{m11}_{ds}",
                method=m11,
                dataset=ds,
                cmd=cmd,
                queue="reservation",
                time="0-10:00:00",
                mem="70G",
                cpus=10,
                expected_files=[expected],
            )
        )
    infer_names11 = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace(".jsonl", "")
        jn = f"{m11}_bbq_{s}"
        infer_names11.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m11,
                dataset=f"bbq:{s}",
                cmd=[
                    "python",
                    "11_chain_of_thought.py",
                    "infer_bbq",
                    "--model",
                    MODEL_KEY,
                    "--strategy",
                    STRATEGY,
                    "--model_tag",
                    tag11,
                    "--source_file",
                    src,
                    "--output_dir",
                    str(out11 / "bbq"),
                ],
                queue="reservation",
                time="0-08:00:00",
                mem="70G",
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f"{m11}_bbq_eval",
            method=m11,
            dataset="bbq",
            cmd=[
                "python",
                "11_chain_of_thought.py",
                "eval_bbq",
                "--model_dir",
                str(out11 / "bbq"),
                "--output_file",
                str(res11 / "bbq" / f"bbq_metrics_{tag11}.csv"),
                "--model_name",
                tag11,
            ],
            queue="reservation",
            time="0-02:00:00",
            mem="40G",
            cpus=6,
            dependency_names=infer_names11,
            expected_files=[str(res11 / "bbq" / f"bbq_metrics_{tag11}.csv")],
        )
    )

    # method12: fine-tuned + CoT (native on crows/stereo/bbq)
    m12 = "m12_finetune_cot"
    tag12 = "m12_ftcot_sr"
    out12 = OUTPUT_ROOT / m12
    res12 = RESULTS_ROOT / m12
    for ds, subcmd in [("crowspairs", "eval_crows"), ("stereoset", "eval_stereoset")]:
        cmd = [
            "python",
            "12_finetune_cot.py",
            subcmd,
            "--model",
            MODEL_KEY,
            "--model_path",
            FT_MODEL_PATH,
            "--strategy",
            STRATEGY,
            "--model_tag",
            tag12,
            "--output_dir",
            str(out12 / ds),
            "--results_dir",
            str(res12 / ds),
        ]
        expected = str(res12 / ds / (f"crows_pairs_metrics_overall_{tag12}.csv" if ds == "crowspairs" else f"stereoset_metrics_{tag12}.csv"))
        jobs.append(
            JobSpec(
                name=f"{m12}_{ds}",
                method=m12,
                dataset=ds,
                cmd=cmd,
                queue="reservation",
                time="0-10:00:00",
                mem="70G",
                cpus=10,
                expected_files=[expected],
            )
        )
    infer_names12 = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace(".jsonl", "")
        jn = f"{m12}_bbq_{s}"
        infer_names12.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m12,
                dataset=f"bbq:{s}",
                cmd=[
                    "python",
                    "12_finetune_cot.py",
                    "infer_bbq",
                    "--model",
                    MODEL_KEY,
                    "--model_path",
                    FT_MODEL_PATH,
                    "--strategy",
                    STRATEGY,
                    "--model_tag",
                    tag12,
                    "--source_file",
                    src,
                    "--output_dir",
                    str(out12 / "bbq"),
                ],
                queue="reservation",
                time="0-08:00:00",
                mem="70G",
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f"{m12}_bbq_eval",
            method=m12,
            dataset="bbq",
            cmd=[
                "python",
                "12_finetune_cot.py",
                "eval_bbq",
                "--model_dir",
                str(out12 / "bbq"),
                "--output_file",
                str(res12 / "bbq" / f"bbq_metrics_{tag12}.csv"),
                "--model_name",
                tag12,
            ],
            queue="reservation",
            time="0-02:00:00",
            mem="40G",
            cpus=6,
            dependency_names=infer_names12,
            expected_files=[str(res12 / "bbq" / f"bbq_metrics_{tag12}.csv")],
        )
    )


def build_jobs() -> List[JobSpec]:
    jobs: List[JobSpec] = []

    # method3 finetune model (reuse existing model) on all datasets
    m3 = "m3_finetune_llama"
    tag3 = "m3_finetune_allstrat"
    add_evalshared_job(jobs, method=m3, dataset="crowspairs", model_path=FT_MODEL_PATH, model_tag=tag3, queue="offload")
    add_evalshared_job(jobs, method=m3, dataset="stereoset", model_path=FT_MODEL_PATH, model_tag=tag3, queue="offload")
    for ds in ADDL_DATASETS:
        add_evalshared_job(jobs, method=m3, dataset=ds, model_path=FT_MODEL_PATH, model_tag=f"{tag3}_{ds}", queue="offload")
    add_unqover_jobs(
        jobs,
        method=m3,
        model_path=FT_MODEL_PATH,
        model_tag_prefix=tag3,
        inference_mode="off",
        inference_strategy=None,
        queue="offload",
    )
    add_generic_bbq_jobs(
        jobs,
        method=m3,
        model_path=FT_MODEL_PATH,
        model_tag=tag3,
        inference_mode="off",
        inference_strategy=None,
        queue="offload",
    )

    # method4 inference-time instruction on all datasets
    m4 = "m4_inference_instruction"
    tag4 = "m4_instr_sr"
    add_evalshared_job(jobs, method=m4, dataset="crowspairs", model_path=None, model_tag=tag4, inference_mode="strategy", inference_strategy=STRATEGY, queue="offload")
    add_evalshared_job(jobs, method=m4, dataset="stereoset", model_path=None, model_tag=tag4, inference_mode="strategy", inference_strategy=STRATEGY, queue="offload")
    for ds in ADDL_DATASETS:
        add_evalshared_job(
            jobs,
            method=m4,
            dataset=ds,
            model_path=None,
            model_tag=f"{tag4}_{ds}",
            inference_mode="strategy",
            inference_strategy=STRATEGY,
            queue="offload",
        )
    add_unqover_jobs(
        jobs,
        method=m4,
        model_path=None,
        model_tag_prefix=tag4,
        inference_mode="strategy",
        inference_strategy=STRATEGY,
        queue="offload",
    )
    add_generic_bbq_jobs(
        jobs,
        method=m4,
        model_path=None,
        model_tag=tag4,
        inference_mode="strategy",
        inference_strategy=STRATEGY,
        queue="offload",
    )

    add_native_method_jobs(jobs)

    # Additional datasets via eval_shared for prompt-based methods (proxy evaluation path)
    proxy_methods = [
        ("m5_soft_prompting", None, "m5_proxy_sr"),
        ("m6_reasoning_post", None, "m6_proxy_sr"),
        ("m11_chain_of_thought", None, "m11_proxy_sr"),
        ("m12_finetune_cot", FT_MODEL_PATH, "m12_proxy_sr"),
    ]
    for method, model_path, tag in proxy_methods:
        for ds in ADDL_DATASETS:
            add_evalshared_job(
                jobs,
                method=method,
                dataset=ds,
                model_path=model_path,
                model_tag=f"{tag}_{ds}",
                inference_mode="strategy",
                inference_strategy=STRATEGY,
                queue="offload",
            )
        add_unqover_jobs(
            jobs,
            method=method,
            model_path=model_path,
            model_tag_prefix=tag,
            inference_mode="strategy",
            inference_strategy=STRATEGY,
            queue="offload",
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
    out = subprocess.check_output(submit, text=True).strip()
    return out.split(";")[0].strip()


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirs = ensure_dirs(ts)
    run_dir = dirs["run_dir"]
    log_out_dir = dirs["log_out"]
    log_err_dir = dirs["log_err"]

    jobs = build_jobs()
    name_to_id: Dict[str, str] = {}
    submit_log = []

    for job in jobs:
        dep_ids = [name_to_id[d] for d in job.dependency_names if d in name_to_id]
        try:
            jid = submit_job(job, dep_ids, log_out_dir, log_err_dir)
            job.job_id = jid
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
                }
            )
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
                    "error": exc.output,
                }
            )
            print(f"[ERROR] submit failed for {job.name}: {exc}")

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
    }
    manifest_path = run_dir / "jobs_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] manifest={manifest_path}")


if __name__ == "__main__":
    main()
