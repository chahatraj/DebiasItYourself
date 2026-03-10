#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path('/scratch/craj/diy')
EXP_DIR = ROOT / 'src' / '3_experiments'
SLURM_DIR = ROOT / 'src' / '4_slurmjobs'
RESV_RUNNER = SLURM_DIR / 'run_single_baseline.slurm'
ANY_RUNNER = SLURM_DIR / 'run_single_baseline_anynode.slurm'
VENV = '/home/craj/nanotron-env/bin/activate'

OUTPUT_ROOT = ROOT / 'outputs' / 'new_outputs'
RESULTS_ROOT = ROOT / 'results' / 'new_results'
LOG_ROOT = Path('/scratch/craj/logs/diy/new_methods_multistrat')
TRACK_ROOT = ROOT / 'tracking' / 'new_methods_multistrat'

RESERVATION = 'craj_278'
PARTITION = 'contrib-gpuq'
RESV_NODE = 'gpu029'
RESV_QOS = 'cs_dept'
OFFLOAD_QOS = 'gpu'

MODEL_KEY = 'llama_8b'
SOFT_PROMPT_DIR = '/scratch/craj/diy/outputs/8_soft_prompt/sp_embed_20260228_1/models/learnable'

STRATEGIES = [
    'counter_imaging',
    'individuating',
    'perspective_taking',
    'positive_contact',
]

STRATEGY_TAG = {
    'counter_imaging': 'ci',
    'individuating': 'ind',
    'perspective_taking': 'pt',
    'positive_contact': 'pc',
}

FT_MODEL_PATHS = {
    'counter_imaging': '/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-counter-imaging-opinion-action-event-allversions',
    'individuating': '/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-individuating-opinion-action-event-allversions',
    'perspective_taking': '/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-perspective-taking-opinion-action-event-allversions',
    'positive_contact': '/scratch/craj/diy/outputs/7_finetuned_models/finetuned_ms-full-positive-contact-opinion-action-event-allversions',
}

BBQ_SOURCE_FILES = [
    'Age.jsonl',
    'Disability_status.jsonl',
    'Gender_identity.jsonl',
    'Nationality.jsonl',
    'Physical_appearance.jsonl',
    'Race_ethnicity.jsonl',
    'Race_x_gender.jsonl',
    'Race_x_SES.jsonl',
    'Religion.jsonl',
    'SES.jsonl',
    'Sexual_orientation.jsonl',
]

# Fast add-on datasets requested in addition to core 3.
QUICK_DATASETS = ['winobias', 'winogender', 'honest']


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
    notes: str = ''
    job_id: Optional[str] = None


def ensure_dirs(ts: str) -> Dict[str, Path]:
    run_dir = TRACK_ROOT / f'run_{ts}'
    log_out = LOG_ROOT / 'out'
    log_err = LOG_ROOT / 'err'
    for p in [OUTPUT_ROOT, RESULTS_ROOT, run_dir, log_out, log_err]:
        p.mkdir(parents=True, exist_ok=True)
    return {'run_dir': run_dir, 'log_out': log_out, 'log_err': log_err}


def add_evalshared_job(
    jobs: List[JobSpec],
    *,
    method: str,
    dataset: str,
    model_path: Optional[str],
    model_tag: str,
    inference_mode: str,
    inference_strategy: Optional[str],
    queue: str,
    time: str,
    mem: str = '70G',
    cpus: int = 10,
) -> None:
    out_dir = OUTPUT_ROOT / method / 'evalshared'
    res_dir = RESULTS_ROOT / method / 'evalshared'
    cmd = [
        'python',
        '7_eval_shared.py',
        '--dataset',
        dataset,
        '--model',
        MODEL_KEY,
        '--output_dir',
        str(out_dir),
        '--results_dir',
        str(res_dir),
        '--model_tag',
        model_tag,
        '--inference_instruction_mode',
        inference_mode,
    ]
    if inference_strategy:
        cmd += ['--inference_strategy', inference_strategy]
    if model_path:
        cmd += ['--model_path', model_path]

    exp_files: List[str] = []
    if dataset == 'crowspairs':
        exp_files.append(str(res_dir / f'crows_pairs_metrics_overall_{model_tag}.csv'))
    elif dataset == 'stereoset':
        exp_files.append(str(res_dir / f'stereoset_metrics_{model_tag}.csv'))
    elif dataset == 'winobias':
        exp_files.append(str(res_dir / 'winobias' / model_tag / f'winobias_metrics_overall_{model_tag}.csv'))
    elif dataset == 'winogender':
        exp_files.append(str(res_dir / 'winogender' / model_tag / f'winogender_metrics_overall_{model_tag}.csv'))
    elif dataset == 'honest':
        exp_files.append(str(res_dir / 'honest' / model_tag / f'honest_metrics_overall_{model_tag}.csv'))

    jobs.append(
        JobSpec(
            name=f'{method}_{dataset}_{model_tag}'[:120],
            method=method,
            dataset=dataset,
            cmd=cmd,
            queue=queue,
            time=time,
            mem=mem,
            cpus=cpus,
            expected_files=exp_files,
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
    queue: str,
    infer_time: str,
) -> None:
    bbq_out = OUTPUT_ROOT / method / 'bbq'
    bbq_res = RESULTS_ROOT / method / 'bbq'
    infer_names: List[str] = []

    instr_suffix = 'instr_off' if inference_mode == 'off' else f'instr_{inference_strategy}'

    for src in BBQ_SOURCE_FILES:
        s = src.replace('.jsonl', '')
        jn = f'{method}_bbq_{s}_{model_tag}'[:120]
        infer_names.append(jn)
        cmd = [
            'python',
            '13_bbq_inference_instruction.py',
            '--model',
            MODEL_KEY,
            '--source_file',
            src,
            '--output_dir',
            str(bbq_out),
            '--model_tag',
            model_tag,
            '--inference_instruction_mode',
            inference_mode,
        ]
        if inference_strategy:
            cmd += ['--inference_strategy', inference_strategy]
        if model_path:
            cmd += ['--model_path', model_path]
        expected = str(bbq_out / f'bbq_preds_{model_tag}_{instr_suffix}_{s}.csv')
        jobs.append(
            JobSpec(
                name=jn,
                method=method,
                dataset=f'bbq:{s}',
                cmd=cmd,
                queue=queue,
                time=infer_time,
                mem='70G',
                cpus=10,
                expected_files=[expected],
            )
        )

    eval_out = bbq_res / f'bbq_metrics_{model_tag}.csv'
    eval_cmd = [
        'python',
        '8_bbq_eval_shared.py',
        '--model_dir',
        str(bbq_out),
        '--output_file',
        str(eval_out),
        '--model_name',
        model_tag,
    ]
    jobs.append(
        JobSpec(
            name=f'{method}_bbq_eval_{model_tag}'[:120],
            method=method,
            dataset='bbq',
            cmd=eval_cmd,
            queue=queue,
            time='0-02:00:00',
            mem='40G',
            cpus=6,
            dependency_names=infer_names,
            expected_files=[str(eval_out)],
        )
    )


def add_native_method_jobs_for_strategy(
    jobs: List[JobSpec],
    *,
    strategy: str,
    strategy_tag: str,
    ft_model_path: str,
) -> None:
    # m5 native core
    m5 = 'm5_soft_prompting'
    tag5 = f'm5_softprompt_{strategy_tag}'
    out5 = OUTPUT_ROOT / m5
    res5 = RESULTS_ROOT / m5

    jobs.append(
        JobSpec(
            name=f'{m5}_crowspairs_{strategy_tag}',
            method=m5,
            dataset='crowspairs',
            cmd=[
                'python', '5_soft_prompting.py', 'eval_crows',
                '--soft_prompt_dir', SOFT_PROMPT_DIR,
                '--strategy', strategy,
                '--model', MODEL_KEY,
                '--output_dir', str(out5 / 'crowspairs'),
                '--results_dir', str(res5 / 'crowspairs'),
                '--model_tag', tag5,
            ],
            queue='reservation',
            time='0-08:00:00',
            mem='70G',
            cpus=10,
            expected_files=[str(res5 / 'crowspairs' / f'crows_pairs_metrics_overall_{tag5}.csv')],
        )
    )
    jobs.append(
        JobSpec(
            name=f'{m5}_stereoset_{strategy_tag}',
            method=m5,
            dataset='stereoset',
            cmd=[
                'python', '5_soft_prompting.py', 'eval_stereoset',
                '--soft_prompt_dir', SOFT_PROMPT_DIR,
                '--strategy', strategy,
                '--model', MODEL_KEY,
                '--output_dir', str(out5 / 'stereoset'),
                '--results_dir', str(res5 / 'stereoset'),
                '--model_tag', tag5,
            ],
            queue='reservation',
            time='0-10:00:00',
            mem='70G',
            cpus=10,
            expected_files=[str(res5 / 'stereoset' / f'stereoset_metrics_{tag5}.csv')],
        )
    )

    m5_infer = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace('.jsonl', '')
        jn = f'{m5}_bbq_{s}_{strategy_tag}'[:120]
        m5_infer.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m5,
                dataset=f'bbq:{s}',
                cmd=[
                    'python', '5_soft_prompting.py', 'infer_bbq',
                    '--soft_prompt_dir', SOFT_PROMPT_DIR,
                    '--strategy', strategy,
                    '--model', MODEL_KEY,
                    '--source_file', src,
                    '--output_dir', str(out5 / 'bbq'),
                    '--model_tag', tag5,
                ],
                queue='reservation',
                time='0-08:00:00',
                mem='70G',
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f'{m5}_bbq_eval_{strategy_tag}',
            method=m5,
            dataset='bbq',
            cmd=[
                'python', '8_bbq_eval_shared.py',
                '--model_dir', str(out5 / 'bbq'),
                '--output_file', str(res5 / 'bbq' / f'bbq_metrics_{tag5}.csv'),
                '--model_name', tag5,
            ],
            queue='reservation',
            time='0-02:00:00',
            mem='40G',
            cpus=6,
            dependency_names=m5_infer,
            expected_files=[str(res5 / 'bbq' / f'bbq_metrics_{tag5}.csv')],
        )
    )

    # m6 native core
    m6 = 'm6_reasoning_post'
    tag6 = f'm6_postcorr_{strategy_tag}'
    out6 = OUTPUT_ROOT / m6
    res6 = RESULTS_ROOT / m6

    for ds, subcmd in [('crowspairs', 'eval_crows'), ('stereoset', 'eval_stereoset')]:
        exp = f'crows_pairs_metrics_overall_{tag6}.csv' if ds == 'crowspairs' else f'stereoset_metrics_{tag6}.csv'
        jobs.append(
            JobSpec(
                name=f'{m6}_{ds}_{strategy_tag}',
                method=m6,
                dataset=ds,
                cmd=[
                    'python', '6_reasoning_token_post_experiment.py', subcmd,
                    '--model', MODEL_KEY,
                    '--debias_method', 'post_inference_correction',
                    '--strategy', strategy,
                    '--model_tag', tag6,
                    '--output_dir', str(out6 / ds),
                    '--results_dir', str(res6 / ds),
                ],
                queue='reservation',
                time='0-10:00:00',
                mem='70G',
                cpus=10,
                expected_files=[str(res6 / ds / exp)],
            )
        )

    m6_infer = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace('.jsonl', '')
        jn = f'{m6}_bbq_{s}_{strategy_tag}'[:120]
        m6_infer.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m6,
                dataset=f'bbq:{s}',
                cmd=[
                    'python', '6_reasoning_token_post_experiment.py', 'infer_bbq',
                    '--model', MODEL_KEY,
                    '--debias_method', 'post_inference_correction',
                    '--strategy', strategy,
                    '--model_tag', tag6,
                    '--source_file', src,
                    '--output_dir', str(out6 / 'bbq'),
                ],
                queue='reservation',
                time='0-08:00:00',
                mem='70G',
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f'{m6}_bbq_eval_{strategy_tag}',
            method=m6,
            dataset='bbq',
            cmd=[
                'python', '6_reasoning_token_post_experiment.py', 'eval_bbq',
                '--model_dir', str(out6 / 'bbq'),
                '--output_file', str(res6 / 'bbq' / f'bbq_metrics_{tag6}.csv'),
                '--model_name', tag6,
            ],
            queue='reservation',
            time='0-02:00:00',
            mem='40G',
            cpus=6,
            dependency_names=m6_infer,
            expected_files=[str(res6 / 'bbq' / f'bbq_metrics_{tag6}.csv')],
        )
    )

    # m11 native core
    m11 = 'm11_chain_of_thought'
    tag11 = f'm11_cot_{strategy_tag}'
    out11 = OUTPUT_ROOT / m11
    res11 = RESULTS_ROOT / m11

    for ds, subcmd in [('crowspairs', 'eval_crows'), ('stereoset', 'eval_stereoset')]:
        exp = f'crows_pairs_metrics_overall_{tag11}.csv' if ds == 'crowspairs' else f'stereoset_metrics_{tag11}.csv'
        jobs.append(
            JobSpec(
                name=f'{m11}_{ds}_{strategy_tag}',
                method=m11,
                dataset=ds,
                cmd=[
                    'python', '11_chain_of_thought.py', subcmd,
                    '--model', MODEL_KEY,
                    '--strategy', strategy,
                    '--model_tag', tag11,
                    '--output_dir', str(out11 / ds),
                    '--results_dir', str(res11 / ds),
                ],
                queue='reservation',
                time='0-10:00:00',
                mem='70G',
                cpus=10,
                expected_files=[str(res11 / ds / exp)],
            )
        )

    m11_infer = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace('.jsonl', '')
        jn = f'{m11}_bbq_{s}_{strategy_tag}'[:120]
        m11_infer.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m11,
                dataset=f'bbq:{s}',
                cmd=[
                    'python', '11_chain_of_thought.py', 'infer_bbq',
                    '--model', MODEL_KEY,
                    '--strategy', strategy,
                    '--model_tag', tag11,
                    '--source_file', src,
                    '--output_dir', str(out11 / 'bbq'),
                ],
                queue='reservation',
                time='0-08:00:00',
                mem='70G',
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f'{m11}_bbq_eval_{strategy_tag}',
            method=m11,
            dataset='bbq',
            cmd=[
                'python', '11_chain_of_thought.py', 'eval_bbq',
                '--model_dir', str(out11 / 'bbq'),
                '--output_file', str(res11 / 'bbq' / f'bbq_metrics_{tag11}.csv'),
                '--model_name', tag11,
            ],
            queue='reservation',
            time='0-02:00:00',
            mem='40G',
            cpus=6,
            dependency_names=m11_infer,
            expected_files=[str(res11 / 'bbq' / f'bbq_metrics_{tag11}.csv')],
        )
    )

    # m12 native core (ft model per strategy)
    m12 = 'm12_finetune_cot'
    tag12 = f'm12_ftcot_{strategy_tag}'
    out12 = OUTPUT_ROOT / m12
    res12 = RESULTS_ROOT / m12

    for ds, subcmd in [('crowspairs', 'eval_crows'), ('stereoset', 'eval_stereoset')]:
        exp = f'crows_pairs_metrics_overall_{tag12}.csv' if ds == 'crowspairs' else f'stereoset_metrics_{tag12}.csv'
        jobs.append(
            JobSpec(
                name=f'{m12}_{ds}_{strategy_tag}',
                method=m12,
                dataset=ds,
                cmd=[
                    'python', '12_finetune_cot.py', subcmd,
                    '--model', MODEL_KEY,
                    '--model_path', ft_model_path,
                    '--strategy', strategy,
                    '--model_tag', tag12,
                    '--output_dir', str(out12 / ds),
                    '--results_dir', str(res12 / ds),
                ],
                queue='reservation',
                time='0-10:00:00',
                mem='70G',
                cpus=10,
                expected_files=[str(res12 / ds / exp)],
            )
        )

    m12_infer = []
    for src in BBQ_SOURCE_FILES:
        s = src.replace('.jsonl', '')
        jn = f'{m12}_bbq_{s}_{strategy_tag}'[:120]
        m12_infer.append(jn)
        jobs.append(
            JobSpec(
                name=jn,
                method=m12,
                dataset=f'bbq:{s}',
                cmd=[
                    'python', '12_finetune_cot.py', 'infer_bbq',
                    '--model', MODEL_KEY,
                    '--model_path', ft_model_path,
                    '--strategy', strategy,
                    '--model_tag', tag12,
                    '--source_file', src,
                    '--output_dir', str(out12 / 'bbq'),
                ],
                queue='reservation',
                time='0-08:00:00',
                mem='70G',
                cpus=10,
            )
        )
    jobs.append(
        JobSpec(
            name=f'{m12}_bbq_eval_{strategy_tag}',
            method=m12,
            dataset='bbq',
            cmd=[
                'python', '12_finetune_cot.py', 'eval_bbq',
                '--model_dir', str(out12 / 'bbq'),
                '--output_file', str(res12 / 'bbq' / f'bbq_metrics_{tag12}.csv'),
                '--model_name', tag12,
            ],
            queue='reservation',
            time='0-02:00:00',
            mem='40G',
            cpus=6,
            dependency_names=m12_infer,
            expected_files=[str(res12 / 'bbq' / f'bbq_metrics_{tag12}.csv')],
        )
    )


def build_jobs() -> List[JobSpec]:
    jobs: List[JobSpec] = []

    # Phase 1: core datasets first (crowspairs, stereoset, bbq)
    for strategy in STRATEGIES:
        strategy_tag = STRATEGY_TAG[strategy]
        ft_model = FT_MODEL_PATHS[strategy]

        # m3 (FT model eval path) core
        tag3 = f'm3_finetune_{strategy_tag}'
        add_evalshared_job(
            jobs,
            method='m3_finetune_llama',
            dataset='crowspairs',
            model_path=ft_model,
            model_tag=tag3,
            inference_mode='off',
            inference_strategy=None,
            queue='offload',
            time='0-10:00:00',
        )
        add_evalshared_job(
            jobs,
            method='m3_finetune_llama',
            dataset='stereoset',
            model_path=ft_model,
            model_tag=tag3,
            inference_mode='off',
            inference_strategy=None,
            queue='offload',
            time='0-10:00:00',
        )
        add_generic_bbq_jobs(
            jobs,
            method='m3_finetune_llama',
            model_path=ft_model,
            model_tag=tag3,
            inference_mode='off',
            inference_strategy=None,
            queue='offload',
            infer_time='0-08:00:00',
        )

        # m4 (inference instruction) core
        tag4 = f'm4_instr_{strategy_tag}'
        add_evalshared_job(
            jobs,
            method='m4_inference_instruction',
            dataset='crowspairs',
            model_path=None,
            model_tag=tag4,
            inference_mode='strategy',
            inference_strategy=strategy,
            queue='offload',
            time='0-10:00:00',
        )
        add_evalshared_job(
            jobs,
            method='m4_inference_instruction',
            dataset='stereoset',
            model_path=None,
            model_tag=tag4,
            inference_mode='strategy',
            inference_strategy=strategy,
            queue='offload',
            time='0-10:00:00',
        )
        add_generic_bbq_jobs(
            jobs,
            method='m4_inference_instruction',
            model_path=None,
            model_tag=tag4,
            inference_mode='strategy',
            inference_strategy=strategy,
            queue='offload',
            infer_time='0-08:00:00',
        )

        # m5/m6/m11/m12 native core (reservation)
        add_native_method_jobs_for_strategy(
            jobs,
            strategy=strategy,
            strategy_tag=strategy_tag,
            ft_model_path=ft_model,
        )

    # Phase 2: quick add-on datasets (aim for <=2h)
    for strategy in STRATEGIES:
        strategy_tag = STRATEGY_TAG[strategy]
        ft_model = FT_MODEL_PATHS[strategy]

        # m3 and m4 direct evalshared
        for ds in QUICK_DATASETS:
            add_evalshared_job(
                jobs,
                method='m3_finetune_llama',
                dataset=ds,
                model_path=ft_model,
                model_tag=f'm3_finetune_{strategy_tag}_{ds}',
                inference_mode='off',
                inference_strategy=None,
                queue='offload',
                time='0-02:00:00',
            )
            add_evalshared_job(
                jobs,
                method='m4_inference_instruction',
                dataset=ds,
                model_path=None,
                model_tag=f'm4_instr_{strategy_tag}_{ds}',
                inference_mode='strategy',
                inference_strategy=strategy,
                queue='offload',
                time='0-02:00:00',
            )

        # proxy evalshared path for m5/m6/m11/m12 on quick datasets
        proxy_methods = [
            ('m5_soft_prompting', None, f'm5_proxy_{strategy_tag}'),
            ('m6_reasoning_post', None, f'm6_proxy_{strategy_tag}'),
            ('m11_chain_of_thought', None, f'm11_proxy_{strategy_tag}'),
            ('m12_finetune_cot', ft_model, f'm12_proxy_{strategy_tag}'),
        ]
        for method, model_path, tag in proxy_methods:
            for ds in QUICK_DATASETS:
                add_evalshared_job(
                    jobs,
                    method=method,
                    dataset=ds,
                    model_path=model_path,
                    model_tag=f'{tag}_{ds}',
                    inference_mode='strategy',
                    inference_strategy=strategy,
                    queue='offload',
                    time='0-02:00:00',
                )

    return jobs


def submit_job(job: JobSpec, dep_job_ids: List[str], log_out_dir: Path, log_err_dir: Path) -> str:
    runner = str(RESV_RUNNER if job.queue == 'reservation' else ANY_RUNNER)
    submit = [
        'sbatch',
        '--parsable',
        '--job-name',
        job.name[:120],
        '--partition',
        PARTITION,
        '--nodes',
        '1',
        '--ntasks',
        '1',
        '--gres',
        'gpu:A100.80gb:1',
        '--cpus-per-task',
        str(job.cpus),
        '--mem',
        job.mem,
        '--time',
        job.time,
        '--output',
        str(log_out_dir / f'{job.name}.%j.out.txt'),
        '--error',
        str(log_err_dir / f'{job.name}.%j.err.txt'),
        '--export',
        f'ALL,WORKDIR={job.workdir},VENV_PATH={VENV}',
    ]
    if job.queue == 'reservation':
        submit += ['--reservation', RESERVATION, '--qos', RESV_QOS, '--nodelist', RESV_NODE]
    else:
        submit += ['--qos', OFFLOAD_QOS]
    if dep_job_ids:
        submit += ['--dependency', 'afterok:' + ':'.join(dep_job_ids), '--kill-on-invalid-dep=yes']

    submit += [runner] + job.cmd
    res = subprocess.run(submit, text=True, capture_output=True)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, submit, output=res.stdout, stderr=res.stderr)
    out = (res.stdout or '').strip()
    return out.split(';')[0].strip()


def main() -> None:
    # preflight FT path checks
    missing = [p for p in FT_MODEL_PATHS.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f'Missing FT model paths: {missing}')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirs = ensure_dirs(ts)
    run_dir = dirs['run_dir']
    log_out_dir = dirs['log_out']
    log_err_dir = dirs['log_err']

    jobs = build_jobs()
    print(f'[INFO] Prepared jobs={len(jobs)}')

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
                    'job_name': job.name,
                    'job_id': jid,
                    'method': job.method,
                    'dataset': job.dataset,
                    'queue': job.queue,
                    'deps': dep_ids,
                    'cmd': ' '.join(shlex.quote(x) for x in job.cmd),
                    'expected_files': job.expected_files,
                }
            )
            print(f'[SUBMIT] {jid} {job.name} queue={job.queue} deps={dep_ids}')
        except subprocess.CalledProcessError as exc:
            submit_log.append(
                {
                    'job_name': job.name,
                    'job_id': None,
                    'method': job.method,
                    'dataset': job.dataset,
                    'queue': job.queue,
                    'deps': dep_ids,
                    'cmd': ' '.join(shlex.quote(x) for x in job.cmd),
                    'expected_files': job.expected_files,
                    'error_stdout': exc.output,
                    'error_stderr': exc.stderr,
                }
            )
            print(f'[ERROR] submit failed for {job.name}: rc={exc.returncode} stderr={(exc.stderr or "").strip()}')

    manifest = {
        'timestamp': ts,
        'strategies': STRATEGIES,
        'quick_datasets': QUICK_DATASETS,
        'root': str(ROOT),
        'output_root': str(OUTPUT_ROOT),
        'results_root': str(RESULTS_ROOT),
        'log_out_dir': str(log_out_dir),
        'log_err_dir': str(log_err_dir),
        'jobs_total': len(jobs),
        'jobs_submitted': sum(1 for j in jobs if j.job_id),
        'jobs': [asdict(j) for j in jobs],
        'submit_log': submit_log,
    }
    manifest_path = run_dir / 'jobs_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f'[INFO] manifest={manifest_path}')


if __name__ == '__main__':
    main()
