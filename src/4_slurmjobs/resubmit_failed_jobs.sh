#!/bin/bash
set -euo pipefail

ROOT="/scratch/craj/diy"
SLURM_DIR="${ROOT}/src/4_slurmjobs"
BASELINES_DIR="${ROOT}/src/5_baselines"
EXPERIMENTS_DIR="${ROOT}/src/3_experiments"
RUNNER="${SLURM_DIR}/run_single_baseline.slurm"
LOG_DIR="${SLURM_DIR}/submissions"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/retry_failed_submit_${STAMP}.log"

RESERVATION="${RESERVATION:-craj_278}"
PARTITION="${PARTITION:-contrib-gpuq}"
NODELIST="${NODELIST:-gpu029}"
GPU_REQ="${GPU_REQ:-gpu:A100.80gb:1}"
QOS="${QOS:-cs_dept}"
VENV_PATH="${VENV_PATH:-/home/craj/nanotron-env/bin/activate}"

mkdir -p "${LOG_DIR}" /scratch/craj/logs/diy/out /scratch/craj/logs/diy/err

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

submit_job() {
  local name="$1"
  local dep="$2"
  local time_limit="$3"
  local mem="$4"
  local cpus="$5"
  local workdir="$6"
  shift 6

  local dep_args=()
  local qos_args=()
  if [[ -n "${dep}" ]]; then
    dep_args=(--dependency "${dep}")
  fi
  if [[ -n "${QOS}" ]]; then
    qos_args=(--qos "${QOS}")
  fi

  local job_id
  job_id=$(
    sbatch --parsable \
      --job-name "${name}" \
      --partition "${PARTITION}" \
      --reservation "${RESERVATION}" \
      --nodelist "${NODELIST}" \
      --nodes 1 \
      --ntasks 1 \
      --gres "${GPU_REQ}" \
      --cpus-per-task "${cpus}" \
      --mem "${mem}" \
      --time "${time_limit}" \
      --kill-on-invalid-dep=yes \
      --output "/scratch/craj/logs/diy/out/${name}.%j.out.txt" \
      --error "/scratch/craj/logs/diy/err/${name}.%j.err.txt" \
      --export "ALL,WORKDIR=${workdir},VENV_PATH=${VENV_PATH}" \
      "${dep_args[@]}" \
      "${qos_args[@]}" \
      "${RUNNER}" \
      "$@"
  )

  printf "[SUBMIT] %s %s dep='%s' time=%s mem=%s cpus=%s cmd=%q\n" \
    "${job_id}" "${name}" "${dep}" "${time_limit}" "${mem}" "${cpus}" "$*" | tee -a "${LOG_FILE}" >&2
  echo "${job_id}"
}

submit_train_eval_pair() {
  local name_prefix="$1"
  local train_time="$2"
  local eval_time="$3"
  local train_mem="$4"
  local eval_mem="$5"
  local train_cpus="$6"
  local eval_cpus="$7"
  local dataset="$8"
  local train_script="$9"
  local eval_script="${10}"
  shift 10
  local extra_eval_args=("$@")

  if [[ -z "${dataset}" ]]; then
    echo "ERROR: submit_train_eval_pair called without dataset for ${name_prefix}" >&2
    return 1
  fi

  local train_id eval_id
  train_id="$(submit_job "${name_prefix}_train" "" "${train_time}" "${train_mem}" "${train_cpus}" "${BASELINES_DIR}" python "${train_script}" --dataset "${dataset}")"
  if [[ "${#extra_eval_args[@]}" -gt 0 ]]; then
    eval_id="$(submit_job "${name_prefix}_eval" "afterok:${train_id}" "${eval_time}" "${eval_mem}" "${eval_cpus}" "${BASELINES_DIR}" python "${eval_script}" --dataset "${dataset}" "${extra_eval_args[@]}")"
  else
    eval_id="$(submit_job "${name_prefix}_eval" "afterok:${train_id}" "${eval_time}" "${eval_mem}" "${eval_cpus}" "${BASELINES_DIR}" python "${eval_script}" --dataset "${dataset}")"
  fi
  echo "${eval_id}"
}

echo "Retry submission log: ${LOG_FILE}"
echo "Start: $(date -Is)" | tee -a "${LOG_FILE}"
echo "Reservation=${RESERVATION} Partition=${PARTITION} Node=${NODELIST} GPU=${GPU_REQ}" | tee -a "${LOG_FILE}"

# crowspairs retries
submit_train_eval_pair "retry_crowspairs_dpo" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  crowspairs 3_dpo_train.py 3_dpo_evaluate.py >/dev/null

submit_train_eval_pair "retry_crowspairs_peft" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  crowspairs 6_peft_train.py 6_peft_evaluate.py >/dev/null

submit_train_eval_pair "retry_crowspairs_debias_nlg" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  crowspairs 10_debias_nlg_train.py 10_debias_nlg_evaluate.py >/dev/null

submit_train_eval_pair "retry_crowspairs_mbias" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  crowspairs 11_mbias_train.py 11_mbias_evaluate.py >/dev/null

submit_train_eval_pair "retry_crowspairs_debias_llms" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  crowspairs 12_debias_llms_train.py 12_debias_llms_evaluate.py >/dev/null

submit_train_eval_pair "retry_crowspairs_biasedit" "0-20:00:00" "0-06:00:00" "80G" "60G" "12" "10" \
  crowspairs 2_biasedit_train.py 2_biasedit_evaluate.py >/dev/null

# stereoset retries
submit_train_eval_pair "retry_stereoset_dpo" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  stereoset 3_dpo_train.py 3_dpo_evaluate.py >/dev/null

submit_train_eval_pair "retry_stereoset_peft" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  stereoset 6_peft_train.py 6_peft_evaluate.py >/dev/null

submit_train_eval_pair "retry_stereoset_debias_nlg" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  stereoset 10_debias_nlg_train.py 10_debias_nlg_evaluate.py >/dev/null

submit_train_eval_pair "retry_stereoset_mbias" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  stereoset 11_mbias_train.py 11_mbias_evaluate.py >/dev/null

submit_train_eval_pair "retry_stereoset_debias_llms" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  stereoset 12_debias_llms_train.py 12_debias_llms_evaluate.py >/dev/null

submit_train_eval_pair "retry_stereoset_biasedit" "0-20:00:00" "0-06:00:00" "80G" "60G" "12" "10" \
  stereoset 2_biasedit_train.py 2_biasedit_evaluate.py >/dev/null

# bbq retries (failed methods only)
submit_train_eval_pair "retry_bbq_dpo" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  bbq 3_dpo_train.py 3_dpo_evaluate.py >/dev/null

submit_train_eval_pair "retry_bbq_peft" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  bbq 6_peft_train.py 6_peft_evaluate.py >/dev/null

submit_train_eval_pair "retry_bbq_debias_nlg" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  bbq 10_debias_nlg_train.py 10_debias_nlg_evaluate.py >/dev/null

submit_train_eval_pair "retry_bbq_mbias" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  bbq 11_mbias_train.py 11_mbias_evaluate.py >/dev/null

submit_train_eval_pair "retry_bbq_debias_llms" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" \
  bbq 12_debias_llms_train.py 12_debias_llms_evaluate.py >/dev/null

regift_train_id="$(submit_job "retry_bbq_regift_train" "" "2-00:00:00" "120G" "16" "${BASELINES_DIR}" \
  python 14_regift_train.py --batch_size 2 --max_length 768)"
submit_job "retry_bbq_regift_eval" "afterok:${regift_train_id}" "0-10:00:00" "80G" "12" "${BASELINES_DIR}" \
  python 14_regift_evaluate.py --model_path /scratch/craj/diy/outputs/3_baselines/regift/models >/dev/null

# Timeout-prone eval-only methods: split BBQ by source_file and allocate longer walltime.
for src in Age.jsonl Disability_status.jsonl Gender_identity.jsonl Nationality.jsonl Physical_appearance.jsonl Race_ethnicity.jsonl Religion.jsonl SES.jsonl Sexual_orientation.jsonl; do
  tag="${src%.jsonl}"
  submit_job "retry_bbq_decap_${tag}" "" "4-00:00:00" "60G" "10" "${BASELINES_DIR}" \
    python 1_decap_evaluate.py --source_file "${src}"
  submit_job "retry_bbq_selfdebias_${tag}" "" "0-18:00:00" "60G" "10" "${BASELINES_DIR}" \
    python 15_selfdebias_evaluate.py --source_file "${src}"
done

# Retry failed additional-benchmark array task (BOLD, task 4)
submit_job "retry_evalshared_bold" "" "0-10:00:00" "50G" "8" "${EXPERIMENTS_DIR}" \
  python 7_eval_shared.py --dataset bold --model llama_8b \
  --output_dir /scratch/craj/diy/outputs/10_additional_benchmarks/baselines/baseline_evalshared_all_20260306_1249 \
  --results_dir /scratch/craj/diy/results/10_additional_benchmarks/baselines/baseline_evalshared_all_20260306_1249 \
  --batch_size 8 --max_length 1024 --max_samples 3000

echo "End: $(date -Is)" | tee -a "${LOG_FILE}"
echo "Submitted retries. Monitor with: squeue -u craj -o '%.18i %.9P %.28j %.2t %.10M %.6D %R'" | tee -a "${LOG_FILE}"
