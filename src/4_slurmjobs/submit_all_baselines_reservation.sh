#!/bin/bash
set -euo pipefail

ROOT="/scratch/craj/diy"
BASELINES_DIR="${ROOT}/src/5_baselines"
SLURM_DIR="${ROOT}/src/4_slurmjobs"
RUNNER="${SLURM_DIR}/run_single_baseline.slurm"
LOG_DIR="${SLURM_DIR}/submissions"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/baseline_submit_${STAMP}.log"

RESERVATION="${RESERVATION:-craj_278}"
PARTITION="${PARTITION:-contrib-gpuq}"
NODELIST="${NODELIST:-gpu029}"
GPU_REQ="${GPU_REQ:-gpu:A100.80gb:1}"
QOS="${QOS:-cs_dept}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}" /scratch/craj/logs/diy/out /scratch/craj/logs/diy/err

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

join_ids_colon() {
  local joined=""
  local id
  for id in "$@"; do
    [[ -z "${id}" ]] && continue
    if [[ -z "${joined}" ]]; then
      joined="${id}"
    else
      joined="${joined}:${id}"
    fi
  done
  echo "${joined}"
}

dep_afterany_from_ids() {
  local ids
  ids="$(join_ids_colon "$@")"
  if [[ -n "${ids}" ]]; then
    echo "afterany:${ids}"
  fi
}

submit_job() {
  local name="$1"
  local dep="${2:-}"
  local time_limit="$3"
  local mem="$4"
  local cpus="$5"
  shift 5

  local dep_args=()
  local qos_args=()
  if [[ -n "${dep}" ]]; then
    dep_args=(--dependency "${dep}")
  fi
  if [[ -n "${QOS}" ]]; then
    qos_args=(--qos "${QOS}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    local fake_id="DRYRUN_${name}_$RANDOM"
    printf "[DRYRUN] %s dep='%s' time=%s mem=%s cpus=%s cmd=%q\n" \
      "${fake_id}" "${dep}" "${time_limit}" "${mem}" "${cpus}" "$*" | tee -a "${LOG_FILE}" >&2
    echo "${fake_id}"
    return 0
  fi

  local job_id
  if ! job_id=$(
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
      --output "/scratch/craj/logs/diy/out/${name}.%j.out.txt" \
      --error "/scratch/craj/logs/diy/err/${name}.%j.err.txt" \
      --kill-on-invalid-dep=yes \
      --export "ALL,WORKDIR=${BASELINES_DIR},VENV_PATH=/home/craj/nanotron-env/bin/activate" \
      "${dep_args[@]}" \
      "${qos_args[@]}" \
      "${RUNNER}" \
      "$@"
  ); then
    echo "[ERROR] sbatch failed for ${name}" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
  if [[ -z "${job_id}" ]]; then
    echo "[ERROR] Empty job id for ${name}" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi

  printf "[SUBMIT] %s %s dep='%s' time=%s mem=%s cpus=%s cmd=%q\n" \
    "${job_id}" "${name}" "${dep}" "${time_limit}" "${mem}" "${cpus}" "$*" | tee -a "${LOG_FILE}" >&2
  echo "${job_id}"
}

submit_train_eval_pair() {
  local key="$1"        # short method key
  local dataset="$2"
  local tier_dep="$3"
  local train_time="$4"
  local eval_time="$5"
  local train_mem="$6"
  local eval_mem="$7"
  local train_cpus="$8"
  local eval_cpus="$9"
  local train_script="${10}"
  local eval_script="${11}"
  local extra_eval_a="${12:-}"
  local extra_eval_b="${13:-}"

  local train_name="bl_${dataset}_${key}_train"
  local eval_name="bl_${dataset}_${key}_eval"

  local train_id eval_dep eval_id
  train_id="$(submit_job "${train_name}" "${tier_dep}" "${train_time}" "${train_mem}" "${train_cpus}" python "${train_script}" --dataset "${dataset}")"
  eval_dep="afterok:${train_id}"

  if [[ -n "${extra_eval_a}" && -n "${extra_eval_b}" ]]; then
    eval_id="$(submit_job "${eval_name}" "${eval_dep}" "${eval_time}" "${eval_mem}" "${eval_cpus}" python "${eval_script}" --dataset "${dataset}" "${extra_eval_a}" "${extra_eval_b}")"
  else
    eval_id="$(submit_job "${eval_name}" "${eval_dep}" "${eval_time}" "${eval_mem}" "${eval_cpus}" python "${eval_script}" --dataset "${dataset}")"
  fi
  echo "${eval_id}"
}

echo "Submission log: ${LOG_FILE}"
echo "Reservation=${RESERVATION} Partition=${PARTITION} Node=${NODELIST} GPU=${GPU_REQ} DRY_RUN=${DRY_RUN}" | tee -a "${LOG_FILE}"
echo "Start: $(date -Is)" | tee -a "${LOG_FILE}"

prev_dataset_dep=""

for dataset in crowspairs stereoset bbq; do
  echo "---- DATASET ${dataset} ----" | tee -a "${LOG_FILE}"

  dataset_jobs=()
  base_dep="${prev_dataset_dep}"

  # Submit in short->medium->long order; scheduler fills up to 4 GPUs.
  dataset_jobs+=("$(submit_job "bl_${dataset}_biasfreebench_eval" "${base_dep}" "0-03:30:00" "50G" "8" python 7_biasfreebench_evaluate.py --dataset "${dataset}")")
  dataset_jobs+=("$(submit_job "bl_${dataset}_reduce_social_bias_eval" "${base_dep}" "0-03:30:00" "50G" "8" python 13_reduce_social_bias_evaluate.py --dataset "${dataset}")")

  if [[ "${dataset}" == "bbq" ]]; then
    dataset_jobs+=("$(submit_job "bl_bbq_decap_eval" "${base_dep}" "0-06:00:00" "60G" "10" python 1_decap_evaluate.py)")
    dataset_jobs+=("$(submit_job "bl_bbq_selfdebias_eval" "${base_dep}" "0-06:00:00" "60G" "10" python 15_selfdebias_evaluate.py)")
  fi

  if [[ "${dataset}" == "bbq" ]]; then
    dataset_jobs+=("$(submit_train_eval_pair "fairsteer" "${dataset}" "${base_dep}" "0-08:00:00" "0-04:00:00" "60G" "60G" "12" "10" 4_fairsteer_train.py 4_fairsteer_evaluate.py --components_dir "/scratch/craj/diy/outputs/3_baselines/fairsteer/models")")
  else
    dataset_jobs+=("$(submit_train_eval_pair "fairsteer" "${dataset}" "${base_dep}" "0-08:00:00" "0-04:00:00" "60G" "60G" "12" "10" 4_fairsteer_train.py 4_fairsteer_evaluate.py --components_dir "/scratch/craj/diy/outputs/3_baselines/fairsteer/models_${dataset}/model_${dataset}_all")")
  fi
  dataset_jobs+=("$(submit_train_eval_pair "bba" "${dataset}" "${base_dep}" "0-06:00:00" "0-03:30:00" "60G" "50G" "10" "8" 8_bba_train.py 8_bba_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "cal" "${dataset}" "${base_dep}" "0-06:00:00" "0-03:30:00" "60G" "50G" "10" "8" 9_cal_train.py 9_cal_evaluate.py)")

  dataset_jobs+=("$(submit_train_eval_pair "dpo" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 3_dpo_train.py 3_dpo_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "lftf" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 5_lftf_train.py 5_lftf_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "peft" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 6_peft_train.py 6_peft_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "debias_nlg" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 10_debias_nlg_train.py 10_debias_nlg_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "mbias" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 11_mbias_train.py 11_mbias_evaluate.py)")
  dataset_jobs+=("$(submit_train_eval_pair "debias_llms" "${dataset}" "${base_dep}" "0-12:00:00" "0-05:00:00" "70G" "60G" "12" "10" 12_debias_llms_train.py 12_debias_llms_evaluate.py)")

  if [[ "${dataset}" == "crowspairs" || "${dataset}" == "stereoset" ]]; then
    bt_train_id="$(submit_job "bl_${dataset}_biasedit_train" "${base_dep}" "0-20:00:00" "80G" "12" python 2_biasedit_train.py --dataset "${dataset}")"
    bt_eval_id="$(submit_job "bl_${dataset}_biasedit_eval" "afterok:${bt_train_id}" "0-06:00:00" "60G" "10" python 2_biasedit_evaluate.py --dataset "${dataset}")"
    dataset_jobs+=("${bt_eval_id}")
  fi

  if [[ "${dataset}" == "bbq" ]]; then
    regift_train_id="$(submit_job "bl_bbq_regift_train" "${base_dep}" "1-00:00:00" "120G" "16" python 14_regift_train.py)"
    regift_eval_id="$(submit_job "bl_bbq_regift_eval" "afterok:${regift_train_id}" "0-10:00:00" "80G" "12" python 14_regift_evaluate.py --model_path /scratch/craj/diy/outputs/3_baselines/regift/models)"
    dataset_jobs+=("${regift_eval_id}")
  fi

  dataset_dep="$(dep_afterany_from_ids "${dataset_jobs[@]}")"
  if [[ -z "${dataset_dep}" ]]; then
    dataset_dep="${base_dep}"
  fi
  prev_dataset_dep="${dataset_dep}"
  echo "Dataset ${dataset} final dependency: ${prev_dataset_dep}" | tee -a "${LOG_FILE}"
done

echo "End: $(date -Is)" | tee -a "${LOG_FILE}"
echo "Submitted all jobs. Track with: squeue -u craj -o '%.18i %.15P %.30j %.10T %.12M %.6D %R'" | tee -a "${LOG_FILE}"
echo "Submission log written to: ${LOG_FILE}"
