#!/bin/bash

#SBATCH --job-name=job_array
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --reservation=craj_278
#SBATCH --nodelist=gpu029
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=0-10:00:00
#SBATCH --output=/scratch/craj/logs/diy/out/job_array.%A_%a.out.txt
#SBATCH --error=/scratch/craj/logs/diy/err/job_array.%A_%a.err.txt
#SBATCH --requeue

set -euo pipefail

VENV_PATH="${VENV_PATH:-/home/craj/nanotron-env/bin/activate}"
WORKDIR="${WORKDIR:-/scratch/craj/diy/src/3_experiments}"
ARRAY_FILE="${ARRAY_FILE:-}"

if [[ -z "${ARRAY_FILE}" ]]; then
  echo "ARRAY_FILE is required."
  exit 1
fi

if [[ ! -f "${ARRAY_FILE}" ]]; then
  echo "Array file not found: ${ARRAY_FILE}"
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

source "${VENV_PATH}"
cd "${WORKDIR}"

readarray -t CMDS < "${ARRAY_FILE}"
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if (( INDEX < 0 || INDEX >= ${#CMDS[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range for array ${ARRAY_FILE}"
  exit 1
fi

CMD="${CMDS[$INDEX]}"
if [[ -z "${CMD// }" ]]; then
  echo "Empty command at index ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "[$(date -Is)] Task ${SLURM_ARRAY_TASK_ID} running: ${CMD}"
echo "[$(date -Is)] Host: $(hostname)"
echo "[$(date -Is)] Python: $(command -v python)"
eval "${CMD}"
echo "[$(date -Is)] Task ${SLURM_ARRAY_TASK_ID} completed"
