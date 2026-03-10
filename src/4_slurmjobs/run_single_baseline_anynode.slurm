#!/bin/bash
#SBATCH --job-name=baseline_anynode
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=cs_dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/craj/logs/diy/out/baseline_anynode.%j.out.txt
#SBATCH --error=/scratch/craj/logs/diy/err/baseline_anynode.%j.err.txt

set -euo pipefail

VENV_PATH="${VENV_PATH:-/home/craj/nanotron-env/bin/activate}"
WORKDIR="${WORKDIR:-/scratch/craj/diy/src/5_baselines}"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch run_single_baseline_anynode.slurm <command> [args ...]" >&2
  exit 2
fi

mkdir -p /scratch/craj/logs/diy/out /scratch/craj/logs/diy/err

source "${VENV_PATH}"
cd "${WORKDIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date -Is)] Host: $(hostname)"
echo "[$(date -Is)] CWD: $(pwd)"
echo "[$(date -Is)] Python: $(command -v python)"
echo "[$(date -Is)] Running: $*"

"$@"

echo "[$(date -Is)] Completed."
