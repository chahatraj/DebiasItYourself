#!/bin/bash
#SBATCH --job-name=base_rebalance_ctl
#SBATCH --partition=contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch/craj/logs/diy/baseline_methods/out/base_rebalance_ctl.%j.out.txt
#SBATCH --error=/scratch/craj/logs/diy/baseline_methods/err/base_rebalance_ctl.%j.err.txt

set -euo pipefail

GPU_MAX_SUBMIT="${GPU_MAX_SUBMIT:-40}"
POLL_SECS="${POLL_SECS:-90}"

mkdir -p /scratch/craj/logs/diy/baseline_methods/out /scratch/craj/logs/diy/baseline_methods/err

cd /scratch/craj/diy
exec /home/craj/nanotron-env/bin/python -u /scratch/craj/diy/src/4_slurmjobs/rebalance_baseline_qos.py \
  --gpu-max-submit "${GPU_MAX_SUBMIT}" \
  --poll-secs "${POLL_SECS}"
