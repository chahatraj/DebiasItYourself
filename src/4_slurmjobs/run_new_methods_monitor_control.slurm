#!/bin/bash
#SBATCH --job-name=newmeth_monitor_ctl
#SBATCH --partition=contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch/craj/logs/diy/new_methods/out/newmeth_monitor_ctl.%j.out.txt
#SBATCH --error=/scratch/craj/logs/diy/new_methods/err/newmeth_monitor_ctl.%j.err.txt

set -euo pipefail

MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
POLL_SECS="${POLL_SECS:-240}"

cd /scratch/craj/diy
exec /home/craj/nanotron-env/bin/python -u /scratch/craj/diy/src/4_slurmjobs/monitor_new_methods.py \
  --manifest "${MANIFEST_PATH}" \
  --poll-secs "${POLL_SECS}"
