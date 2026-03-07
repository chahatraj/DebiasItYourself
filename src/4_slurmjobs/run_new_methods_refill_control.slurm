#!/bin/bash
#SBATCH --job-name=newmeth_refill_ctl
#SBATCH --partition=contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch/craj/logs/diy/new_methods/out/newmeth_refill_ctl.%j.out.txt
#SBATCH --error=/scratch/craj/logs/diy/new_methods/err/newmeth_refill_ctl.%j.err.txt

set -euo pipefail

MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
TARGET_QUEUED="${TARGET_QUEUED:-110}"
POLL_SECS="${POLL_SECS:-120}"

cd /scratch/craj/diy
exec /home/craj/nanotron-env/bin/python -u /scratch/craj/diy/src/4_slurmjobs/refill_new_methods_submissions.py \
  --manifest "${MANIFEST_PATH}" \
  --target-queued "${TARGET_QUEUED}" \
  --poll-secs "${POLL_SECS}"
