#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /scratch/craj/diy/tracking/new_methods/run_*/jobs_manifest.json" >&2
  exit 2
fi

MANIFEST_PATH="$1"
TARGET_QUEUED="${TARGET_QUEUED:-110}"
REFILL_POLL_SECS="${REFILL_POLL_SECS:-120}"
MONITOR_POLL_SECS="${MONITOR_POLL_SECS:-240}"

REFILL_JOB=$(sbatch --parsable \
  --export="ALL,MANIFEST_PATH=${MANIFEST_PATH},TARGET_QUEUED=${TARGET_QUEUED},POLL_SECS=${REFILL_POLL_SECS}" \
  /scratch/craj/diy/src/4_slurmjobs/run_new_methods_refill_control.slurm)

MONITOR_JOB=$(sbatch --parsable \
  --export="ALL,MANIFEST_PATH=${MANIFEST_PATH},POLL_SECS=${MONITOR_POLL_SECS}" \
  /scratch/craj/diy/src/4_slurmjobs/run_new_methods_monitor_control.slurm)

echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "REFILL_CONTROL_JOB=${REFILL_JOB}"
echo "MONITOR_CONTROL_JOB=${MONITOR_JOB}"
