#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/craj/diy"
SLURM_DIR="${ROOT}/src/4_slurmjobs"
LOG_DIR="/scratch/craj/logs/diy/new_methods"
mkdir -p "${LOG_DIR}"

SUBMIT_LOG="${LOG_DIR}/submit_new_methods.$(date +%Y%m%d_%H%M%S).log"
MONITOR_LOG="${LOG_DIR}/monitor_new_methods.$(date +%Y%m%d_%H%M%S).log"
REFILL_LOG="${LOG_DIR}/refill_new_methods.$(date +%Y%m%d_%H%M%S).log"

cd "${ROOT}"

/home/craj/nanotron-env/bin/python "${SLURM_DIR}/submit_new_methods_matrix.py" > "${SUBMIT_LOG}" 2>&1

MANIFEST=$(grep -Eo '/scratch/craj/diy/tracking/new_methods/run_[0-9_]+/jobs_manifest.json' "${SUBMIT_LOG}" | tail -n1)
if [[ -z "${MANIFEST}" ]]; then
  echo "Failed to detect manifest path from submit log: ${SUBMIT_LOG}" >&2
  exit 1
fi

nohup /home/craj/nanotron-env/bin/python -u "${SLURM_DIR}/monitor_new_methods.py" \
  --manifest "${MANIFEST}" \
  --poll-secs 300 \
  > "${MONITOR_LOG}" 2>&1 &
MONITOR_PID=$!

nohup /home/craj/nanotron-env/bin/python -u "${SLURM_DIR}/refill_new_methods_submissions.py" \
  --manifest "${MANIFEST}" \
  --target-queued 110 \
  --poll-secs 120 \
  > "${REFILL_LOG}" 2>&1 &
REFILL_PID=$!

echo "SUBMIT_LOG=${SUBMIT_LOG}"
echo "MANIFEST=${MANIFEST}"
echo "MONITOR_LOG=${MONITOR_LOG}"
echo "REFILL_LOG=${REFILL_LOG}"
echo "MONITOR_PID=${MONITOR_PID}"
echo "REFILL_PID=${REFILL_PID}"
