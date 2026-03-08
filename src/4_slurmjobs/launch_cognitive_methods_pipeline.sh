#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/craj/diy"
SLURM_DIR="${ROOT}/src/4_slurmjobs"
LOG_DIR="/scratch/craj/logs/diy/cognitive_methods"
mkdir -p "${LOG_DIR}"

SUBMIT_LOG="${LOG_DIR}/submit_cognitive_methods.$(date +%Y%m%d_%H%M%S).log"
cd "${ROOT}"

/home/craj/nanotron-env/bin/python "${SLURM_DIR}/submit_cognitive_methods_matrix.py" > "${SUBMIT_LOG}" 2>&1

MANIFEST=$(grep -Eo '/scratch/craj/diy/tracking/cognitive_methods/run_[0-9_]+/jobs_manifest.json' "${SUBMIT_LOG}" | tail -n1)
if [[ -z "${MANIFEST}" ]]; then
  echo "Failed to detect manifest from ${SUBMIT_LOG}" >&2
  exit 1
fi

CONTROL_LOG="${LOG_DIR}/control_cognitive_methods.$(date +%Y%m%d_%H%M%S).log"
"${SLURM_DIR}/submit_new_methods_control_jobs.sh" "${MANIFEST}" > "${CONTROL_LOG}" 2>&1

echo "SUBMIT_LOG=${SUBMIT_LOG}"
echo "MANIFEST=${MANIFEST}"
echo "CONTROL_LOG=${CONTROL_LOG}"
