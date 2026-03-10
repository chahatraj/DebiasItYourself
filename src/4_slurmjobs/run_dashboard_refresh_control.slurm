#!/bin/bash
#SBATCH --job-name=diy_dash_refresh_ctl
#SBATCH --partition=contrib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch/craj/logs/diy/dashboard/out/diy_dash_refresh_ctl.%j.out.txt
#SBATCH --error=/scratch/craj/logs/diy/dashboard/err/diy_dash_refresh_ctl.%j.err.txt

set -euo pipefail

INTERVAL_SECS="${INTERVAL_SECS:-600}"
MAX_CYCLES="${MAX_CYCLES:-0}"

mkdir -p /scratch/craj/logs/diy/dashboard/out /scratch/craj/logs/diy/dashboard/err
cd /scratch/craj/diy

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo "[$(date '+%F %T')] cycle=${cycle} refresh start"
  python /scratch/craj/diy/src/7_visualizations/refresh_full_methods_baselines_dashboard.py
  echo "[$(date '+%F %T')] cycle=${cycle} refresh end"

  if [[ "$MAX_CYCLES" -gt 0 && "$cycle" -ge "$MAX_CYCLES" ]]; then
    echo "[$(date '+%F %T')] reached MAX_CYCLES=${MAX_CYCLES}, exiting"
    break
  fi

  sleep "$INTERVAL_SECS"
done
