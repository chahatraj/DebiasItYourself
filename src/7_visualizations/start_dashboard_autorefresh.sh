#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/craj/diy"
LOG_DIR="$ROOT/tracking/baseline_methods"
LOG_FILE="$LOG_DIR/dashboard_autorefresh.log"
PID_FILE="$LOG_DIR/dashboard_autorefresh.pid"
INTERVAL_SECONDS="${1:-600}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "dashboard autorefresh already running (pid=$OLD_PID)"
    exit 0
  fi
fi

nohup bash -lc "
  cd '$ROOT'
  while true; do
    echo \"[\$(date '+%F %T')] refresh start\"
    python src/7_visualizations/refresh_full_methods_baselines_dashboard.py
    echo \"[\$(date '+%F %T')] refresh done\"
    sleep '$INTERVAL_SECONDS'
  done
" >>"$LOG_FILE" 2>&1 &

NEW_PID="$!"
echo "$NEW_PID" >"$PID_FILE"
echo "dashboard autorefresh started (pid=$NEW_PID, interval=${INTERVAL_SECONDS}s)"
