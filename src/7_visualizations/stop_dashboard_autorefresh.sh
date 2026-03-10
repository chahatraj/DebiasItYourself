#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/craj/diy"
PID_FILE="$ROOT/tracking/baseline_methods/dashboard_autorefresh.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "dashboard autorefresh not running (no pid file)"
  exit 0
fi

PID="$(cat "$PID_FILE" || true)"
if [[ -z "$PID" ]]; then
  rm -f "$PID_FILE"
  echo "dashboard autorefresh not running (empty pid file)"
  exit 0
fi

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "dashboard autorefresh stopped (pid=$PID)"
else
  echo "dashboard autorefresh not running (stale pid=$PID)"
fi

rm -f "$PID_FILE"
