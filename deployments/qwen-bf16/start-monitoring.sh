#!/usr/bin/env bash
set -euo pipefail

# Run separately from vLLM so Jupyter can retain port 8888.
root=/workspace/qwen-bf16
mountpoint -q /workspace || { echo 'Persistent volume is not mounted.' >&2; exit 1; }
exec 9>/tmp/qwen-bf16-metrics.lock
flock -n 9 || exit 0
export METRICS_HOST=0.0.0.0
export METRICS_PORT=8889
export METRICS_TOKEN_FILE="$root/metrics-token"
exec python3 "$root/metrics-relay.py"
