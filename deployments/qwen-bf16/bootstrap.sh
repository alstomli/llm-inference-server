#!/usr/bin/env bash
set -euo pipefail

# The global volume is object-backed. Environments, locks and caches stay local.
root=/workspace/qwen-bf16
runtime=/opt/qwen-bf16
if ! mountpoint -q /workspace; then
  echo '/workspace must be the attached persistent volume.' >&2
  exit 1
fi
mkdir -p "$runtime" "$root"
exec 9>"/tmp/qwen-bf16.lock"
if ! flock -n 9; then
  echo 'The BF16 deployment is already running.'
  exit 0
fi

export HF_HOME="$runtime/hf-cache"
export HF_XET_CACHE="$runtime/xet-cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export VLLM_CACHE_ROOT="$runtime/vllm-cache"
export PYTHONUNBUFFERED=1

if [[ ! -f "$runtime/environment-ready" ]]; then
  python3 -m venv "$runtime/venv"
  "$runtime/venv/bin/python" -m pip install --no-cache-dir \
    'vllm==0.32.0' 'transformers>=5.8.0,<6'
  touch "$runtime/environment-ready"
fi

"$runtime/venv/bin/python" "$root/prepare_model.py"
exec "$runtime/venv/bin/python" "$root/serve.py"
