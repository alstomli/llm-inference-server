#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${QLORA_ENV_ROOT:-/opt/qlora}"
export HF_HOME="$root/hf-cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export UNSLOTH_DISABLE_STATISTICS=1
export TORCHINDUCTOR_CACHE_DIR="$root/torch-cache"
export TRITON_CACHE_DIR="$root/triton-cache"
export PYTHONUNBUFFERED=1
"$root/venv/bin/python" "$source_dir/qlora_smoke.py" --validate-only "$@"
exec "$root/venv/bin/python" "$source_dir/qlora_smoke.py" "$@"
