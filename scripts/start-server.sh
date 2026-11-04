#!/usr/bin/env bash
set -euo pipefail

# Both deployments use the same image; the existing 4bit profile is the default.
profile="${MODEL_PROFILE:-4bit}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
case "$profile" in
  4bit)
    export MODEL_ID="${MODEL_ID:-avyukth/Qwen3.8-27B-AWQ-INT4}"
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
    ;;
  bf16)
    export BF16_ROOT="${BF16_ROOT:-/workspace/qwen-bf16}"
    mountpoint -q /workspace || { echo '/workspace must be the attached persistent volume.' >&2; exit 1; }
    export MODEL_ID="$BF16_ROOT/model"
    export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-bf16}"
    export HF_HOME=/opt/qwen-bf16/hf-cache
    export HF_XET_CACHE=/opt/qwen-bf16/xet-cache
    export HF_HUB_DISABLE_XET=1
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    export VLLM_CACHE_ROOT=/opt/qwen-bf16/vllm-cache
    exec 9>/tmp/qwen-bf16.lock
    flock -n 9 || { echo 'The BF16 deployment is already running.' >&2; exit 1; }
    ;;
  *) echo 'MODEL_PROFILE must be 4bit or bf16.' >&2; exit 1 ;;
esac

# The gateway targets SERVED_MODEL_NAME. With LoRA enabled that alias belongs to
# the adapter, and the base must have a separate name in vLLM's model registry.
base_model_name="${VLLM_BASE_MODEL_NAME:-$SERVED_MODEL_NAME}"
lora_module=""
if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
  [[ "$profile" == bf16 ]] || { echo 'LORA_ADAPTER_PATH is supported only for MODEL_PROFILE=bf16.' >&2; exit 1; }
  base_model_name="${VLLM_BASE_MODEL_NAME:-${SERVED_MODEL_NAME}-base}"
  lora_module="$(python3 /usr/local/lib/validate-lora-adapter.py \
    --path "$LORA_ADAPTER_PATH" --base-model "$MODEL_ID" \
    --base-name "$base_model_name" --name "$SERVED_MODEL_NAME" \
    --max-rank "${MAX_LORA_RANK:-8}")"
elif [[ "$base_model_name" != "$SERVED_MODEL_NAME" ]]; then
  echo 'Without LORA_ADAPTER_PATH, VLLM_BASE_MODEL_NAME must match SERVED_MODEL_NAME.' >&2
  exit 1
fi

# Bind only within the Pod unless a secure ingress is configured.
host="${VLLM_HOST:-127.0.0.1}"
if [[ "$host" != "127.0.0.1" && -z "${VLLM_API_KEY:-}" ]]; then
  echo 'Set VLLM_API_KEY before binding to a public interface.' >&2
  exit 1
fi
args=(serve "${MODEL_ID:-avyukth/Qwen3.8-27B-AWQ-INT4}"
  --served-model-name "$base_model_name"
  --host "$host" --port "${VLLM_PORT:-8000}"
  --max-model-len "${MAX_MODEL_LEN:-40960}"
  --max-num-seqs "${MAX_NUM_SEQS:-2}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --enable-prefix-caching
  --custom-histogram-buckets '{"time_to_first_token":[0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0,20.0,40.0,80.0,160.0,640.0,2560.0]}'
  )
if [[ "$profile" == bf16 ]]; then
  args+=(--dtype bfloat16 --cpu-offload-gb "${CPU_OFFLOAD_GB:-40}"
    --reasoning-parser qwen3 --enforce-eager
    --safetensors-load-strategy eager --limit-mm-per-prompt '{"image":1,"video":0}')
else
  args+=(--language-model-only)
fi
if [[ -n "$lora_module" ]]; then
  args+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-8}" --lora-modules "$lora_module")
fi

start_model() {
  if [[ "$profile" == bf16 ]]; then
    python3 /usr/local/lib/prepare-bf16-model.py
  fi
  exec vllm "${args[@]}"
}
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

metrics=false
chat=false
jupyter=false
case "${METRICS_ENABLED:-false}" in
  false|0) ;;
  true|1) metrics=true ;;
  *) echo 'METRICS_ENABLED must be true or false.' >&2; exit 1 ;;
esac
case "${CHAT_ENABLED:-false}" in
  false|0) ;;
  true|1) chat=true ;;
  *) echo 'CHAT_ENABLED must be true or false.' >&2; exit 1 ;;
esac
case "${JUPYTER_ENABLED:-false}" in
  false|0) ;;
  true|1) jupyter=true ;;
  *) echo 'JUPYTER_ENABLED must be true or false.' >&2; exit 1 ;;
esac
if [[ "$jupyter" == true && -z "${JUPYTER_PASSWORD:-${JUPYTER_TOKEN:-}}" ]]; then
  echo 'Jupyter requires its existing authentication token.' >&2
  exit 1
fi
if [[ "$metrics" == false && "$chat" == false && "$jupyter" == false ]]; then
  start_model
fi

# Keep the credential on the persistent volume, never in the image.
if [[ "$metrics" == true && ! -r "${METRICS_TOKEN_FILE:-/workspace/monitoring/metrics-token}" ]]; then
  echo 'Metrics token file is missing or unreadable; refusing to start.' >&2
  exit 1
fi
if [[ "$chat" == true && ! -r "${CHAT_TOKEN_FILE:-/workspace/chat/chat-token}" ]]; then
  echo 'Chat token file is missing or unreadable; refusing to start.' >&2
  exit 1
fi

children=()
cleanup() {
  trap '' TERM INT
  kill -TERM "${children[@]}" 2>/dev/null || true
  wait "${children[@]}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ "$jupyter" == true ]]; then
  python3 /usr/local/lib/start-jupyter.py &
  children+=("$!")
fi
if [[ "$metrics" == true ]]; then
  python3 /usr/local/lib/metrics-relay.py &
  children+=("$!")
fi
if [[ "$chat" == true ]]; then
  python3 /usr/local/lib/chat-gateway.py &
  children+=("$!")
fi
start_model &
children+=("$!")
echo "Started vLLM (profile: ${profile}; metrics: ${metrics}; chat: ${chat}; Jupyter: ${jupyter})."

# Exit if either service stops, so a relay failure cannot remain unnoticed.
set +e
wait -n "${children[@]}"
status=$?
set -e
echo "A service exited (status ${status}); stopping both services."
exit "$status"
