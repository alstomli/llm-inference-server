#!/usr/bin/env bash
set -euo pipefail

# BF16 is the sole model profile; pass its default to every child service.
profile="${MODEL_PROFILE-bf16}"
[[ "$profile" == bf16 ]] || { echo 'MODEL_PROFILE must be bf16.' >&2; exit 1; }
export MODEL_PROFILE="$profile"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
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

# The gateway targets SERVED_MODEL_NAME. With LoRA enabled that alias belongs to
# the adapter, and the base must have a separate name in vLLM's model registry.
base_model_name="${VLLM_BASE_MODEL_NAME:-$SERVED_MODEL_NAME}"
lora_module=""
if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
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
args=(serve "$MODEL_ID"
  --served-model-name "$base_model_name"
  --host "$host" --port "${VLLM_PORT:-8000}"
  --max-model-len "${MAX_MODEL_LEN:-40960}"
  --max-num-seqs "${MAX_NUM_SEQS:-2}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --enable-prefix-caching
  --custom-histogram-buckets '{"time_to_first_token":[0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0,20.0,40.0,80.0,160.0,640.0,2560.0]}'
  )
args+=(--dtype bfloat16 --cpu-offload-gb "${CPU_OFFLOAD_GB:-40}"
  --reasoning-parser qwen3
  --safetensors-load-strategy eager --limit-mm-per-prompt '{"image":1,"video":0}')
# Both features are opt-in. Weight offload is independent of the native KV tier.
case "${KV_OFFLOAD_CPU_GIB:-0}" in
  0) ;;
  64)
    python3 /usr/local/lib/kv-runtime-config.py check-memory
    args+=(--mamba-cache-mode align --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":68719476736,"blocks_per_chunk":1,"offload_prompt_only":false}}')
    ;;
  *) echo 'KV_OFFLOAD_CPU_GIB must be 0 or 64.' >&2; exit 1 ;;
esac
case "${CUDA_DECODE_GRAPHS:-false}" in
  false|0) args+=(--enforce-eager) ;;
  true|1)
    [[ "${CPU_OFFLOAD_GB:-40}" == 0 ]] || {
      echo 'Decode graphs require zero weight offload.' >&2; exit 1;
    }
    case "$MAX_NUM_SEQS" in
      2) capture_sizes='[1,2]' ;;
      8) capture_sizes='[1,2,4,8]' ;;
      *) echo 'Decode graphs require the supported two- or eight-sequence profile.' >&2; exit 1 ;;
    esac
    args+=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$capture_sizes}" --cudagraph-metrics)
    ;;
  *) echo 'CUDA_DECODE_GRAPHS must be true or false.' >&2; exit 1 ;;
esac
if [[ "${KV_RUNTIME_OBSERVE:-false}" == true ]]; then
  args+=(--worker-cls kv_runtime.Worker)
fi
if [[ "${KV_RUNTIME_PROOF:-false}" == true ]]; then
  [[ "$host" == 127.0.0.1 && "${KV_RUNTIME_OBSERVE:-false}" == true ]] || {
    echo 'KV proof requires the local observed engine.' >&2; exit 1;
  }
  args+=(--middleware kv_runtime.ProofMiddleware)
  python3 /usr/local/lib/kv-runtime-config.py prepare-proof
fi
if [[ -n "$lora_module" ]]; then
  args+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-8}" --lora-modules "$lora_module")
fi

# An explicitly selected parser can be staged while gateway web search remains
# disabled, allowing private model/API verification before advertising readiness.
# No parser is guessed from model names and default serving flags stay unchanged.
if [[ -n "${CHAT_WEB_SEARCH_TOOL_PARSER:-}" ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser "$CHAT_WEB_SEARCH_TOOL_PARSER")
fi
if [[ "${CHAT_WEB_SEARCH_ENABLED:-false}" == true ]]; then
  [[ "${CHAT_WEB_SEARCH_MODEL_VERIFIED:-false}" == true && -n "${CHAT_WEB_SEARCH_TOOL_PARSER:-}" && -n "${CHAT_WEB_SEARCH_VERIFIED_MODEL:-}" ]] || {
    echo 'Web search requires explicit model/parser compatibility verification.' >&2
    exit 1
  }
  [[ "${CHAT_GATEWAY_IMPLEMENTATION:-go}" == go ]] || {
    echo 'Web search requires the Go chat gateway.' >&2
    exit 1
  }
fi

start_model() {
  python3 /usr/local/lib/prepare-bf16-model.py
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
if [[ "$chat" == true ]]; then
  case "${CHAT_GATEWAY_IMPLEMENTATION:-go}" in
    go|python) ;;
    *) echo 'CHAT_GATEWAY_IMPLEMENTATION must be go or python.' >&2; exit 1 ;;
  esac
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
  if [[ "${CHAT_GATEWAY_IMPLEMENTATION:-go}" == python ]]; then
    python3 /usr/local/lib/chat-gateway.py &
  else
    /usr/local/bin/chat-gateway &
  fi
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
