#!/usr/bin/env bash
set -euo pipefail

# Bind only within the Pod unless a secure ingress is configured.
host="${VLLM_HOST:-127.0.0.1}"
if [[ "$host" != "127.0.0.1" && -z "${VLLM_API_KEY:-}" ]]; then
  echo 'Set VLLM_API_KEY before binding to a public interface.' >&2
  exit 1
fi
args=(serve "${MODEL_ID:-avyukth/Qwen3.8-27B-AWQ-INT4}"
  --served-model-name "${SERVED_MODEL_NAME:-qwen}"
  --host "$host" --port "${VLLM_PORT:-8000}"
  --max-model-len "${MAX_MODEL_LEN:-40960}"
  --max-num-seqs "${MAX_NUM_SEQS:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --enable-prefix-caching
  --custom-histogram-buckets '{"time_to_first_token":[0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5.0,5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9,10.0,20.0,40.0,80.0,160.0,640.0,2560.0]}'
  --language-model-only)
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "$VLLM_API_KEY")
fi

metrics=false
chat=false
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
if [[ "$metrics" == false && "$chat" == false ]]; then
  exec vllm "${args[@]}"
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

if [[ "$metrics" == true ]]; then
  python3 /usr/local/lib/metrics-relay.py &
  children+=("$!")
fi
if [[ "$chat" == true ]]; then
  python3 /usr/local/lib/chat-gateway.py &
  children+=("$!")
fi
vllm "${args[@]}" &
children+=("$!")
echo "Started vLLM (metrics relay: ${metrics}; chat gateway: ${chat})."

# Exit if either service stops, so a relay failure cannot remain unnoticed.
set +e
wait -n "${children[@]}"
status=$?
set -e
echo "A service exited (status ${status}); stopping both services."
exit "$status"
