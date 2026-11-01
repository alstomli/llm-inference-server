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
