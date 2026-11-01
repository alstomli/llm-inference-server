#!/bin/sh
set -eu
: "${RUNPOD_METRICS_TOKEN:?Set the private metrics token in Railway variables}"
: "${RUNPOD_METRICS_TARGET:?Set the Runpod metrics hostname without a scheme or path}"
case "$RUNPOD_METRICS_TARGET" in
  *[!a-zA-Z0-9.-]*) echo 'Invalid metrics hostname' >&2; exit 1 ;;
esac
umask 077
printf '%s' "$RUNPOD_METRICS_TOKEN" > /tmp/metrics-token
cat > /tmp/prometheus.yml <<EOF
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: vllm
    scheme: https
    scrape_timeout: 12s
    authorization:
      type: Bearer
      credentials_file: /tmp/metrics-token
    static_configs:
      - targets: ["$RUNPOD_METRICS_TARGET"]
EOF
exec /bin/prometheus \
  --config.file=/tmp/prometheus.yml \
  --web.listen-address="[::]:${PORT:-9090}" \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time="${METRICS_RETENTION_TIME:-7d}" \
  --storage.tsdb.retention.size="${METRICS_RETENTION_SIZE:-512MB}"
