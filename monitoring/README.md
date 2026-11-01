# Local monitoring

Grafana runs at http://127.0.0.1:3000/d/inference-baseline and Prometheus at http://127.0.0.1:9090 on this PC. Both bind only to loopback. Start or stop them with `docker compose up -d` and `docker compose stop` from this directory. Docker Desktop must remain running for collection to continue.

The Runpod inference API remains on 127.0.0.1:8000 inside the Pod. The metrics relay forwards only `/metrics`, with a required bearer token, through Runpod's HTTPS proxy on port 8888. It combines vLLM metrics and GPU readings from `nvidia-smi`. Every other path returns 404 after authentication. This is a small single-user monitoring endpoint, not a general API gateway.

Local secrets are in `.env` (Grafana admin login) and `.secrets/metrics-token` (monitoring connection). Neither is committed. The matching token on the Pod must be saved at `/workspace/monitoring/metrics-token` with mode 600. Do not put it in command-line arguments or logs. Update the Prometheus target when changing Pods.

## Automatic startup (inference image 0.1.1 and later)

The relay source is `../scripts/metrics-relay.py`, included in the image at `/usr/local/lib/metrics-relay.py`. The Pod uses `METRICS_ENABLED=true` and `METRICS_HOST=0.0.0.0`, with the existing HTTP port 8888. Its image entrypoint launches both vLLM and the relay on every Pod/container start. The token remains at `/workspace/monitoring/metrics-token` on the network volume; no credential is embedded in the image. Shutdown stops both services, and failure of either service ends the launcher so a broken relay cannot silently leave the container appearing healthy.

The live image is now `0.2.0`, which also starts the separate chat gateway when `CHAT_ENABLED=true`. Chat uses port 8001 and `/workspace/chat/chat-token`; monitoring retains its existing port and token.

During model loading, the relay returns 503 for authenticated scrapes until vLLM becomes ready. Prometheus reconnects on its next successful scrape. A replacement Pod must retain the token and volume, and its changed hostname must be configured in Prometheus.

## Legacy manual startup (inference image 0.1.0)

The older image does not include the relay. Only for that image, use the persistent copy at `/workspace/monitoring/metrics-relay.py` and start it manually after a restart:

```bash
METRICS_HOST=0.0.0.0 METRICS_PORT=8888 nohup python3 /workspace/monitoring/metrics-relay.py >> /workspace/monitoring/metrics-relay.log 2>&1 < /dev/null &
```

Do not run that manual command alongside the automatic relay in image 0.1.1, because both would try to bind port 8888. Model weights and the token persist on the network volume. Local Prometheus and Grafana history persist in Docker volumes; local Prometheus keeps at most 15 days or 2 GB. The Railway deployment has its own persistent volumes and 7-day/512-MB block retention; see `railway/README.md`.

The dashboard uses metrics confirmed on vLLM 0.32.0. Rates need multiple scrapes, and latency percentiles need recent requests. Idle GPU utilization near zero is normal. Throughput is averaged over the displayed rate window and should not be read as per-request generation speed. Admission rejections do not include every HTTP failure. This setup does not collect prompts or generated text.

Validation: `python scripts/test_metrics_relay.py` from the project root checks authentication, route restrictions, upstream failures, and GPU unit conversion. `docker compose exec -T prometheus promtool check config /etc/prometheus/prometheus.yml` checks the collector configuration.
