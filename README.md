# Qwen inference server on Runpod

This project packages the vLLM configuration tested on one RTX PRO 4500. The Runpod Pod serves Qwen and runs a token-protected metrics relay. Prometheus and Grafana run on Railway; an optional local Docker stack remains available. See the separate [Grafana](https://github.com/alstomli/llm-grafana) and [Prometheus](https://github.com/alstomli/llm-prometheus) repositories. The SSH tunnel below describes an alternative that requires an SSH server.

## What runs where

- Runpod Pod: vLLM, model API and internal `/metrics` on `127.0.0.1:8000`; authenticated metrics relay on port 8888 and authenticated chat gateway on port 8001.
- Runpod network volume: Hugging Face model cache and vLLM compilation cache under `/workspace`.
- Railway: Prometheus and Grafana, with persistent monitoring history independent of your computer.
- Your computer: optional Prometheus and Grafana via Docker Compose.

## Prepare the inference image

The live Pod `fj04zaugwd1u83` now uses `0.2.0`, with automatic monitoring and chat gateway startup. The Docker Hub release digest is `sha256:5389868284fadfd95d8b58499607099580f957756810cbd7684a843f06dc69cd`. Keep published release tags unchanged; use a new version for later code changes.

Build an **x86-64 Linux** image on a machine with Docker and registry access. The Docker Hub namespace for this project is `zihan000`:

```bash
docker buildx build --platform linux/amd64 -t zihan000/qwen-inference:0.2.0 --push .
```

Then create a Runpod custom template or update the existing Pod with that image, one RTX PRO 4500 and the existing network volume mounted at `/workspace`. Do not override the image entrypoint. The image starts vLLM automatically and defaults to the tested model and 4096-token, single-sequence configuration. The official base is pinned to vLLM 0.32.0.

For automatic metrics collection, set `METRICS_ENABLED=true` and `METRICS_HOST=0.0.0.0` in the Pod environment, expose HTTP port 8888, and retain the existing token file `/workspace/monitoring/metrics-token` on the network volume. The token must contain at least 32 characters. It is not embedded in the image. The relay script is included at `/usr/local/lib/metrics-relay.py`; the older copy on `/workspace` is no longer the startup source. The launcher starts the enabled services on every container start, forwards shutdown, and exits if any service exits. With both `METRICS_ENABLED=false` and `CHAT_ENABLED=false` (the defaults), startup runs vLLM only.

Test the launcher without a GPU or model download:

```bash
docker run --rm --entrypoint python3 --mount type=bind,source="$PWD/scripts",target=/tests,readonly zihan000/qwen-inference:0.2.0 /tests/test_start_server.py
```

Test a replacement Pod before retiring the existing Pod. A network volume is tied to its data center and must be selected at deployment. Current Runpod Pod pricing and GPU availability apply. If a container exits or a machine fails, recovery depends on Runpod's behavior and Pod availability; this image alone does not guarantee high availability or restart-on-failure.

## Chat gateway

See [the chatbot project](https://github.com/alstomli/qwen-chat) for the frontend and later deployment instructions. `scripts/chat-gateway.py` exposes only authenticated readiness and streaming chat routes, counts the actual vLLM chat-template tokens before generation, trims old exchanges to fit the context, and limits concurrent generation to one request. It uses a separate chat token; the metrics token is not reused. Set `CHAT_ENABLED=true`, `CHAT_HOST=0.0.0.0`, `CHAT_PORT=8001`, and retain `/workspace/chat/chat-token` on the persistent volume when deploying. Keep vLLM private, retain the existing monitoring settings, and expose port 8001 only when ready to publish the authenticated gateway. Without `CHAT_ENABLED=true`, existing inference/monitoring behavior is preserved.

The deployed gateway origin is `https://fj04zaugwd1u83-8001.proxy.runpod.net`. The matching chat token is retained in the local chatbot's ignored `.env.local` and on the Pod's persistent volume; no credential is stored in the image. Verify it with `python scripts/check-chat-deployment.py --wait 420`. This checks access boundaries, readiness, actual-token context trimming, and one real streamed model response. Verify Railway collection with `python ../llm-grafana/scripts/check-deployment.py`.

Live validation completed on 2026-10-31 (America/New_York): missing/wrong chat tokens rejected; unrelated routes blocked; model ready; oversized earlier chat turns trimmed; a real Qwen response streamed to `[DONE]`. The local Next.js interface also returned a real model reply. Railway inference and GPU collection both returned healthy (`1`).

For rollback, restore image `zihan000/qwen-inference:0.1.1` and the prior HTTP port list `8888`, retaining the network volume and metrics settings. Remove or disable the chat settings; this returns the Pod to inference plus monitoring without the new chat gateway.

## Access securely

By default, vLLM listens only on the Pod's loopback interface. Enable Runpod SSH access with your public key. In the Pod's Connect panel, locate its SSH host and mapped TCP port, then on the monitoring computer establish a tunnel (replace placeholders with those actual values):

```bash
ssh -N -L 18000:127.0.0.1:8000 -p RUNPOD_SSH_PORT root@RUNPOD_SSH_HOST
```

Keep that connection open. On the same computer, `curl http://127.0.0.1:18000/health` should return successfully, and `/v1/chat/completions` can be called through that tunnel. Closing it stops access until it is reconnected.

For a public HTTP endpoint, configure `VLLM_HOST=0.0.0.0` and a strong `VLLM_API_KEY` through Runpod's secret/environment settings, and expose port 8000 only after configuring appropriate ingress controls. An API key does not necessarily protect every vLLM route, including operational endpoints; use an authenticated reverse proxy or network restrictions. Do not publish Prometheus or Grafana without protecting them. Do not commit keys to this project.

## Monitoring

This repository contains the metrics **relay**, not Grafana or Prometheus.
The relay runs with vLLM on the Pod and combines model metrics with GPU readings.
It requires the bearer token saved at `/workspace/monitoring/metrics-token` and
exposes `/metrics` through HTTP port 8888. While vLLM loads, authenticated scrapes
return 503. The collector retries automatically.

Deploy or run the monitoring services from their own repositories:

- [llm-grafana](https://github.com/alstomli/llm-grafana): dashboard, UI editing,
  JSON snapshots, Grafana image, and end-to-end cloud monitoring checks.
- [llm-prometheus](https://github.com/alstomli/llm-prometheus): scrape target,
  retention settings, collector image, and local collector checks.

The current metrics origin is `https://fj04zaugwd1u83-8888.proxy.runpod.net`.
After a Pod migration, update `RUNPOD_METRICS_TARGET` in Prometheus and
`RUNPOD_CHAT_URL` in the frontend deployment.

For local relay verification, retain the token in the ignored
`.secrets/metrics-token` file and run `python scripts/check-monitoring.py`.
Collector health checks live in the sibling Prometheus repository. Unit checks are `python scripts/test_metrics_relay.py`;
the chat and startup tests remain under `scripts/` and run in the inference image.

## Known baseline limitations

- The tested community INT4 checkpoint's quality has not been benchmarked against the original model; pin a model revision after choosing the exact release.
- The manual test reported FlashInfer CUDA toolkit warnings and fallback sampling. Image CUDA compatibility and kernel behavior need validation on the GPU.
- Model weights live on the network volume, but image dependencies live in the container image; backing up project files and long-lived data separately remains necessary.
- `MAX_NUM_SEQS=1` deliberately limits concurrent sequences even if the cache reports a higher theoretical capacity. Increase only after load testing.
