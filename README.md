# Qwen inference server on Runpod

## CI and image releases

GitHub Actions builds the Linux image and runs the existing gateway, relay, and startup tests on pull requests and pushes to `main`, without a GPU. The **Build and publish release** workflow is manually triggered on `main` with a new version such as `0.4.0`. It tests the image before publishing to Docker Hub and records its commit and image digest in the workflow summary. Configure GitHub Actions secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (a Docker Hub access token). The Docker Hub image repository must be publicly readable for the existing-tag check. Never reuse a published version; only this workflow is serialized, so avoid simultaneous external pushes to the same tag.

These workflows do not deploy or restart a Pod. Image publication requires the workflow to be run explicitly. The live deployment checker also accepts `RUNPOD_CHAT_URL` and `RUNPOD_CHAT_TOKEN` as environment variables, taking precedence over the local chatbot's `.env.local`. Running that checker sends a real inference request to the selected service.

The separate **Deploy or roll back Runpod image** workflow is manual and requires a GitHub environment with variable `RUNPOD_POD_ID` and secrets `RUNPOD_API_KEY` and `RUNPOD_CHAT_TOKEN`. Select that environment and provide a published `zihan000/qwen-inference@sha256:...` digest from the release summary. Use separate environments and Pods for testing and production; configure appropriate GitHub environment access controls. Only one deployment per environment can run at a time, so do not configure multiple environments for the same Pod or update it concurrently elsewhere.

The workflow records the previous configured image, updates only `imageName`, waits to observe a service interruption, checks the Runpod-reported image and runs the authenticated live chat check. If deployment fails it attempts to restore the previous image and repeat those checks; the run remains failed even if rollback succeeds. For a manual rollback, run the same workflow with a previously recorded digest. A previous image expressed as a tag is restored as that tag, so its registry contents must remain unchanged. Interruption or cancellation of the runner can prevent automatic rollback; use the recorded previous image and inspect the Pod manually. Set optional environment secret `RUNPOD_METRICS_TOKEN` to check the metrics relay after a successful deployment; this check reports separately and does not trigger rollback. Runpod reset timing, digest support, and recovery must be tested on a non-production Pod before production use; the interruption check can fail closed if the reset is too fast to observe. Deployment causes downtime and may incur GPU charges.

This project packages the vLLM configuration tested on one RTX PRO 4500. The Runpod Pod serves Qwen and runs a token-protected metrics relay. Prometheus and Grafana run on Railway; an optional local Docker stack remains available. See the separate [Grafana](https://github.com/alstomli/llm-grafana) and [Prometheus](https://github.com/alstomli/llm-prometheus) repositories. The SSH tunnel below describes an alternative that requires an SSH server.

## What runs where

- Runpod Pod: vLLM, model API and internal `/metrics` on `127.0.0.1:8000`; authenticated metrics relay on port 8888 and authenticated chat gateway on port 8001.
- Runpod network volume: Hugging Face model cache and vLLM compilation cache under `/workspace`.
- Railway: Prometheus and Grafana, with persistent monitoring history independent of your computer.
- Your computer: optional Prometheus and Grafana via Docker Compose.

## Prepare the inference image

The live Pod `fj04zaugwd1u83` now uses `0.3.0`, with automatic monitoring and configurable chat gateway limits. The Docker Hub release digest is `sha256:2bc9682524bfbfcd44067d5cd74ddabc9dc05f02e42559497d3642246b0ae2bb`. Deployed November 1, 2026; the model reached readiness with a 40,960-token context ceiling. Keep published release tags unchanged; use a new version for later code changes.

Build an **x86-64 Linux** image on a machine with Docker and registry access. The Docker Hub namespace for this project is `zihan000`:

```bash
docker buildx build --platform linux/amd64 -t zihan000/qwen-inference:0.2.0 --push .
```

Then create a Runpod custom template or update the existing Pod with that image, one RTX PRO 4500 and the existing network volume mounted at `/workspace`. Do not override the image entrypoint. The image starts vLLM automatically and defaults to the model above with a 40,960-token context and one sequence. This configuration has reached readiness on the live GPU; maximum-length and sustained-load performance have not been benchmarked. Set `MAX_MODEL_LEN` to override it consistently for both vLLM and the chat gateway. Existing Pod environment overrides take precedence; rebuild the image and restart the Pod to apply source changes. The official base is pinned to vLLM 0.32.0.

For automatic metrics collection, set `METRICS_ENABLED=true` and `METRICS_HOST=0.0.0.0` in the Pod environment, expose HTTP port 8888, and retain the existing token file `/workspace/monitoring/metrics-token` on the network volume. The token must contain at least 32 characters. It is not embedded in the image. The relay script is included at `/usr/local/lib/metrics-relay.py`; the older copy on `/workspace` is no longer the startup source. The launcher starts the enabled services on every container start, forwards shutdown, and exits if any service exits. With both `METRICS_ENABLED=false` and `CHAT_ENABLED=false` (the defaults), startup runs vLLM only.

Test the launcher without a GPU or model download:

```bash
docker run --rm --entrypoint python3 --mount type=bind,source="$PWD/scripts",target=/tests,readonly zihan000/qwen-inference:0.2.0 /tests/test_start_server.py
```

Test a replacement Pod before retiring the existing Pod. A network volume is tied to its data center and must be selected at deployment. Current Runpod Pod pricing and GPU availability apply. If a container exits or a machine fails, recovery depends on Runpod's behavior and Pod availability; this image alone does not guarantee high availability or restart-on-failure.

## Chat gateway

See [the chatbot project](https://github.com/alstomli/qwen-chat) for the frontend and later deployment instructions. `scripts/chat-gateway.py` exposes only authenticated readiness and streaming chat routes, counts the actual vLLM chat-template tokens before generation, trims old exchanges to fit the context, and limits concurrent generation to one request. It uses a separate chat token; the metrics token is not reused. Set `CHAT_ENABLED=true`, `CHAT_HOST=0.0.0.0`, `CHAT_PORT=8001`, and retain `/workspace/chat/chat-token` on the persistent volume when deploying. Keep vLLM private, retain the existing monitoring settings, and expose port 8001 only when ready to publish the authenticated gateway. Without `CHAT_ENABLED=true`, existing inference/monitoring behavior is preserved.

The deployed gateway origin is `https://fj04zaugwd1u83-8001.proxy.runpod.net`. The matching chat token is retained in the local chatbot's ignored `.env.local` and on the Pod's persistent volume; no credential is stored in the image. Verify it with `python scripts/check-chat-deployment.py --wait 420`. This checks readiness and a streamed response to a short greeting. Verify Railway collection with `python ../llm-grafana/scripts/check-deployment.py`.

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

## Per-request chat limits

The updated frontend can send a `settings` object with `inputCharacters`, `outputTokens`, `contextTokens`, `historyMessages`, `historyBytes`, `requestBytes`, `timeoutSeconds`, and `safetyTokens`. These preferences lower the deployment ceilings in `scripts/chat-gateway.py`; they cannot raise vLLM capacity. The gateway enforces character, byte, history, token, and timeout budgets and forwards only model generation options to vLLM. The newest input character count follows browser UTF-16 length; history/request sizes use UTF-8 bytes. Output plus safety reserve must leave input space. Generation timeout covers tokenization, upstream prefill, and streaming, and cancellation releases gateway capacity.

Authenticated `/health` includes `settingsVersion: 1` and `limits`. Its context ceiling is the smaller of the gateway configuration, its application ceiling, and the running model's `max_model_len` from private `/v1/models`. The model capacity is rechecked per request. If capacity cannot be verified, readiness/generation fails instead of guessing. Rebuild/restart the gateway and deploy the matching frontend together; older gateways do not support these controls. Live smoke tests on November 1, 2026 verified output truncation at 128 tokens, input rejection with a 5-character limit, old-exchange trimming with a 512-token context budget, and rejection above the deployed 40,960-token ceiling. The published image also passed 10 gateway and 8 startup checks.
