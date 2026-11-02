# Qwen inference server on Runpod

## CI and image releases

GitHub Actions builds the Linux image and runs the existing gateway, relay, and startup tests when pull requests are opened, updated, or reopened, without a GPU. Inference image checks do not run again when a pull request is merged into `main`; they can also be started manually. The **Build and publish release** workflow is manually triggered on `main` with a new version such as `0.4.0`. It tests the image before publishing to Docker Hub and records its commit and image digest in the workflow summary. Configure GitHub Actions secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (a Docker Hub access token). The Docker Hub image repository must be publicly readable for the existing-tag check. Never reuse a published version; only this workflow is serialized, so avoid simultaneous external pushes to the same tag.

These workflows do not deploy or restart a Pod. Image publication requires the workflow to be run explicitly. The live deployment checker also accepts `RUNPOD_CHAT_URL` and `RUNPOD_CHAT_TOKEN` as environment variables, taking precedence over the local chatbot's `.env.local`. Running that checker sends a real inference request to the selected service.

The separate **Deploy or roll back Runpod image** workflow is manual and requires a GitHub environment with variable `RUNPOD_POD_ID` and secrets `RUNPOD_API_KEY` and `RUNPOD_CHAT_TOKEN`. Select that environment and provide a published `zihan000/qwen-inference@sha256:...` digest from the release summary. Use separate environments and Pods for testing and production; configure appropriate GitHub environment access controls. Only one deployment per environment can run at a time, so do not configure multiple environments for the same Pod or update it concurrently elsewhere.

The workflow records the previous configured image, updates `imageName` and an optional concurrency override, waits to observe a service interruption, checks the Runpod-reported image and runs the authenticated live chat check. The optional `max_num_seqs` workflow input accepts `1` or `2`; an empty value preserves image-only behavior. When supplied, the job preserves the full existing environment, changes only `MAX_NUM_SEQS`, and sends image and environment in one update. A value of `2` also requires health to advertise two active and six waiting slots. If deployment fails it attempts to restore the previous image and, when overridden, the previous environment, and repeat those checks; the run remains failed even if rollback succeeds. For a manual rollback, run the same workflow with a previously recorded digest. A previous image expressed as a tag is restored as that tag, so its registry contents must remain unchanged. Interruption or cancellation of the runner can prevent automatic rollback; use the recorded previous image and inspect the Pod manually. Set optional environment secret `RUNPOD_METRICS_TOKEN` to check the metrics relay after a successful deployment; this check reports separately and does not trigger rollback. Runpod reset timing, digest support, and recovery must be tested on a non-production Pod before production use; the interruption check can fail closed if the reset is too fast to observe. Deployment causes downtime and may incur GPU charges.

This project packages the vLLM configuration tested on one RTX PRO 4500. The Runpod Pod serves Qwen and runs a token-protected metrics relay. Prometheus and Grafana run on Railway; an optional local Docker stack remains available. See the separate [Grafana](https://github.com/alstomli/llm-grafana) and [Prometheus](https://github.com/alstomli/llm-prometheus) repositories. The SSH tunnel below describes an alternative that requires an SSH server.

## What runs where

- Runpod Pod: vLLM, model API and internal `/metrics` on `127.0.0.1:8000`; authenticated metrics relay on port 8888 and authenticated chat gateway on port 8001.
- Runpod network volume: Hugging Face model cache and vLLM compilation cache under `/workspace`.
- Railway: Prometheus and Grafana, with persistent monitoring history independent of your computer.
- Your computer: optional Prometheus and Grafana via Docker Compose.

## Prepare the inference image

The live Pod migrated to `rp93m3noz3l9lt` on November 2, 2026, with automatic monitoring and configurable chat gateway limits. Its configured image is `zihan000/qwen-inference@sha256:e52015057d9e1826c059df3683790ceea28f7711c2692189e1535a75634417ba`; the model is ready with a 40,960-token context ceiling. Keep published release tags unchanged; use a new version for later code changes.

Build an **x86-64 Linux** image on a machine with Docker and registry access. The Docker Hub namespace for this project is `zihan000`:

```bash
docker buildx build --platform linux/amd64 -t zihan000/qwen-inference:0.2.0 --push .
```

Then create a Runpod custom template or update the existing Pod with that image, one RTX PRO 4500 and the existing network volume mounted at `/workspace`. Do not override the image entrypoint. The image starts vLLM automatically and defaults to the model above with a 40,960-token context and up to two sequences, with no batching collection window. The earlier single-sequence configuration reached readiness on the live GPU; the new two-sequence configuration requires GPU load validation before rollout. Set `MAX_MODEL_LEN` to override it consistently for both vLLM and the chat gateway. Existing Pod environment overrides take precedence; rebuild the image and restart the Pod to apply source changes. The official base is pinned to vLLM 0.32.0.

For automatic metrics collection, set `METRICS_ENABLED=true` and `METRICS_HOST=0.0.0.0` in the Pod environment, expose HTTP port 8888, and retain the existing token file `/workspace/monitoring/metrics-token` on the network volume. The token must contain at least 32 characters. It is not embedded in the image. The relay script is included at `/usr/local/lib/metrics-relay.py`; the older copy on `/workspace` is no longer the startup source. The launcher starts the enabled services on every container start, forwards shutdown, and exits if any service exits. With both `METRICS_ENABLED=false` and `CHAT_ENABLED=false` (the defaults), startup runs vLLM only.

The vLLM startup configuration sets TTFT histogram boundaries every 0.1 seconds from 0.1 through 10.0 seconds and retains all existing boundaries, including the finer sub-100ms boundaries and the 0.25s and 0.75s boundaries. Other histogram families keep their defaults. New bucket series record observations only after the updated server starts; older measurements cannot be reconstructed at the finer resolution. Deploying the change restarts the Pod and resets its in-memory prefix cache.

Test the launcher without a GPU or model download:

```bash
docker run --rm --entrypoint python3 --mount type=bind,source="$PWD/scripts",target=/tests,readonly zihan000/qwen-inference:0.2.0 /tests/test_start_server.py
```

Test a replacement Pod before retiring the existing Pod. A network volume is tied to its data center and must be selected at deployment. Current Runpod Pod pricing and GPU availability apply. If a container exits or a machine fails, recovery depends on Runpod's behavior and Pod availability; this image alone does not guarantee high availability or restart-on-failure.

## Chat gateway

See [the chatbot project](https://github.com/alstomli/qwen-chat) for the frontend and later deployment instructions. `scripts/chat-gateway.py` exposes only authenticated readiness and streaming chat routes, counts the actual vLLM chat-template tokens before generation, trims old exchanges to fit the context, and admits up to two active requests plus six FIFO waiting requests per gateway process. It uses a separate chat token; the metrics token is not reused. Set `CHAT_ENABLED=true`, `CHAT_HOST=0.0.0.0`, `CHAT_PORT=8001`, and retain `/workspace/chat/chat-token` on the persistent volume when deploying. Keep vLLM private, retain the existing monitoring settings, and expose port 8001 only when ready to publish the authenticated gateway. Without `CHAT_ENABLED=true`, existing inference/monitoring behavior is preserved.

The deployed gateway origin is `https://rp93m3noz3l9lt-8001.proxy.runpod.net`. The matching chat token is retained in the local chatbot's ignored `.env.local` and on the Pod's persistent volume; no credential is stored in the image. Verify it with `python scripts/check-chat-deployment.py --wait 420`. This checks readiness and a streamed response to a short greeting. Verify Railway collection with `python ../llm-grafana/scripts/check-deployment.py`.

Live validation completed on 2026-10-31 (America/New_York): missing/wrong chat tokens rejected; unrelated routes blocked; model ready; oversized earlier chat turns trimmed; a real Qwen response streamed to `[DONE]`. The local Next.js interface also returned a real model reply. Railway inference and GPU collection both returned healthy (`1`).

For rollback, restore image `zihan000/qwen-inference:0.1.1` and the prior HTTP port list `8888`, retaining the network volume and metrics settings. Remove or disable the chat settings; this returns the Pod to inference plus monitoring without the new chat gateway.

## Request admission and queue

The gateway immediately admits up to **2 active requests** (including preparation and streaming) and holds up to **6 additional waiting requests** in arrival order. vLLM continuously batches available work; the gateway has no collection window and does not wait for a second request. Admission is in-memory and per process; keep one gateway worker per model instance. Restarting the process drops its queued requests.

Authenticated `/health` includes `admission.activeRequests`, `waitingRequests`, `maxConcurrentRequests`, and `maxWaitingRequests`. Cancellation or disconnect removes waiting requests without forwarding them to the model; finishing/cancelling an active stream closes its upstream connection and hands the slot to the oldest waiter. Waiting consumes the existing request timeout rather than starting a fresh generation budget. Queue expiration returns HTTP 504 with `X-Chat-Rejection: queue_timeout`; a full queue returns 429 with `queue_full`. The independent rolling limit of 20 accepted requests per minute returns 429 with `rate_limit`. Reservations count when accepted, including requests later cancelled, expired or rejected by token fitting; queue-full/rate-limit rejections do not count. The companion frontend recognizes these allowlisted reasons without displaying upstream error bodies.

For rollout, build a new immutable image release and set the Pod's `MAX_NUM_SEQS=2` explicitly; an existing override of `1` takes precedence over the new launcher default and keeps the gateway at one active request. Deploy the companion frontend for specific queue/rate/timeout messages. Existing hosting/proxy execution limits also bound how long a request can wait; on Vercel the chat route declares 300 seconds total, including waiting. Deployment/restart and live GPU validation require separate approval. Local HTTP tests use a fake model and do not establish GPU batching performance.

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

The current metrics origin is `https://rp93m3noz3l9lt-8888.proxy.runpod.net`.
After a Pod migration, update `RUNPOD_METRICS_TARGET` in Prometheus and
`RUNPOD_CHAT_URL` in the frontend deployment. Also update `RUNPOD_POD_ID` in
the GitHub deployment environment (`prod` currently uses `rp93m3noz3l9lt`)
so future deployment workflows target the migrated Pod.

For local relay verification, retain the token in the ignored
`.secrets/metrics-token` file and run `python scripts/check-monitoring.py`.
Collector health checks live in the sibling Prometheus repository. Unit checks are `python scripts/test_metrics_relay.py`;
the chat and startup tests remain under `scripts/` and run in the inference image.

## Known baseline limitations

- The tested community INT4 checkpoint's quality has not been benchmarked against the original model; pin a model revision after choosing the exact release.
- The manual test reported FlashInfer CUDA toolkit warnings and fallback sampling. Image CUDA compatibility and kernel behavior need validation on the GPU.
- Model weights live on the network volume, but image dependencies live in the container image; backing up project files and long-lived data separately remains necessary.
- The launcher now defaults to `MAX_NUM_SEQS=2`. The gateway admits at most two active requests and follows a lower `MAX_NUM_SEQS=1` override. Raising the engine setting above two does not raise gateway concurrency. Validate long-context KV-cache pressure, preemption, throughput and latency on the GPU; two simultaneous maximum-length requests are not guaranteed to fit.

## Per-request chat limits

`settings.reuseCachedPrefixes` is a boolean that defaults to `true`. When false, the gateway generates a fresh cryptographic `cache_salt` for every generation request, preventing reuse of earlier requests' cached prefixes. Client-supplied salts are not forwarded. This does not disable cache writes or release reserved GPU memory. The launcher explicitly enables vLLM prefix caching. Health advertises support via `limits.reuseCachedPrefixes: true`, while retaining settings version 1 for existing clients. Deploy the updated inference image before enabling the frontend control; the frontend also supports older gateways with the toggle unavailable.

The updated frontend can send a `settings` object with `inputCharacters`, `outputTokens`, `contextTokens`, `historyMessages`, `historyBytes`, `requestBytes`, `timeoutSeconds`, and `safetyTokens`. These preferences lower the deployment ceilings in `scripts/chat-gateway.py`; they cannot raise vLLM capacity. The gateway enforces character, byte, history, token, and timeout budgets and forwards only model generation options to vLLM. The newest input character count follows browser UTF-16 length; history/request sizes use UTF-8 bytes. Output plus safety reserve must leave input space. Generation timeout covers FIFO queue wait, model-capacity verification, tokenization, upstream prefill, and streaming, and cancellation releases gateway capacity.

Authenticated `/health` includes `settingsVersion: 1` and `limits`. Its context ceiling is the smaller of the gateway configuration, its application ceiling, and the running model's `max_model_len` from private `/v1/models`. The model capacity is rechecked per request. If capacity cannot be verified, readiness/generation fails instead of guessing. Rebuild/restart the gateway and deploy the matching frontend together; older gateways do not support these controls. Live smoke tests on November 1, 2026 verified output truncation at 128 tokens, input rejection with a 5-character limit, old-exchange trimming with a 512-token context budget, and rejection above the deployed 40,960-token ceiling. The published image also passed 10 gateway and 8 startup checks.

## Opt-in development token diagnostics

`CHAT_DIAGNOSTICS_ENABLED=true` enables diagnostic capability on the gateway; it defaults off. `/health` reports `diagnosticsVersion: 1` when enabled and `0` otherwise. Authenticated chat callers additionally send `X-Chat-Diagnostics: 1` per request. A disabled gateway rejects that header with 403, and normal requests neither emit diagnostics nor request extra usage. The matching development/Preview frontend must also opt in. Do not enable this by default for a normal production deployment.

The response header with the same name contains a compact version-1 JSON summary: server-generated request ID, configured model, exact first/final tokenizer counts, last available count, gateway-trimmed message count, validated output/context/safety budgets, and prefix-reuse policy. Counts include the system message and exact served chat template; "before" means before gateway trimming, after the browser has already applied its own limits. The gateway records counts from existing `/tokenize` calls, without additional calls. Errors retain available metadata; final input remains null if no fitting input was found. No prompt text, answer content, token IDs, credentials or salt is returned in the diagnostic header or logged by this feature.

Only opted-in generations add `stream_options: {"include_usage": true}` to the model request. The original streaming bytes, finish reasons, cancellation cleanup, admission limits and token-fitting rules remain intact; the frontend handles any final usage-only event. Missing final usage during stop/disconnect is not reported as zero. This feature does not expose private vLLM routes or change cache reuse.

### Response timing diagnostics

Opted-in streams also contain one named SSE `event: chat-timing` before `[DONE]`, with `version: 1`, the same `requestId` as the diagnostic header, and an allowlisted `timings` object in milliseconds. Original upstream SSE frames are preserved; ordinary requests remain a byte-for-byte passthrough. Timing consumers must handle the named event separately from completion content. Older frontends ignore it; the matching frontend exports it with token diagnostics. No prompts or answer text are added to metadata or logs.

- `contextFitMs`: time inside the admitted context-fitting loop, including all tokenizer requests, template counting, and old-exchange trimming. Excludes the earlier model-capacity check and request validation.
- `upstreamFirstContentMs`: generation POST start to the first complete SSE event with nonempty assistant content observed at the gateway. Includes upstream connection, scheduling, prefill and transport; it is not vLLM's engine TTFT.
- `gatewayFirstContentMs`: chat handler entry to that same first content event, including validation, model-capacity lookup and context fitting. Authentication middleware runs before this boundary.
- `gatewayTotalMs`: chat handler entry to observing `[DONE]`, or detecting stream EOF/failure. It excludes delivery from the gateway to the browser.

All durations use the gateway's monotonic clock. Empty/role-only chunks and usage-only events do not establish first content; stream chunks are never counted as tokens. Parsing handles split UTF-8 and event boundaries, with a 1 MiB limit on an incomplete diagnostic SSE event. No first content means null, including timeout before output. A disconnected caller may never receive the final event. Pre-generation errors retain the existing diagnostic header but have no timing event. Keep using Grafana for engine TTFT, ITL and queue time; do not subtract percentiles measured at different boundaries.
