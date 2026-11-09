# Qwen inference server on Runpod

The 4bit and BF16 Pods can use one vLLM-based image with `MODEL_PROFILE=4bit`
(default) or `MODEL_PROFILE=bf16`. See the [official BF16 image deployment](deployments/qwen-bf16/VLLM-IMAGE.md)
for shared settings, the one-time migration, and versioned-image rollback.

For a trained local adapter, see the [BF16 LoRA serving configuration](deployments/qwen-bf16/LORA-SERVING.md).
It preserves the chatbot's BF16 alias while registering a separate private base
alias, with adapter validation and rollback instructions. Activation requires
the updated application image and serving checks; saving an adapter alone does
not enable it in chat.

BF16 Pod `ftntkktzid9f4p` is verified on its first official image release **0.10.1**,
with the same 40,960-token context and two active requests as the 4bit Pod, using
40 GiB CPU offloading. Its [verification record and deployment guide](deployments/qwen-bf16/VLLM-IMAGE.md)
include Jupyter on 8888, protected metrics on 8889, chat on 8001, observed memory
pressure, and the tested workload limits. The [earlier script-based runbook](deployments/qwen-bf16/README.md)
is retained as a historical record. The 4bit live deployment was retained.

## CI and image releases

The chat gateway is implemented in Go; vLLM and model tooling remain Python.
The image builds and tests the Go binary before packaging it. See the
[gateway migration and operations guide](docs/GATEWAY-GO.md) for local checks,
compatibility guarantees, and the explicit Python rollback option.

GitHub Actions builds the Linux image and runs the existing gateway, relay, and startup tests when pull requests are opened, updated, or reopened, without a GPU. Inference image checks do not run again when a pull request is merged into `main`; they can also be started manually. The **Build and publish release** workflow is manually triggered on `main` with a new version such as `0.4.0`. It tests the image before publishing to Docker Hub and records its commit and image digest in the workflow summary. Configure GitHub Actions secrets `DOCKER_USERNAME` and `DOCKER_PASSWORD` (a Docker Hub access token). The Docker Hub image repository must be publicly readable for the existing-tag check. Never reuse a published version; only this workflow is serialized, so avoid simultaneous external pushes to the same tag.

These workflows do not deploy or restart a Pod. Image publication requires the workflow to be run explicitly. The live deployment checker also accepts `RUNPOD_CHAT_URL` and `RUNPOD_CHAT_TOKEN` as environment variables, taking precedence over the local chatbot's `.env.local`. Running that checker sends a real inference request to the selected service.

The separate **Deploy or roll back Runpod image** workflow is manual and requires a GitHub environment with variable `RUNPOD_POD_ID` and secrets `RUNPOD_API_KEY` and `RUNPOD_CHAT_TOKEN`. Select that environment and provide a published `zihan000/qwen-inference@sha256:...` digest from the release summary. Use separate environments and Pods for testing and production; configure appropriate GitHub environment access controls. Only one deployment per environment can run at a time, so do not configure multiple environments for the same Pod or update it concurrently elsewhere.

The workflow asks for the target environment, image digest, and model profile. It records the previous configured image, updates only `imageName`, waits to observe a service interruption, checks the Runpod-reported image, and runs the authenticated live chat check. Existing Pod settings, including `MAX_NUM_SEQS`, are preserved; concurrency is managed on the Pod rather than through a workflow input. If deployment fails, it attempts to restore the previous image and repeat the checks; the run remains failed even if rollback succeeds. For a manual rollback, run the same workflow with a previously recorded digest. A previous image expressed as a tag is restored as that tag, so its registry contents must remain unchanged. Interruption or cancellation of the runner can prevent automatic rollback; use the recorded previous image and inspect the Pod manually. Set optional environment secret `RUNPOD_METRICS_TOKEN` to check the metrics relay after a successful deployment; this check reports separately and does not trigger rollback. Runpod reset timing, digest support, and recovery must be tested on a non-production Pod before production use; the interruption check can fail closed if the reset is too fast to observe. Deployment causes downtime and may incur GPU charges.

This project packages the vLLM configuration tested on one RTX PRO 4500. The Runpod Pod serves Qwen and runs a token-protected metrics relay. Prometheus and Grafana run on Railway; an optional local Docker stack remains available. See the separate [Grafana](https://github.com/alstomli/llm-grafana) and [Prometheus](https://github.com/alstomli/llm-prometheus) repositories. The SSH tunnel below describes an alternative that requires an SSH server.

## What runs where

- Runpod Pod: vLLM, model API and internal `/metrics` on `127.0.0.1:8000`; authenticated metrics relay on port 8888 and authenticated chat gateway on port 8001.
- Runpod network volume: Hugging Face model cache and vLLM compilation cache under `/workspace`.
- Railway: Prometheus and Grafana, with persistent monitoring history independent of your computer.
- Your computer: optional Prometheus and Grafana via Docker Compose.

## Compare concurrent chat performance

The [text/image concurrency benchmark](docs/CHAT-BENCHMARK.md) measures client
TTFT, reported completion-token throughput and gateway diagnostics for one text
request, two text requests, and text plus image. It uses two stable principals
and the same BF16 base model for all scenarios, without changing server settings.

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

See [the chatbot project](https://github.com/alstomli/qwen-chat) for the frontend and later deployment instructions. `scripts/chat-gateway.py` exposes only authenticated readiness and streaming chat routes, counts the actual vLLM chat-template tokens before generation, trims old exchanges to fit the context, and admits up to two active requests plus six waiting requests per gateway process, with FIFO promotion among accounts eligible to run. It uses a separate chat token; the metrics token is not reused. Set `CHAT_ENABLED=true`, `CHAT_HOST=0.0.0.0`, `CHAT_PORT=8001`, and retain `/workspace/chat/chat-token` on the persistent volume when deploying. Keep vLLM private, retain the existing monitoring settings, and expose port 8001 only when ready to publish the authenticated gateway. Without `CHAT_ENABLED=true`, existing inference/monitoring behavior is preserved.

The deployed gateway origin is `https://rp93m3noz3l9lt-8001.proxy.runpod.net`. The matching chat token is retained in the local chatbot's ignored `.env.local` and on the Pod's persistent volume; no credential is stored in the image. Verify it with `python scripts/check-chat-deployment.py --wait 420`. This checks readiness and a streamed response to a short greeting. Verify Railway collection with `python ../llm-grafana/scripts/check-deployment.py`.

Live validation completed on 2026-10-31 (America/New_York): missing/wrong chat tokens rejected; unrelated routes blocked; model ready; oversized earlier chat turns trimmed; a real Qwen response streamed to `[DONE]`. The local Next.js interface also returned a real model reply. Railway inference and GPU collection both returned healthy (`1`).

For rollback, restore image `zihan000/qwen-inference:0.1.1` and the prior HTTP port list `8888`, retaining the network volume and metrics settings. Remove or disable the chat settings; this returns the Pod to inference plus monitoring without the new chat gateway.

## Request admission and queue

The gateway immediately admits up to **2 active requests** (including preparation and streaming) and holds up to **6 additional waiting requests** with FIFO promotion among eligible accounts. Each account defaults to **1 active request**, **1 waiting request**, and **6 accepted requests per rolling 60 seconds** on that Pod. vLLM continuously batches available work; the gateway has no collection window and does not wait for a second request. Admission is in-memory and per process; keep one gateway worker per model instance. Restarting the process drops its queued requests.

Authenticated `/health` includes `admission.activeRequests`, `waitingRequests`, `maxConcurrentRequests`, and `maxWaitingRequests`. Cancellation or disconnect removes waiting requests without forwarding them to the model; finishing/cancelling an active stream closes its upstream connection and hands the slot to the oldest eligible waiter after upstream cleanup completes. An account waiting on its active request cannot block other eligible accounts, and new requests cannot bypass older eligible waiters. Waiting consumes the existing request timeout rather than starting a fresh generation budget. Queue expiration returns HTTP 504 with `X-Chat-Rejection: queue_timeout`; a full queue returns 429 with `queue_full`. The independent rolling limit of 20 accepted requests per minute returns 429 with `rate_limit`. Reservations count when accepted, including requests later cancelled, expired or rejected by token fitting; queue-full/rate-limit rejections do not count. The companion frontend recognizes these allowlisted reasons without displaying upstream error bodies.


The private chat contract requires a gateway bearer token followed by one canonical hyphenated UUID in `X-Chat-User-Id`. Uppercase UUIDs normalize to lowercase. The frontend derives this value once from the verified Supabase user and active-session check; inbound browser headers and body identity fields are ignored. Missing or malformed identity returns 400 with `X-Chat-Rejection: user_identity` before reading the body or consuming quota. Health remains bearer-only and advertises `accountQuotaVersion: 1` and `accountQuotasEnforced: true`. Requests rejected before reservation (identity, JSON/schema validation, or admission limits) do not consume quota; accepted requests later cancelled, timed out, or failed, including image decoding/token fitting, do. Account capacity returns 429/`user_capacity`; account rate returns 429/`user_rate_limit`. Only rate rejections include integer `Retry-After` (1–60 seconds), including the existing global `rate_limit` reason.

Account limits can be configured at process startup: `CHAT_USER_MAX_CONCURRENT=1`, `CHAT_USER_MAX_WAITING=1`, and `CHAT_USER_RATE_LIMIT_PER_MINUTE=6`. All must be integers within the corresponding global ceiling (2 active, 6 waiting, 20 accepted/minute); active/rate must be positive and waiting may be zero. Global ceilings remain unchanged. Adapter/base aliases share the same account budget. All state is process-local and lazily pruned on admission/release; run exactly one gateway worker per Pod. Restart clears its rate window and reservations. Separate Pods have independent budgets. See [the quota decision](docs/adr-account-quotas.md).

Quota rollout order: deploy this gateway temporarily with explicit `CHAT_REQUIRE_USER_ID=false` only for old callers; missing identities share one bounded legacy account while malformed identities still fail 400. Then deploy the matching frontend that forwards verified identity. Its chat/status routes fail 503 unless health confirms version 1 **and enforcement true**, so hold frontend traffic activation until both selected gateways are restarted with `CHAT_REQUIRE_USER_ID=true` (the default) and verified. This sequence can produce temporary 503 during the transition; compatibility is not a frontend bypass. Re-enable traffic only after strict health and an authenticated streamed request succeed for each Pod. Normal rollback retains strict enforcement with a compatible frontend, or disables chat. Reverting to an old frontend plus temporary gateway compatibility is an explicit operator security tradeoff, not the default rollback; never silently disable enforcement.

The direct `scripts/check-chat-deployment.py` checker verifies strict quota capability before sending inference. It sends stable trusted service principal `00000000-0000-4000-8000-000000000001` by default; override using `--user-id` or `CHAT_CHECK_USER_ID` with a canonical UUID. This principal is a trusted bearer-holder smoke-check identity, shares the ordinary account budget across checker runs, and is not an anonymous bypass or a real end-user identity. Do not rotate IDs to evade the rate limit.

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

### Image attachments on BF16

`CHAT_IMAGE_INPUT_ENABLED=true` opts the BF16 gateway into image input. Its authenticated `/health` advertises the attachment limits when enabled. A chat request can include one top-level `image` field alongside the existing text `messages` array. Use an inline base64 JPEG, PNG, or WebP up to 256 KiB decoded and 2,048 pixels on either edge. The complete JSON still has to fit the request byte limit. The gateway attaches the image to the newest user message for both vLLM tokenization and completion. The reply still streams as text.

When a trained adapter is registered, send `useAdapter: false` for an image request. The gateway uses the BF16 base model for that request. Roll out with the flag enabled on the BF16 test Pod first. Ask a question that depends on a known test image and compare the tokenizer count with the completion's reported prompt tokens for the same prompt. Repeat with a different image and with a text-only request before enabling the flag on the live Pod.

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
