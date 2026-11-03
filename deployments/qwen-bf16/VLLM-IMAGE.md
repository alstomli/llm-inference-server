# Official BF16 image deployment

The active BF16 hardware is now RTX PRO 6000 Pod `o6f2qoctk798tq`, with
`CPU_OFFLOAD_GB=0`. See [the RTX PRO 6000 migration guide](RTX6000-MIGRATION.md)
for current endpoints, configuration and verification. The RTX PRO 4500 SE
measurements and 40 GiB offload settings below describe the historical deployment.

The BF16 Pod's first official release uses the same `zihan000/qwen-inference`
image and versioned-image/digest workflow as the 4bit Pod. Its image starts from
`vllm/vllm-openai:v0.32.0`; application code and Jupyter dependencies are installed
at image build time. Container startup does not install vLLM or execute the old
volume bootstrap. The original BF16 weights stay on global volume
`spicy_coffee_meerkat` at `/workspace/qwen-bf16/model`.

## Completed first official verification

Verified November 2, 2026 (America/New_York). Release **0.10.1** is the first
accepted official BF16 image; `previous_official_release` is null. The 0.10.0
candidate was not accepted because Jupyter rejected browser proxy-Origin
requests, and it is not an official rollback predecessor.

```text
zihan000/qwen-inference@sha256:f0f6c699b2075ed714e67860c5afe26c411dc287ec957eb33382765f54ac25a7
```

- [Release workflow](https://github.com/alstomli/llm-inference-server/actions/runs/41575958803)
  built, tested, and published the Linux/AMD64 image from source merge
  `5b84e08522e06f4fe90c956e740cf1fbd3cc2048`.
- [Implementation PR #13](https://github.com/alstomli/llm-inference-server/pull/13)
  and [proxy fix PR #14](https://github.com/alstomli/llm-inference-server/pull/14)
  are merged. Final CI passed 62 tests, including cookie-only Origin checks,
  XSRF enforcement, and real terminal WebSocket input/output.
- [Sanitized verification record](verification-0.10.1.json) contains the actual
  external phase results and resource/admin evidence. The accepted runtime
  record is also on the global volume at
  `/workspace/qwen-bf16/runtime-verification.json`.

| Live check | Observed result |
| --- | --- |
| Packaged runtime | vLLM 0.32.0; PyTorch 2.13.0+cu130; Transformers 5.19.0; no legacy virtual environment |
| Checkpoint reuse | 18 cached weight shards; 1,199 BF16 tensors; original revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Model and serving limits | `qwen-bf16`; 40,960-token context; two active requests and six waiting |
| Prefix control | Prefix reuse advertised; a real request with reuse disabled completed |
| Authentication/routes | Correct chat token accepted; missing/wrong tokens rejected; non-chat routes blocked |
| Excess context | Request beyond the serving context rejected with HTTP 400 |
| Single synthetic generation | Nonempty response completed with `stop`; first text 50.633 s, total 52.378 s |
| Two short synthetic generations | HTTP intervals overlapped; two active requests observed; both completed (10.471 s and 12.112 s) |
| Cancellation | Client closed after first text; active capacity was released |
| Jupyter | Existing credential login and actual proxy terminal command worked on port 8888 |
| Monitoring | Both Pod scrapes and both GPU exporters healthy; original target identities retained |

The external phases ran from `2026-11-03T04:23:17Z` to
`2026-11-03T04:31:07Z` (the preceding evening in New York). They used four
synthetic BF16 requests: one prefix-disabled generation, two overlapping short
generations, and one cancelled generation. These timings describe that workload;
they are not a throughput benchmark.

The engine reported **15.19 GiB available KV cache and 220,073 GPU KV tokens**.
Its estimated concurrency at 40,960 tokens was 5.37; actual admission remains
configured to two. Cache estimates do not replace workload tests.

After the external tests, GPU use was **27,749 / 32,623 MiB**. Cgroup memory was:

| Measurement | Observed result |
| --- | --- |
| Current usage | 86,301,044,736 bytes (80.37 GiB) |
| Peak usage | 94,000,001,024 bytes (approximately 87.54 GiB) |
| Cgroup limit | 93,999,996,928 bytes (87.54 GiB) |
| Inactive file cache | 15,755,755,520 bytes (14.67 GiB) |
| Current minus inactive file cache | 70,545,289,216 bytes (65.70 GiB), a working-memory estimate |
| `memory.events max` | 8,567; unchanged during the request tests |
| OOM / OOM-kill events | 0 / 0 |

Loading reached the RAM cap and reclaimed file cache. No OOM or OOM-kill was
recorded, but this is a memory-pressure limitation of the present Pod. Two
simultaneous full-context generations were **not** benchmarked; the verified
concurrent workload was two short prompts.

After the frontend browser checks, current RAM was 80,542,801,920 bytes
(75.01 GiB). The peak, limit, and `memory.events max` count were unchanged,
and OOM/OOM-kill counts remained zero.

The local frontend at `http://127.0.0.1:3001` subsequently passed actual browser
checks against both Pods in one conversation: correct 4bit/BF16 replies and
assistant labels, selector locked during generation, and selected model/labels
retained after reload. See [frontend PR #7](https://github.com/alstomli/qwen-chat/pull/7).
Its screenshot is retained in the project workspace at
`images-in-chat/model-selector-live-two-pods.jpg`.

## Request diagnostics verified November 3, 2026

Only `CHAT_DIAGNOSTICS_ENABLED=true` was added to the BF16 Pod's persistent
environment. Its 0.10.1 image, other environment settings, ports, and startup
configuration were retained. After the restart, the gateway and hosted frontend
both reported the BF16 model ready with diagnostics version 1; the hosted
frontend also reported diagnostics available for the 4bit model. Context remained
40,960 tokens, with two active and six waiting slots. The frontend server's
own diagnostics gate was already enabled on the hosted deployment; the local
frontend's setting was not changed.

Actual hosted browser checks confirmed both model controls were enabled and
independent: the 4bit diagnostic checkbox was unchecked while the BF16 checkbox
was checked. Exactly one opted-in BF16 request completed through the hosted
frontend with model `qwen-bf16` and finish reason `stop`. Its diagnostic record
reported 47 input tokens before and after fitting, 47 actual vLLM prompt tokens,
and seven output tokens, using an output budget of 128, a safety reserve of 32,
and context budget 40,960. Timing metadata was accepted only after its request ID
matched the request record. This was a functional check of one post-restart
request, not a measurement of diagnostics overhead.

The BF16 toggle was then restored to off, and its prior output preference of
7,680 tokens was restored. Jupyter's existing credential login and actual proxy
terminal remained usable; both existing Prometheus targets and GPU exporters
were healthy. The BF16 gateway returned to zero active and zero waiting requests.
Post-request private vLLM metrics also showed zero running and waiting requests.
GPU usage was 27,747 / 32,623 MiB. The container retained the same
93,999,996,928-byte RAM cap with zero OOM and OOM-kill events. Its memory-cap
pressure counter was 8,579; this may include profiling and is not an assertion
that the counter stayed unchanged during inference. The raw resource record is
on the volume at `/workspace/qwen-bf16/diagnostics-resources.json`.
The original first-release JSON remains the historical November 2 record; this
later configuration check does not rewrite that evidence.

## Port roles on the two Pods

| Pod | HTTP ports | Jupyter |
| --- | --- | --- |
| 4bit (`rp93m3noz3l9lt`) | 8888: authenticated metrics; 8001: authenticated chat | No Jupyter launcher is configured in its current deployment |
| BF16 (`ftntkktzid9f4p`) | 8888: authenticated Jupyter; 8889: authenticated metrics; 8001: authenticated chat | Enabled on 8888 |

Both keep vLLM private at `127.0.0.1:8000`. The same port number on different
Pods identifies separate services. Runpod's generic Jupyter/8888 port label does
not start Jupyter or identify the process listening there; on the 4bit Pod,
8888 is the metrics relay.

## Shared configuration

Both profiles use the same configuration keys and defaults:

| Setting | Shared default |
| --- | --- |
| `MAX_MODEL_LEN` | `40960` |
| `MAX_NUM_SEQS` | `2` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `VLLM_HOST`, `VLLM_PORT` | `127.0.0.1`, `8000` |
| `CHAT_PORT` | `8001` |

`MODEL_PROFILE=4bit` is the default and retains the AWQ checkpoint and `qwen`
served alias. `MODEL_PROFILE=bf16` selects the original local checkpoint, BF16
dtype, `qwen-bf16` alias, eager execution/loading, and `CPU_OFFLOAD_GB=40`.
The offloading budget is configurable. Both profiles explicitly enable prefix
caching and retain the same gateway API, request limits, and metrics collection.

The prior 30 GiB offload configuration reported a 37,205-token GPU KV cache,
below the requested context. The accepted image therefore uses 40 GiB of CPU offloading. Its observed KV
capacity and host-memory behavior are recorded above; repeat verification when
changing the image, settings, or hardware.

Image tests check configuration; the live results above establish the present
serving configuration for the tested workloads. Two configured sequences do not establish that two simultaneous
maximum-context requests fit; record the exact GPU workload tested. Do not
silently reduce the requested context or concurrency if the trial cannot fit.

The image's environment is authoritative for serving settings. The older
`config.json`, `serve.py`, and `bootstrap.sh` describe the preceding script-based
setup; the official image uses only the checkpoint staging data from the volume.
`prepare_model.py` runs from the image, verifies the pinned checkpoint metadata,
and reuses staged files. Locks, temporary downloads, and caches remain local
because the global volume does not provide POSIX locking/atomic rename.

## One-time migration

Before changing the Pod, inspect its current settings and checkpoint, and retain
a private read-only settings snapshot for migration diagnosis. Preserve the
existing global-volume attachment, 50 GB container disk, and all existing
nonconflicting environment values. Stage a separate chat token at
`/workspace/chat/chat-token`; do not reuse the monitoring credential.

Set these non-secret environment values on BF16 Pod `ftntkktzid9f4p`:

```text
MODEL_PROFILE=bf16
MAX_MODEL_LEN=40960
MAX_NUM_SEQS=2
CPU_OFFLOAD_GB=40
GPU_MEMORY_UTILIZATION=0.90
SERVED_MODEL_NAME=qwen-bf16
BF16_ROOT=/workspace/qwen-bf16
CHAT_ENABLED=true
CHAT_DIAGNOSTICS_ENABLED=true
CHAT_HOST=0.0.0.0
CHAT_PORT=8001
CHAT_TOKEN_FILE=/workspace/chat/chat-token
METRICS_ENABLED=true
METRICS_HOST=0.0.0.0
METRICS_PORT=8889
METRICS_TOKEN_FILE=/workspace/qwen-bf16/metrics-token
JUPYTER_ENABLED=true
JUPYTER_PORT=8888
JUPYTER_SERVER_ROOT=/workspace
JUPYTER_ALLOW_ORIGIN=https://ftntkktzid9f4p-8888.proxy.runpod.net
```

`CHAT_DIAGNOSTICS_ENABLED=true` makes request diagnostics available from the
BF16 gateway. Its health response advertises `diagnosticsVersion=1`. A frontend
must also enable its own server-side `CHAT_DIAGNOSTICS_ENABLED=true` gate before
it exposes the BF16 model's independent Request diagnostics control in Advanced
settings. The Pod flag alone does not enable diagnostics in every frontend
environment. This enables the capability, not diagnostics on every
request: the user still opts in per model, and opted-in requests carry
`X-Chat-Diagnostics: 1`. Ordinary requests retain their normal streaming behavior.
The same image is used; changing this Pod environment value requires a restart
for the gateway to read it.

Retain the existing private `JUPYTER_PASSWORD`: it is Jupyter's existing token,
not a new password. The image uses it in process configuration and suppresses
token-bearing startup URLs. Its allowed browser Origin is the exact HTTPS
Runpod proxy, derived from `RUNPOD_POD_ID` and `JUPYTER_PORT` or explicitly set
with validated `JUPYTER_ALLOW_ORIGIN`. This allows cookie-authenticated API and
terminal WebSocket requests when the proxy forwards an internal Host header.
Other Origins remain blocked; token authentication and XSRF checks stay enabled. Do not print it or place it in Git. Jupyter uses the
same root directory and authentication; its installed versions are pinned to the
prior template's JupyterLab 4.6.4, Jupyter Server 2.21.1, and Notebook 7.6.3. The
4bit profile does not start Jupyter unless explicitly enabled. SSH was not
listening on the prior BF16 Pod, and this migration does not enable it.

Apply the published image digest, environment, empty container start-command
override, and HTTP ports `8888,8889,8001` in one Pod update. Retain TCP port 22.
An old start-command override would run the legacy volume bootstrap instead of
the image entrypoint and must be removed. This one-time migration configuration
is separate from normal official release rollback. Do not perform an env-only
reset followed by an image reset, or trigger another same-image deployment just
to check readiness.

The update resets the container and reloads BF16. The observed earlier global-
volume reload took about 32 minutes, so readiness verification allows 60 minutes.
The authenticated gateway starts early but reports unavailable until vLLM is
ready. Jupyter remains available during model loading. The model API stays local;
port 8001 publishes only the token-protected chat gateway. Metrics stay on 8889,
so Prometheus target identities and historical series are preserved.

## Verification

Use the existing chat checker with a privately supplied BF16 chat credential:

```sh
RUNPOD_CHAT_URL=https://ftntkktzid9f4p-8001.proxy.runpod.net \
python scripts/check-chat-deployment.py --expected-model qwen-bf16 \
  --wait 3600 --read-timeout 180
```

Also require missing/wrong chat tokens to fail, actual context to be 40,960,
reported admission capacity to be two active/six waiting, a real BF16 response,
concurrent short requests, cancellation, correct context rejection, Jupyter
login protection/access, and both Prometheus targets/GPU exporters healthy.
A short generation test does not benchmark two maximum-context generations.

## Release and rollback

The existing `prod` GitHub environment keeps its 4bit bindings
`RUNPOD_POD_ID` and `RUNPOD_CHAT_TOKEN`. Add `RUNPOD_BF16_POD_ID` and the separate
secret `RUNPOD_BF16_CHAT_TOKEN`; reuse the existing `RUNPOD_API_KEY` account
credential. BF16 selection fails closed if either binding is absent or its Pod
ID equals the 4bit Pod. Secrets never enter the image.

Build/publish a new image version on `main` through **Build and publish release**.
The verified 0.10.1 digest above is the first official BF16 baseline. No earlier
PyTorch image or unverified 0.10.0 candidate is an official BF16 release. Later
normal rollback can return to this verified baseline.

For later deploys or rollbacks, use **Deploy or roll back Runpod image**, choose
`target=prod`, `model_profile=bf16`, and the intended published image digest.
These are the workflow's only inputs. The first official BF16 baseline is
already verified, so deployments use normal versioned-image rollback: a failed
update restores the previous official image and verifies it, just as on the
4bit Pod. Existing Pod configuration, including `MAX_NUM_SEQS`, is preserved;
concurrency is managed on the Pod. Stable profile, volume, entrypoint, and ports
stay configured.
