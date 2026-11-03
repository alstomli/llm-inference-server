# BF16 migration to RTX PRO 6000

The original unquantized checkpoint is served by RTX PRO 6000 Blackwell Server
Edition Pod `o6f2qoctk798tq`. CPU weight offloading is disabled. The old RTX PRO
4500 SE Pod `ftntkktzid9f4p` is the retained migration rollback target; it is
stopped only after the new server and its application connections are verified.

## Runtime configuration

Reuse the first official image without rebuilding it:

```text
zihan000/qwen-inference@sha256:f0f6c699b2075ed714e67860c5afe26c411dc287ec957eb33382765f54ac25a7
```

Attach global volume `spicy_coffee_meerkat` at `/workspace`. Keep the existing
checkpoint, manifest, chat token file and metrics token file. Startup validates
and reuses the cached 18 BF16 weight shards; do not run the historical bootstrap.
The image entrypoint owns startup; leave the Pod start-command override empty.

| Setting | Value |
| --- | --- |
| `MODEL_PROFILE` | `bf16` |
| `CPU_OFFLOAD_GB` | `0` |
| `MAX_MODEL_LEN` | `40960` |
| `MAX_NUM_SEQS` | `2` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `SERVED_MODEL_NAME` | `qwen-bf16` |
| `BF16_ROOT` | `/workspace/qwen-bf16` |
| `CHAT_DIAGNOSTICS_ENABLED` | `true` |
| `JUPYTER_ALLOW_ORIGIN` | `https://o6f2qoctk798tq-8888.proxy.runpod.net` |

Other service settings and credentials are retained from the prior BF16 Pod.
The explicit offload value is important: the shared image's BF16 default is
still 40 GiB for compatibility with the smaller Pod. Pod environment overrides
that default. No weight quantization is introduced. Eager execution, checkpoint
revision and loading strategy remain unchanged for this hardware migration.

Keep vLLM private at `127.0.0.1:8000`. Expose authenticated chat on HTTP 8001,
authenticated metrics on HTTP 8889 and authenticated Jupyter on HTTP 8888.
Container disk remains 50 GB. Do not expose the raw vLLM port.

## Cutover

After readiness, authentication, generation and token-latency checks pass:

1. Update only the BF16 connection in Vercel Production and local frontend env
   files to `https://o6f2qoctk798tq-8001.proxy.runpod.net`; retain its existing
   bearer token. Redeploy/restart those frontend runtimes to load the new URL.
2. Set GitHub environment `prod` variable `RUNPOD_BF16_POD_ID=o6f2qoctk798tq`.
   Existing BF16 token and 4bit deployment variables remain unchanged. Future
   deploy/rollback workflow runs target the new Pod and preserve its zero-offload
   environment override.
3. Replace the old BF16 hostname in Prometheus `RUNPOD_METRICS_TARGETS` with
   `o6f2qoctk798tq-8889.proxy.runpod.net`. Retain the 4bit hostname, credentials,
   startup script, retention settings and `/prometheus` data volume. Redeploy
   Prometheus. The new instance has its own time series; old history is retained.
4. Verify both current scrape targets and GPU exporters are up, both frontend
   model status cards are ready, and a real hosted BF16 reply succeeds.
5. Stop the old BF16 Pod. Keep the Pod definition and global volume; do not
   terminate it or delete data. The 4bit deployment is not changed.

The UI model name remains `Qwen3.8-27B-bf16`; selecting BF16 now routes to the new
hardware. Browser conversation selection and per-model settings remain valid.

## Rollback

Before reverting routing, start the retained old BF16 Pod and verify its original
offload configuration is healthy. Restore the frontend BF16 URL, GitHub BF16 Pod
ID and Prometheus BF16 target to `ftntkktzid9f4p` (chat 8001, metrics 8889).
Redeploy/restart affected frontends and Prometheus, verify a reply and scrapes,
then stop the new Pod if required. Keep all tokens and the shared volume.
Image-only rollback on the new Pod continues to use the normal digest workflow.

## Verification

Verified November 3, 2026 (America/New_York). The image digest was confirmed in
Runpod after the settings reset. The actual vLLM process command contained
`--cpu-offload-gb 0`, and the running environment retained context 40,960 and
two sequences. Jupyter login with the existing credential and an actual terminal
command succeeded on the new hostname.

| Check | Observed result |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell Server Edition; 97,887 MiB total VRAM |
| Weights | Original revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 18 cached shards; 1,199 BF16 tensors |
| Weight loading | 390.59 seconds; model GPU memory 51.01 GiB |
| KV cache | 32.34 GiB; 468,610 token capacity; engine estimate 11.44 contexts of 40,960 tokens |
| Authentication and limits | Correct chat token accepted; missing/wrong token rejected; excess context rejected with HTTP 400 |
| First synthetic generation after startup | Completed with `stop`; first content 34.756 seconds (cold request) |
| Subsequent 128-token latency test | Engine mean ITL 39.733 ms across 127 observations; client first content 0.401 seconds |
| Concurrency | Two simultaneous active requests observed; both 128-token replies completed |
| Local frontend | Both model cards ready; real BF16 reply completed through the migrated connection |
| Hosted frontend | Both cards ready; opted-in diagnostic reply completed with 53 input and 14 output tokens |
| Hosted timings | First content at browser 695.7 ms; completed in 1,198.9 ms; gateway total 612.397 ms |
| Routing attribution | New Pod's successful `stop` counter increased 2 to 3 and generated tokens 400 to 414; old Pod counters unchanged |
| Monitoring | Current 4bit/new BF16 targets and both GPU exporters healthy after Prometheus redeploy |
| Old Pod | Stopped after successful hosted cutover; retained with global volume for rollback |

The 128-token latency test followed one cold generation and used engine metric
counter differences around that single request. It is not a controlled comparison
with the user's earlier approximately two-second ITL, nor a maximum-context or
high-concurrency benchmark. About 25 decode tokens/second is the reciprocal of
the measured mean ITL. Two maximum-context generations were not tested.

Post-test VRAM was 87,877 / 97,887 MiB, including preallocated KV cache. Container
RAM was 61,338,382,336 bytes, with peak 68,484,599,808 and limit 187,999,997,952.
Memory-cap pressure, OOM and OOM-kill counters were all zero. Idle engine running
and waiting request counts were both zero.

Vercel Production deployment `5v9YwUbMFkRnMX5FmjXgBX4UF9U7` redeployed the
existing reviewed application source `b89fee635cc65d0a658ac1384cc3b1b7c7d89d52`
with the updated BF16 URL; build duration was 26 seconds. Only that Production
binding was changed, and the existing token was retained. The local frontend's
original checkout and active production build directory each received only the
BF16 URL change. Diagnostics was restored to off in the hosted session after the
test; numeric settings were retained. The 4bit runtime and its connections were
not modified.

The sanitized measurements are in [rtx6000-verification.json](rtx6000-verification.json).
Startup/resource evidence and a copy of this guide are also retained under
`/workspace/qwen-bf16/` on the global volume. Client stream-event gaps are recorded
separately from engine ITL because an SSE event need not equal one token.
