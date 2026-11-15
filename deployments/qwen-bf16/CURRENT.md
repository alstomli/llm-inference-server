# BF16 current runtime

The retained release/profile was migrated to Pod `zwwucot0v5wy0b` on **November 15, 2026**. Authenticated gateway readiness was confirmed at **20:43:21 UTC / 15:43:21 America/New_York**, preserving the previously accepted BF16 runtime.

The eight-sequence profile was accepted on **November 14, 2026 at 20:18:19 UTC / 15:18:19 America/New_York** on the preceding Pod `66j3wfha6suokg`. The [sanitized acceptance receipt](../../docs/BATCH-EIGHT-ACCEPTANCE-2026-11-14.json) records that rollout's configuration, bounded outcomes and final recovery state; it contains no prompts, model outputs, credentials or private snapshots. Its performance observations remain dated evidence from that preceding Pod.

## Observed and retained configuration

| Item | Value |
| --- | --- |
| Pod / GPU | `zwwucot0v5wy0b` / NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Persistent global volume / mount | `cmvzszznk000007jtg17399qp` / `/workspace` |
| Release | `0.18.0` |
| Immutable image index | `zihan000/qwen-inference@sha256:0851736fd6db7a2dd6f7440fea5cf3f76ed060f74dbc47e44944f440d88e5549` |
| `MODEL_PROFILE` / `MAX_NUM_SEQS` | BF16 / 8 |
| Gateway active / waiting / accepted per minute | 8 / 24 / 80 |
| Per-account active / waiting / accepted per minute | 1 / 1 / 6, shared by base and adapter |
| Context / maximum batched tokens | 40,960 / 8,192 |
| `GPU_MEMORY_UTILIZATION` / `CPU_OFFLOAD_GB` | 0.90 / 0 (no CPU weight offloading) |
| CPU KV tier | 64 GiB budget; 68,695,359,488 native mapped bytes |
| Decode graphs | `FULL_DECODE_ONLY`, compilation mode 0, captures `[1,2,4,8]` |

Sizes 3/5/6/7 pad to the next captured bucket; the profile does not capture every size through eight. CPU KV residency and CPU weight offloading are separate settings. Source fallbacks remain **2 active / 6 waiting / 20 accepted per minute**; the retained 8/24/80 profile uses explicit overrides. Account defaults remain 1/1/6, with strict verified identity and the existing eligible-account FIFO policy.

## Release and dependency provenance

Implementation [PR #40](https://github.com/alstomli/llm-inference-server/pull/40) was merged as source commit [`3ca4caf0ba9b797a5e55ebb2fc697dfa8425d087`](https://github.com/alstomli/llm-inference-server/commit/3ca4caf0ba9b797a5e55ebb2fc697dfa8425d087). The main-only [0.18.0 release run](https://github.com/alstomli/llm-inference-server/actions/runs/43633361505) built and checked the published image before rollout.

The runtime inherits the exact preceding digest `59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d`, originally based on vLLM 0.32.0. The pinned Go builder recompiles the gateway; maintained launcher/runtime helpers overlay the inherited vendor/Jupyter stack without a pip install or mutable vendor-tag resolution. Model, adapter, context, GPU utilization, zero CPU weight offload and the 64 GiB CPU KV budget remain unchanged. Future vendor/model/adapter changes require their own reviewed release.

## November 15 migration and readiness

The new Pod retains the exact image digest, BF16 model, rank-eight adapter, 40,960-token context, eight-sequence engine, 8/24/80 gateway limits, zero CPU weight offload, 64 GiB CPU KV tier and `[1,2,4,8]` decode-graph captures. `/workspace` still contains all 18 model shards and the existing adapter and service token files. Authenticated API comparison confirmed all 30 environment entries match the original Pod except `JUPYTER_ALLOW_ORIGIN`, corrected to the new Pod's exact HTTPS proxy origin. Existing credential values were preserved.

The stale Jupyter origin failed the launcher's exact-Pod origin guard during initial startup. Updating the Pod environment automatically recreated its container; no additional explicit restart was required. During recovery, the Runpod plugin's redacted GET values could not preserve three fields in a full environment update (`JUPYTER_PASSWORD`, `METRICS_TOKEN_FILE`, `TAVILY_API_KEY`); authenticated direct API recovery restored their original values and verified the complete environment against the still-readable preceding Pod. Use an unredacted private snapshot/API response when replacing the environment map; never send redaction placeholders as configuration.

Read-only checks confirmed gateway **200** with `ready=true`, both base/adapter model registrations, native CPU KV mapping of **68,695,359,488 bytes**, the expected engine flags, and zero OOM/OOM-kill events or swap. Public authentication checks at **20:43:58 UTC** returned **200** for the gateway (8001), Jupyter (8888) and metrics relay (8889), with `gpu_exporter_up=1`. Missing/invalid credentials were rejected with **401** by gateway/metrics and **403** by Jupyter. The model engine listens only on `127.0.0.1:8000`; its unexposed public proxy returns **404**. No Jupyter kernel was created for these checks.

Two bounded functional checks through the production hosted chat UI completed successfully: one base-model request and one trained-adapter request, each with a 128-token output cap and the expected exact response. The signed-in UI showed the model online and the correct base/adapter labels. On the new metrics instance, native request-success counters increased from **0 to 2** between **20:43:40 and 20:45:37 UTC**, both with stop completions; length/abort/error/repetition counters remained zero. Final engine running/waiting queues were empty, and the sole BF16 monitoring target was healthy with `up=1` and `gpu_exporter_up=1`. These were hosted application checks; the runtime readiness checks sent no direct inference requests.

Migration verification is limited to configuration, authenticated service readiness, access controls and those two hosted functional checks. No load benchmark or profile tuning was performed. The November 14 workload timings and graph observations below were not remeasured on this Pod.

## November 14 bounded acceptance and limits of the evidence

All **32/32 scored requests completed correctly**, with no extra smoke requests. Each wave used eight distinct trusted test principals and reached eight active gateway requests; no real users or accounts were created:

| Wave | Workload | Correct / completed | Peak active |
| --- | --- | --- | --- |
| Short | 512 input tokens per request | 8 / 8 | 8 |
| Medium | 2,048 input tokens per request | 8 / 8 | 8 |
| Mixed | 4 base-text, 2 adapter, 2 vision requests | 8 / 8 | 8 |
| Near context | 40,800 input + 128 output cap + 32 safety reserve | 8 / 8 | 8 |

Near-context direct-gateway E2E peaked at **61.800 seconds**, below the frontend's 240-second application deadline. These are functional samples, not a sustained-capacity benchmark, hosted latency qualification, reliable p95 estimate or guarantee for every eight-request workload.

Native statistics contained **78 FULL unpadded-8 / padded-8 observations** associated with generated outputs. This is execution-backed evidence supporting graph replay inference; it is **not a directly instrumented replay-call count**. No additional graph instrumentation or full GPU benchmark was run for this acceptance.

Both **2/2 direct gateway cancellation probes passed**, including resource recovery. Frontend [deadline support (#27)](https://github.com/alstomli/qwen-chat/pull/27) enforces 240 seconds from route entry with shorter requested/web limits. The [chat-only platform opt-in (#28)](https://github.com/alstomli/qwen-chat/pull/28) was verified through exact-source READY deployments and official local resolver/Lambda emission of `supportsCancellation: true`; no hosted browser-disconnect generation probe was added. Direct gateway cancellation and hosted configuration verification are distinct evidence.

Final checks observed authenticated metrics **200**, unauthenticated metrics **401**, a healthy GPU exporter, empty engine queues and zero native transfer-usage gauges. Remaining RAM was **53.45 GiB**, shared memory available **23.57 GiB**, with no OOM/OOM-kill events or swap. Zero transfer-usage gauges describe idle transfers, not empty persistent CPU cache residency. The independent watchdog was disarmed, its exit verified, and the accepted profile retained **without rollback**.

## Prior dated state and operational entry points

The [November 13 snapshot](https://github.com/alstomli/llm-inference-server/blob/3ca4caf0ba9b797a5e55ebb2fc697dfa8425d087/deployments/qwen-bf16/CURRENT.md) remains a historical record of preceding Pod `66j3wfha6suokg` on release 0.17.0 / digest `59ff84…`, two sequences and captures `[1,2]`, with the same context, GPU utilization and CPU KV/weight-offload settings. Its implementation was `7b10046`, with the dated report at `82fe9a9` in [#37](https://github.com/alstomli/llm-inference-server/pull/37). The earlier [#36](https://github.com/alstomli/llm-inference-server/pull/36) residency experiment is not current image provenance.

- [Batch-eight controls and rollout procedure](../../docs/BATCH-EIGHT-ROLLOUT.md).
- [Repository operations](../../README.md), [Go gateway and Python fallback](../../docs/GATEWAY-GO.md), and [checkpoint staging](prepare_model.py).
- [First official image record](VLLM-IMAGE.md), [earlier RTX PRO 6000 migration](RTX6000-MIGRATION.md), [original template measurements](README.md), and [LoRA setup](LORA-SERVING.md) retain historical/setup context.

An image-only rollback to **0.17.0** while retaining `MAX_NUM_SEQS=8` fails that release's two-sequence graph startup guard. Restore the prior digest **and** `MAX_NUM_SEQS=2`, with gateway overrides restored to **2/6/20** or removed to recover their prior defaults. Preserve the CPU KV/weight-offload and graph flags from the fresh rollback snapshot; do not disable features ad hoc to bypass the guard. The image-only deployment workflow preserves environment settings and cannot perform this profile restoration by itself.

Use private current service bindings rather than copying old Pod origins. Future rollback requires a fresh approved snapshot and reviewed immutable image/configuration; do not reuse the completed rollout's lease or safety files. This file records the approved Pod migration; its tracked changes are documentation only.
