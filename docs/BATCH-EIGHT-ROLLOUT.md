# Batch-eight serving profile and rollout

This is the approved target configuration and release procedure, not a claim that source integration has already changed the running service. The [dated runtime snapshot](../deployments/qwen-bf16/CURRENT.md) retains the preceding two-sequence deployment record.

| Setting | Source fallback | Explicit production target |
| --- | --- | --- |
| `MAX_NUM_SEQS` | 2 | 8 |
| `CHAT_MAX_CONCURRENT` | min(2, engine) | 8 |
| `CHAT_MAX_WAITING` | 6 | 24 |
| `CHAT_RATE_LIMIT_PER_MINUTE` | 20 | 80 |
| `CHAT_USER_MAX_CONCURRENT` | 1 | 1 |
| `CHAT_USER_MAX_WAITING` | 1 | 1 |
| `CHAT_USER_RATE_LIMIT_PER_MINUTE` | 6 | 6 |

Base/adapter modes share each account's quota. Keep strict identity enforcement, eligible-account FIFO, per-account isolation, cancellation cleanup and the existing web/vision contracts. Queue wait remains part of the request timeout. These limits describe admission, not a guarantee that eight maximum-context requests fit in GPU memory.

Keep `MAX_MODEL_LEN=40960`, `GPU_MEMORY_UTILIZATION=0.90`, `CPU_OFFLOAD_GB=0`, `KV_OFFLOAD_CPU_GIB=64` and `CUDA_DECODE_GRAPHS=true`. The eight-sequence graph profile uses `FULL_DECODE_ONLY`, compilation mode zero, captures `[1,2,4,8]`, and native padding for sizes 3/5/6/7. The existing two-sequence profile retains `[1,2]`. Unsupported graph profiles or nonzero CPU weight offload fail closed; CPU KV offload retains the existing RAM/shared-memory reserve guards.

## Exact image dependency boundary

The normal Dockerfile retains the digest-pinned Go builder and rebuilds the gateway from reviewed source. Its runtime base is `zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d`, the previously deployed stack originally based on vLLM 0.32.0. It overlays the Go binary and all maintained launcher, model-preparation, adapter-validation, Jupyter, metrics, Python fallback and KV helper files. There is no runtime pip install or mutable vendor-tag resolution. A future vendor/dependency refresh requires a separate reviewed base update. CI and main-only releases use this same Dockerfile; the historical derivative Dockerfile is not selected by a branch exception.

## Controlled rollout order

1. Pass exact-head image CI and frontend deadline tests; merge reviewed source. Publish a new unused immutable version through the main-only release workflow. Verify image provenance and inherited vendor/Jupyter versions against the pinned base before rollout.
2. Deploy the frontend's 240-second Vercel application deadline first. It starts at route entry, covers auth/validation/readiness/queue/generation, respects shorter requests and web deadlines, and propagates aborts. Its platform route limit remains 300 seconds for headroom. Local custom timeouts remain available.
3. Take a fresh private configuration/image/volume/model/adapter snapshot and bounded recovery lease with an independent rollback owner. Preserve all unrelated environment, credentials and service bindings. Parent operations owns this one controlled deployment; merging source alone does not restart or change a Pod.
4. Deploy the reviewed immutable image and explicit target settings above. Verify actual vLLM sequence/capture arguments, memory reserves, strict authenticated health/admission, base and adapter streams, vision and web capability contracts, cancellation and idle recovery. Live acceptance is limited to the approved two cancellation probes and 32 scored requests from eight users; queue-full/rejection coverage remains in local fixtures. This is a correctness check, not a full GPU capacity benchmark.
5. If acceptance fails, restore the fresh prior digest and exact configuration, verify service recovery and idle metrics, then disarm recovery controls. On success, record the deployed digest/effective settings and retain the new profile only after the independent owner verifies cleanup.

Do not reuse archived snapshots, replay old leases, mutate hosted SQL, change models/adapters or silently alter gateway defaults. Historical benchmark reports and measured evidence remain unchanged.
