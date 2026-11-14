# KV offload and CUDA graphs enabled — November 13, 2026

Both features were enabled and left running on the existing BF16 pod after a
bounded base-model proof and a clean final restart. The final service uses the
native vendor worker, with no proof middleware, replay callback, control socket
or temporary proof environment.

| Setting | Final value |
| --- | --- |
| Native CPU KV budget | 64 GiB; 68,695,359,488 bytes mapped in 1,337 whole chunks |
| Cache groups | Three Mamba groups and one full-attention group; LBHNC; 784-token alignment |
| CUDA graphs | Compilation mode 0; FULL_DECODE_ONLY; capture sizes 1 and 2 |
| Production controls | BF16, context 40,960, maximum two sequences, GPU utilization 0.90, weight offload 0 |
| GPU cache | Automatic allocation; 671 blocks / 465,833 tokens after the final restart |
| Runtime image | 0.17.0, pinned to `zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d` |

The image derives from the exact previous production digest and changes only
the startup launcher and optional helpers. Implementation source:
`7b10046a6e3508bd8ba7dffa2a3dc9b0f8d59a69`. Model, LoRA/vision allocations, gateway limits and
unrelated environment values stayed unchanged.

The proof used the same 3,200-token prompt and cache salt for GPU reuse and CPU
restoration. GPU reuse hit 3,136 device tokens. A one-use local Unix operation
reset only idle GPU prefix state while preserving the CPU connector. The next
request hit 3,136 host tokens, loaded exactly 359,464,960 bytes, and computed
only the 64-token suffix. Both returned the expected answer. A different salt
produced a cold miss. All six scored proof responses passed, including the
single two-request wave; 539 completed native graph replay calls were
observed during requests, with 31 calls in the two-request wave. Standard
telemetry independently showed FULL dispatch at batch sizes one and two.

The final clean startup removed both proof environment variables, captured
native graphs again, passed two concurrent base-model correctness streams,
and reported FULL batch-two dispatch. Authenticated health and metrics,
unauthorized metrics rejection, gateway admission/account quotas, context,
LoRA/image capabilities, empty queues, zero active transfers and the native
CPU mapping all passed. At final verification, RAM reserve was
57,374,945,280 bytes and shared-memory reserve was
25,304,637,440 bytes; no OOM or swap growth occurred. The external
watchdog was disarmed only after these checks.

Both the full implementation image CI and pinned derivative release passed;
the release ran 121 CPU tests. Private rollback transition and monitoring-outage
regressions also passed. No adapter, image or web inference and no full capacity
load test were submitted. These results establish enablement and bounded
correctness, rather than a throughput or latency improvement.

[Numerical evidence](KV-GRAPH-ENABLEMENT-2026-11-13.json) ·
[Configuration and recovery design](KV-GRAPH-ENABLEMENT.md) ·
[Pull request #37](https://github.com/alstomli/llm-inference-server/pull/37)
