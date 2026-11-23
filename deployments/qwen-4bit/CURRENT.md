# 4bit current runtime

Accepted **November 20, 2026** on its own Pod (`kdm8qghqezo72e`, since replaced; see [Pod migration](#pod-migration)) by the profile acceptance harness (`scripts/acceptance/acceptance.py --profile 4bit`). All five waves passed at concurrency 6; the near-context wave failed at 7. Raw results stay in the git-ignored `acceptance-results/`; this page holds the sanitized record. No prompts, model outputs or credentials are recorded.

## Configuration

| Item | Value |
| --- | --- |
| Pod / GPU | `stfd3sdb8furvq` / NVIDIA RTX PRO 4500 Blackwell (32 GB), 1 GPU, EU-CZ-1 |
| Volume | global network volume `melodic_lavender_mandrill` at `/workspace` |
| Release | `0.22.0`, built from [`611fcaf`](https://github.com/alstomli/llm-inference-server/commit/611fcaf5669ddcdd21fd547bf18e03a93e11d219) by the [release run](https://github.com/alstomli/llm-inference-server/actions/runs/44677730887) |
| Image | `zihan000/qwen-inference@sha256:76d381db56d038f79745f5f36a6f1a2cbaae3d66e922be6a75a4ca8e2c23ec06` |
| Checkpoint | `RedHatAI/Qwen3.8-27B-INT4` at `71dface38d4cbfa34121f1d446d4a0fa1025f13c`, compressed-tensors W4A16 with vision tower ([ADR 0001](../../docs/adr/0001-int4-w4a16-for-4bit-profile.md)) |
| `MODEL_PROFILE` / alias | `4bit` / `qwen-4bit`, no adapter |
| `MAX_NUM_SEQS` | 6 |
| Gateway active / waiting | 6 / 18 (derived from `MAX_NUM_SEQS`) |
| Gateway accepted per minute | 50 (`CHAT_RATE_LIMIT_PER_MINUTE`) |
| Context / KV cache | 40,960 tokens / `--kv-cache-dtype fp8`, prefix caching on |
| `GPU_MEMORY_UTILIZATION` / CPU weight offload | 0.90 / none |
| Image input, diagnostics | on |
| Web search | on for `qwen-4bit` (`qwen3_xml` parser, Tavily key from the `prod` environment) |

The Pod's settings come from its environment, not this page: concurrency is set with the deploy script's `DEPLOY_MAX_NUM_SEQS`, which also drops explicit gateway slot overrides so the gateway follows the engine.

## Acceptance

Each wave sends exactly `concurrency` known-answer requests at once through the public gateway and checks answers, headers, exact token counts and resources.

| Concurrency | short | medium + cancellation | mixed (text + vision) | near-context | tool (web search) |
| --- | --- | --- | --- | --- | --- |
| 5 | pass | pass | pass | pass: first text p50 61 s, end max 100 s | pass, 5 sources |
| **6** | **pass** | **pass** | **pass** | **pass: first text p50 72 s, end max 121 s** | **pass, 5 sources** |
| 7 | pass | pass | pass | **fail**: 1 of 7 streams ended with no data after 127 s | pass |

Near-context requests carry 40,800 input tokens, a 128-token output cap and a 32-token safety reserve. At 6 the slowest finished in 121 s, under the frontend's 240-second deadline. The fit test (#50) measured KV-cache room for about 5.09 full-context sequences, so six simultaneous full-context requests cannot all be resident at once; the longer first-text times at 6 and 7 are consistent with the engine waiting for KV-cache space, though this run did not record engine queue metrics.

Resources at the start of each run: GPU memory 28.6 of 32.6 GB, container RAM about 25 of 58 GiB, no OOM events or swap.

## Web-search verification

Before enabling web search, `qwen-4bit` was checked against the Pod's private vLLM API with the gateway's exact tool schemas and `enable_thinking: false`. All checks passed: automatic `search_web` selection for a current-price question, no tool call for a stable question, a forced named `search_web` call, a `read_page` call with `sourceId: "S1"` after an assistant/tool round trip, at most one call with `parallel_tool_calls: false`, tokenizer counts that include tools (426 vs 83 tokens), `usage.completion_tokens` on every response, and a streamed final answer citing `[S1]` with tools omitted. Failure handling (malformed calls, provider errors, empty results, cancellation) is gateway behavior shared with BF16 and was not re-verified here; the acceptance `tool` wave exercised a live search end to end.

## Differences from BF16

- The vision row of the mixed wave asks only about the image. The 4-bit model kept following an earlier "reply only 4" instruction when the image question was appended after it, while a plain image question was answered correctly; the wave tests vision, not instruction overriding.
- Requests to this Pod must not send `useAdapter`: the gateway answers 400 on a profile without an adapter (the frontend already omits it).
- `deployments/qwen-4bit/config.json` is not needed on the volume; the pinned profile names its model (#66).

## Deployment history

- [Staged web search](https://github.com/alstomli/llm-inference-server/actions/runs/44682124082) and [enabled it for `qwen-4bit`](https://github.com/alstomli/llm-inference-server/actions/runs/44682827432) on 0.22.0; both passed the live checks.
- An [earlier staged deploy](https://github.com/alstomli/llm-inference-server/actions/runs/44680097781) rolled back automatically because Runpod v2 omits empty settings from its environment report (#68); rollback passed its checks.
- Concurrency 6 and 7 were applied with the deploy script; the Pod was returned to 6 after the 7 run.

## Pod migration

On November 20, 2026 the 4bit workload moved from `kdm8qghqezo72e` to `8l7n75h7m4tma8` because the old Pod's machine was no longer available. The new Pod has the same GPU type, data center, ports and global network volume, so the model, chat token and metrics token carried over on the volume. Its image and environment were copied from the old Pod (the new Pod keeps its own Jupyter password and SSH key) and it passed the live chat check at 6 active / 18 waiting and the metrics relay check. The acceptance figures above were measured on the old Pod with the same image and settings.

Updated with the move: `prod` variable `RUNPOD_4BIT_POD_ID`, the Railway Prometheus `RUNPOD_METRICS_TARGETS` (the new target reports up), and the live Grafana Pod picker, which now lists every vLLM Pod and defaults to this one. The old Pod is stopped, not terminated. The hosted qwen-chat on Vercel had no 4bit gateway binding before the move, so the frontend is set up for this Pod rather than moved to it.

On November 23, 2026 the 4bit workload moved from `8l7n75h7m4tma8` to `stfd3sdb8furvq`. The new Pod has the same GPU type, data center, ports, global network volume, image and settings. It passed the live chat check at 6 active / 18 waiting and the metrics relay check. The acceptance figures above were not re-measured.

Updated with this move: `prod` variable `RUNPOD_4BIT_POD_ID`, the Railway Prometheus `RUNPOD_METRICS_TARGETS` and the live Grafana Pod picker default. The old Pod is stopped, not terminated. The hosted qwen-chat `RUNPOD_4BIT_CHAT_URL` on Vercel (Production and Preview) was still pending when this note was written.
