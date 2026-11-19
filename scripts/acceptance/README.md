# Profile acceptance

Run this from your workstation against a serving Pod before it takes user traffic. It sends known-answer request waves through the public gateway (`:8001`) and checks the engine, gateway and resources through the RunPod exec API. It never changes the Pod's configuration.

It replaces the one-off `.tools/batch8-rollout-20261114` harness. The live-Pod lease, watchdog, snapshot and rollback parts were dropped: a new profile is accepted on its own idle Pod, and rolling back means pointing traffic at another profile.

## Prerequisites

- `RUNPOD_API_KEY` in the repo `.env`.
- The gateway token in `.secrets/<profile>-chat-token`, or pass `--token-file`.
- `aiohttp`, which is imported by `scripts/benchmark-capacity.py` (the harness uses its SSE parser and percentile helper).

## Run

```powershell
python scripts/acceptance/acceptance.py --profile 4bit --pod <POD_ID> --concurrency 4 all
```

`all` runs these steps in order:
- builds a tokenizer-exact prompt corpus on the Pod, with no inference
- records a baseline status
- runs the waves `short`, `medium` (with queued and active cancellation probes), `mixed`, `nearcontext` and `tool`
- writes `acceptance.json`

Results go to `acceptance-results/<profile>-<UTC>/`, which is git-ignored. To rerun a single wave against an existing corpus, pass the same `--run-dir`.

| Check | `bf16` | `4bit` |
|---|---|---|
| Base alias | `qwen-bf16-base` | `qwen-4bit` |
| Adapter rows in the mixed wave | yes (`qwen-bf16`) | none, `--enable-lora` must be absent |
| KV cache dtype | default | `--kv-cache-dtype fp8` |
| RAM / shm headroom | 32 / 8 GiB | 8 / 4 GiB |

To pick `--concurrency` for a new profile, start low and raise it. The highest value whose `nearcontext` wave passes at the full `--context` is the profile's concurrency.

## Test

```powershell
python scripts/acceptance/test_acceptance.py
```
