# Private vLLM continuous-batching benchmark

`scripts/benchmark-capacity.py` measures the private base-text engine on the
inference host. It bypasses the authenticated gateway. Its results do not change
or describe website admission quotas, per-account limits, image input, tools,
LoRA inference, or public network performance.

Install `aiohttp` for the runner and tests. The CLI also needs the deployment's
existing `transformers` and tokenizer files; it never downloads model files.

```sh
python3 scripts/benchmark-capacity.py --mode continuous \
  --name medium-c8 --workload medium --concurrency 8 \
  --duration 120 --wall-seconds 300 --request-deadline 180 \
  --max-requests 1024 --corpus-size 1024 --seed 12345 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --runtime-fingerprint YOUR_VERIFIED_RUNTIME_HASH \
  --output benchmark-results/capacity
python3 scripts/benchmark-capacity.py --analyze --output benchmark-results/capacity
python3 -m unittest discover -s scripts -p test_benchmark_capacity.py -v
```

Only credential-free HTTP loopback origins are accepted. The optional upstream
API key is read from `VLLM_API_KEY`; it is never written to artifacts. Run names
are unique and existing request/sample files are never overwritten. Raw results
belong under the already ignored `benchmark-results/` directory. Artifacts
contain numerical measurements and configuration, without prompts or responses.

The default continuous mode immediately replaces each completed request up to
the selected concurrency. `--arrival-rate` switches to seeded Poisson arrivals:
scheduling continues independently of completion, a separate bounded client
backlog holds excess work, and overflow becomes an explicit `client_drop` outcome.
The hard submitted-request maximum is 1024, including requests waiting inside
vLLM. Client backlog is separately limited to 1024. This does not measure an
unbounded server queue or authorize raising production gateway limits.

The short, medium, and long workloads use exactly 512/128, 2048/256, and
8192/512 input/output tokens. Mixed workload selection uses weights 50/35/15.
Synthetic text is seeded and tokenized before timing, with the installed chat
template and thinking disabled. The exact generation suffix is retained. Unique
early user prefixes and a verified empty prefix cache between scenarios isolate
cross-request reuse. The runner sends token IDs to `/v1/completions` so input
length is exact. Forced-length mode sets `ignore_eos` and `min_tokens`; the usage
frame must confirm exact input/output counts. `--natural-stop` is a separate
validation mode, not evidence about forced-length throughput.

TTFT and end-to-end latency begin at scheduled arrival, including client queue
wait. Engine submission time is recorded separately. Each request has a true
wall deadline of at most180 seconds from scheduled arrival. Stage deadlines
cancel remaining work and preserve partial reports. Every scheduled request
must have an outcome. HTTP errors, malformed/truncated SSE, missing usage,
token-count mismatches, drops, deadlines, and cancellations cannot count as
success. After at least20 outcomes, over5% failures stop the scenario.

Output throughput uses successful output tokens divided by total scenario time,
including final drain; arrival-window throughput is also reported. Streaming
usage counts tokens, not chunks. Average time per output token is the observed
first-to-last content-chunk interval divided by usage output tokens minus one;
it is approximate because chunks can contain multiple tokens. Chunk-gap
percentiles are separate measurements. Periodic samples capture engine running
and waiting counts, KV occupancy, preemptions, GPU utilization/VRAM, host CPU,
and client backlog. GPU power/clocks/throttling need separate runtime evidence.

Analysis groups by runtime fingerprint, sequence ceiling, and batched-token
budget. It excludes warmups, natural-stop validation, staggered pairs, sustained
open-loop runs, and tuning comparisons from homogeneous capacity estimates.
The knee is the smallest tested concurrency within5% of the best successful
output throughput in that group. Chat capacity requires both short and medium
at the same concurrency and exact settings, each with at least100 outcomes,
success at least99%, p95 TTFT at most5s, and approximate p95 TPOT at most100ms.
Smaller samples are provisional. A largest passing value is a tested lower bound
unless a failing boundary was observed.

Before any exclusive live test, capture exact original configuration privately,
verify an independent external restoration watchdog, and reserve recovery time.
The deployment launcher supervises vLLM: killing its model child stops all
services. Any temporary alternate supervisor must own and stop load generation
before engine restoration, preserve image/model/volume/environment, provide
ordinary ingress isolation, and enforce an absolute self-expiring lease.
Restore only fields actually changed, verify complete configuration equality,
then verify original runtime settings, health, metrics, base/adapter streams,
gateway quotas, and absence of benchmark processes or queued requests.
