# Concurrent text and image benchmark

`scripts/benchmark-chat.py` compares one text request, two simultaneous text
requests, and simultaneous text plus image requests against the **same BF16
gateway and base weights**. It changes no server configuration. By default it
warms the text and image paths sequentially, then runs the three scenarios once
(seven total completions). Repetitions are separated by a 65-second cooldown.
The prompts and output budgets are identical; actual completion lengths may vary.

## Run

Use an already-enabled image-capable gateway with diagnostics
(`CHAT_DIAGNOSTICS_ENABLED=true`) and two concurrent slots. The script checks
these capabilities before sending completions. It does not turn them on or
restart the Pod. It bypasses the website and measures the client-to-gateway
path; it cannot measure browser upload, Vercel or UI rendering overhead.

Set `RUNPOD_BF16_CHAT_URL` and `RUNPOD_BF16_CHAT_TOKEN` privately in the process
environment. Credentials are never accepted on the command line or included
in reports. Provide an existing representative image within the gateway's byte,
dimension and format limits (currently 256 KiB and 2,048 pixels per edge).
File size is checked locally; the gateway checks actual decoding and dimensions.
Use the same image when comparing configurations and note its dimensions
separately. No generated image is needed.

```powershell
python scripts/benchmark-chat.py --image D:/AILearn/images-in-chat/example.png --output benchmark-results/baseline.json
```

Choose an existing image; `example.png` is a placeholder. Reports never overwrite
an existing file, and `benchmark-results/` is ignored by Git. The script uses
only Python's standard library and requires Python 3.10 or later.

The default two fixed benchmark UUIDs represent separate trusted service
principals. Reuse them between runs; do not rotate identities to bypass quotas.
`--user-a` and `--user-b` can select two distinct authorized principals, sharing
their normal account budgets. Defaults consume five requests for A and two for
B with warmup, within the gateway's default six requests per account per minute
when those principals have no other traffic. Lower configured limits can still
reject the run. Failures stop the matrix, save partial results and exit nonzero;
there are no automatic retries that could hide rate limits. Allow the rate
window to clear before rerunning.

`--repeats 3` collects more samples with cooldown between rounds.
`--skip-warmup` captures an unwarmed sequence instead; record that distinction.
`--cache-mode isolated` (default) requests fresh prefix-cache salts to avoid
giving later identical text prompts a prefix-cache advantage. This does **not**
clear multimodal processor/encoder caches. `--cache-mode reuse` measures the
normal prefix-reuse path. Compare reports using the same cache mode. A base
model selection is verified on every response when adapter switching exists;
the benchmark does not mix adapter text with base-model image inference.

## Interpret the report

Each request includes:

- `ttft_seconds`: client request start to first nonempty text content. Role-only
  SSE events do not count. Includes network, admission, tokenization and inference.
- `completion_tokens`: actual upstream usage, never estimated from SSE chunks.
- `output_tokens_per_total_second`: completion tokens divided by request start
  to `[DONE]`, including TTFT.
- `post_ttft_tokens_per_second`: completion tokens divided by first content to
  `[DONE]`. This is an **approximate decode rate**, including final stream
  overhead and all tokens in the first chunk; it is not a per-token latency or
  exact GPU decode speed. Compare sufficiently long outputs of similar lengths.
- `gateway_timings_ms`: `contextFitMs` covers the context-fitting/tokenization
  loop; `upstreamFirstContentMs` covers completion submission to first model text;
  `gatewayFirstContentMs` covers gateway entry to first text. These do not fully
  separate queueing, image validation, CPU processing and GPU encoding.
- `diagnostics`: allowlisted model, input count, output budget and cache mode.

Concurrent jobs use separate threads synchronized immediately before their
requests. Client lifetime overlap and start/end offsets are recorded, but these
**do not prove simultaneous GPU scheduling**. Correlate with existing vLLM
running/waiting requests, prefill/decode timing, GPU utilization, KV-cache use and
preemptions. The health snapshot includes starting admission counts; background
traffic can still affect results. Run controlled comparisons during a quiet period.

If the text request slows primarily in the mixed scenario, investigate shared
image processing/encoding and prefill contention. If both concurrent scenarios
slow similarly, investigate general shared GPU capacity and scheduling. Increased
`contextFitMs` points toward preparation before inference; increased
`upstreamFirstContentMs` points toward the completion path. These are clues,
not proof of a particular cause. GPU scheduler settings, CUDA graphs and image
resolution should only be changed after collecting a baseline.

Reports omit credentials, endpoints, principal IDs, prompt/response text and
image data. HTTP failures record only status codes; no raw response bodies or
exception URLs are saved. Incomplete streams, missing usage, stream errors and
missing finish reasons are failures. Socket `--timeout` bounds individual
blocking operations, not the entire benchmark wall time. Ctrl+C stops the run
and saves completed scenarios after in-flight threads return.

## Offline validation

```powershell
python scripts/test_benchmark_chat.py
```

Tests exercise token accounting, SSE framing and errors, concurrent loopback
HTTP requests, stable distinct principals, base-model consistency, report privacy,
preflight failures and redirect refusal. They use fake model outputs and establish
no live GPU performance result.
