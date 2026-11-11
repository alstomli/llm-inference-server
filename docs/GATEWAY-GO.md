# Go chat gateway

The image starts `/usr/local/bin/chat-gateway` when `CHAT_ENABLED=true`.
The gateway validates the private bearer token and trusted account ID, applies
admission and rate limits, fits context through vLLM's tokenizer API, and streams
model responses. vLLM still runs in Python on its private loopback port.

## Architecture decision

Replace the complete Python gateway with one Go service. Keep admission,
upstream cancellation, and streaming under one lifecycle owner. A permanent
Go proxy in front of a Python policy service would split ownership and make
cancellation and capacity accounting harder to verify.

Use the Go standard HTTP library and a pinned `golang.org/x/image` dependency
for WebP. The Docker builder pins Go 1.27.3, runs vet and race tests, and builds a
static Linux amd64 binary. The vLLM base image and model startup arguments remain
unchanged. This migration makes no GPU throughput or latency improvement claim.

## Configuration and trust

Existing environment variables and the token file remain the interface:
`CHAT_HOST`, `CHAT_PORT`, `CHAT_TOKEN_FILE`, `SERVED_MODEL_NAME`, `VLLM_PORT`,
`VLLM_API_KEY`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MODEL_PROFILE`,
`CHAT_DIAGNOSTICS_ENABLED`, `CHAT_IMAGE_INPUT_ENABLED`, `LORA_ADAPTER_PATH`,
`VLLM_BASE_MODEL_NAME`, `CHAT_REQUIRE_USER_ID`, `CHAT_USER_MAX_CONCURRENT`,
`CHAT_USER_MAX_WAITING`, and `CHAT_USER_RATE_LIMIT_PER_MINUTE`.

`MODEL_PROFILE` defaults to `bf16`; both gateway implementations reject other
profiles. `SERVED_MODEL_NAME` defaults to `qwen-bf16`. BF16 base and adapter
aliases remain configurable.

The bearer secret is read at process startup. Next.js authenticates the user
through Supabase and sends `X-Chat-User-Id`; the gateway accepts that identity
only under its private bearer contract. Never expose the bearer token to a
browser or forward a browser-selected identity without server verification.
The gateway does not contact Supabase itself.

Default admission is two active requests and six waiting globally, with one
active and one waiting per account. Rolling limits are 20 accepted requests per
minute globally and six per account. Waiting time counts toward the request
deadline. Accepted requests consume their rate allowance even if subsequently
cancelled. Queues and rate windows are process-local and reset on restart.
Multiple replicas do not share these limits.

The Go gateway deliberately tightens malformed-input and resource handling:
it rejects invalid UTF-8 and lone Unicode surrogates before JSON conversion,
including in otherwise ignored fields. It caps upstream JSON responses at
4 MiB and complete buffered SSE events at 1 MiB on image/diagnostic streams.
Request bodies remain capped at 640,000 bytes. Header reads are limited to five
seconds and the HTTP header budget is 16 KiB; body reads and individual
downstream writes have ten-second deadlines. JSON requests must use UTF-8.
Only the exact documented route paths are served; trailing-slash variants
return 404 instead of the reference framework's redirect.
These controls can reject malformed or stalled traffic previously accepted by
the reference. Normal frontend request and response contracts are preserved.

## Verification

From `gateway/`, using the pinned Go toolchain:

```sh
go vet ./...
go test -race ./...
go build -trimpath -o chat-gateway ./cmd/chat-gateway
```

The Python reference is retained in `scripts/chat-gateway.py`. Run the external
HTTP contract suite against the built Go binary and the Python reference:

```sh
CHAT_GATEWAY_BINARY="$PWD/gateway/chat-gateway" python scripts/test_go_gateway_contract.py
```

The runner requires Python with FastAPI, Uvicorn, HTTPX, and Pillow. It uses local
fake upstreams, disposable test credentials, and no GPU or production services.
The image contains the binary at the runner's default location. CI and release
checks also run the existing Python suite and launcher supervision tests.

The existing frontend integration runner can exercise Go by setting
`CHAT_GATEWAY_BINARY` to the compiled executable along with its usual
`CHAT_TEST_FIXTURE_PYTHON` and optional image-test settings. The backend fixture
starts the Go process against the fake model with disposable credentials.

```sh
docker build --platform linux/amd64 -t inference-go-check .
docker run --rm --entrypoint python3 --tmpfs /workspace \
  -v "$PWD/scripts:/tests:ro" inference-go-check /tests/test_go_gateway_contract.py
docker run --rm --entrypoint python3 --tmpfs /workspace \
  -v "$PWD/scripts:/tests:ro" inference-go-check /tests/test_start_server.py
```

## Rollout and rollback

Creating or merging a PR does not publish an image or restart a Pod. Follow the
existing manual release and deployment workflows, using a new immutable image
version. Before production rollout, verify authenticated readiness, streamed
text, BF16 image input and adapter modes, quota rejection, stop/disconnect
cleanup, and actual GPU capacity recovery on a controlled deployment.

The default is `CHAT_GATEWAY_IMPLEMENTATION=go`. To explicitly select the
quota-compatible Python implementation in the same image, set
`CHAT_GATEWAY_IMPLEMENTATION=python` and restart through the normal deployment
process. Invalid values refuse chat-enabled startup. Exactly one gateway starts;
do not run independent Go and Python schedulers against the same model process.
Keep `CHAT_REQUIRE_USER_ID=true` for the frontend's account quota contract.

Rollback can also use the prior immutable, quota-compatible Python image.
Keep the persistent volume and token file. Never downgrade to an image that
silently omits account enforcement. Startup continues to stop the other managed
services if any child exits.
On termination, the Go gateway immediately cancels active requests and waits up
to ten seconds for handler cleanup before forcing connections closed. It does
not finish generating outstanding replies during shutdown.
