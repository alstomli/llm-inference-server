# Bounded web search agent (disabled by default)

The Go chat gateway owns a request-local tool loop. It asks the selected model
whether to call `search_web({query})` or `read_page({sourceId})`, executes those
tools through a replaceable `WebProvider`, and streams a separate final answer.
The first provider is Tavily. No database, page cache, raw page logging, or
production provisioning is added. Search queries and selected page URLs are
sent to Tavily; snippets and extracted text are sent to the private model.

## Capability gate and compatibility status

**The deployed Qwen models, adapters, chat templates and tool parsers have not
been tested by this change. Keep web search disabled until controlled serving
verification succeeds.** Local HTTP fixtures verify the OpenAI wire contract,
not model quality or actual parser compatibility. Current deployment defaults
remain unchanged. This change does not deploy, publish an image, or call models.

Enable only after verifying the exact served alias with:

- `CHAT_WEB_SEARCH_ENABLED=true` (default false)
- `TAVILY_API_KEY` on the gateway server only
- `CHAT_WEB_SEARCH_MODEL_VERIFIED=true` (default false)
- `CHAT_WEB_SEARCH_VERIFIED_MODEL` equal to the exact selected served alias
- `CHAT_WEB_SEARCH_TOOL_PARSER` equal to the parser verified with that alias
- `CHAT_GATEWAY_IMPLEMENTATION=go` (the existing default)

The launcher adds `--enable-auto-tool-choice` and the explicitly selected
`--tool-call-parser` only when enabled and verification settings are present.
It never guesses a parser from a Qwen model name. The gateway gates each selected
alias separately, so verifying an adapter does not verify its base model.

Compatibility verification must cover non-streaming automatic tool selection,
a named forced `search_web` call, `read_page` with an assistant/tool round trip,
`parallel_tool_calls:false`, exact tokenizer support for messages **and tools**,
`/tokenize` prompt counting, and final SSE content with tools omitted. Non-stream
responses must expose `usage.completion_tokens` including all generated tokens
(including reasoning, if serving semantics include it). Planning rejects missing
or impossible usage. Every call explicitly keeps the existing gateway's
`enable_thinking:false` setting; reasoning-mode changes need separate validation.

Use the exact vLLM image, tokenizer/template revision, model and adapter intended
for rollout. Verify malformed calls, provider failure, empty results, context
exhaustion, output caps, stop/disconnect and queue cancellation. Record evidence
before setting verification flags. Verification is an operator assertion, not a
runtime proof: health does not make paid search or production generation probes.

## API version 1

An authenticated `GET /health` adds `webSearchVersion:1`,
`webSearchAvailable:boolean`, and `webSearch:{provider:"tavily",reason,limits}`.
Reasons are `disabled`, `missing_key`, `unverified_model`, or `ready`.
The health capability describes the default served alias. Selecting another
alias still requires its own verification match. `webSearch.models` contains
`{model,useAdapter,available,reason}` for each supported adapter/base selection;
clients use the matching `useAdapter` entry for their current selection.

Requests may include `webSearch:{version:1,mode:"off"|"auto"|"required"}`.
Omitting this field is off. Client-supplied tool definitions, URLs and old source
registries do not authorize tools. Auto may answer without searching; required
forces the first model action to search. Docs mode and images cannot be combined
with active web search. Unavailable requests return 503 with rejection
`web_unavailable`; malformed fields return 400 `web_invalid`; incompatible modes
return 400 `web_incompatible`. Normal off requests use the existing stream path.

Accepted streams carry `X-Chat-Web-Search:1`. Named SSE events share a random
request-local `runId`, monotonically increasing `sequence`, and `version:1`:

- `web-status`: `status` is planning, searching, reading, or answering.
- `web-sources`: cumulative `sources` with `sourceId` (`S1` etc.), title, URL,
  UTC `retrievedAt`, and `evidenceKind` (search or page). Text is never included.
- `web-complete`: `outcome` completed or failed, optional machine-readable reason.

The final ordinary OpenAI content frames stay unchanged. A successful
`web-complete` precedes `[DONE]`. Errors after acceptance emit failed completion
and the existing stream error frame. Disconnect emits no new content. Empty or
failed searches with no usable evidence fail honestly; they do not return a
successful searched-answer state. Final premature EOF, tool calls, and invalid
finish reasons fail. Clients must resolve `[S1]` only against the same run's
server-supplied source registry, never fabricated model links or prior runs.

## Resource bounds and trust

One admitted request holds the existing account/global slot throughout all
rounds, including provider work. Limits are 2 search attempts, 3 page attempts,
5 model generation calls (up to 4 planners plus final), 6,000 evidence tokens
total, and 1,500 tokens per page. Search snippets are individually capped at 600
tokens. Evidence uses the serving tokenizer; provider JSON is capped at 1 MiB,
with at most 5 results per search and 10 unique source IDs. Initial page text is
bounded before tokenization. No automatic provider retries occur.

`outputTokens` is an aggregate generated-token budget: reserve
`min(2048, outputTokens/2)` for the final answer; each planner requests at most
512 tokens from the remainder and deducts actual reported completion usage.
The final request stays within its reserve. A small user budget may prevent
valid tool calls; it fails rather than silently raising the budget. Context
fitting counts full messages and the same tool schemas before each planner,
reserving final output and user safety tokens. Initial fitting can trim complete
historic pairs; once tool turns begin, context exhaustion fails rather than
silently deleting evidence or tool-call/result pairs.

The overall deadline is `min(timeoutSeconds,240)` and starts before admission.
Provider calls receive at most 15 seconds. After queueing, a quarter of the
remaining time (up to 30 seconds) is reserved for final fitting/synthesis; planners
and provider work use the earlier work deadline. All upstream work inherits
disconnect cancellation. Individual downstream writes retain ten-second limits.

Tools accept one strict argument object, never arbitrary function names or URLs.
`read_page` looks up IDs from this run only and never fetches a URL directly from
the gateway host: Tavily performs extraction. The gateway rejects credentialed,
local/private literal, non-HTTP, control/space-containing and unusual-port URLs.
Returned extraction URLs must match the registered URL. Source titles and URLs
are untrusted display metadata; snippets/pages are explicitly marked untrusted
in the model prompt and tool results. Prompt instructions reduce injection risk
but do not prove a model will ignore malicious content. No tool can access
files, mutate state, send messages, or execute code.

Reference contracts: [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[Tavily Extract](https://docs.tavily.com/documentation/api-reference/endpoint/extract),
[vLLM tool calling](https://docs.vllm.ai/en/stable/features/tool_calling/).

## Local verification

From `gateway/`, run `go test ./...` and `go vet ./...` with the pinned Go
toolchain. `web_test.go` uses fake models/providers and temporary local HTTP
servers, never credentials or production model calls. Also run race tests where
a C toolchain is available, and the existing Python/Go ordinary-chat parity
suite to check off-mode behavior.
