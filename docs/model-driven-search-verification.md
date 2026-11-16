# Model-driven search verification

Semantic acceptance is **blocked**. The implementation remains in draft PRs; no merge or manual deployment was performed. The existing Vercel integration automatically built a frontend preview after PR creation.

The frontend automatically enables native `auto` tool mode only for supported normal text requests. The gateway policy handles intent, context, explicit browse/no-browse instructions and request UTC date. Existing resource budgets, parser, quotas, timeouts and cancellation remain in force.

## Verified implementation

- Frontend: 106 tests, TypeScript check, production webpack build, browser search/direct/citation/retry/Stop fixture, and desktop/mobile/320px layouts passed. Webpack was used because Turbopack rejects dependencies linked outside an isolated worktree root.
- Gateway: `go test ./...` and `go vet ./...` passed, including date retention, token-fitted planner-only instruction, native auto/no-provider path, strict truncated-response rejection, and blank/whitespace final stream failure.
- Existing configured deployment smoke harness passed authenticated readiness/account quotas and a nonempty completed reply. It validates the existing deployment, not the new policy.

## Actual configured-model check

Tests used a temporary loopback-only gateway compiled from this branch against the configured BF16 base model and provider, with a private recording proxy for synthetic prompts. Requests were sequential with the existing stable smoke principal and at least 11 seconds between starts. No production configuration, provider/model selection, parser, performance settings or budgets were changed. Test processes and uploads were removed; public capability metadata was checked unchanged.

The 28-case matrix covered 14 direct tasks, four clarifications and ten searches, including paraphrases, supplied text, explicit instructions, changing facts, referenced pages and follow-ups. After the initial short-handoff correction, 27/28 tool decisions matched the expected category; 26/28 produced completed nonempty output. These are separate metrics, and neither proves grounded acceptance.

Observed defects included a stable KV paraphrase exhausting the 512-token planner; a supplied current URL being declined without tools; an empty final stream previously reported completed; and an Apple retail-price claim citing a reseller comparator and conditional carrier credits. The final scoped correction strengthens neutral querying, exact fact/source/time/condition matching, source identity, insufficient-evidence disclosure, referenced-page workflow and planner/final separation. Blank final streams now fail closed.

A single focused rerun of all ten searched cases plus two formerly failing stable paraphrases and one clarification tests that correction. Its receipt is linked below. It completed 13/13 cases, with 11/13 expected tool decisions and 10/13 completed nonempty outputs. Both stable paraphrases unnecessarily searched. The primary iPhone price question, availability/price recommendation, and price follow-up each ended with final finish `stop` and zero user-facing content; all three now correctly fail closed. The referenced URL now triggers tools, but the model reads an article about the target and still presents an unverified current-target summary; this is only a tool-selection improvement, not grounded task acceptance. The equivalent iPhone paraphrase and natural-language price lookup honestly report that the current Apple price could not be verified. No further prompt tuning or parser/budget relaxation was attempted.

The planner brevity instruction is advisory for this model. The full matrix did not retain exact final finish reasons; the focused rerun does. The existing stream contract accepts both `stop` and `length`; completed metadata alone must not be treated as a fully correct or fully grounded answer. This small semantic check is not a load test and does not establish general factual accuracy or privacy compliance.

The evaluated code is `d60242bf02cc68382ee076a9f3a588f8e4df19f5`; the subsequent evidence-only commit does not change it. No aggregate grounded-answer pass rate is claimed.

Compact synthetic receipts: [case evidence](evidence/model-driven-search.json). Full synthetic receipts, raw public tool evidence and diagnostic wire traces remain in local `.tools/model-search-semantic-results.json`, `.tools/model-search-grounding-results.json`, and `.tools/model-search-diagnostic-results.json` artifacts rather than product history or this repository. Frontend companion: [qwen-chat PR 29](https://github.com/alstomli/qwen-chat/pull/29); gateway: [PR 44](https://github.com/alstomli/llm-inference-server/pull/44).
