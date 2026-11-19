# Domain Docs

How the engineering skills should consume domain documentation for the Qwen chat stack. **This repo holds the shared domain language** for all four repos (`llm-inference-server`, `qwen-chat`, `llm-prometheus`, `llm-grafana`); the others point here.

## Before exploring, read these

- **`GLOSSARY.md`** at this repo's root: system-wide terms (profile, alias, adapter, acceptance…), shared by all four repos.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.
- **Older decisions** written before this layout: `docs/adr-account-quotas.md`.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## Layout

Single-context: one `GLOSSARY.md` plus `docs/adr/` at this repo's root, serving the whole stack. `qwen-chat` may add its own `GLOSSARY.md` for terms that exist only in the frontend; it never redefines a term from this glossary.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0001 (INT4 W4A16 for the 4bit profile), but worth reopening because…_
