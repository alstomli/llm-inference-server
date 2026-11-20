# llm-inference-server

Self-hosted Qwen3.8-27B serving on RunPod: vLLM launcher, Go chat gateway, deploy scripts and CI. This repo is also the **hub** for the Qwen chat stack: issues, specs and shared domain docs for `qwen-chat`, `llm-prometheus` and `llm-grafana` live here.

## Agent skills

### Issue tracker

GitHub Issues in this repo are the single tracker for all four repos; tickets carry a `repo:*` label. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) plus one `repo:*` label per ticket. See `docs/agents/triage-labels.md`.

### Pod migration

Moving a profile to another Pod, or finding where a Pod ID is configured: follow `docs/POD-MIGRATION.md`. Four of its settings live in hosted services, outside any repo.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` here are the shared language for the whole stack. See `docs/agents/domain.md`.
