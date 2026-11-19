# Issue tracker: GitHub (hub for the Qwen chat stack)

Issues and specs for **all four repos** of the Qwen chat stack live as GitHub issues in **this repo**, `alstomli/llm-inference-server`. The other repos (`qwen-chat`, `llm-prometheus`, `llm-grafana`) do not use their own Issues; their `docs/agents/issue-tracker.md` points here. Use the `gh` CLI for all operations.

## Cross-repo conventions

- **One spec, one issue here.** A feature that spans repos gets a single spec issue in this repo. Its tickets are sub-issues of that spec.
- **Every ticket names its repo** with exactly one label: `repo:llm-inference-server`, `repo:qwen-chat`, `repo:llm-prometheus` or `repo:llm-grafana`. The code for a ticket goes in a branch and PR in that repo.
- **Blocking edges may cross repos.** All tickets live here, so a `qwen-chat` ticket can be blocked by an `llm-inference-server` ticket using native dependencies (see **Blocking**).
- **PRs close tickets by full reference.** A PR in another repo uses `Closes alstomli/llm-inference-server#<n>`. A PR in this repo can use `Closes #<n>`.
- When working in another repo's checkout, pass `--repo alstomli/llm-inference-server` to every `gh issue` command.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Make an issue a sub-issue of a parent**: `gh issue create --parent <parent> ...`, or `gh issue edit <parent> --add-sub-issue <child>` afterwards (`gh` 2.94+). Without sub-issues, put `Part of #<parent>` at the top of the child body.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`
- **Blocking**: GitHub's native issue dependencies. Add an edge with `gh api --method POST repos/alstomli/llm-inference-server/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric database id (`gh api repos/alstomli/llm-inference-server/issues/<n> --jq .id`), not its `#number`. Where dependencies aren't available, put a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `alstomli/llm-inference-server`, with the right `repo:*` label for tickets.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above (with `--repo alstomli/llm-inference-server` outside this checkout).

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: a sub-issue of the map, labelled `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: as in **Blocking** above.
- **Frontier query**: list the map's open children, drop any with an open blocker or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
