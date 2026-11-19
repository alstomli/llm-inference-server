# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual label strings used in the hub tracker (`alstomli/llm-inference-server`), which holds the issues for all four repos.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

## Repo labels

Every ticket also carries exactly one label naming the repo its code lands in:

| Label                       | Repo                             |
| --------------------------- | -------------------------------- |
| `repo:llm-inference-server` | vLLM launcher, Go gateway, deploy, CI |
| `repo:qwen-chat`            | Next.js frontend                 |
| `repo:llm-prometheus`       | Prometheus scrape config         |
| `repo:llm-grafana`          | Grafana dashboards               |
