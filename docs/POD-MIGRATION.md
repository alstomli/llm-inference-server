# Pod migration

Use this when a profile (`bf16` or `4bit`) moves to another Pod: its machine is no longer available, it needs another GPU, or it moves to a new volume. Every setting in [Point everything at the new Pod](#point-everything-at-the-new-pod) names the Pod, and a move is done only when each one names the new Pod and [Verify](#verify) passes. Grep alone misses most of them: four live outside any repo.

## Prepare the new Pod

1. Create it in the data center of the profile's network volume (a volume is tied to its data center), with the profile's GPU, the volume mounted at `/workspace`, and ports `8001/http`, `8888/http`, `8889/http` and `22/tcp`.
2. Check the volume. On the same volume the model and both token files carry over. On a new volume, the model goes under `/workspace/qwen-bf16` or `/workspace/qwen-4bit`, and the user places the token files with the old values: the chat token at `/workspace/chat/chat-token` and the metrics token at `/workspace/monitoring/metrics-token`, each holding only the value.
3. Copy the old Pod's image and settings:

   ```bash
   python scripts/deploy-runpod.py copy-env --profile 4bit <old-pod> <new-pod>
   ```

   It keeps the new Pod's own `JUPYTER_PASSWORD`, restarts the Pod, runs the live chat check against the new Pod, and restores the new Pod's previous image and settings if the check fails.
4. Check the metrics relay:

   ```bash
   RUNPOD_METRICS_URL=https://<new-pod>-8889.proxy.runpod.net python scripts/check-monitoring.py
   ```

## Point everything at the new Pod

| Where | Setting | New value |
| --- | --- | --- |
| This repo, GitHub environment `prod` | `RUNPOD_BF16_POD_ID` or `RUNPOD_4BIT_POD_ID` | `<new-pod>` (`gh variable set <name> -e prod -b <new-pod>`) |
| Vercel project `qwen-chat`, **Production and Preview** | `RUNPOD_BF16_CHAT_URL` or `RUNPOD_4BIT_CHAT_URL` | `https://<new-pod>-8001.proxy.runpod.net` |
| Local `qwen-chat/.env.local` | the same chat URL | the same value |
| Railway project `easygoing-heart`, service `prometheus` | `RUNPOD_METRICS_TARGETS` (comma list of every profile's relay) | swap in `<new-pod>-8889.proxy.runpod.net` |
| The same service, BF16 only | `RUNPOD_METRICS_TARGET` (single-target fallback) | `<new-pod>-8889.proxy.runpod.net` |
| Live Grafana, dashboard `inference-baseline` | `pod` variable's saved default | `<new-pod>-8889.proxy.runpod.net`, shown as `<new-pod>` |
| `llm-prometheus` repo | `.env.example`, `prometheus.example.yml`, README, `scripts/test_start_prometheus.py` | the new relay host |
| `llm-grafana` repo | `dashboards/inference.json` default, README, `scripts/test_flux2_dashboard.py` | the new Pod |
| This repo | `deployments/qwen-<profile>/CURRENT.md` | the new Pod, plus a migration note |

Notes per row:

- **Vercel:** each environment holds its own variables, so check Production and Preview separately. The chat token (`RUNPOD_<PROFILE>_CHAT_TOKEN`) changes only when the token file changed; the user enters it. A settings change reaches the site only after a Production redeploy, and Vercel asks the user for an emailed code to redeploy.
- **Railway:** a variable change is staged until the service is deployed.
- **Grafana:** back up the dashboard JSON to `llm-grafana/.secrets/dashboard-backups/` before saving. The `pod` query stays `label_values(up{job="vllm"}, instance)`, so every scraped Pod remains selectable; only the default changes. The login is `llm-grafana/.secrets/railway-grafana.env`, and the site address is `BASE` in `llm-grafana/scripts/check-deployment.py`.
- **Repos:** one branch and one PR per repo, each linking this repo's PR.

## Verify

- Prometheus lists the new relay with health `up`: read `/api/v1/targets` through Grafana's data source proxy (`/api/datasources/proxy/uid/prometheus/api/v1/targets`).
- The live chat site reports the model ready: `/api/status?model=<bf16|4bit>` from a signed-in browser session returns `"ready": true`.
- Stop the old Pod; it stays stopped, never terminated, until the user decides otherwise:

  ```bash
  python scripts/deploy-runpod.py stop <old-pod>
  ```
