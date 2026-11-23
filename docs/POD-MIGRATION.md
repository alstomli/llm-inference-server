# Pod migration

Use this when a profile (`bf16` or `4bit`) moves to another Pod: its machine is no longer available, it needs another GPU, or it moves to a new volume. Every setting in [Point everything at the new Pod](#point-everything-at-the-new-pod) names the Pod, and a move is done only when each one names the new Pod and [Verify](#verify) passes. Grep alone misses most of them: three live in hosted services, outside any repo, and one in an ignored local file. `deployments/qwen-<profile>/CURRENT.md` is the only tracked record of a profile's current Pod.

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
| Vercel project `qwen-chat`, **Production**, and **Preview** where it has a binding | `RUNPOD_BF16_CHAT_URL` or `RUNPOD_4BIT_CHAT_URL` | `https://<new-pod>-8001.proxy.runpod.net` |
| Local `qwen-chat/.env.local` | the same chat URL | the same value |
| Railway project `easygoing-heart`, service `prometheus` | `RUNPOD_BF16_POD_ID` or `RUNPOD_4BIT_POD_ID` | `<new-pod>`, then deploy the service |
| This repo | `deployments/qwen-<profile>/CURRENT.md` | the new Pod, plus a migration note |

Notes per row:

- **Vercel:** each environment holds its own variables, so check Production and Preview separately. The chat token (`RUNPOD_<PROFILE>_CHAT_TOKEN`) changes only when the token file changed; the user enters it. A settings change reaches the site only after a Production redeploy, and Vercel asks the user for an emailed code to redeploy.
- **Railway:** these variables share their names with the `prod` ones but are set separately. The value is the bare Pod ID; startup adds `-8889.proxy.runpod.net` and the `profile` label, and refuses a hostname or URL. Change only the moved profile's ID and keep the other one set: with only one ID set, Prometheus stops scraping the other profile. A variable change is staged until the service is deployed.
- **`CURRENT.md`:** one branch and one PR in this repo. `llm-prometheus`, `llm-grafana` and the live Grafana dashboard need no change.

## Verify

- Prometheus lists `<new-pod>-8889.proxy.runpod.net` with the moved profile's `profile` label and health `up`: read `/api/v1/targets` through Grafana's data source proxy (`/api/datasources/proxy/uid/prometheus/api/v1/targets`, on the site named by `BASE` in `llm-grafana/scripts/check-deployment.py`, with the login in `llm-grafana/.secrets/railway-grafana.env`). The other profile's target is still listed; it is down if its Pod is stopped.
- The Grafana deployment check passes for the moved profile, run in `llm-grafana`:

  ```bash
  python scripts/check-deployment.py --profile <bf16|4bit>
  ```
- The `inference-baseline` dashboard, opened without a `var-pod` in its URL, shows the new Pod once that profile is selected under **Profile** (it opens on `4bit`). Allow the new target a minute or two of scrapes first. A URL that names the old Pod can still open on it, and the old Pod stays selectable while the time range covers its data.
- The live chat site reports the model ready: `/api/status?model=<bf16|4bit>` from a signed-in browser session returns `"ready": true`.
- Stop the old Pod; it stays stopped, never terminated, until the user decides otherwise:

  ```bash
  python scripts/deploy-runpod.py stop <old-pod>
  ```
