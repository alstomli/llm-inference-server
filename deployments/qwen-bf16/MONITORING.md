# Monitoring the BF16 Pod

The [official vLLM image deployment](VLLM-IMAGE.md) supersedes this historical
PyTorch-template startup procedure. Its same-context/concurrency trial requires
live verification; this document retains the earlier deployment record.

Pod `ftntkktzid9f4p` uses Jupyter on HTTP port 8888 and a separate, bearer-token
protected metrics relay on HTTP port 8889. vLLM remains on `127.0.0.1:8000`.
The original BF16 bootstrap is documented in the deployment branch/PR:
https://github.com/alstomli/llm-inference-server/pull/11.

## Persistent files

On global volume `spicy_coffee_meerkat`, mounted at `/workspace`:

- `/workspace/qwen-bf16/start-monitoring.sh`: the launcher in this directory.
- `/workspace/qwen-bf16/metrics-relay.py`: an unchanged copy of
  `scripts/metrics-relay.py` from this repository.
- `/workspace/qwen-bf16/metrics-token`: the existing private monitoring token,
  reused with the user's approval. Its value is absent from Git, images, logs,
  screenshots, and this document.

The container start command is:

```sh
bash -c 'bash /workspace/qwen-bf16/start-monitoring.sh >>/tmp/qwen-monitoring.log 2>&1 & bash /workspace/qwen-bf16/bootstrap.sh >>/tmp/qwen-bf16-bootstrap.log 2>&1 & exec /start.sh'
```

Expose HTTP ports `8888,8889`, retaining TCP port 22, the existing PyTorch image,
50 GB container disk, and global volume. Applying this Runpod edit resets the
container and reloads BF16. Jupyter starts through the template's `/start.sh`.
The relay uses only Python's standard library and `nvidia-smi`; it has a separate
local lock at `/tmp/qwen-bf16-metrics.lock` and log at `/tmp/qwen-monitoring.log`.

## Metrics and access checks

The endpoint is `https://ftntkktzid9f4p-8889.proxy.runpod.net/metrics`.
The relay combines vLLM's private `/metrics` with the same GPU utilization,
memory, temperature, power, and exporter-health metrics as the existing Pod.

Checks before exposing the port passed: missing token 401, wrong token 401,
correct token 200 with vLLM/GPU metrics, and an authenticated model route 404.
The relay returns 503 while vLLM is loading, so Prometheus may report this target
down until model initialization completes.

Prometheus retains `rp93m3noz3l9lt-8888.proxy.runpod.net` and adds this new target
under the same `vllm` job. Existing `instance` labels distinguish the Pods and
preserve their historical series. The Grafana dashboard's Pod variable filters
the existing panels by instance before aggregation. Its Prometheus and Grafana
changes are documented in their sibling repositories.

## Operator commands

In the Pod terminal:

```sh
tail -n 30 /tmp/qwen-monitoring.log
tail -n 30 /tmp/qwen-bf16-bootstrap.log
curl --fail http://127.0.0.1:8000/health
```

From the workspace, use the existing relay checker with the new endpoint and
the existing credential (supply it privately through the environment):

```sh
RUNPOD_METRICS_URL=https://ftntkktzid9f4p-8889.proxy.runpod.net python scripts/check-monitoring.py
```

To roll back this addition, restore the original BF16-only container start
command and HTTP port list `8888`, then save the Pod configuration. Retain the
global volume. Remove the new target from Prometheus separately; the original
Pod and its history remain available.
