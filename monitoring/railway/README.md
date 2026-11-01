# Railway deployment

## Live deployment

Deployed and verified on 2026-10-31 in Railway project `easygoing-heart`, production environment, EU West (Amsterdam).

- Dashboard: https://grafana-production-5c5e.up.railway.app/d/inference-baseline
- Project: https://railway.com/project/0bfbf322-de90-4f56-bd09-2401e145dbb2
- Cloud login: user `admin`; password saved locally in `../.secrets/railway-grafana.env` and as the Grafana service variable in Railway. This is separate from the local Grafana password.
- Both services have persistent volumes, one replica, and limits of 0.5 vCPU / 0.5 GB RAM each.
- Prometheus has no public domain; Grafana uses its private address and requires login over public HTTPS.
- Verification: `python scripts/check-railway-deployment.py` from the repository root checks HTTPS health, authentication, 20 panels, private data source health, and live inference/GPU collection.
- Cloud collection continues while this PC is off. The Runpod Pod and metrics relay must remain running. Cloud history starts from this deployment; previous local history was not migrated.

Deploy two services in one Railway project/environment. Use the custom pinned images below. Keep the current local monitoring stack until the cloud dashboard is verified. Existing local history is not included in these images; the new Prometheus service starts a new history.

Build from the monitoring directory:

```powershell
docker buildx build --platform linux/amd64 --load -f railway/prometheus.Dockerfile -t zihan000/qwen-inference:prometheus-0.1.1 .
docker buildx build --platform linux/amd64 --load -f railway/grafana.Dockerfile -t zihan000/qwen-inference:grafana-0.1.0 .
```

Images contain configuration and dashboard definitions, not credentials. Push these exact tags after validation.

## Prometheus service

- Image: `zihan000/qwen-inference:prometheus-0.1.1`
- Name: `prometheus`
- Volume: `/prometheus`, default Hobby volume capacity; retention defaults to 7 days or 512 MB of blocks, leaving headroom for WAL, head blocks, and compaction
- Variables: `PORT=9090`, `RUNPOD_METRICS_TARGET=hrqjt6otlehqgh-8888.proxy.runpod.net`, `RUNPOD_METRICS_TOKEN` from the existing local monitoring secret
- Set `RAILWAY_RUN_UID=0` if required for Railway volume permissions, per Railway's volume documentation
- No public domain or TCP proxy; private networking only
- Disable service sleeping, one replica, restart on failure
- Health endpoint: `/-/ready`, port 9090

## Grafana service

- Image: `zihan000/qwen-inference:grafana-0.1.0`
- Name: `grafana`
- Volume: `/var/lib/grafana`, default Hobby volume capacity
- Variables: `PORT=3000`, `GF_SERVER_HTTP_PORT=3000`, `PROMETHEUS_URL=http://prometheus.railway.internal:9090`, `GF_SECURITY_ADMIN_USER=admin`, `GF_SECURITY_ADMIN_PASSWORD` as a Railway secret (a separate cloud admin password)
- Set `RAILWAY_RUN_UID=0` if required for Railway volume permissions
- Public Railway HTTPS domain, target port 3000; set `GF_SERVER_ROOT_URL` to that HTTPS address and `GF_SECURITY_COOKIE_SECURE=true`
- Keep sign-up and anonymous access disabled
- Disable service sleeping, one replica, restart on failure
- Health endpoint: `/api/health`, port 3000

Volumes are billed for used space, rather than the entire capacity. Set each service to one replica with 0.5 vCPU and 0.5 GB memory ceilings for the initial personal deployment. These ceilings are not a spending cap, and the actual billed usage can differ from local container measurements.

An account spending limit can affect other projects in the same workspace; do not change it without checking the user's intent. Service usage is billed on the existing Hobby account. Obtain a deployment budget before starting the services.

## Validation

Check both services' health; Grafana's data source must reach Prometheus privately; `up{job="vllm"}` and `gpu_exporter_up` must equal 1. Test that Grafana requires login and Prometheus has no public URL. Verify the dashboard shows live metrics after a few scrapes.

The model remains on Runpod and its API stays private. Inference image `0.1.1` starts the relay automatically when `METRICS_ENABLED=true`; the existing Pod also sets `METRICS_HOST=0.0.0.0`. After a restart, collection resumes once the model is ready. See `../README.md` for legacy image 0.1.0 manual startup. Railway requires the existing token to collect metrics; do not put it in source code, image layers, public URLs, or logs.
