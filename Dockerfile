FROM golang:1.27.3-bookworm@sha256:dc0963f32053e4bc451d24d87e3892d42d01021b61f093e98eb95c0352bb84b0 AS gateway-build
WORKDIR /src
COPY gateway/go.mod gateway/go.sum ./
RUN go mod download
COPY gateway/ ./
RUN go vet ./... && go test -race ./... \
 && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /out/chat-gateway ./cmd/chat-gateway

# Preserve the exact deployed vendor/Jupyter dependency stack (originally vLLM 0.32.0).
# Vendor refreshes are a separate reviewed change; do not resolve mutable base tags or pip here.
FROM zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d
COPY scripts/start-jupyter.py /usr/local/lib/start-jupyter.py
COPY scripts/start-server.sh /usr/local/bin/start-server
COPY scripts/kv-runtime-config.py /usr/local/lib/kv-runtime-config.py
COPY scripts/kv_runtime.py /usr/local/lib/python3.12/dist-packages/kv_runtime.py
COPY scripts/metrics-relay.py /usr/local/lib/metrics-relay.py
COPY scripts/chat-gateway.py /usr/local/lib/chat-gateway.py
COPY --from=gateway-build /out/chat-gateway /usr/local/bin/chat-gateway
COPY scripts/validate-lora-adapter.py /usr/local/lib/validate-lora-adapter.py
COPY deployments/qwen-bf16/prepare_model.py /usr/local/lib/prepare-bf16-model.py
RUN chmod +x /usr/local/bin/start-server
ENV MODEL_PROFILE=bf16
ENTRYPOINT ["/usr/local/bin/start-server"]
