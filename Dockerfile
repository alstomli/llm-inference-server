FROM golang:1.27.3-bookworm@sha256:dc0963f32053e4bc451d24d87e3892d42d01021b61f093e98eb95c0352bb84b0 AS gateway-build
WORKDIR /src
COPY gateway/go.mod gateway/go.sum ./
RUN go mod download
COPY gateway/ ./
RUN go vet ./... && go test -race ./... \
 && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /out/chat-gateway ./cmd/chat-gateway

FROM vllm/vllm-openai:v0.32.0
# Preserve the vendor stack, including its existing dependency overrides.
# Jupyter must not introduce any additional pip-check conflicts.
RUN (python3 -m pip check > /tmp/pip-check-before || test "$?" -eq 1) \
 && python3 -m pip freeze > /tmp/vllm-constraints.txt \
 && python3 -m pip install --no-cache-dir -c /tmp/vllm-constraints.txt \
      'jupyterlab==4.6.4' 'jupyter-server==2.21.1' 'notebook==7.6.3' \
 && (python3 -m pip check > /tmp/pip-check-after || test "$?" -eq 1) \
 && sort /tmp/pip-check-before > /tmp/pip-check-before-sorted \
 && sort /tmp/pip-check-after > /tmp/pip-check-after-sorted \
 && diff -u /tmp/pip-check-before-sorted /tmp/pip-check-after-sorted \
 && rm /tmp/vllm-constraints.txt /tmp/pip-check-*
COPY scripts/start-jupyter.py /usr/local/lib/start-jupyter.py
COPY scripts/start-server.sh /usr/local/bin/start-server
COPY scripts/metrics-relay.py /usr/local/lib/metrics-relay.py
COPY scripts/chat-gateway.py /usr/local/lib/chat-gateway.py
COPY --from=gateway-build /out/chat-gateway /usr/local/bin/chat-gateway
COPY scripts/validate-lora-adapter.py /usr/local/lib/validate-lora-adapter.py
COPY deployments/qwen-bf16/prepare_model.py /usr/local/lib/prepare-bf16-model.py
RUN chmod +x /usr/local/bin/start-server
ENV MODEL_PROFILE=bf16
ENTRYPOINT ["/usr/local/bin/start-server"]
