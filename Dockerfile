FROM vllm/vllm-openai:v0.32.0
COPY scripts/start-server.sh /usr/local/bin/start-server
COPY scripts/metrics-relay.py /usr/local/lib/metrics-relay.py
COPY scripts/chat-gateway.py /usr/local/lib/chat-gateway.py
RUN chmod +x /usr/local/bin/start-server
ENV HF_HOME=/workspace/huggingface
ENV VLLM_CACHE_ROOT=/workspace/vllm-cache
ENTRYPOINT ["/usr/local/bin/start-server"]
