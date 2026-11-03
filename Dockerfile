FROM vllm/vllm-openai:v0.32.0
# Resolve administrative UI dependencies at build time without upgrading vLLM's stack.
RUN python3 -m pip freeze > /tmp/vllm-constraints.txt \
 && python3 -m pip install --no-cache-dir -c /tmp/vllm-constraints.txt \
      'jupyterlab==4.6.4' 'jupyter-server==2.21.1' 'notebook==7.6.3' \
 && python3 -m pip check \
 && rm /tmp/vllm-constraints.txt
COPY scripts/start-jupyter.py /usr/local/lib/start-jupyter.py
COPY scripts/start-server.sh /usr/local/bin/start-server
COPY scripts/metrics-relay.py /usr/local/lib/metrics-relay.py
COPY scripts/chat-gateway.py /usr/local/lib/chat-gateway.py
COPY deployments/qwen-bf16/prepare_model.py /usr/local/lib/prepare-bf16-model.py
RUN chmod +x /usr/local/bin/start-server
ENV HF_HOME=/workspace/huggingface
ENV VLLM_CACHE_ROOT=/workspace/vllm-cache
ENTRYPOINT ["/usr/local/bin/start-server"]
