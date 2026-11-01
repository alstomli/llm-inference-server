FROM prom/prometheus:v3.5.0
COPY --chmod=755 railway/start-prometheus.sh /usr/local/bin/start-prometheus
ENTRYPOINT ["/usr/local/bin/start-prometheus"]
