FROM grafana/grafana:12.1.0
COPY grafana/dashboards /etc/grafana/dashboards
COPY railway/dashboards.yml /etc/grafana/provisioning/dashboards/dashboards.yml
COPY railway/datasource.yml /etc/grafana/provisioning/datasources/prometheus.yml
ENV PROMETHEUS_URL=http://prometheus.railway.internal:9090
ENV GF_USERS_ALLOW_SIGN_UP=false
ENV GF_AUTH_ANONYMOUS_ENABLED=false
ENV GF_SERVER_HTTP_ADDR=0.0.0.0
ENV GF_SERVER_HTTP_PORT=3000
