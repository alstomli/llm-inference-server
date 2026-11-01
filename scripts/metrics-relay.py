#!/usr/bin/env python3
"""Expose only monitoring data; protect every request with a bearer token."""
import csv
import hmac
import io
import os
from pathlib import Path
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen


def gpu_metrics():
    fields = ['index', 'utilization.gpu', 'memory.used', 'memory.total', 'temperature.gpu', 'power.draw']
    output = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=5).stdout
    names = ['gpu_utilization_percent', 'gpu_memory_used_bytes', 'gpu_memory_total_bytes', 'gpu_temperature_celsius', 'gpu_power_watts']
    lines = []
    for row in csv.reader(io.StringIO(output)):
        index = str(int(row[0]))
        for name, value in zip(names, row[1:]):
            try:
                number = float(value.strip())
            except ValueError:
                continue
            if 'memory' in name:
                number *= 1024 * 1024
            lines.append(f'{name}{{gpu="{index}"}} {number}')
    return '\n'.join(lines) + '\n'


def collect_metrics():
    with urlopen('http://127.0.0.1:8000/metrics', timeout=8) as response:
        text = response.read().decode()
    try:
        text += '\n' + gpu_metrics() + 'gpu_exporter_up 1\n'
    except (OSError, subprocess.SubprocessError, ValueError):
        text += '\ngpu_exporter_up 0\n'
    return text.encode()


def handler_for(token):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            supplied = self.headers.get('Authorization', '')
            if not hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
                self.send_response(401)
                self.end_headers()
                return
            if self.path != '/metrics':
                self.send_response(404)
                self.end_headers()
                return
            try:
                body = collect_metrics()
            except Exception:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass  # Never log credentials or request headers.
    return Handler


if __name__ == '__main__':
    token = Path(os.environ.get('METRICS_TOKEN_FILE', '/workspace/monitoring/metrics-token')).read_text().strip()
    if len(token) < 32:
        raise SystemExit('A bearer token of at least 32 characters is required')
    ThreadingHTTPServer((os.environ.get('METRICS_HOST', '127.0.0.1'), int(os.environ.get('METRICS_PORT', '8888'))), handler_for(token)).serve_forever()
