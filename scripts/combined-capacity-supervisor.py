"""Temporary, fixed-profile supervisor; the external watchdog owns pod rollback."""
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import requests
from combined_capacity_runtime import CPU_BYTES, atomic, resources, safe, secure_root, validate_lease

ROOT = secure_root(os.environ['CAPACITY_SESSION_ROOT'])
LEASE = json.loads((ROOT / 'lease.json').read_text())
ORIGINAL_ENV = dict(os.environ)
BASELINE = resources()
engine = runner = None


def event(kind, **data):
    with (ROOT / 'supervisor-events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time': time.time(), 'kind': kind, **data}) + '\n')


def spawn(argv, env, name):
    with (ROOT / (name + '.private.log')).open('ab') as log:
        return subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def terminate(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A dead leader alone does not establish worker release.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stock(reason):
    event('restore_enabled_production', reason=reason)
    atomic(ROOT / 'restore-requested.json', {'time': time.time(), 'reason': reason, 'lease_id': LEASE['lease_id']})
    terminate(runner)
    terminate(engine)
    until = min(time.time() + 60, LEASE['exclusive_end'])
    while time.time() < until:
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                  '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
        if not active:
            break
        time.sleep(1)
    else:
        raise RuntimeError('Owned GPU worker release could not be verified')
    # Original container environment still contains native64 + production graphs.
    os.execve('/usr/local/bin/start-server', ['/usr/local/bin/start-server'], ORIGINAL_ENV)


class Maintenance(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Retry-After', '120')
        self.end_headers()
        self.wfile.write(b'{"error":"Scheduled capacity validation"}')

    do_POST = do_GET

    def log_message(self, *args):
        pass


def main():
    global engine, runner
    validate_lease(LEASE, time.time())
    assert time.time() - LEASE['created'] < 180, 'Bootstrap lease is stale'
    hashes = json.loads((ROOT / 'source-files.json').read_text())
    import hashlib
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    safe(resources(), BASELINE, projected_cpu_bytes=CPU_BYTES)
    server = http.server.ThreadingHTTPServer(('0.0.0.0', 8001), Maintenance)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child_env = {**ORIGINAL_ENV, 'CHAT_ENABLED': 'false', 'CAPACITY_SESSION_ROOT': str(ROOT),
                 'JUPYTER_SERVER_ROOT': str(ROOT), 'PATH': str(ROOT / 'bin') + os.pathsep + ORIGINAL_ENV['PATH'],
                 'PYTHONPATH': str(ROOT) + os.pathsep + ORIGINAL_ENV.get('PYTHONPATH', '')}
    engine = spawn(['/usr/local/bin/start-server'], child_env, 'engine')
    started = time.time()
    ready_at = None
    session = requests.Session()
    session.trust_env = False
    headers = {'Authorization': 'Bearer ' + os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
    while time.time() < LEASE['restore_at']:
        now = time.time()
        memory = resources()
        safe(memory, BASELINE)
        if (ROOT / 'restore.request').exists():
            break
        if engine.poll() is not None:
            stock('engine_or_sidecar_exit')
        if ready_at is None:
            if now - started > LEASE['startup_cap_seconds']:
                stock('startup_deadline')
            try:
                response = session.get('http://127.0.0.1:8000/health', headers=headers, timeout=2)
                if response.status_code == 200:
                    ready_at = now
                    assert (ROOT / 'allocation.json').exists() and (ROOT / 'launch.json').exists()
                    runner = spawn(['python3', str(ROOT / 'combined-capacity-run.py')], child_env, 'runner')
                    event('engine_ready', startup_seconds=now - started, runner_pid=runner.pid)
            except requests.RequestException:
                pass
        if now >= LEASE['testing_stop_at'] and runner is not None and runner.poll() is None:
            terminate(runner)
            event('testing_hard_stop')
        progress = json.loads((ROOT / 'progress.json').read_text()) if (ROOT / 'progress.json').exists() else None
        phase = 'loading' if ready_at is None else 'running' if runner.poll() is None else 'awaiting_backup'
        # A healthy engine and its /tmp-rooted Jupyter remain available while
        # the tester backs partial evidence after an ordinary runner failure.
        # OOM/headroom/engine failures still restore immediately above.
        atomic(ROOT / 'status.json', {'schema_version': 1, 'run_id': LEASE['run_id'], 'lease_id': LEASE['lease_id'],
               'time': now, 'phase': phase, 'engine_pid': engine.pid, 'runner_pid': runner.pid if runner else None,
               'ready_at': ready_at, 'memory': memory, 'progress': progress,
               'runner_returncode': runner.returncode if runner else None,
               'testing_stop_at': LEASE['testing_stop_at'], 'restore_at': LEASE['restore_at'],
               'exclusive_end': LEASE['exclusive_end']})
        time.sleep(2)
    server.shutdown()
    stock('deadline_or_operator_restore')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        event('supervisor_failure', error_type=type(error).__name__)
        stock('supervisor_failure')
