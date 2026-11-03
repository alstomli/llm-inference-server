"""Retain the Runpod template's token-authenticated Jupyter access."""
import os
import re
from urllib.parse import urlsplit
from pathlib import Path


def configuration(env):
    token = env.get('JUPYTER_PASSWORD') or env.get('JUPYTER_TOKEN')
    if not token:
        raise ValueError('Jupyter requires its existing authentication token')
    port = int(env.get('JUPYTER_PORT', '8888'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid Jupyter port')
    pod = env.get('RUNPOD_POD_ID', '')
    origin = env.get('JUPYTER_ALLOW_ORIGIN', '')
    if not origin and pod:
        origin = f'https://{pod}-{port}.proxy.runpod.net'
    if origin:
        url = urlsplit(origin)
        match = re.fullmatch(r'([a-z0-9]+)-([0-9]+)\.proxy\.runpod\.net', url.netloc)
        if (url.scheme != 'https' or not match or url.path not in ('', '/')
                or url.query or url.fragment or int(match.group(2)) != port
                or (pod and match.group(1) != pod)):
            raise ValueError('Jupyter origin must be the exact HTTPS proxy for this Pod and port')
        origin = 'https://' + url.netloc
    root = Path(env.get('JUPYTER_SERVER_ROOT', '/workspace')).resolve()
    if not root.is_dir():
        raise ValueError('Jupyter root directory does not exist')
    return {
        'IdentityProvider': {'token': token},
        'ServerApp': {'ip': '0.0.0.0', 'port': port, 'port_retries': 0,
                      'root_dir': str(root), 'preferred_dir': str(root),
                      'allow_root': True, 'open_browser': False,
                      'allow_remote_access': True, 'trust_xheaders': True,
                      'allow_origin': origin,
                      'terminado_settings': {'shell_command': ['/bin/bash']},
                      'log_level': 'WARN'},
        'FileContentsManager': {'delete_to_trash': False},
        'LabApp': {'log_level': 'WARN'},
    }


if __name__ == '__main__':
    from jupyterlab.labapp import LabApp
    from traitlets.config import Config
    # Tokens stay in the process environment/configuration, never the command line.
    # initialize_server forwards this config into ServerApp; launch_instance
    # does not forward keyword arguments in Jupyter Server 2.21.1.
    server = LabApp.initialize_server(argv=[], config=Config(configuration(os.environ)))
    server.start()
