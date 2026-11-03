"""Retain the Runpod template's token-authenticated Jupyter access."""
import os
from pathlib import Path


def configuration(env):
    token = env.get('JUPYTER_PASSWORD') or env.get('JUPYTER_TOKEN')
    if not token:
        raise ValueError('Jupyter requires its existing authentication token')
    port = int(env.get('JUPYTER_PORT', '8888'))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid Jupyter port')
    root = Path(env.get('JUPYTER_SERVER_ROOT', '/workspace')).resolve()
    if not root.is_dir():
        raise ValueError('Jupyter root directory does not exist')
    return {
        'IdentityProvider': {'token': token},
        'ServerApp': {'ip': '0.0.0.0', 'port': port, 'port_retries': 0,
                      'root_dir': str(root), 'preferred_dir': str(root),
                      'allow_root': True, 'open_browser': False,
                      'allow_remote_access': True, 'trust_xheaders': True,
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
