"""Update one Runpod image and restore the previous image if verification fails."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE = re.compile(r'zihan000/qwen-inference@sha256:[0-9a-f]{64}\Z')
POD = re.compile(r'[A-Za-z0-9_-]+\Z')
SCRIPT = Path(__file__).with_name('check-chat-deployment.py')


def api(method, path, body=None):
    request = Request('https://rest.runpod.io/v1' + path,
                      data=json.dumps(body).encode() if body is not None else None,
                      headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                               'User-Agent': 'llm-inference-deployer/1.0',
                               'Content-Type': 'application/json'}, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except HTTPError as error:
        raise RuntimeError(f'Runpod API {method} failed: HTTP {error.code}') from None
    except URLError:
        raise RuntimeError(f'Runpod API {method} connection failed') from None


def summary(message):
    print(message, flush=True)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as output:
            output.write(message + '\n\n')


def check_transition(base, timeout=180):
    """Require an observed interruption so old-container readiness cannot pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(Request(base + '/health'), timeout=5) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        except (URLError, TimeoutError, OSError):
            return
        if status != 401:
            return
        time.sleep(2)
    raise RuntimeError('Did not observe service interruption; cannot verify new container is serving')


def deploy_image(path, image, base):
    # Runpod's update operation resets the Pod; do not separately restart it.
    api('PATCH', path, {'imageName': image})
    check_transition(base)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if api('GET', path).get('imageName') == image:
            break
        time.sleep(3)
    else:
        raise RuntimeError('Runpod did not report the requested image')
    subprocess.run([sys.executable, str(SCRIPT), '--wait', '900'], check=True,
                   timeout=1100)


def main():
    pod, image = os.environ['RUNPOD_POD_ID'], os.environ['DEPLOY_IMAGE']
    if not POD.fullmatch(pod) or not IMAGE.fullmatch(image):
        raise ValueError('Invalid Pod ID or image digest')
    for name in ('RUNPOD_API_KEY', 'RUNPOD_CHAT_TOKEN'):
        if not os.environ.get(name):
            raise ValueError(f'Missing {name}')
    base = f'https://{pod}-8001.proxy.runpod.net'
    os.environ['RUNPOD_CHAT_URL'] = base
    path = '/pods/' + pod
    current = api('GET', path)
    previous = current.get('imageName')
    if not isinstance(previous, str) or not re.fullmatch(r'[A-Za-z0-9_./:@-]+', previous):
        raise RuntimeError('Cannot safely determine the previous image')
    if current.get('desiredStatus') != 'RUNNING':
        raise RuntimeError('Pod must be running before deployment')
    if image == previous:
        raise RuntimeError('Requested image is already configured')
    summary(f'Pod: `{pod}`; previous image: `{previous}`; requested image: `{image}`; commit: `{os.environ.get("GITHUB_SHA", "local")}`')
    # A timeout during PATCH can leave its outcome unknown, so attempt rollback
    # even if the API did not acknowledge the update.
    try:
        deploy_image(path, image, base)
    except Exception as failure:
        summary(f'Deployment failed ({type(failure).__name__}); attempting to restore previous image.')
        try:
            deploy_image(path, previous, base)
            summary('Rollback passed the live chat checks.')
        except Exception as rollback:
            summary(f'ROLLBACK FAILED ({type(rollback).__name__}): inspect the Pod immediately. Previous image: `{previous}`')
        raise RuntimeError('Deployment failed; inspect workflow logs and rollback status') from failure
    summary('Deployment passed the live chat checks.')


if __name__ == '__main__':
    main()
