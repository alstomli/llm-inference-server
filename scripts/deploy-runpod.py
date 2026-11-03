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


def deploy_image(path, image, base, env=None):
    # Runpod's update operation resets the Pod; do not separately restart it.
    update = {'imageName': image}
    if env is not None:
        update['env'] = env
    api('PATCH', path, update)
    check_transition(base)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        reported = api('GET', path)
        if reported.get('imageName') == image and (env is None or reported.get('env') == env):
            break
        time.sleep(3)
    else:
        raise RuntimeError('Runpod did not report the requested image/environment')
    bf16 = os.environ.get('DEPLOY_MODEL_PROFILE', '4bit') == 'bf16'
    wait = 3600 if bf16 else 900
    subprocess.run([sys.executable, str(SCRIPT), '--wait', str(wait),
                    '--expected-model', 'qwen-bf16' if bf16 else 'qwen',
                    '--read-timeout', '180' if bf16 else '30'], check=True,
                   timeout=wait + (900 if bf16 else 200))


def deployment_target(env, profile):
    if profile == 'bf16':
        pod = env.get('RUNPOD_BF16_POD_ID', '')
        token = env.get('RUNPOD_BF16_CHAT_TOKEN', '')
        if not pod or not token:
            raise ValueError('Configure RUNPOD_BF16_POD_ID and RUNPOD_BF16_CHAT_TOKEN for this environment')
        if pod == env.get('RUNPOD_POD_ID'):
            raise ValueError('The BF16 target must be different from the 4bit Pod')
        return pod, token
    return env.get('RUNPOD_POD_ID', ''), env.get('RUNPOD_CHAT_TOKEN', '')


def main():
    profile = os.environ.get('DEPLOY_MODEL_PROFILE', '4bit')
    pod, chat_token = deployment_target(os.environ, profile)
    image = os.environ['DEPLOY_IMAGE']
    os.environ['RUNPOD_CHAT_TOKEN'] = chat_token
    if not POD.fullmatch(pod) or not IMAGE.fullmatch(image):
        raise ValueError('Invalid Pod ID or image digest')
    first = os.environ.get('DEPLOY_FIRST_OFFICIAL_RELEASE', 'false')
    if profile not in ('4bit', 'bf16') or first not in ('true', 'false'):
        raise ValueError('Invalid deployment profile or first-release setting')
    if first == 'true' and profile != 'bf16':
        raise ValueError('First official release mode is only for the BF16 migration')
    first_release = first == 'true'
    sequences = os.environ.get('DEPLOY_MAX_NUM_SEQS', '')
    if sequences not in ('', '1', '2'):
        raise ValueError('DEPLOY_MAX_NUM_SEQS must be empty, 1 or 2')
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
    if profile == 'bf16':
        if current.get('env', {}).get('MODEL_PROFILE') != 'bf16':
            raise RuntimeError('Complete the one-time BF16 Pod profile/entrypoint/ports migration first')
        if not first_release and not IMAGE.fullmatch(previous):
            raise RuntimeError('No previous official BF16 image; use first official release mode')
    previous_env = current.get('env') if sequences else None
    if sequences and (not isinstance(previous_env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in previous_env.items())):
        raise RuntimeError('Cannot safely preserve the current Pod environment')
    target_env = {**previous_env, 'MAX_NUM_SEQS': sequences} if sequences else None
    if image == previous and target_env == previous_env and not first_release:
        raise RuntimeError('Requested image is already configured')
    summary(f'Pod: `{pod}`; previous image: `{previous}`; requested image: `{image}`; commit: `{os.environ.get("GITHUB_SHA", "local")}`')
    # A timeout during PATCH can leave its outcome unknown, so attempt rollback
    # even if the API did not acknowledge the update.
    try:
        if sequences:
            summary(f'Applying MAX_NUM_SEQS={sequences}; preserving other environment variables.')
            deploy_image(path, image, base, env=target_env)
            if sequences == '2':
                request = Request(base + '/health', headers={
                    'Authorization': 'Bearer ' + os.environ['RUNPOD_CHAT_TOKEN'],
                    'User-Agent': 'qwen-chat-deployment-check/1.0',
                })
                with urlopen(request, timeout=15) as response:
                    admission = json.load(response).get('admission', {})
                if admission.get('maxConcurrentRequests') != 2 or admission.get('maxWaitingRequests') != 6:
                    raise RuntimeError('Deployed gateway did not advertise two active and six waiting slots')
                summary('Gateway advertises 2 active requests and 6 waiting requests.')
        else:
            deploy_image(path, image, base)
    except Exception as failure:
        if first_release:
            summary('First official BF16 deployment failed; no previous official BF16 release exists. Diagnose the migration and retry in first-release mode; no automatic legacy rollback was attempted.')
            raise RuntimeError('First official BF16 deployment failed') from failure
        summary(f'Deployment failed ({type(failure).__name__}); attempting to restore previous image.')
        try:
            if sequences:
                deploy_image(path, previous, base, env=previous_env)
            else:
                deploy_image(path, previous, base)
            summary('Rollback passed the live chat checks.')
        except Exception as rollback:
            summary(f'ROLLBACK FAILED ({type(rollback).__name__}): inspect the Pod immediately. Previous image: `{previous}`')
        raise RuntimeError('Deployment failed; inspect workflow logs and rollback status') from failure
    summary('First official BF16 release passed the live chat checks.' if first_release else 'Deployment passed the live chat checks.')


if __name__ == '__main__':
    main()
