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
    wait = 3600
    subprocess.run([sys.executable, str(SCRIPT), '--wait', str(wait),
                    '--expected-model', 'qwen-bf16',
                    '--read-timeout', '180'], check=True,
                   timeout=wait + 900)


def deployment_target(env, profile='bf16'):
    if profile != 'bf16':
        raise ValueError('Only the bf16 deployment profile is supported')
    pod = env.get('RUNPOD_BF16_POD_ID', '')
    token = env.get('RUNPOD_BF16_CHAT_TOKEN', '')
    if not pod or not token:
        raise ValueError('Configure RUNPOD_BF16_POD_ID and RUNPOD_BF16_CHAT_TOKEN for this environment')
    return pod, token


def web_environment(current, action, verified_alias, key):
    """Only these named operations may change the server-owned web settings."""
    target = dict(current)
    if current.get('MODEL_PROFILE') != 'bf16' or current.get('SERVED_MODEL_NAME') != 'qwen-bf16' or current.get('VLLM_BASE_MODEL_NAME') != 'qwen-bf16-base':
        raise ValueError('Web configuration requires the existing BF16 base/adapter aliases')
    if current.get('CHAT_GATEWAY_IMPLEMENTATION', 'go') != 'go':
        raise ValueError('Web configuration requires the Go gateway')
    if action == 'disable':
        target.update(CHAT_WEB_SEARCH_ENABLED='false', CHAT_WEB_SEARCH_MODEL_VERIFIED='false', CHAT_WEB_SEARCH_VERIFIED_MODEL='')
        return target
    if not key or len(key) > 512 or any(char.isspace() for char in key):
        raise ValueError('Configure the server-only Tavily production environment secret')
    target.update(TAVILY_API_KEY=key, CHAT_WEB_SEARCH_TOOL_PARSER='qwen3_xml')
    if action == 'stage':
        if verified_alias:
            raise ValueError('Parser staging must not assert a verified alias')
        target.update(CHAT_WEB_SEARCH_ENABLED='false', CHAT_WEB_SEARCH_MODEL_VERIFIED='false', CHAT_WEB_SEARCH_VERIFIED_MODEL='')
    elif action in ('enable-base', 'enable-adapter'):
        alias = 'qwen-bf16-base' if action == 'enable-base' else 'qwen-bf16'
        if verified_alias != alias:
            raise ValueError('Enable requires the exact independently verified alias')
        if current.get('CHAT_WEB_SEARCH_TOOL_PARSER') != 'qwen3_xml' or not current.get('TAVILY_API_KEY'):
            raise ValueError('Stage the private parser and provider key before enabling web search')
        target.update(CHAT_WEB_SEARCH_ENABLED='true', CHAT_WEB_SEARCH_MODEL_VERIFIED='true', CHAT_WEB_SEARCH_VERIFIED_MODEL=alias)
    else:
        raise ValueError('Invalid web configuration action')
    return target


def check_web_capability(base, action, environment):
    request = Request(base + '/health', headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_BF16_CHAT_TOKEN'], 'User-Agent': 'qwen-web-deployment-check/1.0'})
    with urlopen(request, timeout=15) as response:
        health = json.load(response)
    web = health.get('webSearch', {})
    profiles = web.get('models', [])
    if health.get('webSearchVersion') != 1 or len(profiles) != 2:
        raise RuntimeError('Versioned web profile capability unavailable')
    if action in ('stage', 'disable'):
        if web.get('reason') != 'disabled' or any(profile.get('available') for profile in profiles):
            raise RuntimeError('Staged or disabled gateway advertised web search readiness')
    else:
        alias = environment['CHAT_WEB_SEARCH_VERIFIED_MODEL']
        if not any(profile.get('model') == alias and profile.get('available') and profile.get('reason') == 'ready' for profile in profiles):
            raise RuntimeError('Verified alias did not advertise web search readiness')
        if any(profile.get('available') for profile in profiles if profile.get('model') != alias):
            raise RuntimeError('Unverified alias advertised web search readiness')


def main():
    profile = os.environ.get('DEPLOY_MODEL_PROFILE', 'bf16')
    pod, chat_token = deployment_target(os.environ, profile)
    image = os.environ['DEPLOY_IMAGE']
    web_action = os.environ.get('DEPLOY_WEB_SEARCH_ACTION', 'keep')
    verified_alias = os.environ.get('DEPLOY_WEB_SEARCH_VERIFIED_ALIAS', '')
    if web_action not in ('keep', 'stage', 'enable-base', 'enable-adapter', 'disable'):
        raise ValueError('Invalid web action or target profile')
    if web_action == 'keep' and verified_alias:
        raise ValueError('Unchanged web settings must not assert a verified alias')
    os.environ['RUNPOD_BF16_CHAT_TOKEN'] = chat_token
    if not POD.fullmatch(pod) or not IMAGE.fullmatch(image):
        raise ValueError('Invalid Pod ID or image digest')
    first = os.environ.get('DEPLOY_FIRST_OFFICIAL_RELEASE', 'false')
    if first not in ('true', 'false'):
        raise ValueError('Invalid deployment profile or first-release setting')
    first_release = first == 'true'
    sequences = os.environ.get('DEPLOY_MAX_NUM_SEQS', '')
    if sequences not in ('', '1', '2'):
        raise ValueError('DEPLOY_MAX_NUM_SEQS must be empty, 1 or 2')
    for name in ('RUNPOD_API_KEY', 'RUNPOD_BF16_CHAT_TOKEN'):
        if not os.environ.get(name):
            raise ValueError(f'Missing {name}')
    base = f'https://{pod}-8001.proxy.runpod.net'
    os.environ['RUNPOD_BF16_CHAT_URL'] = base
    path = '/pods/' + pod
    current = api('GET', path)
    previous = current.get('imageName')
    if not isinstance(previous, str) or not re.fullmatch(r'[A-Za-z0-9_./:@-]+', previous):
        raise RuntimeError('Cannot safely determine the previous image')
    if current.get('desiredStatus') != 'RUNNING':
        raise RuntimeError('Pod must be running before deployment')
    if current.get('env', {}).get('MODEL_PROFILE') != 'bf16':
        raise RuntimeError('Complete the one-time BF16 Pod profile/entrypoint/ports migration first')
    if not first_release and not IMAGE.fullmatch(previous):
        raise RuntimeError('No previous official BF16 image; use first official release mode')
    changes_env = bool(sequences or web_action != 'keep')
    previous_env = current.get('env') if changes_env else None
    if changes_env and (not isinstance(previous_env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in previous_env.items())):
        raise RuntimeError('Cannot safely preserve the current Pod environment')
    target_env = dict(previous_env) if changes_env else None
    if sequences:
        target_env['MAX_NUM_SEQS'] = sequences
    if web_action != 'keep':
        target_env = web_environment(target_env, web_action, verified_alias, os.environ.get('TAVILY_API_KEY', ''))
    if image == previous and target_env == previous_env and not first_release:
        raise RuntimeError('Requested image is already configured')
    summary(f'Pod: `{pod}`; previous image: `{previous}`; requested image: `{image}`; commit: `{os.environ.get("GITHUB_SHA", "local")}`')
    # A timeout during PATCH can leave its outcome unknown, so attempt rollback
    # even if the API did not acknowledge the update.
    try:
        if changes_env:
            summary('Applying allowlisted deployment settings; preserving other environment variables. No credential values are recorded.')
            deploy_image(path, image, base, env=target_env)
            if sequences == '2':
                request = Request(base + '/health', headers={
                    'Authorization': 'Bearer ' + os.environ['RUNPOD_BF16_CHAT_TOKEN'],
                    'User-Agent': 'qwen-chat-deployment-check/1.0',
                })
                with urlopen(request, timeout=15) as response:
                    admission = json.load(response).get('admission', {})
                if admission.get('maxConcurrentRequests') != 2 or admission.get('maxWaitingRequests') != 6:
                    raise RuntimeError('Deployed gateway did not advertise two active and six waiting slots')
                summary('Gateway advertises 2 active requests and 6 waiting requests.')
        else:
            deploy_image(path, image, base)
        if web_action != 'keep':
            check_web_capability(base, web_action, target_env)
            summary(f'Web configuration action `{web_action}` passed its profile capability checks.')
    except Exception as failure:
        if first_release:
            summary('First official BF16 deployment failed; no previous official BF16 release exists. Diagnose the migration and retry in first-release mode; no automatic legacy rollback was attempted.')
            raise RuntimeError('First official BF16 deployment failed') from failure
        summary(f'Deployment failed ({type(failure).__name__}); attempting to restore previous image.')
        try:
            if changes_env:
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
