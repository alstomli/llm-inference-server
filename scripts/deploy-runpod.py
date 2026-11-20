"""Update one Runpod image and restore the previous image if verification fails."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE = re.compile(r'zihan000/qwen-inference@sha256:[0-9a-f]{64}\Z')
POD = re.compile(r'[A-Za-z0-9_-]+\Z')
SCRIPT = Path(__file__).with_name('check-chat-deployment.py')
CONCURRENCIES = tuple(str(n) for n in range(2, 9))
# Runpod REST API v2; v1 (rest.runpod.io) was retired in November 2026.
API_BASE = 'https://api.runpod.io/v2'


class Profile(NamedTuple):
    settings: str  # prefix of the per-profile Pod ID, chat URL and chat token settings
    alias: str  # the gateway's model: base-plus-adapter on bf16, the base on 4bit
    base_alias: str | None  # the separate base alias; None when the profile has no adapter
    label: str

    @property
    def web_aliases(self):
        """Web-search enable action -> the alias it verifies; one web-search model per alias."""
        if self.base_alias is None:
            return {'enable-base': self.alias}
        return {'enable-base': self.base_alias, 'enable-adapter': self.alias}


PROFILES = {'bf16': Profile('RUNPOD_BF16', 'qwen-bf16', 'qwen-bf16-base', 'BF16'),
            '4bit': Profile('RUNPOD_4BIT', 'qwen-4bit', None, '4-bit')}


def api(method, path, body=None):
    request = Request(API_BASE + path,
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


def deploy_image(path, image, base, env=None, concurrency=None, profile='bf16'):
    # Runpod's update operation resets the Pod; do not separately restart it.
    update = {'image': image}
    if env is not None:
        update['env'] = env
    api('PATCH', path, update)
    check_transition(base)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        reported = api('GET', path)
        if reported.get('image') == image and (env is None or reported.get('env') == env):
            break
        time.sleep(3)
    else:
        raise RuntimeError('Runpod did not report the requested image/environment')
    wait = 3600
    checker = [sys.executable, str(SCRIPT), '--wait', str(wait), '--profile', profile,
               '--expected-model', PROFILES[profile].alias, '--read-timeout', '180']
    if concurrency is not None:
        checker += ['--expected-concurrency', str(concurrency)]
    subprocess.run(checker, check=True, timeout=wait + 900)


def deployment_target(env, profile='bf16'):
    if profile not in PROFILES:
        raise ValueError('The deployment profile must be bf16 or 4bit')
    settings = PROFILES[profile].settings
    pod = env.get(settings + '_POD_ID', '')
    token = env.get(settings + '_CHAT_TOKEN', '')
    if not pod or not token:
        raise ValueError(f'Configure {settings}_POD_ID and {settings}_CHAT_TOKEN for this environment')
    if env.get('RUNPOD_BF16_POD_ID') == env.get('RUNPOD_4BIT_POD_ID'):
        raise ValueError('The BF16 and 4-bit profiles must use different Pods')
    return pod, token


def web_environment(current, action, verified_alias, key, profile='bf16'):
    """Only these named operations may change the server-owned web settings."""
    target = dict(current)
    expected = PROFILES[profile]
    base_names = (expected.base_alias,) if expected.base_alias else (None, expected.alias)
    if current.get('MODEL_PROFILE') != profile or current.get('SERVED_MODEL_NAME') != expected.alias or current.get('VLLM_BASE_MODEL_NAME') not in base_names:
        raise ValueError(f'Web configuration requires the existing {expected.label} aliases')
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
        alias = expected.web_aliases.get(action)
        if alias is None or verified_alias != alias:
            raise ValueError('Enable requires the exact independently verified alias')
        if current.get('CHAT_WEB_SEARCH_TOOL_PARSER') != 'qwen3_xml' or not current.get('TAVILY_API_KEY'):
            raise ValueError('Stage the private parser and provider key before enabling web search')
        target.update(CHAT_WEB_SEARCH_ENABLED='true', CHAT_WEB_SEARCH_MODEL_VERIFIED='true', CHAT_WEB_SEARCH_VERIFIED_MODEL=alias)
    else:
        raise ValueError('Invalid web configuration action')
    return target


def check_web_capability(base, action, environment, profile='bf16'):
    expected = PROFILES[profile]
    request = Request(base + '/health', headers={'Authorization': 'Bearer ' + os.environ[expected.settings + '_CHAT_TOKEN'], 'User-Agent': 'qwen-web-deployment-check/1.0'})
    with urlopen(request, timeout=15) as response:
        health = json.load(response)
    web = health.get('webSearch', {})
    models = web.get('models', [])
    if health.get('webSearchVersion') != 1 or len(models) != len(expected.web_aliases):
        raise RuntimeError('Versioned web profile capability unavailable')
    if action in ('stage', 'disable'):
        if web.get('reason') != 'disabled' or any(model.get('available') for model in models):
            raise RuntimeError('Staged or disabled gateway advertised web search readiness')
    else:
        alias = environment['CHAT_WEB_SEARCH_VERIFIED_MODEL']
        if not any(model.get('model') == alias and model.get('available') and model.get('reason') == 'ready' for model in models):
            raise RuntimeError('Verified alias did not advertise web search readiness')
        if any(model.get('available') for model in models if model.get('model') != alias):
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
    if web_action.startswith('enable-') and PROFILES[profile].web_aliases.get(web_action) != verified_alias:
        raise ValueError(f'{web_action} on the {profile} profile requires its exact independently verified alias')
    settings, label = PROFILES[profile].settings, PROFILES[profile].label
    os.environ[settings + '_CHAT_TOKEN'] = chat_token
    if not POD.fullmatch(pod) or not IMAGE.fullmatch(image):
        raise ValueError('Invalid Pod ID or image digest')
    first = os.environ.get('DEPLOY_FIRST_OFFICIAL_RELEASE', 'false')
    if first not in ('true', 'false'):
        raise ValueError('Invalid deployment profile or first-release setting')
    first_release = first == 'true'
    sequences = os.environ.get('DEPLOY_MAX_NUM_SEQS', '')
    if sequences not in ('',) + CONCURRENCIES:
        raise ValueError('DEPLOY_MAX_NUM_SEQS must be empty or an integer from 2 through 8')
    if not os.environ.get('RUNPOD_API_KEY'):
        raise ValueError('Missing RUNPOD_API_KEY')
    base = f'https://{pod}-8001.proxy.runpod.net'
    os.environ[settings + '_CHAT_URL'] = base
    path = '/pods/' + pod
    current = api('GET', path)
    previous = current.get('image')
    if not isinstance(previous, str) or not re.fullmatch(r'[A-Za-z0-9_./:@-]+', previous):
        raise RuntimeError('Cannot safely determine the previous image')
    if current.get('status') != 'RUNNING':
        raise RuntimeError('Pod must be running before deployment')
    if current.get('env', {}).get('MODEL_PROFILE') != profile:
        raise RuntimeError(f'The {label} Pod must have MODEL_PROFILE={profile}; complete its one-time profile/entrypoint/ports setup first')
    if not first_release and not IMAGE.fullmatch(previous):
        raise RuntimeError(f'No previous official {label} image; use first official release mode')
    changes_env = bool(sequences or web_action != 'keep')
    previous_env = current.get('env') if changes_env else None
    if changes_env and (not isinstance(previous_env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in previous_env.items())):
        raise RuntimeError('Cannot safely preserve the current Pod environment')
    target_env = dict(previous_env) if changes_env else None
    if sequences:
        # Gateway slots then follow the engine: N active and 3N waiting.
        target_env['MAX_NUM_SEQS'] = sequences
        for key in ('CHAT_MAX_CONCURRENT', 'CHAT_MAX_WAITING'):
            target_env.pop(key, None)
    if web_action != 'keep':
        target_env = web_environment(target_env, web_action, verified_alias, os.environ.get('TAVILY_API_KEY', ''), profile)
    if image == previous and target_env == previous_env and not first_release:
        raise RuntimeError('Requested image is already configured')
    summary(f'Pod: `{pod}`; previous image: `{previous}`; requested image: `{image}`; commit: `{os.environ.get("GITHUB_SHA", "local")}`')
    # A timeout during PATCH can leave its outcome unknown, so attempt rollback
    # even if the API did not acknowledge the update.
    try:
        if changes_env:
            summary('Applying allowlisted deployment settings; preserving other environment variables. No credential values are recorded.')
            deploy_image(path, image, base, env=target_env, concurrency=int(sequences) if sequences else None, profile=profile)
        else:
            deploy_image(path, image, base, profile=profile)
        if web_action != 'keep':
            check_web_capability(base, web_action, target_env, profile)
            summary(f'Web configuration action `{web_action}` passed its profile capability checks.')
    except Exception as failure:
        if first_release:
            summary(f'First official {label} deployment failed; no previous official {label} release exists. Diagnose the migration and retry in first-release mode; no automatic legacy rollback was attempted.')
            raise RuntimeError(f'First official {label} deployment failed') from failure
        summary(f'Deployment failed ({type(failure).__name__}); attempting to restore previous image.')
        try:
            if changes_env:
                deploy_image(path, previous, base, env=previous_env, profile=profile)
            else:
                deploy_image(path, previous, base, profile=profile)
            summary('Rollback passed the live chat checks.')
        except Exception as rollback:
            summary(f'ROLLBACK FAILED ({type(rollback).__name__}): inspect the Pod immediately. Previous image: `{previous}`')
        raise RuntimeError('Deployment failed; inspect workflow logs and rollback status') from failure
    summary(f'First official {label} release passed the live chat checks.' if first_release else 'Deployment passed the live chat checks.')


if __name__ == '__main__':
    main()
