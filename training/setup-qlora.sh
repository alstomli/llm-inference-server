#!/usr/bin/env bash
set -euo pipefail
root="${QLORA_ENV_ROOT:-/opt/qlora}"
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$root"
# Reuse the image's existing CUDA/PyTorch binaries without changing them.
python3 -m venv --system-site-packages "$root/venv"
python3 - "$root/serving-constraints.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import sys
pins = []
for package in importlib.metadata.distributions():
    name = package.metadata['Name']
    if name.lower() in ('torch', 'torchvision', 'torchaudio', 'triton', 'xformers'):
        pins.append(f'{name}=={package.version}')
Path(sys.argv[1]).write_text('\n'.join(sorted(set(pins))) + '\n')
PY
"$root/venv/bin/python" -m pip install --progress-bar off \
  --constraint "$root/serving-constraints.txt" -r "$source_dir/requirements-qlora.txt"
"$root/venv/bin/python" -m pip freeze > "$root/packages.freeze.txt"
