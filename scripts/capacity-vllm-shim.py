#!/usr/bin/env python3
"""Installed only as the owned temporary PATH entry bin/vllm."""
import hashlib
import json
import os
from pathlib import Path
import sys
from combined_capacity_runtime import atomic, secure_root, test_argv, value

root = secure_root(os.environ['CAPACITY_SESSION_ROOT'])
frozen = json.loads((root / 'args-contract.json').read_text())
args = test_argv(['serve', *sys.argv[2:]] if sys.argv[1:2] == ['serve'] else sys.argv[1:], frozen)
atomic(root / 'launch.json', {'max_num_seqs': int(value(args, '--max-num-seqs')),
       'max_num_batched_tokens': int(value(args, '--max-num-batched-tokens')),
       'compilation_config': json.loads(value(args, '--compilation-config')),
       'kv_transfer_config': json.loads(value(args, '--kv-transfer-config')),
       'histogram_buckets': json.loads(value(args, '--custom-histogram-buckets')),
       'original_contract_sha256': hashlib.sha256((root / 'args-contract.json').read_bytes()).hexdigest(),
       'native_argv_nonallowlisted_values_preserved': True})
os.execv('/usr/local/bin/vllm', ['/usr/local/bin/vllm', *args])
