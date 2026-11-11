# Serve the selected QLoRA adapter through the BF16 chatbot option

This is a prepared deployment configuration, not a record of a completed rollout.
The exact adapter still needs an actual vLLM load and streamed-chat check.
The existing image is based on vLLM 0.32.0; a new application image containing
this startup change and the updated gateway is required before setting these
environment variables. The current training-only Pod entrypoint must then be
cleared to restore the image's normal serving entrypoint.

## Selected artifact

```text
/workspace/qlora-smoke/runs/20261104T195736282433Z/adapter
```

This is the exported epoch-1 adapter from `checkpoint-25`, selected using the
lowest validation loss in the 100-training/20-validation, three-epoch run.
Its weight file SHA-256 is
`85c5a90007c7340f7a987b595d61ed4d3325305a583d352312d8d51143657cd4`.
All 992 saved tensors were finite and matched that checkpoint exactly.
Rank is 8 and alpha is 16. This is a selected training artifact, not a claim of
better generated answers or an independently verified serving deployment.

The model stays at `/workspace/qwen-bf16/model`. It is loaded in BF16 for
serving; the adapter was trained against an NF4-quantized copy of that same
checkpoint, so compare generated answers in the serving setup too. No merge,
Unsloth serving process or modification to the original weights is needed.

## Configuration for the existing BF16 Pod

Target Pod: `o6f2qoctk798tq` (RTX PRO 6000). Retain its existing image settings,
ports, authentication, diagnostics and persistent volume when applying the new
application image. BF16 is the only supported serving profile.

```dotenv
MODEL_PROFILE=bf16
SERVED_MODEL_NAME=qwen-bf16
VLLM_BASE_MODEL_NAME=qwen-bf16-base
LORA_ADAPTER_PATH=/workspace/qlora-smoke/runs/20261104T195736282433Z/adapter
MAX_LORA_RANK=8
CPU_OFFLOAD_GB=0
MAX_MODEL_LEN=40960
MAX_NUM_SEQS=2
GPU_MEMORY_UTILIZATION=0.90
```

`SERVED_MODEL_NAME` remains the gateway's fixed upstream request model. With
`LORA_ADAPTER_PATH` present, that alias belongs to the adapter. The base alias is
`VLLM_BASE_MODEL_NAME`, defaulting to `${SERVED_MODEL_NAME}-base` in LoRA mode.
These names must differ. Without an adapter, the base name defaults to
`SERVED_MODEL_NAME`; a different explicit base name is rejected so chat cannot
silently target a nonexistent model.

The existing frontend BF16 option expects `qwen-bf16`, so it will use the
adapter without a frontend deployment. The base remains registered privately
as `qwen-bf16-base` for comparison. There is no new frontend model selector.
vLLM's base-model metric label may consequently appear as `qwen-bf16-base`;
the existing Prometheus targets and ports stay the same.

Startup adds `--enable-lora`, `--max-lora-rank 8` and a JSON `--lora-modules`
descriptor with the adapter name, local path and base-model lineage. The raw
vLLM port remains private. Dynamic adapter-loading endpoints are not enabled.
The gateway still exposes only its existing authenticated chat/health routes.

Before services start, `validate-lora-adapter.py` checks the local adapter's
configuration, base-model path, rank capacity, paired linear LoRA tensor shapes,
and complete safetensors header/data layout. It does not load the base, read
every weight value, convert adapter weights, or replace the artifacts. Static
vLLM loading remains the final compatibility check. The BF16 deployment supports this setting; unsupported serving profiles are
rejected before startup.

vLLM 0.32's adapter model card has a `parent` but no context ceiling. The gateway
reads the limit from that exact parent's model card and retains its configured
ceiling. Missing/invalid capacity or an absent adapter fails readiness and chat;
it never falls back to an unrelated model. Tokenization and generation both
continue requesting the adapter alias.

## Activation and verification

1. Build/test and publish a new versioned application image through the existing
   release workflow; record its immutable digest and the rollback image.
2. Apply that image and the settings above to the existing BF16 Pod, clear its
   temporary training-only start command, and restart it. This replaces the
   container-local environment, including `/opt/qlora/venv`. The original model,
   training data, runs and adapter under `/workspace` remain on the attached
   persistent volume. Reinstall the training venv if another training window is
   needed later.
3. Inspect private `/v1/models` from inside the Pod: it must list base
   `qwen-bf16-base` and adapter `qwen-bf16`, with the adapter's root equal to the
   selected path and parent equal to the base alias. Check loading logs for
   adapter failures. Seeing a base response alone does not establish activation.
4. Check authenticated gateway `/health`: ready, model `qwen-bf16`, context
   40960, two active slots and six waiting slots. Check authenticated metrics
   on 8889 and Jupyter on 8888.
5. Send a real streamed gateway request and confirm completion, then check the
   hosted chatbot's BF16 option with a fresh conversation. Compare the base and
   adapter on new questions for factual accuracy, completeness and desired
   teaching style. Saved training loss is not a substitute for this check.

Weight loading remains subject to the existing global-volume throughput. This
change does not copy the BF16 model to the container disk or speed up that load.

## Choose the base or adapter per request

With the updated gateway and the existing LoRA environment settings, authenticated
`/health` advertises `adapterToggleVersion: 1` and `adapterToggleAvailable: true`
only when the adapter's registered parent matches the configured base alias and
both capacities can be verified. Gateways without a configured adapter advertise
version 0 and do not accept this option. No additional Pod environment variable
is needed to enable switching.

Chat requests may include `"useAdapter": true` (adapter) or `"useAdapter": false`
(base). Omitting it preserves the existing default adapter route. Values must
be JSON booleans; a browser-supplied model alias never selects a destination.
The gateway chooses only its configured adapter/base pair and uses that same
alias for capacity checks, tokenization, generation and diagnostic metadata.
Successful streams confirm the selection in `X-Chat-Adapter-Enabled: true|false`.
Both modes share the same admission queue, timeout and rate limit.

Deploy the companion frontend to expose its BF16-only **Use trained adapter**
switch. It saves the preference per conversation and the mode per new answer.
The mode changes between requests without restarting the engine or reloading
weights. Installing the gateway update still uses the normal image rollout.
An older frontend keeps using the existing default adapter route; the new
frontend hides the switch when an older gateway does not advertise support.

The local fixture can exercise the feature without a GPU by setting
`CHAT_FIXTURE_ADAPTER=true` for `scripts/chat_test_fixture.py`. This fixture is
test-only and must not be used as the serving entrypoint.

## Rollback

Remove **both** `LORA_ADAPTER_PATH` and `VLLM_BASE_MODEL_NAME`, then restart with
the accepted application image (or restore the previously recorded image digest).
Keep `SERVED_MODEL_NAME=qwen-bf16` and the original limits. The gateway then
targets the original BF16 base under its previous alias. Leaving a distinct
base-name override without an adapter intentionally refuses startup. Keep the
adapter and training run on the volume for later use.

Sources: [vLLM 0.32 LoRA serving](https://docs.vllm.ai/en/v0.32.0/features/lora/),
[adapter model cards](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/models/serving.py),
[supported rank capacities](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/lora.py).
