# Environment Variables Reference

This file is the complete reference for every environment variable that affects a TT Symbiote model deployment on `tt-inference-server`. The table gives a one-line summary; the sections below expand on each variable's behavior and any gotchas specific to TT Symbiote models.

## Quick Reference Table

| Variable | Set by | Purpose |
|---|---|---|
| `VLLM_USE_V1` | operator / `serve.sh` | Must be `"1"` for `tt-vllm-plugin` |
| `HF_MODEL` | operator / `serve.sh` | HuggingFace model repo used as `--model` arg |
| `HF_TOKEN` | operator | HuggingFace auth token for gated model downloads |
| `JWT_SECRET` | operator | Seeds the vLLM API key |
| `SERVICE_PORT` | operator | HTTP port (default `8000`) |
| `MESH_DEVICE` | workflow / operator | Device type string: `"N150"`, `"N300"`, `"T3K"`, `"(1,8)"`, etc. |
| `TT_VISIBLE_DEVICES` | workflow | Comma-separated physical device IDs |
| `TT_MODEL_SPEC_JSON_PATH` | workflow | Path to JSON-serialized `ModelSpec` for current deployment |
| `VLLM_API_KEY` | workflow | Derived from `JWT_SECRET` if not set |
| `TT_LLAMA_TEXT_VER` | server startup | Selects Llama implementation variant |
| `TT_QWEN3_TEXT_VER` | server startup | Selects Qwen3 implementation variant |
| `TT_SYMBIOTE_RUN_MODE` | model / operator | TT Symbiote run mode (`NORMAL`, `DPL`, `SEL`, `CPU`) |
| `TT_SYMBIOTE_DISPATCHER` | model / operator | Active dispatcher selection |

---

## Variable Descriptions

### `VLLM_USE_V1`

**Must be set to `"1"`.** The `tt-vllm-plugin` is built against the vLLM v1 engine architecture, which introduced a redesigned scheduler, async execution loop, and the `TTWorker` interface. If this variable is absent or set to `"0"`, vLLM will load its legacy v0 engine and the plugin's worker class will not be registered, causing an immediate startup failure.

```bash
export VLLM_USE_V1=1
```

### `HF_MODEL`

The HuggingFace Hub repository ID (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`) or a local filesystem path to a model directory. This value is forwarded as the `--model` argument to the vLLM engine. Inside `initialize_vllm_model`, the `hf_model_path` parameter is derived from this variable after the engine resolves it to a local path via the HuggingFace cache.

For TT Symbiote models, `HF_MODEL` must point to the same architecture that the model implementation expects. A mismatch between the repo's `config.json` and what the model class reads will typically surface as a shape error during weight loading.

### `HF_TOKEN`

A HuggingFace access token (starting with `hf_`). Required for gated model repositories such as Llama 3 family models. The token is used by the `huggingface_hub` library when downloading model weights and tokenizer files. If the model's weights are already present in the local HuggingFace cache (`~/.cache/huggingface/hub/`) from a prior download, this token is not needed at inference time — but it is still required for the initial pull.

Set this in a `.env` file or as a secret in the deployment environment. Never hard-code it in source files.

### `JWT_SECRET`

A secret string used to seed the API key that the vLLM server enforces on incoming requests. If `VLLM_API_KEY` is not explicitly set, the server workflow derives the API key from `JWT_SECRET`. Clients must send `Authorization: Bearer <derived-key>` with every request.

For TT Symbiote model developers running local integration tests, a placeholder value such as `JWT_SECRET=test-secret` is sufficient. The exact derivation algorithm is in the `tt-inference-server` workflow scripts.

### `SERVICE_PORT`

The TCP port on which the vLLM HTTP server listens. Defaults to `8000`. When running multiple model servers on the same host (e.g., separate N150 cards for different models), each deployment must use a different `SERVICE_PORT`.

### `MESH_DEVICE`

Controls the mesh shape passed to `ttnn.open_mesh_device()`. See [`device_lifecycle.md`](./device_lifecycle.md) for the full mapping table. For TT Symbiote model developers, the key behavior is:

- The model receives a `mesh_device` with exactly the shape implied by this variable.
- If your model requires a specific mesh shape (e.g., `1×8` for a T3K-optimized attention kernel), assert it at the start of `initialize_vllm_model`.
- If `MESH_DEVICE` is not set, the worker uses a default appropriate for the detected hardware, but relying on this default is fragile in multi-card deployments.

### `TT_VISIBLE_DEVICES`

Controls which physical Tenstorrent device IDs are visible to the process. The operator sets this variable in the process environment before launching the worker — the server workflow's container or process launch configuration places it there. `TTWorker.__init__` calls `_configure_visible_devices()` to read the existing value and apply the device filter during initialization; the worker does not override a pre-set `TT_VISIBLE_DEVICES`, it reads it. Models must not modify this variable.

On a multi-device host with multiple N300 modules, individual modules can be isolated by listing only their IDs. Each N300 card contains 2 Wormhole dies, each presenting as a separate device ID, so isolating one N300 card requires listing both of its IDs (e.g., `TT_VISIBLE_DEVICES=0,1` for the first N300 card, `TT_VISIBLE_DEVICES=2,3` for the second). A T3K cannot be split into sub-groups; all 8 device IDs must be listed together.

See [`device_lifecycle.md`](./device_lifecycle.md) for full mesh-grid mapping and device-ID conventions.

### `TT_MODEL_SPEC_JSON_PATH`

Filesystem path to a JSON file containing a serialized `ModelSpec` object. The `ModelSpec` describes the deployment configuration: model architecture, quantization settings, target hardware, and tuning parameters specific to this combination. The worker reads `TT_MODEL_SPEC_JSON_PATH` at startup and passes the deserialized `ModelSpec` to `initialize_vllm_model` (or makes it accessible via the model config dict).

For TT Symbiote models, the `ModelSpec` is the primary mechanism for communicating hardware-specific tuning knobs (e.g., which kernel variant to use, what sequence length paddings to apply) without hard-coding them in the model source.

### `VLLM_API_KEY`

The API key that vLLM enforces on HTTP requests. If not explicitly set, the server workflow derives it from `JWT_SECRET`. When running automated tests against a locally deployed server, set `VLLM_API_KEY` directly to avoid depending on the derivation logic:

```bash
export VLLM_API_KEY=my-test-key
# Then in test client:
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer my-test-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/...", "prompt": "Hello", "max_tokens": 32}'
```

### `TT_LLAMA_TEXT_VER`

Selects which Llama text-generation implementation variant the server loads. The `tt-vllm-plugin` ships multiple Llama implementations that differ in supported sequence lengths, quantization schemes, or hardware targets. The server startup script inspects `TT_LLAMA_TEXT_VER` and imports the corresponding module.

TT Symbiote Llama models must be registered under a version string that matches what this variable is expected to contain for the target deployment. If a new variant is added, its version string must be documented alongside this variable.

### `TT_QWEN3_TEXT_VER`

Analogous to `TT_LLAMA_TEXT_VER` but for Qwen3 model implementations. The same versioning conventions apply. If deploying a TT Symbiote Qwen3 model, verify that the value of `TT_QWEN3_TEXT_VER` in the deployment environment matches the version string under which the model is registered in the server's model registry.

### `TT_SYMBIOTE_RUN_MODE`

Controls the TT Symbiote runtime execution mode. Accepted values:

| Value | Behavior |
|---|---|
| `NORMAL` | Standard on-device inference (default) |
| `DPL` | Data-parallel mode; splits the batch across chips for higher throughput when serving concurrent requests |
| `SEL` | Selective execution mode; skips layers or experts that are predicted to be unimportant for the current input |
| `CPU` | Runs the model on CPU using reference implementations; used for correctness debugging only, not for production |

The model implementation reads `TT_SYMBIOTE_RUN_MODE` to branch into the appropriate execution path. When running integration tests against `tt-inference-server`, keep this at `NORMAL` unless specifically testing DPL or SEL behavior.

### `TT_SYMBIOTE_DISPATCHER`

Selects the active TT Symbiote dispatcher, which controls how operations are scheduled and batched for submission to the device. The set of valid values is model-family-specific and defined in each model's dispatcher registry.

In most standard deployments this variable does not need to be set explicitly — the model defaults to its recommended dispatcher for the detected hardware. Set it explicitly only when benchmarking dispatcher variants or when the default selection is known to be suboptimal for a specific `MESH_DEVICE` configuration.

---

**Next:** [Chapter 6 — Integration Checklist and Worked Example](../ch6_integration_checklist/index.md)
