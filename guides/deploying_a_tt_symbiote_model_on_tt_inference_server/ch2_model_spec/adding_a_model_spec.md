# Adding a ModelSpec for a TT Symbiote Model

This page walks through every step needed to register a TT Symbiote model in `tt-inference-server`. All edits go into `workflows/model_spec.py` unless noted. Commit hashes, HuggingFace repo names, and performance values in the examples below are placeholders — replace them with real values from your integration.

---

## Step 1 — Define an `ImplSpec`

Create one `ImplSpec` for the symbiote codebase. This object is reused across all weight variants and device types.

```python
from workflows.model_spec import ImplSpec

symbiote_impl = ImplSpec(
    impl_id="tt_symbiote",           # snake_case; used as a registry key and in model_id
    impl_name="tt-symbiote",         # human-readable name shown in reports
    repo_url="https://github.com/tenstorrent/tt-symbiote",
    code_path="tt_symbiote/models/my_symbiote_model",  # path within the repo
)
```

The `impl_id` must be unique across all entries in `model_spec.py`. It is also used by `run_vllm_api_server.py` to select the correct model registration branch when multiple implementations coexist in the same Docker image.

---

## Step 2 — Define a `DeviceModelSpec` per supported device

Create one `DeviceModelSpec` for each hardware target the symbiote model supports. The values below are starting points for an 8 B parameter model on N300 and T3K; tune them based on actual profiling.

### N300 entry

```python
from workflows.model_spec import DeviceModelSpec, DeviceTypes

symbiote_8b_n300 = DeviceModelSpec(
    device=DeviceTypes.N300,
    max_concurrency=1,          # maximum simultaneous requests; raise after perf validation
    max_context=32768,          # maximum sequence length in tokens
    vllm_args={
        "max-model-len":            "32768",
        "max-num-seqs":             "1",
        "block-size":               "64",     # required for Tenstorrent KV cache layout
        "max-num-batched-tokens":   "32768",  # see schema_reference.md for formula options
        "enable-chunked-prefill":   "false",  # must be false; TT hardware does not support it
    },
    override_tt_config={
        "optimizations": "performance",   # "performance" or "accuracy"
    },
    env_vars={
        "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
        "MESH_DEVICE":  "N300",
        "ARCH_NAME":    "wormhole_b0",
    },
)
```

### T3K entry

```python
symbiote_8b_t3k = DeviceModelSpec(
    device=DeviceTypes.T3K,
    max_concurrency=32,
    max_context=131072,
    vllm_args={
        "max-model-len":            "131072",
        "max-num-seqs":             "32",
        "block-size":               "64",
        "max-num-batched-tokens":   "2048",   # decode: max_concurrency * block_size = 32 * 64
        "enable-chunked-prefill":   "false",
    },
    override_tt_config={
        "optimizations":      "performance",
        "trace_region_size":  80000000,    # 80 MB; increase if trace compilation fails
    },
    env_vars={
        "WH_ARCH_YAML": "wormhole_b0_80_arch_eth_dispatch.yaml",
        "MESH_DEVICE":  "T3K",
        "ARCH_NAME":    "wormhole_b0",
    },
)
```

---

## Step 3 — Create the `ModelSpecTemplate`

Assemble the template with the weight repos, the `ImplSpec`, and the list of `DeviceModelSpec` entries.

```python
from workflows.model_spec import ModelSpecTemplate, InferenceEngine, ModelStatusTypes

symbiote_8b_template = ModelSpecTemplate(
    weights=[
        "myorg/my-symbiote-model-8b",   # HuggingFace repo string for the base model
        # add instruct / chat variants here if they share the same impl
    ],
    impl=symbiote_impl,
    tt_metal_commit="<tt_metal_commit_sha>",  # commit tested with this spec
    vllm_commit="<vllm_commit_sha>",          # tt-vllm fork commit tested with this spec
    inference_engine=InferenceEngine.VLLM,
    device_model_specs=[
        symbiote_8b_n300,
        symbiote_8b_t3k,
    ],
)
```

Finally, append the template to the module-level `spec_templates` list so it is picked up by `expand_to_specs()`:

```python
spec_templates = [
    # ... existing entries ...
    symbiote_8b_template,
]
```

### How `spec_templates` connects to `MODEL_SPECS`

Adding a `ModelSpecTemplate` to `spec_templates` is what causes your model to appear in `MODEL_SPECS`. At module load time, `model_spec.py` calls `expand_to_specs()` on each template in `spec_templates` and flattens the results into the global `MODEL_SPECS` dict, keyed by `model_id`:

```python
MODEL_SPECS = {
    spec.model_id: spec
    for template in spec_templates
    for spec in template.expand_to_specs()
}
```

`expand_to_specs()` takes the Cartesian product of `weights` × `device_model_specs` on the template and returns one fully instantiated `ModelSpec` per combination. Fields such as `status` and `docker_image` are derived or defaulted during expansion. If your model is not showing up in `run.py`, verify that the template is present in `spec_templates` and that `expand_to_specs()` produces at least one entry (i.e., both `weights` and `device_model_specs` are non-empty).

---

### `uses_tensor_model_cache` and `has_builtin_warmup`

`uses_tensor_model_cache` and `has_builtin_warmup` are declared on `ModelSpecTemplate` and propagated to each expanded `ModelSpec` instance during `expand_to_specs()`. You should explicitly set them on the `ModelSpecTemplate` if your model requires non-default behaviour:

- **`uses_tensor_model_cache`** — set to `True` if the model uses tt-metal's tensor model cache (i.e., it serializes compiled tensors to disk for faster subsequent loads). Defaults to `False`.
- **`has_builtin_warmup`** — set to `True` if the model's `initialize_vllm_model()` performs its own warmup internally (so the serving framework should not issue a redundant warmup call). Defaults to `False`.

If neither flag applies, no action is needed — the defaults are safe for standard integrations.

---

## `model_id` naming conventions

The system derives `model_id` automatically from the template fields using the pattern:

```
id_{impl_id}_{model_name_slug}_{device_type_lowercase}
```

For the N300 entry above this produces `id_tt_symbiote_my-symbiote-model-8b_n300`. The conventions to follow:

- `impl_id` is lowercase and snake_case.
- The model name slug is the HuggingFace repo name after the `/`, lowercased and with underscores replaced by hyphens.
- The device suffix is the enum name lowercased: `n150`, `n300`, `t3k`, `galaxy`, `p150`, etc.

Do not hand-write `model_id` strings; they are computed to guarantee uniqueness across the registry.

---

## `vllm_args` in detail

Every key in `vllm_args` maps one-to-one to a `vllm serve` CLI flag by prepending `--`.

See the [`vllm_args` key reference](./schema_reference.md#vllm_args-keys) in Chapter 2's schema reference for the full flag list.

All values are strings. Integer flags like `max-model-len` still need to be passed as string literals in the dict.

---

## `override_tt_config` in detail

The `override_tt_config` dict is serialized to a JSON string and passed to the TT vLLM plugin's worker initialization. The worker's `device_params_from_override_tt_config()` function consumes it.

See the [`override_tt_config` key reference](./schema_reference.md#override_tt_config-keys) for the full key list.

An empty dict `{}` is valid and tells the worker to use all defaults.

---

## Required environment variables for first deployment

The following variables must be present in your shell or in a `.env` file at the repository root before running `run.py` for the first time:

| Variable | Purpose | Notes |
|----------|---------|-------|
| `HF_TOKEN` | Authenticates with HuggingFace Hub to download gated model weights | Required for any model behind an access gate; even public models may need it for high-volume downloads. |
| `JWT_SECRET` | Shared secret used by the vLLM server to validate Bearer tokens on the OpenAI-compatible API | Can be any string; set it consistently across the server and any clients. |
| `SERVICE_PORT` | TCP port the vLLM HTTP server binds to | Defaults to `8000`. Override when that port is already in use or when running multiple servers in parallel. |

The `run.py` setup step prompts for `HF_TOKEN` and `JWT_SECRET` on first run and writes them to `.env` for subsequent invocations. `SERVICE_PORT` can also be supplied via `--service-port` on the command line, which takes precedence over the environment variable.

---

**Next:** [`workflow_commands.md`](./workflow_commands.md)
