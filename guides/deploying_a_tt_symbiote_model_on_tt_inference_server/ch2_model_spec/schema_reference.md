# Schema Reference — ModelSpec Dataclasses

This file documents every dataclass and enum in `tt-inference-server`'s ModelSpec configuration system. All definitions live under `workflows/model_spec.py` in the repository root. The classes are used at Python import time to populate the `MODEL_SPECS` dictionary and at runtime to serialize deployment configuration for `run.py` and the vLLM server startup scripts.

---

## `ImplSpec`

`ImplSpec` is an implementation pointer. It identifies where the model code lives — which repository, which subdirectory, and which commit — so that the Docker image and tooling can be audited against a specific source tree.

```python
@dataclass
class ImplSpec:
    impl_id: str        # snake_case identifier used as a lookup key, e.g. "tt_transformers"
    impl_name: str      # human-readable display name, e.g. "tt-transformers"
    repo_url: str       # full HTTPS URL of the implementation repo
    code_path: str      # path within the repo to the model module, e.g. "models/demos/llama3"
```

For a TT Symbiote model the `repo_url` would point to the symbiote repository (e.g. `"https://github.com/tenstorrent/tt-symbiote"`) and `code_path` would be the Python module path inside that repo that contains the model class (e.g. `"tt_symbiote/models/my_model"`).

**Example:**

```python
symbiote_impl = ImplSpec(
    impl_id="tt_symbiote",
    impl_name="tt-symbiote",
    repo_url="https://github.com/tenstorrent/tt-symbiote",
    code_path="tt_symbiote/models/my_symbiote_model",
)
```

---

## `DeviceModelSpec`

`DeviceModelSpec` holds all configuration that is specific to one device type. A single `ModelSpecTemplate` typically carries a list of these — one per supported hardware target — and the system expands them into individual `ModelSpec` instances automatically.

```python
@dataclass
class DeviceModelSpec:
    device: DeviceTypes          # target hardware, from the DeviceTypes enum
    max_concurrency: int         # maximum batch size (number of simultaneous sequences)
    max_context: int             # maximum sequence length in tokens
    vllm_args: dict              # extra CLI arguments appended to the vLLM launch command
    override_tt_config: dict     # key-value overrides passed to the model's init for
                                 # precision and performance tuning
    env_vars: dict               # environment variables injected only for this device
    perf_reference: dict         # optional; maps metric names to target values used by the
                                 # `reports` workflow to determine ModelStatusTypes
```

### `DeviceTypes` enum

The valid values for the `device` field are members of the `DeviceTypes` enum:

| Value | Hardware |
|-------|----------|
| `N150` | Single Wormhole N150 card |
| `N300` | Single Wormhole N300 card (2-chip) |
| `T3K` | Eight-chip Wormhole T3000 server |
| `GALAXY` | Galaxy cluster (multi-T3K) |
| `P100` | Blackhole P100 card |
| `P150` | Blackhole P150 card |
| `P150X4` | Four-card Blackhole configuration |
| `P150X8` | Eight-card Blackhole configuration |
| `P300` | Blackhole P300 card |

### `vllm_args` keys

The `vllm_args` dictionary maps vLLM CLI flag names to string values. The keys use hyphen-separated names that correspond directly to `vllm serve` flags:

| Key | vLLM flag | Notes |
|-----|-----------|-------|
| `"max-model-len"` | `--max-model-len` | Maximum sequence length; must match `max_context` |
| `"max-num-seqs"` | `--max-num-seqs` | Maximum concurrent sequences; must match `max_concurrency` |
| `"block-size"` | `--block-size` | KV cache block size; use `"64"` for Tenstorrent hardware |
| `"max-num-batched-tokens"` | `--max-num-batched-tokens` | Maximum total tokens processed in a single forward pass. For decode-focused deployments set this to `max_concurrency * block_size` (e.g., 32 × 64 = 2048 for a T3K decode config). For prefill-heavy or single-sequence deployments it may equal `max_context`. Do **not** use `max_context * max_concurrency` — that formula produces values far larger than the hardware forward-pass budget. |
| `"enable-chunked-prefill"` | `--enable-chunked-prefill` | Must be `"false"` for TT hardware — chunked prefill is not supported |

All values must be strings even when the underlying flag takes an integer.

### `override_tt_config` keys

The `override_tt_config` dictionary is passed as a JSON string to the TT vLLM plugin's worker initialization, where it is consumed by `device_params_from_override_tt_config()`. Recognized keys:

| Key | Type | Description |
|-----|------|-------------|
| `"optimizations"` | `str` | Selects the optimization profile. Valid values: `"performance"` (maximize throughput) or `"accuracy"` (maximize numerical fidelity). |
| `"is_embedding_model"` | `bool` | Set to `true` for embedding models. When `true` the worker auto-configures `num_command_queues=2` and adjusts L1 cache sizing. |
| `"trace_region_size"` | `int` | Size in bytes reserved for TTNN trace buffers. Default is `50000000` (50 MB). Increase for models with large trace graphs. |
| `"num_command_queues"` | `int` | Number of command queues opened on the device. Normally `1`; set to `2` for embedding workloads. |
| `"worker_l1_size_bytes"` | `int` | Per-worker L1 SRAM allocation in bytes. Controls how much local memory each Tensix core can use during a kernel launch. |

Unrecognized keys in `override_tt_config` are silently ignored, so it is safe to include device-specific keys that are only meaningful for certain hardware types.

### Per-device `env_vars`

The `env_vars` dict on `DeviceModelSpec` is merged on top of the model-level `env_vars` when the final environment is assembled for the Docker container. Common per-device keys:

| Variable | Example value | Purpose |
|----------|---------------|---------|
| `WH_ARCH_YAML` | `"wormhole_b0_80_arch_eth_dispatch.yaml"` | Selects the Wormhole architecture descriptor |
| `MESH_DEVICE` | `"T3K"` | Tells the plugin which hardware mesh to open |
| `ARCH_NAME` | `"wormhole_b0"` | Hardware architecture string for tt-metal |

### `perf_reference`

`perf_reference` is an **optional** flat `dict[str, float]` on `DeviceModelSpec`. It maps metric names to absolute target values (e.g., tokens per second). There is no tier structure inside the dict — it is a single set of targets. The `reports` workflow compares every measured benchmark value against its corresponding entry in this dict to determine the `ModelStatusTypes` level.

In practice the values in `perf_reference` are set to the **Complete-tier performance bar**. The `reports` workflow uses a configurable margin (e.g., 80 % of each target) to grant `Functional` status when results are close but not yet at the full target. A model reaches `Complete` status only when all measured values meet or exceed every value in `perf_reference` directly.

```python
perf_reference={
    "decode_throughput_tps": 4800.0,
    "prefill_throughput_tps": 12000.0,
    "decode_1_prefill_128_decode_128_throughput_tok_s": 1200.0,
    "decode_32_prefill_128_decode_128_throughput_tok_s": 38400.0,
}
```

Keys are metric name strings as produced by the `benchmarks` workflow; values are the Complete-tier float thresholds for that metric. `perf_reference` can be omitted (or set to `{}`) for an initial integration — the model will remain at `Experimental` status and the `reports` workflow will skip threshold validation for that device.

---

## `ModelSpecTemplate`

`ModelSpecTemplate` is a factory. It stores the fields that are shared across all (weight, device) combinations for a model family and produces fully instantiated `ModelSpec` objects via its `expand_to_specs()` method.

```python
@dataclass
class ModelSpecTemplate:
    weights: list[str]                  # list of HuggingFace model repo strings,
                                        # e.g. ["meta-llama/Llama-3.1-8B-Instruct"]
    impl: ImplSpec                      # implementation pointer
    tt_metal_commit: str                # tt-metal commit SHA validated against this spec
    vllm_commit: str                    # tt-vllm fork commit SHA validated against this spec
    inference_engine: InferenceEngine   # which serving backend to use
    device_model_specs: list[DeviceModelSpec]  # one entry per supported device
    uses_tensor_model_cache: bool = False  # set True if the model serializes compiled tensors
                                           # to disk via tt-metal's tensor cache; propagated
                                           # to each expanded ModelSpec during expand_to_specs()
    has_builtin_warmup: bool = False    # set True if initialize_vllm_model() performs its own
                                        # warmup internally; suppresses the framework's
                                        # redundant warmup call on the expanded ModelSpec
```

Calling `expand_to_specs()` takes the Cartesian product of `weights` × `device_model_specs` and returns one `ModelSpec` per combination. The resulting specs are keyed by `model_id` in the global `MODEL_SPECS` dictionary.

> **Note — fields derived during expansion:** `status` and `docker_image` are fields on the *expanded* `ModelSpec` instance that are set by `expand_to_specs()`. `uses_tensor_model_cache` and `has_builtin_warmup` are declared on `ModelSpecTemplate` and propagated to each expanded `ModelSpec` instance during `expand_to_specs()`. Set them directly on `ModelSpecTemplate` before expansion — see the `ModelSpec` section below and the note in `adding_a_model_spec.md`.

### `InferenceEngine` enum

| Value | Use case |
|-------|----------|
| `InferenceEngine.VLLM` | LLM and VLM models served through the vLLM continuous-batching engine |
| `InferenceEngine.MEDIA` | Multi-modal models (image generation, audio, video) served through the TT Media Server |
| `InferenceEngine.FORGE` | CNN-style models using the PyTorch XLA/Forge backend |

For TT Symbiote language models use `InferenceEngine.VLLM`.

---

## `ModelSpec`

`ModelSpec` is the fully expanded, per-weight-per-device instance. It is the object that gets serialized to JSON and passed to the vLLM container via `TT_MODEL_SPEC_JSON_PATH`.

```python
@dataclass
class ModelSpec:
    model_id: str                    # unique programmatic key used as dict key in MODEL_SPECS,
                                     # e.g. "id_tt_symbiote_my-model-8b_n300"
    hf_model_repo: str               # HuggingFace repo used to download weights
    model_name: str                  # human-readable CLI slug matched by run.py --model,
                                     # e.g. "my-model-8b"
    device_type: DeviceTypes         # which device this spec targets
    device_model_spec: DeviceModelSpec  # the device-specific configuration block
    env_vars: dict                   # merged env vars (model-level + device-level)
    uses_tensor_model_cache: bool    # whether the model uses tt-metal's tensor cache;
                                     # derived/defaulted during expand_to_specs()
    has_builtin_warmup: bool         # whether initialize_vllm_model performs its own warmup;
                                     # derived/defaulted during expand_to_specs()
    impl_id: str                     # copied from ModelSpecTemplate.impl.impl_id during
                                     # expansion; used by the server startup script to locate
                                     # and load the correct model implementation module
    docker_image: str                # fully qualified Docker image tag validated for this spec;
                                     # set during expand_to_specs()
    status: ModelStatusTypes         # readiness level; set during expand_to_specs()
```

> **`model_id` vs. `model_name`:** `model_id` is the internal dict key used by the registry (never passed on the CLI). `model_name` is the human-readable slug matched by `run.py --model <name>`. Both fields are set automatically during `expand_to_specs()` — do not hand-write them.

> **`impl_id` derivation:** `ModelSpec.impl_id` is derived from `ModelSpecTemplate.impl.impl_id` via `expand_to_specs()`. Do not set it directly; it is copied automatically from the `ImplSpec` attached to the template.

### `ModelStatusTypes` enum

| Value | Meaning |
|-------|---------|
| `ModelStatusTypes.Experimental` | Initial integration; benchmarks not yet run |
| `ModelStatusTypes.Functional` | All `perf_reference` metric targets are met within a configurable margin (e.g., measured value ≥ 80 % of each target) |
| `ModelStatusTypes.Complete` | All `perf_reference` metric targets are fully met (measured value ≥ 100 % of each target) |

`perf_reference` is a flat dict of absolute metric thresholds — there are no separate "Functional tier" and "Complete tier" sub-dicts. The same set of target values governs both status levels; the difference is whether results satisfy the targets directly (`Complete`) or come within the configurable margin (`Functional`). Thresholds are model-specific absolute values (e.g., tokens/s) set in `DeviceModelSpec.perf_reference`. New TT Symbiote integrations should start with `Experimental` and graduate to `Functional` once benchmarks are run and results fall within the margin of the `perf_reference` targets.

---

## How `TT_MODEL_SPEC_JSON_PATH` is used

When `run.py` launches the Docker container it:

1. Calls `apply_runtime_args()` on the selected `ModelSpec` to apply any CLI-level overrides.
2. Serializes the resulting `ModelSpec` to a JSON file on the host filesystem.
3. Passes the path to that file as the `TT_MODEL_SPEC_JSON_PATH` environment variable into the container.

Inside the container, `run_vllm_api_server.py` reads the file via `_load_model_spec_json()`. It then:

- Extracts the `vllm_args` dict from `device_model_spec` and appends each key-value pair as a `--flag value` argument to the `vllm serve` command.
- Extracts `override_tt_config` and serializes it as a JSON string passed to `--override-tt-config`.
- Sets the hardware mesh environment variables (`MESH_DEVICE`, `WH_ARCH_YAML`, `ARCH_NAME`) from `env_vars`.
- Registers the model's implementation class with vLLM's `ModelRegistry` using `impl_id` to locate the correct module.

The net effect is that every configurable aspect of the vLLM launch — flags, device mesh, hardware tuning, and model registration — flows from the single `ModelSpec` JSON rather than from scattered shell variables.

---

**Next:** [`adding_a_model_spec.md`](./adding_a_model_spec.md)
