# serving_stack.md

This document describes the static architecture of the `tt-inference-server` serving stack: which layers exist, how each layer connects to the next, and what each layer is responsible for. The companion document [`request_lifecycle.md`](./request_lifecycle.md) describes how a request flows through this stack at runtime.

## The three-layer serving stack

The stack is composed of three layers stacked vertically.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — vLLM OpenAI API server                               │
│  vllm.entrypoints.openai.api_server                             │
│  Exposes /v1/completions, /v1/chat/completions over HTTP        │
│  Owns tokenization, scheduling, and sampling logic              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2 — TTPlatform vLLM platform plugin                      │
│  tt_vllm_plugin.platform.tt_platform.TTPlatform                 │
│  Registers Tenstorrent hardware with vLLM's platform registry   │
│  Renames model architecture strings, enforces constraints       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3 — Model implementation                                 │
│  e.g., TTLlamaForCausalLM in generator_vllm.py                  │
│  Implements initialize_vllm_model(), prefill_forward(),         │
│  decode_forward() — runs on ttnn mesh device                    │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1 — vLLM OpenAI API server

The top layer is the unmodified (or lightly patched) vLLM OpenAI API server, launched by the workflow orchestrator via:

```python
import runpy
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

Before this call, the workflow orchestrator reads the active `DeviceModelSpec` and prepends its `vllm_args` list to `sys.argv` so that vLLM's argument parser sees the device-specific flags (e.g., `--block-size`, `--max-model-len`, `--tensor-parallel-size`) as if they had been typed on the command line.

A representative `sys.argv` injection looks like:

```python
import sys

device_args = model_spec.vllm_args  # e.g. ["--block-size", "64", "--max-model-len", "131072"]
sys.argv = sys.argv[:1] + device_args + sys.argv[1:]
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

The API server then owns the full HTTP lifecycle: parsing requests, running the tokenizer, feeding the scheduler, collecting generated tokens, detokenizing, and returning the response. The model implementation never sees raw text.

### Layer 2 — TTPlatform vLLM platform plugin

`TTPlatform` is a subclass of vLLM's `Platform` base class, registered via the `tt-vllm-plugin` package's entry point. When vLLM initialises, it queries the active platform for hardware capabilities; `TTPlatform` intercepts those queries to enforce Tenstorrent-specific constraints.

#### `VLLM_USE_V1=1` requirement

The `tt-vllm-plugin` only supports vLLM's v1 engine architecture. The `TTPlatform.supports_v1()` method returns `True` unconditionally, but the serving stack also requires `VLLM_USE_V1=1` to be set in the environment before the vLLM process starts. If the environment variable is absent or set to `0`, the v1 engine will not be selected and `TTPlatform`'s worker and loader registrations will not be invoked correctly.

> **Warning:** `VLLM_USE_V1` must be set in the shell environment *before* the Python process starts — do **not** set it via `os.environ["VLLM_USE_V1"] = "1"` inside a script after any `vllm` module has been imported. vLLM reads this variable at import time; if any `vllm` module is imported before the variable is set, the wrong engine is silently selected and no error is raised.

#### Architecture string renaming

vLLM resolves a model class by reading the `architectures` field from a HuggingFace `config.json`. For a LLaMA model, this field typically reads `["LlamaForCausalLM"]`. `TTPlatform` intercepts the class lookup and renames the architecture string:

```python
# Inside TTPlatform.get_model_arch_name() or equivalent hook
ARCH_RENAMES = {
    "LlamaForCausalLM": "TTLlamaForCausalLM",
    "MistralForCausalLM": "TTMistralForCausalLM",
    # ... other supported architectures
}
```

This redirection ensures that vLLM loads the TT-specific model class (e.g., `TTLlamaForCausalLM` from `generator_vllm.py`) instead of vLLM's built-in CUDA implementation.

#### Capability constraints enforced by `TTPlatform`

`TTPlatform` disables several vLLM features that are not yet supported on Tenstorrent hardware:

| Feature | Status | Notes |
|---------|--------|-------|
| Chunked prefill | Disabled | `supports_chunked_prefill()` returns `False` |
| Prefix caching | Disabled | `supports_prefix_caching()` returns `False` |
| Speculative decoding | Disabled | Not yet implemented in the TT worker path |
| `n > 1` (multiple completions per prompt) | Rejected | `TTPlatform.check_and_update_config()` validates and rejects requests with `n > 1` |
| `best_of` | Rejected | Not supported; requests with `best_of > 1` are rejected |
| `prompt_logprobs` | Rejected | Not available; the model returns sampled tokens only |

These constraints are checked at request validation time, before any tokens reach the model.

### Layer 3 — Model implementation

The model implementation is the only layer that touches Tenstorrent hardware directly. It must satisfy the interface that `TTModelLoader` and the TT worker expect.

## TTWorker

`TTWorker` is the vLLM worker class for Tenstorrent hardware, implemented in `tt_worker.py`. At startup it calls `open_mesh_device()` to initialise the `ttnn.MeshDevice` and then triggers the vLLM loader pipeline. `TTModelLoader` (not `TTWorker` directly) calls `initialize_vllm_model()` on the resolved model class. `TTWorker` also invokes `allocate_kv_cache` to pre-allocate the KV cache pool on device. During inference, `TTWorker` receives scheduling steps from the vLLM engine, builds the `TTModelInput` for each step, and dispatches the forward call through `TTModelRunner`.

## TTModelRunner

`TTModelRunner` is the component inside `TTWorker` responsible for dispatching each inference step to the model implementation. After `TTWorker` builds a `TTModelInput` for the current scheduling step, it delegates to `TTModelRunner`, which determines whether a prefill or decode step is needed and calls the model's `prefill_forward()` or `decode_forward()` method accordingly. `TTModelRunner` also owns the pre-allocated host-pinned input/output tensor buffers that are reused across TTNN trace replays during decode.

## TTModelLoader

`TTModelLoader` is vLLM's model loader hook for TT hardware. It replaces vLLM's standard `DefaultModelLoader` and is registered automatically when the `tt-vllm-plugin` package is installed. Its responsibilities are:

1. **Validate** that the model class resolved by `TTPlatform`'s architecture rename implements `initialize_vllm_model()`. If the method is absent, `TTModelLoader` raises a clear error before any device memory is allocated.
2. **Receive the mesh device** — `TTWorker` has already called `open_mesh_device()` during its initialisation and passes the resulting `mesh_device` handle to `TTModelLoader`; `TTModelLoader` does not open the device itself.
3. **Instantiate the model** by calling `initialize_vllm_model()` with the parameters it needs:

```python
model = model_class.initialize_vllm_model(
    hf_config=hf_config,
    mesh_device=mesh_device,
    max_batch_size=max_batch_size,
    max_seq_len=max_seq_len,
    tt_data_parallel=tt_data_parallel,
    optimizations=optimizations,
)
```

The returned object is stored as the active model for the worker's lifetime. All subsequent `prefill_forward()` and `decode_forward()` calls are dispatched to this instance.

## Device setup

Device initialisation is centralised in `tt_worker.py`. The key function is `open_mesh_device()`:

```python
import os
import ttnn

MESH_DEVICE_SHAPES = {
    "T3K": (1, 8),
    "N300": (1, 2),
    "N150": (1, 1),
    "TG":   (4, 8),
}

def open_mesh_device() -> ttnn.MeshDevice:
    mesh_name = os.environ["MESH_DEVICE"]  # e.g. "T3K"
    mesh_shape = MESH_DEVICE_SHAPES[mesh_name]
    # Configure fabric topology based on shape
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        # ... dispatch mode, fabric topology, etc.
    )
    return mesh_device
```

The `MESH_DEVICE` environment variable is the sole control point for hardware topology. Setting `MESH_DEVICE=T3K` gives a `(1, 8)` mesh device grid across eight N300 modules (each N300 module contains two Wormhole chips, for 16 Wormhole chips total); `MESH_DEVICE=N150` gives a single-chip `(1, 1)` mesh. The `mesh_device` handle returned by `open_mesh_device()` is passed directly into `initialize_vllm_model()` and is the only device context the model implementation ever receives.

## Tokenization ownership

Tokenization is owned entirely by the vLLM API server (Layer 1). vLLM loads the tokenizer from the path supplied to `--model` at startup, using HuggingFace's `AutoTokenizer`:

```python
# Inside vLLM — not in model code
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

The model implementation receives integer token IDs via `TTModelInput.input_ids` — it never handles raw text strings. Detokenization of generated tokens also happens inside vLLM, not inside the model. This is significant for TT Symbiote model authors: if your model's `forward` method today accepts or returns strings, those calls must be removed before deploying through `tt-inference-server`.

---

**Next:** [request_lifecycle.md](./request_lifecycle.md)
