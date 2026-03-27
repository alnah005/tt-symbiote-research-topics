# Weight Discovery and Loading

## How Weights Reach the Model

### Default Weight Source: HuggingFace Hub

tt-inference-server downloads model weights from HuggingFace Hub during host setup. `setup_host.py` invokes:

```bash
huggingface-cli download <hf_repo> --local-dir <cache_dir>
```

The `HF_TOKEN` environment variable must be set if the repository is gated (e.g., Llama, Mistral, or any private model). The files land in a local directory managed by the HuggingFace cache — typically under `~/.cache/huggingface/hub/` unless overridden by `HF_HOME` or `HUGGINGFACE_HUB_CACHE`.

### How the Model Receives the Weight Path

The server does not pass a raw filesystem path to `initialize_vllm_model()`. Instead, it passes an `hf_config` argument, which is a `transformers.PretrainedConfig` object loaded from the downloaded model directory. The checkpoint root can be recovered from this object:

```python
def initialize_vllm_model(self, hf_config, vllm_config, mesh_device):
    # Recover the local checkpoint directory from the config object.
    checkpoint_dir = hf_config._name_or_path
    # checkpoint_dir is an absolute path like:
    # /home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/<hash>/
    ...
```

`hf_config._name_or_path` is set by `transformers` to the directory (or HuggingFace repo ID) from which the config was loaded. When the server loads a locally-cached checkpoint it will always be an absolute filesystem path.

> **Warning — internal attribute:** `_name_or_path` is a private attribute of `transformers.PretrainedConfig` with no public API stability guarantee; it could be renamed or removed in a future `transformers` release. Access it defensively and fall back to the `HUGGINGFACE_HUB_DOWNLOAD_DIR` environment variable (or the equivalent path derived from `HF_HOME` / `HUGGINGFACE_HUB_CACHE`) if the attribute is absent:
>
> ```python
> checkpoint_dir = getattr(hf_config, "_name_or_path", None) or os.environ.get("HUGGINGFACE_HUB_DOWNLOAD_DIR")
> ```
>
> In practice, `_name_or_path` is used throughout existing `tt-transformers` models (e.g., `generator_vllm.py`) and is de facto stable for the current integration. Treat it as internal nonetheless.

## Supported Checkpoint Formats

tt-inference-server always downloads **standard HuggingFace-format checkpoints**. The files in the checkpoint directory are either:

- **`.safetensors` shards** — the modern, memory-safe format; preferred when available.
- **`.bin` shards** — legacy PyTorch pickle format; still common for older or community models.

The model is entirely responsible for loading these files. Common approaches:

### Loading safetensors

```python
from safetensors.torch import load_file

checkpoint_dir = hf_config._name_or_path
# Load a single shard.
state_dict = load_file(f"{checkpoint_dir}/model.safetensors")

# Or load multiple shards and merge.
import os, glob
shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
state_dict = {}
for path in shard_paths:
    state_dict.update(load_file(path))
```

### Loading .bin shards

```python
import torch, os, glob

checkpoint_dir = hf_config._name_or_path
shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "pytorch_model-*.bin")))
state_dict = {}
for path in shard_paths:
    state_dict.update(torch.load(path, map_location="cpu"))
```

### Converting to TTNN tensors on load

For weights that will live permanently on device, convert them with `ttnn.as_tensor` immediately after loading from disk. Provide a `cache_file_name` to avoid repeating the conversion on subsequent startups:

```python
import ttnn

weight_tensor = ttnn.as_tensor(
    state_dict["model.layers.0.self_attn.q_proj.weight"],
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    cache_file_name=f"{cache_dir}/layer_0_q_proj.bin",
)
```

## Custom Weight Formats

Some TT Symbiote models require pre-converted weights — for example, weights already tiled into `bfloat8_b` format or pre-sharded into a DRAM layout for a multi-chip mesh. Two strategies are supported:

1. **Convert inside `initialize_vllm_model`**: Read the standard HF checkpoint, apply the conversion, and store the result to disk for future runs. This is the most portable approach and requires no changes to the deployment environment.

2. **Pre-compute and store alongside the HF checkpoint**: Run a one-time conversion script offline and place the converted files in the checkpoint directory (or a sibling directory). `initialize_vllm_model` then loads the pre-converted files directly, skipping the conversion step entirely.

In both cases, **conversion logic lives inside the model**, not in the server.

## ModelSpec Flags Related to Weight Loading

### `uses_tensor_model_cache`

When `uses_tensor_model_cache = True` in `ModelSpec`, tt-inference-server expects a side-channel directory for pre-converted TTNN weight tensors. `setup_host.py` creates this directory and communicates its path to the model via an environment variable set by that script. The exact variable name is defined per-model in the `ModelSpec` — check `workflows/model_spec.py` for your specific model to confirm the name before hard-coding it.

Use this flag when your model calls `ttnn.as_tensor(..., cache_file_name=...)` and needs a stable, writable location that is separate from the read-only HF checkpoint snapshot directory:

```python
import os

# The cache-directory variable name is set by setup_host.py and is defined
# per-model in ModelSpec (see workflows/model_spec.py for the exact name).
# Use a safe lookup that covers the two most common names; add others as needed.
cache_dir = (
    os.environ.get("TENSOR_MODEL_CACHE_PATH")  # common default name
    or os.environ.get("MODEL_CACHE_DIR")        # alternate name used by some ModelSpecs
)
if not cache_dir:
    raise RuntimeError(
        "No tensor model cache directory found in environment. "
        "Check the ModelSpec definition in workflows/model_spec.py for the "
        "exact environment variable name set by setup_host.py."
    )

weight_tensor = ttnn.as_tensor(
    raw_weight,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    cache_file_name=os.path.join(cache_dir, "layer_0_ffn_w1.bin"),
)
```

On the first run, `ttnn.as_tensor` converts and writes the cache file. On subsequent runs it reads the cached file directly, skipping host-side conversion entirely.

### `has_builtin_warmup`

When `has_builtin_warmup = True` in `ModelSpec`, tt-inference-server does not inject HTTP warmup requests before opening the public API endpoint. Set this flag when `initialize_vllm_model` already runs warmup trace captures internally — for example, when the model calls `ttnn.capture_trace` for each supported sequence-length bucket during initialization.

If `has_builtin_warmup = False` (the default), the server sends a short synthetic prompt through the full inference stack before marking the server as ready, to ensure any JIT compilation or trace capture triggered by the first real request has already occurred.

## What the Server Does NOT Do

`TTModelLoader.download_model()` and `TTModelLoader.load_weights()` both raise `NotImplementedError` — this is intentional. There is no server-managed weight loading pipeline. If you see these methods in the codebase, treat them as explicit markers that weight loading is the model's responsibility and must be implemented inside `initialize_vllm_model`.

---

**Next:** [`tokenization.md`](./tokenization.md)
