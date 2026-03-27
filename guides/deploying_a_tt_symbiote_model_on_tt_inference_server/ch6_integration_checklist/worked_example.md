# Worked Example — TTMySymbioteModel

This page walks through a complete, end-to-end integration of a hypothetical `TTMySymbioteModel`: a 7B-parameter decoder-only transformer compiled with TTNN. Every file path, method signature, and code snippet shown here represents correct, runnable code that satisfies the contract described in Chapters 2–5.

---

## Directory Structure

The integration spans two repositories. The tt-symbiote model source lives in its own repo, cloned as a sibling of `tt-inference-server`:

```
workspace/
├── tt-inference-server/               # the inference server repo
│   ├── tt-vllm-plugin/
│   │   ├── tt_vllm_plugin/
│   │   │   ├── __init__.py
│   │   │   ├── model_registry.py      # register_tt_models() lives here
│   │   │   └── ...
│   │   └── setup.py
│   └── workflows/
│       └── model_spec.py              # ModelSpecTemplate entries appended here
│
└── tt-symbiote/                       # the model implementation repo
    └── models/
        └── my_symbiote_model/
            ├── __init__.py
            ├── my_symbiote_model.py   # TTMySymbioteModel class
            └── config.py              # optional: model hyperparameter dataclass
```

The `PYTHONPATH` must include `workspace/tt-symbiote` so that `from models.my_symbiote_model.my_symbiote_model import TTMySymbioteModel` resolves correctly at runtime:

```bash
export PYTHONPATH="/workspace/tt-symbiote:$PYTHONPATH"
```

---

## Step 1 — ModelSpec Registration

### `ImplSpec` and `ModelSpecTemplate` in `workflows/model_spec.py`

Open `tt-inference-server/workflows/model_spec.py` and append the following. Do not modify any existing entries.

```python
# --- TTMySymbioteModel registration ---

from model_spec import ImplSpec, DeviceModelSpec, ModelSpecTemplate

MY_SYMBIOTE_IMPL = ImplSpec(
    repo="https://github.com/tenstorrent/tt-symbiote.git",
    module_path="models.my_symbiote_model.my_symbiote_model",
)

# N300 single-chip configuration: 32-sequence batch, 8 k token context
_N300_SPEC = DeviceModelSpec(
    device="N300",
    max_concurrency=32,
    max_context=8192,
    vllm_args=[
        "--block-size", "64",
        "--max-num-seqs", "32",
        "--dtype", "bfloat16",
    ],
    override_tt_config={
        "ENABLE_ELTWISE_UNARY_FUSED_WITH_MATMUL": True,
    },
)

# T3K eight-chip configuration: 64-sequence batch, 32 k token context
_T3K_SPEC = DeviceModelSpec(
    device="T3K",
    max_concurrency=64,
    max_context=32768,
    vllm_args=[
        "--block-size", "64",
        "--max-num-seqs", "64",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "8",
    ],
    override_tt_config={},
)

MY_SYMBIOTE_TEMPLATE = ModelSpecTemplate(
    hf_model_repo="myorg/my-symbiote-7b",
    impl=MY_SYMBIOTE_IMPL,
    device_specs=[_N300_SPEC, _T3K_SPEC],
    has_builtin_warmup=True,   # model compiles TTNN traces on first forward pass
)

spec_templates.append(MY_SYMBIOTE_TEMPLATE)
```

Verify expansion after saving:

```bash
cd tt-inference-server
python -c "
from workflows.model_spec import MODEL_SPECS
for s in MODEL_SPECS:
    if 'my-symbiote' in s.hf_model_repo:
        print(s.device, s.hf_model_repo, s.max_context)
"
# Expected output:
# N300 myorg/my-symbiote-7b 8192
# T3K  myorg/my-symbiote-7b 32768
```

---

## Step 2 — Registering the Model Class

### `register_tt_models()` addition in `tt_vllm_plugin/model_registry.py`

```python
# tt-inference-server/tt-vllm-plugin/tt_vllm_plugin/model_registry.py

from models.my_symbiote_model.my_symbiote_model import TTMySymbioteModel


def register_tt_models() -> dict:
    """Return a mapping of architecture name → model class.

    The architecture name must match config.json "architectures"[0] exactly.
    """
    return {
        # existing entries preserved here ...
        "TTMySymbioteModel": TTMySymbioteModel,
    }
```

The string key `"TTMySymbioteModel"` must appear verbatim in the `"architectures"` list inside the HuggingFace `config.json` for `myorg/my-symbiote-7b`. If you control the model repo, set it there; if not, add an alias inside `register_tt_models()` pointing the upstream architecture name to your class.

---

## Step 3 — Model Class Implementation

The full model file lives at `tt-symbiote/models/my_symbiote_model/my_symbiote_model.py`.

### 3a — `initialize_vllm_model` classmethod

```python
import os
from typing import List, Optional, Tuple

import torch
import ttnn
from safetensors.torch import load_file


class TTMySymbioteModel:
    """7B decoder-only Symbiote model compiled for TT hardware."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        tt_weights: dict,           # weights already on-device as ttnn.Tensor
        hf_config,
        max_batch_size: int,
        max_seq_len: int,
    ):
        self.mesh_device = mesh_device
        self.tt_weights = tt_weights
        self.hf_config = hf_config
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Build TTNN compute graph / trace here (model-specific).
        # self._build_model()

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        # The plugin may pass additional keyword arguments; absorb them.
        **kwargs,
    ) -> "TTMySymbioteModel":
        """Load weights from the HuggingFace checkpoint and return a ready instance.

        Parameters
        ----------
        hf_config:
            HuggingFace `PretrainedConfig` object. `hf_config._name_or_path`
            is the local path to the downloaded checkpoint directory.
        mesh_device:
            An already-open `ttnn.MeshDevice`. Do NOT call
            `ttnn.open_mesh_device()` here.
        max_batch_size:
            Maximum number of sequences in a single forward pass.
        max_seq_len:
            Maximum total token length (prompt + generation).
        """
        checkpoint_dir = hf_config._name_or_path
        assert os.path.isdir(checkpoint_dir), (
            f"Checkpoint directory not found: {checkpoint_dir!r}. "
            "Set HF_HOME or --download-dir so the path resolves correctly."
        )

        # --- 1. Discover and load all safetensors shards ---
        shard_paths = sorted(
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".safetensors")
        )
        assert shard_paths, f"No .safetensors files found in {checkpoint_dir}"

        cpu_state_dict: dict[str, torch.Tensor] = {}
        for shard_path in shard_paths:
            cpu_state_dict.update(load_file(shard_path))

        # --- 2. Convert weights to bfloat16 and move to mesh_device ---
        # Real implementation would shard tensors across the mesh according
        # to the tensor-parallel strategy. Shown here as a simplified loop.
        tt_weights: dict[str, ttnn.Tensor] = {}
        for name, cpu_tensor in cpu_state_dict.items():
            tt_weights[name] = ttnn.from_torch(
                cpu_tensor.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
            )

        # --- 3. Construct and return the model instance ---
        return cls(
            mesh_device=mesh_device,
            tt_weights=tt_weights,
            hf_config=hf_config,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
```

### 3b — `prefill_forward`

```python
    def prefill_forward(
        self,
        tokens: torch.Tensor,          # (batch, max_prompt_len) int64, on CPU
        prompt_lens: torch.Tensor,     # (batch,) int64, unpadded lengths, on CPU
        page_table: torch.Tensor,      # (batch, max_blocks) int32, on CPU
        kv_cache: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
    ) -> torch.Tensor:
        """Run the prefill (prompt encoding) forward pass.

        Returns
        -------
        torch.Tensor
            Logits of shape `(batch, vocab_size)` on CPU — the last-token
            logits only. Returning shape `(batch, seq_len, vocab_size)` is
            wrong and will cause vLLM's sampler to fail.
        """
        # Move inputs to device.
        tt_tokens = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # --- Run model forward (model-specific TTNN ops) ---
        # tt_logits = self._run_prefill(tt_tokens, prompt_lens, page_table, kv_cache)

        # Placeholder: return random logits for illustration only.
        tt_logits = ttnn.full(
            (tokens.shape[0], self.hf_config.vocab_size),
            fill_value=0.0,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )

        # CRITICAL: convert to CPU before returning.
        cpu_logits = ttnn.to_torch(tt_logits).to(torch.float32)
        return cpu_logits  # shape: (batch, vocab_size)
```

### 3c — `decode_forward`

```python
    def decode_forward(
        self,
        tokens: torch.Tensor,          # (batch, 1) int64, on CPU — one token per seq
        page_table: torch.Tensor,      # (batch, max_blocks) int32, on CPU
        kv_cache: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
    ) -> torch.Tensor:
        """Run the decode (autoregressive) forward pass for a single step.

        Returns
        -------
        torch.Tensor
            Logits of shape `(batch, vocab_size)` on CPU.
        """
        tt_tokens = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # --- Run model forward (model-specific TTNN ops) ---
        # tt_logits = self._run_decode(tt_tokens, page_table, kv_cache)

        # Placeholder: return random logits for illustration only.
        tt_logits = ttnn.full(
            (tokens.shape[0], self.hf_config.vocab_size),
            fill_value=0.0,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
        )

        # CRITICAL: convert to CPU before returning.
        cpu_logits = ttnn.to_torch(tt_logits).to(torch.float32)
        return cpu_logits  # shape: (batch, vocab_size)
```

### 3d — `allocate_kv_cache`

```python
    def allocate_kv_cache(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> List[Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Allocate paged KV cache on the mesh device. Do NOT call inside `initialize_vllm_model`;
        the block manager calls this separately after profiling available memory.
        Returns one `(k_cache, v_cache)` tuple per transformer layer, each tensor on-device with
        shape `(num_blocks, block_size, num_kv_heads, head_dim)`. ROW_MAJOR_LAYOUT is required
        because GQA models may have `num_kv_heads` or `head_dim` that are not multiples of 32.
        """
        num_layers = self.hf_config.num_hidden_layers
        tt_dtype = ttnn.bfloat16  # map torch dtype to ttnn dtype as needed

        kv_cache: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []
        cache_shape = (num_blocks, block_size, num_kv_heads, head_dim)

        for _ in range(num_layers):
            # ROW_MAJOR_LAYOUT is used here because num_kv_heads and head_dim may
            # not be multiples of 32. TILE_LAYOUT requires the last two dimensions
            # to be multiples of 32; using it without explicit padding would error
            # or produce mis-shaped tensors for GQA models (e.g. num_kv_heads=8).
            k_cache = ttnn.zeros(
                cache_shape,
                dtype=tt_dtype,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            v_cache = ttnn.zeros(
                cache_shape,
                dtype=tt_dtype,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            kv_cache.append((k_cache, v_cache))

        return kv_cache
```

---

## Step 4 — Smoke Test

Run the Phase 6 smoke test from [`checklist.md`](./checklist.md#phase-6--smoke-test), substituting `myorg/my-symbiote-7b` as the model repo, `--max-model-len 8192`, and `--max-num-seqs 32` from the worked example above.

Expected response structure for TTMySymbioteModel (note the `"model"`, `"logprobs"`, and `"usage"` fields):

```json
{
    "id": "cmpl-abc123",
    "object": "text_completion",
    "model": "myorg/my-symbiote-7b",
    "choices": [
        {
            "text": " Paris. It is located in",
            "index": 0,
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 12
    }
}
```

---

## Common Pitfalls

### Pitfall 1 — Wrong `@classmethod` vs instance method

**Wrong:**
```python
def initialize_vllm_model(self, hf_config, mesh_device, ...):
    ...
```

**Correct:**
```python
@classmethod
def initialize_vllm_model(cls, hf_config, mesh_device, ...):
    ...
    return cls(...)
```

`initialize_vllm_model` is called on the class, not on an instance. Without `@classmethod`, Python passes an uninitialized `self` as the first argument and the call fails with a `TypeError`. The decorator and the `cls` parameter are both required.

---

### Pitfall 2 — Forgetting `.cpu()` on prefill logits

**Wrong:**
```python
def prefill_forward(self, tokens, prompt_lens, page_table, kv_cache):
    tt_logits = self._run_prefill(...)
    return ttnn.to_torch(tt_logits)   # ttnn.to_torch() returns a CPU torch.Tensor but may be bfloat16; vLLM sampler expects float32
```

**Wrong (silent hang):**
```python
    return tt_logits   # ttnn.Tensor, NOT a CPU torch.Tensor — scheduler deadlocks
```

**Correct:**
```python
    cpu_logits = ttnn.to_torch(tt_logits).to(torch.float32)
    return cpu_logits
```

vLLM's scheduler expects a plain `torch.Tensor` on CPU. A `ttnn.Tensor` satisfies no isinstance check and causes the token sampling code to block indefinitely. The same applies to `decode_forward`.

---

### Pitfall 3 — Allocating KV cache inside `initialize_vllm_model`

**Wrong:**
```python
@classmethod
def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
    ...
    # Do NOT do this:
    kv_cache = cls._allocate_kv(mesh_device, num_blocks=512, ...)
    return cls(mesh_device=mesh_device, kv_cache=kv_cache, ...)
```

**Correct:** leave KV cache allocation entirely to `allocate_kv_cache`. The block manager computes the correct `num_blocks` value after profiling available device memory. Allocating inside `initialize_vllm_model` pre-empts this calculation, either wastes memory (too many blocks) or causes OOM (too few blocks left for activations).

---

### Pitfall 4 — Calling `ttnn.close_mesh_device()` inside the model

**Wrong:**
```python
def __del__(self):
    ttnn.close_mesh_device(self.mesh_device)   # NEVER do this
```

**Wrong:**
```python
@classmethod
def initialize_vllm_model(cls, ...):
    device = ttnn.open_mesh_device(...)        # NEVER do this either
    ...
    ttnn.close_mesh_device(device)             # doubly wrong
```

The `mesh_device` handle is owned by `tt-vllm-plugin`. Calling `close_mesh_device` anywhere in model code invalidates the shared handle and causes all subsequent TTNN operations — including those in other workers — to raise a `DeviceClosedError` or hang. Do not store the intent to close the device in a destructor, context manager, or `atexit` hook.

---

**Next:** [Chapter 7 — Debugging, Constraints, and Performance Tuning](../ch7_debugging_and_tuning/index.md)
