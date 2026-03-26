# Plan: Profile TTNNBailingMoEAttention in Isolation (Standalone)

## Objective

Create a standalone pytest that instantiates ONLY the `TTNNBailingMoEAttention` module (one layer) from the Ling-mini-2.0 model, runs prefill and decode through it in isolation, and captures timing via DispatchManager and Tracy device profiler. The full model generation loop is NOT used.

---

## File to Create

**Path:** `models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py`

(Inside the tt-metal repo, alongside existing tests.)

---

## Ling-mini-2.0 Attention Config

| Parameter | Value |
|-----------|-------|
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 |
| `hidden_size` | 2048 |
| `head_dim` | 128 |
| `partial_rotary_factor` | 0.5 |
| `use_qk_norm` | True |
| `rope_theta` | 600000 |
| `max_position_embeddings` | 32768 |
| `num_hidden_layers` | 20 |

---

## TTNNBailingMoEAttention Internal Structure

The module (created by `from_torch()`) contains these sub-modules:
- `q_proj`: `TTNNLinearIColShardedWRowSharded` -- Q projection (sharded input, row-sharded weights)
- `k_proj`: `TTNNLinearIReplicatedWColSharded` -- K projection (replicated input, col-sharded weights)
- `v_proj`: `TTNNLinearIReplicatedWColSharded` -- V projection (same as K)
- `dense`: `TTNNLinearIReplicatedWColSharded` -- output projection
- `query_layernorm`: `TTNNRMSNorm` (when `use_qk_norm=True`)
- `key_layernorm`: `TTNNRMSNorm` (when `use_qk_norm=True`)
- `rope`: `TTNNRotaryPositionEmbedding` (because `partial_rotary_factor=0.5 < 1.0`)
- `sdpa`: `TTNNSDPAAttention`
- `_rotary_setup`: `BailingRotarySetup` (created during `move_weights_to_device_impl`)

### Critical Runtime Dependencies on `_fallback_torch_layer`

The module accesses these at runtime:
- `self._fallback_torch_layer.layer_idx` -- used in both prefill and decode for KV cache indexing
- `self._fallback_torch_layer.config` -- used in `move_weights_to_device_impl` for RoPE parameters

Therefore, the original PyTorch attention module MUST stay alive. The `from_torch()` classmethod preserves it as `_fallback_torch_layer`.

---

## Implementation Details

### Imports

```python
import os
import time

import pytest
import torch
from transformers import AutoModelForCausalLM

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, DistributedConfig
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNBailingMoEAttention,
    TTNNPagedAttentionKVCache,
)
```

### Pytest Fixtures

Use the same device/mesh fixtures as `test_ling_mini_2_0.py`:

```python
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
```

These are resolved by the global `conftest.py` at `/home/ttuser/salnahari/tt-metal/conftest.py` (lines 302, 528).

### Helper: Setup Attention Module

```python
def setup_standalone_attention(mesh_device, layer_idx=0, batch_size=1):
    """Extract a single attention layer from the HF model and prepare it on device.

    Steps:
    1. Load the full HF model (needed for weights)
    2. Extract one attention layer
    3. Create TTNNBailingMoEAttention via from_torch()
    4. Delete the full model to free CPU memory
    5. Set device, preprocess weights, move to device

    Returns:
        (ttnn_attn, model_config)
    """
    model = AutoModelForCausalLM.from_pretrained(
        "inclusionAI/Ling-mini-2.0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model_config = model.config

    # Extract the attention sub-module from the specified layer
    torch_attn = model.model.layers[layer_idx].attention

    # Create TTNN module (this splits fused QKV, permutes weights, creates sub-modules)
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn)

    # Delete the full model to free ~4GB CPU RAM.
    # IMPORTANT: torch_attn is still alive because ttnn_attn._fallback_torch_layer holds a reference.
    del model

    # Set device on the standalone module.
    # Cannot use set_device(model, mesh_device) since we don't have a nn.Module tree.
    # Instead, call TTNNModule methods directly.
    ttnn_attn.to_device(mesh_device)
    if mesh_device.get_num_devices() > 1:
        ttnn_attn.set_device_state(DistributedConfig(mesh_device))

    # Recursively set device on all child TTNNModules
    from models.experimental.tt_symbiote.core.module import TTNNModule
    for attr_name in dir(ttnn_attn):
        try:
            attr = getattr(ttnn_attn, attr_name)
        except Exception:
            continue
        if isinstance(attr, TTNNModule) and attr is not ttnn_attn:
            attr.to_device(mesh_device)
            if mesh_device.get_num_devices() > 1:
                attr.set_device_state(DistributedConfig(mesh_device))

    # Preprocess and move weights to device
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    return ttnn_attn, model_config
```

**IMPORTANT NOTE on child module device propagation:** The `TTNNModule.move_weights_to_device_impl()` base class iterates `self.__dict__` and calls `move_weights_to_device()` on child TTNNModules. But `move_weights_to_device()` asserts `self.device is not None`. So we must ensure all child modules have their device set BEFORE calling `ttnn_attn.preprocess_weights()` / `move_weights_to_device()`.

A simpler alternative: iterate `ttnn_attn.__dict__` directly:
```python
for name, child in ttnn_attn.__dict__.items():
    if isinstance(child, TTNNModule):
        child.to_device(mesh_device)
        if mesh_device.get_num_devices() > 1:
            child.set_device_state(DistributedConfig(mesh_device))
```

### Helper: Create Paged KV Cache

```python
def create_standalone_kv_cache(model_config, mesh_device, batch_size=1):
    """Create paged KV cache for standalone attention testing."""
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,  # 20 -- must accommodate layer_idx
        num_kv_heads=model_config.num_key_value_heads,  # 4
        head_dim=model_config.head_dim,  # 128
        config=config,
        device=None,
    ).to_device(mesh_device)
```

### Helper: Create Input Tensors

```python
def create_hidden_states(hidden_size, seq_length, batch_size, device):
    """Create hidden_states tensor distributed correctly for the mesh device."""
    hidden = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16)

    num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    if num_devices > 1:
        mesh_mapper = ttnn.ShardTensor2dMesh(device, device.shape, (0, -1))
    else:
        mesh_mapper = None

    return ttnn.from_torch(
        hidden,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
```

**Why ShardTensor2dMesh(device, device.shape, (0, -1))?**
This matches the default distribution strategy in `DistributedConfig.__post_init__()` (run_config.py line 74). On T3K (1x8), this shards dim=-1 (hidden_size=2048) across 8 devices, giving 256 per device. The `q_proj` (`TTNNLinearIColShardedWRowSharded`) expects input sharded along columns, which is exactly what dim=-1 sharding provides.

### The Test Function

```python
def test_profile_bailing_attention_standalone(mesh_device):
    """Profile TTNNBailingMoEAttention in isolation -- prefill + decode."""
    batch_size = 1
    prefill_seq_len = 128
    num_decode_tokens = 32

    # --- Setup ---
    ttnn_attn, model_config = setup_standalone_attention(
        mesh_device, layer_idx=0, batch_size=batch_size
    )
    torch.set_grad_enabled(False)

    hidden_size = model_config.hidden_size

    # === WARMUP ===
    print("Running warmup...")
    paged_cache = create_standalone_kv_cache(model_config, mesh_device, batch_size)

    # Warmup prefill
    h_warmup = create_hidden_states(hidden_size, prefill_seq_len, batch_size, mesh_device)
    cache_pos_warmup = torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)
    _ = ttnn_attn(
        h_warmup,
        (None, None),  # position_embeddings -- unused by TTNNBailingMoEAttention
        attention_mask=None,
        past_key_values=paged_cache,
        cache_position=cache_pos_warmup,
    )
    ttnn.synchronize_device(mesh_device)

    # Warmup decode (a few tokens)
    for i in range(3):
        h_dec = create_hidden_states(hidden_size, 1, batch_size, mesh_device)
        cp_dec = torch.tensor([prefill_seq_len + i], dtype=torch.long)
        _ = ttnn_attn(
            h_dec,
            (None, None),
            attention_mask=None,
            past_key_values=paged_cache,
            cache_position=cp_dec,
        )
    ttnn.synchronize_device(mesh_device)

    # === PROFILED RUN ===
    # Reset KV cache
    paged_cache = create_standalone_kv_cache(model_config, mesh_device, batch_size)
    DispatchManager.clear_timings()

    # --- Profiled Prefill ---
    print(f"\nProfiled prefill (seq_len={prefill_seq_len})...")
    h_pf = create_hidden_states(hidden_size, prefill_seq_len, batch_size, mesh_device)
    cache_pos_pf = torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)

    start = time.time()
    out_pf = ttnn_attn(
        h_pf,
        (None, None),
        attention_mask=None,
        past_key_values=paged_cache,
        cache_position=cache_pos_pf,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.time() - start) * 1000
    print(f"  Prefill latency: {prefill_ms:.2f} ms")

    # --- Profiled Decode ---
    print(f"\nProfiled decode ({num_decode_tokens} tokens)...")
    decode_times = []
    for i in range(num_decode_tokens):
        cur_pos = prefill_seq_len + i
        h_dec = create_hidden_states(hidden_size, 1, batch_size, mesh_device)
        # Pass cache_position as torch.LongTensor to avoid extra ttnn->torch->ttnn round-trip
        cp_dec = torch.tensor([cur_pos], dtype=torch.long)

        start = time.time()
        out_dec = ttnn_attn(
            h_dec,
            (None, None),
            attention_mask=None,
            past_key_values=paged_cache,
            cache_position=cp_dec,
        )
        ttnn.synchronize_device(mesh_device)
        decode_times.append(time.time() - start)

    avg_decode_ms = sum(decode_times) / len(decode_times) * 1000
    min_decode_ms = min(decode_times) * 1000
    max_decode_ms = max(decode_times) * 1000
    print(f"  Avg decode latency: {avg_decode_ms:.2f} ms")
    print(f"  Min: {min_decode_ms:.2f} ms, Max: {max_decode_ms:.2f} ms")

    # --- Save stats ---
    DispatchManager.save_stats_to_file("bailing_attention_standalone_timing_stats.csv")

    # --- Assertions ---
    assert out_pf is not None, "Prefill output should not be None"
    assert len(decode_times) == num_decode_tokens, "All decode iterations should complete"
    print("\nTest passed.")
```

### Key Details for the Implementer

1. **`position_embeddings=(None, None)`**: The `TTNNBailingMoEAttention.forward()` accepts `position_embeddings` in its signature but never uses it. The module uses its internal `_rotary_setup` (a `BailingRotarySetup` instance created during `move_weights_to_device_impl()`) to compute cos/sin tables. Pass `(None, None)` as a dummy.

2. **`cache_position` as `torch.LongTensor`**: Pass it as a plain torch tensor, NOT as a ttnn.Tensor. The decode path (line 2670-2678 of attention.py) has code that converts ttnn.Tensor cache_position to torch and back -- this is a known round-trip. By passing torch directly, we skip the ttnn->torch conversion (the torch->ttnn conversion for `cur_pos_tt` is unavoidable).

3. **`attention_mask=None`**: For prefill, the SDPA module handles causal masking internally (`is_causal=True`). For decode, paged SDPA handles masking via the page table and cur_pos. No explicit mask is needed.

4. **Input tensor shape:** For prefill: `[batch_size, seq_length, hidden_size]`. For decode: `[batch_size, 1, hidden_size]`. The module checks `hidden_states.shape[1]` to route to prefill vs decode (line 2825-2827).

5. **`ttnn.synchronize_device(mesh_device)`**: Must be called after each forward call to ensure device ops are complete before reading the host-side timer. Without this, the host timer would only measure dispatch time, not actual compute.

6. **Deleting the full model**: After `from_torch()`, the TTNN module holds a reference to the original torch attention via `_fallback_torch_layer`. When we `del model`, the rest of the model is freed, but the single attention layer stays alive. This saves ~4GB CPU RAM.

---

## How to Run

### Environment Setup

```bash
cd /home/ttuser/salnahari
source tt_bashrc
cd tt-metal
```

### Run 1: DispatchManager Timing Only (Fastest)

```bash
pytest models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py::test_profile_bailing_attention_standalone -sv
```

Output files:
- `bailing_attention_standalone_timing_stats.csv` -- raw per-op timing entries
- `bailing_attention_standalone_timing_stats_pivot.csv` -- aggregated by func_name/module_name

### Run 2: Tracy Device Profiler (Cycle-Accurate)

```bash
TT_METAL_DEVICE_PROFILER=1 pytest models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py::test_profile_bailing_attention_standalone -sv
```

This generates `ops_perf_results.csv` with device kernel durations, FPU utilization, and NoC bandwidth utilization for every op.

### Run 3: Full Tracy Capture (GUI-Inspectable)

```bash
pytest_full models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py::test_profile_bailing_attention_standalone -sv
```

(`pytest_full` is aliased to `python3 -m tracy -v -r -p -m pytest`)

### Run 4: With Signposts for Module Boundary Markers

```bash
TT_SYMBIOTE_SIGNPOST_MODE=all TT_METAL_DEVICE_PROFILER=1 pytest models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py::test_profile_bailing_attention_standalone -sv
```

### Run 5: Serialized Profiling (Most Accurate Per-Op Timing)

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SYNC=1 pytest models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py::test_profile_bailing_attention_standalone -sv
```

---

## Expected Op Breakdown Per Forward Call

### Prefill Path (`_forward_prefill`)

| # | Operation | Notes |
|---|-----------|-------|
| 1 | `q_proj(hidden_states)` | matmul + reduce_scatter |
| 2 | `all_gather(hidden_states)` | For K/V replicated input |
| 3 | `k_proj(hidden_states_replicated)` | col-sharded matmul |
| 4 | `v_proj(hidden_states_replicated)` | col-sharded matmul |
| 5 | `deallocate(hidden_states_replicated)` | |
| 6 | `all_gather(Q)`, `all_gather(K)`, `all_gather(V)` | via `_maybe_all_gather` |
| 7 | 3x `reshape` + 3x `permute` | [B,S,H*D] -> [B,H,S,D] |
| 8 | `_apply_qk_norm` | 2x RMSNorm |
| 9 | `rotary_embedding_llama(Q)` | RoPE |
| 10 | `rotary_embedding_llama(K)` | RoPE |
| 11 | `paged_fill_cache(K, V)` | KV cache fill |
| 12 | `scaled_dot_product_attention` | SDPA with causal mask |
| 13 | `reshape` + `dense()` | output projection |

### Decode Path (`_forward_decode_paged`)

| # | Operation | Known Bottleneck? |
|---|-----------|-------------------|
| 1 | `q_proj(hidden_states)` | |
| 2 | `all_gather(hidden_states)` | Inter-device BW |
| 3 | `k_proj`, `v_proj` | |
| 4 | 3x `all_gather(Q,K,V)` | Inter-device BW |
| 5 | `concat([Q,K,V])` + reshape | |
| 6 | **`_to_replicated` (HOST ROUND-TRIP)** | YES: device->host->device |
| 7 | `nlp_create_qkv_heads_decode` | |
| 8 | 2x `to_memory_config` (L1) | Data movement |
| 9 | `_apply_qk_norm` | 2x RMSNorm |
| 10 | 2x `rotary_embedding_llama` (HEIGHT_SHARDED) | |
| 11 | 2x `to_memory_config` (kv_mem) | Data movement |
| 12 | `paged_update_cache(K,V)` | |
| 13 | `paged_sdpa_decode` | Attention compute |
| 14 | `to_memory_config` + `nlp_concat_heads_decode` | |
| 15 | `dense()` | |
| 16 | Optional `slice` + `reshape` | |

**Primary optimization target:** Step 6 (`_to_replicated`) does a full device->host->device round-trip to convert mesh topology metadata from all-gathered to replicated. This is needed because `paged_sdpa_decode` requires replicated topology.

---

## What the Implementer Must Do

1. Create `models/experimental/tt_symbiote/tests/test_profile_bailing_attention.py` in the tt-metal repo
2. Implement the test using the structure above
3. Do NOT modify any existing code in tt-metal
4. Do NOT use `register_module_replacement_dict` or `model.generate()`
5. Use real weights from the HF model (not random) for realistic profiling
6. Ensure `_fallback_torch_layer` reference stays alive (do not delete `torch_attn`)
7. The test must print latency numbers and save CSV timing stats
8. The test must work on both single-device (N150) and multi-device (T3K) configurations

---

## Key Files Reference

| File | Line | What |
|------|------|------|
| `modules/attention.py` | 2251-2843 | `TTNNBailingMoEAttention` class |
| `modules/attention.py` | 62-75 | `PagedAttentionConfig` dataclass |
| `modules/attention.py` | 76-268 | `TTNNPagedAttentionKVCache` class |
| `modules/rope.py` | 313-432 | `BailingRotarySetup` class |
| `core/module.py` | 60-159 | `TTNNModule` base class |
| `core/run_config.py` | 178-314 | `DispatchManager` class |
| `core/run_config.py` | 64-98 | `DistributedConfig` class |
| `tests/test_attention.py` | 44-109 | Existing pattern for standalone attention testing |
| `tests/test_ling_mini_2_0.py` | 32-54 | KV cache creation pattern |

All paths are relative to `models/experimental/tt_symbiote/`.
