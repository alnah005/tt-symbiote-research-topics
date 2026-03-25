# Plan: Fix Layer 10 Decode Path Accuracy in TTNNBailingMoEAttention

**Date:** 2026-03-25 (Revised)
**Target:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
**Class:** `TTNNBailingMoEAttention._forward_decode_paged()`
**Model:** Ling-mini-2.0 (BailingMoeV2)

---

## REVISION HISTORY

### 2026-03-25 v2: Alternative Fix After Option A Failed

**What was tried:** Option A - Remove padding from shard_h (change from 32 to 4)

**Why it failed:** TTNN requires tile-aligned sharding. Error:
```
Physical shard shape (4, 128) must be tile {32, 32} sized!
```

**Conclusion:** We MUST keep the padding to 32 for tile alignment. The fix must address the tensor shape/layout mismatch instead.

---

## 1. Problem Summary

### Symptoms
- Garbled text generation when only TTNN attention is enabled
- PCC degradation from 0.99 to 0.43 over 29 decode iterations
- **Layer 10 specifically degrades first** (min PCC 0.5286) while Layer 0 remains stable (avg PCC 0.9687)
- Prefill path works correctly (all layers PCC > 0.999)
- Problem appears at decode step 1 (not cumulative from prefill)

### Observed Per-Layer Accuracy
| Layer | Avg PCC | Min PCC | Behavior |
|-------|---------|---------|----------|
| Layer 0 | 0.9687 | 0.9069 | Stable |
| Layer 10 | 0.7288 | 0.5286 | **Severe degradation from step 1** |
| Layer 19 | 0.9038 | 0.8318 | Secondary degradation |

### Model Configuration
- **Q heads (nh):** 16
- **KV heads (nkv):** 4
- **GQA group_size:** 4 (16 / 4)
- **Head dimension (dh):** 128
- **Total layers:** 20
- **partial_rotary_factor:** < 1.0 (uses partial RoPE)
- **use_qk_norm:** True

---

## 2. Root Cause Analysis (REVISED)

### 2.1 CONFIRMED: Tensor Shape Mismatch

**Evidence from tt_transformers working implementation (`models/tt_transformers/tt/attention.py`):**

The working implementation uses `ttnn.experimental.paged_update_cache` with these tensor shapes:
- **K/V input shape:** `[1, num_kv_heads, batch_size, head_dim]` (comment: "k_heads, [seqlen, n_kv_heads, bsz, head_dim]")
- **Cache shape:** `[max_batch_size, n_kv_heads, max_seq_len, head_dim]`

**Evidence from paged_update_cache unit test (`tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py`):**

```python
input_shape = [1, num_users, num_heads, head_dim]  # [1, batch, heads, head_dim]
cache_shape = [num_users, num_heads, max_seq_len, head_dim]  # [batch, heads, seq, head_dim]

# Padding is done to 32 heads, then reshape back to logical shape
x_pad = torch.nn.functional.pad(x, (0, 0, 0, 32 - num_heads), "constant", 0)
xt = ttnn.Tensor(x_pad, input_dtype).to(ttnn.TILE_LAYOUT)
xt = ttnn.reshape(xt, ttnn.Shape(input_shape))  # Reshape to original logical shape
```

**Current Ling implementation (WRONG):**
```python
# After RoPE, K/V shape is [batch, heads, seq, head_dim] = [1, 4, 1, 128]
# Permute to [S, B, H, D] = [1, 1, 4, 128]
key_states = ttnn.permute(key_states, (2, 0, 1, 3))  # seq, batch, heads, head_dim
```

This produces shape `[seq, batch, heads, head_dim]` but `paged_update_cache` expects `[1, batch, heads, head_dim]`.

### 2.2 The Shape Convention Mismatch

| Dimension | Expected by paged_update_cache | Current Ling Shape |
|-----------|-------------------------------|-------------------|
| Dim 0 | 1 (constant) | seq_length (1 for decode) |
| Dim 1 | batch_size | batch_size |
| Dim 2 | num_kv_heads | num_kv_heads |
| Dim 3 | head_dim | head_dim |

While the shapes happen to match for decode (`seq=1`), the semantic interpretation may differ, and the padding/reshaping pattern used in the working implementation is NOT being followed.

### 2.3 The Working Pattern

From `test_paged_update_cache.py`:
1. Create tensor with logical shape `[1, batch, heads, head_dim]`
2. Pad heads dimension to tile boundary (32) with zeros
3. Convert to TILE_LAYOUT
4. **Reshape back to original logical shape** (this is critical!)
5. Apply height sharding based on batch count, not head count
6. Call `paged_update_cache`

The Ling code does NOT do step 4 (reshape back to logical shape after padding).

---

## 3. NEW Implementation Plan

### Option B: Match tt_transformers Pattern (RECOMMENDED)

Remove the custom sharding entirely and follow the tt_transformers pattern that directly calls `paged_update_cache` without explicit height sharding.

**Current problematic code (lines 2851-2883):**
```python
# Permute B H S D -> S B H D for paged kernels
query_states = ttnn.permute(query_states, (2, 0, 1, 3))
key_states = ttnn.permute(key_states, (2, 0, 1, 3))
value_states = ttnn.permute(value_states, (2, 0, 1, 3))

# ... multi-device handling ...

# Update paged KV cache
tile_size = 32
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size

core_grid = ttnn.CoreGrid(y=1, x=batch_size)
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
value_states = ttnn.to_memory_config(value_states, shard_cfg)

past_key_values.paged_update_on_device(
    key_states,
    value_states,
    layer_idx=layer_idx,
    current_pos=cur_pos_tt,
)
```

**New code following tt_transformers:**
```python
# After RoPE, shape is [batch, heads, seq, head_dim] = [1, 4, 1, 128]
# Permute to [seq, heads, batch, head_dim] = [1, 4, 1, 128] for paged_update_cache
# Expected input: [1, num_kv_heads, batch, head_dim]
key_states = ttnn.permute(key_states, (2, 1, 0, 3))  # [seq, heads, batch, head_dim]
value_states = ttnn.permute(value_states, (2, 1, 0, 3))

# Multi-device: convert all-gathered topology -> replicated for paged kernels
if self.device.get_num_devices() > 1:
    query_states = self._to_replicated(query_states)
    key_states = self._to_replicated(key_states)
    value_states = self._to_replicated(value_states)

# NO custom sharding - pass K/V directly to paged_update_on_device
# The kernel handles the tensor layout internally
past_key_values.paged_update_on_device(
    key_states,
    value_states,
    layer_idx=layer_idx,
    current_pos=cur_pos_tt,
)
```

### Option C: Use Proper Padding + Reshape Pattern

If sharding is required for performance, follow the exact pattern from the unit test:

```python
# After RoPE, shape is [batch, heads, seq, head_dim] = [1, 4, 1, 128]
# Permute to paged_update_cache expected: [1, batch, heads, head_dim]
key_states = ttnn.permute(key_states, (2, 0, 1, 3))  # [1, 1, 4, 128]
value_states = ttnn.permute(value_states, (2, 0, 1, 3))

# Convert to torch for padding (if needed)
key_torch = ttnn.to_torch(key_states)
value_torch = ttnn.to_torch(value_states)

# Pad heads to 32 with zeros
key_padded = torch.nn.functional.pad(key_torch, (0, 0, 0, 32 - self.num_kv_heads), "constant", 0)
value_padded = torch.nn.functional.pad(value_torch, (0, 0, 0, 32 - self.num_kv_heads), "constant", 0)

# Convert back to TTNN with TILE_LAYOUT
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
key_states = ttnn.from_torch(key_padded, device=self.device, layout=ttnn.TILE_LAYOUT,
                              dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
value_states = ttnn.from_torch(value_padded, device=self.device, layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

# CRITICAL: Reshape back to original logical shape
key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))

# Apply height sharding based on BATCH, not HEADS
compute_grid_size = self.device.compute_with_storage_grid_size()
num_cores = batch_size
shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
input_shard_spec = ttnn.ShardSpec(
    shard_grid,
    [key_states.volume() // key_states.padded_shape[-1] // num_cores, key_states.padded_shape[-1]],
    ttnn.ShardOrientation.ROW_MAJOR,
)
input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

key_states = ttnn.to_memory_config(key_states, input_mem_config)
value_states = ttnn.to_memory_config(value_states, input_mem_config)

past_key_values.paged_update_on_device(
    key_states,
    value_states,
    layer_idx=layer_idx,
    current_pos=cur_pos_tt,
)
```

### Option D: Use nlp_create_qkv_heads_decode

The cleanest solution is to use the same `ttnn.experimental.nlp_create_qkv_heads_decode` function used by tt_transformers, which produces correctly shaped tensors:

```python
# Instead of separate Q/K/V projections and manual reshape/permute,
# use the fused QKV approach with nlp_create_qkv_heads_decode

# After QKV matmul, instead of manual split:
(
    q_heads_1BQD,
    k_heads_1BKD,
    v_heads_1BKD,
) = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# These tensors are already in the correct shape [1, batch, heads, head_dim]
# and can be passed directly to paged_update_cache
```

This requires restructuring to use fused QKV, which may not be feasible for the current modular architecture.

---

## 4. Recommended Approach: Option B

**Rationale:**
1. Simplest change - just remove the custom sharding code
2. Aligns with working tt_transformers pattern
3. Does not require host<->device transfers (unlike Option C)
4. Does not require architectural changes (unlike Option D)

### Step-by-Step Implementation

#### Step 1: Fix the permute order

Current:
```python
# Permute B H S D -> S B H D
query_states = ttnn.permute(query_states, (2, 0, 1, 3))
key_states = ttnn.permute(key_states, (2, 0, 1, 3))
value_states = ttnn.permute(value_states, (2, 0, 1, 3))
```

New (match tt_transformers K/V shape `[seqlen, n_kv_heads, bsz, head_dim]`):
```python
# Permute B H S D -> S H B D for K/V, S B H D for Q
# K/V: [batch, heads, seq, head_dim] -> [seq, heads, batch, head_dim]
query_states = ttnn.permute(query_states, (2, 0, 1, 3))  # Q stays [S, B, H, D]
key_states = ttnn.permute(key_states, (2, 1, 0, 3))  # K to [S, H, B, D]
value_states = ttnn.permute(value_states, (2, 1, 0, 3))  # V to [S, H, B, D]
```

#### Step 2: Remove the custom height sharding

Delete these lines (2863-2874):
```python
tile_size = 32
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size

core_grid = ttnn.CoreGrid(y=1, x=batch_size)
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
value_states = ttnn.to_memory_config(value_states, shard_cfg)
```

#### Step 3: Pass K/V directly to cache update

Keep the existing call:
```python
past_key_values.paged_update_on_device(
    key_states,
    value_states,
    layer_idx=layer_idx,
    current_pos=cur_pos_tt,
)
```

#### Step 4: Verify Q shape for SDPA

The `paged_sdpa_decode` expects Q in shape `[1, batch, num_heads, head_dim]`. Verify:
```python
# After permute, Q shape should be [1, 1, 16, 128]
# This matches [1, batch, num_heads, head_dim] for batch_size=1
assert query_states.shape == (1, batch_size, self.num_heads, self.head_dim)
```

---

## 5. Validation

### Test 1: Per-Layer Cache PCC

```python
def test_decode_cache_pcc_per_layer(device):
    """Verify K/V cache contents match torch reference after decode step."""
    # ... setup ...
    for layer_idx in [0, 10, 19]:
        k_cache_tt = read_cache_to_torch(paged_cache, layer_idx, "k")
        v_cache_tt = read_cache_to_torch(paged_cache, layer_idx, "v")
        k_ref, v_ref = get_torch_kv(model, layer_idx)

        pcc_k = compute_pcc(k_ref, k_cache_tt)
        pcc_v = compute_pcc(v_ref, v_cache_tt)

        assert pcc_k > 0.99, f"Layer {layer_idx} K cache PCC {pcc_k:.4f} < 0.99"
        assert pcc_v > 0.99, f"Layer {layer_idx} V cache PCC {pcc_v:.4f} < 0.99"
```

### Test 2: Multi-Step Decode Stability

```python
def test_multi_step_decode_pcc_stability(device):
    """Verify PCC does not degrade over decode steps."""
    for step in range(30):
        output = run_decode_step(model, step)
        pcc = compute_pcc(output, reference[step])
        assert pcc > 0.98, f"Step {step}: PCC {pcc:.4f} < 0.98"
```

---

## 6. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Layer 10 min PCC | 0.5286 | > 0.98 |
| Layer 10 avg PCC | 0.7288 | > 0.99 |
| Layer 19 min PCC | 0.8318 | > 0.98 |
| Multi-step decay | 0.99 -> 0.43 | Stable > 0.98 |
| Generated text | Garbled | Coherent |

---

## 7. Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `TTNNBailingMoEAttention._forward_decode_paged()` lines 2851-2883

---

## 8. References

- `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py` - Working tt_transformers implementation (lines 595-610 for shape comments, 604-609 for paged_update_cache call)
- `/home/ttuser/salnahari/tt-metal/tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py` - Unit test showing correct padding/reshape pattern
- `guides/paged_sdpa_decode_for_gqa/` - Full API documentation
