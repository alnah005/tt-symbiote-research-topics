# PLAN: Bailing Attention T3K Reshape Volume Fix

## Problem Statement

After the nlp_concat_heads_decode sharding fix, a new error appeared:

```
RuntimeError: Invalid arguments to reshape (new_volume == old_volume assertion failed)
```

This error occurs in the decode path when trying to pad and reshape cos/sin tensors for RoPE.

## Previous Fix (nlp_concat_heads_decode Sharding)

The previous fix added HEIGHT_SHARDED conversion before `nlp_concat_heads_decode`:
- Added `sdpa_output_memcfg` with `shape=(32, self.head_dim)`
- Added `ttnn.to_memory_config(attn_output, sdpa_output_memcfg)` before concat_heads

## Current Error: Reshape Volume Mismatch

## Root Cause Analysis

### The Error Location

The reshape error occurs at `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50`:
```cpp
TT_FATAL(new_volume == old_volume, "Invalid arguments to reshape");
```

### Analysis of `_forward_decode_paged()` Code Flow

In `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` at lines 2661-2739:

1. **Position embeddings input** (line 2662):
   ```python
   cos, sin = position_embeddings
   ```
   - From HuggingFace `rotary_emb`, these are **torch tensors** with shape `[B, S, rotary_dim]` = `[1, 1, 64]`
   - `rotary_dim = head_dim * partial_rotary_factor = 128 * 0.5 = 64`

2. **Problem: torch.layout vs ttnn.Layout** (lines 2665-2668):
   ```python
   if cos.layout != ttnn.TILE_LAYOUT:
       cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
   ```
   - For torch tensors, `cos.layout` returns `torch.strided` (PyTorch's layout attribute)
   - This is NOT equal to `ttnn.TILE_LAYOUT` (different types)
   - So the condition is True, but `ttnn.to_layout()` requires a ttnn.Tensor, not torch.Tensor
   - **This should fail with TypeError!**

3. **Subsequent operations also require ttnn tensors** (lines 2683-2700):
   - `ttnn.unsqueeze()` - requires ttnn.Tensor
   - `ttnn.pad()` - requires ttnn.Tensor
   - `ttnn.to_memory_config()` - requires ttnn.Tensor

### Why Reshape Error Instead of TypeError?

The reshape error suggests that either:
1. Some code path converts torch tensors to ttnn before these operations
2. The test is using a different position_embeddings source (e.g., BailingRotarySetup which returns ttnn tensors)
3. There's implicit conversion happening in a newer version

### The Actual Issue: Volume Mismatch After Padding

Assuming cos/sin are converted to ttnn somehow, the reshape error occurs because:

1. **Original cos/sin shape**: `[1, 1, 1, rotary_dim]` = `[1, 1, 1, 64]` (after unsqueeze)
2. **After padding** (lines 2699-2700):
   ```python
   cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=1.0)
   ```
   - Result: `[1, 1, 1, 128]` (padded to head_dim)

3. **Sharded memory config** (lines 2713-2718):
   ```python
   cos_sin_memcfg = ttnn.create_sharded_memory_config(
       shape=(ttnn.TILE_SIZE, self.head_dim),  # (32, 128) = 4096 elements per shard
       core_grid=ttnn.CoreGrid(y=1, x=batch_size),  # 1 core for batch=1
       ...
   )
   ```

4. **Volume mismatch**:
   - cos/sin tensor: `1*1*1*128 = 128` elements
   - Shard config expects: `32*128 = 4096` elements per shard
   - **Cannot reshape 128 elements into 4096 elements!**

### Root Cause: Incorrect cos_sin_memcfg Shard Shape

The shard shape `(TILE_SIZE, head_dim) = (32, 128)` was copied from the Q/K memory config pattern but is **wrong for cos/sin**:
- Q/K shape: `[1, batch, num_heads, head_dim]` = `[1, 1, 16, 128]` = 2048 elements
- cos/sin shape: `[1, batch, 1, head_dim]` = `[1, 1, 1, 128]` = 128 elements

The cos/sin tensors have `1` in the head dimension (they're broadcast across all heads), not `num_heads`.

## Solution

Two issues need to be fixed:

### Issue 1: Position Embeddings Type Conversion

Position embeddings from HuggingFace are **torch tensors**, but the decode path expects **ttnn tensors**. The code needs to convert torch -> ttnn at the start.

### Issue 2: Incorrect cos_sin_memcfg Shard Shape

The current shard shape `(32, 128)` expects 4096 elements, but cos/sin only have 128 elements.

**Fix**: The cos/sin don't need HEIGHT_SHARDED memory config at all. The `rotary_embedding_llama` kernel can operate on DRAM tensors, or we should use a shard shape that matches the actual tensor dimensions.

### Exact Fix for TTNNBailingMoEAttention._forward_decode_paged()

**Part 1: Convert torch tensors to ttnn at the start** (add after line 2662):

```python
cos, sin = position_embeddings

# Convert torch tensors to ttnn if needed (HuggingFace returns torch tensors)
if isinstance(cos, torch.Tensor):
    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
    cos = ttnn.from_torch(
        cos.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    sin = ttnn.from_torch(
        sin.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
```

**Part 2: Fix the cos_sin_memcfg shard shape** (lines 2711-2718):

The current code uses:
```python
cos_sin_memcfg = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # WRONG: (32, 128) = 4096 elements
    ...
)
```

**Option A - Remove HEIGHT_SHARDED for cos/sin entirely:**

The `rotary_embedding_llama` kernel in decode mode may not actually require HEIGHT_SHARDED cos/sin. Check if DRAM_MEMORY_CONFIG works:

```python
# Skip sharding for cos/sin - keep in DRAM
# (rotary_embedding_llama may accept DRAM inputs)
# cos = ttnn.to_memory_config(cos, cos_sin_memcfg)  # Remove this
# sin = ttnn.to_memory_config(sin, cos_sin_memcfg)  # Remove this
```

**Option B - Use correct shard shape for cos/sin:**

If HEIGHT_SHARDED is required, the shard shape must match the tensor dimensions:

```python
# cos/sin shape after padding: [1, batch, 1, head_dim]
# For HEIGHT_SHARDED, shard_shape should be (1, head_dim) NOT (32, head_dim)
cos_sin_memcfg = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # Tile-aligned: (32, 128)
    core_grid=ttnn.CoreGrid(y=1, x=batch_size),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
# Pad cos/sin to tile-aligned shape (32 in height dim)
cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 31), (0, 0)), value=1.0)  # [1,1,1,128] -> [1,1,32,128]
sin = ttnn.pad(sin, ((0, 0), (0, 0), (0, 31), (0, 0)), value=0.0)
cos = ttnn.to_memory_config(cos, cos_sin_memcfg)
sin = ttnn.to_memory_config(sin, cos_sin_memcfg)
```

### Recommended Fix: Use BailingRotarySetup

The cleanest solution is to use `BailingRotarySetup` (already in rope.py) which pre-computes cos/sin with correct topology at initialization. This avoids the torch->ttnn conversion overhead in the forward pass.

**In TTNNBailingMoEAttention.__init__ or from_torch:**

```python
self.rotary_setup = BailingRotarySetup(
    device=self.device,
    head_dim=self.head_dim,
    max_seq_len=max_seq_len,
    rope_theta=config.rope_theta,
    partial_rotary_factor=self.partial_rotary_factor,
)
```

**In _forward_decode_paged:**

```python
# Get cos/sin from pre-computed setup (already ttnn with replicated topology)
cos, sin = self.rotary_setup.get_cos_sin_for_decode(cache_position)
# cos/sin are already [1, batch, 1, rotary_dim] in TILE_LAYOUT
```

### Bailing Model Specifics

For Bailing (from the attention module):
- `num_heads` = 16 (attention heads)
- `num_kv_heads` = 4 (KV heads)
- `head_dim` = 128
- `partial_rotary_factor` = 0.5
- `rotary_dim` = 64

## Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `_forward_decode_paged()` in `TTNNBailingMoEAttention` class
   - Line ~2662: Add torch-to-ttnn conversion for position embeddings
   - Lines ~2711-2732: Fix cos_sin_memcfg or remove HEIGHT_SHARDED for cos/sin

## Testing

After the fix, run:
```bash
pytest models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py::test_full_forward_decode_with_paged_cache -v
```

## Status

- [x] Implement HEIGHT_SHARDED conversion before nlp_concat_heads_decode (previous fix)
- [ ] Fix torch-to-ttnn conversion for position embeddings
- [ ] Fix cos_sin_memcfg volume mismatch (or remove sharding for cos/sin)
- [ ] Test end-to-end decode flow

## Related Files

- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` - Bailing attention implementation
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py` - RoPE implementation with BailingRotarySetup
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/tensor_utils.py` - Tensor conversion utilities
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py` - Test file
- `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp` - Reshape error source

## Error Source Location

```
/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape_common.cpp:50
TT_FATAL(new_volume == old_volume, "Invalid arguments to reshape");
```

This error is triggered when `ttnn.to_memory_config()` tries to reshape the tensor to fit the sharded memory config, but the volumes don't match.
