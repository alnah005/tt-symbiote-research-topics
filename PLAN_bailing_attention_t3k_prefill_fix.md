# Plan: Fix Bailing Attention T3K Prefill with DynamicCache

**Date:** 2026-03-25
**Author:** Architect Agent
**Status:** Draft

## Problem Description

The `TTNNBailingMoEAttention` class fails during prefill on T3K (1x8 mesh, 8 devices) when using `DynamicCache`. The SDPA operation receives incorrectly shaped K and V tensors.

### Error Message

```
TT_FATAL: K and V hidden dim must match. Got K: 16, V: 16
TTNNSDPAAttention: ttnn SDPA failed, falling back to matmul attention.
Q=Shape([1, 16, 32, 128]) K=Shape([1, 4, 32, 16]) V=Shape([1, 4, 32, 16]) is_causal=True
TT_FATAL: The width of the first tensor must be equal to the height of the second tensor. Mismatch: width=128 height=16
```

### Expected Shapes
- Q: `[1, 16, 32, 128]` (batch=1, num_heads=16, seq=32, head_dim=128) - CORRECT
- K: `[1, 4, 32, 128]` (batch=1, num_kv_heads=4, seq=32, head_dim=128) - Expected
- V: `[1, 4, 32, 128]` (batch=1, num_kv_heads=4, seq=32, head_dim=128) - Expected

### Actual Shapes
- K: `[1, 4, 32, 16]` - head_dim is 16 instead of 128
- V: `[1, 4, 32, 16]` - head_dim is 16 instead of 128

The head_dim (128) appears to be incorrectly sharded across 8 devices (128/8=16).

### Affected Tests
- `test_ling_attention_prefill` with `MESH_DEVICE=T3K`
- Only fails when using `DynamicCache` (not `TTNNPagedAttentionKVCache`)

### Passing Tests
- `test_ling_attention_prefill_with_paged_cache` - PASSES with same configuration

## Root Cause Analysis

### 1. Tensor Mapping Issue

When the test creates the input tensor with `ReplicateTensorToMesh`:
```python
mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
hidden_states_tt = ttnn.from_torch(
    hidden_states.to(torch.bfloat16),
    device=mesh_device,
    mesh_mapper=mesh_mapper,
    ...
)
```

The tensor is correctly replicated. However, subsequent operations in the attention module may not preserve this replication.

### 2. Shape Operations on Multi-Device Tensors

The `_split_qkv` and `_apply_qk_norm` functions use `ttnn.reshape` and `ttnn.permute`:

```python
# In _split_qkv
query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
key_states = ttnn.permute(key_states, (0, 2, 1, 3))

# In _apply_qk_norm
k_reshaped = ttnn.reshape(key_states, (batch_kv * num_kv_heads * seq_length_k, head_dim_k))
key_states = ttnn.reshape(k_normed, (batch_kv, num_kv_heads, seq_length_k, head_dim_k))
```

When these operations are applied to multi-device tensors, the tensor semantics (replicated vs sharded) may not be correctly preserved or the shapes may be computed using device-local dimensions instead of global dimensions.

### 3. DynamicCache Path vs Paged Cache Path

The paged cache path passes because:
- It uses `TTNNPagedAttentionKVCache` which stores KV states on-device
- It doesn't go through the torch-ttnn conversion in the KV cache update

The DynamicCache path fails because:
- It converts K/V tensors to torch via `TorchTTNNTensor.to_torch`
- The conversion doesn't use a mesh_composer, potentially causing shape issues
- The tensors are then moved back to device without proper mesh mapping

### 4. Key Code Paths

**Fused QKV Path** (when `distributed=False`):
1. `hidden_states` -> `query_key_value` linear projection
2. `_split_qkv()` - slice and reshape QKV
3. `_apply_qk_norm()` - reshape for norm, then reshape back
4. `_apply_partial_rope()` - apply RoPE
5. KV cache update (DynamicCache path)
6. SDPA - fails here with wrong K/V shapes

## Proposed Solution

### Option 1: Fix Multi-Device Tensor Handling in `_split_qkv`

Before splitting, ensure the QKV tensor shape is correctly interpreted for multi-device:

```python
def _split_qkv(self, qkv: ttnn.Tensor, batch_size: int, seq_length: int):
    # For multi-device, ensure we use the correct global shape
    num_devices = self.device.get_num_devices() if hasattr(self.device, 'get_num_devices') else 1
    qkv_shape = list(qkv.shape)

    # Check if last dimension is sharded and needs to be gathered
    expected_qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    if qkv_shape[-1] != expected_qkv_size and num_devices > 1:
        # QKV was incorrectly sharded, need to gather
        qkv = ttnn.all_gather(qkv, dim=-1, num_links=1)

    # Continue with split...
```

### Option 2: Fix Multi-Device Tensor Handling in `_apply_qk_norm`

Ensure reshape operations use global shapes:

```python
def _apply_qk_norm(self, query_states: ttnn.Tensor, key_states: ttnn.Tensor):
    if not self.use_qk_norm:
        return query_states, key_states

    # Use stored dimensions instead of reading from tensor shape
    # which may be device-local on multi-device mesh
    batch_size, num_heads, seq_length, head_dim = query_states.shape

    # Verify head_dim is correct (not sharded)
    if head_dim != self.head_dim:
        # Shape is device-local, need to handle differently
        ...
```

### Option 3: Ensure Proper Mesh Mapping for All Operations (Recommended)

The most robust solution is to ensure all tensors maintain their replication status:

1. **Add explicit replication after linear projections**:
   ```python
   qkv = self.query_key_value(hidden_states)
   if self.device.get_num_devices() > 1:
       # Ensure QKV is properly replicated
       qkv = self._ensure_replicated(qkv)
   ```

2. **Add mesh handling utilities**:
   ```python
   def _ensure_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
       """Ensure tensor is replicated across all devices in the mesh."""
       if self.device.get_num_devices() <= 1:
           return tensor

       # Convert to torch and back with explicit replication
       torch_t = ttnn.to_torch(
           tensor,
           mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0)
       )
       # Take first batch (since replicated, all batches are identical)
       torch_t = torch_t[:tensor.shape[0]]

       return ttnn.from_torch(
           torch_t,
           device=self.device,
           layout=tensor.layout,
           dtype=tensor.dtype,
           mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
       )
   ```

3. **Fix DynamicCache path** to use mesh_composer:
   ```python
   else:
       # DynamicCache path
       cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

       # Convert with proper mesh handling
       mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0) if self.device.get_num_devices() > 1 else None
       k_torch = ttnn.to_torch(key_states, mesh_composer=mesh_composer)
       v_torch = ttnn.to_torch(value_states, mesh_composer=mesh_composer)

       # Slice to original batch size
       k_torch = k_torch[:batch_size]
       v_torch = v_torch[:batch_size]

       key_states, value_states = past_key_values.update(
           k_torch, v_torch,
           layer_idx,
           cache_kwargs,
       )

       # Convert back with replication
       mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
       key_states = ttnn.from_torch(key_states, device=self.device, mesh_mapper=mesh_mapper, ...)
       value_states = ttnn.from_torch(value_states, device=self.device, mesh_mapper=mesh_mapper, ...)
   ```

## Step-by-Step Implementation Plan

### Phase 1: Diagnose Exact Failure Point

1. Add debug prints after each operation in `_forward_prefill`:
   - After QKV linear projection
   - After `_split_qkv`
   - After `_apply_qk_norm`
   - Before SDPA call

2. Log tensor shapes and mesh topology at each step

3. Identify exactly where the shape becomes incorrect

### Phase 2: Implement Fix

1. Based on diagnosis, implement the appropriate fix from Options 1-3 above

2. If the issue is in QKV split/reshape:
   - Update `_split_qkv` to handle multi-device tensors
   - Add explicit shape validation

3. If the issue is in QK norm:
   - Update `_apply_qk_norm` to use stored dimensions
   - Add explicit replication check

4. If the issue is general mesh handling:
   - Add `_ensure_replicated` utility method
   - Apply it after key operations

### Phase 3: Update DynamicCache Path

1. Fix the tensor conversion in the DynamicCache update path
2. Use proper mesh_composer for `ttnn.to_torch`
3. Use proper mesh_mapper for `ttnn.from_torch`

### Phase 4: Validation

1. Run `test_ling_attention_prefill` with `MESH_DEVICE=T3K`
2. Verify all sequence lengths pass (32, 128, 256)
3. Verify PCC >= 0.99 compared to PyTorch reference
4. Run full test suite to ensure no regressions

## Success Criteria

1. **test_ling_attention_prefill** passes on T3K with all sequence lengths
2. **test_ling_attention_prefill_with_paged_cache** continues to pass
3. PCC >= 0.99 between TTNN and PyTorch reference
4. No regressions in other Ling attention tests

## Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `TTNNBailingMoEAttention._forward_prefill()`
   - `TTNNBailingMoEAttention._split_qkv()`
   - `TTNNBailingMoEAttention._apply_qk_norm()`

## Dependencies

- TTNN mesh device operations
- Understanding of `ReplicateTensorToMesh` vs `ShardTensorToMesh`
- `ttnn.reshape` and `ttnn.permute` behavior on multi-device tensors

## Risk Assessment

- **Medium Risk**: Changes to attention core path could affect model accuracy
- **Mitigation**: Comprehensive PCC testing against PyTorch reference
- **Mitigation**: Run all existing attention tests to catch regressions

## Alternative Approaches Considered

1. **Use `distributed=True` mode**: This uses separate Q/K/V projections which may work better on multi-device, but tests explicitly use `distributed=False`

2. **Skip DynamicCache on T3K**: Not acceptable as it limits functionality

3. **Single-device fallback**: Could work but defeats the purpose of T3K support

## Notes

- The paged cache path works, suggesting the issue is specific to the DynamicCache path
- The fix should maintain backward compatibility with single-device operation
- Consider adding explicit tests for multi-device tensor operations
