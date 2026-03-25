# Plan: Fix Bailing (TTNNBailingMoEAttention) Attention Tests on T3K

## Problem Description

The `TTNNBailingMoEAttention` attention module fails on T3K (8-device mesh) with a matmul shape mismatch error during the prefill phase:

```
RuntimeError: TT_FATAL @ matmul_device_operation.cpp:126: a_shape[-1] == b_shape[-2]
The width of the first tensor must be equal to the height of the second tensor. Mismatch: width=128 height=16
```

### Error Analysis

The error occurs in `_matmul_attention` (the SDPA fallback) when computing `Q @ K^T`:
- **Expected**: Q shape `[B, num_heads, S, head_dim]` = `[1, 16, S, 128]`, K shape `[B, num_kv_heads, S, head_dim]` = `[1, 4, S, 128]`
- **Observed**: K tensor has shape `[1, 4, 16, 24]` per device (should be `[1, 4, S, 128]`)

The dimension 24 in the K tensor's last axis (should be 128 = head_dim) indicates the **all-gather operation is not correctly reconstituting the full tensor** from the 8 device shards.

### Root Cause Hypothesis

The issue is in the distributed prefill path (`_use_separate_qkv = True` on T3K):

1. **K/V projections use `TTNNLinearIReplicatedWColSharded`** which shards the output along the last dimension
2. **Each device has**: K output shape `[B, S, kv_size/8]` = `[1, 11, 64]` (where kv_size = 4 * 128 = 512)
3. **`_maybe_all_gather(key_states)`** should gather these to `[1, 11, 512]`
4. **But**: The all-gather appears to be incomplete or on the wrong dimension, resulting in `[1, 4, 16, 24]` after reshape/permute

Likely causes:
- The `_maybe_all_gather` is using `dim=-1` but the tensor may already be reshaped
- The all_gather may be failing silently or returning a partial result
- CCL (Collective Communication Library) synchronization issues on T3K

## Model Configuration Reference

Ling-mini-2.0 (BailingMoeV2):
- `num_attention_heads`: 16
- `num_key_value_heads`: 4
- `hidden_size`: 2048
- `head_dim`: 128
- `partial_rotary_factor`: 0.5 (rotary_dim = 64)

## Affected Code

**File**: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
**Class**: `TTNNBailingMoEAttention`
**Method**: `_forward_prefill` (lines 2660-2848)

## Implementation Plan

### Phase 1: Add Debug Instrumentation

Add shape printing to track where dimensions go wrong:

```python
# In _forward_prefill, after line 2698 (_maybe_all_gather calls):
print(f"DEBUG: After all_gather - Q shape: {query_states.shape}, K shape: {key_states.shape}, V shape: {value_states.shape}")
print(f"DEBUG: Expected - Q: [1, S, {self.num_heads * self.head_dim}], K: [1, S, {self.num_kv_heads * self.head_dim}]")
```

### Phase 2: Fix All-Gather Order

The issue may be that `_maybe_all_gather` is called BEFORE reshape. The sequence should be:
1. Projection outputs: `[B, S, size_per_device]`
2. All-gather: `[B, S, full_size]`
3. Reshape: `[B, S, num_heads, head_dim]`
4. Permute: `[B, num_heads, S, head_dim]`

**Verify the current order is correct**. If projections output tensors that are already 4D, the all-gather dimension may be wrong.

### Phase 3: Validate All-Gather Dimension

In `_maybe_all_gather`, add validation:

```python
def _maybe_all_gather(self, tensor):
    t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    # Debug print before gather
    print(f"DEBUG _maybe_all_gather: input shape {t.shape}, gathering on dim=-1")

    # ... existing code ...

    # Debug print after gather
    print(f"DEBUG _maybe_all_gather: output shape {gathered.shape}")

    # Add shape validation
    if len(gathered.shape) >= 2 and gathered.shape[-1] != expected_full_size:
        raise ValueError(f"All-gather incomplete: expected last dim {expected_full_size}, got {gathered.shape[-1]}")

    return gathered
```

### Phase 4: Fix - Use Synchronous All-Gather with Explicit Synchronization

The async all_gather may have synchronization issues. Force synchronous execution:

```python
def _maybe_all_gather(self, tensor):
    t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor

    if not hasattr(self, 'device') or self.device is None:
        return t

    num_devices = self.device.get_num_devices() if hasattr(self.device, 'get_num_devices') else 1

    if num_devices <= 1:
        return t

    # Always use synchronous all_gather for reliability
    gathered = ttnn.all_gather(t, dim=-1, num_links=1)
    ttnn.synchronize_device(self.device)  # Ensure completion

    if gathered.dtype != ttnn.bfloat16:
        gathered = ttnn.typecast(gathered, ttnn.bfloat16)

    return gathered
```

### Phase 5: Alternative Fix - Replicate K/V Instead of Sharding

If all-gather remains problematic, modify the K/V projections to produce replicated outputs:

```python
# In from_torch, for K/V projections:
# Instead of TTNNLinearIReplicatedWColSharded (col-sharded output)
# Use a linear that produces replicated output

# Option A: Use weights that are sharded but produce replicated output via all-reduce
# Option B: Replicate K/V weights entirely (memory trade-off)
```

### Phase 6: Validate Against Reference Implementation

Compare with how TT-Transformers handles similar GQA scenarios:

```python
# Reference: tt_transformers attention implementation
# Check how they handle:
# 1. K/V projection when num_kv_heads < num_devices
# 2. All-gather before attention
# 3. Tensor layout for SDPA
```

## Test Plan

### Unit Test 1: Projection Shape Validation
```python
def test_kv_projection_shapes_t3k(mesh_device):
    """Verify K/V projection outputs have correct shapes after all-gather."""
    # Create attention module
    # Run projection
    # Assert shapes match expected [B, S, num_kv_heads * head_dim]
```

### Unit Test 2: RoPE Shape Preservation
```python
def test_rope_shape_preservation_t3k(mesh_device):
    """Verify RoPE doesn't corrupt tensor shapes on T3K."""
    # Create tensors with expected shapes
    # Apply partial RoPE
    # Assert output shapes match input shapes
```

### Integration Test: Full Attention
```python
def test_bailing_attention_prefill_t3k(mesh_device):
    """End-to-end test of attention prefill on T3K."""
    # Run prefill with known input
    # Compare output shape and values to PyTorch reference
```

## Success Criteria

1. `test_ling_mini_2_0_with_attention_acceleration` passes on T3K
2. All attention tests pass on single device and T3K
3. Generated text is coherent (passes coherence checks)
4. No regression on N150/N300 configurations

## Estimated Effort

- Phase 1 (Debug): 1 hour
- Phase 2-4 (Fix): 2-3 hours
- Phase 5 (Alternative): 2 hours if needed
- Phase 6 (Validation): 1 hour
- Testing: 2 hours

Total: ~8-10 hours

## Key Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `TTNNBailingMoEAttention._maybe_all_gather()`
   - `TTNNBailingMoEAttention._forward_prefill()`

2. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0_attention.py`
   - Add new diagnostic tests

## References

- Existing Ling attention plans: `PLAN_ling_shape_mismatch_fix.md`, `PLAN_ling_linear_placement_fix.md`
- TT-Transformers attention: `/home/ttuser/salnahari/tt-metal/models/demos/*/tt/llama_attention.py`
- TTNN all_gather docs: `ttnn.all_gather` API documentation
