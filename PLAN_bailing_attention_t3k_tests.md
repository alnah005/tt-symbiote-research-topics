# Plan: Fix Bailing Attention T3K Decode Tests

## Problem Description

The Bailing Attention T3K tests are failing with the following results:

| Test | Status | Details |
|------|--------|---------|
| test_weight_sharing_verification | PASSED | - |
| test_full_forward_prefill_with_paged_cache | FAILED | PCC: 0.967932 (below 0.99) |
| test_full_forward_decode_with_paged_cache | FAILED | RuntimeError: Shard width 128 must match physical width 32 for height sharded |
| test_multi_token_decode_sequence | FAILED | RuntimeError: Shard width 128 must match physical width 32 for height sharded |
| test_integration_with_model_layer | FAILED | PCC: 0.958638 (below 0.99) |

## NEW ERROR: HEIGHT_SHARDED Slice Failure (2026-03-25)

### Problem

The previous fix padded cos/sin to head_dim=128 and converted them to HEIGHT_SHARDED. However, the error now occurs when slicing Q/K tensors:

```
RuntimeError: Shard width 128 must match physical width 32 for height sharded
info: Shard width 128 must match physical width 32 for height sharded
backtrace:
 --- ttnn::operations::data_movement::SliceOperation::invoke<int>(...)
```

### Root Cause

1. Q/K tensors are converted to HEIGHT_SHARDED with shard width = head_dim (128)
2. `apply_partial_rope_decode()` tries to slice Q/K to get the rotary portion (`[..., 0:rotary_dim]` where rotary_dim=64)
3. **Slicing along the shard dimension on a HEIGHT_SHARDED tensor is not allowed**
4. The slice operation fails because shard width (128) doesn't match the sliced width (64)

### Code Flow

In `_forward_decode_paged()` (attention.py):
```python
# Create HEIGHT_SHARDED config with head_dim=128
qk_decode_memcfg = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # (32, 128)
    ...
)

# Call apply_partial_rope_decode with this config
query_states, key_states = apply_partial_rope_decode(
    query_states, key_states, cos, sin, trans_mat_sharded,
    decode_memcfg=qk_decode_memcfg, rotary_dim=rotary_dim
)
```

In `apply_partial_rope_decode()` (rope.py):
```python
# Convert Q/K to HEIGHT_SHARDED
query_states = ttnn.to_memory_config(query_states, decode_memcfg)  # Now HEIGHT_SHARDED

# Try to slice - THIS FAILS
q_rot = ttnn.slice(query_states, [0, 0, 0, 0], [seq_len, batch, num_heads, actual_rotary_dim])
```

### Key Insight: rotary_embedding_llama Kernel Constraints

From the kernel validation code (`rotary_embedding_llama_device_operation.cpp`):

1. **Decode mode requires ALL inputs to be HEIGHT_SHARDED** (lines 86-96)
2. **cos/sin must match input dimensions** (line 130 for prefill)
3. **The kernel does NOT natively support partial rotary**

Since tt_transformers doesn't use partial rotary at all, they don't face this issue.

## Correct Solution: Pad cos/sin with Identity Values (No Slicing)

Instead of slicing Q/K into rotary and pass-through portions, we should:

1. **Pad cos with 1.0 and sin with 0.0** for the pass-through portion
2. **Apply RoPE to the full Q/K** (no slicing)
3. The rotation formula `q * cos + rotate_half(q) * sin` will naturally:
   - **Rotate** the first `rotary_dim` elements (where cos/sin have real values)
   - **Pass through** the remaining elements (where `cos=1.0, sin=0.0` gives `q * 1 + rotate_half(q) * 0 = q`)

### Why This Works

The rotary embedding formula is:
```
q_out[i] = q[i] * cos[i] + rotate_half(q)[i] * sin[i]
```

For the pass-through portion (indices >= rotary_dim):
- `cos[i] = 1.0`
- `sin[i] = 0.0`
- `q_out[i] = q[i] * 1.0 + rotate_half(q)[i] * 0.0 = q[i]` (unchanged)

For the rotary portion (indices < rotary_dim):
- cos/sin have the actual rotation values
- `q_out[i]` is properly rotated

### Implementation Changes

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`

In `_forward_decode_paged()`, change the padding from value=0.0 to:

```python
# BEFORE (WRONG):
cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
sin = ttnn.pad(sin, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

# AFTER (CORRECT):
# Pad cos with 1.0 (identity for cos) and sin with 0.0 (identity for sin)
# This ensures pass-through behavior: q * 1.0 + rotate_half(q) * 0.0 = q
cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=1.0)
sin = ttnn.pad(sin, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
```

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`

Simplify `apply_partial_rope_decode()` to NOT slice Q/K:

```python
def apply_partial_rope_decode(
    query_states: ttnn.Tensor,
    key_states: ttnn.Tensor,
    cos: ttnn.Tensor,  # MUST be padded with 1.0 for pass-through portion
    sin: ttnn.Tensor,  # MUST be padded with 0.0 for pass-through portion
    trans_mat: ttnn.Tensor,
    decode_memcfg: ttnn.MemoryConfig = None,
    rotary_dim: int = None,  # No longer needed if padding is correct
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply partial RoPE for decode mode.

    IMPORTANT: This function assumes cos/sin are ALREADY padded to head_dim with
    identity values (cos=1.0, sin=0.0) for the pass-through portion. This avoids
    slicing HEIGHT_SHARDED tensors which is not supported.
    """
    # Convert Q/K to HEIGHT_SHARDED if memory config provided
    if decode_memcfg is not None:
        query_states = ttnn.to_memory_config(query_states, decode_memcfg)
        key_states = ttnn.to_memory_config(key_states, decode_memcfg)

    # Apply RoPE to full Q/K - no slicing needed
    # The identity padding in cos/sin handles pass-through automatically
    query_rotated = ttnn.experimental.rotary_embedding_llama(
        query_states, cos, sin, trans_mat, is_decode_mode=True
    )
    key_rotated = ttnn.experimental.rotary_embedding_llama(
        key_states, cos, sin, trans_mat, is_decode_mode=True
    )

    # Convert back to DRAM if memory config was provided
    if decode_memcfg is not None:
        query_rotated = ttnn.to_memory_config(query_rotated, ttnn.DRAM_MEMORY_CONFIG)
        key_rotated = ttnn.to_memory_config(key_rotated, ttnn.DRAM_MEMORY_CONFIG)

    return query_rotated, key_rotated
```

## Step-by-Step Implementation Plan

### Step 1: Update cos/sin padding in attention.py

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`

**Location:** `_forward_decode_paged()` around line 2695-2698

**Change:**
```python
# Current (wrong):
if rotary_dim < self.head_dim:
    pad_size = self.head_dim - rotary_dim
    cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
    sin = ttnn.pad(sin, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)

# Fixed:
if rotary_dim < self.head_dim:
    pad_size = self.head_dim - rotary_dim
    # Use identity values for padding: cos=1.0, sin=0.0
    # This ensures: q_pass * 1.0 + rotate_half(q_pass) * 0.0 = q_pass (pass-through)
    cos = ttnn.pad(cos, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=1.0)
    sin = ttnn.pad(sin, ((0, 0), (0, 0), (0, 0), (0, pad_size)), value=0.0)
```

### Step 2: Simplify apply_partial_rope_decode() in rope.py

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`

**Remove the slicing logic** since the identity-padded cos/sin handles partial rotary automatically.

**Before:**
```python
def apply_partial_rope_decode(...):
    if actual_rotary_dim < head_dim:
        # Slice Q/K - THIS FAILS ON HEIGHT_SHARDED
        q_rot = ttnn.slice(query_states, ...)
        q_pass = ttnn.slice(query_states, ...)
        # Apply RoPE only to rotary portion
        q_rot_embedded = ttnn.experimental.rotary_embedding_llama(q_rot, cos, sin, ...)
        # Concatenate back
        query_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)
    else:
        query_rotated = ttnn.experimental.rotary_embedding_llama(query_states, cos, sin, ...)
```

**After:**
```python
def apply_partial_rope_decode(
    query_states: ttnn.Tensor,
    key_states: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    decode_memcfg: ttnn.MemoryConfig = None,
    rotary_dim: int = None,  # Kept for backward compatibility but unused
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply partial RoPE for decode mode with [1, B, H, D] format tensors.

    This function expects cos/sin to be padded to head_dim with identity values:
    - cos padded with 1.0 for pass-through portion
    - sin padded with 0.0 for pass-through portion

    This ensures the rotation formula q*cos + rotate_half(q)*sin = q for pass-through
    elements, avoiding the need to slice HEIGHT_SHARDED tensors.
    """
    # Convert Q/K to HEIGHT_SHARDED if memory config provided
    if decode_memcfg is not None:
        query_states = ttnn.to_memory_config(query_states, decode_memcfg)
        key_states = ttnn.to_memory_config(key_states, decode_memcfg)

    # Apply RoPE to full tensors - identity padding handles pass-through
    query_rotated = ttnn.experimental.rotary_embedding_llama(
        query_states, cos, sin, trans_mat, is_decode_mode=True
    )
    key_rotated = ttnn.experimental.rotary_embedding_llama(
        key_states, cos, sin, trans_mat, is_decode_mode=True
    )

    # Convert back to DRAM if memory config was provided
    if decode_memcfg is not None:
        query_rotated = ttnn.to_memory_config(query_rotated, ttnn.DRAM_MEMORY_CONFIG)
        key_rotated = ttnn.to_memory_config(key_rotated, ttnn.DRAM_MEMORY_CONFIG)

    return query_rotated, key_rotated
```

### Step 3: Update any other places that pad cos/sin

Search for other uses of the padding pattern and ensure they use the correct identity values.

## Files to Modify

1. **`/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`**
   - Change cos padding from `value=0.0` to `value=1.0`
   - Keep sin padding as `value=0.0` (already correct)

2. **`/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`**
   - Remove slicing logic from `apply_partial_rope_decode()`
   - Apply RoPE to full Q/K tensors directly
   - Update docstring to explain identity padding requirement

## Success Criteria

1. **Decode tests pass:** No RuntimeError about shard width mismatch
2. **Correct rotation:** The rotary portion gets rotated, pass-through portion is unchanged
3. **PCC targets met:**
   - Decode tests: PCC >= 0.98 with PyTorch reference
   - The identity padding approach should give exact mathematical equivalence

## Testing Plan

1. Run the existing test suite:
   ```bash
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py -v
   ```

2. Specifically test decode with partial rotary:
   ```bash
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py::test_full_forward_decode_with_paged_cache -v
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py::test_multi_token_decode_sequence -v
   ```

## Mathematical Verification

For a partial rotary factor of 0.5 with head_dim=128 and rotary_dim=64:

**Input Q:**
```
Q = [q0, q1, ..., q63, q64, q65, ..., q127]
     |---rotary---|  |----pass-through----|
```

**Padded cos/sin:**
```
cos = [cos0, cos1, ..., cos63, 1.0, 1.0, ..., 1.0]
sin = [sin0, sin1, ..., sin63, 0.0, 0.0, ..., 0.0]
```

**After rotation:**
```
Q_out[i] = Q[i] * cos[i] + rotate_half(Q)[i] * sin[i]

For i < 64 (rotary):
  Q_out[i] = Q[i] * cos[i] + rotate_half(Q)[i] * sin[i]  # Proper rotation

For i >= 64 (pass-through):
  Q_out[i] = Q[i] * 1.0 + rotate_half(Q)[i] * 0.0 = Q[i]  # Unchanged
```

This is mathematically equivalent to the original partial rotary implementation.

## Alternative Approaches Considered

### Option A: Keep cos/sin in DRAM until after slicing
- **Rejected:** rotary_embedding_llama kernel requires HEIGHT_SHARDED inputs in decode mode

### Option B: Don't use HEIGHT_SHARDED for cos/sin
- **Rejected:** Kernel validation fails with non-HEIGHT_SHARDED inputs in decode mode

### Option C: Slice Q/K in DRAM before converting to HEIGHT_SHARDED
- **Rejected:** Would require different shard configs for rotary vs full head_dim, complicates the logic

### Option D: Use identity padding (CHOSEN)
- **Selected:** Clean solution that avoids slicing entirely
- No changes to kernel requirements
- Mathematically equivalent to original partial rotary

## References

- **tt_transformers RotarySetup:** `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/rope.py`
  - Uses full head_dim for shard width, doesn't support partial rotary

- **rotary_embedding_llama kernel:** `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp`
  - Decode mode validation: lines 78-120
  - HEIGHT_SHARDED requirement: lines 86-96
