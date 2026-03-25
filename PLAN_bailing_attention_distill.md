# Plan: Distill Bailing Attention to Simplest T3K Functional Form

## Overview

This plan describes how to simplify `TTNNBailingMoEAttention` to its minimal functional form for T3K execution. The goal is to remove unnecessary complexity while maintaining correctness on T3K (1x8 mesh device).

**Related Plans:**
- `PLAN_bailing_dynamic_cache_removal.md` - Removes DynamicCache support (prerequisite)
- `PLAN_bailing_single_device_removal.md` - Removes single device flow support (prerequisite)

**Note:** This plan assumes the above two plans have been executed. It focuses on further simplifications beyond those scope.

---

## Current Implementation Analysis

### File Location
`/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
- Class: `TTNNBailingMoEAttention` (lines 2198-2986)

### Ling-mini-2.0 Model Configuration
- `num_attention_heads`: 16
- `num_key_value_heads`: 4 (GQA with 4:1 ratio)
- `head_dim`: 128
- `hidden_size`: 2048
- `partial_rotary_factor`: 0.5 (only 64 of 128 dims get RoPE)
- `use_qk_norm`: True (QK normalization is enabled)

### Current Code Structure (T3K Mode)

| Component | Description | Lines of Code | Can Simplify? |
|-----------|-------------|---------------|---------------|
| `from_torch()` | Initializes projections, QK norm, RoPE, SDPA | ~105 | Moderate |
| `_maybe_all_gather()` | All-gather across mesh devices | ~35 | Minimal |
| `_to_replicated()` | Convert tensor topology for paged kernels | ~30 | Minimal (required) |
| `_apply_qk_norm()` | QK normalization for prefill | ~45 | Required |
| `_apply_qk_norm_decode()` | QK normalization for decode | ~35 | Required |
| `_apply_partial_rope()` | Partial RoPE for prefill | ~25 | Required |
| `_apply_partial_rope_decode()` | Partial RoPE for decode | ~45 | Could merge |
| `_ensure_replicated_tensor()` | Helper for cos/sin conversion | ~55 | Could simplify |
| `_forward_prefill()` | Prefill forward pass | ~160 | Moderate |
| `_forward_decode_paged()` | Decode forward pass with paged KV cache | ~175 | Moderate |
| `forward()` | Main entry point | ~50 | Minimal |

**Total: ~790 lines** (including comments and docstrings)

---

## Complexity Sources

### 1. Tensor Topology Handling (High Complexity)
**Problem:** The implementation handles multiple tensor topologies and requires converting between them.

**Current Code:**
- `_maybe_all_gather()`: Handles both async (ccl_manager) and sync (fallback) all-gather
- `_to_replicated()`: Converts all-gathered tensor to explicitly replicated topology via host round-trip
- `_ensure_replicated_tensor()`: Nested function handling TorchTTNNTensor, ttnn.Tensor, and torch.Tensor

**Why Complex:**
- Position embeddings (cos/sin) arrive with various tensor types and sharding states
- Paged attention kernels require specific tensor topologies
- The framework's default sharding doesn't match what kernels expect

**Simplification Opportunity:**
- Move tensor topology conversion to a dedicated utility module
- Standardize on one tensor type path (remove TorchTTNNTensor handling from attention)

### 2. Separate Prefill/Decode Paths (Medium Complexity)
**Problem:** Prefill and decode have significant code duplication.

**Common Operations:**
- Q/K/V projection
- All-gather
- QK normalization (different tensor shapes)
- RoPE application (different APIs)

**Why Separate:**
- Different tensor layouts: Prefill uses [B, H, S, D], Decode uses [1, B, H, D]
- Different SDPA APIs: Regular SDPA vs paged_sdpa_decode
- Different RoPE APIs: rotary_embedding vs rotary_embedding_llama

**Simplification Opportunity:**
- Create shared helper for projection + all-gather
- Document why separate paths are required

### 3. Partial Rotary Embedding (Medium Complexity)
**Problem:** Model uses partial_rotary_factor=0.5, requiring slice/concat operations.

**Current Code:**
- `_apply_partial_rope()`: Delegates to TTNNRotaryPositionEmbedding which handles partial rotary
- `_apply_partial_rope_decode()`: Manual slice/concat for decode path

**Why Complex:**
- Prefill uses `TTNNRotaryPositionEmbedding` which handles partial rotary internally
- Decode uses `rotary_embedding_llama` kernel which doesn't handle partial rotary

**Simplification Opportunity:**
- Unify to one RoPE implementation that handles both paths
- Or document why they must be different

### 4. Type Conversion/Wrapping (Low Complexity but Noisy)
**Problem:** Code frequently unwraps TorchTTNNTensor and converts between types.

**Examples:**
```python
if hasattr(tensor, "to_ttnn"):
    tensor = tensor.to_ttnn
if query_states.dtype != ttnn.bfloat16:
    query_states = ttnn.typecast(query_states, ttnn.bfloat16)
```

**Simplification Opportunity:**
- Add utility function `ensure_ttnn_bfloat16(tensor)` to consolidate conversions

---

## Step-by-Step Simplification Plan

### Phase 1: Prerequisites (Execute First)
Execute the two existing plans before this one:
1. `PLAN_bailing_dynamic_cache_removal.md`
2. `PLAN_bailing_single_device_removal.md`

### Phase 2: Utility Extraction

#### Step 2.1: Create Tensor Topology Utilities
**New file:** `modules/tensor_utils.py`

```python
def ensure_ttnn_bfloat16(tensor) -> ttnn.Tensor:
    """Convert any tensor to ttnn.Tensor with bfloat16 dtype."""
    # Handle TorchTTNNTensor
    # Handle torch.Tensor
    # Handle dtype conversion

def to_replicated_topology(tensor, device) -> ttnn.Tensor:
    """Convert tensor to explicitly replicated topology for paged kernels."""
    # Current _to_replicated() logic

def ensure_replicated_position_embeddings(cos, sin, device) -> tuple:
    """Ensure cos/sin have replicated topology."""
    # Current _ensure_replicated_tensor() logic
```

**Lines Removed from attention.py:** ~85

#### Step 2.2: Simplify _maybe_all_gather()
**Change:** Remove ccl_manager check, use consistent all_gather API.

**Before:**
```python
def _maybe_all_gather(self, tensor):
    if self._is_distributed:
        gathered = ttnn.experimental.all_gather_async(...)
        ttnn.synchronize_device(self.device)
    else:
        gathered = ttnn.all_gather(t, dim=-1, num_links=1)
    ...
```

**After:**
```python
def _maybe_all_gather(self, tensor):
    t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
    gathered = ttnn.all_gather(t, dim=-1, num_links=1)
    if gathered.dtype != ttnn.bfloat16:
        gathered = ttnn.typecast(gathered, ttnn.bfloat16)
    return gathered
```

**Rationale:** The async variant provides marginal benefit but adds complexity. Standard all_gather is sufficient for correctness. Performance optimization can be re-added later if profiling shows it's a bottleneck.

**Lines Removed:** ~15

### Phase 3: Consolidate RoPE Handling

#### Step 3.1: Document RoPE Divergence
**Add docstring explaining why prefill/decode use different RoPE:**

```python
# NOTE: Prefill uses TTNNRotaryPositionEmbedding which handles partial rotary internally
# via slice/pad/concat. Decode uses rotary_embedding_llama kernel directly which
# requires manual slice/concat because the kernel expects full head_dim input.
# This divergence is required by kernel API differences, not a bug.
```

#### Step 3.2: Move RoPE Slice/Concat to Utility
**Add to `modules/rope.py`:**

```python
def apply_partial_rope_decode(query, key, cos, sin, trans_mat):
    """Apply partial RoPE for decode path using rotary_embedding_llama."""
    # Current _apply_partial_rope_decode() logic
```

**Lines Removed from attention.py:** ~40

### Phase 4: Simplify Forward Passes

#### Step 4.1: Extract Common Projection Logic
**Add private helper:**

```python
def _project_qkv_t3k(self, hidden_states):
    """Project Q/K/V with proper sharding for T3K.

    Returns (query, key, value) after all-gather with shape:
        query: [batch, seq, num_heads * head_dim]
        key/value: [batch, seq, num_kv_heads * head_dim]
    """
    query_states = self.q_proj(hidden_states)
    hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
    key_states = self.k_proj(hidden_states_replicated)
    value_states = self.v_proj(hidden_states_replicated)
    ttnn.deallocate(hidden_states_replicated)

    query_states = self._maybe_all_gather(query_states)
    key_states = self._maybe_all_gather(key_states)
    value_states = self._maybe_all_gather(value_states)

    return query_states, key_states, value_states
```

**Lines Removed (duplicated in prefill/decode):** ~20

#### Step 4.2: Remove Inline _ensure_replicated_tensor
**Replace nested function with call to utility:**

**Before:** `_forward_prefill()` contains 55-line `_ensure_replicated_tensor()` nested function

**After:** `cos, sin = ensure_replicated_position_embeddings(cos, sin, self.device)`

**Lines Removed:** ~50

### Phase 5: Final Cleanup

#### Step 5.1: Consolidate Assertions
**Add single validation at entry:**

```python
def forward(self, ...):
    # T3K mode validation
    if self.device.get_num_devices() <= 1:
        raise RuntimeError(
            "TTNNBailingMoEAttention requires T3K mesh device (8 devices). "
            "Single device mode is not supported."
        )
    if past_key_values is not None and not isinstance(past_key_values, TTNNPagedAttentionKVCache):
        raise ValueError(
            f"Only TTNNPagedAttentionKVCache is supported, got {type(past_key_values).__name__}"
        )
```

#### Step 5.2: Remove Debug/Legacy Code
- Remove commented-out print statements
- Remove unused parameters (position_ids in forward)
- Remove legacy `past_key_value` parameter handling

**Lines Removed:** ~15

---

## Expected Outcome

### Code Reduction Summary

| Phase | Lines Removed | Description |
|-------|---------------|-------------|
| Prerequisites | ~100 | DynamicCache + single device removal |
| Phase 2 | ~100 | Tensor utilities extraction |
| Phase 3 | ~40 | RoPE consolidation |
| Phase 4 | ~70 | Forward pass cleanup |
| Phase 5 | ~15 | Debug/legacy removal |
| **Total** | **~325** | **From ~790 to ~465 lines** |

### Final Structure

```
TTNNBailingMoEAttention (~465 lines)
    __init__                        ~20 lines
    from_torch()                    ~80 lines
    preprocess_weights_impl()        ~5 lines
    move_weights_to_device_impl()   ~40 lines
    _project_qkv_t3k()              ~25 lines (NEW shared helper)
    _apply_qk_norm()                ~30 lines
    _apply_qk_norm_decode()         ~25 lines
    _forward_prefill()             ~100 lines
    _forward_decode_paged()        ~100 lines
    forward()                       ~40 lines

External Utilities:
    modules/tensor_utils.py         ~60 lines (NEW)
    modules/rope.py                 +40 lines (partial decode helper)
```

---

## Success Criteria

1. **Correctness:** Test `test_ling_mini_2_0.py` passes on T3K
   - Generated text is coherent
   - No runtime errors

2. **Simplicity:**
   - Total class lines reduced by ~40%
   - No nested functions
   - Single code path (T3K only)

3. **Maintainability:**
   - Clear separation of concerns (tensor utils, RoPE, attention)
   - Documented rationale for remaining complexity

4. **Performance:** No regression in generation speed
   - Run timing comparison before/after

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Async all_gather removal causes performance regression | Benchmark before/after; re-add if >10% regression |
| Utility extraction introduces bugs | Run tests after each phase |
| RoPE changes break output quality | Add PCC check before/after refactor |

---

## Verification Commands

```bash
# Run T3K test
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v

# Verify no single device code remains
grep -n "num_devices <= 1" models/experimental/tt_symbiote/modules/attention.py

# Verify no DynamicCache references
grep -n "DynamicCache" models/experimental/tt_symbiote/modules/attention.py

# Count lines
wc -l models/experimental/tt_symbiote/modules/attention.py
```

---

## References

- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` - Main implementation
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` - T3K test
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py` - Sharded linear modules
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py` - RoPE implementations
