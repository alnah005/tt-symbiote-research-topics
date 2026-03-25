# PLAN: Fix Ling Attention Module Tests

## Problem Description

The Ling (BailingMoeV2) model attention tests are failing on T3K (8-device Wormhole mesh). There are multiple interrelated issues that need to be addressed.

## Current Test Status

**Test:** `models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py`

**Error:** `KeyError: 'default'` when loading the model:
```
ROPE_INIT_FUNCTIONS[self.rope_type]
KeyError: 'default'
```

This is a model compatibility issue with the HuggingFace transformers version - the `BailingMoeV2RotaryEmbedding` class is looking for a `'default'` key in `ROPE_INIT_FUNCTIONS` that doesn't exist.

## Root Causes

Based on the research cache and existing plans, there are **four distinct issues**:

### Issue 1: HuggingFace Model Compatibility (BLOCKING)

**Location:** Remote HuggingFace model code (`modeling_bailing_moe_v2.py`)

**Problem:** The model's rotary embedding initialization expects a `ROPE_INIT_FUNCTIONS` dictionary to have a `'default'` key, but it doesn't exist. This prevents the model from loading at all.

**Status:** This is a **prerequisite** that must be fixed before any attention-specific tests can run.

### Issue 2: PlacementReplicate AttributeError

**Location:** `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/modules/linear.py:136-144`

**Problem:** `TTNNLinearIColShardedWRowSharded.forward()` checks `placement.dim` but `PlacementReplicate` objects have no `.dim` attribute.

**Error:**
```
AttributeError: 'PlacementReplicate' object has no attribute 'dim'
```

**Existing Plan:** `PLAN_ling_linear_placement_fix.md`

**Fix:** Use `hasattr(placement, 'dim')` before accessing `.dim`:
```python
if len(placements) == 1:
    placement = placements[0]
    if hasattr(placement, 'dim'):
        assert placement.dim == self.input_dim, ...
```

### Issue 3: Matmul Shape Mismatch

**Location:** Attention output projection

**Problem:** `nlp_concat_heads_decode` is called with `num_heads=self.num_heads` (16 total) but on T3K, each device only has `n_local_heads` (16/8 = 2) heads.

**Error:**
```
TT_FATAL @ matmul_device_operation.cpp:126: a_shape[-1] == b_shape[-2]
Mismatch: width=2048 height=256
```

**Existing Plan:** `PLAN_ling_shape_mismatch_fix.md`

**Fix:**
1. Use `n_local_heads = self.num_heads // self.device.get_num_devices()` for `nlp_concat_heads_decode`
2. Add `all_gather_async` before dense projection when in distributed mode

### Issue 4: Attention Decode Path Architecture

**Location:** `TTNNBailingMoEAttention._forward_decode_paged()` in `attention.py`

**Problem:** Multiple architectural issues compared to working TT-Transformers implementation:
- Unnecessary permutes causing precision loss and sharding errors
- Incorrect sharding configuration before `paged_update_cache`
- Missing decode-specific QK norm and RoPE methods

**Existing Plan:** `PLAN_ling_attention_refactor.md`

## Step-by-Step Implementation Plan

### Phase 1: Fix Model Loading (Prerequisite)

**Task 1.1:** Create a local patch for ROPE_INIT_FUNCTIONS

**File:** Create `models/experimental/tt_symbiote/patches/ling_rope_patch.py`

```python
"""Patch for Ling model RoPE initialization."""

def patch_rope_init_functions():
    """Add 'default' key to ROPE_INIT_FUNCTIONS."""
    import sys
    # Get the module if loaded
    module_name = 'transformers_modules.inclusionAI.Ling_hyphen_mini_hyphen_2_dot_0'
    for name, mod in sys.modules.items():
        if 'modeling_bailing_moe_v2' in name:
            if hasattr(mod, 'ROPE_INIT_FUNCTIONS'):
                if 'default' not in mod.ROPE_INIT_FUNCTIONS:
                    # Add default using linear scaling function
                    mod.ROPE_INIT_FUNCTIONS['default'] = mod.ROPE_INIT_FUNCTIONS.get('linear', lambda *args, **kwargs: None)
                    return True
    return False
```

**Alternative:** Use `trust_remote_code=True` with a local copy of the model that has the fix.

### Phase 2: Fix Linear Placement Issue

**Task 2.1:** Update `TTNNLinearIColShardedWRowSharded.forward()` in `linear.py`

**File:** `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/modules/linear.py`

**Lines:** 135-149

**Change:**
```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Forward pass through linear layer."""
    placements = input_tensor.tensor_topology().placements()

    if len(placements) == 1:
        placement = placements[0]
        # PlacementReplicate has no .dim - replicated tensors are valid input
        if hasattr(placement, 'dim'):
            assert (
                placement.dim == self.input_dim
            ), f"Input tensor must be sharded on dimension {self.input_dim}."
    elif len(placements) == 2:
        p0, p1 = placements[0], placements[1]
        if hasattr(p0, 'dim'):
            assert p0.dim == 0, f"Input tensor must be sharded on batch dim (0)."
        if hasattr(p1, 'dim'):
            assert p1.dim == self.input_dim, f"Input tensor must be sharded on dimension {self.input_dim}."
    # Note: Replicated tensors (no placements with .dim) are valid input
```

### Phase 3: Fix Shape Mismatch

**Task 3.1:** Add `n_local_heads` calculation in attention module

**Task 3.2:** Fix `nlp_concat_heads_decode` call to use `n_local_heads`

**Task 3.3:** Add `all_gather_async` before output projection in distributed mode

See `PLAN_ling_shape_mismatch_fix.md` for detailed implementation.

### Phase 4: Refactor Attention Decode Path

**Task 4.1:** Remove unnecessary permutes

**Task 4.2:** Add decode-specific QK norm method

**Task 4.3:** Add decode RoPE method using `rotary_embedding_llama`

**Task 4.4:** Remove explicit sharding before `paged_update_cache`

See `PLAN_ling_attention_refactor.md` for detailed implementation.

## Testing Strategy

### After Phase 1:
```bash
# Verify model loads without KeyError
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0 -x 2>&1 | head -100
```

### After Phase 2:
```bash
# Test linear layer with replicated tensors
pytest models/experimental/tt_symbiote/tests/test_attention.py -v --timeout=0
```

### After Phase 3:
```bash
# Test attention shape handling on T3K
pytest models/experimental/tt_symbiote/tests/test_attention.py -v --timeout=0
```

### After Phase 4:
```bash
# Full Ling model test
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0
```

## Success Criteria

1. **Model Loading:** Model loads without `KeyError` or other import errors
2. **Linear Placement:** No `AttributeError` for `PlacementReplicate.dim`
3. **Shape Matching:** No matmul shape mismatch errors
4. **Text Generation:** Coherent text output (not garbled)
5. **PCC Tests:**
   - Prefill: PCC > 0.99
   - Decode: PCC > 0.98

## Critical Files

| File | Purpose |
|------|---------|
| `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` | Main test file |
| `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/modules/linear.py` | Linear layer with placement issue |
| `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/modules/attention.py` | Attention implementation |
| `/home/ttuser/salnahari/tt-metal-ign/models/experimental/tt_symbiote/modules/moe.py` | MoE module (TTNNBailingMoE) |

## Related Plans

- `PLAN_ling_linear_placement_fix.md` - Detailed fix for PlacementReplicate issue
- `PLAN_ling_shape_mismatch_fix.md` - Detailed fix for shape mismatch issue
- `PLAN_ling_attention_refactor.md` - Comprehensive attention refactoring plan
- `PLAN_ling_attention_ttnn_pytorch_tests.md` - TTNN vs PyTorch equivalence tests

## Dependencies

- PyTorch with transformers library
- TTNN with paged attention support
- T3K hardware (8 Wormhole chips)
- Ling-mini-2.0 model weights from HuggingFace

## Notes

The current blocking issue is the HuggingFace model compatibility problem. Once that is resolved, the other issues (PlacementReplicate, shape mismatch, attention architecture) can be addressed incrementally.

The attention refactoring should follow the TT-Transformers reference implementation which has been verified to work correctly on T3K.
