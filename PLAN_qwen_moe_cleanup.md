# Plan: Enable Sparse Matmul and W1/W3 Fusion as Default for Qwen 3.5 MoE

**Date:** 2026-03-17
**Status:** Draft
**Author:** Architect Agent

---

## 1. Problem Description

The current Qwen 3.5 MoE implementation (`TTNNQwenExperts`) uses environment variables to toggle between:
1. **Sparse matmul** vs **batched matmul** (legacy)
2. **Fused w1/w3 projection** vs **separate w1 and w3 matmuls**

These toggles were useful during development to compare approaches, but now that sparse matmul and w1/w3 fusion have been validated as the optimal path, the legacy code should be removed to:
- Reduce code complexity and maintenance burden
- Remove dead code paths that are no longer tested or used
- Simplify the module for future development
- Reduce memory footprint by eliminating mutually-exclusive weight storage logic

---

## 2. Current State Analysis

### 2.1 File Location

**Primary file to modify:**
```
/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py
```

### 2.2 Environment Variables (to be removed)

| Variable | Default | Purpose |
|----------|---------|---------|
| `TT_QWEN_USE_SPARSE_MATMUL` | `"1"` (enabled) | Toggle sparse_matmul vs batched matmul |
| `TT_QWEN_FUSED_GATE_UP` | `"1"` (enabled) | Toggle fused w1/w3 vs separate matmuls |

Both are defined at the top of `qwen_moe.py` (lines 28-32):

```python
# Feature flag for sparse_matmul optimization
TT_QWEN_USE_SPARSE_MATMUL = os.environ.get("TT_QWEN_USE_SPARSE_MATMUL", "1").lower() in ("1", "true", "yes")

# Feature flag for fused w1/w3 (gate/up) matmul optimization
TT_QWEN_FUSED_GATE_UP = os.environ.get("TT_QWEN_FUSED_GATE_UP", "1").lower() in ("1", "true", "yes")
```

### 2.3 Class Affected: `TTNNQwenExperts`

The `TTNNQwenExperts` class (starting at line 205) has conditional logic throughout:

#### 2.3.1 `preprocess_weights_impl()` (lines 221-276)

**Current behavior:**
- If fused mode (`TT_QWEN_FUSED_GATE_UP and TT_QWEN_USE_SPARSE_MATMUL`):
  - Creates `tt_w1_w3_proj` (concatenated weights)
  - Sets `tt_w1_proj = None` and `tt_w3_proj = None`
- If unfused mode:
  - Creates `tt_w1_proj` and `tt_w3_proj` separately
  - Sets `tt_w1_w3_proj = None`

**After cleanup:**
- Only create `tt_w1_w3_proj` (fused weights)
- Remove `tt_w1_proj` and `tt_w3_proj` entirely

#### 2.3.2 `move_weights_to_device_impl()` (lines 278-364)

**Current behavior:**
- Conditional weight movement based on fused mode
- Creates either `_fused_gate_up_program_config` OR `_gate_up_program_config` based on mode
- Creates `_down_program_config` always

**After cleanup:**
- Only move `tt_w1_w3_proj`
- Only create `_fused_gate_up_program_config`
- Remove `_gate_up_program_config`

#### 2.3.3 `forward()` (lines 366-686)

**Current behavior (lines 466-616):**
```python
if TT_QWEN_USE_SPARSE_MATMUL:
    # ===== SPARSE_MATMUL PATH (optimized) =====
    # ... ~130 lines of sparse matmul code ...

    if TT_QWEN_FUSED_GATE_UP:
        # ===== FUSED GATE/UP PATH =====
        # Single sparse_matmul for w1 and w3
    else:
        # ===== UNFUSED PATH (original) =====
        # Two separate sparse_matmuls
else:
    # ===== BATCHED MATMUL PATH (legacy) =====
    # ... ~25 lines of batched matmul code ...
```

**After cleanup:**
- Remove the outer `if TT_QWEN_USE_SPARSE_MATMUL` condition
- Remove the inner `if TT_QWEN_FUSED_GATE_UP` condition
- Keep only the fused sparse_matmul path

### 2.4 Legacy Code to Remove

| Section | Lines (approx) | Description |
|---------|----------------|-------------|
| Environment variables | 27-32 | Two env var definitions |
| Unfused weight creation | 250-264 | `tt_w1_proj`, `tt_w3_proj` creation |
| Unfused weight movement | 294-297 | `tt_w1_proj`, `tt_w3_proj` to device |
| Unfused program config | 342-350 | `_gate_up_program_config` creation |
| Unfused forward path | 533-559 | Separate w1/w3 sparse_matmuls |
| Batched matmul path | 593-616 | Entire batched matmul section |
| Pad block size logic | 409-412 | `pad_block_size` conditional |

### 2.5 Documentation to Update

The docstring at the top of `qwen_moe.py` (lines 5-21) describes the environment variables and their usage. This needs to be updated to remove references to these toggles.

---

## 3. Files to Modify

| File | Change Type | Description |
|------|-------------|-------------|
| `modules/qwen_moe.py` | Major refactor | Remove legacy paths, simplify `TTNNQwenExperts` |

### 3.1 No Other Files Need Changes

A search confirmed that `TT_QWEN_USE_SPARSE_MATMUL` and `TT_QWEN_FUSED_GATE_UP` are only referenced in `qwen_moe.py`.

---

## 4. Step-by-Step Implementation Plan

### Step 1: Update Module Docstring

**Location:** Lines 5-21

**Action:** Remove the environment variable documentation section. Update the class descriptions to reflect that sparse_matmul with fused w1/w3 is the only implementation.

**Before:**
```python
"""Qwen3.5-35B-A3B specific MoE implementations for TTNN.

This module contains Qwen-specific subclasses that inherit from the GLM base classes
in moe.py. Key differences:
- TTNNQwenMoERouterDecode: Uses softmax activation instead of sigmoid
- TTNNQwenExperts: Uses batched matmul instead of sparse_matmul, with 4D weight reshaping
- TTNNQwen3MoE: Handles Qwen's shared_expert (singular) and optional shared_expert_gate

Environment Variables:
- TT_QWEN_CPU_EXPERTS: Set to "1" to use CPU fallback for experts (for debugging).
  When enabled, TTNNQwenExperts is NOT created and the PyTorch experts are used instead.
- TT_QWEN_USE_SPARSE_MATMUL: Set to "1" (default) to use sparse_matmul optimization,
  or "0" to use batched matmul (legacy path).
- TT_QWEN_FUSED_GATE_UP: Set to "1" (default) to fuse w1 (gate) and w3 (up) projections
  into a single sparse_matmul. This eliminates duplicate memory bandwidth by reading
  the input tensor once instead of twice. Set to "0" to use separate matmuls.
"""
```

**After:**
```python
"""Qwen3.5-35B-A3B specific MoE implementations for TTNN.

This module contains Qwen-specific subclasses that inherit from the GLM base classes
in moe.py. Key differences:
- TTNNQwenMoERouterDecode: Uses softmax activation instead of sigmoid
- TTNNQwenExperts: Uses sparse_matmul with fused w1/w3 projection for efficiency
- TTNNQwen3MoE: Handles Qwen's shared_expert (singular) and optional shared_expert_gate

Environment Variables:
- TT_QWEN_CPU_EXPERTS: Set to "1" to use CPU fallback for experts (for debugging).
  When enabled, TTNNQwenExperts is NOT created and the PyTorch experts are used instead.
"""
```

### Step 2: Remove Environment Variable Definitions

**Location:** Lines 27-32

**Action:** Remove the two environment variable definitions.

**Remove:**
```python
# Feature flag for sparse_matmul optimization
TT_QWEN_USE_SPARSE_MATMUL = os.environ.get("TT_QWEN_USE_SPARSE_MATMUL", "1").lower() in ("1", "true", "yes")

# Feature flag for fused w1/w3 (gate/up) matmul optimization
# When enabled, combines two sparse_matmul calls into one, reducing memory bandwidth by 50%
TT_QWEN_FUSED_GATE_UP = os.environ.get("TT_QWEN_FUSED_GATE_UP", "1").lower() in ("1", "true", "yes")
```

### Step 3: Update `TTNNQwenExperts` Class Docstring

**Location:** Lines 205-218

**Action:** Update to reflect the single implementation path.

**Before:**
```python
class TTNNQwenExperts(TTNNExperts):
    """Qwen-specific experts using batched matmul instead of sparse_matmul.

    This subclass overrides preprocess_weights_impl() to pre-reshape weights to 4D
    and forward() to use batched matmul. This approach works better for Qwen's
    large number of experts (256) with top-8 routing.

    Inheritance:
        - __init__(): Inherited (unchanged)
        - _get_num_experts_per_device(): Inherited (unchanged)
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): OVERRIDDEN - reshapes to 4D, shards on dim=1
        - move_weights_to_device_impl(): OVERRIDDEN - simplified (no reshape needed)
        - forward(): OVERRIDDEN - uses batched matmul instead of sparse_matmul
    """
```

**After:**
```python
class TTNNQwenExperts(TTNNExperts):
    """Qwen-specific experts using sparse_matmul with fused w1/w3 projection.

    This subclass overrides preprocess_weights_impl() to create fused gate/up weights,
    and forward() to use sparse_matmul for efficient expert computation. The fused
    w1/w3 approach reduces memory bandwidth by 50% compared to separate matmuls.

    Inheritance:
        - __init__(): Inherited (unchanged)
        - _get_num_experts_per_device(): Inherited (unchanged)
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): OVERRIDDEN - creates fused w1_w3 weights, 4D shape
        - move_weights_to_device_impl(): OVERRIDDEN - moves fused weights, creates program configs
        - forward(): OVERRIDDEN - uses sparse_matmul with fused gate/up projection
    """
```

### Step 4: Simplify `preprocess_weights_impl()`

**Location:** Lines 221-276

**Action:** Remove conditional logic, keep only fused path.

**After:**
```python
def preprocess_weights_impl(self):
    """Preprocess expert weights: create fused w1_w3 weights for sparse_matmul.

    Creates a single fused weight tensor by concatenating w1 (gate) and w3 (up)
    projections. This enables a single sparse_matmul call instead of two separate
    matmuls, reducing memory bandwidth by 50%.

    Shape: (num_experts, H, I) x2 -> (1, num_experts, H, 2*I)
    """
    # Reshape to 4D on host (torch) before converting to ttnn
    torch_w1_4d = self.torch_w1_proj.unsqueeze(0).to(torch.bfloat16)
    torch_w3_4d = self.torch_w3_proj.unsqueeze(0).to(torch.bfloat16)
    torch_w2_4d = self.torch_w2_proj.unsqueeze(0).to(torch.bfloat16)

    # Create fused w1_w3 weights (gate and up projections concatenated)
    torch_w1_w3_fused = torch.cat([torch_w1_4d, torch_w3_4d], dim=-1)
    self.tt_w1_w3_proj = ttnn.from_torch(
        torch_w1_w3_fused,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
    )
    del torch_w1_w3_fused

    # w2 (down projection) - not fused
    self.tt_w2_proj = ttnn.from_torch(
        torch_w2_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
    )

    del self.torch_w1_proj
    del self.torch_w3_proj
    del self.torch_w2_proj
```

### Step 5: Simplify `move_weights_to_device_impl()`

**Location:** Lines 278-364

**Action:** Remove conditional logic, keep only fused sparse_matmul path.

**After:**
```python
def move_weights_to_device_impl(self):
    """Move preprocessed weights to device and create sparse_matmul program configs."""
    self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
    self.num_devices = self.device.get_num_devices()
    self.num_dispatch_devices = self.device.get_num_devices()

    # Move fused w1_w3 and w2 weights to device
    self.tt_w1_w3_proj = ttnn.to_device(self.tt_w1_w3_proj, self.device)
    self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

    # Create expert mapping tensors for all-to-all ops
    self.expert_mapping_tensors = ttnn.from_torch(
        torch.eye(self.num_devices, dtype=torch.int32)
        .repeat_interleave(self.num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0),
        device=self.device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Create remap topk mask for expert token remap
    self.remap_topk_mask = ttnn.from_torch(
        torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
        device=self.device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Program configs for sparse_matmul
    hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
    intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE

    # Fused gate/up program config (output is 2*intermediate_size)
    self._fused_gate_up_program_config = _make_sparse_matmul_program_config(
        device=self.device,
        out_features=int(self.intermediate_size * 2),
        in0_block_w=min(4, hidden_tiles),
        per_core_M=1,
    )

    # Down projection program config
    self._down_program_config = _make_sparse_matmul_program_config(
        device=self.device,
        out_features=int(self.hidden_size),
        in0_block_w=min(4, intermediate_tiles),
        per_core_M=1,
    )

    self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
```

### Step 6: Simplify `forward()` Method

**Location:** Lines 366-686

**Action:** Remove all conditional paths, keep only the fused sparse_matmul implementation.

The forward method should:
1. Remove the `TT_QWEN_USE_SPARSE_MATMUL` conditional (always use sparse_matmul)
2. Remove the `TT_QWEN_FUSED_GATE_UP` conditional (always use fused path)
3. Remove the batched matmul path entirely (lines 593-616)
4. Simplify padding logic (always use `SPARSITY_BLOCK_SIZE`)

**Key changes in forward():**

**Before (lines 408-412):**
```python
# Padding: Use SPARSITY_BLOCK_SIZE for sparse path, TILE_SIZE for batched path
if TT_QWEN_USE_SPARSE_MATMUL:
    pad_block_size = SPARSITY_BLOCK_SIZE
else:
    pad_block_size = 32  # TILE_SIZE
```

**After:**
```python
pad_block_size = SPARSITY_BLOCK_SIZE
```

**Remove entirely:**
- Lines 533-559: The unfused `else` block with separate w1/w3 sparse_matmuls
- Lines 593-616: The batched matmul path

**Final structure of forward():**
```python
def forward(self, x, topk_experts_indices, topk_experts_weights):
    """Execute expert pipeline with fused sparse_matmul.

    Uses sparse_matmul to only compute activated experts (no repeat).
    Fused w1/w3 projection reduces memory bandwidth by 50%.
    """
    # ... dimension extraction and padding (unchanged) ...

    # Padding: Always use SPARSITY_BLOCK_SIZE
    pad_block_size = SPARSITY_BLOCK_SIZE

    # ... padding logic (unchanged) ...

    # STEP 1-2: ALL_TO_ALL_DISPATCH (unchanged)

    # STEP 3: SPARSE MATMUL COMPUTATION
    # Generate sparsity tensor
    # ... sparsity tensor generation (unchanged) ...

    # Fused gate/up sparse_matmul
    w1_w3_out = ttnn.sparse_matmul(
        x_sparse,
        self.tt_w1_w3_proj,
        sparsity=sparsity_t,
        # ... config (unchanged) ...
    )

    # Split fused output into w1 and w3 components
    # ... slice operations (unchanged) ...

    # SwiGLU activation and down projection (unchanged)

    # STEP 5-7: COMBINE AND WEIGHT (unchanged)
```

### Step 7: Update Comments in `TTNNQwen3MoE.from_torch()`

**Location:** Line 749

**Action:** Update comment to reflect the new implementation.

**Before:**
```python
# KEY DIFFERENCE: Use Qwen-specific experts with batched matmul
```

**After:**
```python
# Use Qwen-specific experts with fused sparse_matmul
```

---

## 5. Success Criteria

### 5.1 Functional Requirements

1. **Behavior unchanged**: The module produces identical outputs to the current implementation with `TT_QWEN_USE_SPARSE_MATMUL=1` and `TT_QWEN_FUSED_GATE_UP=1`
2. **Tests pass**: All existing tests in `test_qwen_moe_accuracy.py` pass without modification
3. **No environment variable dependencies**: The removed env vars no longer affect behavior

### 5.2 Code Quality Requirements

1. **No dead code**: All conditional paths for legacy implementations are removed
2. **No unused variables**: `tt_w1_proj`, `tt_w3_proj`, `_gate_up_program_config` are removed
3. **Updated documentation**: Docstrings accurately describe the implementation
4. **Clean imports**: Remove `os` import if no longer needed (check for `TT_QWEN_CPU_EXPERTS`)

### 5.3 Testing Requirements

1. Run `pytest test_qwen_moe_accuracy.py -v` - all tests pass
2. Run with `TT_QWEN_USE_SPARSE_MATMUL=0` - should have no effect (env var ignored)
3. Run with `TT_QWEN_FUSED_GATE_UP=0` - should have no effect (env var ignored)

---

## 6. Risk Assessment

### 6.1 Low Risk

- The default behavior is already sparse_matmul + fused w1/w3
- We are removing code paths that are not exercised in production
- Tests already validate the sparse_matmul path

### 6.2 Mitigation

- Create a backup branch before changes
- Run full test suite before and after
- Keep `TT_QWEN_CPU_EXPERTS` for debugging (not being removed)

---

## 7. Estimated Effort

| Task | Estimate |
|------|----------|
| Update docstrings | 15 min |
| Remove env vars | 5 min |
| Simplify `preprocess_weights_impl()` | 15 min |
| Simplify `move_weights_to_device_impl()` | 15 min |
| Simplify `forward()` | 30 min |
| Testing and validation | 30 min |
| **Total** | **~2 hours** |

---

## 8. Summary of Changes

| Component | Current State | After Cleanup |
|-----------|---------------|---------------|
| `TT_QWEN_USE_SPARSE_MATMUL` | Env var, default "1" | Removed (always sparse) |
| `TT_QWEN_FUSED_GATE_UP` | Env var, default "1" | Removed (always fused) |
| `tt_w1_proj` | Conditionally created | Removed |
| `tt_w3_proj` | Conditionally created | Removed |
| `_gate_up_program_config` | Conditionally created | Removed |
| Batched matmul path | ~25 lines | Removed |
| Unfused sparse_matmul path | ~27 lines | Removed |
| **Net lines removed** | - | ~100+ lines |

---

## Appendix: Code References

### A.1 Current Weight Attributes After Cleanup

```python
# In TTNNQwenExperts after cleanup:
self.tt_w1_w3_proj     # Fused gate+up weights: [1, E, H, 2*I]
self.tt_w2_proj        # Down projection: [1, E, I, H]
# Removed: self.tt_w1_proj, self.tt_w3_proj
```

### A.2 Current Program Config Attributes After Cleanup

```python
# In TTNNQwenExperts after cleanup:
self._fused_gate_up_program_config  # For fused w1_w3 matmul
self._down_program_config           # For w2 matmul
self._expert_compute_cfg            # Compute kernel config
# Removed: self._gate_up_program_config
```
