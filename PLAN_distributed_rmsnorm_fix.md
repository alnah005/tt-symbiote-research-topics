# PLAN: Distributed RMSNorm Fix for Garbled Output

## Problem Description

Model output becomes garbled after adding Distributed RMS norm to the Ling-mini-2.0 (BailingMoeV2) model. The `TTNNDistributedRMSNorm` module is now mapped for the decoder layer's `input_layernorm`, but the model produces incoherent text.

**Symptoms:**
- Garbled/incoherent text output from the model
- Component-level tests may pass, but full model inference fails
- Issue appears after the Distributed RMS norm was integrated

## Root Cause Analysis

### Issue 1: Weight Shaping Mismatch (HIGH CONFIDENCE)

**Location:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`, lines 116-124

**Problem:** The `TTNNDistributedRMSNorm.move_weights_to_device_impl()` method shapes weights incorrectly for distributed operation:

```python
def move_weights_to_device_impl(self):
    """Move weights to TTNN device."""
    dim = self.torch_layer.weight.shape[0]
    self.weight_distributed = ttnn.as_tensor(
        self.torch_layer.weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // 32, 32]),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=(ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape))),
    )
    self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
```

**Issues:**
1. **dims=(None, 2) sharding is incorrect** - The weight shape is `[1, 1, dim // 32, 32]`, and sharding on dim=2 means sharding the `dim // 32` dimension. This doesn't properly distribute weights across devices.
2. **No handling for non-T3K devices** - The weight is always prepared for 2D mesh, but the forward pass is only decorated for T3K.
3. **Missing weight layout consideration** - The test utility `distributed_norm_test_utils.py` shows weight can be either TILE_LAYOUT or ROW_MAJOR_LAYOUT, and the sharding differs:
   - ROW_MAJOR: Reshape to `(num_mesh_devices, 1, -1, 32)` and shard over dim 0
   - TILE: Reshape to `(1, 1, 1, hidden_dim)` and shard over dim -1

**Reference Implementation (from `distributed_norm_test_utils.py`):**
```python
if weight_layout == ttnn.ROW_MAJOR_LAYOUT:
    weight_shape = (num_mesh_devices, 1, -1, 32)
    weight_shard_dim = 0
else:
    weight_shape = (1, 1, 1, hidden_dim)
    weight_shard_dim = -1

ttnn_weight = ttnn.from_torch(
    torch_weight.reshape(weight_shape),
    dtype=ttnn.bfloat16,
    device=mesh_device,
    layout=weight_layout,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=weight_shard_dim),
)
```

### Issue 2: Missing cluster_axis Parameter (MEDIUM CONFIDENCE)

**Location:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`, lines 134-141

**Problem:** The `all_gather_async` call is missing the `cluster_axis` parameter:

```python
tt_stats = ttnn.experimental.all_gather_async(
    tt_stats,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
)
```

**Comparison with reference implementations:**

1. **tt_distributed_rmsnorm (ccl.py)** uses `tt_all_gather` which sets `cluster_axis=1`.
2. **tt_sharded_distributed_rmsnorm (ccl.py)** explicitly passes `cluster_axis=cluster_axis` (=1).
3. **RMSNorm2D** explicitly passes `cluster_axis=1`.

Missing `cluster_axis` could cause the all_gather to operate on the wrong axis, corrupting the statistics.

### Issue 3: Stats Tensor Reshaping (MEDIUM CONFIDENCE)

**Location:** The `tt_distributed_rmsnorm` function in `ccl.py` includes a reshape step after `rms_norm_pre_all_gather`:

```python
tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
padded_shape = (1, 1, inp.shape[-2], 32)
tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))  # TODO: Figure out why we need this
```

The `TTNNDistributedRMSNorm` does NOT include this reshape step. The comment "TODO: Figure out why we need this" suggests this may be a workaround for a known issue.

### Issue 4: Device Architecture Restriction

**Location:** The `@run_on_devices(DeviceArch.T3K)` decorator on the forward method.

The forward method is only enabled for T3K devices. If running on other mesh configurations (N300, TG), this could cause fallback issues or unexpected behavior.

## Step-by-Step Implementation Plan

### Step 1: Fix Weight Shaping

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`

**Changes:**
```python
def move_weights_to_device_impl(self):
    """Move weights to TTNN device."""
    dim = self.torch_layer.weight.shape[0]
    num_devices = self.device.get_num_devices()

    # Use ShardTensorToMesh with proper weight shape
    # ROW_MAJOR: (num_devices, 1, dim // num_devices // 32, 32), shard on dim 0
    # This matches the pattern in distributed_norm_test_utils.py
    weight_per_device = dim // num_devices
    weight_shape = (num_devices, 1, weight_per_device // 32, 32)

    self.weight_distributed = ttnn.from_torch(
        self.torch_layer.weight.reshape(weight_shape),
        dtype=ttnn.bfloat16,
        device=self.device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
    )
```

### Step 2: Add cluster_axis to all_gather_async

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`

**Changes:**
```python
tt_stats = ttnn.experimental.all_gather_async(
    tt_stats,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
    cluster_axis=1,  # ADD THIS
    mesh_device=self.device,  # ADD THIS - may be required
)
```

### Step 3: Add Stats Reshape (Following ccl.py Pattern)

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py`

**Changes:**
```python
def forward(self, inp):
    original_shape = inp.shape
    if len(original_shape) == 3:
        inp = ttnn.unsqueeze(inp, 1)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)

    # ADD: Reshape stats (following ccl.py pattern)
    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))

    # AllGather stats
    tt_stats = ttnn.experimental.all_gather_async(...)

    # ... rest of the method
```

### Step 4: Consider Using compute_kernel_config

The reference implementations pass `compute_kernel_config` to both pre and post all_gather operations. Add this for consistency:

```python
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

tt_stats = ttnn.rms_norm_pre_all_gather(
    inp,
    dtype=ttnn.bfloat16,
    compute_kernel_config=compute_kernel_config
)
```

### Step 5: Handle Device Architecture Properly

The `@run_on_devices(DeviceArch.T3K)` decorator limits the forward method to T3K only. Consider:

1. Extending to support other distributed devices (N300 with 2 devices)
2. Or falling back to non-distributed `TTNNRMSNorm` for non-T3K devices

### Step 6: Test the Fix

```bash
# Run component test
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0_attention_pcc_stages.py -v --timeout=0

# Run full model test
unset TT_VISIBLE_DEVICES && tt-smi -r
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0
```

## Alternative Approach: Use Existing RMSNorm2D

Instead of fixing `TTNNDistributedRMSNorm`, consider using the well-tested `RMSNorm2D` from `/home/ttuser/salnahari/tt-metal/models/common/modules/rmsnorm/rmsnorm_2d.py` which:

1. Has proper weight shaping with `MeshMapperConfig`
2. Correctly uses `cluster_axis=1` for all_gather
3. Handles both decode and prefill modes
4. Has been tested in production deployments

## Success Criteria

1. **Coherent text generation** - Output should be readable sentences matching the question context
2. **No garbled characters** - No random symbols or unintelligible text
3. **PCC validation** - Component-level PCC > 0.99 for RMSNorm operations
4. **Works on T3K mesh** - Distributed operation functions correctly across all 8 devices

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `models/experimental/tt_symbiote/modules/normalization.py` | Fix weight shaping, add cluster_axis, add stats reshape | HIGH |
| `models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` | May need adjustments for device config | MEDIUM |

## Summary of Root Causes

| Issue | Severity | Confidence | Description |
|-------|----------|------------|-------------|
| Weight shaping | HIGH | HIGH | dims=(None, 2) doesn't properly distribute weights |
| Missing cluster_axis | HIGH | MEDIUM | all_gather may operate on wrong axis |
| Missing stats reshape | MEDIUM | MEDIUM | Stats tensor shape may not match expectations |
| Missing compute_kernel_config | LOW | LOW | May affect numerical precision |

## Status

- [ ] Step 1: Fix weight shaping
- [ ] Step 2: Add cluster_axis to all_gather_async
- [ ] Step 3: Add stats reshape
- [ ] Step 4: Add compute_kernel_config
- [ ] Step 5: Handle device architecture
- [ ] Step 6: Test the fix
