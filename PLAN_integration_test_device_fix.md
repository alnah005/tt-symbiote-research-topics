# Plan: Fix Integration Test Device Setup (MeshShape Mismatch)

## Problem Description

The integration tests in `test_qwen3_linear_attention_ttnn.py` fail with:
```
MeshShape([1, 8]) != MeshShape([1, 1])
```

This error occurs when running tests that use the single-device `device` fixture, but the `TTNNQwen3LinearAttention` layer uses distributed linear layers (`TTNNLinearIReplicatedWColSharded`) that expect an 8-device mesh.

## Root Cause Analysis

### 1. Test Configuration
The single-device tests use:
```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_qwen3_linear_attention_ttnn_vs_pytorch(device):
    ...
    ttnn_layer = TTNNQwen3LinearAttention.from_torch(torch_layer, distributed=False)
```

The test correctly passes `distributed=False` to `from_torch()`, which should use non-distributed linear layers.

### 2. The Actual Bug
Looking at `TTNNQwen3LinearAttention.from_torch()`:

```python
@classmethod
def from_torch(cls, torch_layer, distributed: bool = True):
    ...
    # Choose linear layer classes based on distributed mode
    LinearClsIn = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
    LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
    LinearClsSmall = TTNNLinear  # Always non-sharded for small projections

    # Create TTNN linear projections
    new_layer.in_proj_qkv = LinearClsIn.from_torch(torch_layer.in_proj_qkv)
    new_layer.in_proj_z = LinearClsIn.from_torch(torch_layer.in_proj_z)
    ...
    new_layer.out_proj = LinearClsOut.from_torch(torch_layer.out_proj)
```

This is correct - when `distributed=False`, it uses `TTNNLinear` instead of `TTNNLinearIReplicatedWColSharded`.

### 3. The Real Issue - Weight Preprocessing with Mesh Mappers
The bug is in `TTNNLinearInputReplicatedWeightSharded.move_weights_to_device_impl()` (base class of `TTNNLinearIReplicatedWColSharded`):

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
        )
```

The `ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim)` creates a mesh mapper that expects an 8-device mesh shape. When this tensor is then moved to a single device, TTNN raises the MeshShape mismatch error because the tensor was prepared for sharding across 8 devices but the device is a single device (MeshShape [1, 1]).

### 4. Inheritance Chain
- `TTNNLinearIReplicatedWColSharded` extends `TTNNLinearInputReplicatedWeightSharded`
- `TTNNLinearInputReplicatedWeightSharded` extends `TTNNLinear`
- The `move_weights_to_device_impl()` in `TTNNLinearInputReplicatedWeightSharded` always uses `shard_tensor_to_mesh_mapper`, which creates a mapper expecting the full device mesh

### 5. Why Tests Are Failing
Even though `distributed=False` is passed and `TTNNLinear` is selected:
1. Tests may have been passing before but something changed in the infrastructure
2. OR the test file may have a different test (`test_qwen3_linear_attention_t3k`) that uses `distributed=True` and the mesh_device fixture, and there might be fixture contamination
3. OR there's a mismatch between the error message and the actual failing test

Let me verify by checking the test more carefully:

The failing tests are the **single-device tests** that correctly pass `distributed=False`. The issue is that the `TTNNLinearIReplicatedWColSharded` class is being used somewhere unexpectedly, OR there's a mesh mapper being applied to tensors that shouldn't have one.

## Step-by-Step Implementation Plan

### Step 1: Verify the Actual Failing Test
Run the test with verbose output to identify exactly which test and which operation is failing:
```bash
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py -v -x 2>&1 | head -100
```

### Step 2: Add Conditional Mesh Mapper in TTNNLinearInputReplicatedWeightSharded

The `move_weights_to_device_impl()` should only use mesh mappers when the device is actually a mesh:

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py`

**Current code (lines 237-253):**
```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
        )
    if isinstance(self.tt_bias_host, torch.Tensor):
        self.tt_bias_host = preprocess_linear_bias(
            self.tt_bias_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
        )
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
```

**Proposed fix:**
```python
def move_weights_to_device_impl(self):
    # Only use mesh mapper if device is a multi-device mesh
    num_devices = self.device.get_num_devices() if hasattr(self.device, 'get_num_devices') else 1
    use_mesh_mapper = num_devices > 1

    if isinstance(self.tt_weight_host, torch.Tensor):
        weights_mesh_mapper = ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim) if use_mesh_mapper else None
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=weights_mesh_mapper,
        )
    if isinstance(self.tt_bias_host, torch.Tensor):
        weights_mesh_mapper = ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim) if use_mesh_mapper else None
        self.tt_bias_host = preprocess_linear_bias(
            self.tt_bias_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=weights_mesh_mapper,
        )
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
```

### Step 3: Apply Same Fix to All Sharded Linear Classes

Apply the conditional mesh mapper fix to:
1. `TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl()` (lines 107-123)
2. `TTNNLinearLLamaIColShardedWRowSharded.move_weights_to_device_impl()` (lines 200-216)
3. `TTNNLinearInputReplicatedWeightSharded.move_weights_to_device_impl()` (lines 237-253)

### Step 4: Alternative Approach - Fix at Test Level

If the above fix is too invasive, an alternative is to ensure single-device tests never use sharded linear classes:

In `TTNNQwen3LinearAttention.from_torch()`, add a check:
```python
@classmethod
def from_torch(cls, torch_layer, distributed: bool = True):
    ...
    new_layer = cls(config, distributed=distributed)
    ...
```

And in `move_weights_to_device()` or `preprocess_weights()`, check `self.distributed` flag before using mesh mappers.

### Step 5: Verify the Fix

Run the single-device tests:
```bash
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py::test_qwen3_linear_attention_ttnn_vs_pytorch -v
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py::test_qwen3_linear_attention_seq_lengths -v
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py::test_qwen3_linear_attention_state_propagation -v
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py::test_qwen3_linear_attention_env_toggle -v
```

Run the multi-device test (requires T3K):
```bash
pytest models/experimental/tt_symbiote/tests/test_qwen3_linear_attention_ttnn.py::test_qwen3_linear_attention_t3k -v
```

## Success Criteria

1. All single-device tests pass without MeshShape mismatch errors
2. Multi-device tests (T3K) still work correctly with proper weight sharding
3. No regression in existing functionality
4. PCC comparison between TTNN and PyTorch outputs passes threshold (0.95+)

## Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py`
   - Add conditional mesh mapper usage in `move_weights_to_device_impl()` for all sharded linear classes

## Additional Notes

- The `distributed` flag in `TTNNQwen3LinearAttention.from_torch()` controls which linear class is used
- When `distributed=False`, `TTNNLinear` is used which doesn't use mesh mappers
- The error suggests that either:
  a. The wrong linear class is being instantiated, OR
  b. There's a tensor conversion somewhere that incorrectly applies a mesh mapper

- The fix should ensure mesh mappers are only created and applied when the device actually supports multiple devices
