# TT Symbiote Distributed Primitives

Source files: `models/experimental/tt_symbiote/modules/linear.py`, `models/experimental/tt_symbiote/modules/normalization.py`, `models/experimental/tt_symbiote/modules/rope.py`, `models/experimental/tt_symbiote/core/run_config.py`, `models/experimental/tt_symbiote/core/module.py`

---

## 1. TTNNLinearIColShardedWRowSharded

**Full name:** `TTNNLinearIColShardedWRowSharded` (`linear.py` lines 126-176)

Inherits from `TTNNLinearInputShardedWeightSharded` with fixed `input_dim=-1`, `weight_dim=-2`.

This class implements the row-parallel linear pattern: the weight is row-sharded (split along the input dimension, i.e. `dim=-2` of the transposed `[in, out]` weight), and the input is column-sharded (split along its last dimension). The matmul produces a partial result on each device that must be reduced.

### preprocess_weights_impl

Defined on the parent class `TTNNLinearInputShardedWeightSharded` (`linear.py` lines 103-123). Stores `self.weight` and `self.bias` on the host without converting to TTNN. Conversion is deferred to `move_weights_to_device_impl`.

### move_weights_to_device_impl

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),  # dim=-2
        )
    ...
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

The weight is sharded along `dim=-2` (the row dimension of the transposed weight, i.e. the input dimension of the linear). The bias, if present, is sharded along `dim=-1` (output dimension).

### forward

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
```

Asserts that the input tensor's topology placements are on `dim=-1` (single placement) or on `(dim=0, dim=-1)` (batch + column sharding with two placements). Then:

1. Reshapes input to 4-D.
2. Calls `ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
3. Calls `ttnn.experimental.reduce_scatter_minimal_async` with:
   - `dim=3`
   - `cluster_axis=1`
   - `num_links=1`
   - `topology=ttnn.Topology.Ring`
   - `multi_device_global_semaphore` from `self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1)`
   - `barrier_semaphore` from `self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1)`
4. Adds bias if present.
5. Reshapes to `input_tensor_shape[:-1] + [-1]`.

**Constraint:** The `@run_on_devices(DeviceArch.T3K)` decorator enforces that this forward can only be called when `MESH_DEVICE=T3K` in the environment. Calling it on any other architecture raises a `RuntimeError`.

---

## 2. TTNNLinearLLamaIColShardedWRowSharded

Subclass of `TTNNLinearIColShardedWRowSharded` (`linear.py` lines 197-222).

Overrides `move_weights_to_device_impl` to use `dtype=ttnn.bfloat8_b` instead of `bfloat16`.

Decorated with both `@deallocate_weights_after` and `@run_on_devices(DeviceArch.T3K)`. The `deallocate_weights_after` decorator automatically frees the device weight tensors after the forward pass.

---

## 3. TTNNLinearIReplicatedWColSharded

**Full name:** `TTNNLinearIReplicatedWColSharded` (`linear.py` lines 256-276)

Inherits from `TTNNLinearInputReplicatedWeightSharded` with `weight_dim=-1`.

This class implements the column-parallel linear pattern: the weight is sharded along its last dimension (output dimension), and the input is **replicated** across all devices. Each device computes a partial output along its slice of the output columns. No reduction is needed after the matmul.

### preprocess_weights_impl

Defined on the parent `TTNNLinearInputReplicatedWeightSharded` (`linear.py` lines 233-235). Stores `self.weight` and `self.bias` unmodified on the host.

### move_weights_to_device_impl

```python
def move_weights_to_device_impl(self):
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),  # dim=-1
        )
    ...
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
```

Both weight and bias are sharded along `dim=-1`.

### forward

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
```

1. Ensures `TILE_LAYOUT`.
2. Reshapes to 4-D.
3. Calls `ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
4. Adds bias if present.
5. Reshapes to `input_tensor_shape[:-1] + [-1]`.

No CCL operation is performed in `forward`. The caller is responsible for any subsequent all-gather if the full output dimension is needed. This matches the column-parallel contract.

---

## 4. TTNNDistributedRMSNorm

**Location:** `normalization.py` lines 100-151

Decorated with `@trace_enabled`.

### move_weights_to_device_impl

```python
def move_weights_to_device_impl(self):
    dim = self.torch_layer.weight.shape[0]
    self.weight_distributed = ttnn.as_tensor(
        self.torch_layer.weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // 32, 32]),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=(ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape))),
    )
    self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
```

The norm weight is reshaped to `[1, 1, dim//32, 32]` and then sharded along dimension 2 (the second-to-last axis, which after reshaping corresponds to slices of the hidden dimension). `dims=(None, 2)` in `ShardTensor2dMesh` means: no sharding on mesh axis 0 (rows), shard dimension 2 of the tensor across mesh axis 1 (columns).

### forward

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, inp):
```

Two-phase distributed RMSNorm:

1. `ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)` — computes per-shard statistics.
2. `ttnn.experimental.all_gather_async(tt_stats, dim=-1, cluster_axis=None, ...)` using semaphore handles from `self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1)` with `num_links=1, topology=ttnn.Topology.Linear`.
3. `ttnn.rms_norm_post_all_gather(inp, tt_stats, epsilon=..., weight=self.weight_distributed)` — applies the global statistics with the sharded weight.

Note: `cluster_axis` is not passed to `all_gather_async` here (the call uses `dim=-1` directly and semaphore index 1), so it operates without an explicit cluster axis argument.

---

## 5. TTNNDistributedRotaryPositionEmbedding

**Location:** `rope.py` lines 169-end

Covered in depth in Chapter 3 (Attention). Brief summary here for completeness:

- `move_weights_to_device_impl` builds a transformation matrix (`trans_mat`) of shape `[1, 1, dhead, dhead]` where `dhead = ttnn.TILE_SIZE`. The matrix swaps even/odd pairs and negates for the rotation. It is replicated to all devices via `ttnn.ReplicateTensorToMesh`.
- `forward(q, k, cos, sin)` calls `ttnn.experimental.rotary_embedding_llama` which is designed for multi-device tensor-parallel scenarios.
- No explicit CCL calls are made in the forward pass; the operation is element-wise per device.

---

## 6. Configuration Dataclasses (core/run_config.py)

### CCLManagerConfig

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None      # defaults to 1
    topology: Optional[Any] = None       # defaults to ttnn.Topology.Linear
```

A lightweight configuration holder. It does not itself create semaphores; it stores parameters that callers may use when creating a `TT_CCL` instance.

### DistributedTensorConfig

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any
    mesh_composer: Any
    logical_shape_fn: Optional[Any] = None
```

Pairs a `mesh_mapper` (used when placing a tensor onto the mesh) with a `mesh_composer` (used when gathering it back to a single tensor). The optional `logical_shape_fn` is called to translate a sharded tensor's shape to its logical (full) shape.

The helper `logical_shape_for_batch_channel_sharding(mesh_shape)` (lines 55-61) returns a closure that reconstructs the full logical shape for 2-D batch/channel sharding:

```
logical_shape = [shape[0] * mesh_rows] + shape[1:-1] + [shape[-1] * mesh_cols]
```

### DistributedConfig

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None
```

The top-level distributed configuration object. In `__post_init__`:

- If `tensor_config` is None and `mesh_device.get_num_devices() > 1`, creates a default `DistributedTensorConfig` using `ShardTensor2dMesh` on `(0, -1)` (batch × channel) with the default `logical_shape_for_batch_channel_sharding`.
- If `ccl_manager` is None and `mesh_device.get_num_devices() > 1`, creates a `TT_CCL(mesh_device)` instance.

`get_tensor_config_for_tensor(module_name, tensor)` checks whether the tensor is evenly divisible by `mesh_device.shape[-1]` and `mesh_device.shape[0]`. If not, it falls back to `ReplicateTensorToMesh` and logs a warning. This is the mechanism by which Symbiote modules automatically handle tensors that cannot be evenly sharded.

---

## 7. @run_on_devices Decorator

**Location:** `core/module.py` lines 277-321

```python
def run_on_devices(*allowed_archs: DeviceArch):
    ...
    def wrapper(self, *args, **kwargs):
        mesh_device = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
        if mesh_device not in allowed_set:
            raise RuntimeError(...)
        return func(self, *args, **kwargs)
```

The decorator reads the `MESH_DEVICE` environment variable and looks it up in `MeshShapeToDeviceArch`:

| Env value | DeviceArch |
|-----------|-----------|
| `"N150"`  | `DeviceArch.N150` |
| `"N300"`  | `DeviceArch.N300` |
| `"T3K"`   | `DeviceArch.T3K` |
| `"TG"`    | `DeviceArch.TG` |
| `"P150"`  | `DeviceArch.P150` |
| `"P300"`  | `DeviceArch.P300` |
| `"P150x4"` | `DeviceArch.P150x4` |
| `"P150x8"` | `DeviceArch.P150x8` |
| `"BHGLX"` | `DeviceArch.BHGLX` |

All distributed forward methods in Symbiote (`TTNNLinearIColShardedWRowSharded.forward`, `TTNNLinearIReplicatedWColSharded.forward`, `TTNNDistributedRMSNorm.forward`) use `@run_on_devices(DeviceArch.T3K)`, meaning they will raise `RuntimeError` if `MESH_DEVICE` is not `"T3K"`. No TG / Galaxy variant exists yet in Symbiote.

---

## Navigation

| | |
|---|---|
| Previous | [tt_transformers_parallelism.md](tt_transformers_parallelism.md) |
| Next | [integration_strategy.md](integration_strategy.md) |
| Chapter home | [index.md](index.md) |
