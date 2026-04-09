# Multi-Device Simplification: Current State and Near-Term Strategy

This file analyzes how TT-Lang's grid model interacts with TT-Symbiote's multi-device distribution layer, documents the current multi-device code patterns, and recommends the near-term approach for deploying TT-Lang kernels in multi-device configurations.

## Current Multi-Device Architecture in TT-Symbiote

TT-Symbiote handles multi-device distribution through three cooperating abstractions defined in `core/run_config.py`:

### DistributedConfig

The top-level configuration object that bundles device mesh, tensor sharding, and collective communication:

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None
```

In `__post_init__`, when `mesh_device.get_num_devices() > 1`, it automatically creates:
- A `DistributedTensorConfig` with `ShardTensor2dMesh` (for distributing tensors) and `ConcatMesh2dToTensor` (for gathering results)
- A `TT_CCL` instance (from `models.tt_transformers.tt.ccl`) for collective communication

### DistributedTensorConfig

Controls how tensors are mapped onto the device mesh:

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any       # e.g., ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, -1))
    mesh_composer: Any     # e.g., ttnn.ConcatMesh2dToTensor(mesh_device, mesh_device.shape, (0, -1))
    logical_shape_fn: Optional[Any] = None  # Reconstructs logical shape from shard shape
```

The default sharding strategy splits along batch (dim 0) and the last dimension (dim -1). When a tensor's shape is not evenly divisible by the mesh shape, `get_tensor_config_for_tensor` falls back to replication via `ttnn.ReplicateTensorToMesh`.

### CCLManagerConfig

Configuration for collective communication links:

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None     # defaults to 1
    topology: Optional[Any] = None      # defaults to ttnn.Topology.Linear
```

### Concrete Example: TTNNDistributedRMSNorm

The `TTNNDistributedRMSNorm` class in `modules/normalization.py` demonstrates all three abstractions in action. This is the canonical example of a multi-device module:

**Weight distribution** (in `move_weights_to_device_impl`):

```python
self.weight_distributed = ttnn.as_tensor(
    self.torch_layer.weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // 32, 32]).to(torch.bfloat16),
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=(ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape))),
)
self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
```

The weight is reshaped to be shardable, then distributed across the mesh using `ShardTensor2dMesh` with explicit dimension mapping.

**Forward pass** (decorated with `@run_on_devices(DeviceArch.T3K)`):

```python
def forward(self, inp):
    # Phase 1: per-device partial statistics
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)

    # Phase 2: all-gather statistics across devices
    tt_stats = ttnn.all_gather(
        tt_stats, dim=-1, num_links=1, topology=ttnn.Topology.Linear,
    )

    # Phase 3: per-device normalization using gathered statistics
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats, epsilon=self.torch_layer.variance_epsilon,
        weight=self.weight_distributed,
    )
    tt_stats.deallocate(True)
    return tt_out
```

The pattern is: compute partial results per device, all-gather the reduction statistics, then finalize per device. The `@run_on_devices(DeviceArch.T3K)` decorator restricts this path to T3K (multi-device) hardware.

## TT-Lang's Grid Model: Potential and Limitations

TT-Lang's `@ttl.operation(grid=...)` decorator defines a grid of compute cores for a single device. The `_resolve_grid` function in `ttl_api.py` resolves the grid either as:
- `"auto"` --- queries `device.compute_with_storage_grid_size()` to use all available cores
- An explicit tuple `(cols, rows)` --- specifying exact core dimensions
- A callable --- evaluated at runtime with the kernel's arguments

Within the grid, each core is addressed by `ttl.node(dims=2)` returning `(col, row)`. The `ttl.grid_size(dims=2)` call returns the total grid dimensions. Work is partitioned across cores by the kernel author using these primitives.

### Current Limitation: Single-Device Scope

TT-Lang currently operates at the **single-device level**. Key evidence from the source:

1. **`_compile_ttnn_kernel`** builds a single `CoreRangeSet` from the grid dimensions and dispatches to a single device via `ttnn.generic_op`. There is no multi-device dispatch path.

2. **`_is_mesh_tensor`** detection exists in `ttl_api.py` and is used in `_make_cache_key` to include mesh shape in the cache key, ensuring that per-device shard shapes (which differ from logical shapes) produce separate compilations. But the kernel itself sees only the per-device shard.

3. **`_resolve_grid`** queries a single device's compute grid. It does not span multiple devices.

This means: when a TT-Lang kernel runs on a mesh tensor, it sees the **per-device shard dimensions** (not the full logical tensor), and it runs independently on each device in the mesh. The mesh runtime handles dispatching the kernel to each device with that device's shard.

### Future Potential

TT-Lang's grid model could conceptually extend to multi-device grids where `ttl.node()` addresses cores across devices. This would unify intra-device parallelism (core grid) with inter-device parallelism (mesh) under a single programming model. The `mesh_shape` is already tracked in the cache key, suggesting this extension is anticipated. However, this is not implemented today.

## Recommended Near-Term Approach

The correct strategy for deploying TT-Lang kernels in multi-device settings combines both systems at their respective strengths:

### Architecture: TT-Lang for Per-Device Compute, TT-Symbiote for Cross-Device Coordination

```
                          TT-Symbiote Layer
                    ┌───────────────────────────┐
                    │  DistributedConfig         │
                    │  ShardTensor2dMesh         │
                    │  ConcatMesh2dToTensor      │
                    │  TT_CCL / all_gather       │
                    └─────────┬─────────────────┘
                              │ per-device shards
                    ┌─────────▼─────────────────┐
                    │  TT-Lang Fused Kernel      │
                    │  (runs on each device      │
                    │   independently)           │
                    └───────────────────────────┘
```

### Concrete Pattern for a Distributed Module with TT-Lang

Follow the `TTNNDistributedRMSNorm` pattern, replacing the per-device TTNN ops with TT-Lang kernel calls:

```python
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import trace_enabled

@ttl.operation(grid="auto")
def fused_per_device_op(x_in: ttnn.Tensor, w_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    # ... TT-Lang kernel that operates on per-device shard ...
    pass

@trace_enabled
class TTNNDistributedFusedOp(TTNNModule):
    def move_weights_to_device_impl(self):
        # Distribute weights across mesh --- standard TT-Symbiote pattern
        self.weight_distributed = ttnn.as_tensor(
            self.torch_layer.weight.reshape(...).to(torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device, dims=(None, 2), mesh_shape=list(self.device.shape)
            ),
        )
        self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, inp):
        # Phase 1: TT-Lang fused kernel on per-device shard
        out = ttnn.from_torch(
            torch.zeros(inp.shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=inp.device()
        )
        fused_per_device_op(inp, self.weight_distributed, out)

        # Phase 2: Cross-device collective (stays as TTNN/CCL)
        gathered = ttnn.all_gather(
            out, dim=-1, num_links=1, topology=ttnn.Topology.Linear,
        )

        # Phase 3: Post-gather processing (could be another TT-Lang kernel)
        # ...
        return result
```

### Why This Is the Right Split

1. **TT-Lang excels at fusing sequential ops within a device.** The DFB model keeps intermediates in L1 across what would otherwise be separate DRAM-bouncing TTNN calls. This is the primary value proposition ([Chapter 7](../ch7_fusion_targets/index.md)).

2. **Cross-device communication has fundamentally different semantics.** `all_gather`, `reduce_scatter`, and other collectives involve network links (Ethernet on T3K), not the Tensix core grid. These are well-served by TTNN's CCL primitives and TT-Symbiote's `TT_CCL` wrapper.

3. **The boundary is clean.** TT-Lang kernels consume and produce `ttnn.Tensor` objects. The mesh runtime transparently dispatches per-device. No special multi-device code is needed inside the TT-Lang kernel.

4. **Cache keys already handle mesh tensors correctly.** `_make_cache_key` includes `mesh_key = tuple(arg.device().shape)` when mesh tensors are detected, ensuring that single-device compilations and multi-device shard compilations are cached separately.

### What Changes Per Device Count

| Aspect | Single Device | Multi-Device (T3K) |
|--------|--------------|-------------------|
| TT-Lang kernel code | Unchanged | Unchanged |
| Tensor shapes seen by kernel | Full logical shape | Per-device shard shape |
| `CompilerOptions` | Same | Same (per-device compilation is independent) |
| Weight distribution | `ttnn.to_device(w, device)` | `ttnn.as_tensor(..., mesh_mapper=ShardTensor2dMesh(...))` |
| Cross-device communication | None | `ttnn.all_gather`, `ttnn.reduce_scatter`, etc. |
| Module decorator | None needed | `@run_on_devices(DeviceArch.T3K)` |

### Limitations to Be Aware Of

1. **No cross-device fusion.** You cannot fuse a per-device compute with a cross-device collective into a single TT-Lang kernel. The collective must remain a separate TTNN call between TT-Lang kernel invocations.

2. **Output tensor allocation on mesh.** Pre-allocating the output tensor for a TT-Lang kernel on a mesh device requires knowing the per-device shard shape, not the logical shape. Use the shard dimensions from the input tensor (which already reflects the per-device view).

3. **Profiling on multi-device.** The profiling environment variables (`TTLANG_AUTO_PROFILE`, etc.) produce per-device profiler data. You get one profile per device. Correlating across devices requires Perfetto's timeline view with device IDs.

4. **`@trace_enabled` interaction.** When a module is decorated with `@trace_enabled`, the TT-Lang kernel's `ttnn.generic_op` dispatch is captured in the trace. This works correctly because the compiled kernel uses the standard TTNN dispatch path. However, the JIT compilation must complete before trace capture begins --- the first call (which triggers compilation) cannot be inside a trace region.

---

**End of guide.** Return to [Guide Index](../index.md)
