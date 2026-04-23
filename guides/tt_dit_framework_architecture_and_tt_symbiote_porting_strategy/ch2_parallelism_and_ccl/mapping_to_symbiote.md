# Mapping to TT-Symbiote

## Prerequisites

- [Chapter 2 Index](./index.md): understanding of TT-DiT's 3-axis parallelism.
- [CCLManager](./ccl_manager.md): understanding of semaphore management and persistent buffers.
- [Parallel Linear Layers](./parallel_linear_layers.md): understanding of `ColParallelLinear` and `RowParallelLinear`.
- [Chapter 1 -- Comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md): understanding of architectural differences between TT-DiT `Module` and TT-Symbiote `TTNNModule`.

---

## Overview

TT-Symbiote has its own distributed computing infrastructure, but it was designed primarily for LLM inference workloads. This section compares the two frameworks' approaches to parallelism and collective communication, identifies gaps, and recommends how to extend TT-Symbiote to support DiT-style workloads.

---

## DistributedConfig and DistributedTensorConfig

TT-Symbiote's distributed computing is configured through two dataclasses in `models/experimental/tt_symbiote/core/run_config.py`:

### DistributedTensorConfig

```python
# models/experimental/tt_symbiote/core/run_config.py

@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any        # e.g., ttnn.ShardTensor2dMesh
    mesh_composer: Any      # e.g., ttnn.ConcatMesh2dToTensor
    logical_shape_fn: Optional[Any] = None  # Computes logical shape from sharded shape

    def get_logical_shape(self, sharded_shape):
        if self.logical_shape_fn is not None:
            return self.logical_shape_fn(sharded_shape)
        return sharded_shape
```

This config describes how a tensor is distributed across a mesh and how to reassemble it. The `mesh_mapper` handles host-to-device distribution, and the `mesh_composer` handles device-to-host reassembly.

### DistributedConfig

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None

    def __post_init__(self):
        if self.tensor_config is None and self.mesh_device.get_num_devices() > 1:
            self.tensor_config = DistributedTensorConfig(
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, self.mesh_device.shape, (0, -1)),
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, self.mesh_device.shape, (0, -1)),
                logical_shape_fn=logical_shape_for_batch_channel_sharding(
                    self.mesh_device.shape),
            )
        if self.ccl_manager is None and self.mesh_device.get_num_devices() > 1:
            self.ccl_manager = TT_CCL(self.mesh_device)
```

When no explicit configuration is provided, `DistributedConfig.__post_init__` creates a **default batch-and-channel sharding** strategy:
- Dimension 0 (batch) is sharded along mesh axis 0.
- Dimension -1 (last dim / channels) is sharded along mesh axis 1.

It also creates a `TT_CCL` instance (from `models/tt_transformers/tt/ccl.py`) as the default CCL manager.

---

## Comparison: Parallelism Configuration

| Feature | TT-DiT | TT-Symbiote |
|---|---|---|
| **Config structure** | `DiTParallelConfig` with 3 `ParallelFactor` tuples | `DistributedConfig` + `DistributedTensorConfig` |
| **Parallelism axes** | 3 explicit axes (CFG, SP, TP) with per-axis `(factor, mesh_axis)` | Implicit batch+channel sharding via `ShardTensor2dMesh` dims |
| **Per-parameter sharding** | `Parameter(mesh_axes=[...])` specifies per-dimension sharding | `DistributedTensorConfig.mesh_mapper` applies uniformly to all tensors of a module |
| **Per-module override** | Each layer constructor receives `mesh_axis`, `fsdp_mesh_axis`, `ccl_manager` | Modules can override via `set_output_tensors_config_impl` and `get_tensor_config_for_tensor` |
| **Submesh support** | Explicit submesh creation via `mesh_device.create_submeshes()` | No built-in submesh abstraction |
| **Multiple configs per model** | `DiTParallelConfig` + `EncoderParallelConfig` + `VAEParallelConfig` | Single `DistributedConfig` per model |

The fundamental difference is that TT-DiT's parallelism is **declarative and composable** -- the `DiTParallelConfig` describes the intended parallelism at a high level, and individual layers receive the relevant `mesh_axis` to shard their parameters accordingly. TT-Symbiote's approach is **tensor-level and uniform** -- a `DistributedTensorConfig` describes how to distribute a generic tensor, and the framework applies it broadly.

### TT-DiT's Per-Parameter Sharding

TT-DiT's `Parameter` class accepts `mesh_axes` to specify exactly which tensor dimension maps to which mesh axis:

```python
# ColParallelLinear weight
Parameter(total_shape=[K, N], mesh_axes=[fsdp_axis, tp_axis], ...)
# -> Shard K along fsdp_axis, shard N along tp_axis

# RowParallelLinear weight
Parameter(total_shape=[K, N], mesh_axes=[tp_axis, fsdp_axis], ...)
# -> Shard K along tp_axis, shard N along fsdp_axis
```

This is more flexible than TT-Symbiote's approach, where the mapper applies the same distribution strategy to all tensors. TT-Symbiote's `get_tensor_config_for_tensor` method provides some per-tensor customization, but it is reactive (checking shapes at runtime) rather than declarative.

---

## Comparison: Distributed Linear Variants

TT-Symbiote provides several distributed linear classes in `models/experimental/tt_symbiote/modules/linear.py`:

### TTNNLinearIColShardedWRowSharded

```python
# models/experimental/tt_symbiote/modules/linear.py

class TTNNLinearIColShardedWRowSharded(TTNNLinearInputShardedWeightSharded):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, input_dim=-1, weight_dim=-2)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        ...
        tt_output = ttnn.linear(input_tensor, self.tt_weight, ...)
        tt_output = ttnn.reduce_scatter(
            tt_output, dim=3, num_links=1, cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring,
        )
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        ...
```

This corresponds to TT-DiT's `RowParallelLinear` -- the input is column-sharded (input dim -1 is split), the weight is row-sharded (weight dim -2 is split), and a reduce-scatter follows the matmul.

### TTNNLinearIColShardedWAllReduced

```python
class TTNNLinearIColShardedWAllReduced(TTNNLinearIColShardedWRowSharded):
    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        ...
        tt_output = ttnn.linear(input_tensor, self.tt_weight, ...)
        # Decompose all_reduce into reduce_scatter + all_gather
        tt_output = ttnn.reduce_scatter(tt_output, dim=3, ...)
        tt_output = ttnn.all_gather(tt_output, dim=3, ...)
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        ...
```

This is an all-reduce variant that produces replicated output (useful when the next layer expects replicated input). It decomposes `all_reduce` into `reduce_scatter + all_gather` for trace compatibility.

### TTNNLinearIReplicatedWColSharded

```python
class TTNNLinearIReplicatedWColSharded(TTNNLinearInputReplicatedWeightSharded):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, weight_dim=-1)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor):
        tt_output = ttnn.linear(input_tensor, self.tt_weight, ...)
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        ...
```

This corresponds to TT-DiT's `ColParallelLinear` -- the input is replicated, the weight is column-sharded (weight dim -1 is split), and the output is column-fractured. No CCL operation in the forward pass.

### Comparison Table

| Feature | TT-DiT `ColParallelLinear` | TT-Symbiote `TTNNLinearIReplicatedWColSharded` |
|---|---|---|
| Weight sharding | `Parameter(mesh_axes=[fsdp, tp])` | `shard_tensor_to_mesh_mapper(dim=-1)` |
| FSDP support | Yes, via `fsdp_mesh_axis` with all-gather | No |
| Matmul op | `ttnn.experimental.minimal_matmul` | `ttnn.linear` |
| Activation fusion | Built-in (`fused_activation`, `swiglu`) | Separate `TTNNLinearActivation` class |
| Chunked output | `minimal_matmul_split` | Not supported |
| Bias handling | Sharded via `mesh_axes` | Sharded via `shard_tensor_to_mesh_mapper` |
| Device restriction | None (works on any mesh) | `@run_on_devices(DeviceArch.T3K)` |
| Mesh axis | Configurable via `mesh_axis` param | Hardcoded to `cluster_axis=1` |

| Feature | TT-DiT `RowParallelLinear` | TT-Symbiote `TTNNLinearIColShardedWRowSharded` |
|---|---|---|
| Weight sharding | `Parameter(mesh_axes=[tp, fsdp])` | `shard_tensor_to_mesh_mapper(dim=-2)` |
| FSDP support | Yes | No |
| Reduce op | `ccl_manager.reduce_scatter` (async, semaphores) | `ttnn.reduce_scatter` (synchronous) |
| Bias zero-padding | Explicit in `_prepare_torch_state` | Bias sharded separately |
| Matmul op | `ttnn.experimental.minimal_matmul` | `ttnn.linear` |
| Topology | Configurable (usually Linear) | Hardcoded Ring |
| Mesh axis | Configurable via `mesh_axis` param | Hardcoded `cluster_axis=1` |

---

## Comparison: CCL Infrastructure

### TT-DiT CCLManager vs. TT-Symbiote TT_CCL

TT-Symbiote uses `TT_CCL` from `models/tt_transformers/tt/ccl.py` as its CCL manager:

```python
# models/tt_transformers/tt/ccl.py

class TT_CCL:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.sub_device_crs = ttnn.CoreRangeSet({...})

        # Double-buffered semaphores for 3 axis options (0, 1, no-axis)
        self.barrier_semaphore_handles = [[], [], []]
        self.ag_semaphore_handles = [[], [], []]
        self.rs_semaphore_handles = [[], [], []]

        for i in range(3):
            for _ in range(2):  # double-buffered
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(...))
                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(2)])
                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(3)])
```

| Feature | TT-DiT `CCLManager` | TT-Symbiote `TT_CCL` |
|---|---|---|
| **Semaphore types** | RS, AG, NP, SR, Barrier (5 types) | RS, AG, Barrier (3 types) |
| **Axis support** | 2 axes (0, 1) | 3 options (0, 1, no-axis) |
| **Ping-pong** | Yes, per semaphore type + axis | Yes, double-buffered per type + axis |
| **Persistent buffers** | Yes, cached by `(shape, dim, axis)` | No |
| **All-gather helper** | `all_gather(dim, mesh_axis, use_hyperparams, ...)` | `tt_all_gather()` -- uses `ttnn.experimental.all_gather_async` with semaphores |
| **Reduce-scatter helper** | `reduce_scatter(dim, mesh_axis, use_persistent_buffer, ...)` | `tt_all_reduce()` -- uses `ttnn.experimental.reduce_scatter_minimal_async` with semaphores |
| **Hyperparameter tuning** | `get_ag_hyperparams`, `get_rs_hyperparams` | Hardcoded defaults (`chunks_per_sync=10`, `num_workers_per_link=2`) in `tt_all_reduce`/`tt_all_gather` |
| **Rank normalization** | Auto-pads to rank 4 | Not handled |
| **device_to_host** | Integrated gather + to_torch | Not provided |
| **VAE ops** | `vae_all_gather`, `vae_neighbor_pad`, `vae_slice_reshard` | Not applicable |
| **Reset method** | `reset_global_semaphores()` | Not provided |

The key difference is that TT-DiT's `CCLManager` is a **high-level abstraction** that encapsulates all CCL concerns (buffer allocation, semaphore cycling, hyperparameter selection, rank normalization) behind simple method calls. TT-Symbiote's `TT_CCL` is a **semaphore store with helper functions** -- `tt_all_reduce()` and `tt_all_gather()` provide async CCL operations using `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` with semaphore cycling, but TT-DiT's `CCLManager` additionally handles persistent buffer caching, shape-based hyperparameter tuning, and rank normalization.

### How TT-Symbiote Distributed Linears Use CCL

Despite the async helpers available in `tt_transformers`, TT-Symbiote's distributed linear *modules* call `ttnn.reduce_scatter` and `ttnn.all_gather` directly rather than using `tt_all_reduce()`/`tt_all_gather()`:

```python
# TTNNLinearIColShardedWRowSharded.forward()
tt_output = ttnn.reduce_scatter(
    tt_output, dim=3, num_links=1, cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Ring)
```

This works but has several limitations:
1. **No persistent buffers** -- each call allocates and deallocates output memory.
2. **Bypasses async helpers** -- uses synchronous CCL ops directly instead of the available `tt_all_reduce()`/`tt_all_gather()` functions that provide semaphore-based async operations.
3. **Hardcoded parameters** -- `num_links=1`, `cluster_axis=1`, `topology=Ring` are fixed per class.
4. **No hyperparameter tuning** -- no shape-dependent configuration of chunks or workers (though `tt_all_reduce`/`tt_all_gather` use hardcoded defaults of `chunks_per_sync=10`, `num_workers_per_link=2`).

---

## Gaps in TT-Symbiote's CCL Infrastructure

Based on the analysis, the following gaps would need to be addressed to support DiT-style workloads in TT-Symbiote:

### 1. No CCLManager-Equivalent with Persistent Buffers

TT-Symbiote's `TT_CCL` provides semaphore management but does not handle persistent buffer caching. DiT workloads benefit significantly from persistent buffers because the same CCL operations (with the same shapes) repeat across denoising steps and transformer blocks.

**Recommendation**: Either port TT-DiT's `CCLManager` as a new utility in TT-Symbiote, or extend `TT_CCL` to add persistent buffer caching with the same ping-pong pattern.

### 2. Distributed Linear Modules Bypass Async CCL Helpers

The `tt_transformers` library already provides async CCL helper functions -- `tt_all_reduce()` and `tt_all_gather()` in `models/tt_transformers/tt/ccl.py` -- that use `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` with proper semaphore management. However, TT-Symbiote's distributed linear *modules* (e.g., `TTNNLinearIColShardedWRowSharded`) bypass these helpers and call synchronous `ttnn.reduce_scatter` and `ttnn.all_gather` directly. The infrastructure for async CCL exists; the gap is that the distributed linear layers do not use it.

**Recommendation**: Refactor TT-Symbiote's distributed linear modules to call the existing `tt_all_reduce()`/`tt_all_gather()` helpers (or equivalent wrappers) instead of invoking synchronous CCL ops directly. This would immediately gain async semaphore-based synchronization without needing a new `AsyncCCLManager` class.

### 3. No Multi-Axis Parallelism Config

TT-Symbiote's `DistributedConfig` defaults to batch+channel sharding and does not support the independent 3-axis parallelism that DiT requires. There is no concept of separate CFG-P, SP, and TP axes.

**Recommendation**: Introduce a `DiTDistributedConfig` (or extend `DistributedConfig`) that supports:
- Configurable axis assignment via `ParallelFactor`-style tuples.
- Submesh creation for CFG parallelism.
- Per-parameter sharding directives (like TT-DiT's `mesh_axes`).

### 4. No Submesh Management

TT-Symbiote has no built-in concept of submeshes for CFG parallelism. Each model uses a single `mesh_device`, and the `DistributedConfig` applies to the entire mesh.

**Recommendation**: Add submesh creation support to TT-Symbiote's device management utilities, with the ability to maintain independent `DistributedConfig` instances per submesh.

### 5. No FSDP Support in Distributed Linears

TT-Symbiote's distributed linear variants shard weights along one axis only. TT-DiT's `ColParallelLinear` and `RowParallelLinear` support 2-axis sharding (TP + FSDP), with just-in-time all-gather for the FSDP axis.

**Recommendation**: Add an `fsdp_mesh_axis` parameter to TT-Symbiote's distributed linear classes and implement the pre-matmul weight gathering pattern.

### 6. Hardcoded Mesh Axis and Topology

TT-Symbiote's distributed linears hardcode `cluster_axis=1` and `topology=ttnn.Topology.Ring`. TT-DiT makes these configurable per-layer.

**Recommendation**: Make `cluster_axis` and `topology` configurable parameters on all distributed linear classes, defaulting to the values in the `DistributedConfig`.

### 7. No Shape-Based Hyperparameter Tuning

The `tt_all_reduce()` and `tt_all_gather()` helpers use hardcoded defaults (`chunks_per_sync=10`, `num_workers_per_link=2`) rather than adjusting CCL parameters based on tensor shape. TT-DiT's `get_ag_hyperparams` and `get_rs_hyperparams` improve performance by tuning these values for different tensor sizes.

**Recommendation**: Add shape-based hyperparameter selection to the existing async CCL helpers or introduce a configurable policy on the CCL manager.

---

## Porting Path: Incremental Integration

Given the gaps identified above, the recommended porting path from TT-DiT to TT-Symbiote has three phases:

### Phase 1: CCL Infrastructure

1. Refactor TT-Symbiote's distributed linear modules to use the existing `tt_all_reduce()`/`tt_all_gather()` async helpers from `models/tt_transformers/tt/ccl.py` instead of calling synchronous CCL ops directly.
2. Add persistent buffer caching (either by extending `TT_CCL` or porting TT-DiT's `CCLManager` as a complementary utility).
3. Add `reset_global_semaphores()` support for trace replay.

### Phase 2: Parallelism Configuration

1. Introduce `ParallelFactor` and `DiTParallelConfig` types (or equivalents) in TT-Symbiote's config system.
2. Add submesh creation and per-submesh config support to `DistributedConfig`.
3. Extend `TTNNModule.set_device_state()` to accept submesh-specific configs.

### Phase 3: Distributed Layers

1. Add FSDP support to existing distributed linear classes.
2. Make `cluster_axis` and `topology` configurable.
3. Switch from `ttnn.linear` to `ttnn.experimental.minimal_matmul` for configurable blocking.
4. Add activation fusion support (SwiGLU, GELU) to distributed linears.

This incremental path allows each phase to be validated independently before proceeding to the next, reducing risk and enabling early benchmarking.

---

## Summary Table: Feature Parity

| Feature | TT-DiT Status | TT-Symbiote Status | Gap Severity |
|---|---|---|---|
| Tensor parallelism | Full (configurable axis) | Partial (hardcoded axis 1) | Medium |
| Sequence parallelism | Full | Not supported | High |
| CFG parallelism | Full (submesh-based) | Not supported | High (DiT-specific) |
| FSDP | Full | Not supported | Medium |
| Async CCL ops | Full (semaphore-based) | Available via `tt_all_reduce`/`tt_all_gather` helpers, but distributed linear modules use sync ops directly | Medium |
| Persistent CCL buffers | Full (ping-pong cached) | Not supported | High |
| Hyperparameter tuning | Shape-based | Hardcoded defaults in helpers; no shape-based selection | Low |
| Per-parameter sharding | Full (`mesh_axes`) | Partial (`mesh_mapper`) | Medium |
| VAE CCL ops | Full (neighbor_pad, slice_reshard) | Not applicable | Low (DiT-specific) |
| Matmul kernel | `minimal_matmul` | `ttnn.linear` | Medium |

---

## Key Takeaways

1. **TT-Symbiote's distributed infrastructure was designed for LLM workloads** and lacks the multi-axis parallelism and persistent buffer caching that DiT models require for competitive performance. Async CCL helpers (`tt_all_reduce`/`tt_all_gather`) exist in `tt_transformers` but are not used by TT-Symbiote's distributed linear modules.
2. **The most critical gaps are persistent buffers and wiring up async CCL in distributed linears** -- the async infrastructure exists in `tt_all_reduce()`/`tt_all_gather()`, but the distributed linear modules bypass it; persistent buffer caching is not yet available.
3. **TT-Symbiote already has the building blocks** (semaphore management via `TT_CCL`, distributed linear variants, device management) -- the extensions needed are evolutionary, not revolutionary.
4. **An incremental porting path** (CCL infrastructure first, then config, then layers) allows validation at each step and reuses TT-Symbiote's existing module lifecycle and tracing infrastructure.
5. **Per-parameter `mesh_axes`** is a cleaner API than `mesh_mapper` for fine-grained sharding control, and TT-Symbiote should consider adopting a similar mechanism.

---

**Next:** [Chapter 3 -- Custom Layers and TTNN Operations](../ch3_custom_layers_and_ops/index.md)
