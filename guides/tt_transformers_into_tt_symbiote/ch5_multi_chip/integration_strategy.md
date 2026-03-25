# Integration Strategy: TT Transformers Parallelism into TT Symbiote

This document compares the two systems side-by-side, identifies what can be reused, what must be adapted, and what is currently absent from Symbiote.

---

## 1. What Maps Directly

### 1.1 Sharded Linear Patterns vs Column/Row-Parallel

The conceptual patterns are aligned. The table below maps TT Transformers' terminology to Symbiote's class names:

| Role | TT Transformers approach | Symbiote class | Match quality |
|------|--------------------------|---------------|---------------|
| Column-parallel (QKV, w1, w3) | `ShardTensor2dMesh(..., dims=(...))` on weight; no CCL after matmul | `TTNNLinearIReplicatedWColSharded` | Direct — weight sharded on output dim, input replicated, no post-matmul CCL |
| Row-parallel (wo, w2) | `ShardTensor2dMesh(..., dims=(...))` on weight; `tt_all_reduce` after matmul | `TTNNLinearIColShardedWRowSharded` | Direct — weight sharded on input dim, reduce-scatter after matmul |
| Distributed RMSNorm | `tt_distributed_rmsnorm` / `tt_sharded_distributed_rmsnorm` in `ccl.py` | `TTNNDistributedRMSNorm` | Direct — both use pre-all-gather + all-gather + post-all-gather pattern |
| Distributed RoPE | `RotarySetup` + `ttnn.experimental.rotary_embedding_llama` | `TTNNDistributedRotaryPositionEmbedding` | Direct — both use `rotary_embedding_llama` |

### 1.2 TT_CCL Semaphore Management

Symbiote's `DistributedConfig.__post_init__` already creates a `TT_CCL` instance and stores it as `ccl_manager`. The distributed forward methods in Symbiote access it as `self.device_state.ccl_manager`. This is the same `TT_CCL` class from `ccl.py` that TT Transformers uses. The semaphore cycling APIs (`get_and_cycle_ag_semaphore_handles`, `get_and_cycle_rs_semaphore_handles`, `get_and_cycle_barrier_semaphore_handle`) are identical.

### 1.3 DistributedConfig / DistributedTensorConfig Dataclasses

`DistributedConfig` and `DistributedTensorConfig` in `run_config.py` provide a clean container for mesh mappers and composers. TT Transformers uses `args.cluster_shape` and passes `mesh_device` directly. The Symbiote dataclasses are an adequate abstraction layer; no structural change is needed.

---

## 2. What Must Be Adapted

### 2.1 CCL Call Site: TT_CCL Wrapper vs Raw ttnn.experimental Calls

The central difference is how CCL operations are invoked:

| Aspect | TT Transformers | TT Symbiote (current) |
|--------|----------------|----------------------|
| All-gather | `tt_all_gather(...)` wrapper in `ccl.py` | Direct `ttnn.experimental.all_gather_async(...)` in `normalization.py` and `linear.py` |
| Reduce-scatter | `tt_all_reduce(...)` wrapper (which calls `reduce_scatter_minimal_async`) | Direct `ttnn.experimental.reduce_scatter_minimal_async(...)` in `linear.py` |
| Single-device guard | `if mesh_shape == [1, 1]` in `tt_all_gather` / `tt_all_reduce` | Not present in Symbiote modules |
| Topology selection | Passed as `topology=self.args.ccl_topology()` | Hard-coded `ttnn.Topology.Ring` in `TTNNLinearIColShardedWRowSharded` |
| Link count | Auto-detected via `tt_ccl.get_num_links(cluster_axis)` | Hard-coded `num_links=1` |
| dtype coercion | Handled inside `tt_all_gather` / `tt_all_reduce` | Caller responsibility |

**Recommendation:** Introduce a thin helper in Symbiote that wraps `ttnn.experimental.all_gather_async` and `reduce_scatter_minimal_async` with the same single-device guard and link-count auto-detection that TT Transformers' `tt_all_gather` / `tt_all_reduce` provide. The existing `TT_CCL` instance in `DistributedConfig.ccl_manager` can be used directly. The wrappers need not be identical to TT Transformers — but the single-device guard is especially important: without it, distributed modules will fail on single-device runs.

### 2.2 Topology Rigidity

Symbiote's `TTNNLinearIColShardedWRowSharded` hard-codes `topology=ttnn.Topology.Ring`. TT Transformers passes topology as a parameter (`self.args.ccl_topology()`). For T3K the linear topology may be more appropriate; for Ring-connected fabrics, Ring is correct. The topology should become a parameter of the Symbiote module (or read from `CCLManagerConfig.topology`) rather than a compile-time constant.

### 2.3 Mesh Device Shape vs cluster_shape

| System | How mesh shape is specified |
|--------|-----------------------------|
| TT Transformers | `args.cluster_shape` — a `(rows, cols)` tuple passed at model construction time |
| TT Symbiote | `MESH_DEVICE` environment variable, looked up in `MeshShapeToDeviceArch` at `@run_on_devices` check time; physical shape from `self.device.shape` |

For `ShardTensor2dMesh`, TT Transformers passes `mesh_shape=args.cluster_shape` explicitly. Symbiote's `TTNNDistributedRMSNorm.move_weights_to_device_impl` passes `mesh_shape=list(self.device.shape)`. This is equivalent when `self.device` is the correct mesh device, but TT Transformers' approach is more explicit and does not depend on `self.device` being set before weight loading. The Symbiote approach works as long as `move_weights_to_device_impl` is always called after device assignment — which the `TTNNModule` lifecycle guarantees.

### 2.4 @run_on_devices Scope

All distributed Symbiote modules currently accept only `DeviceArch.T3K`. To support N300 or TG, the decorator arguments must be updated and separate code paths (possibly separate subclasses) must be written because:

- N300 has the same code path as T3K (1xN mesh, `reduce_scatter_minimal_async` path) but different physical link counts.
- TG requires the 2-D `all_gather + fast_reduce_nc` pattern that is not present in any current Symbiote module.

---

## 3. What Is Missing

### 3.1 Prefetcher Integration

TT Transformers has a `Prefetcher` class (`models/tt_transformers/tt/prefetcher.py`). It performs DRAM weight prefetching for decode mode by streaming weights from DRAM to L1 ahead of the matmul. The verified model list in `prefetcher.py` (lines 27-38) covers models including Llama 3.1 8B through 70B, Qwen3, and Gemma3.

Integration with CCL is non-trivial: when a prefetcher is active, matmuls pass `global_cb=self.prefetcher.global_cb` and `sub_device_id=self.prefetcher.worker_sub_device_id`. The CCL calls must also pass `subdevice_id=self.prefetcher.worker_sub_device_id` (see `mlp.py` lines 305-310 and `attention.py` lines 524). Without this, the prefetcher's sub-device isolation is broken.

**TT Symbiote currently has no Prefetcher class or equivalent mechanism.** The `TTNNLinearIColShardedWRowSharded.forward` and related methods do not accept or pass a sub-device ID. Adding prefetcher support requires:

1. A port of `Prefetcher` or a compatible wrapper.
2. Threading `subdevice_id` through all distributed module forward methods.
3. Registering weight tensors with the prefetcher via a callback mechanism analogous to `self.prefetcher.register_callback(register_weights)`.

### 3.2 TG / Galaxy Distributed Path

Symbiote has no modules that implement the TG-specific collective patterns:
- No `all_gather_async + fast_reduce_nc` all-reduce path.
- No intermediate reduce-scatter / all-gather around the MLP element-wise multiply.
- No `slice_mat` post-QKV batch-selection for 4x8 device groups.
- No `use_composite` reduce-scatter + all-gather path.

All `@run_on_devices` decorators in distributed Symbiote modules specify only `DeviceArch.T3K`.

### 3.3 Fused All-Gather-Matmul

TT Transformers uses `ttnn.experimental.all_gather_matmul_async` for the attention output projection on Ring topology without a prefetcher (`attention.py` lines 667-684). This fused operation does not appear anywhere in TT Symbiote. For initial integration on T3K it is not strictly required (the unfused path works), but it represents a gap if maximum decode throughput is needed.

### 3.4 DRAM-Sharded Memory Config

TT Transformers creates DRAM-sharded memory configs via `args.create_dram_sharded_mem_config(in_dim, out_dim)` for `wqkv` and `wo` on T3K. These configs are used to enable the DRAM-sharded matmul kernel paths. Symbiote's `TTNNLinearIColShardedWRowSharded` uses `ttnn.DRAM_MEMORY_CONFIG` (interleaved) rather than sharded DRAM. The impact is kernel selection and L1 usage, not correctness.

---

## 4. Summary Table

| Item | Status | Action required |
|------|--------|-----------------|
| Column-parallel linear (no post-CCL) | Present as `TTNNLinearIReplicatedWColSharded` | None for T3K; extend `@run_on_devices` for other targets |
| Row-parallel linear (reduce-scatter) | Present as `TTNNLinearIColShardedWRowSharded` | Parameterise topology; add single-device guard |
| Distributed RMSNorm | Present as `TTNNDistributedRMSNorm` | Add single-device guard; parameterise topology |
| Distributed RoPE | Present as `TTNNDistributedRotaryPositionEmbedding` | None |
| TT_CCL semaphore management | Present (via `DistributedConfig.ccl_manager`) | None |
| CCL wrapper functions (guards, link auto-detect) | Absent; raw calls used | Write thin helpers or adopt `tt_all_gather` / `tt_all_reduce` |
| N300 (2-chip) distributed path | `@run_on_devices` blocks it | Add `DeviceArch.N300` after verifying single-device guard |
| TG / Galaxy distributed path | Absent | Full new implementation required |
| Prefetcher integration | Absent | Port `Prefetcher` class; thread sub-device IDs |
| Fused all-gather-matmul | Absent | Build if decode throughput demands it |
| DRAM-sharded matmul configs | Absent | Add to linear modules if higher decode throughput is needed |

---

## Navigation

| | |
|---|---|
| Previous | [symbiote_distributed_primitives.md](symbiote_distributed_primitives.md) |
| Chapter home | [index.md](index.md) |
| Series plan | [../plan.md](../plan.md) |
