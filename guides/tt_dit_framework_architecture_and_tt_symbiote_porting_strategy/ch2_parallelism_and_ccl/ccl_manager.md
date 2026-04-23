# CCLManager: Collective Communication Infrastructure

## Prerequisites

- [Chapter 2 Index](./index.md): understanding of the 3-axis parallelism model and how submeshes are created.
- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of TT-DiT's `Module` and `Parameter` classes.

---

## Overview

The `CCLManager` class (`models/tt_dit/parallel/manager.py`) is TT-DiT's central coordinator for all collective communication operations within a submesh. It manages:

1. **SubDevice setup** -- allocating a CoreRangeSet spanning all compute cores for CCL operations.
2. **Semaphore initialization** -- creating ping-pong global semaphores for reduce-scatter, all-gather, neighbor-pad, slice-reshard, and barrier operations.
3. **Persistent buffer caching** -- allocating and reusing ping-pong DRAM buffers for CCL operations that benefit from pre-allocated memory.
4. **Helper methods** -- `all_gather`, `reduce_scatter`, `all_gather_persistent_buffer`, and `device_to_host` that wrap `ttnn.experimental` async CCL ops with correct semaphore and buffer management.
5. **Hyperparameter tuning** -- shape-dependent CCL configuration (chunks_per_sync, num_workers_per_link, num_buffers_per_channel).

Each `CCLManager` instance is bound to a single `ttnn.MeshDevice` (typically a submesh), and stores a `num_links` count and `topology` (usually `ttnn.Topology.Linear`).

---

## SubDevice Setup

The constructor calls `_init_subdevice()` to establish a core range spanning the entire compute grid:

```python
# models/tt_dit/parallel/manager.py

def _init_subdevice(self):
    compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
    self.ccl_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1)
        )}
    )

    _worker_sub_device = ttnn.SubDevice([self.ccl_cores])
    self.ccl_sub_device_id = ttnn.SubDeviceId(0)
```

This `CoreRangeSet` is used for two purposes:
1. **Semaphore scope** -- all global semaphores are created on this core range, ensuring every compute core in the mesh can participate in synchronization.
2. **SubDevice identification** -- the `ccl_sub_device_id` (always 0) is passed to ring joint SDPA and other operations that need to know which SubDevice partition to use.

The compute grid on a Wormhole chip is typically 8x8 (64 cores), so `ccl_cores` spans all of them.

---

## Semaphore Initialization

The `_init_semaphores()` method creates **five categories** of global semaphores, each with **ping-pong pairs** and **per-mesh-axis separation**:

```python
# models/tt_dit/parallel/manager.py

def _init_semaphores(self):
    # Reduce scatter: 3 semaphores * 2 ping-pong = 6 per axis
    rs_n_sems = 3 * 2
    self.rs_ping_pong_semaphores = {
        0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0)
            for _ in range(rs_n_sems)],
        1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0)
            for _ in range(rs_n_sems)],
    }

    # All gather: 2 semaphores * 2 ping-pong = 4 per axis
    ag_n_sems = 2 * 2
    self.ag_ping_pong_semaphores = { ... }

    # Neighbor pad: 1 semaphore * 2 ping-pong = 2 per axis
    np_n_sems = 1 * 2
    self.np_ping_pong_semaphores = { ... }

    # Slice reshard: 1 semaphore * 2 ping-pong = 2 per axis
    sr_n_sems = 1 * 2
    self.sr_ping_pong_semaphores = { ... }

    # Barrier: 1 semaphore * 2 ping-pong = 2 per axis
    barrier_n_sems = 1 * 2
    self.barrier_semaphores = { ... }
```

### Why Ping-Pong?

Each semaphore category maintains **two sets** (indexed 0 and 1), and the CCLManager alternates between them on consecutive CCL calls. This is essential for correctness when running traced workloads:

- In a trace-captured sequence, the same CCL operation appears multiple times.
- If all invocations used the same semaphore, the second invocation's signals could be confused with the first's.
- By alternating between set 0 and set 1, consecutive operations use independent semaphores, preventing race conditions.

The ping-pong index for each category is tracked per mesh axis:

```python
self.rs_ping_pong_idx = [0, 0]  # [axis_0_idx, axis_1_idx]
self.ag_ping_pong_idx = [0, 0]
self.np_ping_pong_idx = [0, 0]
self.sr_ping_pong_idx = [0, 0]
self.barrier_idx = [0, 0]
```

### Per-Axis Separation

Semaphores are stored in a dict keyed by mesh axis (0 or 1). This allows operations along axis 0 (e.g., sequence parallel all-gather) and axis 1 (e.g., tensor parallel reduce-scatter) to use completely independent semaphore pools, avoiding cross-axis interference.

### Semaphore Accessor Pattern

Each getter method follows the same pattern -- return the current semaphore(s), then toggle the ping-pong index:

```python
def get_rs_ping_pong_semaphore(self, mesh_axis):
    cur_idx = self.rs_ping_pong_idx[mesh_axis]
    n_sems = 3  # reduce-scatter uses 3 semaphores per operation
    self.rs_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
    return self.rs_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

def get_ag_ping_pong_semaphore(self, mesh_axis):
    cur_idx = self.ag_ping_pong_idx[mesh_axis]
    n_sems = 2  # all-gather uses 2 semaphores per operation
    self.ag_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
    return self.ag_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]
```

The semaphore counts per operation type reflect the internal requirements of the async CCL kernels:
- **Reduce-scatter**: 3 semaphores (for the multi-phase scatter-reduce protocol).
- **All-gather**: 2 semaphores (for the double-buffered ring protocol).
- **Neighbor-pad, slice-reshard, barrier**: 1 semaphore each.

### Reset

The `reset_global_semaphores()` method resets all semaphore values to 0 across both axes. This is typically called between traced iterations or when reinitializing the pipeline:

```python
def reset_global_semaphores(self):
    for axis in [0, 1]:
        for sem in self.np_ping_pong_semaphores[axis]:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.sr_ping_pong_semaphores[axis]:
            ttnn.reset_global_semaphore_value(sem, 0)
        # ... same for rs, ag semaphores
```

---

## Persistent Buffer Caching

CCL operations can reuse pre-allocated output buffers rather than allocating new memory on each call. The CCLManager maintains a cache of these **ping-pong buffers** keyed by `(shape, dim, mesh_axis)`:

### Reduce-Scatter Buffers

```python
def get_rs_ping_pong_buffer(self, shape, dim, mesh_axis):
    cache_key = (tuple(shape), dim, mesh_axis)

    if cache_key not in self._ping_pong_buffer_cache:
        ttnn.synchronize_device(self.mesh_device)

        buffers = []
        # Output buffer shape: input shape with dim divided by device count
        output_buffer_shape = list(shape)
        output_buffer_shape[dim] //= self.mesh_device.shape[mesh_axis]

        # Intermediate buffer shape: [2] + input shape (double-buffered)
        intermediate_buffer_shape = [2] + list(shape)

        for _ in range(2):  # Two sets for ping-pong
            intermediate_buffer = bf16_tensor(
                torch.empty(intermediate_buffer_shape), device=self.mesh_device)
            output_buffer = bf16_tensor(
                torch.empty(output_buffer_shape), device=self.mesh_device)
            buffers.append([intermediate_buffer, output_buffer])

        self._ping_pong_buffer_cache[cache_key] = buffers
        self._ping_pong_buffer_indices[cache_key] = 0
        ttnn.synchronize_device(self.mesh_device)

    # Alternate between buffer sets
    current_idx = self._ping_pong_buffer_indices[cache_key]
    self._ping_pong_buffer_indices[cache_key] = 1 - current_idx
    return self._ping_pong_buffer_cache[cache_key][current_idx]
```

Key details:
- Each reduce-scatter buffer set contains **two tensors**: an intermediate buffer (with a leading dimension of 2 for double-buffering within the kernel) and an output buffer (with the scatter-reduced shape).
- Two such sets exist (ping-pong), and the manager alternates between them.
- Buffers are allocated lazily on first access and synchronized before and after allocation to ensure all devices are ready.
- The `bf16_tensor` helper (from `utils/tensor.py`) creates a replicated bfloat16 tensor on the mesh device.

### All-Gather Buffers

```python
def get_ag_ping_pong_buffer(self, shape, dim, mesh_axis, dtype=ttnn.bfloat16):
    cache_key = ("ag", tuple(shape), dim, mesh_axis, dtype)

    if cache_key not in self._ping_pong_buffer_cache:
        ttnn.synchronize_device(self.mesh_device)

        buffers = []
        output_buffer_shape = list(shape)
        # All-gather increases the dimension by the device count
        output_buffer_shape[dim] *= self.mesh_device.shape[mesh_axis]

        for _ in range(2):
            output_buffer = ttnn.from_torch(
                torch.empty(output_buffer_shape),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.mesh_device,
            )
            buffers.append(output_buffer)

        self._ping_pong_buffer_cache[cache_key] = buffers
        ...
```

All-gather buffers are simpler -- just a single output tensor per set, sized to hold the gathered result (input dim multiplied by device count). Note the `"ag"` prefix in the cache key to avoid collisions with reduce-scatter entries.

---

## Helper Methods

### all_gather

The `all_gather` method wraps `ttnn.experimental.all_gather_async` with automatic rank handling, semaphore selection, and optional persistent buffers:

```python
def all_gather(self, tensor, /, *, dim, mesh_axis, use_hyperparams,
               use_persistent_buffer=False):
    # Skip if single device on this axis
    if mesh_axis is None or self.mesh_device.shape[mesh_axis] == 1:
        return tensor

    rank = len(tensor.shape)
    if dim < 0:
        dim += rank

    # all_gather_async requires rank-4 tensors; pad if needed
    if rank < 4:
        shape = [1] * (4 - rank) + list(tensor.shape)
        tensor = ttnn.reshape(tensor, shape)
        dim += 4 - rank

    params = self.get_ag_hyperparams(tensor.shape) if use_hyperparams else {}

    tensor = ttnn.experimental.all_gather_async(
        tensor,
        persistent_output_buffer=(
            self.get_ag_ping_pong_buffer(tensor.shape, dim, mesh_axis, ...)
            if use_persistent_buffer else None
        ),
        barrier_semaphore=(
            self.get_barrier_semaphore(mesh_axis)
            if not use_persistent_buffer else None
        ),
        dim=dim,
        multi_device_global_semaphore=self.get_ag_ping_pong_semaphore(mesh_axis),
        num_links=self.num_links,
        topology=self.topology,
        cluster_axis=mesh_axis,
        **params,
    )

    # Remove padding dimensions
    if rank < 4:
        shape = list(tensor.shape)[4 - rank:]
        tensor = ttnn.reshape(tensor, shape)

    return tensor
```

Important behaviors:
- **Short-circuit**: if the mesh has only 1 device on the target axis, the tensor is returned unchanged.
- **Rank padding**: `all_gather_async` requires rank-4 tensors, so lower-rank tensors get leading 1-dimensions prepended, then stripped after the operation.
- **Persistent buffer vs. barrier**: when using persistent buffers, the ping-pong buffer cache ensures correctness across calls. When not using persistent buffers, a barrier semaphore is used instead to synchronize devices before reusing memory.

### all_gather_persistent_buffer

A convenience wrapper that calls `all_gather` with `use_persistent_buffer=True`:

```python
def all_gather_persistent_buffer(self, tensor, /, *, dim, mesh_axis,
                                   use_hyperparams=False):
    return self.all_gather(tensor, dim=dim, mesh_axis=mesh_axis,
                           use_hyperparams=use_hyperparams,
                           use_persistent_buffer=True)
```

This is the most commonly called variant in the codebase -- used in attention blocks for TP all-gather after QKV projections, and in FSDP weight gathering.

### reduce_scatter

```python
def reduce_scatter(self, tensor, /, *, dim, mesh_axis,
                   use_persistent_buffer=False):
    if mesh_axis is None or self.mesh_device.shape[mesh_axis] == 1:
        return tensor

    # Rank padding to 4D (same as all_gather)
    ...

    tensor = ttnn.experimental.reduce_scatter_minimal_async(
        tensor,
        persistent_output_buffers=(
            self.get_rs_ping_pong_buffer(tensor.shape, dim, mesh_axis)
            if use_persistent_buffer else None
        ),
        barrier_semaphore=(
            self.get_barrier_semaphore(mesh_axis)
            if not use_persistent_buffer else None
        ),
        dim=dim,
        multi_device_global_semaphore=self.get_rs_ping_pong_semaphore(mesh_axis),
        num_links=self.num_links,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
        topology=self.topology,
        cluster_axis=mesh_axis,
        **self.get_rs_hyperparams(tensor.shape),
    )

    # Remove rank padding
    ...
    return tensor
```

The reduce-scatter uses `ttnn.experimental.reduce_scatter_minimal_async`, which is the optimized async variant that supports persistent buffers and semaphores. It always applies hyperparameters (unlike `all_gather` which makes them optional).

### device_to_host

This method gathers a distributed tensor to a single torch tensor for host-side processing (e.g., returning results):

```python
def device_to_host(self, tensor, mesh_dims, use_persistent_buffer=True):
    device_tensor = ttnn.to_layout(tensor, ttnn.TILE_LAYOUT)
    for mesh_axis, mesh_dim in enumerate(mesh_dims):
        if mesh_dim is not None:
            device_tensor = self.all_gather(
                device_tensor, dim=mesh_dim, mesh_axis=mesh_axis,
                use_hyperparams=True, use_persistent_buffer=use_persistent_buffer)
    return ttnn.to_torch(ttnn.get_device_tensors(device_tensor)[0])
```

The `mesh_dims` parameter is a list with one entry per mesh axis, specifying which tensor dimension to gather along for that axis (or `None` to skip). For example, `[None, 2]` gathers along dimension 2 on mesh axis 1 only. After gathering, the first device tensor is extracted and converted to PyTorch.

---

## Hyperparameter Tuning

The CCLManager provides shape-dependent hyperparameters for all-gather and reduce-scatter operations:

```python
def get_ag_hyperparams(self, shape):
    if shape[2] > 512:
        return {
            "chunks_per_sync": 16,
            "num_workers_per_link": 3,
            "num_buffers_per_channel": 2,
        }
    else:
        return {
            "chunks_per_sync": 10,
            "num_workers_per_link": 2,
            "num_buffers_per_channel": 2,
        }

def get_rs_hyperparams(self, shape):
    return {
        "chunks_per_sync": 2,
        "num_workers_per_link": 2,
        "num_buffers_per_channel": 2,
    }
```

These parameters control the async CCL kernels:
- **`chunks_per_sync`**: how many data chunks are transferred before a synchronization point. Larger values amortize sync overhead but increase latency for small tensors. The all-gather uses 16 chunks for large sequences (>512 tokens) and 10 for smaller ones.
- **`num_workers_per_link`**: number of Ethernet worker cores per link. More workers increase bandwidth utilization but consume more cores. The all-gather scales from 2 to 3 workers for large tensors.
- **`num_buffers_per_channel`**: double-buffering level within each worker channel. Always 2 in the current implementation.

Reduce-scatter uses a fixed configuration (2 chunks_per_sync) because it transfers smaller amounts of data per device.

---

## VAE-Specific CCL Operations

The `parallel/config.py` module also defines three CCL helper functions specifically for VAE operations. These are implemented as standalone functions rather than CCLManager methods, though they take a `ccl_manager` parameter for semaphore access:

### vae_all_gather

```python
# models/tt_dit/parallel/config.py

def vae_all_gather(ccl_manager, x, cluster_axis=1, dim=3, reshape=True):
    if x.device().shape[cluster_axis] == 1:
        return x

    global_semaphores = ccl_manager.get_ag_ping_pong_semaphore(cluster_axis)
    barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)

    if reshape:
        b, h, w, c = x.shape
        if h != 1:
            x = x.reshape(b, 1, h * w, c)  # Flatten H*W for tile layout

    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # NOTE: sync before all-gather due to barrier_semaphore hang workaround
    ttnn.synchronize_device(x.device())
    x_g = ttnn.experimental.all_gather_async(
        input_tensor=x,
        dim=dim,
        persistent_output_buffer=None,
        multi_device_global_semaphore=global_semaphores,
        topology=ttnn.Topology.Linear,
        cluster_axis=cluster_axis,
        num_links=ccl_manager.num_links,
        num_workers_per_link=4,
        chunks_per_sync=80,
        num_buffers_per_channel=4,
    )

    if reshape and h != 1:
        x_g = x_g.reshape(b, h, w, -1)
    return x_g
```

Key differences from the general `all_gather`:
- **No persistent output buffers** -- the comment explains that VAE runs out of memory with ping-pong buffers.
- **Explicit `synchronize_device`** instead of barrier semaphore -- this is a workaround for a known hang issue when barrier semaphores are used in the integrated pipeline.
- **Higher throughput parameters**: `num_workers_per_link=4`, `chunks_per_sync=80`, `num_buffers_per_channel=4` -- VAE tensors are typically larger (image-resolution feature maps), so more aggressive parallelism is used.
- **Reshape for tile layout**: VAE tensors are often in `[B, H, W, C]` format, which is flattened to `[B, 1, H*W, C]` before the all-gather since tile layout works on the last two dimensions.

### vae_neighbor_pad

```python
def vae_neighbor_pad(ccl_manager, x, cluster_axis=1, dim=0,
                     padding_left=0, padding_right=0, padding_mode="replicate",
                     secondary_cluster_axis=None, secondary_mesh_shape=None):
    global_semaphore = ccl_manager.get_np_ping_pong_semaphore(cluster_axis)
    barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)

    x_pad = ttnn.experimental.neighbor_pad_async(
        x, dim=dim,
        padding_left=padding_left, padding_right=padding_right,
        padding_mode=padding_mode,
        cluster_axis=cluster_axis,
        final_semaphore=global_semaphore,
        barrier_semaphore=barrier_semaphore,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Linear,
        secondary_cluster_axis=secondary_cluster_axis,
        secondary_mesh_shape=secondary_mesh_shape,
    )
    return x_pad
```

Neighbor padding is used in VAE convolution layers where padding at tensor boundaries requires data from adjacent devices. The `secondary_cluster_axis` and `secondary_mesh_shape` parameters support multi-axis VAE parallelism (e.g., `MochiVAEParallelConfig` with time, height, and width axes).

### vae_slice_reshard

```python
def vae_slice_reshard(ccl_manager, x, cluster_axis=1, dim=0,
                      output_shape=88, output_offset=0):
    global_semaphore = ccl_manager.get_sr_ping_pong_semaphore(cluster_axis)
    barrier_semaphore = ccl_manager.get_barrier_semaphore(cluster_axis)

    x_sr = ttnn.experimental.slice_reshard_async(
        x, dim=dim,
        output_dim_shape=output_shape, output_dim_offset=output_offset,
        cluster_axis=cluster_axis,
        final_semaphore=global_semaphore,
        barrier_semaphore=barrier_semaphore,
        num_links=ccl_manager.num_links,
        topology=ttnn.Topology.Linear,
    )
    return x_sr
```

Slice-reshard redistributes tensor slices across devices when the VAE changes its parallelization pattern (e.g., going from time-parallel to spatial-parallel in Mochi).

---

## Summary of TTNN Experimental Ops Used

| Operation | TTNN API | Purpose |
|---|---|---|
| All-gather | `ttnn.experimental.all_gather_async` | Gather sharded tensor slices across devices |
| Reduce-scatter | `ttnn.experimental.reduce_scatter_minimal_async` | Sum partial results and scatter |
| Neighbor pad | `ttnn.experimental.neighbor_pad_async` | Pad tensor boundaries with neighbor data |
| Slice reshard | `ttnn.experimental.slice_reshard_async` | Redistribute tensor slices across devices |
| Semaphore create | `ttnn.create_global_semaphore` | Create a global semaphore on a core range |
| Semaphore reset | `ttnn.reset_global_semaphore_value` | Reset semaphore to initial value |

All of these are async operations that use global semaphores for synchronization rather than blocking device syncs. This enables overlap of computation and communication when used within traced execution.

---

## Key Takeaways

1. **CCLManager centralizes all CCL state** -- semaphores, persistent buffers, and hyperparameters are managed per-submesh rather than scattered across individual layers.
2. **Ping-pong indexing prevents semaphore reuse conflicts** -- every CCL operation alternates between two semaphore sets, which is critical for correctness under trace replay.
3. **Persistent buffers trade memory for performance** -- pre-allocated output buffers avoid allocation overhead on each CCL call, but some components (VAE) cannot afford the memory cost.
4. **VAE CCL operations have distinct requirements** -- larger tensors, no persistent buffers, synchronization workarounds, and specialized operations (neighbor pad, slice reshard) that the DiT transformer does not need.
5. **The async CCL ops (`ttnn.experimental.*_async`) are the performance-critical path** -- they enable overlapped computation and communication via semaphore-based synchronization.

---

**Next:** [`parallel_linear_layers.md`](./parallel_linear_layers.md)
