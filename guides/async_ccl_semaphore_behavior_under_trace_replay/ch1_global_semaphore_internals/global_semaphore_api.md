# GlobalSemaphore API

This file describes the three public APIs that govern `GlobalSemaphore` objects — `ttnn.create_global_semaphore`, `GlobalSemaphore::address()`, and `ttnn.reset_global_semaphore_value` — and then shows exactly how `TT_CCL.__init__` uses those APIs to allocate the full set of double-buffered handles for all cluster-axis variants.

---

## `ttnn.create_global_semaphore`

### Python signature

```python
ttnn.create_global_semaphore(
    mesh_device,      # MeshDevice
    cores,            # CoreRangeSet
    initial_value,    # int (uint32)
    buffer_type=ttnn.BufferType.L1,
)
# returns: GlobalSemaphore object
```

The Python binding is registered via nanobind in `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp` and delegates directly to `CreateGlobalSemaphore` in `tt_metal/tt_metal.cpp`.

`CreateGlobalSemaphore` constructs a `GlobalSemaphore` object, which in its constructor calls `setup_buffer`. That method allocates a HEIGHT_SHARDED `MeshBuffer` in L1 — one 4-byte page per core in the `CoreRangeSet` — and then immediately writes `initial_value` to every core via `reset_semaphore_value`:

```cpp
// tt_metal/impl/buffers/global_semaphore.cpp, setup_buffer()
uint32_t num_cores = cores_.num_cores();
auto shard_parameters = ShardSpecBuffer(cores_, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_cores, 1});
ShardedBufferConfig sem_shard_config = {
    .device = device_,
    .size = num_cores * sizeof(uint32_t),
    .page_size = sizeof(uint32_t),
    .buffer_type = buffer_type,
    .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
    .shard_parameters = std::move(shard_parameters),
};
buffer_ = distributed::AnyBuffer::create(sem_shard_config, address);

if (initial_value.has_value()) {
    this->reset_semaphore_value(initial_value.value());
}
```

The key structural point: `GlobalSemaphore` is a thin wrapper around a sharded buffer. There is one 4-byte slot per core in the `CoreRangeSet`, each slot living at the same L1 address across all cores in the set. The buffer layout (`HEIGHT_SHARDED`, one page per core) is what makes the address uniform across cores.

### What the return value is

`ttnn.create_global_semaphore` returns a Python-wrapped `GlobalSemaphore` C++ object. From the Python side it behaves as an opaque handle — you cannot read back the semaphore value directly from Python; you can only pass the handle to other TTNN APIs (`ttnn.experimental.all_gather_async`, `ttnn.experimental.reduce_scatter_minimal_async`, `ttnn.reset_global_semaphore_value`). Internally, the object stores:

- `buffer_`: the `AnyBuffer` wrapping the sharded `MeshBuffer`
- `device_`: the owning device pointer
- `cores_`: the `CoreRangeSet` over which the semaphore is allocated

---

## `GlobalSemaphore::address()`

### C++ signature

```cpp
// tt_metal/api/tt-metalium/global_semaphore.hpp
DeviceAddr address() const;

// implementation in tt_metal/impl/buffers/global_semaphore.cpp
DeviceAddr GlobalSemaphore::address() const {
    return buffer_.get_buffer()->address();
}
```

`DeviceAddr` is `uint64_t`. The value returned is the L1 byte offset at which each participating core stores its semaphore word. Because the underlying buffer is sharded with a uniform page layout, every core in the `CoreRangeSet` exposes the semaphore at this same L1 address.

### Stability guarantee

The address is determined at allocation time (inside `distributed::AnyBuffer::create`) and does not change for the lifetime of the object. Nothing in `GlobalSemaphore` reallocates the buffer after construction. This stability is the property that makes it safe to bake the address into per-core runtime arguments: once a kernel is given `semaphore.address()` as an RTA, that value remains valid until the `GlobalSemaphore` object is destroyed.

> **Note:** On a `MeshDevice`, the same L1 address is used on all physical devices in the mesh. The `MeshBuffer` allocation path (`CreateGlobalSemaphore` for `MeshDevice*`) allocates the sharded buffer across all devices in the mesh, producing the same per-core L1 address on each. If you need to guarantee that two separately-created semaphores land at the same address (for multi-device synchronization), see `create_global_semaphore_with_same_address` in `ttnn/core/global_semaphore.cpp`. `TT_CCL` does not use that variant; it relies on the async CCL kernels receiving the actual address via RTAs.

---

## `ttnn.reset_global_semaphore_value`

### Python signature

```python
ttnn.reset_global_semaphore_value(
    global_semaphore,  # GlobalSemaphore handle
    reset_value,       # int (uint32)
)
# returns: None
```

### What it does

`reset_global_semaphore_value` calls `GlobalSemaphore::reset_semaphore_value(reset_value)`, which issues a blocking dispatch write:

```cpp
// tt_metal/impl/buffers/global_semaphore.cpp
void GlobalSemaphore::reset_semaphore_value(uint32_t reset_value) const {
    // Blocking write here to ensure that Global Semaphore reset value lands on
    // each physical device before the next program runs.
    // This is to ensure that cross-chip writes to the Global Semaphore are not
    // lost due to device skew.
    std::vector<uint32_t> host_buffer(cores_.num_cores(), reset_value);
    auto mesh_buffer = buffer_.get_mesh_buffer();
    bool using_fast_dispatch = MetalContext::instance().rtoptions().get_fast_dispatch();
    if (using_fast_dispatch) {
        distributed::EnqueueWriteMeshBuffer(
            mesh_buffer->device()->mesh_command_queue(), mesh_buffer, host_buffer, true);  // blocking=true
    } else { ... }
}
```

The `true` argument to `EnqueueWriteMeshBuffer` makes this a blocking write — it waits for the write to complete on every device before returning. The write sets the uint32 semaphore word to `reset_value` on every core in the `CoreRangeSet` on every device in the mesh.

This is the only host-side mechanism for resetting device-side semaphore state. Async CCL kernels leave semaphores in a non-zero state after completing their collective operation; `ttnn.reset_global_semaphore_value(handle, 0)` is how you prepare a handle for its next use.

> **Note:** In the standard single-CQ pattern used by `TT_CCL`, CQ FIFO ordering ensures the reset write runs after all prior CCL kernels finish — no extra `Finish()` or barrier call is required. Because `EnqueueWriteMeshBuffer` enqueues the reset write into the same mesh command queue as the prior CCL dispatch, the CQ executes operations in submission order: CCL kernels complete on device before the reset write executes, making L1 contention impossible in this path. The caller's actual responsibility is narrower: do not call `reset_global_semaphore_value` before the CCL dispatch has been enqueued (i.e., before the Python CCL op call returns), and do not issue the reset on a separate out-of-order command queue that bypasses the CCL dispatch CQ. Multi-CQ usage or direct L1 writes that bypass CQ ordering are out of scope for normal `TT_CCL` usage and would require explicit synchronization.

---

## Handle allocation in `TT_CCL.__init__`

The `TT_CCL` class (present in both `models/tt_transformers/tt/ccl.py` and `models/common/modules/tt_ccl.py`) allocates all semaphore handles in its constructor. The relevant portion of `__init__` is identical in both files:

```python
# models/tt_transformers/tt/ccl.py, TT_CCL.__init__
self.barrier_semaphore_idx = [0, 0, 0]
self.barrier_semaphore_handles = [[], [], []]

self.ag_semaphores_idx = [0, 0, 0]
self.ag_semaphore_handles = [[], [], []]

self.rs_semaphores_idx = [0, 0, 0]
self.rs_semaphore_handles = [[], [], []]

# cluster-axis-0, cluster-axis-1, no-cluster-axis
for i in range(3):
    # double buffered semaphores
    for _ in range(2):
        self.barrier_semaphore_handles[i].append(
            ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
        )

        self.ag_semaphore_handles[i].append(
            [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
        )

        self.rs_semaphore_handles[i].append(
            [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
        )
```

### Structure of the handle arrays

The outer index `i` corresponds to the cluster-axis variant:

| Array slot `i` | `cluster_axis` value |
|---|---|
| 0 | `cluster_axis=0` |
| 1 | `cluster_axis=1` |
| 2 | `cluster_axis=None` |

For each axis variant `i`, the inner loop runs twice (`for _ in range(2)`), creating two double-buffer slots. After `__init__` completes, the handle arrays have the following shapes:

```
barrier_semaphore_handles[3][2]
  barrier_semaphore_handles[i][slot]  → one GlobalSemaphore handle

ag_semaphore_handles[3][2][2]
  ag_semaphore_handles[i][slot]       → list of 2 GlobalSemaphore handles

rs_semaphore_handles[3][2][3]
  rs_semaphore_handles[i][slot]       → list of 3 GlobalSemaphore handles
```

The total number of `GlobalSemaphore` objects created is: `3 axes × 2 slots × (1 + 2 + 3) = 36` handles. Each handle is a distinct L1 buffer allocation; the allocator does not deduplicate them.

### The `CoreRangeSet` used

All handles are allocated over `self.sub_device_crs`, which is constructed to span the full compute grid of the mesh device:

```python
self.sub_device_crs = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(
                self.mesh_device.compute_with_storage_grid_size().x - 1,
                self.mesh_device.compute_with_storage_grid_size().y - 1,
            ),
        )
    }
)
```

This means each `GlobalSemaphore` covers every Tensix core on the mesh device, placing the semaphore word at the same L1 address across all of them.

---

**Next:** [Double-Buffer Design](./double_buffer_design.md)
