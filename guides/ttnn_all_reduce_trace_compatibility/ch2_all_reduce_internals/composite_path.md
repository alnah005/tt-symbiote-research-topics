# Composite Path — All-Gather + Local Sum

The composite path is the code branch inside `ExecuteAllReduceAsync::invoke`
(cluster\_axis overload in
`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`)
that is selected when at least one of the four predicates evaluated in
[`call_chain.md`](./call_chain.md) returns `true`. It implements all-reduce as a
two-stage operation: a cross-device all-gather that replicates every shard to every
device, followed by a local elementwise sum that collapses the replicated dimension
on each device independently.

---

## 1. When is the composite path selected?

The composite branch is entered when any of the following holds:

- `composite_common::use_composite_all_gather(...)` returns `true` — the tensor is
  row-major, the gather dimension is not tile-aligned, or a 2D fabric is active.
- `composite_common::use_composite_reduce_scatter(...)` returns `true` — the scatter
  dimension IS evenly divisible by `num_devices` but the resulting per-device slice is
  not tile-aligned, or the tensor is row-major. Note: when the scatter dimension is
  **not** evenly divisible by `num_devices`, `use_composite_reduce_scatter` returns
  `false` (do NOT use composite); in that case the composite path is entered via
  `dim != composite_dim` (see below), not via this predicate.
- `dim != composite_dim` — `finding_scatter_dim` returned `rank` (past-the-end),
  meaning no dimension divides evenly by `num_devices`, so `composite_dim` was
  forced to 0.
- `composite_for_2d_mesh` — the fabric config is `FABRIC_2D` and the mesh has more
  than one device on each axis.

---

## 2. What the composite path does

```cpp
if (composite_all_gather || composite_reduce_scatter
    || (dim != composite_dim) || composite_for_2d_mesh) {

    composite_dim = 0;

    // Reshape to expose a contiguous batch dimension for the gather.
    // The cluster-axis overload uses a dynamic SmallVector so that the
    // reshape works correctly for tensors of any rank, not just rank-4.
    ttnn::SmallVector<uint32_t> ag_shape_vec(initial_shape.rank());
    std::copy(initial_shape.cbegin() + 2, initial_shape.cend(), ag_shape_vec.begin() + 2);
    ag_shape_vec[0] = 1;
    ag_shape_vec[1] = initial_shape[0] * initial_shape[1];

    auto reshaped_tensor = ttnn::reshape(interleaved_tensor, ttnn::Shape(ag_shape_vec));
    interleaved_tensor.deallocate();

    // Stage 1: all-gather (composite_common::composite_all_gather)
    auto gather_tensor = composite_common::composite_all_gather(
        reshaped_tensor,
        composite_dim,          // dim = 0
        num_preferred_links.value_or(1),
        out_memory_config,
        worker_subdevice_id_opt,
        cluster_axis);
    reshaped_tensor.deallocate();

    // Stage 2: local reduce (moreh_sum on each device independently)
    auto sum_tensor =
        is_float32
            ? local_sum_float32(gather_tensor, 0, num_devices, out_memory_config)
            : local_sum(gather_tensor, 0, out_memory_config);
    gather_tensor.deallocate();

    return ttnn::reshape(sum_tensor, initial_shape);
}
```

> **Note on overload difference:** The non-cluster-axis overload of
> `ExecuteAllReduceAsync::invoke` (lines 152–256 of `all_reduce_async.cpp`) uses the
> hard-coded 4-element form
> `ttnn::Shape({1, initial_shape[0] * initial_shape[1], initial_shape[2], initial_shape[3]})`.
> The cluster-axis overload shown above uses the dynamic `SmallVector` form
> (`ttnn::Shape(ag_shape_vec)`), which copies all dimensions beyond index 1 from the
> original shape. For rank-4 tensors the two forms produce identical shapes; for any
> other rank they differ. Because `ttnn.all_reduce` delegates to the cluster-axis
> overload, the dynamic form is the representative code for this chapter.

### Stage 1 — `composite_common::composite_all_gather`

File: `ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp`

`composite_all_gather` is implemented using `ttnn::prim::all_broadcast` followed by
`ttnn::concat`. `all_broadcast` distributes each device's local shard to all other
devices using `Topology::Linear` (a 1-D scatter), producing one tensor per device.
`ttnn::concat` then joins these tensors along the gather dimension as a **device op**
— it executes on-device, not on the host. Allocation and command recording for the
concat output therefore occur on the device, inside the trace capture window. The
function signature is:

```cpp
ttnn::Tensor composite_all_gather(
    ttnn::Tensor input_tensor,
    const int32_t dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    std::optional<uint32_t> cluster_axis);
```

There is no `GlobalSemaphore` parameter. Synchronisation inside `all_broadcast` is
managed by the primitive op's own local (per-program) semaphores, which are created
as part of the `tt_metal::Program` object and are re-initialised on every program
dispatch. They are not accessible as caller-visible objects.

### Stage 2 — `local_sum` / `local_sum_float32`

```cpp
Tensor local_sum(
    const ttnn::Tensor& gathered_tensor,
    int reduce_dim,
    const std::optional<ttnn::MemoryConfig>& memory_config) {

    auto sum_tensor = ttnn::moreh_sum(
        gathered_tensor,
        reduce_dim,
        /* keep_dim */ true,
        /* output */ std::nullopt,
        memory_config,
        /* device kernel config */ std::nullopt);
    ...
}
```

`ttnn.moreh_sum` is a single-device compute operation. It has no CCL component and
requires no cross-device semaphores. Its output tensor is allocated fresh on each call
through the standard TTNN allocator.

---

## 3. Semaphore state in the composite path

Neither `composite_all_gather` nor `local_sum` accepts or stores any external
`GlobalSemaphore` handle. From a semaphore-state perspective, both stages are
self-contained: all synchronisation uses local semaphores embedded in the dispatched
`tt_metal::Program` objects, which are re-created and re-initialised on every call.

There is **no persistent semaphore state** that persists between invocations of the
composite path. A trace replay does not need to reset or cycle any external semaphore
handle to execute these stages correctly.

---

## 4. Why the composite path is incompatible with trace capture

> **Key finding:** The composite path dynamically allocates intermediate device tensors
> on every call. This is incompatible with trace capture.

Trace capture (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture`) records a
sequence of command-buffer commands that reference device buffers by their fixed L1
or DRAM addresses. When the trace is replayed (`ttnn.execute_trace`), the runtime
re-issues exactly those commands to the same addresses. Any tensor that is allocated
after trace capture begins must therefore live at the **same address on every replay**.
This is why the trace pattern requires persistent buffers: tensors allocated before
capture and kept alive.

The composite path violates this requirement in three places:

1. **`reshaped_tensor`** — allocated by `ttnn::reshape` inside the capture window.
   Its backing buffer is requested from the allocator on each call; the allocator may
   assign a different DRAM bank or offset depending on the current free-list state.

2. **`gather_tensor`** — the output of `composite_all_gather`. It is sized as
   `[num_devices, initial_shape[0] * initial_shape[1], H, W]`: `composite_all_gather`
   calls `ttnn::prim::all_broadcast` to replicate the reshaped tensor (shape
   `[1, initial_shape[0]*initial_shape[1], H, W]`) to all `num_devices` devices, then
   `ttnn::concat` along `composite_dim = 0`, so the first dimension becomes
   `num_devices`, not the second. For example, a typical decode input `[1, 1, 32, H]`
   produces a `gather_tensor` of shape `[8, 1, 32, H]` on an 8-device T3K, not
   `[1, 8, 32, H]`. This allocation is not pre-declared before the trace and will
   receive a fresh address on each call.

3. **`sum_tensor`** — the output of `moreh_sum`. Also freshly allocated.

Each of these tensors is immediately `.deallocate()`-d after use (the `deallocate()`
calls in the code above are explicit). However, during a trace-capture pass, the
allocator records each allocation event as a command. On replay, those commands are
re-issued, and the allocator may place the tensor at a different address because the
free-list state at replay time differs from the state at capture time (prior tensors
may have been freed or allocated in a different order). The result is that the commands
in the trace body reference stale addresses, leading to silent data corruption or
hardware faults.

> **Warning:** If the T3K fabric is switched from a 1-D to a 2-D configuration
> (i.e., `FabricConfig::FABRIC_2D`), the `composite_for_2d_mesh` predicate becomes
> `true` and the composite path is unconditionally selected — even when the tensor
> dimensions would otherwise satisfy the non-composite conditions. A module using
> `ttnn.all_reduce` inside a trace must therefore be tested on the exact fabric
> configuration used in production.

---

**Next:** [`reduce_scatter_all_gather_path.md`](./reduce_scatter_all_gather_path.md)
