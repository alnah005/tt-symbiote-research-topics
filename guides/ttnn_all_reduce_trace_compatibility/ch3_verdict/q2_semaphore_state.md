# Q2 — Does `ttnn.all_reduce` Use Persistent Semaphore State That Could Conflict With Trace Replay?

This file answers the second research question: whether `ttnn.all_reduce` leaves any
persistent semaphore state on device between invocations that could corrupt a trace
replay. The conclusion is no: `ttnn.all_reduce` passes `std::nullopt` for all three
`GlobalSemaphore` argument groups, and the downstream synchronous collective ops use
only per-program local semaphores that are created fresh on every dispatch and cease
to exist after the program completes.

---

## Conclusion

`ttnn.all_reduce` introduces no persistent global semaphore state that the caller must
manage between trace captures or across replays. The semaphore lifecycle is entirely
local to each program dispatch. A replay can be issued any number of times without
any semaphore initialisation step between replays. The remaining risk is entirely in
the buffer address stability of the dynamically allocated `scattered_tensor` intermediate.

This is in sharp contrast with `ttnn.experimental.all_reduce_async`, which requires
the caller to create `GlobalSemaphore` objects before capture and ensure they are
correctly initialised before each replay.

---

## How `ttnn.all_reduce` passes semaphore arguments

The synchronous `ttnn.all_reduce` entry point is
`ttnn::operations::ccl::ExecuteAllReduce::invoke` in
`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`:

```cpp
// all_reduce.cpp lines 44-55
return ::ttnn::experimental::all_reduce_async(
    input_tensor,
    cluster_axis,
    *mesh_device,
    std::nullopt,   // barrier_semaphores
    std::nullopt,   // rs_global_semaphores
    std::nullopt,   // ag_global_semaphores
    ttnn::operations::reduction::ReduceType::Sum,
    memory_config,
    topology_,
    num_links,
    subdevice_id);
```

All three semaphore parameters receive `std::nullopt`. They are forwarded to the
cluster-axis overload of `ExecuteAllReduceAsync::invoke`, where they drive the path
selection inside the non-composite branch:

```cpp
// all_reduce_async.cpp lines 331-358
if (rs_global_semaphores.has_value() && barrier_semaphores.has_value()) {
    scattered_tensor = ttnn::experimental::reduce_scatter_minimal_async(
        interleaved_tensor,
        std::nullopt,
        dim,
        rs_global_semaphores.value(),
        barrier_semaphores.value()[0],
        ...);
} else {
    scattered_tensor = ttnn::reduce_scatter(
        interleaved_tensor,
        dim,
        cluster_axis,
        worker_subdevice_id_opt,
        out_memory_config,
        std::nullopt,
        std::nullopt,
        num_preferred_links,
        topology_,
        ...);
}
```

Because `rs_global_semaphores` and `barrier_semaphores` are both `std::nullopt`, the
`else` branch is always taken: the implementation calls the synchronous `ttnn::reduce_scatter`
op rather than `reduce_scatter_minimal_async`. The same nullopt guard applies to the
all-gather step:

```cpp
// all_reduce_async.cpp lines 361-393
if (ag_global_semaphores.has_value() && barrier_semaphores.has_value()) {
    gathered = ttnn::prim::all_gather_async(...);
} else {
    gathered = ttnn::all_gather(
        scattered_tensor,
        dim,
        cluster_axis,
        worker_subdevice_id_opt,
        out_memory_config,
        std::nullopt,
        num_preferred_links,
        topology_);
}
```

Again, because `ag_global_semaphores` is `std::nullopt`, the synchronous
`ttnn::all_gather` is called.

---

## Local semaphore lifecycle in synchronous collective ops

The synchronous `ttnn::reduce_scatter` and `ttnn::all_gather` implementations create
their synchronisation semaphores as per-program local semaphores — L1 memory that is
allocated as part of the program object, initialised when the program is compiled and
dispatched, and discarded when the program completes. The semaphore values are not
written to any fixed L1 address that persists across program launches.

Tracing records the sequence of command queue commands issued during capture. On
replay, each command is reissued in the same order. Because the local semaphore
addresses are embedded in the kernel binaries that were compiled during capture (not
in device state that survives between calls), the replay sees the same addresses and
initial values that capture saw. No external initialisation step is required.

---

## Contrast with `ttnn.experimental.all_reduce_async`

`ttnn.experimental.all_reduce_async` — exposed via the Python binding of the same name
and used in similar async CCL patterns — requires caller-supplied `GlobalSemaphore`
objects. A concrete example of the async semaphore pattern is
`TTNNLinearIColShardedWRowSharded.forward`, which calls
`ttnn.experimental.reduce_scatter_minimal_async` directly with explicit semaphore
handles (`linear.py` lines 158–172):

```python
# TTNNLinearIColShardedWRowSharded.forward (linear.py lines 158-172)
tt_output = ttnn.experimental.reduce_scatter_minimal_async(
    tt_output,
    persistent_output_buffers=None,
    dim=3,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    cluster_axis=1,
    ...
)
```

A `GlobalSemaphore` occupies a fixed L1 address that is determined at creation time by
`ttnn.create_global_semaphore`. That address is recorded into trace command payloads
during capture. If the semaphore value is not reset to its expected initial state
before each replay, the kernel may spin indefinitely waiting for a signal that was
already consumed by the previous replay.

The key differences between the two variants are summarised below:

| Property | `ttnn.all_reduce` (synchronous) | `ttnn.experimental.all_reduce_async` |
|----------|--------------------------------|--------------------------------------|
| Semaphore type | Local (per-program) | Global (fixed L1 address) |
| Semaphore lifetime | Single program dispatch | Persists on device until deallocated |
| Caller must create semaphores | No | Yes — `ttnn.create_global_semaphore` |
| Caller must reset before replay | No | Yes — semaphore values are consumed |
| Trace-captured address stable | N/A (no fixed address) | Yes — address is fixed; value must be correct |
| Buffer cycling required | No | Yes — typical pattern uses a pool of $N$ semaphore handles |

> **Note:** `TTNNLinearIColShardedWAllReduced` uses the synchronous `ttnn.all_reduce`
> path (no caller-managed semaphores). `TTNNLinearIColShardedWRowSharded` uses the
> async path with `GlobalSemaphore` cycling. These two classes have different trace
> requirements and must not be confused during integration.

---

**Next:** [`q3_requirements_and_limitations.md`](./q3_requirements_and_limitations.md)
