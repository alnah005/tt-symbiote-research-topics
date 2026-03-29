# Non-Composite Path — Reduce-Scatter + All-Gather

When all four composite-path predicates are `false` (the typical case for T3K decode
tensors; see [`call_chain.md`](./call_chain.md)), `ExecuteAllReduceAsync::invoke`
falls through to the non-composite code path. Because `ttnn.all_reduce` passes
`std::nullopt` for all three semaphore arguments, this path uses the synchronous
`ttnn.reduce_scatter` and `ttnn.all_gather` operations rather than their
`_minimal_async` counterparts. This file analyses what those operations do, what
semaphore state they use, and how intermediate tensors are allocated.

---

## 1. The non-composite branch in source

File:
`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`
(cluster\_axis overload, lines 326–398)

```cpp
// Reduce scatter + all gather
bool use_llama_sharded =
    composite_common::use_all_gather_async_llama_sharded(padded_tensor, out_memory_config);
padded_tensor.deallocate();
log_debug(tt::LogOp, "Using reduce scatter + all gather");

ttnn::Tensor scattered_tensor;
if (rs_global_semaphores.has_value() && barrier_semaphores.has_value()) {
    // async path — requires BOTH rs_global_semaphores AND barrier_semaphores to be present
    // NOT taken when called from ttnn.all_reduce (which passes std::nullopt for both)
    scattered_tensor = ttnn::experimental::reduce_scatter_minimal_async(...);
} else {
    // synchronous fallback — taken when EITHER rs_global_semaphores OR barrier_semaphores
    // is absent (std::nullopt). A caller that supplies only one of the two semaphore
    // arguments will fall through to this branch silently.
    scattered_tensor = ttnn::reduce_scatter(
        interleaved_tensor,
        dim,
        cluster_axis,
        worker_subdevice_id_opt,
        out_memory_config,
        std::nullopt,   // intermediate_memory_config
        std::nullopt,   // optional_output_tensor
        num_preferred_links,
        topology_,
        std::nullopt,   // chunks_per_sync
        std::nullopt,   // num_workers_per_link
        std::nullopt);  // num_buffers_per_channel
}
interleaved_tensor.deallocate();

ttnn::Tensor gathered;
if (ag_global_semaphores.has_value() && barrier_semaphores.has_value()) {
    // async path — requires BOTH ag_global_semaphores AND barrier_semaphores to be present
    // NOT taken when called from ttnn.all_reduce
    gathered = ttnn::prim::all_gather_async(...);
} else {
    // synchronous fallback — taken when EITHER ag_global_semaphores OR barrier_semaphores
    // is absent (std::nullopt).
    gathered = ttnn::all_gather(
        scattered_tensor,
        dim,
        cluster_axis,
        worker_subdevice_id_opt,
        out_memory_config,
        std::nullopt,   // sub_device_id (second optional)
        num_preferred_links,
        topology_);
}
scattered_tensor.deallocate();
```

The two `has_value()` guards are the critical switch. The guard condition is
`rs_global_semaphores.has_value() && barrier_semaphores.has_value()` — meaning **both**
must be present for the async branch to be taken. If **either** argument is
`std::nullopt`, the conjunction is `false` and execution falls through to the
synchronous op. Because `ttnn.all_reduce` passes `std::nullopt` for all semaphore
arguments, both guards evaluate to `false` and the synchronous ops are used.

---

## 2. Semaphore state — synchronous `ttnn.reduce_scatter` and `ttnn.all_gather`

The synchronous `ExecuteReduceScatter::invoke`
(`ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp`) and
`ExecuteAllGather::invoke`
(`ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp`) accept no
`GlobalSemaphore` parameters. Their function signatures carry only tuning knobs
(`num_links`, `topology`, `chunks_per_sync`, etc.) and standard memory configuration
options.

Internally, each synchronous CCL op constructs a `tt_metal::Program` object. Device
semaphores required for fabric-level synchronisation are embedded inside that program
as `CreateSemaphore` calls. These local semaphores:

- Are allocated from a per-program semaphore pool that lives in device L1 but is owned
  by the `Program` object, not by any user-visible handle.
- Are re-created and re-initialised to their reset value on every program dispatch.
- Are destroyed when the `Program` object goes out of scope at the end of the op call.

There is therefore **no persistent semaphore state** between successive calls to
`ttnn.reduce_scatter` or `ttnn.all_gather` when invoked through `ttnn.all_reduce`.
Each call starts from a clean state regardless of what the previous call left in L1.

> **Note:** This is the fundamental distinction from the async CCL ops. Async ops
> (`reduce_scatter_minimal_async`, `all_gather_async`) require that caller-supplied
> `GlobalSemaphore` handles are pre-created at known L1 addresses and that their state
> is managed (cycled) externally across calls. The synchronous ops have no such
> requirement because they manage their own ephemeral semaphore state.

---

## 3. Intermediate tensor allocation — `scattered_tensor`

After `ttnn.reduce_scatter` completes, the output `scattered_tensor` holds the
per-device partial result. For the decode case (sequence length = 1), the input
shape is `[1, 1, 1, out_features]` and the scattered shape is
`[1, 1, 1, out_features // num_devices]` (for a tensor scattered along dimension 3
across 8 devices, each device holds $out\_features / 8$ elements in the last
dimension). For prefill with sequence length $S$, the input shape is
`[1, 1, S, out_features]` and the scattered shape is
`[1, 1, S, out_features // num_devices]`.

`scattered_tensor` is allocated by `ttnn.reduce_scatter` as its output tensor using
the TTNN standard allocator. It is **not** pre-allocated before trace capture; it is
created inside the op's output allocation step.

This raises a question analogous to the composite-path concern: does this allocation
produce a different address on each call, making the trace unsafe?

The answer is that **`scattered_tensor` is a dynamically-allocated DRAM intermediate
and therefore faces the same trace incompatibility risk as the composite-path
intermediates.** TTNN trace captures metal-level command buffers; it does **not** pin
dynamically-allocated DRAM buffer addresses at capture time. A DRAM buffer allocated
inside an op during trace capture will be re-allocated by the standard allocator on
each replay and may land at a different DRAM bank or offset depending on the
free-list state at that moment. The result is that the commands in the trace body
can reference stale addresses on subsequent replays, leading to silent data corruption
or hardware faults — identical in mechanism to the `reshaped_tensor`/`gather_tensor`
problem documented for the composite path in `composite_path.md` §4.

> **Warning:** A caller implementing a trace-safe non-composite all-reduce must
> pre-allocate a persistent output buffer for `scattered_tensor` **before**
> `ttnn.begin_trace_capture` is called, and pass it as the `output_tensor`
> argument to `ttnn.reduce_scatter`. Without this pre-allocation, the non-composite
> path carries the same address-instability risk as the composite path. The
> compatibility verdict in §4 is therefore conditional on this pre-allocation
> precaution being taken.
>
> **This pre-allocation cannot be done through `ttnn.all_reduce`.** Inspecting
> `all_reduce_async.cpp` (cluster\_axis overload, lines 344–358) shows that the
> internal call to `ttnn::reduce_scatter` passes `std::nullopt` as both
> `intermediate_memory_config` (position 6) and `optional_output_tensor`
> (position 7). The `optional_output_tensor` argument is hardcoded as `std::nullopt`
> and the `ttnn.all_reduce` Python API exposes no parameter to override it. The only
> way to supply a pre-allocated buffer is to bypass `ttnn.all_reduce` entirely and
> call `ttnn.reduce_scatter` + `ttnn.all_gather` directly in
> `TTNNLinearIColShardedWAllReduced.forward`.

---

## 4. Summary — trace compatibility of the non-composite path

| Property | Non-composite (synchronous RS + AG) |
|---|---|
| External `GlobalSemaphore` handles | None — all semaphores are per-program local |
| Persistent semaphore state between calls | None |
| Intermediate tensor `scattered_tensor` | Dynamically allocated inside op; **potential trace risk** unless pre-allocated as a persistent buffer before capture |
| Dynamic allocations during replay | Risk present if `scattered_tensor` is not pre-allocated |
| Trace compatibility verdict | **Conditionally compatible** — requires `scattered_tensor` to be pre-allocated before trace capture (pass as `output_tensor` to `ttnn.reduce_scatter`), and fabric config must be 1-D |

The non-composite synchronous path does not use external global semaphores, so there
is no persistent semaphore state to manage across replays. However, `scattered_tensor`
is a dynamic DRAM intermediate allocation (see §3 above), which is a potential trace
compatibility risk identical in mechanism to the composite-path intermediate
allocations. The path is compatible with `ttnn.begin_trace_capture` /
`ttnn.execute_trace` **only when** `scattered_tensor` is pre-allocated as a persistent
buffer before capture begins and the fabric configuration keeps all four
composite-path predicates `false`.

---

**Next:** [`contrast_with_async_variants.md`](./contrast_with_async_variants.md)
