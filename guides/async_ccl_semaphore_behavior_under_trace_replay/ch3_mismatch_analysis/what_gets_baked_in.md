# What Gets Baked In: Tracing a Capture with tt_all_reduce

This file walks through the precise sequence of host-side state changes and device-side dispatch events that occur when a decode trace is captured for a model that calls `tt_all_reduce` with `use_composite=False` on a TG mesh. By the end of this file you will know exactly which semaphore handles get embedded into the immutable DRAM command buffer at `end_trace_capture` time, what the host counter points to after capture completes, and why those two values are immediately and permanently inconsistent.

---

## The Code Path Under Analysis

`tt_all_reduce` with `use_composite=False` is the TG path in `models/tt_transformers/tt/ccl.py`. When this path is taken, the function calls two async CCL ops in sequence:

```python
gathered_tensor = ttnn.experimental.all_gather_async(
    input_tensor,
    persistent_output_buffer=None,
    dim=dim,
    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
    # ...
    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
    # ...
)

reduced_tensor = ttnn.experimental.fast_reduce_nc(
    gathered_tensor,
    # ...
)
```

The two `get_and_cycle_*` calls each read the current index, return the corresponding handle, and advance the index modulo 2. The returned handles are passed directly to `ttnn.experimental.all_gather_async`, which writes their `.address()` values into the per-core RTA slots via `override_runtime_arguments` in `all_gather_async_minimal_default_helper_override_runtime_arguments` (in `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`):

```cpp
// In all_gather_async_minimal_default_helper_override_runtime_arguments:
worker_reader_sender_runtime_args[2] = out_ready_semaphore.address();  // out_ready_semaphore L1 address
worker_writer_sender_runtime_args[3] = out_ready_semaphore.address();  // out_ready_semaphore L1 address
worker_writer_sender_runtime_args[5] = barrier_semaphore.value().address();  // barrier_semaphore L1 address
```

These are the addresses that flow into the DRAM command buffer.

---

## State Before the Compile Run

The `_capture_decode_trace_text` path in `models/tt_transformers/tt/generator.py` runs a compile run first:

```python
# Compile run (no trace capture)
self._decode_forward_no_trace_text(tokens, current_pos, ...)
```

Let the state of `TT_CCL` at the start of the compile run be:

- `ag_semaphores_idx[semaphore_index]` = `N` (some value in `{0, 1}`)
- `barrier_semaphore_idx[semaphore_index]` = `M` (some value in `{0, 1}`)

where `semaphore_index = cluster_axis` for a concrete `cluster_axis` value of 0 or 1.

For the `cluster_axis=0` edge case with the older CCL file, see the Note in this chapter's [index](./index.md) and [Chapter 1](../ch1_global_semaphore_internals/double_buffer_design.md).

The compile run calls `tt_all_reduce`, which calls:

1. `get_and_cycle_ag_semaphore_handles(cluster_axis)`: returns `ag_semaphore_handles[semaphore_index][N]`; advances `ag_semaphores_idx[semaphore_index]` to `(N+1) % 2`.
2. `get_and_cycle_barrier_semaphore_handle(cluster_axis)`: returns `barrier_semaphore_handles[semaphore_index][M]`; advances `barrier_semaphore_idx[semaphore_index]` to `(M+1) % 2`.

After the compile run:

- `ag_semaphores_idx[semaphore_index]` = `(N+1) % 2`
- `barrier_semaphore_idx[semaphore_index]` = `(M+1) % 2`
- The device L1 semaphore words for `ag_semaphore_handles[semaphore_index][N]` and `barrier_semaphore_handles[semaphore_index][M]` are reset to 0 by kernel self-reset (compile run is synchronous): the reader kernel resets `out_ready_sem` to 0 as its final action (`minimal_default_reader.cpp` line 295) and the writer kernel resets `barrier_sem` to 0 (`minimal_default_writer.cpp` line 272). When the compile run returns, both semaphore words are clean.

> **Note:** The compile run is not traced. The program cache is populated on this run, so the capture run that follows will be a cache hit. The cycling counters have advanced: the next `get_and_cycle_*` call will return slot `(N+1) % 2` for AG and slot `(M+1) % 2` for barrier.

For simplicity in the remainder of this file, define:

- `N'` = `(N+1) % 2` — the post-compile AG index
- `M'` = `(M+1) % 2` — the post-compile barrier index

---

## State at the Start of Trace Capture

`_capture_decode_trace_text` then calls `ttnn.begin_trace_capture` and runs `ttnn_decode_forward`, which calls `tt_all_reduce` again.

At the moment `tt_all_reduce` is invoked inside the trace bracket:

- `ag_semaphores_idx[semaphore_index]` = `N'`
- `barrier_semaphore_idx[semaphore_index]` = `M'`

The two `get_and_cycle_*` calls inside `tt_all_reduce`:

1. `get_and_cycle_ag_semaphore_handles(cluster_axis)`: returns `ag_semaphore_handles[semaphore_index][N']`; advances `ag_semaphores_idx[semaphore_index]` to `(N'+1) % 2 = N`.
2. `get_and_cycle_barrier_semaphore_handle(cluster_axis)`: returns `barrier_semaphore_handles[semaphore_index][M']`; advances `barrier_semaphore_idx[semaphore_index]` to `(M'+1) % 2 = M`.

The returned handles — `ag_semaphore_handles[semaphore_index][N']` and `barrier_semaphore_handles[semaphore_index][M']` — are passed to `ttnn.experimental.all_gather_async`. Because the program is already in the cache (from the compile run), this is a cache hit. `override_runtime_arguments` is called, which writes:

- `ag_semaphore_handles[semaphore_index][N'][dir].address()` for each direction `dir` in `{0, 1}` into the RTA slots for `out_ready_semaphore` (slots `[2]` on the reader and `[3]` on the writer per direction). Note: `ag_semaphore_handles[semaphore_index][N']` is a **list of 2 `GlobalSemaphore` objects** (one per direction); the C++ layer accesses each element as `semaphore.at(dir).address()`, baking **two distinct L1 addresses** into the trace per AG semaphore slot. Consequently, resetting requires calling `reset_global_semaphore_value` for each of the 2 handles in the list.
- `barrier_semaphore_handles[semaphore_index][M'].address()` into the RTA slot for `barrier_semaphore` (slot `[5]` on the writer).

These addresses are written into the sysmem bypass buffer that the trace recorder is capturing.

---

## What end_trace_capture Freezes

When `ttnn.end_trace_capture` is called, `FDMeshCommandQueue::record_end` calls `assemble_dispatch_commands`, which reads the bypass buffer and builds `MeshTraceDescriptor::ordered_trace_data`. The RTA values embedded in the bypass buffer at this moment — specifically `ag_semaphore_handles[semaphore_index][N'][dir].address()` for each direction `dir` in `{0, 1}` (two L1 addresses, as noted above) and `barrier_semaphore_handles[semaphore_index][M'].address()` — are copied verbatim into `ordered_trace_data`.

`populate_mesh_buffer` then uploads this data to DRAM. From this point forward, the DRAM command buffer is immutable.

> **Key insight:** The handles frozen into the trace are the ones selected by the `get_and_cycle_*` calls that ran inside the `begin_trace_capture` / `end_trace_capture` bracket — that is, slot `N'` for AG and slot `M'` for barrier. These are the capture-time handles.

---

## State After Capture Completes

After `end_trace_capture`:

- `ag_semaphores_idx[semaphore_index]` = `N` (the counter wrapped back)
- `barrier_semaphore_idx[semaphore_index]` = `M` (the counter wrapped back)
- The DRAM command buffer is frozen with slot `N'` and slot `M'` addresses.

The host counter is now pointing at slot `N`, while the trace is permanently bound to slot `N'`. These are the opposite double-buffer slots:

```
slot index:  0         1
             ┌─────────┬─────────┐
ag handles:  │ handle A │ handle B │
             └─────────┴─────────┘
                 ^               ^
                 │               │
             host counter      trace-baked
             (pointing here)   (baked here)
             ag_idx = N         N' = (N+1)%2
             (e.g., 0)          (e.g., 1)
```

The same inversion holds for the barrier semaphore index.

---

## What Happens on Each Replay

Each `execute_trace` call causes the hardware prefetcher to fetch and execute the DRAM command buffer assembled at capture time. The prefetcher writes the frozen RTA values — containing `ag_semaphore_handles[semaphore_index][N'][dir].address()` for each direction `dir` in `{0, 1}` (two L1 addresses, as noted above) and `barrier_semaphore_handles[semaphore_index][M'].address()` — into the per-core kernel config space before dispatching the kernels.

No Python code runs during replay. No `get_and_cycle_*` call is made. The host counter remains at `N` / `M` and does not advance.

Every replay, from the first to the last, uses exactly the same L1 addresses: those of slot `N'` and slot `M'`. The host counter continues to point at the opposite slots.

> **Warning:** If a non-traced `tt_all_reduce` call is made between two trace replays (for example, a prefill step that is not covered by the trace), that call will invoke `get_and_cycle_ag_semaphore_handles(cluster_axis)`, which returns the current host-counter slot and advances the index. The first non-traced call after capture uses slot `N` (no collision — the trace uses `N'`). The second non-traced call would use slot `N'` — the same slot the trace is bound to — causing a collision. The collision risk alternates with every additional non-traced call. This is the pattern analyzed in `failure_modes.md` Case B.

---

## Summary of State Transitions

| Moment | `ag_semaphores_idx[si]` | AG handle in use | `barrier_semaphore_idx[si]` | Barrier handle in use |
|---|---|---|---|---|
| Before compile run | `N` | — | `M` | — |
| After compile run | `N'` = `(N+1)%2` | slot `N` was used | `M'` = `(M+1)%2` | slot `M` was used |
| During capture bracket | `N'` | slot `N'` being dispatched | `M'` | slot `M'` being dispatched |
| After `end_trace_capture` | `N` | — (trace baked: `N'`) | `M` | — (trace baked: `M'`) |
| After replay 1 | `N` (unchanged) | trace uses `N'` | `M` (unchanged) | trace uses `M'` |
| After replay 2 | `N` (unchanged) | trace uses `N'` | `M` (unchanged) | trace uses `M'` |

The divergence is immediate and permanent after the capture: the host counter points at slot `N` while the trace is bound to slot `N'`.

---

**Next:** [Traceability of Async CCL Ops](traceability_of_async_ccl_ops.md)
