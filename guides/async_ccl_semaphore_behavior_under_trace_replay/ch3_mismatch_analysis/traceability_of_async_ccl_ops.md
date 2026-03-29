# Traceability of Async CCL Ops

This file answers the question of whether `ttnn.experimental.reduce_scatter_minimal_async` and `ttnn.experimental.all_gather_async` can be placed inside a `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` bracket. The short answer is yes, with one non-negotiable caveat around semaphore state management. By the end of this file you will understand why no architectural barrier prevents their inclusion, why the persistent output buffer argument does not pose a problem, and exactly what the caller is responsible for managing across the capture-replay boundary.

---

## Architectural Question: Can These Ops Be Traced?

The question of traceability is really two questions:

1. Does the dispatch path for `reduce_scatter_minimal_async` or `all_gather_async` use any mechanism that is incompatible with the trace recorder?
2. Are there arguments whose values are not captured correctly by the trace?

The answer to question 1 is no. Both ops go through the standard `device_operation` program cache path. On a program cache hit, `override_runtime_arguments` is called to write the current RTA values (buffer addresses, semaphore addresses) into the live command sequence. This is the same path used by all other traced ops. The trace recorder, operating in bypass mode, captures the resulting command stream entries for these ops exactly as it captures entries for any other op. There is no architectural incompatibility.

As shown in Chapter 2, the mesh trace path assembles an immutable DRAM command buffer at `end_trace_capture` time. The RTA values written by `override_runtime_arguments` during the capture bracket — including semaphore addresses — are embedded in that buffer verbatim. The fact that these ops use async Ethernet data movement does not change the trace capture mechanism.

> **Note:** The trace records the host-side dispatch command stream, not the device-side kernel execution timeline. Async Ethernet transfers that begin during trace replay are launched by the replayed command stream in the same way they are launched by the original command stream. The trace mechanism is agnostic to whether a kernel is async or synchronous.

---

## Persistent Output Buffer: Not a Problem

Both `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` accept an optional `persistent_output_buffer` argument. In virtually all tt-transformers uses, this argument is `None`:

```python
gathered_tensor = ttnn.experimental.all_gather_async(
    input_tensor,
    persistent_output_buffer=None,   # output tensor allocated fresh each call
    # ...
)
```

```python
reduced = ttnn.experimental.reduce_scatter_minimal_async(
    input_tensor,
    persistent_output_buffers=None,  # output tensor allocated fresh each call
    # ...
)
```

When `persistent_output_buffer` is `None`, the op allocates a new output tensor on each invocation. The output buffer's DRAM address changes from call to call. This is not a problem for tracing because output buffer addresses are also written as RTAs by `override_runtime_arguments`. The address written during the capture bracket (the address of the output buffer allocated during that capture invocation) is embedded into the DRAM command buffer. On each replay, the hardware prefetcher uses that same frozen output buffer address.

This means the output tensor produced by a traced async CCL op is always written to the same allocation that existed at capture time — which is the standard behavior for all traced ops with DRAM output tensors. The caller must ensure that output tensor lifetime extends across all replays, which is the case when the output tensor is held by the Python model object (as it is in the `tt_out_trace` variable in `_capture_decode_trace_text`).

> **Note:** If `persistent_output_buffer` were non-`None`, it would similarly be written as an RTA and frozen at capture time. The persistent-buffer case is actually simpler from a trace perspective because the same output buffer is reused by design.

---

## What the Trace Does Not Capture: Device L1 Semaphore Values

The trace captures the host-side dispatch command stream. It does not capture or restore device L1 state. The device-side semaphore word — the `uint32_t` at `semaphore.address()` in L1 — is written by the async CCL kernels to signal completion during replay, exactly as it is during any non-traced execution. The trace has no mechanism to reset this word between replays.

This is the critical distinction: semaphore addresses (RTAs) are frozen at capture time and replayed correctly; semaphore values (device L1 words) are modified by kernels at runtime and must be managed by the caller.

---

## Critical Caveat: Semaphore State Must Be Managed Across the Boundary

Because semaphore addresses are snapshotted into the DRAM command buffer at capture time, any `reduce_scatter_minimal_async` or `all_gather_async` op placed inside a trace bracket is permanently bound to the capture-time handles for all replays. A model is free to place these ops inside a trace bracket, subject to the following requirements:

1. **Before each `execute_trace`:** the device L1 semaphore words for the capture-time handles must be reset to 0 using `ttnn.reset_global_semaphore_value`. If a previous replay left them non-zero, the next replay's kernels may pass through their wait condition immediately rather than waiting for a completion signal, producing silent data corruption.

2. **Host counter alignment:** The host-side `TT_CCL` cycling counters must be managed so that any non-traced CCL call that occurs outside the trace bracket does not use the same handles that the trace is bound to. As shown in [`what_gets_baked_in.md`](what_gets_baked_in.md), the host counter points to the opposite double-buffer slot after capture. If the counter is left uncorrected, non-traced CCL calls and trace replays will share the same semaphore handles.

3. **Initial device state at capture time must match initial device state at replay time:** the device L1 semaphore words for the capture-time handles must be at 0 at the start of the capture bracket, and they must be at 0 at the start of each replay. Chapter 4 explains how to structure the capture and replay sequence to satisfy this requirement.

> **Key insight:** There is no architectural barrier to placing `reduce_scatter_minimal_async` or `all_gather_async` inside a trace bracket. The constraint is entirely about semaphore state management: the capture-time handles are frozen into the trace, and the caller is responsible for ensuring those handles have clean device state (value = 0) before each replay, and that no other code path uses those handles concurrently with a replay.

---

## What the Existing tt-transformers Decode Trace Does

The existing `_capture_decode_trace_text` and `_decode_forward_trace_text` paths in `models/tt_transformers/tt/generator.py` perform no semaphore reset and no counter management around `execute_trace`:

```python
# In _decode_forward_trace_text (models/tt_transformers/tt/generator.py):
for i, trace_id in self.trace_ids_decode[sampling_on_device].items():
    ttnn.execute_trace(self.model_args[i].mesh_device, trace_id, cq_id=0, blocking=False)
```

There is no call to `ttnn.reset_global_semaphore_value` before `execute_trace`, and no assignment to `TT_CCL` index fields. This is correct behavior for the existing code because the current decode trace in tt-transformers does not place async CCL ops inside the trace bracket. The existing traced decode path uses non-async CCL variants (barrier-free or composite paths) that do not go through `reduce_scatter_minimal_async` / `all_gather_async` with global semaphores.

The `tt_all_reduce` with `use_composite=False` path (the TG `all_gather_async` + `fast_reduce_nc` sequence) and direct `tt_all_gather` calls via `all_gather_async` are the paths that use `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle`. Adding trace support for those paths requires explicit semaphore management as described in Chapter 4.

---

**Next:** [Failure Modes](failure_modes.md)
