# Chapter 4: Correct Synchronization Strategies for Traced Async CCL

This chapter presents the concrete patterns required to make trace replay work correctly when the traced model calls `ttnn.experimental.all_gather_async` or `ttnn.experimental.reduce_scatter_minimal_async`. Chapters 1 through 3 established the problem: semaphore addresses are baked into the `TraceNode` at `end_trace_capture` time, the host-side cycling counters diverge from the trace-baked indices after the first capture, and device-side semaphore values left non-zero by a completed kernel will cause incorrect behavior on the next replay. This chapter explains the two independent dimensions of the fix — resetting host counters and resetting device semaphore values — and shows how to structure the capture and replay phases so that both dimensions remain consistent across every `execute_trace` call.

---

## Prerequisites from prior chapters

- **Chapter 1** — `GlobalSemaphore` handle, L1 semaphore address, address stability, double-buffer slots, the three cycling counter arrays (`barrier_semaphore_idx`, `ag_semaphores_idx`, `rs_semaphores_idx`), `get_and_cycle_*` methods, handle array layout in `TT_CCL.__init__`
- **Chapter 2** — Why semaphore addresses are runtime arguments (RTAs), not compile-time args; the `override_runtime_arguments` path; how `create_trace_node` snapshots RTA data into `TraceNode.rta_data` at `end_trace_capture` time; how replay restores those snapshots on every `execute_trace`
- **Chapter 3** — The host-counter / trace-handle mismatch (the counter advances to `(N+1) % 2` while every replay reuses handle `N`); the two failure modes (silent corruption from stale non-zero device semaphore values, and hang from a kernel that skips its wait); that `reduce_scatter_minimal_async` and `all_gather_async` can be placed inside a trace bracket as long as semaphore state is managed correctly

---

## State machine: correct synchronization around each replay

The following diagram shows the required ordering of operations for each decode step. The host (left column) and device (right column) are separated by a vertical bar; each row is one logical operation.

```
Host                                           | Device
-----------------------------------------------|-----------------------------------------------
1. Enqueue reset_global_semaphore_value(s)     | [write 0 to L1 semaphore word arrives here,
   for each capture-time handle                |  ordered before trace kernels by CQ FIFO]
                                               |
2. Reset TT_CCL index fields to capture-time   |
   values (barrier_semaphore_idx,              |
   ag_semaphores_idx, rs_semaphores_idx)        |
                                               |
3. ttnn.execute_trace(blocking=False) -------->| Kernels run; CCL kernels wait on semaphore,
                                               | complete, write non-zero to semaphore as
                                               | terminal action, then self-reset to 0
                                               |
4. [next decode token preparation on host]     | [device running or idle]
                                               |
5. Repeat from step 1                          |
```

The key invariant: at the moment the CCL kernels begin (step 3), the L1 semaphore words for all capture-time handles must read 0, and the host-side index fields must point to those same capture-time handles so that any subsequent non-traced CCL call also selects them.

---

## Files in reading order

1. [Resetting Host Counters](resetting_host_counters.md) — The snapshot/restore pattern for `TT_CCL` index fields; why this is necessary even when device-side resets are done; code pattern
2. [Resetting Device Semaphore Values](resetting_device_semaphore_values.md) — How to call `reset_global_semaphore_value` for each capture-time handle; timing and CQ FIFO ordering
3. [Structuring the Capture](structuring_the_capture.md) — Recommended checklist for pre-capture, capture, and pre-replay setup
4. [Existing Patterns in tt-transformers](existing_patterns_in_tt_transformers.md) — Audit of `generator.py` and related traced code; which paths need semaphore management and which do not

---

## What's next

After completing Chapter 4, proceed to [Chapter 5 — Implementing Trace Support: A Step-by-Step Guide](../ch5_implementation_guide/index.md).
