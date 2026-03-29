# Chapter 3 — The Host-Counter / Trace-Handle Mismatch

This chapter precisely characterizes the inconsistency that arises when the host-side cycling counter in `TT_CCL` advances while trace replay locks the semaphore handle to its capture-time value. By the end of this chapter you will understand the exact sequence of state changes that creates the mismatch, whether `reduce_scatter_minimal_async` and `all_gather_async` can be placed inside a `begin_trace_capture` / `end_trace_capture` bracket, and what failure modes arise when the mismatch is left uncorrected.

---

## Prerequisites from Chapters 1 and 2

Chapter 1 established:

- `TT_CCL` maintains three index arrays (`barrier_semaphore_idx`, `ag_semaphores_idx`, `rs_semaphores_idx`), each of length 3 (one slot per cluster-axis variant: `cluster_axis=0` at index 0, `cluster_axis=1` at index 1, `cluster_axis=None` at index 2), each cycling modulo 2.
- `get_and_cycle_ag_semaphore_handles(cluster_axis)` returns `ag_semaphore_handles[semaphore_index][current_idx]` and advances `ag_semaphores_idx[semaphore_index]` to `(current_idx + 1) % 2`, where `semaphore_index = 2 if cluster_axis is None else cluster_axis`.

  > **Note — two implementations, different behavior for `cluster_axis=0`:** The expression `2 if cluster_axis is None else cluster_axis` above is the correct form found in `models/common/modules/tt_ccl.py`. The older `models/tt_transformers/tt/ccl.py` uses `2 if not cluster_axis else cluster_axis` instead. Because `not 0` is `True` in Python, the older file maps `cluster_axis=0` to `semaphore_index=2` — the same slot used by `cluster_axis=None` — rather than to slot 0. Readers working with `models/tt_transformers/tt/ccl.py` must account for this: for `cluster_axis=0` workloads with that file, the actual `semaphore_index` used at runtime is **2**, not 0. This pre-existing bug is documented in Chapter 1.

- `get_and_cycle_barrier_semaphore_handle(cluster_axis)` follows the same pattern on `barrier_semaphore_idx`.
- The `GlobalSemaphore` object's `.address()` is stable for the object's lifetime.
- `ttnn.reset_global_semaphore_value(handle, value)` dispatches a write command that sets the device L1 semaphore word to the given value.

Chapter 2 established:

- Semaphore addresses are passed as runtime arguments (RTAs), not compile-time arguments. They are written into per-core L1 config space via `override_runtime_arguments` on every program cache hit.
- On a `MeshDevice`, the capture path assembles an immutable DRAM command buffer (`MeshTraceDescriptor::ordered_trace_data`) at `end_trace_capture` time via `assemble_dispatch_commands`. The semaphore addresses written by `override_runtime_arguments` during the capture bracket are embedded verbatim in that command buffer.
- Each `execute_trace` call issues an `add_prefetch_exec_buf` command that causes the hardware prefetcher to replay the immutable DRAM buffer wholesale. There is no per-replay rewrite of semaphore addresses.

---

## Timeline Diagram

The diagram below tracks host counter state and device semaphore state across one compile run, one trace capture, and four trace replays. `N` denotes the index value of `ag_semaphores_idx[semaphore_index]` at the start of the compile run, and `M` denotes the initial value of `barrier_semaphore_idx[semaphore_index]`.

```
OPERATION                     HOST STATE                      DEVICE STATE
                              ag_idx  barrier_idx             device L1 semaphore word
                              ------  -----------             (for the handle baked in)
─────────────────────────────────────────────────────────────────────────────────────────
[start]                       N       M                       both handles at 0 (clean)

compile run
  get_and_cycle_ag (N)        (N+1)%2     M                       handle[N] used, self-reset to 0
  get_and_cycle_barrier (M)   (N+1)%2     (M+1)%2                 barrier[M] used, self-reset to 0

  (compile run ends; device semaphores reset to 0 by kernel self-reset (compile run is synchronous))

─────────────────────────────────────────────────────────────────────────────────────────

begin_trace_capture
  override_runtime_arguments writes address of handle[(N+1)%2] and barrier[(M+1)%2]

  get_and_cycle_ag ((N+1)%2)      N    (M+1)%2               handle[(N+1)%2] snapshotted
  get_and_cycle_barrier ((M+1)%2) N    M                     barrier[(M+1)%2] snapshotted

end_trace_capture
  assemble_dispatch_commands bakes handle[(N+1)%2].address() and barrier[(M+1)%2].address()
  into the immutable DRAM command buffer.

POST-CAPTURE STATE:
  ag_semaphores_idx[si]    = N          <── does not match baked slot (N+1)%2
  barrier_semaphore_idx[si]= M          <── does not match baked slot (M+1)%2

─────────────────────────────────────────────────────────────────────────────────────────

execute_trace (replay 1)
  Hardware prefetcher replays DRAM buffer.
  RTAs contain handle[(N+1)%2].address() and barrier[(M+1)%2].address().
  Host counter: ag_idx=N, barrier_idx=M  (unchanged; no Python runs)
  Device L1: handle[(N+1)%2] and barrier[(M+1)%2] may be non-zero if kernel self-reset
             hasn't completed yet (blocking=False race): kernels reset semaphores to 0
             as their final action, but with blocking=False the host returns before that
             reset executes on device.

─────────────────────────────────────────────────────────────────────────────────────────

execute_trace (replay 2)
  Same DRAM buffer replayed verbatim.
  Same capture-time addresses: handle[(N+1)%2] and barrier[(M+1)%2].
  Device L1: handle[(N+1)%2] and barrier[(M+1)%2] may still be non-zero from replay 1's
              in-flight self-reset (blocking=False race): replay 2's kernels may begin
              incrementing the same semaphore address before replay 1's kernels have
              finished resetting it to 0. reset_global_semaphore_value before each replay
              guarantees clean state regardless of prior kernel completion.

─────────────────────────────────────────────────────────────────────────────────────────

execute_trace (replay 3) ... (replay 4) ...
  Identical to replay 2. The mismatch is permanent:
    host counter stays at N / M;
    trace always uses (N+1)%2 / (M+1)%2.
─────────────────────────────────────────────────────────────────────────────────────────
```

> **Key insight:** The host counter diverges from the trace-baked slot immediately after the first capture. Every subsequent `execute_trace` replays the same capture-time addresses regardless of the host counter's current value, because the DRAM command buffer is immutable. The counter and the trace are permanently out of phase unless explicitly corrected before each replay.

---

## Learning Objectives

After reading this chapter you will be able to answer:

1. What is the exact sequence of state changes — in both the host counter and the device semaphore word — during a trace capture that contains one `tt_all_reduce` with `use_composite=False`?
2. After capture completes, at which double-buffer slot does the host counter point, and at which slot does the trace operate?
3. Can `reduce_scatter_minimal_async` and `all_gather_async` be placed inside a `begin_trace_capture` / `end_trace_capture` bracket? Under what conditions?
4. What happens if device-side semaphore values are not reset before each replay?
5. What happens if the host counter is not managed relative to a non-traced interleaved CCL call?

---

## What's Next

Read the files in this order:

| File | Topic |
|---|---|
| [`what_gets_baked_in.md`](./what_gets_baked_in.md) | Exact state walk-through during a decode trace capture with one `tt_all_reduce`; tracking host counter before and after; showing the divergence |
| [`traceability_of_async_ccl_ops.md`](./traceability_of_async_ccl_ops.md) | Whether `reduce_scatter_minimal_async` and `all_gather_async` can be placed inside a trace bracket; the persistent output buffer non-issue; the critical semaphore management caveat |
| [`failure_modes.md`](./failure_modes.md) | What goes wrong if host counter is not corrected; what goes wrong if device semaphore values are not reset; silent corruption vs. deadlock |
