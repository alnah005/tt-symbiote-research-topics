# Failure Modes: What Goes Wrong and Why

This file catalogs the concrete failure modes that arise when the host-counter / trace-handle mismatch is not corrected. Two distinct problems are possible: host counter misalignment (which affects non-traced CCL calls that run alongside traced replays) and device-side semaphore value accumulation (which affects every replay). Each problem has a different symptom — silent data corruption in one case, and a hang or incorrect result in the other. By the end of this file you will know how to recognize each failure mode and why it arises from the mechanisms described in the preceding two files.

---

## Problem 1: Host Counter Not Corrected

### What the state looks like

After trace capture completes, the host counter points to the slot opposite to the one baked into the trace. Using the notation from [`what_gets_baked_in.md`](what_gets_baked_in.md):

- Trace-baked AG slot: `N'` = `(N+1) % 2`
- Host counter value: `ag_semaphores_idx[semaphore_index]` = `N`
- Trace-baked barrier slot: `M'` = `(M+1) % 2`
- Host counter value: `barrier_semaphore_idx[semaphore_index]` = `M`

Every `execute_trace` uses slot `N'` and slot `M'`. No Python code runs during replay, so the host counter does not advance.

### Case A: Purely traced decode loop with no interleaved non-traced CCL calls

If the decode loop consists entirely of `execute_trace` calls with no non-traced CCL calls between them, the host counter mismatch has no effect on the trace itself. The trace always uses its capture-time handles, and nothing else is competing for those handles. The only observable consequence is that the host counter is in the wrong state after the decode loop ends — if a subsequent non-traced call is made later, it will use the wrong starting slot.

> **Note:** In practice, a purely traced decode loop with no non-traced CCL calls is unusual. Prefill, sampling, and embedding update steps that are not part of the trace may each invoke CCL ops, and those invocations will advance the host counter and potentially collide with the trace's handles.

### Case B: Interleaved non-traced CCL call (e.g., prefill between two replays)

This is the dangerous case. Suppose the host counter is at `N` after trace capture. A prefill step, which is not traced, calls `tt_all_reduce`. The call invokes `get_and_cycle_ag_semaphore_handles(cluster_axis)`, which:

1. Returns `ag_semaphore_handles[semaphore_index][N]` — the slot opposite to what the trace uses.
2. Advances `ag_semaphores_idx[semaphore_index]` to `N'`.

The prefill CCL op now runs on slot `N`. A subsequent `execute_trace` runs on slot `N'`. These are different handles, so there is no immediate collision. But the host counter is now at `N'`, and the next non-traced CCL call will return slot `N'` — the same handle the trace uses.

The collision pattern depends on the calling sequence:

```
trace replay 1   → uses slot N'
non-traced call  → uses slot N    (counter advances to N')
trace replay 2   → uses slot N'
non-traced call  → uses slot N'   (non-traced kernel dispatched to slot N' — same slot as trace; collision)
```

> **Note:** This Case B failure is not fixed by `reset_global_semaphore_value` alone — device semaphore resets address Problem 2 but the handle collision at step 4 requires host-counter correction (Chapter 4).

After two replay-plus-call cycles, the non-traced calls and the trace replays are using the same handle concurrently. If both are dispatched to the same command queue in rapid succession, the non-traced call's kernel may observe a semaphore value that was intended as the trace kernel's completion signal, or vice versa. The result is data corruption — the all-gather or reduce-scatter produces the wrong output — without any hang or error message.

> **Warning:** The collision between a non-traced CCL call and a trace replay is not detectable at the API level. Both calls dispatch to the same command queue and complete without error. The corruption manifests only in the numerical output values.

### Case C: Prefill that itself calls multiple CCL ops

If a non-traced prefill step calls `tt_all_reduce` more than once (for example, one per transformer layer), each call advances the host counter by one step. After an even number of non-traced calls, the counter is back at `N` — safe. After an odd number, the counter is at `N'` — dangerous. The number of CCL calls per prefill step therefore determines whether the non-traced calls land on the safe or dangerous handle.

This makes the failure intermittent with respect to model depth and batch size: it may appear or disappear as the number of transformer layers or attention heads changes, which is a hallmark of subtle semaphore-state bugs.

---

## Problem 2: Device-Side Semaphore Values Not Reset Before Replay

### What the kernel expects

Async CCL kernels coordinate via a semaphore protocol. The exact protocol depends on the kernel variant, but the general pattern is:

- The reader kernel initializes a local variable `sem_target = 0`. Before each data movement phase, it calls `noc_semaphore_wait_min(..., out_ready_sem, sem_target + 1)`, blocking until `out_ready_sem` reaches at least `sem_target + 1`. On the first wait, this means waiting for the value to be at least 1; on the second wait, at least 2; and so on. The writer kernel signals readiness by sending NOC atomic increment writes to the reader's `out_ready_sem` address, incrementing the accumulated count.
- The reader kernel's terminal action is to reset `out_ready_sem` to 0: `noc_semaphore_set(..., out_ready_sem, 0)`. This resets the accumulator so that the next use begins with the semaphore at 0 and the first wait again checks for at least 1.

The semaphore is designed to start at 0. The kernel never waits for 0 — 0 is the starting value before the protocol runs and the value after the terminal reset, but not a wait target. `reset_global_semaphore_value(handle, 0)` before each replay pre-resets the semaphore so that the next replay's reader begins with `sem_target = 0` and correctly waits for the writer's first increment to reach 1.

With `blocking=False`, if the second replay is dispatched before the first replay's reader has executed its terminal reset, the second replay's writer sends atomic increments to `out_ready_sem` while it may still hold a non-zero value (because the first replay's reader has not yet zeroed it). The accumulated count becomes at least 1, which satisfies the second replay's reader's first wait condition immediately — a skip-through. This skip-through occurs not because of a stale value from a prior completed replay, but because overlapping execution means the semaphore was never properly reset between replays.

### What happens after the first replay and the correct requirement

The `all_gather_async` reader kernel (`minimal_default_reader.cpp` line 295) resets `out_ready_sem` to 0 as its final action; the writer similarly resets `barrier_sem` to 0. Both async CCL kernels self-reset their semaphores to 0 when they run to completion. With `blocking=True` the host waits for completion before returning, so those self-resets are done before the next dispatch — an explicit `reset_global_semaphore_value` is not strictly required in that case. With `blocking=False` — the common production path — the host may dispatch the next `execute_trace` before the prior replay's kernels have reached their self-reset. Because commands in the CQ FIFO are consumed in order, an explicit `ttnn.reset_global_semaphore_value` enqueued before `execute_trace` guarantees the reset write reaches the cores before the next replay's CCL kernels begin reading, regardless of whether the prior kernels have finished their self-reset. The DRAM command buffer and `execute_trace` itself do not reset device L1; the caller-enqueued reset is the only mechanism that guarantees clean state for `blocking=False` replays.

### The skip-through failure mode

The practical effect of a missing reset depends on the exact kernel logic:

- If the reader waits for the semaphore to reach a specific count before treating a buffer as available, a spurious increment from the second writer's replay may satisfy that wait condition prematurely. The reader proceeds to its data movement phase without waiting for the actual data to be in the expected state — silent data corruption.
- If the kernel uses the semaphore as a counter that accumulates across phases (writing `ring_size` per phase and checking for multiples), the extra increment from the overlapping second replay shifts the expected threshold. The kernel may complete a phase prematurely, producing incorrect results.
- In some configurations, the collision may cause a kernel to wait for a count that is never reached (because one side consumed the increment out of order), producing a hang.

> **Warning:** Whether the failure manifests as silent corruption or a hang depends on the specific kernel logic and the timing of the overlap. Both outcomes are possible from the same root cause (missing reset before `blocking=False` replays). Silent corruption is the harder failure to diagnose because the op completes successfully and returns a tensor, but the tensor contains wrong values.

### Which handles must be reset

For `tt_all_reduce` with `use_composite=False` on cluster axis `a`, the handles that must be reset before each replay are:

- All handles in the list `ag_semaphore_handles[a][N']` (a list of 2 `GlobalSemaphore` objects).
- `barrier_semaphore_handles[a][M']` (a single `GlobalSemaphore` object).

where `N'` and `M'` are the capture-time slot indices (the values that were current during the `begin_trace_capture` / `end_trace_capture` bracket).

> **Note:** For `tt_all_reduce` with `use_composite=True`, the path calls two async ops in sequence — `reduce_scatter_minimal_async` followed by `all_gather_async` — each consuming its own `get_and_cycle_*` calls. The rs and ag index variables advance independently, so the capture-time slot indices for the two ops are tracked separately. Let `rs_N'` be the capture-time rs slot index, `barrier_M'_first` the barrier slot consumed by reduce_scatter, `ag_N'` the capture-time ag slot index, and `barrier_M'_second` the barrier slot consumed by all_gather. All four handle groups are embedded as RTAs in the immutable DRAM command buffer and must be reset before each replay:
>
> - `rs_semaphore_handles[a][rs_N']` — a list of 3 `GlobalSemaphore` objects (for reduce_scatter)
> - `barrier_semaphore_handles[a][barrier_M'_first]` — the barrier handle used by reduce_scatter
> - `ag_semaphore_handles[a][ag_N']` — a list of 2 `GlobalSemaphore` objects (for all_gather)
> - `barrier_semaphore_handles[a][barrier_M'_second]` — the barrier handle used by all_gather
>
> Resetting only the rs and first barrier handles while omitting `ag_semaphore_handles` and the second barrier handle leaves the all_gather kernel unprotected: a `blocking=False` race on the omitted handles can cause the all_gather writer to collide with the prior replay's reader, producing silent data corruption on every replay after the first.

---

## Summary of Failure Modes

| Condition | Symptom | Category |
|---|---|---|
| Host counter not corrected; no non-traced CCL calls interleaved | No immediate failure; host counter is wrong for any post-loop non-traced call | Latent defect |
| Host counter not corrected; non-traced CCL calls interleaved after an odd number of cycles | Non-traced call and trace replay share the same handle; semaphore signals corrupt each other | Silent data corruption |
| Device semaphore values not reset before replay | With `blocking=False`, overlapping replays cause skip-through: second reader's `noc_semaphore_wait_min` satisfied immediately by first reader's not-yet-complete self-reset | Silent data corruption or hang |
| Both problems present simultaneously | Compound: handle collision and stale semaphore values; unpredictable | Silent data corruption or hang |

Chapter 4 describes the concrete remediation steps for both problems: how to reset device semaphore values before each replay, how to restore host counter values to their capture-time state, and how to structure the capture so that device state at capture time matches device state at replay time.

---

**Next chapter:** [Chapter 4 — Correct Synchronization Strategies for Traced Async CCL](../ch4_synchronization_strategies/index.md)
