# Resetting Device Semaphore Values

By the end of this file you will understand why async CCL kernels leave device-side semaphore values in a non-zero state, why those values must be reset to 0 before each trace replay, how to enumerate the handles that need resetting, and how the CQ FIFO ordering guarantee ensures the reset reaches device L1 before the CCL kernels begin.

---

## Why kernels leave semaphores non-zero

Async CCL kernels use the global semaphore as a multi-phase coordination signal. In the reduce-scatter-minimal-async implementation, the kernels self-reset semaphore values to 0 as terminal actions before exiting. The actual RS kernels are `ring_reduce_scatter_minimal_async_reader.cpp` and `ring_reduce_scatter_minimal_async_writer.cpp`, located in `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/`. The reader resets `out_ready_sem` to 0 at line 275 (after the final batch loop iteration); the writer resets `barrier_sem` to 0 at line 226 and `batch_ready_sem` to 0 at line 479.

> **Note on file attribution:** `minimal_default_reader.cpp` and `minimal_default_writer.cpp` are the **all_gather_async** kernels, located in `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/`. They are not part of the reduce_scatter_minimal_async implementation. Any reference to those file names in the context of RS kernel behavior refers to the wrong operation.

This self-reset behavior means that under normal execution — including non-traced use and trace replay where the device-side state starts at 0 — the kernels clean up after themselves.

The problem arises at the boundary between replays. If the device-side value is 0 at the start of a replay (either because the kernel reset it as expected, or because an explicit host reset was performed), the kernel behaves correctly. If, however, the semaphore was not properly reset to 0 before a replay begins — for example, because a previous operation left it in a transient state, or because an explicit reset was skipped in the belief that the kernel's self-reset would always complete before the next replay — then the CCL kernel at the next replay reads a non-zero value immediately and skips or misinterprets its wait condition, producing silent data corruption.

> **Note:** Kernels self-reset to 0 as a terminal action. This self-reset is reliable for sequential non-traced invocations where each call completes before the next begins. Under trace replay with `blocking=False`, the host cannot safely assume the terminal reset has occurred before enqueueing the next replay. An explicit reset before each `execute_trace` removes this dependency.

---

## Which handles to reset

The handles that need resetting are exactly those whose addresses were baked into the trace at capture time. These are the handles selected by `get_and_cycle_*` during the `begin_trace_capture` / `end_trace_capture` bracket, which correspond to the pre-capture index values recorded in [Resetting Host Counters](resetting_host_counters.md).

Given `captured_ag_idx`, `captured_rs_idx`, and `captured_barrier_idx` (the snapshots taken before `begin_trace_capture`), and assuming the trace uses `cluster_axis=0` (TG all-reduce path) as the example:

```python
semaphore_index = 2  # for cluster_axis=0 — see note below

# All-gather semaphore handles at the capture-time slot (a list of 2 GlobalSemaphore objects)
ag_handles = tt_ccl.ag_semaphore_handles[semaphore_index][captured_ag_idx[semaphore_index]]
for handle in ag_handles:
    ttnn.reset_global_semaphore_value(handle, 0)

# Reduce-scatter semaphore handles at the capture-time slot (a list of 3 GlobalSemaphore objects)
rs_handles = tt_ccl.rs_semaphore_handles[semaphore_index][captured_rs_idx[semaphore_index]]
for handle in rs_handles:
    ttnn.reset_global_semaphore_value(handle, 0)

# Barrier semaphore handle at the capture-time slot (a single GlobalSemaphore object).
# For use_composite=False: only one barrier cycling call is made, so only this slot is baked.
barrier_handle = tt_ccl.barrier_semaphore_handles[semaphore_index][captured_barrier_idx[semaphore_index]]
ttnn.reset_global_semaphore_value(barrier_handle, 0)

# For use_composite=True: get_and_cycle_barrier_semaphore_handle is called TWICE —
# once for the RS op (bakes slot N) and once for the AG op (bakes slot (N+1)%2).
# Both slots are baked into the trace and both must be reset before each replay.
barrier_M_prime = captured_barrier_idx[semaphore_index]
barrier_handle_ag = tt_ccl.barrier_semaphore_handles[semaphore_index][(barrier_M_prime + 1) % 2]
ttnn.reset_global_semaphore_value(barrier_handle_ag, 0)
```

> The capture bakes two distinct barrier slot addresses; both must be reset before each replay.

> **Note on `cluster_axis=0` → `semaphore_index=2`:** The older `models/tt_transformers/tt/ccl.py` computes the index as `2 if not cluster_axis else cluster_axis`. Because `not 0` is `True` in Python, `cluster_axis=0` resolves to `semaphore_index=2` — the same slot used by `cluster_axis=None`. Only `cluster_axis=1` resolves to `semaphore_index=1`. The correct mapping is therefore:
> - `cluster_axis=None` → `semaphore_index=2`
> - `cluster_axis=0` → `semaphore_index=2` (pre-existing bug in the older file; `not 0` is `True`)
> - `cluster_axis=1` → `semaphore_index=1`
>
> The newer `models/common/modules/tt_ccl.py` uses an `is None` check and correctly maps `cluster_axis=0` to `semaphore_index=0`. See Chapter 1 for the handle array layout and Chapter 3 for the failure mode produced by resetting the wrong slot.

The counts come from the `TT_CCL.__init__` allocation in both `models/tt_transformers/tt/ccl.py` and `models/common/modules/tt_ccl.py`:

- `ag_semaphore_handles[i][slot]` is a list of 2 `GlobalSemaphore` objects — 2 reset calls per slot
- `rs_semaphore_handles[i][slot]` is a list of 3 `GlobalSemaphore` objects — 3 reset calls per slot
- `barrier_semaphore_handles[i][slot]` is a single `GlobalSemaphore` object — 1 reset call per slot for `use_composite=False`; 2 reset calls (slots N and (N+1)%2) for `use_composite=True`

If the traced model makes CCL calls on multiple cluster-axis variants (for example, both `cluster_axis=0` and `cluster_axis=1` in a 2D mesh model, or `cluster_axis=None` on T3K), the reset loop must cover each `semaphore_index` that the trace uses.

---

## Timing: enqueue before execute_trace

`ttnn.reset_global_semaphore_value` dispatches a write command into CQ0. `ttnn.execute_trace` also dispatches into CQ0. Because CQ0 is a FIFO, the reset write command is guaranteed to execute on the device before the trace replay begins. The CCL kernels inside the trace therefore always see 0 in the semaphore word when they start their wait loop.

This ordering guarantee holds specifically because both operations target the same CQ. If the model ever uses a second command queue (CQ1), a fence or barrier between queues would be required. In all current tt-transformers and tt_ccl uses, CQ0 is the single command queue, so the FIFO guarantee is sufficient.

```python
# Correct ordering: resets enqueued before execute_trace on the same CQ.

ttnn.reset_global_semaphore_value(ag_handles[0], 0)   # CQ0 write
ttnn.reset_global_semaphore_value(ag_handles[1], 0)   # CQ0 write
ttnn.reset_global_semaphore_value(rs_handles[0], 0)   # CQ0 write
ttnn.reset_global_semaphore_value(rs_handles[1], 0)   # CQ0 write
ttnn.reset_global_semaphore_value(rs_handles[2], 0)   # CQ0 write
ttnn.reset_global_semaphore_value(barrier_handle, 0)  # CQ0 write

ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
# All six reset writes are ordered before the trace replay by CQ FIFO.
```

---

## Relationship to blocking=True

With `blocking=True`, `execute_trace` does not return until the device has finished the trace and the kernels have performed their self-resets. The device-side semaphore value is 0 before the next call to `execute_trace`. In that case, the explicit resets described here are harmless redundancy: they enqueue six writes that arrive at the device, find the semaphore already at 0, and write 0 again with no effect.

With `blocking=False` — the standard production mode in the tt-transformers generator — the next `execute_trace` call may be enqueued while the device is still executing the previous replay. The explicit resets before each replay ensure that the reset writes are ordered before the next replay's kernels regardless of how far ahead the host gets. This is the only correct behavior under `blocking=False`.

---

## What happens if this step is skipped

If the device-side reset is skipped before a replay under `blocking=False`:

- If the previous replay's kernels have already self-reset the semaphore to 0 by the time the new replay's CCL kernels begin their wait, the replay succeeds. This is a timing race that may appear to work in testing but is not guaranteed.
- If the new replay's CCL kernels begin before the previous replay's terminal self-reset has occurred, the kernel reads a non-zero value immediately, skips its wait, and proceeds to the next phase using stale or partial data. The result is silent numerical corruption rather than a hang, because the kernel does not deadlock — it just uses the wrong data.

This failure mode (described in Chapter 3, `failure_modes.md`) is particularly difficult to diagnose because it produces incorrect outputs rather than a crash or timeout.

---

**Next:** [Structuring the Capture](structuring_the_capture.md)
