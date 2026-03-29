# Structuring the Capture

By the end of this file you will understand why the capture run itself must start with the same device-side semaphore state as each future replay, how to structure the three phases (compile run, capture run, repeated replay) so that host counter state and device L1 state are consistent at every transition, and what goes wrong if the device semaphores are reset only before replays but not before the capture.

---

## Why the capture run's initial device state matters

The `TraceNode` stores a snapshot of RTA data — specifically, the L1 addresses of the semaphore handles baked in at `end_trace_capture` time. It does not store any snapshot of device L1 semaphore word values. The trace replay mechanism restores RTAs but does not restore L1 state.

This means: if the capture run executes with the device semaphore word at some value V (because a previous operation left it non-zero), and the replay subsequently starts with the device semaphore word at 0 (because the replay pre-reset was performed), the kernel behavior during capture and during replay differs. Whether this difference produces incorrect results depends on the specific kernel logic, but the correct approach is to make the initial device state for the capture run match the initial device state for every replay: all capture-time semaphore words at 0 before `begin_trace_capture`.

---

## Recommended checklist

The following numbered steps form a complete procedure for integrating async CCL semaphore management into a trace capture and replay workflow.

### Phase 1: Compile run (warm-up / program cache build)

1. Before the compile run, call `ttnn.reset_global_semaphore_value(handle, 0)` for every `GlobalSemaphore` handle that the model will use — not just the capture-time slots, but all double-buffer slots for all axis variants that the model touches. This ensures the compile run starts from a fully clean state.

2. Run the compile-run forward pass. This populates the program cache, triggers `override_runtime_arguments` on all ops, and advances the `TT_CCL` index fields by one step per CCL op.

3. After the compile run, note the current index field values. These are the post-compile-run state; the capture will begin from here.

### Phase 2: Capture run

4. Reset all semaphore device values to 0 again — the same full reset as step 1. This ensures the capture run and every future replay begin from the same device state.

   > **Key insight:** Resetting only the handles that the trace will use (the capture-time slots) is not sufficient at this stage, because you do not yet know which slots those are. Reset all handles to be safe, or reset only the slots at the current index values before incrementing.

5. Record the current `TT_CCL` index field values immediately before `begin_trace_capture`. These are the capture-time starting indices:

   ```python
   captured_barrier_idx = list(tt_ccl.barrier_semaphore_idx)
   captured_ag_idx      = list(tt_ccl.ag_semaphores_idx)
   captured_rs_idx      = list(tt_ccl.rs_semaphores_idx)
   ```

6. Call `ttnn.begin_trace_capture(mesh_device, cq_id=0)`.

7. Run the model forward pass. Each `get_and_cycle_*` call inside the bracket selects the handle at the pre-capture index and advances the index. `override_runtime_arguments` writes the L1 addresses of the selected handles into the RTA slots.

8. Call `ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)`. This snapshots the current RTA data — including the semaphore addresses selected in step 7 — into the `TraceNode`. The index fields are now at `(captured_*_idx + 1) % 2` for each axis variant used.

9. The `captured_*_idx` values recorded in step 5 are the authoritative restore targets. Store them alongside `trace_id`.

### Phase 3: Each replay

10. Restore the `TT_CCL` index fields to the capture-time values:

    ```python
    tt_ccl.barrier_semaphore_idx = list(captured_barrier_idx)
    tt_ccl.ag_semaphores_idx     = list(captured_ag_idx)
    tt_ccl.rs_semaphores_idx     = list(captured_rs_idx)
    ```

11. For each cluster-axis variant used in the trace, and for each semaphore type, call `ttnn.reset_global_semaphore_value(handle, 0)` for every `GlobalSemaphore` in the capture-time slot:

    - For each active `semaphore_index` (2 for `cluster_axis=0`, 1 for `cluster_axis=1`, 2 for `cluster_axis=None`) — see note below:
      - For the `ag_semaphore_handles[semaphore_index][captured_ag_idx[semaphore_index]]` list (2 handles): 2 calls
      - For the `rs_semaphore_handles[semaphore_index][captured_rs_idx[semaphore_index]]` list (3 handles): 3 calls
      - For barrier handles: **2 handles (use_composite=True): 2 calls** — one for the RS op (`barrier_semaphore_handles[semaphore_index][captured_barrier_idx[semaphore_index]]`, slot N) and one for the AG op (`barrier_semaphore_handles[semaphore_index][(captured_barrier_idx[semaphore_index]+1)%2]`, slot (N+1)%2). For `use_composite=False`, only 1 barrier cycling call is made, so only slot N is baked and only 1 reset call is needed.

12. Call `ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)`.

13. Repeat steps 10–12 for every decode step.

> **Note on `cluster_axis=0` → `semaphore_index=2`:** See the equivalent note in [`resetting_device_semaphore_values.md`](./resetting_device_semaphore_values.md).

---

## Why step 4 (reset before capture) is required, not just step 11 (reset before replay)

Suppose the compile run (step 2) consumed one CCL slot (the run advanced `ag_semaphores_idx` from 0 to 1 for the relevant axis). The capture run (step 7) uses slot 1 (index at 1 when `begin_trace_capture` was called, if the index was not separately managed). If at that point the device-side value for slot 1 is non-zero — left by the compile run's `all_gather_async` kernel — then the capture run's kernel skips or shortens its wait, producing a different dispatch pattern than will occur during replay (where the device reset in step 11 ensures the kernel sees 0).

The replay will execute the same command sequence that was captured, but the device-side conditions at kernel start differ between capture and replay. Depending on the specific kernel logic, this can produce silent differences in output. The safe invariant is: reset all handles before the capture run so that the capture-run kernel behavior matches the replay-run kernel behavior.

---

## Summary table

| Phase | Host index action | Device semaphore action |
|---|---|---|
| Before compile run | (no special action needed) | Reset all handles for all axis variants to 0 |
| After compile run, before capture | Record current index values as `captured_*_idx` | Reset all handles to 0 |
| During capture (`begin_trace_capture` to `end_trace_capture`) | Indices advance normally via `get_and_cycle_*` | Kernels run, self-reset to 0 at completion |
| Before each `execute_trace` | Restore `TT_CCL.*_idx` to `captured_*_idx` | Reset capture-time handles to 0 via `reset_global_semaphore_value` |

---

**Next:** [Existing Patterns in tt-transformers](existing_patterns_in_tt_transformers.md)
