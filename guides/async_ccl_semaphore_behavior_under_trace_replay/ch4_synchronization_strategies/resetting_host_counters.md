# Resetting Host Counters

By the end of this file you will understand why the host-side cycling counters inside `TT_CCL` must be restored to their capture-time values before each `execute_trace` call, how to determine what those capture-time values are, and what bug is produced if this step is skipped.

---

## The problem in one sentence

When `end_trace_capture` is called, the `get_and_cycle_*` methods have already advanced each index by one step beyond the capture-time starting values. Every subsequent replay reuses the capture-time semaphore addresses, but the host counters now point to the opposite double-buffer slot. Any non-traced CCL call that happens after the trace replay — for example, a prefill step that is not traced, or a sampling op that calls into the same `tt_all_gather` path — will select the wrong handle and, if a replay is still in flight, will collide with the handles the trace is actively using.

---

## Identifying the capture-time index values

The key observation is that `get_and_cycle_*` first returns the current index and then advances it. Therefore, the capture-time handle selected by the first `get_and_cycle_ag_semaphore_handles(cluster_axis)` call inside the trace bracket is the value of `ag_semaphores_idx[semaphore_index]` at the moment `begin_trace_capture` was called, not after it.

The correct procedure is:

1. Before calling `begin_trace_capture`, record the current values of all index fields that the traced forward pass touches.
2. Run the capture (`begin_trace_capture` → model forward → `end_trace_capture`). The `get_and_cycle_*` calls inside the bracket advance each index by one step.
3. Store the recorded (pre-capture) index values as the set to restore before each replay.

For a decode trace that touches `cluster_axis=0` (for `tt_all_reduce`) and `cluster_axis=None` (for single-axis `tt_all_gather`), the relevant fields in `models/tt_transformers/tt/ccl.py` (the older `TT_CCL`) are:

- `barrier_semaphore_idx[0]` and `barrier_semaphore_idx[2]`
- `ag_semaphores_idx[0]` and `ag_semaphores_idx[2]`
- `rs_semaphores_idx[0]` and `rs_semaphores_idx[2]`

For the newer `models/common/modules/tt_ccl.py` (same field names, same layout), the same fields apply.

---

## Code pattern: snapshot before capture, restore before each replay

```python
# --- Before begin_trace_capture ---

# Snapshot index fields for every cluster-axis variant that the traced model uses.
# Adjust the list of (semaphore_index) values to match what the model actually calls.
captured_barrier_idx = list(tt_ccl.barrier_semaphore_idx)   # copy, not reference
captured_ag_idx      = list(tt_ccl.ag_semaphores_idx)
captured_rs_idx      = list(tt_ccl.rs_semaphores_idx)

trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
tt_out_trace = model.ttnn_decode_forward(...)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

# --- Before each execute_trace ---

# Step 1: Restore index fields so the host counter matches the capture-time state.
tt_ccl.barrier_semaphore_idx = list(captured_barrier_idx)
tt_ccl.ag_semaphores_idx     = list(captured_ag_idx)
tt_ccl.rs_semaphores_idx     = list(captured_rs_idx)

# Step 2: Reset device semaphore values (see resetting_device_semaphore_values.md).

# Step 3: Execute the trace.
ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
```

Neither `TT_CCL` in `models/tt_transformers/tt/ccl.py` nor the one in `models/common/modules/tt_ccl.py` currently provides a `snapshot_semaphore_indices` or `restore_semaphore_indices` method; the pattern above must be implemented at the call site (in `generator.py` or a model-specific trace wrapper) until such methods are added. Chapter 5 describes the recommended approach for adding them.

---

## Why this step is necessary even when device-side semaphore values are reset

Device-side semaphore resets (issuing `ttnn.reset_global_semaphore_value` before `execute_trace`) ensure that the kernels baked into the trace start from a clean L1 state. They do not touch the host-side index fields.

If the host index is left at `(N+1) % 2` after the capture, the trace replay itself is unaffected — the replay dispatches the command stream snapshot, which always uses handle `N`. But any call to `get_and_cycle_ag_semaphore_handles(cluster_axis)` that occurs after the replay returns handle `(N+1) % 2` — the slot that was not reset before the replay and that the trace did not use. If the replay is still executing (because it was launched with `blocking=False`), handle `(N+1) % 2` now has an unknown device-side state, and a non-traced CCL op using it may produce incorrect results.

Resetting the host counter eliminates this divergence so that post-replay non-traced calls and the next replay both operate on the same handle, under the same device-side reset invariant. Even when no non-traced CCL calls are currently interleaved, resetting the host counter costs nothing and is the correct invariant to maintain for forward compatibility: future changes to the model or the generator that introduce non-traced CCL calls will not silently break.

---

**Next:** [Resetting Device Semaphore Values](resetting_device_semaphore_values.md)
