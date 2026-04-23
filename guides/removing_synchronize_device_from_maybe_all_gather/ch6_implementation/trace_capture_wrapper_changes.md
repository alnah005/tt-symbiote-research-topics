# Trace Capture Wrapper Changes

This file describes the changes needed in the `TracedRun` capture and replay logic to account for the cycling semaphore indices introduced by the Type B2 changes in [`structural_changes.md`](./structural_changes.md). By the end of this file you will have a numbered checklist that can serve as a code-review reference for every capture-and-replay bracket that covers `_maybe_all_gather` with `all_gather_async`.

> **Note:** Type A (delete `synchronize_device` only, keep synchronous `ttnn.all_gather`) requires no changes to the trace capture wrapper. The content in this file applies exclusively to the Type B2 path.

---

## Why Wrapper Changes Are Necessary

When `all_gather_async` is called during trace capture, the specific `GlobalSemaphore` handle objects whose L1 device addresses were live at capture time are **baked into the trace buffer**. The trace stores the device-side L1 address of each semaphore, not a symbolic reference. At replay time, the Metal Trace hardware replays exactly those baked-in addresses.

The cycling semaphore mechanism in `TT_CCL` advances an integer index (`ag_semaphores_idx`, `barrier_semaphore_idx`) on each call to `get_and_cycle_ag_semaphore_handles` or `get_and_cycle_barrier_semaphore_handle`. After a trace capture that contains N calls to `all_gather_async`, the index has advanced by N positions. On the next `execute_trace` call, if the index is not restored to its pre-capture value, `get_and_cycle_ag_semaphore_handles` will select the **wrong** semaphore handle — one whose L1 address does not match what is baked into the trace. The async CCL kernel will wait on the wrong semaphore and either deadlock or complete without correct synchronization.

The fix is a snapshot-and-restore protocol: record the index values before capture, then restore them before each `execute_trace`.

---

## Identifying the Correct cluster_axis

Before implementing the snapshot-and-restore protocol, identify which `cluster_axis` value `_maybe_all_gather` uses. This determines which slot within the `TT_CCL` semaphore arrays must be snapshotted and restored.

From [Chapter 1, `call_sites_and_control_flow.md`](../ch1_maybe_all_gather_anatomy/call_sites_and_control_flow.md), the `cluster_axis` passed to `_maybe_all_gather` at each call site determines the axis:

- `cluster_axis=1` is the typical value for T3K (ring along the 8-device axis).
- If the module uses a different axis for the all_gather than the model's outer reduce_scatter uses, the two may require independent index tracking within `TT_CCL`.

> **Warning:** Do not assume that `_maybe_all_gather`'s `cluster_axis` matches the model's outer reduce_scatter axis. They may differ. Verify the axis values at both call sites (in `TTNNQwen3FullAttention.forward` and `TTNNQwen3LinearAttention.forward`) before implementing the snapshot logic.

---

## Checklist: Pre-Capture Steps

Perform these steps **before calling `ttnn.begin_trace_capture`**:

1. **Record the current `ag_semaphores_idx` for the `cluster_axis` used by `_maybe_all_gather`.**

   ```python
   # Snapshot the cycling index before capture
   ag_idx_snapshot = self.tt_ccl.ag_semaphores_idx[cluster_axis]
   barrier_idx_snapshot = self.tt_ccl.barrier_semaphore_idx[cluster_axis]
   ```

2. **Obtain the capture-time semaphore handles** (the handles that will be selected during capture).

   ```python
   # These handles will be cycled during capture; record them before the index advances
   capture_ag_handles = self.tt_ccl.ag_semaphore_handles[cluster_axis][ag_idx_snapshot % 2]
   capture_barrier_handle = self.tt_ccl.barrier_semaphore_handles[cluster_axis][barrier_idx_snapshot % 2]
   ```

3. **Reset the capture-time semaphore values to zero** before any device ops in the capture bracket read them.

   ```python
   for handle in capture_ag_handles:
       ttnn.reset_global_semaphore_value(handle, 0)
   ttnn.reset_global_semaphore_value(capture_barrier_handle, 0)
   ```

4. **Call `ttnn.begin_trace_capture`.** The next call to `_maybe_all_gather` inside the capture bracket will call `get_and_cycle_ag_semaphore_handles(cluster_axis)`, which selects the handle at the current (pre-snapshot) index and advances the index. The selected handle's L1 address is baked into the trace.

---

## Checklist: Post-Capture Steps

Perform these steps **after calling `ttnn.end_trace_capture`**:

5. **Store the snapshot values as the replay restore point.** The indices have advanced during capture; the restore point is the pre-capture values recorded in step 1.

   ```python
   # Store as instance fields or as part of the TracedRun state
   self._replay_ag_idx_restore = ag_idx_snapshot
   self._replay_barrier_idx_restore = barrier_idx_snapshot
   ```

---

## Checklist: Pre-Replay Steps (before each `execute_trace`)

Perform these steps **before every call to `ttnn.execute_trace`**, including the first replay:

6. **Restore the `TT_CCL` index fields to the pre-capture snapshot values.**

   ```python
   self.tt_ccl.ag_semaphores_idx[cluster_axis] = self._replay_ag_idx_restore
   self.tt_ccl.barrier_semaphore_idx[cluster_axis] = self._replay_barrier_idx_restore
   ```

7. **Re-identify the capture-time handles** using the restored index values.

   ```python
   replay_ag_handles = self.tt_ccl.ag_semaphore_handles[cluster_axis][
       self._replay_ag_idx_restore % 2
   ]
   replay_barrier_handle = self.tt_ccl.barrier_semaphore_handles[cluster_axis][
       self._replay_barrier_idx_restore % 2
   ]
   ```

8. **Reset the semaphore values to zero** so the baked-in `all_gather_async` completion wait starts from a clean state.

   ```python
   for handle in replay_ag_handles:
       ttnn.reset_global_semaphore_value(handle, 0)
   ttnn.reset_global_semaphore_value(replay_barrier_handle, 0)
   ```

9. **Call `ttnn.execute_trace`.** The trace replays with the correct semaphore L1 addresses from step 4, and the device-side completion semaphores are in a known-zero initial state from step 8.

---

## Condensed Checklist (for Code Review)

```
PRE-CAPTURE:
  [1] ag_idx_snapshot        = tt_ccl.ag_semaphores_idx[cluster_axis]
  [2] barrier_idx_snapshot   = tt_ccl.barrier_semaphore_idx[cluster_axis]
  [3] reset_global_semaphore_value(capture_ag_handles[*], 0)
  [4] reset_global_semaphore_value(capture_barrier_handle, 0)
  [5] ttnn.begin_trace_capture(...)

POST-CAPTURE:
  [6] ttnn.end_trace_capture(...)
  [7] store ag_idx_snapshot, barrier_idx_snapshot as replay restore point

PRE-REPLAY (repeat before every execute_trace):
  [8]  tt_ccl.ag_semaphores_idx[cluster_axis]      = ag_idx_snapshot
  [9]  tt_ccl.barrier_semaphore_idx[cluster_axis]  = barrier_idx_snapshot
  [10] reset_global_semaphore_value(replay_ag_handles[*], 0)
  [11] reset_global_semaphore_value(replay_barrier_handle, 0)
  [12] ttnn.execute_trace(...)
```

---

## Relationship to the async_ccl_semaphore_behavior_under_trace_replay Guide

This protocol is analogous to the snapshot-and-restore pattern documented in the `async_ccl_semaphore_behavior_under_trace_replay` guide for `Attention` in `models/tt_transformers/tt/attention.py`. The new element specific to `_maybe_all_gather` is that its `cluster_axis` may differ from the axis used by the model's outer reduce_scatter operation. Where that guide snapshots a single axis's indices, the wrapper here must independently snapshot and restore indices for each distinct `cluster_axis` value appearing in the trace bracket.

If the trace bracket covers both:
- The model's outer reduce_scatter (`cluster_axis=0` on a 2D mesh, for example), and
- `_maybe_all_gather`'s all_gather (`cluster_axis=1`),

then **both** sets of indices must be snapshotted and restored independently. Refer to the `async_ccl_semaphore_behavior_under_trace_replay` guide for the outer reduce_scatter handling, and use the checklist above only for the `_maybe_all_gather` semaphores.

> **Warning:** Applying the restore for the wrong `cluster_axis` (e.g., restoring axis 0's index when the trace baked in axis 1's handles) will cause the trace to present a stale or wrong semaphore address on replay, resulting in a deadlock or silent numerical error. Always verify the `cluster_axis` values at each call site before implementing the restore logic.

---

## Validation

After implementing the wrapper changes, run the multi-replay stability test described in [Chapter 7, `multi_replay_stability.md`](../ch7_validation/multi_replay_stability.md). A correct implementation produces bit-identical or near-identical outputs across N≥10 consecutive trace replays. A deadlock on replay 2 or later indicates that the semaphore reset (steps 10–11 above) was not executed before `execute_trace`.
