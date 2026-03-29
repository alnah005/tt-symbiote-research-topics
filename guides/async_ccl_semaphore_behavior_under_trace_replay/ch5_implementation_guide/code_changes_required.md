# Code Changes Required for Trace-Compatible Async CCL

This file contains the concrete code changes needed to make async CCL operations safe inside a tt-metal mesh trace. The changes fall into four categories: adding index snapshot/restore helpers so the host-side `TT_CCL` state is deterministic across replays; adding device semaphore reset calls so the on-device counters are zero before each `execute_trace`; restructuring the trace capture site to interleave those two concerns correctly; and applying those same structures at every trace replay site. The four files in the codebase that currently lack these guards — `models/tt_transformers/tt/generator.py` (TG text and TG vision paths), `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`, and `models/demos/deepseek_v3/tt/ccl.py` — all need the patterns described here.

---

## Index Snapshot and Restore Helpers

### Why this is necessary

During trace capture, `get_and_cycle_ag_semaphore_handles`, `get_and_cycle_rs_semaphore_handles`, and `get_and_cycle_barrier_semaphore_handles` each advance their respective index arrays by one position modulo 2. After `end_trace_capture` returns, those indices point to the slot *after* the one whose L1 addresses were baked into the DRAM command buffer. Any subsequent non-traced CCL call — or the next `execute_trace` call itself when the surrounding Python logic re-invokes `get_and_cycle_*` — will consume the wrong handle slot and write to a different L1 address than the one the trace kernel expects.

The fix is to freeze the capture-time index values in Python variables immediately before `begin_trace_capture`, and then write those values back into `tt_ccl` before every `execute_trace`.

### Snapshot (taken once, before `begin_trace_capture`)

```python
# Capture host indices at the moment the trace will consume them.
# These are lists of length 3 (one entry per semaphore_index slot).
captured_ag_idx      = list(tt_ccl.ag_semaphores_idx)
captured_rs_idx      = list(tt_ccl.rs_semaphores_idx)
captured_barrier_idx = list(tt_ccl.barrier_semaphore_idx)
```

> **Note:** Take the snapshot *before* resetting device semaphores and *before* calling `begin_trace_capture`. The device reset uses the captured values to know which handles to reset (see the next section), so the snapshot must come first.

### Restore (repeated before every `execute_trace`)

```python
# Restore host indices to the values they held at capture time.
# This must happen before execute_trace, not after.
tt_ccl.ag_semaphores_idx      = list(captured_ag_idx)
tt_ccl.rs_semaphores_idx      = list(captured_rs_idx)
tt_ccl.barrier_semaphore_idx  = list(captured_barrier_idx)
```

> **Warning:** Do not use `= captured_ag_idx` (without `list(...)`). That would share the same list object, and any later `get_and_cycle_*` call that mutates `tt_ccl.ag_semaphores_idx` in-place would also corrupt your snapshot variables. Always copy.

---

## Device Semaphore Reset Before Each Replay

### Why this is necessary

The DRAM command buffer baked during trace capture encodes the L1 addresses of the handles that were current at capture time. On each replay, the kernels at those addresses read their semaphore values to determine whether they should proceed or wait. If a previous replay's kernels have not yet written their self-reset (possible with `blocking=False`) or if no reset was ever enqueued from the host, the kernels see a non-zero counter and either deadlock (if they are waiting for a zero) or skip an expected synchronization barrier.

`ttnn.reset_global_semaphore_value(handle, 0)` dispatches a write command through the CQ FIFO. Because the CQ FIFO is ordered, any reset enqueued before `execute_trace` is guaranteed to complete on device before the trace begins executing.

### Determining the semaphore index

The `semaphore_index` (`si`) that a model uses is determined at CCL construction time by `cluster_axis`:

| `cluster_axis` value | `models/tt_transformers/tt/ccl.py` (older) | `models/common/modules/tt_ccl.py` (newer) |
|---|---|---|
| `None` | `2` (because `not None` is `True`) | `2` (explicit `None` check) |
| `0` | `2` (because `not 0` is `True`) | `0` |
| `1` | `1` | `1` |

> **Warning:** In the older `models/tt_transformers/tt/ccl.py`, `cluster_axis=0` maps to `semaphore_index=2`, not 0, because the guard is `2 if not cluster_axis else cluster_axis` and `not 0` evaluates to `True`. Verify which file your model's `TT_CCL` is instantiated from before writing reset loops. Using the wrong `si` silently resets the wrong handle, leaving the correct one with a stale value.

### Reset for `use_composite=False` (AG or RS alone, `cluster_axis=None`)

When the traced operation is a standalone `all_gather_async` or `reduce_scatter_minimal_async` on `cluster_axis=None`, the trace consumed one AG handle group and one barrier handle. The semaphore index is `si = 2`.

```python
si = 2  # semaphore_index for cluster_axis=None

ag_idx      = captured_ag_idx[si]
barrier_idx = captured_barrier_idx[si]

# Reset the 2 AG semaphore handles (one per ring direction).
for ag_sem in tt_ccl.ag_semaphore_handles[si][ag_idx]:
    ttnn.reset_global_semaphore_value(ag_sem, 0)

# Reset the 1 barrier semaphore handle.
ttnn.reset_global_semaphore_value(
    tt_ccl.barrier_semaphore_handles[si][barrier_idx], 0
)
```

> **Note:** `ag_semaphore_handles[si][ag_idx]` is a list of 2 `GlobalSemaphore` objects — one for each direction in the ring. Both must be reset. `barrier_semaphore_handles[si][barrier_idx]` is a single object, not a list.

### Reset for `use_composite=True` (`tt_all_reduce`, `cluster_axis=None`)

`tt_all_reduce` with `use_composite=True` makes four sequential cycling calls during the traced region:

1. `get_and_cycle_rs_semaphore_handles(si)` — consumes RS slot at `rs_idx`, advances to `(rs_idx + 1) % 2`
2. `get_and_cycle_barrier_semaphore_handles(si)` — consumes barrier slot at `barrier_idx`, advances to `(barrier_idx + 1) % 2`
3. `get_and_cycle_ag_semaphore_handles(si)` — consumes AG slot at `ag_idx`, advances to `(ag_idx + 1) % 2`
4. `get_and_cycle_barrier_semaphore_handles(si)` — consumes barrier slot at `(barrier_idx + 1) % 2`, advances to `(barrier_idx + 2) % 2`

All four groups are baked into the DRAM command buffer. All four must be reset before each replay.

```python
si = 2  # semaphore_index for cluster_axis=None

rs_idx         = captured_rs_idx[si]
ag_idx         = captured_ag_idx[si]
barrier_rs_idx = captured_barrier_idx[si]             # barrier slot consumed by RS op
barrier_ag_idx = (captured_barrier_idx[si] + 1) % 2  # barrier slot consumed by AG op

# Reset the 3 RS semaphore handles (one per ring position).
for rs_sem in tt_ccl.rs_semaphore_handles[si][rs_idx]:
    ttnn.reset_global_semaphore_value(rs_sem, 0)

# Reset the barrier handle associated with the RS op.
ttnn.reset_global_semaphore_value(
    tt_ccl.barrier_semaphore_handles[si][barrier_rs_idx], 0
)

# Reset the 2 AG semaphore handles (one per ring direction).
for ag_sem in tt_ccl.ag_semaphore_handles[si][ag_idx]:
    ttnn.reset_global_semaphore_value(ag_sem, 0)

# Reset the barrier handle associated with the AG op.
ttnn.reset_global_semaphore_value(
    tt_ccl.barrier_semaphore_handles[si][barrier_ag_idx], 0
)
```

> **Key insight:** `barrier_ag_idx` is derived from `captured_barrier_idx[si]` by adding 1, not from `captured_ag_idx`. The barrier index and the AG index are independent counters. The RS op consumes `captured_barrier_idx[si]` and the AG op consumes `(captured_barrier_idx[si] + 1) % 2`, because both barrier cycling calls happen against the same `barrier_semaphore_idx` array, which advances once per call.

---

## Complete Capture Wrapper: Numbered Checklist

Follow these steps in order when restructuring a trace capture site.

1. **Run the compile (warm-up) pass.** Execute the traced operations once outside any trace bracket. This populates the program cache. After this pass, the host indices have advanced by one position from their initialization values.

2. **[Recommended] Reset device semaphores after the compile pass.** The compile pass runs kernels that self-reset their semaphores, but if the compile pass itself ran without a prior reset, device state may be non-zero. Resetting now is defensive and costs very little.

3. **Snapshot host indices.** Immediately before the device semaphore reset for capture, read the current index arrays out of `tt_ccl`:
   ```python
   captured_ag_idx      = list(tt_ccl.ag_semaphores_idx)
   captured_rs_idx      = list(tt_ccl.rs_semaphores_idx)
   captured_barrier_idx = list(tt_ccl.barrier_semaphore_idx)
   ```
   These values identify which handle slots the trace will consume.

4. **Reset device semaphores for the capture-time handles.** Using the snapshot values, enqueue `reset_global_semaphore_value` for every handle that will be consumed during the capture (see the reset code above for `use_composite=False` and `use_composite=True`). These resets must be enqueued before `begin_trace_capture`.

5. **Call `begin_trace_capture`.** The device is now in a clean state for the handles the trace will use.

6. **Call the traced operations.** The `get_and_cycle_*` calls inside these operations will consume the handle slots identified by the snapshot and advance the host indices past them.

7. **Call `end_trace_capture`.** The DRAM command buffer is assembled and becomes immutable. The host indices now point to the slot *after* the captured slot.

8. **Do not restore host indices yet.** The captured values are already stored in `captured_ag_idx` etc. The post-capture host indices are irrelevant to replay and will be overwritten before the first `execute_trace`.

> **Note:** Steps 3 and 4 are ordered this way — snapshot first, then reset — because the reset code uses the snapshot values to identify which handles to reset. If you reset first and then snapshot, you will snapshot the same values but have wasted a read. More importantly, if you intend to place the snapshot before the compile pass, the compile pass will advance the indices and your snapshot will be stale. Always snapshot immediately before `begin_trace_capture`.

---

## Complete Replay Wrapper: Numbered Checklist

Follow these steps in order before every `execute_trace` call in the decode loop.

1. **Reset device semaphore values.** Using the frozen `captured_*_idx` variables, enqueue `reset_global_semaphore_value` for every handle slot the trace uses. For `use_composite=False`, that is the AG handles and one barrier handle. For `use_composite=True`, that is the RS handles, the AG handles, and two barrier handles.

2. **Restore host index arrays.** Overwrite the live `tt_ccl` index fields with the captured values:
   ```python
   tt_ccl.ag_semaphores_idx      = list(captured_ag_idx)
   tt_ccl.rs_semaphores_idx      = list(captured_rs_idx)
   tt_ccl.barrier_semaphore_idx  = list(captured_barrier_idx)
   ```

3. **Call `execute_trace`.** The CQ FIFO guarantees the reset commands from step 1 will complete before the trace begins.
   ```python
   ttnn.execute_trace(mesh_device, cq_id=0, tid=trace_id, blocking=True)
   ```

4. **Do not call `get_and_cycle_*` between replays** unless those calls are for a different `semaphore_index` slot that the trace does not use. Any cycling call that touches the captured `si` will advance the index again, so the restore in step 2 of the next iteration will still correct it — but be aware that the device-side handles touched by such an intervening call will need their own reset management.

> **Warning:** Use `blocking=True` in `execute_trace` unless you have carefully analyzed the self-reset timing of every kernel in the trace and confirmed that the next replay's reset commands will always be enqueued *after* all previous kernels have written their self-resets. In practice, `blocking=False` introduces a race between the self-reset writes from replay N and the host-enqueued resets for replay N+1. The host-enqueued resets will win if they are enqueued after the kernels complete, but there is no guarantee without blocking. The chapter 3 analysis shows that this race produces silent corruption (skip-through), not a hard error, making it very difficult to detect.

---

## Identifying Which Handles the Trace Uses

When writing the reset loop for a model you did not author, you need to determine: (a) which `semaphore_index` the model uses, (b) whether the trace contains `use_composite=True` or `use_composite=False` ops, and (c) which index slots were current at capture time.

### Step 1: Find the `TT_CCL` instantiation

Search for `TT_CCL(` in the model directory. Check whether the import resolves to `models/tt_transformers/tt/ccl.py` or `models/common/modules/tt_ccl.py`. The `semaphore_index` formula differs between the two files (see the table in the "Determining the semaphore index" section above).

### Step 2: Find the `cluster_axis` argument

Look at how `TT_CCL` is constructed. The `cluster_axis` argument at construction time determines `semaphore_index`. Common patterns:

```python
# cluster_axis=None → si=2 in both files
tt_ccl = TT_CCL(mesh_device, cluster_axis=None, ...)

# cluster_axis=0 → si=2 in older file, si=0 in newer file
tt_ccl = TT_CCL(mesh_device, cluster_axis=0, ...)

# cluster_axis=1 → si=1 in both files
tt_ccl = TT_CCL(mesh_device, cluster_axis=1, ...)
```

### Step 3: Find the traced CCL call

Locate the `all_gather_async`, `reduce_scatter_minimal_async`, or `tt_all_reduce` call that sits between `begin_trace_capture` and `end_trace_capture`. Check its `use_composite` argument (for `tt_all_reduce`) or whether it is a standalone AG or RS call.

### Step 4: Derive captured indices from the snapshot

Once you have `captured_ag_idx`, `captured_rs_idx`, and `captured_barrier_idx`, the handle slots are:

- `tt_ccl.ag_semaphore_handles[si][captured_ag_idx[si]]` — list of 2 objects
- `tt_ccl.rs_semaphore_handles[si][captured_rs_idx[si]]` — list of 3 objects (only for `use_composite=True`)
- `tt_ccl.barrier_semaphore_handles[si][captured_barrier_idx[si]]` — single object (first barrier)
- `tt_ccl.barrier_semaphore_handles[si][(captured_barrier_idx[si] + 1) % 2]` — single object (second barrier, only for `use_composite=True`)

> **Key insight:** You do not need to inspect the DRAM command buffer to know which handles the trace uses. The snapshot taken before `begin_trace_capture` captures that information entirely. The `get_and_cycle_*` functions are deterministic: given the index values at the start of the traced region, you can predict exactly which handles were consumed.
