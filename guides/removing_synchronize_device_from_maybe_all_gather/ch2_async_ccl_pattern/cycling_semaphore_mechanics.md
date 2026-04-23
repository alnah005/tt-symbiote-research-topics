# Cycling Semaphore Mechanics in TT_CCL

This file documents the double-buffer semaphore design in the `TT_CCL` class (`models/tt_transformers/tt/ccl.py`). By the end you will understand the three-axis pool structure, how `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle` advance the slot index, why double-buffering is required inside a trace to prevent semaphore aliasing, and what structural change is necessary before `_maybe_all_gather` can adopt this pattern.

Source file: `models/tt_transformers/tt/ccl.py` in the `tt-metal` repository.

---

## Double-Buffer Initialization in `TT_CCL.__init__`

```python
# From TT_CCL.__init__ in models/tt_transformers/tt/ccl.py

class TT_CCL:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        # Three slot indices: [cluster_axis=0, cluster_axis=1, cluster_axis=None]
        self.barrier_semaphore_idx = [0, 0, 0]     # why: tracks which double-buffer slot is "current" for each axis
        self.barrier_semaphore_handles = [[], [], []]

        self.ag_semaphores_idx = [0, 0, 0]         # why: same per-axis cycling index for all_gather semaphores
        self.ag_semaphore_handles = [[], [], []]

        self.rs_semaphores_idx = [0, 0, 0]         # why: same for reduce_scatter semaphores
        self.rs_semaphore_handles = [[], [], []]

        # cluster-axis-0, cluster-axis-1, no-cluster-axis (indices 0, 1, 2 respectively)
        for i in range(3):
            # double buffered semaphores — two slots per axis variant
            for _ in range(2):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                    # why: GlobalSemaphore allocated in L1 across the full compute grid;
                    #      initial value 0; two handles per axis = two independent L1 addresses
                )

                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                    # why: all_gather_async needs TWO semaphore handles per call (one for
                    #      the send-side, one for the receive-side completion signal);
                    #      two slots of two handles each = 4 GlobalSemaphore objects per axis
                )

                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                    # why: reduce_scatter_minimal_async needs THREE semaphore handles per call;
                    #      two slots of three handles each = 6 GlobalSemaphore objects per axis
                )
```

After `__init__` completes, the pool contains (per axis):

| Handle type | Slots | Handles per slot | Total per axis |
|---|---|---|---|
| `barrier_semaphore_handles[i]` | 2 | 1 | 2 |
| `ag_semaphore_handles[i]` | 2 | 2 | 4 |
| `rs_semaphore_handles[i]` | 2 | 3 | 6 |

The pool has 3 index slots (`i = 0, 1, 2`), giving a total `GlobalSemaphore` count of `3 × (2 + 4 + 6) = 36` objects. However, the cycling methods only actively select indices 1 and 2: `semaphore_index = 2 if not cluster_axis else cluster_axis` maps `cluster_axis=None` and `cluster_axis=0` both to index 2, and `cluster_axis=1` to index 1. Index 0 is allocated by `__init__` but is not reached by the standard calling convention.

---

## The `get_and_cycle_*` Methods

### `get_and_cycle_ag_semaphore_handles`

```python
# From TT_CCL in models/tt_transformers/tt/ccl.py

def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
    semaphore_index = 2 if not cluster_axis else cluster_axis
    # why: maps cluster_axis=None → index 2, cluster_axis=0 → index 2 (falsy), cluster_axis=1 → index 1
    #      `not cluster_axis` is True for both None and 0, so both collapse to index 2;
    #      this selects the correct axis-specific pool from the three-element list

    current_idx = self.ag_semaphores_idx[semaphore_index]
    # why: reads the current slot pointer (0 or 1) for this axis

    self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
    # why: advances the slot pointer for the NEXT call; modular arithmetic ensures
    #      it cycles: 0 → 1 → 0 → 1 → ...

    return self.ag_semaphore_handles[semaphore_index][current_idx]
    # why: returns the [list of 2 GlobalSemaphore handles] for the CURRENT slot before
    #      the index was advanced; the caller receives slot 0 on the first call, slot 1
    #      on the second, slot 0 on the third, and so on
```

### `get_and_cycle_barrier_semaphore_handle`

```python
def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
    semaphore_index = 2 if not cluster_axis else cluster_axis
    # why: same axis-to-index mapping as above — cluster_axis=None → index 2,
    #      cluster_axis=0 → index 2 (falsy), cluster_axis=1 → index 1

    current_idx = self.barrier_semaphore_idx[semaphore_index]
    self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
    # why: same modular cycling; barrier slot advances independently of ag slot

    return self.barrier_semaphore_handles[semaphore_index][current_idx]
    # why: returns a single GlobalSemaphore handle (not a list); the barrier
    #      semaphore has only one handle per slot
```

> **Note:** For models like T3K that only use `cluster_axis=1` or `None`, `cluster_axis=0` is never passed — so the collapsed index-2 bucket is not a practical issue. But the mapping is worth noting: `not cluster_axis` evaluates to `True` for both `None` and `0`.

> **Note:** The two methods cycle their indices independently. A call to `get_and_cycle_ag_semaphore_handles()` does not advance the barrier index, and vice versa. The Python call site (e.g., `Attention.forward_decode` or `tt_all_gather`) is responsible for calling both methods in the same dispatch to obtain a matched pair of handles from the same slot.

---

## Why Cycling Is Required Inside a Trace

A `GlobalSemaphore` is an L1 memory object. When a `GlobalSemaphore` handle is passed to `all_gather_async`, the device-side kernel writes the handle's L1 address into its program arguments. During trace capture, that L1 address is baked into the trace command buffer.

Consider a traced loop that runs two consecutive `all_gather_async` calls (as occurs when multiple layers are stacked inside a single trace or when a pipelined decode unrolls two steps):

**Without cycling (single handle, reused):**
```
Step N:   all_gather_async(semaphore=handle_A)
           → kernel completes, writes "done" signal to handle_A's L1 address
Step N+1: all_gather_async(semaphore=handle_A)
           → kernel checks handle_A's L1 address before starting
           → L1 address still holds the "done" signal from Step N
           → kernel may interpret the stale signal as an immediate completion
           → output buffer contains garbage from Step N, not Step N+1 data
```

This is semaphore aliasing: the completion signal left by iteration N is misread as the completion signal for iteration N+1.

**With cycling (two handles, alternating):**
```
Step N:   all_gather_async(semaphore=handle_A, slot 0)
           → kernel completes, writes "done" to handle_A's L1 address
Step N+1: all_gather_async(semaphore=handle_B, slot 1)
           → handle_B's L1 address was reset to 0 before this step
           → no stale signal; kernel waits correctly for Step N+1 completion
Step N+2: all_gather_async(semaphore=handle_A, slot 0)  ← cycles back
           → handle_A's L1 address was reset to 0 before this step
```

The cycling plus a pre-replay reset (described in `ch6_implementation/trace_capture_wrapper_changes.md`) ensures that on every replay the same handle addresses that were baked into the trace are presented in the correct initial state.

> **Warning:** The handle addresses are baked into the trace at `end_trace_capture` time. If the cycling index is not restored to its capture-time value before each `execute_trace`, a different handle (with a different L1 address) will be selected by the cycling logic — but the trace will still attempt to write to the capture-time address. The mismatch silently corrupts semaphore signaling.

---

## Structural Requirement for `_maybe_all_gather`

The current `_maybe_all_gather` implementation in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` does not hold a reference to a `TT_CCL` instance. It has no access to cycling semaphore handles.

For `_maybe_all_gather` to use `ttnn.experimental.all_gather_async`, the following structural change is necessary:

- `TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, or their shared base class must accept a `tt_ccl: TT_CCL` parameter in `__init__` and store it as `self.tt_ccl`.
- `_maybe_all_gather` must call `self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)` and `self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)` to obtain the handles for each `all_gather_async` call.
- The parent model or layer stack that constructs the attention modules must pass the same shared `TT_CCL` instance to every attention layer — so that the cycling indices are correctly synchronized across layers and no two layers independently cycle over the same handle slot at the same time.

> **Key finding:** The cycling semaphore pattern is not an optional optimization — it is a correctness requirement for any async CCL op inside a traced loop. `_maybe_all_gather` cannot become trace-compatible by simply removing `synchronize_device` without also obtaining access to cycling semaphore handles. The `TT_CCL` wiring is the structural prerequisite.

For the concrete wiring changes, see [`../ch6_implementation/structural_changes.md`](../ch6_implementation/structural_changes.md).
