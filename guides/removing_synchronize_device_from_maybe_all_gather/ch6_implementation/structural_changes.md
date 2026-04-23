# Structural Changes

This file specifies every code change required to remove `ttnn.synchronize_device()` from `_maybe_all_gather` and, for the Type B2 path, to wire in `TT_CCL` and replace the synchronous `ttnn.all_gather` with `ttnn.experimental.all_gather_async`. By the end of this file you will have a complete list of edits across module constructors and `_maybe_all_gather` implementations, organized so that each edit can be reviewed independently.

---

## Type A — Delete synchronize_device Only

Type A applies when:
- `_maybe_all_gather` currently calls synchronous `ttnn.all_gather` (confirmed by the audit in [Chapter 4](../ch4_symbiote_audit/audit_results.md)).
- The dispatch intent is to keep the synchronous call and rely on CQ0 FIFO ordering for correctness.
- The primary objective is trace enablement without changing the all_gather variant.

### Change 1A: Delete `ttnn.synchronize_device` from `_maybe_all_gather`

Locate `_maybe_all_gather` in:
- `models/tt_symbiote/nn/attention/qwen3_full_attention.py` (or the shared base class — confirm via Chapter 4)
- `models/tt_symbiote/nn/attention/qwen3_linear_attention.py` (if the method is duplicated, not inherited)

Delete the line:

```python
# DELETE this line:
ttnn.synchronize_device(self.mesh_device)
```

No other changes are required for Type A. The CQ0 FIFO ordering guarantee (described in [Chapter 3, `command_queue_ordering_guarantee.md`](../ch3_root_cause_analysis/command_queue_ordering_guarantee.md)) ensures that the synchronous `ttnn.all_gather` output is valid before the downstream op reads it.

**Validation:** Run the functional correctness test described in [Chapter 7, `functional_correctness.md`](../ch7_validation/functional_correctness.md) to confirm that PCC > 0.999 against the reference implementation.

---

## Type B2 — Replace all_gather with all_gather_async + Cycling Semaphores

Type B2 applies when:
- The intent is to adopt the async CCL pattern from `models/tt_transformers/tt/attention.py` for latency optimization or full trace compatibility including a persistent output buffer contract.
- The engineer has confirmed that `TT_CCL` (or equivalent semaphore pool infrastructure) can be threaded through the module constructor chain.

Type B2 consists of three ordered groups of changes.

---

### Change Group B2-1: TT_CCL Wiring in Module Constructors

**Option 1 (Preferred): Share a TT_CCL instance from the parent model**

Modify `TTNNQwen3FullAttention.__init__` and `TTNNQwen3LinearAttention.__init__` (or the shared base class `__init__` if one exists) to accept a `tt_ccl` parameter:

```python
def __init__(
    self,
    ...,
    tt_ccl: TT_CCL,   # <-- add this parameter
    ...
):
    ...
    self.tt_ccl = tt_ccl   # <-- store for use in _maybe_all_gather
```

The parent `LayerStack` or model-level constructor must pass its existing `TT_CCL` instance (already present for the outer reduce_scatter semaphores) to each attention module's constructor. No new `TT_CCL` object is created — the existing one is reused.

> **Note:** If the shared `TT_CCL` instance was created with a specific set of cluster axes for the model's outer CCL ops, verify that the `cluster_axis` values used by `_maybe_all_gather` are covered by the existing semaphore pools. `TT_CCL.__init__` allocates pools for `cluster_axis=0`, `cluster_axis=1`, and `cluster_axis=None` by default; confirm the pools match the call sites.

**Option 2 (Self-contained): Create a per-module semaphore pool in `__init__`**

If threading a shared `TT_CCL` instance through the constructor chain is not immediately feasible, each module can create its own minimal semaphore pool:

```python
from models.tt_transformers.tt.ccl import TT_CCL

def __init__(self, ..., mesh_device, cluster_axis, ...):
    ...
    # Per-module TT_CCL for _maybe_all_gather semaphores only.
    # Trade-off: multiplies GlobalSemaphore count by number of layers.
    # Prefer Option 1 (shared TT_CCL) to minimize L1 usage.
    self.tt_ccl = TT_CCL(
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        ...
    )
```

> **Warning:** Option 2 allocates additional `GlobalSemaphore` L1 buffers on each device for every attention module instance. For a 32-layer model with 2 modules per layer, this is 64 per-module `TT_CCL` instances × 2 semaphore slots × 2 semaphore types = 256 additional `GlobalSemaphore` allocations. This may be acceptable but should be profiled against L1 capacity. Option 1 is preferred.

---

### Change Group B2-2: `_maybe_all_gather` Signature and Body

**Step B2-2a: Add `cluster_axis` parameter if not already present**

The `cluster_axis` argument tells `TT_CCL` which semaphore pool slot to use. If `_maybe_all_gather` already receives `cluster_axis` (to pass to `ttnn.all_gather`), no signature change is needed. If it is captured in the closure or stored as `self.cluster_axis`, refactor to pass it explicitly so the type of axis is visible at the call site:

```python
def _maybe_all_gather(self, x: ttnn.Tensor, cluster_axis: int, ...) -> ttnn.Tensor:
```

**Step B2-2b: Replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`**

Replace the body of the `if self.num_devices > 1:` block:

```python
# BEFORE (synchronous):
x = ttnn.all_gather(
    x,
    dim=...,
    num_links=self.num_links,
    cluster_axis=cluster_axis,
    memory_config=self.all_gather_memory_config,
)
ttnn.synchronize_device(self.mesh_device)   # DELETE

# AFTER (async with cycling semaphores):
x = ttnn.experimental.all_gather_async(
    x,
    dim=...,
    persistent_output_buffer=None,           # program cache provides buffer stability
    multi_device_global_semaphore=(
        self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)
        # cycling semaphore handle: alternates slot 0 / slot 1 on each call
    ),
    barrier_semaphore=(
        self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
        # barrier semaphore for completion signaling to downstream device kernels
    ),
    num_links=self.num_links,
    topology=self.ccl_topology,              # must match existing ring topology config
    memory_config=self.all_gather_memory_config,
)
# synchronize_device removed — completion is signaled via GlobalSemaphore to device;
# CQ0 FIFO ordering ensures downstream ops wait for all_gather_async output
```

> **Note:** The `persistent_output_buffer=None` argument causes `all_gather_async` to allocate an output buffer on the compile (warm-up) run and cache its address in the program cache. On subsequent calls — including trace capture and trace replay — the cached address is reused via `override_runtime_arguments`. This provides the buffer address stability required by Metal Trace. Do not pass a manually allocated buffer unless you have confirmed that its address will not change across replays.

> **Note:** Match `topology` and `num_links` to the values used by the existing `ttnn.all_gather` call. These can be found in the source at the call site identified in [Chapter 4](../ch4_symbiote_audit/audit_results.md). A common configuration for T3K (1×8 ring on cluster_axis=1) is `topology=ttnn.Topology.Ring` and `num_links=1` or `num_links=2` depending on the model configuration.

**Step B2-2c: Confirm call sites in `TTNNQwen3FullAttention.forward` and `TTNNQwen3LinearAttention.forward`**

After changing `_maybe_all_gather`, verify that every call site in the forward methods passes the correct `cluster_axis` value. Typical values:
- `cluster_axis=1` for tensor-parallel all_gather on the device-ring axis in T3K
- `cluster_axis=0` if the model uses a 2D mesh and gathers on the inter-node axis

Refer to the call site documentation in [Chapter 1, `call_sites_and_control_flow.md`](../ch1_maybe_all_gather_anatomy/call_sites_and_control_flow.md) for the specific axis used by each module.

---

### Change Group B2-3: Delete synchronize_device

After completing B2-1 and B2-2, the `ttnn.synchronize_device(self.mesh_device)` line is already removed in step B2-2b above. Confirm that no other `synchronize_device` calls remain in `_maybe_all_gather` or in the forward methods of the affected modules.

If the method is defined in a base class, verify that the deletion does not affect any subclass that might rely on the synchronize behavior for a different reason. The audit in [Chapter 4](../ch4_symbiote_audit/audit_results.md) should list all subclasses; review each.

---

## Summary of All Edits

| File | Change | Path type |
|---|---|---|
| `qwen3_full_attention.py` (or shared base) | Delete `ttnn.synchronize_device(...)` line | A and B2 |
| `qwen3_linear_attention.py` (if not base class) | Delete `ttnn.synchronize_device(...)` line | A and B2 |
| `qwen3_full_attention.py` (or shared base) | Add `tt_ccl: TT_CCL` constructor parameter; store as `self.tt_ccl` | B2 only |
| `qwen3_linear_attention.py` (if not base class) | Same `tt_ccl` constructor change | B2 only |
| `_maybe_all_gather` body | Replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async` + cycling semaphore args | B2 only |
| Parent `LayerStack` or model constructor | Pass shared `TT_CCL` instance to each attention module constructor | B2 only |
| Trace capture wrapper (`TracedRun`) | Semaphore index snapshot + reset before capture and before each replay | B2 only |

The trace capture wrapper changes are described in [`trace_capture_wrapper_changes.md`](./trace_capture_wrapper_changes.md).
