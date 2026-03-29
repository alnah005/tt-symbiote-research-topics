# Contrast with Async Variants

`ttnn.all_reduce` is one of three collective reduction patterns used across tt-symbiote
modules. This file contrasts it with two async variants — `ttnn.experimental.all_reduce_async`
and `ttnn.experimental.reduce_scatter_minimal_async` — to clarify why the synchronous
form avoids the cycling semaphore complexity that the async forms require.

---

## 1. `ttnn.all_reduce` — synchronous, no explicit semaphore args

`ttnn.all_reduce` passes `std::nullopt` for all three semaphore arguments, which the
C++ implementation forwards directly to the cluster-axis overload of
`ExecuteAllReduceAsync::invoke`. Because both `has_value()` guards in that overload
evaluate to `false`, execution falls through to the synchronous `ttnn.reduce_scatter`
and `ttnn.all_gather` fallbacks — which use only per-program local semaphores and
carry no persistent semaphore state between calls. The synchronous path is
conditionally trace-compatible on a 1-D fabric T3K configuration when `scattered_tensor`
is pre-allocated as a persistent buffer before trace capture.

For the Python call site and the C++ `std::nullopt` forwarding block, see
[`call_chain.md` §2](./call_chain.md). For the full `has_value()` guard logic and
the `scattered_tensor` pre-allocation requirement, see
[`reduce_scatter_all_gather_path.md` §1 and §4](./reduce_scatter_all_gather_path.md).

---

## 2. `ttnn.experimental.all_reduce_async` — requires caller-supplied `GlobalSemaphore` objects

### What the async overload requires

`ttnn.experimental.all_reduce_async` is an overloaded operation. The two primary
overloads differ in their semaphore type signatures:

**Non-cluster-axis overload** (lines 21–31 of `all_reduce_async.hpp`):

```cpp
static ttnn::Tensor invoke(
    const ttnn::Tensor& input_tensor,
    uint32_t num_devices,
    const std::vector<GlobalSemaphore>& barrier_semaphores,       // non-optional
    const std::vector<GlobalSemaphore>& rs_global_semaphores,     // non-optional
    const std::vector<GlobalSemaphore>& ag_global_semaphores,     // non-optional
    ttnn::operations::reduction::ReduceType math_op,
    ...);
```

All three semaphore vectors are **non-optional** (`const std::vector<GlobalSemaphore>&`).
This overload always calls `reduce_scatter_minimal_async` and `prim::all_gather_async`
unconditionally — there is no synchronous fallback.

**Cluster-axis overload** (lines 33–44 of `all_reduce_async.hpp`) — the one that
`ttnn.all_reduce` delegates to:

```cpp
static ttnn::Tensor invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    MeshDevice& mesh_device,
    const std::optional<std::vector<GlobalSemaphore>>& barrier_semaphores,    // optional
    const std::optional<std::vector<GlobalSemaphore>>& rs_global_semaphores,  // optional
    const std::optional<std::vector<GlobalSemaphore>>& ag_global_semaphores,  // optional
    ttnn::operations::reduction::ReduceType math_op,
    ...);
```

All three semaphore vectors are **optional** (`const std::optional<std::vector<GlobalSemaphore>>&`).
When `std::nullopt` is passed for either `rs_global_semaphores` or `barrier_semaphores`,
the implementation falls back to synchronous `ttnn.reduce_scatter`. Similarly, when
either `ag_global_semaphores` or `barrier_semaphores` is absent, it falls back to
synchronous `ttnn.all_gather`.

> **Important:** A developer who looks up `ttnn.experimental.all_reduce_async` and
> encounters the non-cluster-axis overload first will see non-optional semaphore
> parameters. Calling the cluster-axis overload as though semaphores are required
> (i.e., not wrapping them in `std::optional`) will fail to compile against the
> optional-accepting signature, and vice versa. Always identify which overload you
> need before constructing the call site.

For both overloads, the caller must supply pre-created `GlobalSemaphore` objects —
each created via `ttnn.create_global_semaphore` at a fixed L1 address — and is
responsible for ensuring those semaphores are in the correct state before each
invocation.

When this overload is used inside a trace, the firmware kernels read and write the
semaphore L1 addresses that were recorded during capture. Because `GlobalSemaphore`
objects live at fixed L1 addresses for their entire lifetime, the hardware accesses the
correct memory locations on every replay. However, the semaphores are **not** reset
between trace replays by the trace mechanism itself. If a semaphore is left in a
non-zero state at the end of one replay, the next replay starts with corrupted
synchronisation state, which causes the CCL to hang or produce incorrect results.

---

## 3. `ttnn.experimental.reduce_scatter_minimal_async` — the cycling semaphore pattern

### Usage in `TTNNLinearIColShardedWRowSharded`

File: `models/experimental/tt_symbiote/modules/linear.py`

```python
tt_output = ttnn.experimental.reduce_scatter_minimal_async(
    tt_output,
    persistent_output_buffers=None,
    dim=3,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

Two `get_and_cycle_*` calls are made. Each call returns a different `GlobalSemaphore`
handle from a double-buffered pool and advances the pool's internal index.

### The `TT_CCL` cycling mechanism

File: `models/common/modules/tt_ccl.py`

```python
class TT_CCL:
    def __init__(self, mesh_device):
        # double-buffered semaphores for each cluster axis (0, 1, no-axis)
        for i in range(3):
            for _ in range(2):  # two slots per axis
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0))
                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(2)])
                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(3)])

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if cluster_axis is None else cluster_axis
        current_idx = self.rs_semaphores_idx[semaphore_index]
        self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[semaphore_index][current_idx]
```

On each call to `get_and_cycle_rs_semaphore_handles`, the pool index advances modulo 2.
This means successive calls alternate between slot 0 and slot 1 of the pre-created
`GlobalSemaphore` pool.

### Why cycling semaphores are necessary for async CCL inside a trace

Inside a trace, the hardware kernels for an async CCL op communicate with each other
using `GlobalSemaphore` L1 addresses encoded directly in the recorded command buffer.
The kernels rely on those semaphores starting at zero at the beginning of each
collective operation. There are two problems that cycling solves:

1. **Residual state from the previous replay.** An async op uses a semaphore to signal
   completion. At the end of a replay, the semaphore may still hold its signalled value.
   When the next replay begins, the kernel reads the same L1 address and finds it
   already set — it interprets this as "already completed" and skips synchronisation,
   corrupting the result. By alternating between two semaphore slots, each replay
   uses a slot that was last written by the replay two iterations ago, giving enough
   time for the slot to be drained and reset.

2. **In-flight overlap.** When two successive replays overlap in the command queue
   (non-blocking `ttnn.execute_trace`), the second replay's kernels may start
   dispatching before the first replay's kernels have finished. Cycling ensures the
   two replays use different semaphore slots, so the second replay's initial check
   does not observe the in-progress state of the first replay's semaphores.

The double-buffering depth (2 slots) is chosen to match the maximum pipeline depth
of non-blocking trace execution.

> **Note:** `TT_CCL` creates all `GlobalSemaphore` objects before any trace is captured
> (at `DistributedConfig.__post_init__` time, which runs during module instantiation).
> The L1 addresses of these semaphores are therefore fixed and stable before
> `ttnn.begin_trace_capture` is called. The cycling pointer advances in Python host
> code — it is not part of the trace itself — so the correct semaphore handle is passed
> to each invocation regardless of whether that invocation is inside a trace or not.

---

## 4. Why `ttnn.all_reduce` avoids this complexity

`ttnn.all_reduce` passes `std::nullopt` for all semaphore arguments. The resulting
synchronous fallback path (`ttnn.reduce_scatter` + `ttnn.all_gather`) does not use
`GlobalSemaphore` objects at all. The consequences are:

- **No residual semaphore state.** Each call dispatches a fresh `tt_metal::Program`
  with freshly initialised per-program semaphores. The hardware resets these semaphores
  as part of program setup; no state from the previous call is visible.
- **No cycling required.** Because there are no external semaphore handles, there is
  no pool index to advance and no double-buffering to maintain.
- **No pre-capture setup.** The caller does not need to allocate any
  `GlobalSemaphore` objects, register them with a `TT_CCL` instance, or call any
  `get_and_cycle_*` method.

The trade-off is performance: the synchronous ops block the command queue until the
collective completes, preventing overlap with subsequent compute ops. The async ops
allow the next compute op to be dispatched while the collective is still in progress,
reducing end-to-end latency in pipelined decode loops. Modules that need that latency
reduction (such as `TTNNLinearIColShardedWRowSharded`) use the async variant with
explicit cycling semaphores; modules that prioritise simplicity and trace robustness
(such as `TTNNLinearIColShardedWAllReduced`) use the synchronous `ttnn.all_reduce`.

---

## 5. Comparison table

`ttnn.experimental.all_reduce_async` has two distinct overloads with different semaphore type signatures. The table distinguishes them:

| Property | `ttnn.all_reduce` | `all_reduce_async` — non-cluster-axis overload | `all_reduce_async` — cluster-axis overload (used by `ttnn.all_reduce`) | `ttnn.experimental.reduce_scatter_minimal_async` |
|---|---|---|---|---|
| Semaphore argument type | None (`std::nullopt`) | `const std::vector<GlobalSemaphore>&` (non-optional) | `const std::optional<std::vector<GlobalSemaphore>>&` (optional) | Required (`GlobalSemaphore`) |
| Synchronous fallback available | Yes (always) | No — always async | Yes — falls back to sync when either semaphore arg is `std::nullopt` | No — always async |
| Semaphore type used | Local (per-program) | Global (fixed L1 address) | Global (fixed L1 address) when supplied; local otherwise | Global (fixed L1 address) |
| Cycling required | No | Yes (double-buffered pool) | Yes, when semaphores are supplied | Yes (double-buffered pool) |
| `TT_CCL` dependency | No | Yes | Yes, when semaphores are supplied | Yes |
| Pre-capture setup | None | `GlobalSemaphore` must be created before capture | `GlobalSemaphore` must be created before capture (if used) | `GlobalSemaphore` must be created before capture |
| Blocks command queue | Yes | No | No (when semaphores supplied); Yes (synchronous fallback) | No |
| Trace compatible (1-D fabric, tile-aligned, `scattered_tensor` pre-allocated) | Yes | Yes, with cycling | Yes, with cycling (when semaphores supplied) | Yes, with cycling |

---

**Next:** [Chapter 3 — Trace Compatibility Verdict and Requirements](../ch3_verdict/index.md)
