# Double-Buffer Design

This file explains why `TT_CCL` maintains two alternating semaphore handles per cluster-axis variant rather than one, shows how the three host-side index arrays and the three `get_and_cycle_*` methods implement that alternation, illustrates the handle selection sequence across four consecutive CCL calls, and states the key invariant that both the host counter and the device semaphore value must be consistent before every CCL invocation.

---

## Motivation: why two handles per axis?

Async CCL operations — `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` — use global semaphores to signal completion between device kernels. A kernel on one device increments the semaphore word at `semaphore.address()` when it finishes sending or receiving data; the kernel on the other end spins waiting for the semaphore to reach a target value (typically the ring size or a completion count).

If the same semaphore handle were reused for two back-to-back CCL operations on the same axis, the second operation's kernels could observe a non-zero semaphore value left by the first operation before their own transmission has started. The kernel would interpret that leftover count as an early completion signal and proceed — producing corrupted results.

The double-buffer design avoids this by maintaining two independent handles for each axis variant. Operation N uses handle at slot 0; operation N+1 uses the handle at slot 1. By the time operation N+2 arrives and selects slot 0 again, operation N's device-side activity has completed and its semaphore has been (or can be) reset to zero before reuse. The two handles "leapfrog" each other, giving each operation a fresh semaphore while the previous operation's signal is still live on the other slot.

> **Note:** The double-buffer design relaxes timing: the caller does not need to wait for operation N to fully complete before issuing operation N+1, because N+1 operates on a different semaphore. However, before slot 0 is used again (operation N+2), its device-side semaphore value must be reset to zero. Double-buffering reduces the frequency of required resets but does not eliminate them.

---

## The three host-side index arrays

`TT_CCL.__init__` initializes three index arrays, each of length 3:

```python
self.barrier_semaphore_idx = [0, 0, 0]
self.ag_semaphores_idx     = [0, 0, 0]
self.rs_semaphores_idx     = [0, 0, 0]
```

Each array has three slots:

| Array position | Cluster axis |
|---|---|
| 0 | `cluster_axis=0` |
| 1 | `cluster_axis=1` |
| 2 | `cluster_axis=None` |

Each slot is an integer that cycles between 0 and 1 (modulo 2). It records which double-buffer slot the *next* call to the corresponding `get_and_cycle_*` method will return.

The mapping from `cluster_axis` argument to array slot is computed the same way in all three methods:

```python
semaphore_index = 2 if cluster_axis is None else cluster_axis
```

So `cluster_axis=None` maps to slot 2, `cluster_axis=0` maps to slot 0, and `cluster_axis=1` maps to slot 1.

---

## The `get_and_cycle_*` methods in full

All three cycling methods follow the same pattern. Here they are as they appear in both `models/tt_transformers/tt/ccl.py` and `models/common/modules/tt_ccl.py`:

```python
def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
    semaphore_index = 2 if cluster_axis is None else cluster_axis
    current_idx = self.barrier_semaphore_idx[semaphore_index]
    self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
    return self.barrier_semaphore_handles[semaphore_index][current_idx]

def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
    semaphore_index = 2 if cluster_axis is None else cluster_axis
    current_idx = self.ag_semaphores_idx[semaphore_index]
    self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
    return self.ag_semaphore_handles[semaphore_index][current_idx]

def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
    semaphore_index = 2 if cluster_axis is None else cluster_axis
    current_idx = self.rs_semaphores_idx[semaphore_index]
    self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
    return self.rs_semaphore_handles[semaphore_index][current_idx]
```

The sequence for a single call is:

1. Compute `semaphore_index` from `cluster_axis`.
2. Read `current_idx = *_idx[semaphore_index]` — this is the slot that will be returned.
3. Write `*_idx[semaphore_index] = (current_idx + 1) % 2` — the next call will return the other slot.
4. Return the handle (or list of handles) at `*_handles[semaphore_index][current_idx]`.

> **Note:** The `models/common/modules/tt_ccl.py` version uses `2 if cluster_axis is None else cluster_axis` (an identity test). The older `models/tt_transformers/tt/ccl.py` version uses `2 if not cluster_axis else cluster_axis` (a truthiness test). These two forms are **not** equivalent: in Python, `not 0` evaluates to `True`, so the older form returns `2` when `cluster_axis=0`, mapping axis-0 incorrectly to slot 2 (the `cluster_axis=None` slot) instead of slot 0. The `is None` form is correct; the older truthiness form is a pre-existing bug in that file.

The return value of each method:

| Method | Return type |
|---|---|
| `get_and_cycle_barrier_semaphore_handle` | Single `GlobalSemaphore` handle |
| `get_and_cycle_ag_semaphore_handles` | List of 2 `GlobalSemaphore` handles |
| `get_and_cycle_rs_semaphore_handles` | List of 3 `GlobalSemaphore` handles |

The list form for AG and RS is because those ops require multiple semaphores per invocation to coordinate the different phases of the ring protocol (e.g., sender and receiver signals). Each element in the returned list is a distinct `GlobalSemaphore` with its own L1 address.

---

## 4-call sequence illustration

Consider four consecutive `tt_all_reduce` calls on `cluster_axis=0`, using the non-composite path (`use_composite=False`). Each call to `tt_all_reduce` invokes `all_gather_async` and therefore calls `get_and_cycle_ag_semaphore_handles(cluster_axis=0)` and `get_and_cycle_barrier_semaphore_handle(cluster_axis=0)`.

Initial state: `ag_semaphores_idx[0] = 0`, `barrier_semaphore_idx[0] = 0`.

**Call 1: first `tt_all_reduce`**

```
ag_semaphores_idx[0]:      0 → 1   returns ag_semaphore_handles[0][0]   ← slot 0
barrier_semaphore_idx[0]:  0 → 1   returns barrier_semaphore_handles[0][0]  ← slot 0
```

State after call 1: `ag_semaphores_idx[0] = 1`, `barrier_semaphore_idx[0] = 1`.

**Call 2: second `tt_all_reduce`**

```
ag_semaphores_idx[0]:      1 → 0   returns ag_semaphore_handles[0][1]   ← slot 1
barrier_semaphore_idx[0]:  1 → 0   returns barrier_semaphore_handles[0][1]  ← slot 1
```

State after call 2: `ag_semaphores_idx[0] = 0`, `barrier_semaphore_idx[0] = 0`.

**Call 3: third `tt_all_reduce`** — same as call 1; returns slot 0 again.

**Call 4: fourth `tt_all_reduce`** — same as call 2; returns slot 1 again.

Timeline view (host state row, device state row):

```
Time →    │ Call 1            │ Call 2            │ Call 3            │ Call 4
──────────┼───────────────────┼───────────────────┼───────────────────┼───────────────
Host idx  │ 0 → 1             │ 1 → 0             │ 0 → 1             │ 1 → 0
Handle    │ slot 0            │ slot 1            │ slot 0            │ slot 1
──────────┼───────────────────┼───────────────────┼───────────────────┼───────────────
Device    │ slot 0: active    │ slot 0: completing│ slot 0: reused    │ slot 1: reused
semaphore │ slot 1: zero      │ slot 1: active    │ slot 1: completing│ slot 0: completing
```

Before call 3 reuses slot 0, the device-side semaphore word at `ag_semaphore_handles[0][0][*].address()` must have been reset to 0. If it was not reset (because it still holds the non-zero value written by call 1's kernels), the call 3 kernels will immediately read the stale non-zero value as if they had already completed — producing silent data corruption.

---

## The key invariant

> **Key insight:** The double-buffer design establishes the following invariant that must hold before every CCL invocation:
>
> 1. The host-side cycling counter (`*_idx[semaphore_index]`) points to the slot that will be passed to the CCL op.
> 2. The device-side semaphore word at every handle in that slot is 0 (reset to its initial state).
>
> Both conditions must hold simultaneously. If condition (1) is wrong — the host counter points to the wrong slot — then the CCL op receives the wrong handle and may collide with a concurrent or prior use of the correct slot. If condition (2) is wrong — the device semaphore is non-zero when the CCL op starts — then the kernel interprets the stale value as an early completion signal and skips the wait.

Double-buffering provides time for condition (2) to be satisfied: while call N is running on slot 0, call N+1 is free to run on slot 1 without waiting for slot 0 to be reset. But by the time call N+2 needs slot 0 again, slot 0's semaphore must have been explicitly reset to 0 via `ttnn.reset_global_semaphore_value`.

In normal (non-traced) execution this reset is the caller's responsibility and must be managed explicitly. In traced execution, the interaction between the host-side counter state and the trace-baked handle addresses introduces additional complexity, which is the subject of the remaining chapters.

---

**Next:** [Chapter 2 — How Semaphore Addresses Flow into Kernel Runtime Arguments](../ch2_semaphore_rta_path/index.md)
