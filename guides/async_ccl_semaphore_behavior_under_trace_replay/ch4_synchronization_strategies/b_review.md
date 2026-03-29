# B Review — Chapter 4: Synchronization Strategies — Pass 1

## Issue 1 — Wrong `semaphore_index` for `cluster_axis=0` (implementation-breaking)

**Files:** `resetting_device_semaphore_values.md` line 24; `structuring_the_capture.md` line 61

Both files state that `cluster_axis=0` maps to `semaphore_index=0`. This is incorrect.

The actual implementation in `models/tt_transformers/tt/ccl.py` (lines 118, 124, 130) is:

```python
semaphore_index = 2 if not cluster_axis else cluster_axis
```

Because `not 0` evaluates to `True` in Python, `cluster_axis=0` maps to `semaphore_index=2`, the same slot used by `cluster_axis=None`. It is only `cluster_axis=1` that maps to `semaphore_index=1`.

The correct mapping is:
- `cluster_axis=None` → `semaphore_index=2`
- `cluster_axis=0` → `semaphore_index=2`
- `cluster_axis=1` → `semaphore_index=1`

A reader following `resetting_device_semaphore_values.md` would write `semaphore_index = 0` for `cluster_axis=0` and would call `ttnn.reset_global_semaphore_value` on `ag_semaphore_handles[0][...]`, `rs_semaphore_handles[0][...]`, and `barrier_semaphore_handles[0][...]` — none of which are the actual capture-time handles. The handles at slot 2 that the trace actually uses would remain unreset, producing the silent corruption failure mode that Chapter 3 describes.

The same wrong mapping appears in `structuring_the_capture.md` step 11, which lists "(0 for `cluster_axis=0`, 1 for `cluster_axis=1`, 2 for `cluster_axis=None`)" as the active `semaphore_index` values.

Note: the plan.md propagates the same error by describing the method as `2 if cluster_axis is None else cluster_axis`, but the actual code uses `not cluster_axis`.

---

## Issue 2 — `index.md` is missing the required "What's next" section pointing to Chapter 5

**File:** `index.md`

The `index.md` ends after the "Files in reading order" list. It has no "What's next" section pointing to `../ch5_implementation_guide/index.md`. Every other chapter's `index.md` (ch1, ch3) ends with a "What's next" block naming the next chapter and linking to it. The plan.md conventions section specifies: "Every chapter's `index.md` ends with a 'What's next' section listing files in reading order." The absence of this link breaks the reading chain at the end of Chapter 4 and leaves a reader with no navigational pointer to Chapter 5.

---

# B Review — Chapter 4: Synchronization Strategies — Pass 2

## Issue 1 — Wrong source file attribution for kernel self-reset behavior (misleads verification)

**File:** `resetting_device_semaphore_values.md`, paragraph 1 (lines 9–11)

The chapter states: "In the reduce-scatter-minimal-async implementation (`minimal_default_reader.cpp` line 295 and `minimal_default_writer.cpp` line 272), the kernels self-reset the semaphore to 0 as their terminal action before exiting."

Those two files — `minimal_default_reader.cpp` and `minimal_default_writer.cpp` — belong to the `all_gather_async` kernel directory (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/`), not to `reduce_scatter_minimal_async`. The actual `reduce_scatter_minimal_async` kernels are `ring_reduce_scatter_minimal_async_reader.cpp` (280 lines) and `ring_reduce_scatter_minimal_async_writer.cpp` (503 lines) in `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/`.

The cited line 295 of `minimal_default_reader.cpp` does contain a `noc_semaphore_set(..., 0)` — but it is in the AG kernel. A reader who follows the citation to verify the claim (or to understand which RS kernel code drives the reset behavior) will look in the wrong file and wrong operation. The RS writer does contain analogous self-reset logic (`noc_semaphore_set(..., 0)` calls at lines 226 and 479 of `ring_reduce_scatter_minimal_async_writer.cpp`), but the cited file names are incorrect for the claim being made.

---

## Issue 2 — `use_composite=True` path in `tt_all_reduce` omits AG semaphore and second barrier cycling call

**File:** `existing_patterns_in_tt_transformers.md`, line 51

The chapter states: `tt_all_reduce` with `use_composite=True` "calls `get_and_cycle_rs_semaphore_handles(cluster_axis)` and `get_and_cycle_barrier_semaphore_handle(cluster_axis)`" — implying only two cycling calls for this path.

The actual code in `models/tt_transformers/tt/ccl.py` (lines 260–291) shows four cycling calls for `use_composite=True`:
1. `get_and_cycle_rs_semaphore_handles(cluster_axis)` — for `reduce_scatter_minimal_async`
2. `get_and_cycle_barrier_semaphore_handle(cluster_axis)` — for `reduce_scatter_minimal_async`
3. `get_and_cycle_ag_semaphore_handles(cluster_axis)` — for the subsequent `all_gather_async`
4. `get_and_cycle_barrier_semaphore_handle(cluster_axis)` — for the subsequent `all_gather_async`

A reader using this chapter to implement pre-replay semaphore resets for a model that calls `tt_all_reduce` with `use_composite=True` would reset only the RS handles and one barrier handle per slot, leaving the AG handles and the second barrier handle unreset. The AG and second-barrier handles are still baked into the trace (they are each a separate `override_runtime_arguments` write) and still require resetting before each replay. Skipping them produces the silent corruption failure mode described in Chapter 3.

---

## Change Log — Agent A Pass 1

- Issue 1 (semaphore_index for cluster_axis=0): In `resetting_device_semaphore_values.md`, changed `semaphore_index = 0  # for cluster_axis=0` to `semaphore_index = 2  # for cluster_axis=0 — see note below` and added a note block after the code snippet explaining that the older `models/tt_transformers/tt/ccl.py` uses `2 if not cluster_axis else cluster_axis`, making `cluster_axis=0` map to `semaphore_index=2` (same as `None`) due to `not 0` being `True`, while the newer `models/common/modules/tt_ccl.py` correctly maps `cluster_axis=0` to `semaphore_index=0`, with references to Chapter 1 and Chapter 3. In `structuring_the_capture.md`, changed the step 11 mapping list from "(0 for `cluster_axis=0`, 1 for `cluster_axis=1`, 2 for `cluster_axis=None`)" to "(2 for `cluster_axis=0`, 1 for `cluster_axis=1`, 2 for `cluster_axis=None`) — see note below" and added the same explanatory note block after step 13, before the "Why step 4" section.
- Issue 2 (index.md missing What's next): Added a `## What's next` section at the end of `index.md` with a horizontal rule separator and a link to `../ch5_implementation_guide/index.md`, matching the pattern used in other chapter index files.

---

## Change Log — Agent A Pass 2

- Issue 1 (RS kernel attribution): Corrected `resetting_device_semaphore_values.md` paragraph 1. Removed the false attribution of kernel self-reset behavior to `minimal_default_reader.cpp` (line 295) and `minimal_default_writer.cpp` (line 272) as "the reduce-scatter-minimal-async implementation." Those files are the **all_gather_async** kernels in `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/`. Replaced with accurate attribution to the actual RS kernels: `ring_reduce_scatter_minimal_async_reader.cpp` and `ring_reduce_scatter_minimal_async_writer.cpp` in `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/`. Verified self-reset locations in those files: reader resets `out_ready_sem` to 0 at line 275; writer resets `barrier_sem` at line 226 and `batch_ready_sem` at line 479. Added an explicit note block distinguishing the AG `minimal_default_reader/writer.cpp` files from the RS kernel files.
- Issue 2 (use_composite=True four calls): Updated `existing_patterns_in_tt_transformers.md` line 51 description of the `use_composite=True` path in `tt_all_reduce`. Verified against `models/tt_transformers/tt/ccl.py` lines 264–286: the path makes four cycling calls, not two. Replaced the two-call description with a numbered list of all four calls: (1) `get_and_cycle_rs_semaphore_handles` for RS, (2) `get_and_cycle_barrier_semaphore_handle` for RS, (3) `get_and_cycle_ag_semaphore_handles` for AG, (4) `get_and_cycle_barrier_semaphore_handle` for AG. Added explicit emphasis that pre-replay resets must cover all four handle groups, and that resetting only the RS handles and one barrier leaves the AG handles and second barrier handle unreset, producing silent corruption.

---

# B Review — Chapter 4: Synchronization Strategies — Pass 3

## Issue 1 — Second barrier handle for `use_composite=True` missing from reset code and checklist (implementation-breaking)

**Files:** `resetting_device_semaphore_values.md` (code example, lines 28–42); `structuring_the_capture.md` (step 11, line 64)

The `use_composite=True` path in `tt_all_reduce` calls `get_and_cycle_barrier_semaphore_handle(cluster_axis)` **twice** — once for the RS op and once for the AG op. Verified in `models/tt_transformers/tt/ccl.py` lines 265 and 286. Because `barrier_semaphore_idx` is a separate counter that advances on each call, these two calls consume two consecutive double-buffer slots. If the pre-capture `barrier_semaphore_idx[si]` is N, both `barrier_semaphore_handles[si][N]` and `barrier_semaphore_handles[si][(N+1)%2]` are baked into the trace.

The code example in `resetting_device_semaphore_values.md` resets only one barrier handle:

```python
barrier_handle = tt_ccl.barrier_semaphore_handles[semaphore_index][captured_barrier_idx[semaphore_index]]
ttnn.reset_global_semaphore_value(barrier_handle, 0)
```

The second barrier handle (`barrier_semaphore_handles[si][(captured_barrier_idx[si]+1)%2]`) is omitted. This handle is still baked into the trace and still requires resetting before each replay.

`structuring_the_capture.md` step 11 likewise states "For `barrier_semaphore_handles[semaphore_index][captured_barrier_idx[semaphore_index]]` (1 handle): 1 call" — correct only for `use_composite=False` (which makes a single barrier cycling call). For `use_composite=True`, this must read "2 handles, 2 calls."

This directly contradicts the correct analysis in `existing_patterns_in_tt_transformers.md` (lines 57–58), which identifies "the second barrier handle" as a distinct handle group that must be reset. A reader following the code example or checklist for a `use_composite=True` model will miss resetting the second barrier handle, producing the silent corruption failure mode described in Chapter 3.

## Change Log — Agent A Pass 3

- resetting_device_semaphore_values.md: added second barrier handle reset for use_composite=True
- structuring_the_capture.md: updated step 11 barrier count to reflect use_composite=True needing 2 resets

---

# B Review — Chapter 4: Synchronization Strategies — Pass 4

No feedback — chapter approved.

---

# B Review — Cross-Chapter Final Pass (Ch4 fix)

## Issue 1 — Per-replay ordering inconsistency between Ch4 index.md and Ch5

`ch4_synchronization_strategies/index.md` state machine (Steps 1–2) had: (1) Reset TT_CCL index fields, then (2) Enqueue reset_global_semaphore_value — opposite of the ordering in `ch5_implementation_guide/index.md` and `code_changes_required.md` (device reset first, then host restore).

**Fix applied:** Swapped steps 1 and 2 in the `ch4/index.md` state machine diagram so the ordering is now: (1) Enqueue reset_global_semaphore_value(s), (2) Reset TT_CCL index fields, (3) execute_trace. This matches Ch5 and the guide-level Quick Reference.

Note: Both orderings are functionally equivalent since the device reset and host index restore are independent operations. The fix is for consistency only.
