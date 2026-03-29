# B Review — Chapter 3: Mismatch Analysis — Pass 1

## Issue 1 — `index.md` timeline diagram: baked-slot label and post-capture counter label are inverted

**File:** `index.md`, lines 45–56 (the `begin_trace_capture` block and `POST-CAPTURE STATE` block)

**What the diagram says:**

```
begin_trace_capture
  override_runtime_arguments writes address of handle[N%2] and barrier[M%2]
  get_and_cycle_ag (N%2)      (N+1)%2  ...   handle[N%2] snapshotted
  get_and_cycle_barrier (M%2) (N+1)%2  (M+1)%2

POST-CAPTURE STATE:
  ag_semaphores_idx[si]    = (N+1)%2  <── does not match baked slot N%2
  barrier_semaphore_idx[si]= (M+1)%2  <── does not match baked slot M%2
```

**What is actually correct (as shown consistently in `what_gets_baked_in.md`):**

The diagram defines `N` as the ag index value *before* the compile run. After the compile run, the ag index is `(N+1)%2`. When `get_and_cycle_ag_semaphore_handles` is called *inside* the trace bracket, it reads the current index — which is `(N+1)%2` — returns slot `(N+1)%2`, and advances the counter to `((N+1)%2 + 1)%2 = N`. Therefore:

- The handle **baked into the trace** is at slot `(N+1)%2` (not `N%2`).
- The **post-capture host counter** is `N` (not `(N+1)%2`).

The diagram has these two values exactly swapped. With concrete values (N=0): the diagram claims baked=slot 0 and post-capture counter=1, but the correct answer is baked=slot 1 and post-capture counter=0. `what_gets_baked_in.md` is internally consistent and correct; the index.md timeline diagram contradicts it on the identity of the baked slot.

**Why this matters:** A reader who learns from the index.md diagram which slot is baked will identify the wrong semaphore handle as the one locked into the trace. Any reset or counter-management logic they build from that understanding will target the wrong double-buffer slot.

**Fix:** In the `begin_trace_capture` block, change `handle[N%2]` / `barrier[M%2]` to `handle[(N+1)%2]` / `barrier[(M+1)%2]` throughout. In the `POST-CAPTURE STATE` block, change `ag_semaphores_idx[si] = (N+1)%2` to `= N` and `barrier_semaphore_idx[si] = (M+1)%2` to `= M`, and update the baked-slot annotations to `(N+1)%2` and `(M+1)%2`.

---

## Issue 2 — `failure_modes.md` note on `use_composite=True` omits `ag_semaphore_handles` and second barrier handle from the reset list

**File:** `failure_modes.md`, lines 95–97 (the note block at the bottom of the "Correct requirement" section)

**What the note says:**

> The handles baked in for `reduce_scatter_minimal_async` (in the `use_composite=True` path) follow the same pattern: the `rs_semaphore_handles[a][N']` list (a list of 3 `GlobalSemaphore` objects) and `barrier_semaphore_handles[a][M']` must be reset before each replay.

**What is actually correct:**

`use_composite=True` in `tt_all_reduce` calls two async CCL ops in sequence: first `reduce_scatter_minimal_async` (which consumes `get_and_cycle_rs_semaphore_handles` and one `get_and_cycle_barrier_semaphore_handle` call), then `all_gather_async` (which consumes `get_and_cycle_ag_semaphore_handles` and a *second* `get_and_cycle_barrier_semaphore_handle` call). All four handle groups are written as RTAs and baked into the trace. The full reset list for the `use_composite=True` path is:

- `rs_semaphore_handles[a][rs_N']` — a list of 3 `GlobalSemaphore` objects
- `barrier_semaphore_handles[a][barrier_M'_first]` — the barrier used by reduce_scatter
- `ag_semaphore_handles[a][ag_N']` — a list of 2 `GlobalSemaphore` objects
- `barrier_semaphore_handles[a][barrier_M'_second]` — the barrier used by all_gather

The note omits `ag_semaphore_handles` and the second barrier handle entirely.

**Why this matters:** A reader implementing replay semaphore management for the `use_composite=True` path who follows this note will reset only the rs and one barrier handle, leaving the ag semaphore handles dirty. The next replay's `all_gather_async` kernel will see stale non-zero semaphore values and produce silent data corruption.

**Fix:** Expand the note to list all four handle groups that must be reset for the `use_composite=True` path, and introduce separate capture-time index variables for the rs call and the ag call (they advance independently because they are distinct `get_and_cycle_*` calls).

---

# B Review — Chapter 3: Mismatch Analysis — Pass 2

## Issue 1 — `semaphore_index` mapping for `cluster_axis=0` is wrong in `index.md` and `what_gets_baked_in.md`

**Files:** `index.md` line 12; `what_gets_baked_in.md` line 55

**What the chapter says:**

`index.md` (Prerequisites section):

> `get_and_cycle_ag_semaphore_handles(cluster_axis)` returns `ag_semaphore_handles[semaphore_index][current_idx]` ... where `semaphore_index = 2 if cluster_axis is None else cluster_axis`.

`what_gets_baked_in.md` (State Before the Compile Run section):

> `semaphore_index = cluster_axis` for a concrete `cluster_axis` value of 0 or 1.

**What the actual code does:**

All three `get_and_cycle_*` methods in `models/tt_transformers/tt/ccl.py` use:

```python
semaphore_index = 2 if not cluster_axis else cluster_axis
```

In Python, `not 0` is `True`, so `cluster_axis=0` evaluates to `semaphore_index=2` — the same slot used by `cluster_axis=None`. `cluster_axis=1` correctly maps to `semaphore_index=1`. The guide's expression `2 if cluster_axis is None else cluster_axis` would be semantically correct Python (identity test vs. truthiness test), but the guide description in `what_gets_baked_in.md` goes further and explicitly asserts "semaphore_index = cluster_axis for a concrete cluster_axis value of 0 or 1", which is wrong for `cluster_axis=0`: the actual mapping is `semaphore_index=2`, not `0`.

**Why this matters:**

A reader implementing semaphore management for a model that calls `tt_all_reduce` with `cluster_axis=0` will, if they follow the guide, perform counter reads, counter resets, and handle resets against `ag_semaphore_handles[0]`, `rs_semaphore_handles[0]`, and `barrier_semaphore_handles[0]` — the wrong array slots. The actual baked handles live at slot index 2. Resetting or snapshotting the wrong slots leaves the real capture-time handles unreset, producing silent data corruption on every replay after the first.

**Fix:** In `index.md`, change `semaphore_index = 2 if cluster_axis is None else cluster_axis` to `semaphore_index = 2 if not cluster_axis else cluster_axis` and add a note that this means `cluster_axis=0` is treated identically to `cluster_axis=None` (both map to `semaphore_index=2`). In `what_gets_baked_in.md`, remove the claim `semaphore_index = cluster_axis for a concrete cluster_axis value of 0 or 1` and replace it with the correct mapping: `cluster_axis=None` → 2, `cluster_axis=0` → 2, `cluster_axis=1` → 1.

---

## Change Log — Agent A Pass 1

- Issue 1 (index.md timeline diagram): In the `begin_trace_capture` block, changed `handle[N%2]` to `handle[(N+1)%2]` and `barrier[M%2]` to `barrier[(M+1)%2]` in the `override_runtime_arguments` comment, both `get_and_cycle_*` call labels, and the `end_trace_capture` `assemble_dispatch_commands` line. In the `POST-CAPTURE STATE` block, changed `ag_semaphores_idx[si] = (N+1)%2` to `= N` and `barrier_semaphore_idx[si] = (M+1)%2` to `= M`, with annotations updated to `(N+1)%2` and `(M+1)%2` for the baked slots. In the replay blocks, updated all references from `handle[N%2]`/`barrier[M%2]` to `handle[(N+1)%2]`/`barrier[(M+1)%2]`, and corrected the "mismatch is permanent" summary to read `host counter stays at N / M; trace always uses (N+1)%2 / (M+1)%2`.
- Issue 2 (failure_modes.md use_composite note): Replaced the single-sentence note that listed only `rs_semaphore_handles[a][N']` and one barrier with an expanded note that introduces four separate capture-time index variables (`rs_N'`, `barrier_M'_first`, `ag_N'`, `barrier_M'_second`) and lists all four handle groups that must be reset before each replay: `rs_semaphore_handles[a][rs_N']` (3 GlobalSemaphore objects), `barrier_semaphore_handles[a][barrier_M'_first]`, `ag_semaphore_handles[a][ag_N']` (2 GlobalSemaphore objects), and `barrier_semaphore_handles[a][barrier_M'_second]`. Added an explicit warning that omitting the ag and second barrier handles causes silent data corruption.

## Change Log — Agent A Pass 2

- Issue (semaphore_index mapping): In `index.md`, added a note block immediately after the `semaphore_index = 2 if cluster_axis is None else cluster_axis` bullet explaining that this expression is the correct form from `models/common/modules/tt_ccl.py`, whereas the older `models/tt_transformers/tt/ccl.py` uses `2 if not cluster_axis else cluster_axis`, which maps `cluster_axis=0` to `semaphore_index=2` (not 0) due to Python's truthiness of `not 0`. The note instructs readers using the older file that for `cluster_axis=0` workloads the actual `semaphore_index` at runtime is 2, with a cross-reference to Chapter 1. In `what_gets_baked_in.md`, added a caveat note block immediately after the `semaphore_index = cluster_axis` statement clarifying that this equivalence holds only for `models/common/modules/tt_ccl.py` (`is None` check); the older `models/tt_transformers/tt/ccl.py` (`not cluster_axis` check) incorrectly resolves `cluster_axis=0` to `semaphore_index=2`, with a cross-reference to Chapter 1.

# B Review — Chapter 3: Mismatch Analysis — Pass 3

## Issue 1 — `failure_modes.md`: kernel self-reset means the "left non-zero after completion" claim is wrong

**File:** `failure_modes.md`, lines 69–70 ("What happens after the first replay" section)

**What the guide says:**

> After `execute_trace` completes for the first time, the async CCL kernels have run to completion. They have written a non-zero value to the device L1 semaphore word at `ag_semaphore_handles[semaphore_index][N'].address()` and `barrier_semaphore_handles[semaphore_index][M'].address()`. The semaphore word is now non-zero.

**What the kernel actually does:**

The `minimal_default_reader.cpp` kernel (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp`) resets `out_ready_sem` to 0 as its final operation:

```cpp
noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);  // line 295
```

Similarly, the writer kernel resets `barrier_sem` to 0 after the ring barrier:

```cpp
noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);  // minimal_default_writer.cpp line 272
```

If a kernel runs to completion, both semaphore words are reset to 0 by the kernel itself. The guide's claim that "the semaphore word is now non-zero" after a completed replay is incorrect for the `all_gather_async` kernels.

**What the actual failure mechanism is:**

With `execute_trace(..., blocking=False)`, the host returns before device kernel execution finishes. If the host immediately dispatches a second `execute_trace`, that second replay's kernels may launch on device before the first replay's reader kernel has reached its reset at line 295. The second replay's writer then sends atomic increments to `out_ready_sem`, which may collide with the still-running first replay. This is a race condition, not a "stale non-zero value left by a completed kernel."

**Why this matters:**

A reader who internalizes the guide's explanation will believe that fully-completed replays always leave semaphores non-zero, and will always call `reset_global_semaphore_value` regardless of execution mode. This is a wrong conceptual model. The correct model is: if a replay is dispatched with `blocking=True` (host waits for device completion), kernels will have already reset semaphores to 0, and calling `reset_global_semaphore_value` before the next replay is redundant but harmless. If dispatched with `blocking=False`, the reset before the next replay is necessary precisely because the prior replay's kernels may not have completed their self-reset. An implementer using `blocking=True` in a test harness would wrongly conclude that they have reproduced the failure condition described by the guide, when in fact the blocker semantics guarantee the kernels have already reset.

**Fix:** Replace the claim that kernels leave the semaphore non-zero after completion with the correct statement: the `all_gather_async` reader kernel resets `out_ready_sem` to 0 as its final action, and the barrier semaphore is likewise reset by the writer. The failure mode with `blocking=False` execution is that the host may dispatch the next replay before the first replay's kernels have reached their self-reset, causing the second replay's writer to send increments that collide with the first replay's still-running reader. The `reset_global_semaphore_value` call before each replay must be enqueued ahead of the next `execute_trace` to guarantee clean state regardless of whether prior kernels have self-reset.

---

## Issue 2 — `index.md` timeline: `N+1` and `M+1` without modulo in the compile-run block produce invalid index values when N=1 or M=1

**File:** `index.md`, lines 40–41 (the compile-run block of the timeline diagram)

**What the diagram says:**

```
compile run
  get_and_cycle_ag (N)        N+1     M                       handle[N] used, left non-0
  get_and_cycle_barrier (M)   N+1     M+1                     barrier[M] used, left non-0
```

**What is correct:**

The cycling counter for `ag_semaphores_idx` and `barrier_semaphore_idx` is always in `{0, 1}` — it advances modulo 2. After `get_and_cycle_ag_semaphore_handles` reads index `N` and advances, the new value is `(N+1) % 2`, not `N+1`. When N=1, `N+1=2` is not a valid index value; `(N+1)%2=0` is. The rest of the diagram (lines 48–82) consistently and correctly uses the `(N+1)%2` form. Only lines 40–41 drop the modulo, making the notation inconsistent and wrong for the N=1 (or M=1) case.

**Why this matters:**

A reader tracing through the diagram with concrete values N=1, M=1 will see `ag_idx=2` and `barrier_idx=2` after the compile run — values that are out of range for an array of length 2. If that reader then tries to reconcile this with the `(N+1)%2` labels later in the diagram, they must infer the missing modulo themselves. If they do not catch the inconsistency, they may believe the counter reaches index 2 and that a length-3 array is somehow involved, or they may misidentify which slot (0 vs 1) is active post-compile-run when N=1.

**Fix:** On lines 40–41, replace `N+1` with `(N+1)%2` and `M+1` with `(M+1)%2` throughout the compile-run HOST STATE column, consistent with the rest of the diagram.

## Change Log — Agent A Pass 3

- Issue 1 (failure_modes.md kernel self-reset): Replaced the "What happens after the first replay" section to correctly state that both async CCL kernels self-reset their semaphores to 0 as their final action (`out_ready_sem` in the reader, `barrier_sem` in the writer), meaning a completed replay does not leave semaphores non-zero. Added explanation that with `blocking=True` the explicit `reset_global_semaphore_value` is redundant but harmless, while with `blocking=False` it is necessary to prevent a race where the second replay's writer sends atomic increments before the first replay's reader has finished its self-reset. Updated the "skip-through failure mode" section to frame the failure as a race-condition collision between overlapping replays (transient non-zero due to in-flight self-reset) rather than a stale value left by a completed kernel. Updated the "correct requirement" section to state that the explicit reset guarantees CQ FIFO ordering regardless of prior kernel completion state. Updated the composite `use_composite=True` note to frame the omission of ag and second barrier handles as leaving those handles unprotected against the blocking=False race rather than "stale non-zero."
- Issue 2 (index.md compile-run N+1 without modulo): In the compile-run block of the timeline diagram (lines 40–41), replaced `N+1` with `(N+1)%2` in the ag_idx HOST STATE column and `M+1` with `(M+1)%2` in the barrier_idx HOST STATE column, making the notation consistent with the rest of the diagram which already used the modulo form.

# B Review — Chapter 3: Mismatch Analysis — Pass 4

## Issue 1 — `index.md` timeline diagram: `M+1` without modulo in the `begin_trace_capture` block

**File:** `index.md`, line 50 (the `get_and_cycle_ag` row inside the `begin_trace_capture` block)

**What the diagram says:**

```
  get_and_cycle_ag ((N+1)%2)      N    M+1                   handle[(N+1)%2] snapshotted
```

The `M+1` in the barrier_idx column represents the intermediate state of `barrier_semaphore_idx[si]` after the ag cycle call but before the barrier cycle call. This is intended to convey that the barrier index is still at `(M+1)%2` (its post-compile-run value) — but the expression is written as `M+1` without the modulo reduction.

**What is correct:**

The barrier cycling counter is always in `{0, 1}` — it advances modulo 2. The intermediate barrier_idx value at this point in the diagram is `(M+1)%2`. When M=1, `M+1=2`, which is not a valid counter value; `(M+1)%2=0` is. The rest of the diagram consistently uses the `(N+1)%2` / `(M+1)%2` form everywhere the modulo matters. Pass 3 applied the same fix to the compile-run block (lines 40–41) but the identical problem remains on this line in the `begin_trace_capture` block.

**Why this matters:**

A reader tracing through the diagram with concrete values M=1 will see `barrier_idx=2` as an intermediate state inside the capture bracket — a value that is out of range for a length-2 array. They may conclude that the counter reached 2 and attempt to reconcile it with the `(M+1)%2` labels elsewhere, or they may misidentify slot 0 as the active barrier slot when in fact `(M+1)%2 = (1+1)%2 = 0` — which happens to be the same digit but reached by incorrect arithmetic. More practically, a reader implementing counter tracking from the diagram with M=1 would produce an invalid index value if they follow the notation literally.

**Fix:** On line 50 of the timeline diagram, change `M+1` to `(M+1)%2` in the barrier_idx column, consistent with the rest of the diagram and with the Pass 3 fix applied to lines 40–41.

## Change Log — Agent A Pass 4

- index.md begin_trace_capture block: replaced bare M+1 with (M+1)%2 in barrier_idx intermediate column

# B Review — Chapter 3: Mismatch Analysis — Pass 5

## Issue 1 — `index.md` timeline diagram: replay blocks still say semaphores are "left non-zero after completion", contradicting the Pass 3 fix to `failure_modes.md`

**File:** `index.md`, lines 67 and 74 (the `execute_trace (replay 1)` and `execute_trace (replay 2)` blocks)

**What the diagram says:**

```
execute_trace (replay 1)
  ...
  Device L1: handle[(N+1)%2] and barrier[(M+1)%2] left non-zero after completion.

execute_trace (replay 2)
  ...
  Device L1: handle[(N+1)%2] and barrier[(M+1)%2] still non-zero from replay 1.
              (if not reset before replay 2, kernel may skip the wait — corruption)
```

**What is correct (as established by the Pass 3 fix to `failure_modes.md`):**

Pass 3 corrected `failure_modes.md` to state that the `all_gather_async` reader kernel resets `out_ready_sem` to 0 as its final action (`minimal_default_reader.cpp` line 295: `noc_semaphore_set(..., out_ready_sem, 0)`) and the writer kernel resets `barrier_sem` to 0 after the ring barrier (`minimal_default_writer.cpp` line 272). If replay 1 runs to completion before replay 2 is dispatched, both semaphore words are already 0 — they are not left non-zero. The failure mode for `blocking=False` is a race condition where replay 2's kernels launch before replay 1's kernels have reached their self-reset, not a stale non-zero value left by a completed replay.

The `index.md` timeline diagram was not updated to match the Pass 3 correction. It still asserts "left non-zero after completion" (line 67) and "still non-zero from replay 1" (line 74), directly contradicting the corrected `failure_modes.md`.

**Why this matters:**

A reader who reads the timeline diagram (typically before reading `failure_modes.md` in full) internalizes the wrong model: that a completed replay leaves semaphores non-zero. This would cause them to (a) believe a `blocking=True` test harness exposes the same failure as `blocking=False` production code, (b) misunderstand the reason why `reset_global_semaphore_value` is necessary (they would think it clears stale values rather than resolving a race), and (c) reach incorrect conclusions about whether a hang or corruption arises from replay overlap or from a prior completed replay. The diagram is the first concrete artifact a reader sees for this chapter; the wrong model it instills is not corrected until deep in `failure_modes.md`.

**Fix:** In the `execute_trace (replay 1)` block, replace "left non-zero after completion" with a note that with `blocking=False` the host returns before device kernels finish; the semaphore words may be transiently non-zero if the next replay is dispatched before the self-reset at the end of the prior kernels. In the `execute_trace (replay 2)` block, replace "still non-zero from replay 1" with language that captures the race: the prior replay's kernels may not have completed their self-reset to 0 before the next replay's kernels begin incrementing the same semaphore address.

## Change Log — Agent A Pass 5

- index.md replay blocks: updated device semaphore labels from "left non-zero" to reflect blocking=False race condition

# B Review — Chapter 3: Mismatch Analysis — Pass 6

## Issue 1 — `what_gets_baked_in.md`: Warning block incorrectly identifies which slot the trace uses

**File:** `what_gets_baked_in.md`, line 147 (the Warning block in "What Happens on Each Replay")

**What the guide says:**

> If a non-traced `tt_all_reduce` call is made between two trace replays (for example, a prefill step that is not covered by the trace), that call will invoke `get_and_cycle_ag_semaphore_handles(cluster_axis)`, which returns slot `N` (the current host-counter value) and advances the index to `N'`. This means the non-traced call uses slot `N` — **the slot that the trace is also using on every replay.**

**What is actually correct:**

Throughout this chapter, `N` denotes the post-capture host counter value, and `N'` = `(N+1)%2` denotes the capture-time slot baked into the trace. The DRAM command buffer is frozen with the addresses of `ag_semaphore_handles[semaphore_index][N']`, not `ag_semaphore_handles[semaphore_index][N]`. The trace uses slot `N'` on every replay. The host counter after capture is `N` — the opposite slot.

The Warning block states that the non-traced call uses slot `N` — "the slot that the trace is also using" — but the trace uses `N'`, not `N`. Slot `N` and slot `N'` are different double-buffer slots. The first non-traced call uses the slot the trace does NOT use. There is no collision on the first non-traced call.

This is directly contradicted by `failure_modes.md` Case B, which correctly states: "A subsequent `execute_trace` runs on slot `N'`. These are different handles, so there is no immediate collision." The dangerous collision in Case B arises on the second non-traced call, after the counter has advanced to `N'`.

**Why this matters:**

A reader following the Warning block will believe that any single non-traced `tt_all_reduce` call immediately creates a handle collision with the trace. The correct picture is more subtle: the first non-traced call after capture uses the non-trace slot (N) and is safe; the second non-traced call uses the trace slot (N') and collides. A reader acting on the Warning block's wrong model may implement collision detection or reset logic that targets the wrong slot (`N` instead of `N'`), leaving the actual dangerous case (second non-traced call at `N'`) undetected.

**Fix:** In the Warning block, change "the slot that the trace is also using on every replay" to "the slot opposite to the one the trace uses." Note that the collision with the trace's slot (`N'`) occurs on the next non-traced call, after the counter has advanced from `N` to `N'` — cross-reference `failure_modes.md` Case B for the full collision sequence.

## Change Log — Agent A Pass 6

- what_gets_baked_in.md Warning block: corrected slot collision description — first non-traced call uses N (no collision), second uses N' (collision)

# B Review — Chapter 3: Mismatch Analysis — Pass 7

## Issue 1 — `what_gets_baked_in.md` and `index.md`: compile-run device semaphore state incorrectly described as non-zero after completion

**Files:** `what_gets_baked_in.md` line 68; `index.md` lines 40–43 (compile-run block of the timeline diagram)

**What `what_gets_baked_in.md` says:**

> The device L1 semaphore words for `ag_semaphore_handles[semaphore_index][N]` and `barrier_semaphore_handles[semaphore_index][M]` are now non-zero (left by the kernel after completion).

**What `index.md` says:**

```
  get_and_cycle_ag (N)        (N+1)%2     M                       handle[N] used, left non-0
  get_and_cycle_barrier (M)   (N+1)%2     (M+1)%2                 barrier[M] used, left non-0

  (compile run ends; device semaphores left at non-zero values)
```

**What is actually correct:**

Pass 3 of this review established — and the kernel source confirms — that the `all_gather_async` reader kernel resets `out_ready_sem` to 0 as its final operation (`minimal_default_reader.cpp` line 295: `noc_semaphore_set(..., out_ready_sem, 0)`) and the writer kernel resets `barrier_sem` to 0 after the ring barrier (`minimal_default_writer.cpp` line 272). These self-resets occur unconditionally when the kernels run to completion.

The compile run is a normal (non-traced) forward pass executed synchronously. When it returns, device kernels have run to completion and have performed their self-resets. The device L1 semaphore words for slot `N` (AG) and slot `M` (barrier) are therefore 0 after the compile run — not non-zero.

**Why this matters:**

A reader following `what_gets_baked_in.md` concludes that before `begin_trace_capture`, device semaphores for slots N and M are non-zero, and therefore a device-side semaphore reset before `begin_trace_capture` is required to clear stale compile-run values. The correct understanding is the opposite: those slots are already at 0 because the compile run's kernels self-reset. The slots that are relevant to reset before capture are the slots that will actually be used during the capture bracket — slots N' and M' (the post-compile-run counter values) — and the motivation for resetting them before capture is not stale compile-run values but ensuring that whatever may have left those slots non-zero (e.g., a prior capture run) does not corrupt the capture's initial device state. An implementer who follows the incorrect model may believe the compile-run slots (N and M) require resetting before capture, while the correct targets are the capture-time slots (N' and M'). This is directly load-bearing for the semaphore reset logic described in Chapter 4.

**Fix:** In `what_gets_baked_in.md` line 68, replace "are now non-zero (left by the kernel after completion)" with "are now 0: the kernels self-reset `out_ready_sem` and `barrier_sem` to 0 as their final actions (see `failure_modes.md` Problem 2)." In `index.md` lines 40–43, replace "left non-0" with "self-reset to 0 on completion" for both the ag and barrier rows, and replace the annotation "(compile run ends; device semaphores left at non-zero values)" with "(compile run ends; device semaphores self-reset to 0 by kernels on completion)".

## Change Log — Agent A Pass 7

- what_gets_baked_in.md: fixed post-compile-run device semaphore state claim
- index.md: fixed post-compile-run device semaphore labels in compile-run block

# B Review — Chapter 3: Mismatch Analysis — Pass 8

## Issue 1 — `failure_modes.md`: kernel wait protocol description is wrong — the reader kernel waits for an incrementing positive threshold, not for 0

**File:** `failure_modes.md`, lines 62–65 ("What the kernel expects" section)

**What the guide says:**

> Before the kernel starts processing, it waits for the semaphore at `semaphore.address()` to reach 0 (initial state, indicating the buffer is available) or waits for a specific completion count to be written by peer kernels.
> After the kernel finishes its work, it writes a non-zero value to the semaphore (for example, `ring_size` or a direction count) to signal completion to the next stage.
> The semaphore is designed to start at 0. The kernel's wait condition is defined relative to this initial state.

**What the kernel actually does:**

The `minimal_default_reader.cpp` kernel initializes a local variable `sem_target = 0` (line 160) and waits using `noc_semaphore_wait_min(..., out_ready_sem, sem_target + 1)` (lines 216–217, 269–270), incrementing `sem_target` after each wait. The first wait checks that `out_ready_sem` has reached 1 (not 0). Subsequent waits check for 2, 3, etc. The writer kernel sends atomic increments to the reader's `out_ready_sem` address, which the reader reads as an accumulating count. The reader's final action is `noc_semaphore_set(..., out_ready_sem, 0)` (line 295), resetting the counter for the next use.

The claim that the kernel "waits for the semaphore to reach 0 (initial state, indicating the buffer is available)" is factually wrong for `out_ready_sem`. The kernel never waits for 0 — 0 is the starting value before the protocol runs, and the final value after the reset, but not a wait target.

The second bullet — "after the kernel finishes its work, it writes a non-zero value to the semaphore" — is also incorrect as a description of the terminal state: the reader's final write to `out_ready_sem` is the reset to 0 (line 295), not a non-zero signal. The writer does send atomic increments (non-zero contributions) during its data movement phase, but those increments are intermediate signals, not the final state the kernel "writes after finishing."

**Why this matters:**

A reader learning the semaphore protocol from this section will build the wrong mental model: they will believe the kernel blocks on 0 → unblocks when written to non-zero → finishes and leaves a non-zero value. The correct model is: kernel tracks `sem_target` locally, blocks until the semaphore reaches `sem_target + 1` (incremented by the writer's atomic increments), advances `sem_target`, and finally resets the semaphore to 0. This distinction is load-bearing: it explains why `reset_global_semaphore_value(handle, 0)` is the correct pre-replay reset value (it resets the accumulator to 0 so that the next replay's reader starts waiting for 1 again, as expected by `sem_target=0`). A reader who believes the kernel waits for 0 cannot understand why the reset is necessary at all, since a completed kernel already leaves the semaphore at 0 (via self-reset). The correct framing makes the `blocking=False` race mechanically precise: the second replay's writer increments `out_ready_sem` from some intermediate value (e.g., 1) to 2 while the first replay's reader is still waiting for its own `sem_target+1`; if the reader has already advanced past that count but not yet reset, the second writer's increment shifts the accumulated count in a way that may satisfy or block the second replay's reader at the wrong phase.

## Change Log — Agent A Pass 8

- failure_modes.md: corrected kernel semaphore wait protocol — reader waits for ≥1 (not 0), terminal action is reset to 0 (not write non-zero)

# B Review — Chapter 3: Mismatch Analysis — Pass 9

## Issue 1 — `failure_modes.md` Case B diagram: "left non-zero" annotations contradict the kernel self-reset established in Pass 3 and misattribute the corruption cause

**File:** `failure_modes.md`, lines 38–43 (the collision-sequence diagram in Case B of Problem 1)

**What the diagram says:**

```
trace replay 1   → uses slot N'   (device L1 for N' left non-zero)
non-traced call  → uses slot N    (counter advances to N')
trace replay 2   → uses slot N'   (device L1 for N' still non-zero from replay 1 — corruption)
non-traced call  → uses slot N'   (counter advances to N; same handle as trace replay 2 — collision)
```

**What is actually correct:**

Pass 3 of this review established — and the kernel source confirms — that the `all_gather_async` reader kernel resets `out_ready_sem` to 0 as its terminal action (`minimal_default_reader.cpp` line 295) and the writer resets `barrier_sem` to 0 (`minimal_default_writer.cpp` line 272). A completed replay does not leave semaphore words non-zero. The phrase "device L1 for N' left non-zero" (replay 1 row) and "device L1 for N' still non-zero from replay 1" (replay 2 row) assert a completed state of non-zero, which is inconsistent with the self-reset behavior described in Problem 2's corrected "What happens after the first replay" section.

Furthermore, the corruption in Case B is caused by handle collision: at the fourth step in the sequence, a non-traced call dispatches a CCL op to the same physical semaphore handle (`N'`) that is also used by the trace. When both a non-traced kernel and a traced kernel run against the same semaphore address concurrently, their atomic increments and wait conditions interfere. The "(device L1 for N' still non-zero from replay 1 — corruption)" annotation in the replay 2 row attributes the corruption to residual non-zero semaphore state from a prior replay rather than to simultaneous handle use by two independent kernel dispatches. A reader who has learned the correct model from Problem 2 will find these annotations contradictory and will be left uncertain whether Case B's corruption is caused by the device semaphore value problem or the handle collision problem.

**Why this matters:**

A reader who internalizes "device L1 for N' left non-zero" in the Case B diagram will believe that even in Case B (host counter problem), the device semaphore is always non-zero after a replay. This directly contradicts the corrected model from Pass 3: completed kernels self-reset to 0, and the non-zero state is a transient race condition specific to `blocking=False` concurrent dispatch. An implementer testing Case B behavior with `blocking=True` (where self-resets are guaranteed to complete before the next dispatch) would observe that the device semaphore is 0 at replay 2, yet still see corruption — because the corruption cause in Case B is handle collision, not stale device state. If the diagram had attributed the corruption to handle collision, the implementer would understand why `blocking=True` does not eliminate Case B's corruption.

**Fix:** Remove the "(device L1 for N' left non-zero)" parenthetical from the replay 1 row and "(device L1 for N' still non-zero from replay 1 — corruption)" from the replay 2 row. Replace the replay 2 annotation with a note that accurately identifies the corruption cause: the second non-traced call (step 4) dispatches a CCL kernel against handle `N'` — the same handle baked into the trace — and the two kernels' semaphore signals interfere. If device semaphore values are also not reset, a compounding Problem 2 race is possible, but that is a separate concern; Case B's core corruption is the simultaneous use of the same handle by uncoordinated kernel dispatches.

## Change Log — Agent A Pass 9

- failure_modes.md Case B diagram: removed incorrect "device L1 left non-zero" annotations; replaced with accurate handle collision description

# B Review — Chapter 3: Mismatch Analysis — Pass 10

## Issue 1 — `what_gets_baked_in.md` lines 95 and 104: `ag_semaphore_handles[semaphore_index][N']` is a list of 2 objects, not a single object with `.address()`

**File:** `what_gets_baked_in.md`, lines 95 and 104

**What the file says (line 95):**

> `ag_semaphore_handles[semaphore_index][N'].address()` into the RTA slots for `out_ready_semaphore` (slots `[2]` on the reader and `[3]` on the writer for each direction).

**What the file says (line 104):**

> specifically `ag_semaphore_handles[semaphore_index][N'].address()` and `barrier_semaphore_handles[semaphore_index][M'].address()` — are copied verbatim into `ordered_trace_data`.

**What is actually correct:**

`ag_semaphore_handles[semaphore_index][N']` is a **list of 2** `GlobalSemaphore` objects — confirmed by `TT_CCL.__init__` in both `models/tt_transformers/tt/ccl.py` and `models/common/modules/tt_ccl.py`:

```python
self.ag_semaphore_handles[i].append(
    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
)
```

The Python list does not have an `.address()` method. The C++ `override_runtime_arguments` receives the list as `multi_device_global_semaphore` and accesses each element by direction index via `semaphore.at(dir).address()` (confirmed at `all_gather_async_default_program_factory.cpp` line 829). Specifically, `semaphore.at(0).address()` and `semaphore.at(1).address()` are each written into separate per-direction per-core RTA slots. Two distinct L1 addresses — one per direction — are baked into the trace for the AG semaphore, not one.

`failure_modes.md` line 97 correctly describes this as "a list of 2 `GlobalSemaphore` objects" that must each be reset. The `what_gets_baked_in.md` notation `ag_semaphore_handles[semaphore_index][N'].address()` contradicts this and implies a single address.

**Why this matters:**

A reader building the reset logic from `what_gets_baked_in.md` will call `ttnn.reset_global_semaphore_value` once for the AG handle slot, targeting what they believe is a single `GlobalSemaphore`. The correct implementation requires iterating over both elements in the list — `ag_semaphore_handles[semaphore_index][N'][0]` and `ag_semaphore_handles[semaphore_index][N'][1]` — and calling `reset_global_semaphore_value` on each. An implementer who resets only one of the two will leave one direction's semaphore unreset, causing the per-direction reader kernel for that direction to observe a stale count at the start of the next replay, producing silent data corruption for that direction's data movement phase.

**Fix:** Replace `ag_semaphore_handles[semaphore_index][N'].address()` (both occurrences) with a description that makes clear this is a list of 2 handles, each with its own address written per direction — e.g., "`ag_semaphore_handles[semaphore_index][N'][dir].address()` for each direction `dir` in `{0, 1}`".

## Change Log — Agent A Pass 9

- failure_modes.md Case B diagram: removed incorrect "device L1 left non-zero" annotations; replaced with accurate handle collision description

## Change Log — Agent A Pass 10

- what_gets_baked_in.md: corrected ag_semaphore_handles notation — it is a list of 2 GlobalSemaphore objects, accessed per direction; two addresses baked into trace

# B Review — Chapter 3: Mismatch Analysis — Pass 11

## Issue 1 — `failure_modes.md` Case B diagram: step 3 annotation attributes the wrong problem as the corruption cause

**File:** `failure_modes.md`, lines 37–42 (the collision-sequence diagram in Case B)

**What the diagram says:**

```
trace replay 1   → uses slot N'
non-traced call  → uses slot N    (counter advances to N')
trace replay 2   → uses slot N'   (two kernels share slot N' L1 address — concurrent increments corrupt sem_target counting)
non-traced call  → uses slot N'   (non-traced kernel dispatched to slot N' — same slot as trace)
```

**What is actually correct:**

Case B illustrates Problem 1: the host-counter mismatch causing a non-traced call to use the same handle as the trace. The actual Problem 1 collision occurs at step 4, where the second non-traced call uses slot N' — the same slot the trace uses. The step 4 annotation correctly identifies this.

The step 3 annotation "(two kernels share slot N' L1 address — concurrent increments corrupt sem_target counting)" is describing a Problem 2 race: trace replay 2 is dispatched (with `blocking=False`) while trace replay 1 may still be running on slot N'. These are two *trace* kernels concurrently using the same handle — the overlap between consecutive trace replays that `reset_global_semaphore_value` is designed to prevent. This is a Problem 2 (device semaphore) issue, not a Problem 1 (host counter) issue.

No non-traced call has yet used slot N' at step 3. The step 3 "two kernels" are both from the trace, not from a trace plus a non-traced call. The Case B-specific corruption (host counter collision) does not occur until step 4.

**Why this matters:**

A reader learning Case B from this diagram will see the step 3 annotation and conclude that the Problem 1 collision (host counter mismatch) manifests at trace replay 2, caused by concurrent kernel use of the slot. They will then infer that fixing Problem 2 (adding `reset_global_semaphore_value` before each replay) eliminates Case B corruption, because `reset_global_semaphore_value` resolves the concurrent replay race identified in the step 3 annotation. This is wrong: `reset_global_semaphore_value` eliminates the step 3 Problem 2 race but does not prevent the step 4 Problem 1 collision, which requires host-counter correction. An implementer who adds only device semaphore resets and observes that step 3's race is gone may incorrectly conclude Case B is fully resolved, while the step 4 collision (the actual Case B issue) remains and produces silent data corruption when the calling pattern reaches that point.

**Fix:** Replace the step 3 annotation with language that identifies it as a Problem 2 blocking=False race (two consecutive trace replays sharing the same handle slot because no reset was performed before replay 2), distinct from the Problem 1 collision at step 4. For example: `trace replay 2 → uses slot N'  (Problem 2 race: replay 1 may still be running on N'; missing reset_global_semaphore_value)`. The step 4 annotation can remain as is, since it correctly identifies the Problem 1 handle-collision cause.

## Change Log — Agent A Pass 11

- failure_modes.md Case B diagram: removed incorrect step 3 annotation; moved collision annotation to step 4; added clarifying note that Case B requires host-counter correction

# B Review — Chapter 3: Mismatch Analysis — Pass 12

## Issue 1 — `failure_modes.md` summary table: Problem 2 row still uses the pre-Pass-3 incorrect causal description

**File:** `failure_modes.md`, line 121 (the "Summary of Failure Modes" table)

**What the table says:**

| Device semaphore values not reset before replay | Kernels see stale non-zero value; may skip wait condition | Silent data corruption or hang |

**What is actually correct (as established by Pass 3 and Pass 8 corrections to this same file):**

The body of `failure_modes.md` — specifically "What happens after the first replay" (lines 73–79) and "The skip-through failure mode" (lines 81–91) — was corrected in Pass 3 and Pass 8 to state:

- The async CCL kernels self-reset `out_ready_sem` to 0 as their terminal action (`minimal_default_reader.cpp` line 295) and `barrier_sem` to 0 (`minimal_default_writer.cpp` line 272). A completed replay does not leave semaphore words non-zero.
- The failure mechanism with `blocking=False` is a race: the second replay is dispatched before the first replay's reader kernel has reached its self-reset. The second replay's writer sends atomic increments to the semaphore address while the first replay's reader has not yet zeroed it. The accumulated count satisfies the second replay's reader's first `noc_semaphore_wait_min` condition immediately — a skip-through due to overlapping execution, not a stale value from a prior completed replay.

The summary table row still says "stale non-zero value" and "skip wait condition," which is the pre-Pass-3 model that was corrected. The summary table directly contradicts the body text of the same file.

**Why this matters:**

The summary table functions as a quick-reference checklist. A reader who scans the table (as is common for reference tables at the end of an analysis section) learns: "the failure is caused by a stale non-zero value left by a completed replay." This wrong model leads to two concrete implementation errors:

1. The reader may believe that using `blocking=True` (where kernels complete their self-reset before the host returns) eliminates the need for `reset_global_semaphore_value` — because if the failure is a stale value from a completed replay, and `blocking=True` guarantees the kernels have self-reset to 0, then there is no stale value. The correct model — that the race only exists with `blocking=False` — supports this same conclusion, but the reader's reasoning is grounded in a wrong causal mechanism and will break down if they later switch to `blocking=False` without reconsidering their reset logic.

2. More directly: the reader learns to diagnose "skip wait condition" as the symptom. The corrected body text explains the skip-through as: the second replay's writer increments `out_ready_sem` from a transiently non-zero value (left by an in-flight first replay) to some count that satisfies the second replay's reader's `sem_target + 1` threshold immediately. The skip-through is therefore a threshold-satisfaction event on `out_ready_sem`, not a "wait for 0" skip. A reader who has internalized the correct protocol from Pass 8 (reader waits for ≥1, not for 0) will find "may skip wait condition" ambiguous — skip which wait condition? The table's description does not match the corrected kernel protocol.

**Fix:** Replace the summary table row's symptom column with language consistent with the corrected body: "With `blocking=False`, the prior replay's reader may not have self-reset the semaphore to 0 before the next replay's writer sends atomic increments; the accumulated count may satisfy the next replay's reader's wait threshold prematurely, producing incorrect output." Category can remain "Silent data corruption or hang."

## Change Log — Agent A Pass 12

- failure_modes.md summary table: corrected Problem 2 symptom — replaced "stale non-zero value" with accurate blocking=False overlapping-replay description

# B Review — Chapter 3: Mismatch Analysis — Pass 13

No feedback — chapter approved.

# B Review — Chapter 3: Mismatch Analysis — Pass 14

No feedback — chapter approved.
