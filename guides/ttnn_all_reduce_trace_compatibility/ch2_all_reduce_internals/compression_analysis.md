# Change Log — B Review Pass 1 Fixes

- **Item 1 — `call_chain.md` §4.3: `use_composite_reduce_scatter` predicate inverted.** Rewrote the §4.3 description to correctly state that when the scatter dimension is not evenly divisible by `num_devices`, the function returns `false` (do NOT use composite). The composite path is only considered after the divisibility check passes; the guide previously had this logic inverted, which would have led a reader to select the wrong path for non-divisible dimensions. Verified against `composite_common.cpp` lines 46–48.

- **Item 2 — `composite_path.md` §2: `concat` described as host-side.** Corrected the description of `ttnn::concat` in `composite_all_gather` from "host side" to a device op. The call at `composite_common.cpp` line 379 is `ttnn::concat(broadcasted_tensors, gather_dim)`, which executes on-device; allocation and command recording for the concat output occur on the device inside the trace capture window.

- **Item 3 — `call_chain.md` §3: async `all_gather` guard omits `barrier_semaphores.size() == 2` constraint.** Added an implementation note to §3 documenting the `TT_FATAL(barrier_semaphores.value().size() == 2, ...)` assertion (line 362 of `all_reduce_async.cpp`) that fires immediately inside the async `all_gather` guard. Callers implementing the async path must supply exactly 2 elements: index `[0]` for `reduce_scatter_minimal_async` and index `[1]` for `all_gather_async`.

- **Item 4 — `contrast_with_async_variants.md` §1: synchronous fallback scope incorrect.** Clarified that the `has_value()` guard and synchronous fallback only exist in the cluster-axis overload of `ExecuteAllReduceAsync::invoke` (lines 258–399). The non-cluster-axis overload (lines 152–256) takes its semaphore arguments as non-optional and always calls async ops unconditionally — there is no synchronous fallback in that overload. Added a scope note to prevent readers from incorrectly concluding the fallback is available in both overloads.

- **Item 5 — `call_chain.md` §4.3: early-exit for non-divisible dimensions omitted.** Restructured the `use_composite_reduce_scatter` bullet list to lead with the early-exit condition (`if (input_shape[scatter_dim] % num_devices != 0) return false`) before presenting the tile-alignment checks. The predicate only proceeds to the tile-alignment and row-major checks after confirming divisibility; the guide previously presented tile-alignment as a top-level condition without noting this dependency. Verified against `composite_common.cpp` lines 46–59.

# Change Log — B Review Pass 2 Fixes

- **Item 1 — `reduce_scatter_all_gather_path.md` §3: incorrect claim that `scattered_tensor` has a stable DRAM address during trace replay.** Rewrote §3 to correct the mechanistic error: TTNN trace captures metal-level command buffers and does NOT pin dynamically-allocated DRAM buffer addresses at capture time. The `scattered_tensor` is a dynamic DRAM intermediate that the allocator re-invokes on each replay, potentially landing at a different offset if the free-list state differs. The section now explicitly states that `scattered_tensor` poses the same trace incompatibility risk as the composite-path intermediates, and that a caller must pre-allocate it as a persistent buffer (via `optional_output_tensor`) before `ttnn.begin_trace_capture`. Also updated the §4 summary table and verdict to reflect this conditional compatibility. Verified against `all_reduce_async.cpp` lines 330–358 (no pre-allocation of `scattered_tensor` in the synchronous fallback).

- **Item 2 — `composite_path.md` §2: reshape snippet used the non-cluster-axis overload's hard-coded 4-element form.** Replaced the hard-coded `ttnn::Shape({1, initial_shape[0] * initial_shape[1], initial_shape[2], initial_shape[3]})` (lines 194–196, non-cluster-axis overload) with the correct dynamic `SmallVector` form from the cluster-axis overload (lines 300–305): `ttnn::SmallVector<uint32_t> ag_shape_vec(initial_shape.rank())` with `std::copy` for dimensions beyond index 1 and explicit assignment of `ag_shape_vec[0] = 1`, `ag_shape_vec[1] = initial_shape[0] * initial_shape[1]`, followed by `ttnn::reshape(interleaved_tensor, ttnn::Shape(ag_shape_vec))`. Added a note explaining the overload difference and that for rank-4 tensors the results are identical but for any other rank the expressions differ. Verified against `all_reduce_async.cpp` lines 300–305.

- **Item 3 — `reduce_scatter_all_gather_path.md` §1: guard description described synchronous fallback as "taken when semaphore args are `std::nullopt`" (implying all null).** Updated the code comments and surrounding prose to precisely state the actual guard condition: `rs_global_semaphores.has_value() && barrier_semaphores.has_value()` requires BOTH to be present for the async branch. If EITHER is absent (`std::nullopt`), the conjunction is `false` and the code falls through to the synchronous op silently. The identical asymmetry applies to the `all_gather` guard (`ag_global_semaphores.has_value() && barrier_semaphores.has_value()`). Verified against `all_reduce_async.cpp` lines 331 and 361.

- **Item 4 — `contrast_with_async_variants.md` §2 and §5: comparison table conflated the two overloads of `all_reduce_async`.** Rewrote §2 to distinguish the two overloads explicitly: the non-cluster-axis overload (lines 21–31 of `all_reduce_async.hpp`) takes `const std::vector<GlobalSemaphore>&` (non-optional, no synchronous fallback), while the cluster-axis overload (lines 33–44) takes `const std::optional<std::vector<GlobalSemaphore>>&` (optional, with synchronous fallback when either semaphore arg is absent). Expanded the comparison table in §5 to four columns, separating the two `all_reduce_async` overloads so readers understand which semaphore type signature they will encounter depending on their call site. Added a developer warning about compile-time failures if the wrong overload is targeted. Verified against `all_reduce_async.hpp` lines 21–44.

- **Item 5 — `index.md` summary table verdict: non-composite path claimed "Compatible" without qualification.** Updated the Key finding callout block in `index.md` to replace the unqualified "compatible" claim for the non-composite path with a conditional statement: the path is conditionally compatible, requiring pre-allocation of `scattered_tensor` as a persistent buffer before trace capture begins. Without this precaution, the non-composite path carries the same address-instability risk as the composite path. Also updated Learning objective 3 to reflect that the non-composite path requires this pre-allocation condition to be safe. Consistent with the fix to Issue 1.

# Change Log — B Review Pass 3 Fixes

- **Item 1 — `composite_path.md` §4: `gather_tensor` shape formula was wrong (wrong axis and wrong formula).** Corrected the stated shape from `[1, num_devices * initial_shape[1], H, W]` to `[num_devices, initial_shape[0] * initial_shape[1], H, W]`. `composite_all_gather` is called with `composite_dim = 0`; it calls `ttnn::prim::all_broadcast` to replicate the reshaped tensor (shape `[1, initial_shape[0]*initial_shape[1], H, W]`) across all `num_devices` devices, then `ttnn::concat(broadcasted_tensors, gather_dim)` along dim 0. Concatenating `num_devices` tensors of shape `[1, ...]` along dim 0 produces `[num_devices, ...]`, not `[1, num_devices * initial_shape[1], ...]`. Added a concrete example: a decode input `[1, 1, 32, H]` on an 8-device T3K yields `gather_tensor` of shape `[8, 1, 32, H]`, not `[1, 8, 32, H]`. Verified against `composite_common.cpp` lines 375–379 and `all_reduce_async.cpp` lines 295–313.

- **Item 2 — `contrast_with_async_variants.md` §1: guard condition omitted `barrier_semaphores.has_value()`.** Expanded the guard description to include `barrier_semaphores.has_value()` as a mandatory component of both the reduce-scatter guard (line 331: `rs_global_semaphores.has_value() && barrier_semaphores.has_value()`) and the all-gather guard (line 361: `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`). The previous prose only mentioned `rs_global_semaphores.has_value()` and `ag_global_semaphores.has_value()`, which would lead a developer to incorrectly believe they need only supply the respective CCL semaphore to enter the async branch; omitting `barrier_semaphores` alone would silently fall through to the synchronous path. Verified against `all_reduce_async.cpp` lines 331 and 361.

# Change Log — B Review Pass 4 Fixes

- **Item 1 — `composite_path.md` §2: `local_sum` snippet called `ttnn::moreh_sum` with undefined variable `input_tensor`.** Corrected the variable name in the `moreh_sum` call from `input_tensor` to `gathered_tensor`, which is the parameter declared in the `local_sum` function signature. The previous snippet would produce a compilation error if used as written, and could not be trusted as a reference for implementing `local_sum`.

- **Item 2 — `composite_path.md` §1 bullet 2: `use_composite_reduce_scatter` wrongly attributed as the predicate that fires for non-divisible scatter dimensions.** Rewrote the bullet to correctly state that `use_composite_reduce_scatter` returns `true` only when the scatter dimension IS evenly divisible by `num_devices` but the per-device slice is not tile-aligned or the tensor is row-major. Added an explicit note that when the scatter dimension is not evenly divisible, `use_composite_reduce_scatter` returns `false` (do NOT use composite), and that composite-path selection for the non-divisible case is instead triggered by `dim != composite_dim` (bullet 3). This matches `call_chain.md` §4.3 and `composite_common.cpp` lines 46–48.

- **Item 3 — `contrast_with_async_variants.md` §1: trace compatibility claim omitted `scattered_tensor` pre-allocation requirement.** Extended the trace compatibility statement to explicitly require that `scattered_tensor` be pre-allocated as a persistent buffer before `ttnn.begin_trace_capture` and passed as `optional_output_tensor` to `ttnn.reduce_scatter`. Without this pre-allocation the intermediate is re-allocated by the standard DRAM allocator on each replay and may land at a different address, causing silent data corruption or hardware faults. Consistent with `reduce_scatter_all_gather_path.md` §4.

---

# Compression Analysis: Chapter 2 — ttnn.all_reduce Internal Architecture — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~937 lines
- Estimated post-compression line count: ~800 lines
- Estimated reduction: ~15%

## CRUCIAL Suggestions

### [contrast_with_async_variants.md] ~lines 11–69
**Issue:** Section 1 of `contrast_with_async_variants.md` duplicates content already present in `call_chain.md` and `reduce_scatter_all_gather_path.md`. Specifically:
- The Python `ttnn.all_reduce(...)` call snippet (lines 17–24) is identical to `call_chain.md` §1 lines 27–34.
- The C++ `::ttnn::experimental::all_reduce_async(... std::nullopt ...)` block (lines 31–39) is identical to `call_chain.md` §2 lines 61–73.
- The detailed `has_value()` guard explanation (lines 43–51) repeats `reduce_scatter_all_gather_path.md` §1 lines 72–77 nearly verbatim.
- The "Trace compatibility" paragraph (lines 61–69) restates `reduce_scatter_all_gather_path.md` §4 conclusion with almost identical wording.

**Suggestion:** Replace §1 with a 3–4 sentence summary that states the key facts (no semaphore args, `std::nullopt` forwarded, sync fallback taken, no persistent state) and cross-references `call_chain.md` §2 and `reduce_scatter_all_gather_path.md` §1 for the code details. Remove both code blocks from this file entirely. Estimated saving: ~40 lines.

### [reduce_scatter_all_gather_path.md] ~lines 137–145
**Issue:** The two-bullet sub-list ("Outside a trace" / "Inside a trace capture", lines 139–145) immediately restates the same mechanism explained in the paragraph directly above it (lines 127–136). The bullets add no new information; they re-label and echo the preceding prose.

**Suggestion:** Delete the two-bullet sub-list (lines 137–145). The Warning callout that follows (lines 147–153) already provides the actionable consequence; the bullets are redundant scaffolding. Estimated saving: ~9 lines.

---

## MINOR Suggestions

### [index.md] ~lines 12–34
**Issue:** The "Summary answer — the semaphore question" block (lines 12–34) anticipates and restates conclusions that are fully developed in `call_chain.md` §4.5/§5, `composite_path.md` §4, and `reduce_scatter_all_gather_path.md` §4. Readers who proceed to the subfiles encounter the same material again, including the identical "Key finding" language.

**Suggestion:** Shorten the summary block to 3–4 sentences stating the top-line verdict (non-composite synchronous path taken on T3K; no external semaphores; conditional trace compatibility requiring `scattered_tensor` pre-allocation). Replace the current "Key finding" callout with a single sentence and a link to `composite_path.md` and `reduce_scatter_all_gather_path.md`. Estimated saving: ~12 lines.

### [call_chain.md] ~lines 163–165
**Issue:** The parenthetical in §4.3 — "This is the leading early-exit; the predicate never proceeds to the tile-alignment check in this case." — restates what the immediately preceding sentence already makes explicit ("Returns `false` (do NOT use composite) immediately when: the scatter dimension size is not evenly divisible..."). The word "immediately" already signals early exit.

**Suggestion:** Delete the parenthetical sentence. The leading bullet and "Returns `false` immediately when" phrasing carry the full meaning. Estimated saving: ~1 line.

### [composite_path.md] ~lines 136–146
**Issue:** Section 3 ("Semaphore state in the composite path") states there is no persistent semaphore state in the composite path. This conclusion is repeated in `reduce_scatter_all_gather_path.md` §2 (lines 100–101) and again in `contrast_with_async_variants.md` §4 bullet 1, creating three near-identical paragraphs across three files.

**Suggestion:** Retain the statement in `composite_path.md` §3 as the canonical location. In `reduce_scatter_all_gather_path.md` §2 and `contrast_with_async_variants.md` §4, reduce to a single cross-reference sentence ("As with the composite path, no persistent semaphore state is carried between calls; see `composite_path.md` §3."). Estimated saving: ~8 lines across the two non-canonical files.

### [reduce_scatter_all_gather_path.md] ~lines 83–108
**Issue:** Section 2 includes a block-quote note (lines 104–108) that previews the contrast with async CCL ops. This note essentially summarises what `contrast_with_async_variants.md` covers in full. The note is not load-bearing here because `contrast_with_async_variants.md` is the dedicated file for this comparison.

**Suggestion:** Replace the block-quote note with a single sentence cross-reference: "For the contrast with async CCL ops and the cycling semaphore pattern, see `contrast_with_async_variants.md`." Estimated saving: ~4 lines.

---

## Load-Bearing Evidence

- `call_chain.md` line ~196–207: The predicate truth-table for T3K decode tensors is load-bearing because it is the only location in the chapter that consolidates all four predicate values in a single scannable reference; it cannot be cut without losing the at-a-glance summary.
- `composite_path.md` line ~151–187: The three-item numbered list explaining which specific allocations (`reshaped_tensor`, `gather_tensor`, `sum_tensor`) violate trace requirements is load-bearing because it provides the mechanistic evidence behind the incompatibility verdict; collapsing it would lose the per-allocation detail needed to implement a fix.
- `reduce_scatter_all_gather_path.md` line ~159–165: The §4 summary table is load-bearing because it is the only location that presents the non-composite path's trace-compatibility verdict in tabular form with the pre-allocation condition explicitly stated; it is referenced from `index.md` and `contrast_with_async_variants.md`.
- `contrast_with_async_variants.md` line ~257–266: The four-column comparison table (§5) is load-bearing because it is the only place that places `ttnn.all_reduce`, both `all_reduce_async` overloads, and `reduce_scatter_minimal_async` side-by-side across eight properties; collapsing or merging columns would lose the overload-distinction information that was explicitly added in Agent B Pass 2.
- `contrast_with_async_variants.md` line ~195–217: The two-point explanation of why cycling semaphores are necessary (residual state + in-flight overlap) is load-bearing because it provides the causal reasoning that justifies the double-buffering design; it does not appear in any other file.

## VERDICT
- Crucial updates: yes

---

# Agent A Change Log — C Pass 1 Compression

- **`contrast_with_async_variants.md` §1 (~lines 11–69):** Replaced the duplicated Python call snippet, C++ `std::nullopt` forwarding block, `has_value()` guard explanation, and trace-compatibility paragraph with a 8-sentence summary paragraph plus cross-reference links to [`call_chain.md` §2](./call_chain.md) and [`reduce_scatter_all_gather_path.md` §1 and §4](./reduce_scatter_all_gather_path.md). Estimated saving: ~40 lines.
- **`reduce_scatter_all_gather_path.md` ~lines 137–145:** Removed the two-bullet sub-list ("Outside a trace" / "Inside a trace capture") that re-explained the dynamic-allocation mechanism stated in the paragraph directly above it. The Warning callout that follows was retained unchanged. Estimated saving: ~9 lines.

---

# Compression Analysis: Chapter 2 — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~887 lines
- Estimated post-compression line count: ~870 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions
(re-check of Pass 1 CRUCIAL items only)

### [contrast_with_async_variants.md] §1 — RESOLVED
**Pass 1 issue:** ~40 lines of duplicated Python snippet, C++ `std::nullopt` block, `has_value()` guard prose, and trace-compatibility paragraph repeated material from `call_chain.md` §2 and `reduce_scatter_all_gather_path.md` §1/§4.

**Re-check verdict:** Resolved. The current §1 (lines 11–25) is a compact 8-sentence summary with two cross-reference links and no code blocks. No duplicated content remains. No further action needed.

### [reduce_scatter_all_gather_path.md] ~lines 137–145 — RESOLVED
**Pass 1 issue:** A two-bullet sub-list ("Outside a trace" / "Inside a trace capture") re-explained the dynamic-allocation mechanism stated in the paragraph immediately above it.

**Re-check verdict:** Resolved. The current §3 (lines 113–143) moves directly from the dynamic-allocation explanation to the Warning callout with no intervening bullet list. The redundant bullets are gone. No further action needed.

---

## MINOR Suggestions

### [index.md] ~lines 12–34
**Issue:** The "Summary answer — the semaphore question" block (22 lines) front-loads both the composite-path and non-composite-path verdicts, including the conditional compatibility caveat, before the reader has seen any of the evidence. The same conclusions appear in full in `composite_path.md` §4 and `reduce_scatter_all_gather_path.md` §4. The "Key finding" callout inside the block (lines 23–34) is 12 lines long and restates nearly verbatim what those subfiles conclude.
**Suggestion:** Trim the "Key finding" callout to 3–4 sentences covering only the top-line verdict (composite path dynamically allocates — incompatible; non-composite synchronous fallback — conditionally compatible pending `scattered_tensor` pre-allocation) and replace the remaining detail with a pointer to the subfiles. Estimated saving: ~8 lines.

### [call_chain.md] ~lines 163–165
**Issue:** The parenthetical sentence "This is the leading early-exit; the predicate never proceeds to the tile-alignment check in this case." restates what "Returns `false` (do NOT use composite) immediately when:" already signals. The word "immediately" in the bullet leader already communicates early exit.
**Suggestion:** Delete the parenthetical sentence. Estimated saving: ~1 line.

---

## Load-Bearing Evidence

- `call_chain.md` line ~196–207: The four-row predicate truth-table for T3K decode tensors is load-bearing because it is the only place in the chapter that consolidates all four predicate values in one scannable view; removing it would force readers to re-derive the combined outcome from four separate subsections.
- `composite_path.md` line ~151–187: The three-item numbered list naming `reshaped_tensor`, `gather_tensor`, and `sum_tensor` as the specific allocations that violate trace requirements is load-bearing; it provides per-allocation mechanistic evidence behind the incompatibility verdict and cannot be collapsed without losing the implementation-relevant detail.
- `reduce_scatter_all_gather_path.md` line ~148–165: The §4 summary table is load-bearing because it is the only place the non-composite path's conditional trace-compatibility verdict is presented in tabular form with the `scattered_tensor` pre-allocation condition stated explicitly; referenced from both `index.md` and `contrast_with_async_variants.md`.
- `contrast_with_async_variants.md` line ~208–222: The four-column comparison table (§5) is load-bearing because it is the only location that aligns `ttnn.all_reduce`, both `all_reduce_async` overloads, and `reduce_scatter_minimal_async` side-by-side across eight properties; merging or removing columns would destroy the overload-distinction information added in Agent B Pass 2.
- `contrast_with_async_variants.md` line ~150–172: The two-point explanation of why cycling semaphores are necessary (residual state from the previous replay + in-flight overlap between successive replays) is load-bearing; the causal reasoning for double-buffering does not appear in any other file and is required for a reader to understand why depth-2 cycling is sufficient.

## VERDICT
- Crucial updates: no
