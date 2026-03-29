# Change Log — B Review Pass 1 Fixes

- Item 1: Corrected the inverted `use_composite_reduce_scatter` logic in `q1_trace_compatibility.md`. The original text listed non-divisibility of the scatter dimension by `num_devices` as a condition that returns `true` (forcing composite). `composite_common.cpp` lines 46–48 show the opposite: non-divisibility returns `false`, suppressing the composite path and falling back to the native path. The corrected text removes non-divisibility from the `true`-returning bullet list and adds an explicit `false`-returning note explaining that non-divisibility means composite is not taken.
- Item 2: Corrected the op attributed to `TTNNLinearIColShardedWRowSharded` in `q2_semaphore_state.md`. The original text introduced the contrast section by naming that class as a representative user of `ttnn.experimental.all_reduce_async`. `linear.py` lines 158–172 show that `TTNNLinearIColShardedWRowSharded.forward` calls `ttnn.experimental.reduce_scatter_minimal_async` directly — there is no `all_reduce_async` call in that class. The corrected text accurately describes the class as a concrete example of the async semaphore pattern (using `reduce_scatter_minimal_async` with explicit semaphore handles), without misattributing the `all_reduce_async` op to it.
- Item 3: Corrected the composite path intermediate buffer count in `q1_trace_compatibility.md` (Conclusion section) and in Limitation 1 of `q3_requirements_and_limitations.md`. The original text stated three transient buffers (including the interleaved conversion tensor from `sharded_to_interleaved`). `all_reduce_async.cpp` lines 279–284 show that the conversion tensor is only allocated when `change_mem_config` is `true`, i.e., when the input is sharded. Because `TTNNLinearIColShardedWAllReduced` always passes `memory_config=ttnn.DRAM_MEMORY_CONFIG`, the input is always DRAM interleaved, `change_mem_config` is always `false`, and only two transient buffers (`reshaped_tensor` and `gather_tensor`) are allocated on the composite path for this caller. The corrected text states two buffers for this caller and adds a note that the third appears only for sharded inputs.
- Item 4: Corrected Requirement 5 item 1 in `q3_requirements_and_limitations.md`. The original text claimed that passing `memory_config=ttnn.DRAM_MEMORY_CONFIG` "prevents the `composite_for_2d_mesh` path via sharded input". The `composite_for_2d_mesh` guard checks only `GetFabricConfig() == FABRIC_2D && is_true_2d_mesh(input_tensor)` — fabric config and mesh shape. Memory config has no effect on this guard. The corrected text separates the two independent effects: (a) `DRAM_MEMORY_CONFIG` skips `sharded_to_interleaved` by keeping `change_mem_config = false`, and (b) `FABRIC_1D` is what prevents `composite_for_2d_mesh`, as stated in Requirement 3. The corrected wording makes explicit that item 1 and item 2 guard different code paths and that an implementer must satisfy item 2 independently to block the fabric-based composite path.
- Item 5: Not applied — index.md is excluded from navigation footer requirement per guide rules.

# Change Log — B Review Pass 2 Fixes

- Item 1: Corrected the composite path sub-operation description throughout `index.md` (Q1 table row) and `q1_trace_compatibility.md` (Conclusion section and code comment), and propagated the correction to Limitation 1 and Requirement 3 in `q3_requirements_and_limitations.md`. The original text named the composite path "all-gather + `moreh_sum`". Verified against `all_reduce_async.cpp` lines 295–323 and `composite_common.cpp` lines 330–390: the composite branch calls `composite_common::composite_all_gather` (which uses `ttnn::prim::all_broadcast` + `ttnn::concat`, not a primitive `ttnn::all_gather`) followed by `local_sum` (using `ttnn::moreh_sum` for non-float32 dtypes) or `local_sum_float32` (using `ttnn::transpose` + `ttnn::sum` for float32). All affected text now reads "composite_all_gather (all_broadcast + concat) + local reduce (moreh_sum for non-float32, ttnn::sum via transpose for float32)".
- Item 2: Restructured the `use_composite_reduce_scatter` description in `q1_trace_compatibility.md` to reflect the actual sequential decision flow. The original text presented non-divisibility as a third parallel category alongside the two true-return conditions. Verified against `composite_common.cpp` lines 46–59: the function first performs an early-return `false` if the scatter dimension is not evenly divisible by `num_devices`, then checks row-major layout (return `true`), then checks tile alignment (return based on that). The corrected text presents these as an explicit three-step sequence, clarifying that the divisibility check is a short-circuit guard that exits before the force conditions are evaluated, and that a divisible + tile-aligned + TILE_LAYOUT tensor is guaranteed non-composite.
- Item 3: Replaced the Requirement 2a code snippet in `q3_requirements_and_limitations.md`. The original snippet called `ttnn.experimental.all_reduce_async(tt_output, scattered_buffer, ...)` — the overload at `all_reduce_async.cpp` line 426 that routes to `ttnn::prim::all_reduce_async`, a single-step fused primitive that does not use the reduce-scatter + all-gather two-step path. Verified against `reduce_scatter_nanobind.cpp` lines 65–92: `ttnn.reduce_scatter` exposes an `output_tensor` keyword argument that maps to `optional_output_tensor` in the C++ implementation. The replacement snippet shows the correct two-step pattern: pre-allocating a persistent `scattered_tensor` at the expected shape, then calling `ttnn.reduce_scatter(..., output_tensor=scattered_tensor)` followed by `ttnn.all_gather`, matching the actual execution path taken by `ttnn.all_reduce` on T3K. A note explains why the fused `all_reduce_async` overload must not be used as a drop-in replacement.

---

# Compression Analysis: Chapter 3 — Trace Compatibility Verdict and Requirements — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~676 lines
- Estimated post-compression line count: ~590 lines
- Estimated reduction: ~13%

## CRUCIAL Suggestions

### [q2_semaphore_state.md] ~lines 166–175 (`## Summary` section)
**Issue:** The `## Summary` section at the bottom of the file restates the same three facts already fully established in the `## Conclusion` block (lines 13–17): no persistent global semaphore state, lifecycle is local to each program dispatch, no semaphore initialisation required between replays. The only sentence that adds anything not in the Conclusion is the final one pointing at `scattered_tensor` as the remaining risk — and that point is the subject of Q3 and is covered there in depth. The section is ~9 lines of pure restatement.
**Suggestion:** Delete the entire `## Summary` section (lines 166–175). Move the one novel sentence — "The remaining risk is entirely in the buffer address stability of the dynamically allocated `scattered_tensor` intermediate" — into the Conclusion block as a closing sentence if needed. Net saving: ~8 lines.

### [q1_trace_compatibility.md] ~lines 151–179 (`## The sharded_to_interleaved conversion inside all_reduce_async` section)
**Issue:** This section repeats material already covered in the Conclusion (lines 39–44). The Conclusion already states: the conversion is only allocated when `change_mem_config` is `true`; because `TTNNLinearIColShardedWAllReduced` passes `DRAM_MEMORY_CONFIG`, `input_is_sharded` is `false`, the call is skipped, and the only intermediate that matters is `scattered_tensor`. The dedicated section restates this same chain with the same C++ snippet and the same conclusion, adds a forward-looking note about a future sharded config, and contributes ~29 lines to reach a conclusion already stated 100+ lines earlier.
**Suggestion:** Delete the `## The sharded_to_interleaved conversion inside all_reduce_async` section entirely. The C++ snippet (lines 157–163) and the forward-looking note (lines 176–179) are the only pieces not in the Conclusion; the note can be folded as a single sentence into the Conclusion's `sharded_to_interleaved` bullet. Net saving: ~25 lines.

## MINOR Suggestions

### [index.md] ~lines 22–33 (`## Reading order` bullet descriptions)
**Issue:** Each of the three reading-order bullets (lines 23–26, 27–29, 30–33) describes what its target file covers. This information is already communicated more concisely and completely by the `## Research questions and one-line answers` table directly above. The bullet descriptions are padded summaries that add no navigational value beyond the table.
**Suggestion:** Reduce each bullet to a single-line label, e.g. "1. **`q1_trace_compatibility.md`** — Path selection, T3K routing, and `sharded_to_interleaved` analysis." Drop the sub-sentence expansions on lines 24–26, 28–29, and 31–33. Net saving: ~6 lines.

### [q3_requirements_and_limitations.md] ~lines 45–53 (Requirement 2 warning, options (a)–(c))
**Issue:** The warning box under Requirement 2 lists three options for satisfying intermediate buffer stability. Option (b) ("calling `ttnn.reduce_scatter` and `ttnn.all_gather` directly") is immediately implemented in full detail in Requirement 2a with a code snippet. Option (a) ("switching to `ttnn.experimental.all_reduce_async`") is contraindicated by the note at the end of Requirement 2a (lines 94–101). Option (c) ("verifying empirically that the DRAM allocator returns the same address") is stated as unreliable and not recommended. The three-option enumeration structure is therefore misleading: one option is the correct answer, one is explicitly wrong, and one is labelled unreliable. Enumerating them as parallel choices inflates the warning and obscures the recommendation.
**Suggestion:** Collapse the warning to two sentences: (1) The `ttnn.all_reduce` public API does not expose `optional_output_tensor` for the internal `scattered_tensor`; satisfying this requirement requires calling `ttnn.reduce_scatter` and `ttnn.all_gather` directly (see Requirement 2a). (2) Do not rely on the DRAM allocator returning the same address for identically-shaped tensors. Drop options (a) and (c) as labelled alternatives. Net saving: ~5 lines.

### [q3_requirements_and_limitations.md] ~lines 127–141 (Requirement 5 condition 1 parenthetical)
**Issue:** Requirement 5 condition 1 carries a long parenthetical (lines 130–133) explaining that `DRAM_MEMORY_CONFIG` affects `change_mem_config` but not `composite_for_2d_mesh`, and cross-referencing item 2 and Requirement 3. This correction was introduced in B Review Pass 2 to fix a factual error, but the resulting parenthetical is now longer than the condition statement itself. The same clarification ("fabric and mesh shape are the only inputs; memory config has no bearing on this guard") is implicit from reading Requirement 3 immediately above in Q3.
**Suggestion:** Shorten the parenthetical to: "(ensures `change_mem_config = false`, skipping `sharded_to_interleaved`; this does not affect the `composite_for_2d_mesh` guard — see Requirement 3)." Drop the clause that re-explains what `composite_for_2d_mesh` checks, since that is Requirement 3's job. Net saving: ~2 lines.

### [q2_semaphore_state.md] ~lines 76–99 (second `std::nullopt` guard block)
**Issue:** After explaining the reduce-scatter nullopt guard (lines 52–73), the file presents a near-identical explanation for the all-gather nullopt guard (lines 81–99): same structure (code block showing the `if (has_value && has_value) / else synchronous-op` pattern), same reasoning (nullopt forces the else branch), same conclusion (synchronous op is called). The prose connector ("Again, because `ag_global_semaphores` is `std::nullopt`, the synchronous `ttnn::all_gather` is called.") explicitly signals that this is a repetition of the previous block.
**Suggestion:** After the reduce-scatter code block, add one sentence: "The all-gather step applies the same guard: because `ag_global_semaphores` is `std::nullopt`, the synchronous `ttnn::all_gather` is called (lines 361–393)." Then show only the all-gather code block without the full re-explanation. This preserves the evidence while cutting the duplicated prose. Net saving: ~5 lines.

## Load-Bearing Evidence
- `q1_trace_compatibility.md` line ~87: "An implementer building a path-prediction routine must model this as a sequential guard, not as three independent conditions in parallel." — load-bearing because this is the actionable design rule for the `use_composite_reduce_scatter` three-step decision; cutting it removes the only explicit warning against mis-modelling the guards as OR-conditions.
- `q2_semaphore_state.md` line ~150 (comparison table): the six-row table comparing synchronous vs async semaphore properties — load-bearing because it is the only place in the chapter that presents the `GlobalSemaphore` address-stability requirement and the buffer-cycling pattern in a directly scannable form; prose alone before and after the table does not substitute for its structure.
- `q3_requirements_and_limitations.md` line ~94: the note block beginning "Note: `ttnn.experimental.all_reduce_async(input, buffer_tensor, ...)` maps to a different overload..." — load-bearing because it explicitly warns against using the fused `all_reduce_async` overload as a drop-in replacement for Requirement 2a, which is the most likely incorrect substitution an integrator would attempt.
- `q3_requirements_and_limitations.md` line ~144: the `assert self.out_features % (8 * 32) == 0` code snippet — load-bearing because it is the concrete, copy-pasteable action item for Requirement 5; removing it would leave the requirement without an enforcement mechanism.
- `q1_trace_compatibility.md` line ~221: "The key structural lesson from both tests is that every tensor whose device address must be stable — input, output, and all intermediates — must exist on device before the first call to `ttnn.begin_trace_capture`." — load-bearing because this sentence synthesises the evidence from both test files into the single rule that governs all buffer pre-allocation requirements; it cannot be cut without losing the explicit connection between the evidence and the requirements.

## VERDICT
- Crucial updates: yes

# Agent A Change Log — C Pass 1 Compression

- Item 1: `q2_semaphore_state.md` — Moved the novel sentence about `scattered_tensor` buffer address stability from the `## Summary` section into the `## Conclusion` block as a closing sentence, then removed the entire `## Summary` section (~8 lines saved).
- Item 2: `q1_trace_compatibility.md` — Folded the forward-looking note about future sharded config triggering an additional `sharded_to_interleaved` buffer (the only novel content in the `## The sharded_to_interleaved conversion inside all_reduce_async` section) into the Conclusion's `sharded_to_interleaved` bullet as a closing sentence, then removed the full section (~25 lines saved).

---

# Compression Analysis: Chapter 3 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~633 lines
- Estimated post-compression line count: ~607 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
(re-check of Pass 1 CRUCIAL items only)

### [q2_semaphore_state.md] `## Summary` section — RESOLVED
The section is no longer present. Agent A removed it and moved the novel `scattered_tensor` sentence into the `## Conclusion` block. No action required.

### [q1_trace_compatibility.md] `## The sharded_to_interleaved conversion inside all_reduce_async` section — RESOLVED
The section is no longer present. Agent A folded the forward-looking note into the Conclusion's bullet and removed the section. No action required.

## MINOR Suggestions
### [index.md] ~lines 22–33 (`## Reading order` bullet descriptions)
**Issue:** The three reading-order bullets each carry a sub-sentence expansion that duplicates information already expressed more completely in the `## Research questions and one-line answers` table directly above. The table already names each file, states the one-line answer, and links to the file; the bullet descriptions repeat the same subject matter at greater length without adding navigational value.
**Suggestion:** Reduce each bullet to a single-line label only, dropping the sub-sentence expansions. Net saving: ~6 lines.

### [q3_requirements_and_limitations.md] ~lines 45–53 (Requirement 2 warning, options (a)–(c))
**Issue:** The warning box enumerates three options as if they are parallel alternatives. Option (b) is the correct one, immediately implemented in Requirement 2a. Option (a) is explicitly contraindicated by the note at end of Requirement 2a. Option (c) is explicitly labelled unreliable. The three-option structure is misleading and obscures the actual recommendation.
**Suggestion:** Collapse the warning to two sentences directing readers to Requirement 2a and warning against DRAM allocator reliance. Drop the labelled (a)/(b)/(c) enumeration. Net saving: ~5 lines.

### [q3_requirements_and_limitations.md] ~lines 127–141 (Requirement 5 condition 1 parenthetical)
**Issue:** Condition 1's parenthetical re-explains the `composite_for_2d_mesh` guard in detail, a point that is Requirement 3's job. The parenthetical is now longer than the condition statement itself.
**Suggestion:** Shorten to: "(ensures `change_mem_config = false`, skipping `sharded_to_interleaved`; this does not affect the `composite_for_2d_mesh` guard — see Requirement 3)." Net saving: ~2 lines.

### [q2_semaphore_state.md] ~lines 76–99 (second `std::nullopt` guard block)
**Issue:** The all-gather nullopt guard block (lines 81–99) mirrors the structure and reasoning of the reduce-scatter nullopt guard block immediately preceding it. The prose connector explicitly signals the repetition ("Again, because `ag_global_semaphores` is `std::nullopt`, the synchronous `ttnn::all_gather` is called.").
**Suggestion:** After the reduce-scatter code block, add one sentence referencing lines 361–393 for the parallel all-gather guard, then show only the code block without the re-explanation. Net saving: ~5 lines.

## Load-Bearing Evidence
- `q1_trace_compatibility.md` line ~87: "An implementer building a path-prediction routine must model this as a sequential guard, not as three independent conditions in parallel." — load-bearing because this is the only explicit warning against mis-modelling the `use_composite_reduce_scatter` guards as OR-conditions; removing it leaves integrators without the design rule.
- `q2_semaphore_state.md` line ~150 (comparison table): the six-row table comparing synchronous vs async semaphore properties — load-bearing because it is the only location that presents the `GlobalSemaphore` address-stability and buffer-cycling requirements in a directly scannable form not substitutable by surrounding prose.
- `q3_requirements_and_limitations.md` line ~94: the note beginning "`ttnn.experimental.all_reduce_async(input, buffer_tensor, ...)` maps to a different overload..." — load-bearing because it explicitly warns against using the fused `all_reduce_async` overload as a drop-in for Requirement 2a, which is the most likely incorrect substitution an integrator would attempt.
- `q3_requirements_and_limitations.md` line ~144: the `assert self.out_features % (8 * 32) == 0` code snippet — load-bearing because it is the concrete, copy-pasteable enforcement mechanism for Requirement 5; removing it leaves the requirement without an actionable implementation form.
- `q1_trace_compatibility.md` line ~193: "The key structural lesson from both tests is that every tensor whose device address must be stable — input, output, and all intermediates — must exist on device before the first call to `ttnn.begin_trace_capture`." — load-bearing because this sentence synthesises the evidence from both test files into the single rule governing all buffer pre-allocation requirements.

## VERDICT
- Crucial updates: no
