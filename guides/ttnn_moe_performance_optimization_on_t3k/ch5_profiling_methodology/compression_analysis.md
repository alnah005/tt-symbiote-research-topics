## Change Log — B Feedback Pass 1

**index.md:** Added conditional qualifier — 3-pass BF16 centering only runs in the n_group <= topk_group branch; the group-based path skips it.

**router_latency_profiling.md:** Corrected/removed wrong BF16 subnormal value citation.

## Change Log — B Feedback Pass 2

**tracy_profiling_setup.md:** Added `pre_norm` Tracy zone annotation covering `moe.py:L1477` (`ttnn.mul(routed_out, 1.0/float(n_rs))`) between the `experts_forward` zone and the `reduce_scatter` zone in the Step 5 zone-annotation code.

**ttnn_op_timer_profiling.md:** Added `pre_norm` (`eltwise_mul` scalar, L1477) as a row in the `TTNNMoE.forward` stage mapping table between the experts row and `reduce_scatter`, and added a corresponding row (expected median 2–8 µs) in the Step 6 expected-output table.

**router_latency_profiling.md:** Changed per-pass overhead from "10–30 µs" to "10–15 µs" to be consistent with the concrete example (single-pass baseline ~15 µs, 3-pass total ~35–45 µs implying ~20–30 µs additional overhead across 2 extra passes, i.e. 10–15 µs each).

# Compression Analysis: Ch5 Profiling Methodology — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~985 lines
- Estimated post-compression line count: ~940 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions

### [tracy_profiling_setup.md] ~lines 136–158
**Issue:** The Step 4 harness includes two stub Tracy zones (`"residual_capture"` and `"all_gather_async"`, lines 137–146) whose bodies are empty `pass` statements. The inline comments explicitly say these zones do nothing ("No direct call here; this is a tensor alias." and "zone wraps the model call for coarse annotation."). The note at line 158 then immediately tells the reader that all sub-forward-pass zones are ineffective from outside `moe.py`. The stubs thus mislead before the note corrects — a reader who follows the harness code as written will add empty zones that produce no timeline data, then must read the disclaimer to understand why.
**Suggestion:** Remove the two empty stub zones and their surrounding `with` blocks entirely. Keep only the `"forward_dispatch"` zone wrapping the actual `model(x)` call and the synchronize. Move the disclaimer note to before the harness code (one sentence: "Because `TTNNMoE.forward` is not externally decomposable at the Python level, the harness annotates only the full forward call; for per-stage zones, apply annotations directly inside `moe.py` as in Step 5."). This eliminates ~20 lines of dead code and removes the confusion.

---

## MINOR Suggestions

### [index.md] ~lines 59–62
**Issue:** The "Reading Order" paragraph restates the sequential dependency between the three files, but this information is already present in the per-tool descriptions (lines 21–45) and in the "Research Questions Covered" table (lines 51–56), which lists the files in order with their Q-labels. The paragraph adds only the detail that `router_latency_profiling.md` "adds the precision dimension" — which the table already implies by listing Q7 separately.
**Suggestion:** Trim the paragraph to one sentence covering only the non-obvious dependency (that `router_latency_profiling.md` requires both prior tools). The first two clauses are redundant with the table.

### [ttnn_op_timer_profiling.md] ~lines 3–7
**Issue:** The "Context" section repeats both source file ranges (`moe.py:L1159–L1343` and `moe.py:L1412–L1496`) that appear verbatim in the section headers and table entries later in the same file (lines 171, 186).
**Suggestion:** Remove the source range parentheticals from the Context section; the Q8 label and question text are sufficient context. The ranges are redundant given their appearance in the body.

### [router_latency_profiling.md] ~lines 49–59
**Issue:** The prose paragraph at lines 49–51 ("The harness below runs `TTNNMoERouterDecode.forward` in isolation... making the router's contribution measurable.") is restated almost verbatim in the module docstring at lines 54–59 inside the code block.
**Suggestion:** Remove the module docstring from the code block (lines 54–59) since its three bullet points duplicate what the preceding prose already states. The docstring adds no information not in the paragraph immediately above it.

### [router_latency_profiling.md] ~lines 289–308
**Issue:** The "Device-Side Cycle Counter Measurement" section (lines 289–308) re-explains using `TT_METAL_DEVICE_PROFILER=1` and reading the CSV output, which was already fully documented in `ttnn_op_timer_profiling.md`. The cross-reference at line 291 acknowledges this, yet the section then provides its own `extract_router_ops` helper and re-explains the workflow rather than deferring entirely to the referenced file.
**Suggestion:** Keep the `extract_router_ops` function (it is specific to the router-window filtering use case), but remove the two sentences before it that re-explain how the TTNN op timer CSV approach works (lines 291–292 up to "Filter the CSV"). Replace with a single sentence pointing to `ttnn_op_timer_profiling.md` Step 3 for CSV setup, then present `extract_router_ops` directly as the router-specific addition.

---

## Load-Bearing Evidence
- `tracy_profiling_setup.md` line ~158: `"The zone annotations above are coarse wrappers around the full \`model(x)\` call because \`TTNNMoE.forward\` is not easily split at the Python level without modifying \`moe.py\`."` — load-bearing because it is the disclaimer that confirms the stub zones in the same code block are non-functional, establishing that those stubs are bloat rather than intentional scaffolding.

## VERDICT
- Crucial updates: yes

---

## Change Log — B Feedback Pass 3

**router_latency_profiling.md:** Changed 3-pass overhead multiplier from "2×–3×" to "~2.3×–3×" to be consistent with the concrete example (single-pass baseline ~15 µs, 3-pass total ~35–45 µs → lower bound ratio is 35/15 ≈ 2.3×, not 2×).

## Change Log — C Compression Pass 1

**tracy_profiling_setup.md:** Deleted the two empty stub Tracy zones (`"residual_capture"` and `"all_gather_async"`) from the Step 4 harness; both contained only `pass` and produced no timeline data. Moved the scope-limitation disclaimer (previously a trailing note after the code block) to a blockquote immediately before the code block, so the reader understands the harness constraint before reading the code. Kept the working `"forward_dispatch"` outer zone wrapping the actual `model(x)` call. Removed approximately 20 lines of dead code.

---

# Compression Analysis: Ch5 Profiling Methodology — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~977 lines (index.md: 62, tracy_profiling_setup.md: 293, ttnn_op_timer_profiling.md: 302, router_latency_profiling.md: 320)
- Estimated post-compression line count: ~977 lines
- Estimated reduction: ~0% (no CRUCIAL fixes applied; MINOR items documented only)

## CRUCIAL Suggestions

None remaining. Pass 1 addressed the only CRUCIAL item: the two empty stub Tracy zones in `tracy_profiling_setup.md` Step 4 that contained only `pass` statements and produced no timeline data. No new prose-duplicates-table/diagram or 3x-repetition issues were found in the current state of the files.

## MINOR Suggestions

The following four MINOR items from Pass 1 were correctly documented but not applied (per rules). They remain present in the files and are re-documented here for completeness, with one new item added.

### [index.md] lines 59–62 — Reading Order paragraph (carry-forward from Pass 1)
**Issue:** The "Reading Order" paragraph re-states the sequential dependency between the three files. This information is already implicit in the per-tool descriptions (lines 21–45) and the "Research Questions Covered" table (lines 51–56). The only non-redundant clause is the note that `router_latency_profiling.md` adds a precision dimension that neither timing tool covers alone.
**Suggestion:** Reduce to one sentence covering only that non-obvious dependency. Estimated saving: 2–3 lines.

### [ttnn_op_timer_profiling.md] lines 3–7 — Context section source ranges (carry-forward from Pass 1)
**Issue:** The Context section lists both source file ranges (`moe.py:L1159–L1343` and `moe.py:L1412–L1496`) in parentheses. These same ranges appear verbatim as section headers in Step 3 (lines 171, 186) and throughout the mapping tables below. The Q8 question text is sufficient context for the opening.
**Suggestion:** Remove the parenthetical source range annotations from the Context section. Estimated saving: 1 line.

### [router_latency_profiling.md] lines 54–59 — Module docstring duplicates preceding prose (carry-forward from Pass 1)
**Issue:** The prose at lines 49–51 states that the harness runs `TTNNMoERouterDecode.forward` in isolation and eliminates expert compute from the measurement. The module docstring inside the code block (lines 54–59) restates this as three bullet points with no additional information.
**Suggestion:** Remove the module docstring (lines 54–59). The file-level docstring string adds nothing not stated in the paragraph immediately above. Estimated saving: 7 lines.

### [router_latency_profiling.md] lines 291–292 — Device-side section re-explains CSV setup already in ttnn_op_timer_profiling.md (carry-forward from Pass 1)
**Issue:** Lines 291–292 re-explain using `TT_METAL_DEVICE_PROFILER=1` and the CSV output format, then cross-reference `ttnn_op_timer_profiling.md` where this is already documented in full. The `extract_router_ops` function that follows is specific to this file and should be kept.
**Suggestion:** Replace lines 291–292 (up to "Filter the CSV for rows") with a single sentence deferring to `ttnn_op_timer_profiling.md` Step 3 for CSV setup. Estimated saving: 1–2 lines.

### [router_latency_profiling.md] lines 263–272 — "Expected Results" table note partially restates table content (new)
**Issue:** The paragraph following the Expected Results table (lines 270–272) states: "The gap between the two agreement rates narrows at larger batch sizes because the statistical likelihood of a near-tie between two experts is independent of batch size — but the per-token routing error rate remains the same." The second clause ("but the per-token routing error rate remains the same") contradicts the first clause without adding new information — if the near-tie likelihood is independent of batch size, the per-token error rate being the same is a direct corollary, not an additional fact. The sentence is self-redundant.
**Suggestion:** Delete the trailing clause ("— but the per-token routing error rate remains the same"), leaving the sentence to end after "independent of batch size". Estimated saving: 1 line fragment.

## Load-Bearing Evidence

1. **router_latency_profiling.md, line 13** — "When two expert scores are separated by less than one BF16 step, a single-pass `ttnn.topk` in BF16 may compare two values that round to the same BF16 representation, and will then select based on index rather than value." This is the precise failure mode being guarded against; removing or paraphrasing it would obscure the engineering rationale for the 3-pass trick.

2. **router_latency_profiling.md, lines 40–42** — The explanation of why `topk(k+1)` is used in passes 1 and 2 (to extract the score just below the decision boundary as the centering target) is non-obvious and does not appear in any other file or in the code comments of the harness. Cutting this would leave the `k+1` argument unexplained.

3. **ttnn_op_timer_profiling.md, lines 256–262 (Step 5)** — The five aggregation rules, including the rule to use **median not mean** for individual ops and **min for CCL ops**, are not restated anywhere else in the chapter. These rules are the principal method guidance for producing reliable sweep comparisons.

4. **ttnn_op_timer_profiling.md, line 285** — "CCL ops (`all_gather`, both `all_to_all`, `reduce_scatter`) collectively account for approximately 55–65% of the total forward-pass device time at batch=1 decode." This is the key quantitative finding that motivates the chapter structure and cross-references to Chapters 2 and 3. It must not be cut.

5. **tracy_profiling_setup.md, lines 269–272** — The three "Signs of a Pipeline Stall" bullet points specify the exact timeline signatures (e.g., `shared_experts_and_add` starting only after `reduce_scatter` completes) that indicate a missing overlap. These diagnostic patterns are not derivable from the setup steps alone and are the principal actionable output of the Tracy section.

6. **router_latency_profiling.md, lines 313–315** — The go/no-go threshold for removing the centering trick ("agreement rate ≥ 0.9999 on the actual model's logit distribution") and the caveat that the N(0, 0.15) synthetic distribution is an approximation requiring validation against real logits. This is a safety constraint on a potential optimization; trimming it could lead to premature removal of the centering trick.

## VERDICT
- Crucial updates: no

---

## Change Log — C Compression Pass 2

No changes applied. All CRUCIAL items from Pass 1 were resolved by C Compression Pass 1. No new CRUCIAL issues (prose duplicating a table/diagram with no added information, or content repeated 3+ times) were identified in the current state of the four files. Four MINOR items from Pass 1 remain documented but unapplied per the compression rules; one new MINOR item was identified in `router_latency_profiling.md` lines 270–272.
