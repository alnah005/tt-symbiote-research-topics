# Compression Analysis: Cross-Chapter Redundancy Check — Pass 1

## Summary
- Total files analyzed: 28 (across 7 chapters + guide index)
- Estimated current line count: ~2800 lines
- Estimated post-compression line count: ~2780 lines
- Estimated reduction: ~1%

## CRUCIAL Suggestions

None. The guide has no load-bearing cross-chapter redundancy problems. Repeated facts serve a legitimate pedagogical purpose (re-anchoring the reader who enters at a specific chapter) and do not contradict each other.

## MINOR Suggestions

### 1. [ch3/recurrence_math.md] ~lines 158-171
**Issue:** Ch3 derives "12 MB" per layer and "576 MB" total for state size. Ch7's `performance_summary.md` derives the same number with round-trip bandwidth calculation (1.2 GB). A reader going through Ch3 might wonder about the connection.
**Suggestion:** Add a brief forward-reference note: "For the bandwidth impact of this state size during decode, see Chapter 7."

## Load-Bearing Evidence

### 1. Profiler breakdown (Ch6 index, Ch7 bottleneck)
The exact profiler timing block appears in `ch6_l1_state_management/index.md` and `ch7_performance_analysis/bottleneck_analysis.md`. Ch6 uses it to motivate L1 state work; Ch7 uses it as the anchor for the full bottleneck analysis. Intentional context-setting for readers who enter at either chapter. Not actionable bloat.

### 2. State size arithmetic (12 MB per layer) across four locations
`ch3/recurrence_math.md` (derives 12 MB, 576 MB static), `ch7/performance_summary.md` (derives 12 MB, 1.2 GB round-trip), `ch6/index.md` (states 12 MB, 1.2 GB per step), `ch1/hybrid_architecture.md` (states 576 MB across 48 layers). All numbers are mutually consistent. Ch3 gives static footprint; Ch7 gives round-trip bandwidth.

### 3. TTFT performance table (Ch5 index, Ch7 summary)
The `498 ms -> 94 ms, 5.3x` table appears in both. Ch5 presents it as the chapter's result; Ch7 presents it as part of the full performance dashboard. Numbers are identical.

### 4. DeltaNet recurrence equation (guide index, Ch3, Ch7)
The top-level index provides a one-line summary with a link to Ch3's full derivation. Ch7 uses the equation to explain why prefill is sequential. Both are appropriate uses.

### 5. L1 state validation status (Ch6, Ch7)
Ch7 has the structured table; Ch6 has the same facts in prose. Consistent.

### 6. No contradictions found
All repeated numbers, config names, tensor shapes, and optimization descriptions verified consistent: B=32, Nv_TP=12, Nk_TP=4, Dk=128, Dv=128, num_pairs=384, COMPUTE_HIFI4 with fp32_dest_acc_en=True, conv1d 384 dispatches, 2.26x per-layer cost ratio.

## VERDICT
- Crucial updates: no

---

## Pass 2

**Summary:** 0 crucial updates, 1 minor suggestion
**Crucial updates: no**

### CRUCIAL (must fix before guide is done)

None. The HiFi2/HiFi4 fix in `recurrence_math.md` and the clarifications to `bottleneck_analysis.md` introduced no new cross-chapter redundancy. All precision claims are now consistent across the five locations that reference them (Ch1 `tp_sharding_strategy.md`, Ch3 `recurrence_math.md`, Ch4 `kernel_dispatch.md`, Ch4 `compute_kernel.md`, Ch7 `bottleneck_analysis.md`).

### MINOR Suggestions (optional)

**1. [ch3/recurrence_math.md, Numerical Precision section ~line 186] Forward reference to Ch7 bandwidth analysis**

The Pass 1 MINOR suggestion — adding a forward reference from the Ch3 state-size arithmetic to Ch7 — was not applied and remains outstanding. The Numerical Precision section now also notes that `fp32_dest_acc_en=True` "is critical for the state update where small updates to a large state matrix could otherwise be lost to bfloat16 rounding." Ch7 `bottleneck_analysis.md` echoes the same rationale at line 86 ("The recurrence uses HiFi4 because the iterative state update accumulates numerical error across tokens"). These are additive and not redundant, but a one-sentence cross-reference at the end of the Ch3 Numerical Precision section — "For the decode-time bandwidth cost driven by this 12 MB state, see Chapter 7" — would close the loop for readers who read Ch3 linearly.

### Load-Bearing Evidence

**For MINOR suggestion 1:**

- `ch3/recurrence_math.md` line 150: "This state is read and written every decode step for every GDN layer, making DRAM bandwidth the primary bottleneck. Chapter 6 discusses the L1 state optimization that addresses this." Ch3 already forward-references Ch6 for the optimization path but does not forward-reference Ch7 for the quantitative bandwidth breakdown (1.15 GB/step, 9.78 ms/layer timing). The gap is real but small.
- `ch7/performance_summary.md` lines 40-46: derives the 1.15 GB/step round-trip figure from the same 12 MB per-layer number first established in Ch3. A reader entering at Ch7 can follow back to Ch3 via the Quick Reference table in `index.md`; a reader reading Ch3 in order has no forward pointer to the Ch7 quantitative treatment.

### VERDICT

**Crucial updates: no**

The guide is done. No compression actions are required before publication. The one outstanding MINOR suggestion (forward reference from Ch3 Numerical Precision to Ch7 bandwidth analysis) is carry-forward from Pass 1 and remains optional.
