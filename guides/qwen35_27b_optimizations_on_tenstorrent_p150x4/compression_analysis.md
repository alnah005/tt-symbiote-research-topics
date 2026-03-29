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
