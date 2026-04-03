# Chapter 5 Change Log

## Date: 2026-04-03

## Summary

Three issues identified in the Agent B (Critic) review (`b_review.md`) have been resolved. All cascading numeric dependencies across the three chapter files and the index have been updated.

---

## Issue 1: MLP FLOP Count Omits Gate Projection (High Severity)

**Root cause:** Gemma 4 vision MLP is gated with three weight matrices (gate_proj, up_proj, down_proj), but only two projections were counted.

**Files changed:** `cpu_baseline_profiling.md`, `ttnn_latency_projection.md`, `decision_matrix.md`

**Key numeric changes:**

| Quantity | Before | After |
|----------|--------|-------|
| MLP FLOPs per layer | ~16.7 GFLOPs | ~25.1 GFLOPs |
| Total FLOPs per layer | ~28.9 GFLOPs | ~37.3 GFLOPs |
| 27 encoder layers total | ~780 GFLOPs | ~1,007 GFLOPs |
| Total vision encoder (280 tokens) | ~785 GFLOPs | ~1,010 GFLOPs |
| MLP share of total FLOPs | 57.4% | 67.1% |
| Attention share of total FLOPs | 41.7% | 32.6% |

**Cascading updates (280-token budget, batch=1):**

| Quantity | Before | After |
|----------|--------|-------|
| CPU latency (optimistic) | 13.1 ms | 16.8 ms |
| CPU latency (conservative) | 26.2 ms | 33.7 ms |
| TTNN matmul latency (optimistic) | 6.0 ms | 7.8 ms |
| TTNN matmul latency (conservative) | 8.7 ms | 11.2 ms |
| TTNN total with tracing (optimistic) | 7.0 ms | 9.5 ms |
| TTNN total with tracing (conservative) | 11.0 ms | 14.5 ms |
| Break-even speedup (batch=1, 280) | 2.2x | 2.1x |
| Break-even speedup (batch=8, 280) | 4.5x | 4.8x |
| Break-even speedup (batch=8, 1120) | 5.3x | 5.7x |

All token-budget and batch-scaling tables in both CPU and TTNN files were recomputed. The Gemma 3 SigLIP comparison FLOP figure was updated from ~785 to ~1,010 GFLOPs, and the FLOP ratio from ~5x to ~3.8x. Decision matrix scenario latencies and speedup ratios were updated throughout.

---

## Issue 2: Parameter Count Inconsistency (Low Severity)

**Root cause:** Chapter 5 stated "approximately 550M parameters" while Chapter 1 computed ~569M.

**Files changed:** `index.md`, `cpu_baseline_profiling.md`, `ttnn_latency_projection.md`

**Changes:**
- `index.md` line 29: "approximately 550M" changed to "approximately 570M"
- `cpu_baseline_profiling.md` line 64: "550M-parameter" changed to "570M-parameter"
- `ttnn_latency_projection.md` weight transfer warning: "550M params" changed to "570M params"

The ~1.1 GB weight size estimate remains unchanged (570M x 2 bytes = 1.14 GB, which rounds to ~1.1 GB).

---

## Issue 3: Weight Size Calculation Missing Gate Projection (Medium Severity)

**Root cause:** Per-layer weight traffic omitted the MLP gate_proj weights (1152 x 4304 x 2 = 9.92 MB).

**Files changed:** `ttnn_latency_projection.md`

**Key numeric changes:**

| Quantity | Before | After |
|----------|--------|-------|
| Weight traffic per layer | ~30.5 MB | ~40.4 MB |
| Weight traffic for 27 layers | ~823 MB | ~1,091 MB |
| Bandwidth time (288 GB/s) | 2.9 ms | 3.8 ms |

The workload remains compute-bound at batch=1 since compute time (~7.8-11.2 ms) still exceeds bandwidth time (3.8 ms). Qualitative conclusions are unchanged.

---

## Qualitative Impact

All three issues shifted absolute numbers upward but did not change the qualitative recommendations. TTNN still wins at 140+ tokens (batch=1) and at all token budgets for batch >= 4. The break-even point remains at roughly 70-140 tokens at batch=1, where CPU is competitive or slightly faster. The speedup ratios shifted slightly due to the increased MLP compute, but the overall ordering and recommendations in the decision matrix are preserved.

---

## Agent C — Compression Analysis

### Date: 2026-04-03

### Scope

Redundancy and bloat review across all four Chapter 5 files: `index.md`, `cpu_baseline_profiling.md`, `ttnn_latency_projection.md`, `decision_matrix.md`.

---

### File 1: `index.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "Decision Framework Summary" table (lines 47-54) reproduces the decision matrix table from `decision_matrix.md` (lines 7-15) almost cell-for-cell. Both present the same six scenario rows with the same recommendations. The index table adds no information that the decision_matrix file lacks. This is a full cross-file duplication of the chapter's central artifact.

**MINOR suggestion:** The "Reading Order" section (lines 58-60) restates the file sequence already conveyed by the "Chapter Contents" table directly above it. Three sentences explaining which file to read first, second, and third add no value when the table already lists them in that order with descriptive topic labels. Consider removing the "Reading Order" section or collapsing it into a single sentence within the "Overview" section.

---

### File 2: `cpu_baseline_profiling.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "Key Takeaway" section (lines 188-189) restates three facts already established in the body: (a) encoder layers are ~99% of compute (stated in the Warning on line 103), (b) CPU latency is 17-34 ms at 280 tokens batch=1 (stated in the table on line 124), and (c) latency grows rapidly at higher budgets/batches (stated in the Warning on line 143). Every clause in the takeaway is a verbatim or near-verbatim repetition of content the reader encountered within the same file, at most 50 lines earlier.

**MINOR suggestion:** The "Hardware Assumptions" subsection (lines 53-60) includes a tip about AMX BF16 on Sapphire Rapids that partially overlaps the "Methodology for Estimates" subsection (lines 109-114), which also discusses AMX utilization percentages. The AMX utilization context could be consolidated into a single location -- either the hardware table or the estimation methodology -- rather than split across both.

---

### File 3: `ttnn_latency_projection.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "FLOP Counts Recap" section (lines 22-31) is an explicit duplicate of the summary table from `cpu_baseline_profiling.md` (lines 94-101). It copies the same four rows (patch embedding 1.5, encoder layers 1,007, pooling+projection 3.5, total 1,010) with identical values. The section heading itself acknowledges this is a recap, not new content. A single-line cross-reference ("FLOP counts are in cpu_baseline_profiling.md, Table X") would suffice.

Additionally, the "Summary" table (lines 222-227) re-presents three rows that are a subset of the break-even comparison table (lines 171-180) in the same file. The reader encounters these exact numbers (batch=1 280 tokens: 25.3 ms vs 12.1 ms, 2.1x; batch=1 1120 tokens: 127.2 ms vs 33.8 ms, 3.8x; batch=8 280 tokens: 180.4 ms vs 37.8 ms, 4.8x) twice within 50 lines.

**MINOR suggestion:** The Gemma 3 SigLIP reference comparison table (lines 110-119) lists eight architectural parameters, five of which (hidden_size, num_layers, num_heads, head_dim, intermediate_size) are identical between Gemma 3 and Gemma 4. Rather than a full row-by-row comparison that visually implies differences, a sentence stating the layer dimensions are identical followed by only the differing rows (input, position encoding, GFLOPs) would be more compact and less misleading.

---

### File 4: `decision_matrix.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "How Gemma 3 TTNN Performance Informs Expectations" section (lines 118-128) substantially overlaps the "Reference: Gemma 3 SigLIP TTNN Performance" section in `ttnn_latency_projection.md` (lines 106-128). Both sections make the same four points: (1) MLP matmuls are well-optimized at 1152x4304, (2) attention works at long sequences, (3) non-matmul overhead is significant and tracing helps, (4) weight sharding transfers directly. The decision_matrix version is nearly a bullet-point summary of the ttnn_latency_projection version. One of these should be removed in favor of a cross-reference.

Additionally, the "Summary of Recommendations" section (lines 148-161) restates the decision matrix table from the top of the same file (lines 7-15). Five numbered justification bullets repeat rationale already given in the per-scenario detailed analysis immediately above. Within a single file, the reader encounters the same recommendation three times: in the matrix table, in each scenario's bold "Recommendation" line, and again in the summary.

**MINOR suggestion:** The "Porting Effort vs. Latency Benefit" section (lines 130-146) introduces engineering effort estimates (5-7 weeks, 40-50% reuse) by forward-referencing Chapters 6 and 7. This content is not latency analysis -- it is project-planning justification. Consider moving it to Chapter 7 (implementation roadmap) where effort-vs-benefit analysis is topically native, and replacing it here with a brief cross-reference.

---

### Cross-File Redundancy Summary

| Redundancy | Source | Duplicate | Type |
|------------|--------|-----------|------|
| Decision framework table | `decision_matrix.md` lines 7-15 | `index.md` lines 47-54 | Full table duplication |
| FLOP summary table | `cpu_baseline_profiling.md` lines 94-101 | `ttnn_latency_projection.md` lines 26-31 | Explicit recap of identical data |
| Gemma 3 TTNN lessons | `ttnn_latency_projection.md` lines 106-128 | `decision_matrix.md` lines 118-128 | Paraphrased restatement |
| Break-even numbers | `ttnn_latency_projection.md` lines 171-180 | `ttnn_latency_projection.md` lines 222-227 | Intra-file re-presentation |
| Scenario recommendations | `decision_matrix.md` lines 7-15 | `decision_matrix.md` lines 148-161 | Intra-file restatement |
| AMX utilization context | `cpu_baseline_profiling.md` lines 53-60 | `cpu_baseline_profiling.md` lines 109-114 | Intra-file split coverage |

### Estimated Compressibility

If the identified redundancies were consolidated (cross-references replacing duplicated tables, intra-file summaries removed, Gemma 3 comparison deduplicated, porting-effort section relocated), the combined chapter word count could be reduced by an estimated 15-20% without losing any unique information.
