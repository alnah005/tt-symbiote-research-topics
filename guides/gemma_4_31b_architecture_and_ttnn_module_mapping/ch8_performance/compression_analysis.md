# Chapter 8 Change Log

Changes made in response to Agent B review (`b_review.md`), addressing four
identified issues.

## Issue 1: Weight Memory Inconsistency (index.md, memory_budget.md)

**Problem:** `memory_budget.md` computed BFP8 weight total as ~4,156 MB (from
rounded per-layer values), while `index.md` and Ch6 use the authoritative
~4,143 MB figure derived from exact per-projection bytes.

**Fix:** Changed `memory_budget.md` total from 4,156 MB to 4,143 MB. Added a
rounding note explaining the 13 MB discrepancy between rounded per-layer
multiplication and Ch6's bottom-up calculation. Updated all DRAM budget tables
(BFP8+BF16 and BFP8+BFP8 configurations) and the quantization description
to use 4,143 MB. No change needed in `index.md` (already correct at 4,143 MB).

**Files changed:** `memory_budget.md` (lines 53-62, 69, 170-178, 183-186).

## Issue 2: BFP8+BFP8 131K Headroom (index.md, memory_budget.md)

**Problem:** Both `index.md` and `memory_budget.md` stated ~2.7 GB headroom for
BFP8+BFP8 at S=131K. Actual arithmetic: 12,288 - 9,315 = 2,973 MB = ~2.9 GB.

**Fix:** Changed headroom from ~2.7 GB to ~2.9 GB in both `index.md` (line 36)
and `memory_budget.md` (Maximum Context Length Summary table).

**Files changed:** `index.md` (line 36), `memory_budget.md` (line 194).

## Issue 3: BF16+BF16 Headroom at 8K (memory_budget.md)

**Problem:** `memory_budget.md` claimed ~4.4 GB headroom for BF16+BF16 at ~8K.
Actual: weights 7,899 + KV 740 + activations 2 = 8,641 MB; headroom =
12,288 - 8,641 = 3,647 MB = ~3.6 GB.

**Fix:** Changed headroom from ~4.4 GB to ~3.6 GB in the Maximum Context Length
Summary table.

**Files changed:** `memory_budget.md` (line 192).

## Issue 4: Fused Gate+Up Latency (decode_latency_analysis.md, optimization_roadmap.md)

**Problem:** The fused gate+up projection was listed as ~48 us (the cost of a
single gate or up projection), but fusing concatenates both weight matrices into
a ~29 MB read (14.5 + 14.5 MB). At 300 GB/s this takes ~96 us, not ~48 us.
The 48 us figure understated per-layer cost by 48 us, cascading into all totals.

**Fix:** Changed fused gate+up from ~48 us to ~96 us in both per-layer summary
tables (sliding and global). Updated all cascading values:

- Sliding layer total: 178 us -> 226 us
- Global layer totals: 284/444/1,084 us -> 332/492/1,132 us
- Total decode latency table (all sequence lengths):
  - S=2K: 11.7 ms -> 14.6 ms (~68 tok/s)
  - S=4K: 12.3 ms -> 15.2 ms (~66 tok/s)
  - S=8K: 13.3 ms -> 16.2 ms (~62 tok/s)
  - S=16K: 15.5 ms -> 18.4 ms (~54 tok/s)
  - S=32K: 19.7 ms -> 22.6 ms (~44 tok/s)
- Model comparison table: ~13 ms -> ~16 ms at S=8K
- Key observations: updated percentages and crossover point (~16K -> ~32K)
- Fused gate+up description: clarified ~29 MB combined weight read

Also updated `optimization_roadmap.md` to reflect the new baseline:
- Metal Trace projection: ~13 ms -> ~16 ms baseline, ~7-9 ms -> ~9-11 ms after
- Combined impact estimate: recalculated full optimization waterfall from
  16.2 ms baseline to ~5.2 ms projected optimized (~190 tok/s)
- S=32K optimized: 19.7 ms -> 22.6 ms baseline, ~7-9 ms -> ~9-11 ms projected
- Trace buffer headroom: ~7.4 GB -> ~7.2 GB (from Issue 1 weight correction)
- Metal Trace summary row: 4-6 ms -> 5-7 ms savings

**Files changed:** `decode_latency_analysis.md` (lines 43, 162-167, 180-185,
193-199, 203-213, 225), `optimization_roadmap.md` (lines 23-24, 39, 297,
308-327).

---

# Compression Analysis (Agent C)

## File 1: `index.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The "Key Performance Metrics Summary" table (lines 27-37) provides a single-glance reference consolidating the most important numbers from across the chapter. This is a legitimate index-level summary that serves navigational purpose distinct from the detailed tables in subfiles.

**Minor suggestion:** The "Primary Bottlenecks" section (lines 40-60) restates findings that are fully developed in `decode_latency_analysis.md` (CCL overhead at lines 137-150, SDPA growth at lines 107-126, memory-bound matmuls at lines 15-23) and `optimization_roadmap.md` (heterogeneous layer tracing at lines 28-36). This section could be condensed from four detailed paragraphs to a brief enumeration with forward references, saving roughly 15 lines. The detail belongs in the subfiles, not the index.

---

## File 2: `memory_budget.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The DRAM budget tables at lines 176-193 are the authoritative source for per-configuration headroom calculations. Every other file references these numbers rather than deriving them independently. This is correctly placed, non-redundant content.

**Minor suggestion:** The "Quantization Requirement" prose at lines 68-80 narrates conclusions that a reader can already draw from the immediately preceding tables. Specifically, "At BF16, the model weights alone consume ~7.9 GB of the 12 GB per-device budget, leaving only ~4.1 GB" (line 70-71) is a restatement of the Total weights row in the table at line 56. The paragraph could be reduced to two sentences: the BFP8 requirement statement (line 74) and the BFP4 evaluation note (lines 78-80). The arithmetic connecting the table to the conclusion does not need to be spelled out.

---

## File 3: `decode_latency_analysis.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The per-layer latency summary tables (sliding at lines 156-169, global at lines 173-187) are unique content that synthesizes component latencies into actionable per-layer totals. No other file provides this breakdown.

**Minor suggestion:** The per-projection latency tables (sliding lines 29-38, global lines 49-57) include a "Per-Device Weight (BFP8)" column that exactly duplicates the BFP8 column from `memory_budget.md` (sliding table lines 17-27, global table lines 31-41). Since the latency derivation formula is `weight_bytes / bandwidth`, including the weight sizes is reasonable for self-containment, but the tables could reference memory_budget.md for the weight derivation rather than re-presenting the full per-projection breakdown. A more compact presentation would list only the total weight read and total latency per layer type, with a reference link for per-projection detail.

Additionally, the sliding window explanation at lines 96-99 ("The sliding window bounds the KV cache to 1,024 tokens regardless of the total sequence length") repeats what `memory_budget.md` already states at lines 88-89. A brief cross-reference would suffice.

---

## File 4: `optimization_roadmap.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The prioritized optimization waterfall at lines 308-320 is unique analytical content that shows cumulative impact of applying techniques in sequence. This is the only place where the interaction between optimizations is quantified.

**Minor suggestion:** Several optimization sections re-derive baseline figures that are already established in earlier files. For example:
- Priority 4 (lines 148-150) restates "120 all-reduce operations per decode step contribute an estimated 0.6-1.2 ms" -- this exact figure appears in `index.md` line 51-54 and `decode_latency_analysis.md` lines 148-150.
- Priority 5 (lines 180-183) restates KV cache sizes at S=8,192 and S=131,072 that are already tabled in `memory_budget.md` lines 122-128 and 188-192.
- Priority 1 (lines 23-24) restates the ~16 ms baseline from `decode_latency_analysis.md` line 199.

These repeated baseline numbers could be replaced with short cross-references (e.g., "the 0.6-1.2 ms CCL overhead identified in the latency analysis"), reducing each optimization section by 2-3 lines. Across 8 sections this would trim roughly 20 lines of restatement.

---

## Cross-File Redundancy Summary

| Repeated Content | Appears In | Recommendation |
|-----------------|-----------|----------------|
| Per-device BFP8 weight sizes per projection | memory_budget.md, decode_latency_analysis.md | Keep in memory_budget.md; reference from decode_latency_analysis.md |
| CCL overhead (120 ops, 0.6-1.2 ms) | index.md, decode_latency_analysis.md, optimization_roadmap.md | Define in decode_latency_analysis.md; reference elsewhere |
| Sliding window = 1,024 tokens, constant KV | memory_budget.md, decode_latency_analysis.md | Define in memory_budget.md; reference from decode_latency_analysis.md |
| BFP8 quantization requirement rationale | memory_budget.md (table + prose) | Remove prose restatement of table data |
| Baseline decode latency (~16 ms at S=8K) | decode_latency_analysis.md, optimization_roadmap.md, index.md | Define in decode_latency_analysis.md; reference elsewhere |

**Estimated total reduction:** ~50-60 lines across all four files (~8-10% of combined content), with no loss of information.
