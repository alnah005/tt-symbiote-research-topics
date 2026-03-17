# Compression Analysis — Chapter 4: MoE Forward Pass Op Breakdown

## Crucial updates: no

No crucial cross-file duplications found.

---

## Methodology

Every file in Chapter 4 was read in full and compared against all files in Chapters 1, 2, and 3. The search focused on multi-line blocks — code examples, tables, step-by-step procedures, formula derivations — that appear verbatim or near-verbatim in both a Chapter 4 file and a prior-chapter file, where one copy could be removed with a cross-reference.

---

## Candidates Examined and Ruled Out

### 1. MoE Op Dispatch Order Sequence

`ch3_ttnn_profiling_api/device_profiler_api.md` (lines 85–89) contains a 2-line inline code block listing the MoE op dispatch order:

```
topk → softmax (routing weights) → all_gather → matmul (gate proj) →
silu → matmul (up proj) → matmul (down proj) → reduce_scatter/all_reduce
```

`ch4_moe_op_breakdown/index.md` (lines 53–75) contains a 22-line ASCII diagram of the same forward pass with step numbers, T3K annotations, and phase labels. These are not verbatim duplicates. The ch4 diagram is a substantially more detailed structural document that serves as the conceptual anchor for the entire chapter. Not a crucial duplication.

### 2. Tracy vs. Device Profiler Quick-Reference Table

`ch1_tracy_overview/index.md` (lines 51–57) contains a 5-row summary table (What it times / Activation / Output format / Key question / Blind spot). This table does not appear in any Chapter 4 file. Chapter 4 references the distinction by name in its prerequisites section but does not reproduce the table. No duplication.

### 3. `process_ops_logs.py` Invocation

`ch3_ttnn_profiling_api/reading_op_timing_output.md` (lines 36–54) and `ch2_tracy_setup/output_format.md` (lines 73–78) each contain a multi-step workflow for running `process_ops_logs.py`. Chapter 4's `full_op_sequence_reference.md` (line 40–42) references `process_ops_logs.py` only as a single-sentence instruction within a numbered checklist step — it does not reproduce the multi-step workflow block. Not a crucial duplication.

### 4. Key Model Configuration Table

`ch4_moe_op_breakdown/index.md` (lines 98–106) contains a 6-row configuration table (d_model, d_ff, num_experts, top_k, Hardware, Dtype). The same parameters appear inline as prose in `expert_matmul_phase.md` (line 106) and in the header paragraph of `full_op_sequence_reference.md` (lines 7–8), but not as a reproduced table. The inline prose restatements of configuration constants are not multi-line block duplications; they are contextual anchors. Not a crucial duplication.

### 5. Phase Latency Totals: Per-Phase Files vs. Phase Latency Summary Table

`dispatch_phase.md` (lines 128–129), `expert_matmul_phase.md` (lines 131–135), and `combine_phase.md` (lines 108–109) each contain a phase-total summary row at the end of their respective latency budget tables. `full_op_sequence_reference.md` (lines 110–116) consolidates these totals into a single Phase Latency Summary table.

The figures are the same across both locations, but this is intentional structural design: the phase files provide per-op breakdown with narrative context; the reference file consolidates for use as a checklist. Crucially, all of this duplication is within Chapter 4 itself (intra-chapter), not between Chapter 4 and a prior chapter. The task scope is cross-chapter duplications between Chapter 4 and Chapters 1–3. Not in scope.

### 6. "How to Use This Table as a Ground-Truth Checklist" Procedure

`full_op_sequence_reference.md` (lines 38–65) contains a 6-step numbered procedure for using the op table against a device profiler CSV. This procedure has no counterpart in Chapters 1, 2, or 3 — it is unique to Chapter 4. No duplication.

### 7. AICLK Warning

`ch3_ttnn_profiling_api/device_profiler_api.md` (lines 143–145) and `ch1_tracy_overview/tracy_vs_device_profiler.md` (line 19) both warn against hardcoding 1 GHz for the AICLK conversion. Chapter 4 does not reproduce this warning as a block — it references `DEVICE KERNEL DURATION [ns]` by column name without explaining the cycle conversion. Not a crucial duplication.

---

## Conclusion

Chapter 4 references concepts defined in Chapters 1–3 (Tracy zone names, `DEVICE KERNEL DURATION [ns]`, `process_ops_logs.py`) but does not reproduce multi-line blocks from those chapters verbatim. All latency-table overlaps identified are intra-chapter (within ch4 across its five files), and they serve the intentional role of a phase-level reference consolidating into a master reference table. No content block in Chapter 4 meets the threshold of a crucial cross-chapter duplication where one copy could be removed with a cross-reference to a prior chapter.
