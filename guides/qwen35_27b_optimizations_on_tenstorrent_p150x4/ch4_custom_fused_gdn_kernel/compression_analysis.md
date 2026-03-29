# Compression Analysis: Custom Fused GDN Kernel — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~620 lines
- Estimated post-compression line count: ~480 lines
- Estimated reduction: ~23%

## CRUCIAL Suggestions

None. The redundancy found is cross-file duplication of reference tables and structural repetition within the compute kernel narrative. None undermines clarity or correctness; removing it would save roughly 120-150 lines without losing information.

## MINOR Suggestions

### 1. [index.md] ~lines 131-161
**Issue:** Full 28-row circular buffer table reproduced from `kernel_dispatch.md` lines 131-161. The index file's job is orientation, not specification.
**Suggestion:** Replace with a brief summary ("The program defines 28 circular buffers totaling 218 KB per core; see kernel_dispatch.md for the full table").

### 2. [writer_kernel.md] ~lines 7-26
**Issue:** Writer compile-time args table (7 rows) and runtime args table (4 rows) reproduce identical content from `kernel_dispatch.md` lines 106-116 and 76-84.
**Suggestion:** Replace with cross-reference to kernel_dispatch.md.

### 3. [compute_kernel.md] ~lines 19-51
**Issue:** Re-lists 22 CB indices with names, index numbers, tile counts, and roles. 13 entries are copied verbatim from the CB table in `kernel_dispatch.md`.
**Suggestion:** Keep only the 9 compute-intermediate-only entries and cross-reference kernel_dispatch.md for the rest.

### 4. [reader_kernel.md] ~lines 9-17
**Issue:** Paraphrases reader compile-time and runtime argument tables already in `kernel_dispatch.md` lines 88-105 and 59-84 with same names, values, and descriptions.
**Suggestion:** Replace with a one-line cross-reference.

### 5. [compute_kernel.md] ~lines 179-196
**Issue:** Phase 2 text states it is "structurally identical to Phase 1" then shows a partial code block demonstrating the one difference (no scale multiply).
**Suggestion:** Compress to a single paragraph noting the difference, with no code block (~15 lines saved).

## Load-Bearing Evidence
- `kernel_dispatch.md` line ~88: The compile-time argument tables (reader, writer, compute) are the canonical definition of the kernel interface — load-bearing as the authoritative reference
- `reader_kernel.md` line ~48: "issue_row_reads" function with batched NOC reads and single barrier — load-bearing as the core optimization mechanism, cannot be cut
- `compute_kernel.md` line ~93: The DeltaNet recurrence implementation with FP32 dest accumulation — load-bearing as the correctness-critical compute phase
- `writer_kernel.md` line ~104: HEIGHT_SHARDED L1 memcpy path with `volatile tt_l1_ptr` — load-bearing as the defining artifact of the L1 state approach

## VERDICT
- Crucial updates: no
