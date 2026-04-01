# Chapter 4 Review -- Correctness (Agent B)

## Issue 1: CB count is 28, not 26

`index.md` (line 14) says "26 circular buffer descriptors" and `kernel_dispatch.md` (line 129) says "The program defines 26 circular buffers." However, both the chapter's own CB table and the Python source (`_build_full_fused_device_program` in `gdn_kernel_op.py`, lines 388-418) define **28** distinct CB indices: c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_12, c_13, c_14, c_15, c_16, c_17, c_18, c_19, c_20, c_21, c_24, c_25, c_26, c_27, c_28, c_29, c_31. Count those entries in the chapter's own table and you get 28.

**Fix:** Change "26" to "28" in both `index.md` and `kernel_dispatch.md`.

## Issue 2: Total tile count and L1 calculation are wrong

`kernel_dispatch.md` (line 162) states:

> `(4+4+4+4+1+1+16+16+16+1+1+1+1+4+1+4+4+4+1+1+1+1+4+4+4+4+1+1) * 2048 = 104 tiles * 2 KB = 208 KB`

The parenthesized sum contains 28 addends (consistent with 28 CBs, contradicting the "26" claim). Summing them: 4+4+4+4+1+1+16+16+16+1+1+1+1+4+1+4+4+4+1+1+1+1+4+4+4+4+1+1 = **109**, not 104. The correct L1 total is 109 tiles x 2 KB = **218 KB** (still well within the ~1.2 MB budget, so the qualitative conclusion is unaffected).

**Fix:** Change "104 tiles" to "109 tiles" and "208 KB" to "218 KB" (or alternatively, recount and correct the parenthesized expression).

No other factual errors found. All remaining claims -- runtime/compile-time argument tables, sub-tile extraction math, pair-to-head mapping, CB flow between kernels, gate computation sequence, recurrence steps, and write paths -- match the source code.

## Pass 1

1. **`compute_kernel.md` — four "persistent" CBs are held by the compute kernel but never consumed in any described phase, creating a false impression that RMS norm runs in-kernel.** `cb_norm_w` (c_14), `cb_rms_scale` (c_31), `cb_rms_eps` (c_20), and `cb_reduce_scaler` (c_19) are all waited for at compute kernel startup and popped after the pair loop, but none of the five described phases (L2 Norm Q, L2 Norm K, K Transpose, Gate Computation, DeltaNet Recurrence) consume them. RMS norm is in fact applied post-kernel in Python (`gdn.py` line 330: `ttnn.rms_norm`). A reader implementing the compute kernel would either look for a missing RMS norm phase or implement the kernel without these 4 CBs — the latter would cause a deadlock because the reader pushes to them unconditionally. The guide must state explicitly that the compute kernel holds these CBs as pre-fetched L1 reserves (for a planned kernel extension) but does not currently use them, and that RMS norm is applied by the Python caller post-kernel.

2. **`kernel_dispatch.md` — the `_compute_kernel_hash` code snippet uses an ellipsis that hides 5 of 8 hashed files, misrepresenting the scope of cache invalidation.** The source (`gdn_kernel_op.py` lines 56–57) hashes all 8 kernel paths: `READER_PATH, WRITER_PATH, READER_IAF_PATH, WRITER_IAF_PATH, COMPUTE_PATH, READER_FUSED_PATH, WRITER_FUSED_PATH, COMPUTE_FUSED_PATH`. The guide shows only `[READER_PATH, WRITER_PATH, ..., COMPUTE_FUSED_PATH]`. A developer relying on this guide to add a new kernel file to the hash set would not know which files are already covered, potentially adding a duplicate or omitting a required path. Replace the ellipsis with the full list.

3. **`index.md` "Process Files" table — description for `b_review.md` says "identified CB count and L1 total errors" but both errors have been corrected in the current `kernel_dispatch.md`.** The text "identified CB count and L1 total errors" implies those errors still exist in the chapter. A reader auditing the guide would waste time searching for errors that are no longer present, or distrust the corrected values. Change the description to reflect that the errors were found and subsequently fixed (e.g., "Correctness review (Agent B) — CB count and L1 total errors found and fixed").

4. **`kernel_dispatch.md` CB table — gap at `c_22` and `c_23` is unexplained, inconsistent with the explicit callout of `c_11` and `c_30` as removed.** The CB table jumps from `c_21` to `c_24` without a note. The guide explicitly notes that `c_11` (originally `cb_z`) and `c_30` (originally `cb_rec_out`) were removed during development. The indices `c_22` and `c_23` are simply never allocated, but a reader who notices the skip has no way to know whether those slots were removed (like c_11/c_30) or were always absent. Add a one-line note: "c_22 and c_23 are unallocated (never assigned)."

5. **`reader_kernel.md` — the "Total Read Count Per Pair" table counts 44 reads but the sharded state path issues zero NOC reads, making the "44 reads before a single barrier" claim only true for the DRAM path.** The introduction states "all 44 reads required for a single pair are issued before a single `noc_async_read_barrier()`" as if it is always true. The State Reads section then explains that the HEIGHT_SHARDED path skips the 16 NOC reads entirely (direct L1 memcpy). The table footer also says "All issued before single barrier." An implementer on the sharded path would see only 28 NOC reads (Q+K+V+scalars) before the barrier, and would not find 16 state reads — the "44 total" claim is wrong for that path. The introduction and table should qualify: "44 reads on the DRAM state path; 28 reads on the HEIGHT_SHARDED path."

## Pass 2

All 5 Pass 1 issues are confirmed fixed:

1. `compute_kernel.md` — the callout box (lines 54–58) now explicitly states that `cb_norm_w`, `cb_rms_scale`, `cb_rms_eps`, and `cb_reduce_scaler` are waited on but never consumed by any compute phase, that RMS norm runs post-kernel in Python, and that removing them would deadlock. The CB Dataflow Summary also marks them "held, NOT consumed." Issue resolved.

2. `kernel_dispatch.md` — the `_compute_kernel_hash` snippet now enumerates all 8 paths (`READER_PATH`, `WRITER_PATH`, `READER_IAF_PATH`, `WRITER_IAF_PATH`, `COMPUTE_PATH`, `READER_FUSED_PATH`, `WRITER_FUSED_PATH`, `COMPUTE_FUSED_PATH`) with no ellipsis. Issue resolved.

3. `index.md` — the `b_review.md` row now reads "Correctness review (Agent B)" with no stale error reference. Issue resolved.

4. `kernel_dispatch.md` CB table — rows for `c_22` and `c_23` are present, labelled "*(unused/reserved)*" with an explanation of the index gap. Issue resolved.

5. `reader_kernel.md` — the introduction now states the DRAM path issues 44 NOC reads and the HEIGHT_SHARDED path issues 28. The table and trailing paragraph both qualify both counts. Issue resolved.

No feedback — chapter approved.

## Pass 3

All Pass 1 and Pass 2 fixes are confirmed intact. Verified against `gdn_kernel_op.py`, `gdn_kernel_op_ttnn.py`, and `gdn.py`:

- Runtime arg tables (12 reader / 4 writer) match source lines 361–380 exactly.
- Reader compile-time arg table (11 values, indices 0–10) matches `reader_ct` at lines 447–453.
- Writer compile-time arg table (7 values) matches `writer_ct` at lines 455–459.
- `gdn_full_fused_inplace` line range (609–656) and `_build_full_fused_device_program` line range (325–498) are correct.
- `v_tile_off = 32 = 2 * key_dim_tp / 32` formula is correct (source line 385).
- `num_pairs=384 / num_cores=40` remainder arithmetic (24 cores × 10 pairs, 16 cores × 9 pairs) is correct.
- Scratch buffer partition totals (1792 bytes of 2048-byte tile) are internally consistent.
- NOC read counts (44 DRAM path, 28 sharded path) are correct and consistently qualified throughout `reader_kernel.md`.
- All navigation footers present in all five files.
- All `index.md` links are clickable relative paths.
- No plain-text display equations; all math is in `$$...$$` or code blocks.
- The `c_11` and `c_30` removal notes and `c_22`/`c_23` "unused/reserved" rows are present.
- `gdn.py` post-kernel flow (RMS norm line 330, SiLU line 334) matches the description in `compute_kernel.md`.

No feedback — chapter approved.
