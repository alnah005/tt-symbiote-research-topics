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
