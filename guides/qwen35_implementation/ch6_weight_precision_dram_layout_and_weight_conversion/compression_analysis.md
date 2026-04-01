# Compression Analysis: Chapter 6 — Weight Precision, DRAM Layout, and Weight Conversion — Pass 1

## Summary
- Total files analyzed: 4 (index.md, dtype_choices.md, hf_to_meta_conversion.md, moe_key_protection.md)
- Estimated current line count: ~310 lines
- Estimated post-compression line count: ~290 lines
- Estimated reduction: ~6%

## CRUCIAL Suggestions

### [hf_to_meta_conversion.md, moe_key_protection.md] Pop-protect-reinsert code snippet
**Issue:** The `_is_moe_key` function and pop-protect pattern code block appears in both hf_to_meta_conversion.md (Step 0) and moe_key_protection.md (Protection Pattern section).
**Suggestion:** hf_to_meta_conversion.md should reference moe_key_protection.md for the full pattern; Step 0 description can remain but the code block should be omitted or reduced to a signature.

## MINOR Suggestions

### [dtype_choices.md] DRAM table units
**Issue:** Table mixes "GiB" and "~15.0 GiB" entries; the "~" prefix is inconsistently applied to some estimated values but not others.
**Suggestion:** Apply "~" consistently to all estimated values in the table.

### [moe_key_protection.md] Section header length
**Issue:** "Why Dense-Only Models Are Unaffected" section is three sentences and could be folded into the preceding section.
**Suggestion:** Minor merge to reduce section count.

## Load-Bearing Evidence
- `dtype_choices.md` line ~43: "`fp32_dest_acc_en=False, math_approx_mode=False`" kernel config — load-bearing; exact parameter names verified against source and affect device correctness.
- `hf_to_meta_conversion.md` lines ~40–60: 5-step conversion pipeline with key names — load-bearing reference for anyone adding new weight types or debugging conversion failures.
- `moe_key_protection.md` lines ~9–54: two failure mode examples (split_hf_keys, map_hf_to_meta_keys) — load-bearing; removing either would leave the "why" incomplete for engineers debugging expert weight corruption.

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Chapter 6 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~300 lines (after Pass 1 consolidation of code duplication)
- Estimated post-compression line count: ~290 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions
(none — Pass 1 CRUCIAL item resolved)

## MINOR Suggestions

### [dtype_choices.md] Recurrent state size calculation
**Issue:** "2 MB per layer" claim for recurrent state appears without the derivation `1 × 32 × 128 × 128 × 4 bytes ≈ 2 MB`; some readers may want to verify.
**Suggestion:** Add the derivation inline (one line) or in a code block.

## Load-Bearing Evidence
- `moe_key_protection.md` lines ~64–71: `_is_moe_key` predicate and pop-before-transform code — load-bearing; the exact matching strings (`mlp.experts`, `mlp.gate.`, `mlp.shared_expert`) must be preserved for correctness.
- `hf_to_meta_conversion.md` lines ~120–140: q_proj interleaved per-head reshape logic — load-bearing; the reshape + slice pattern is the only place this non-obvious weight layout is explained.
- `dtype_choices.md` line ~100: DRAM summary table — load-bearing reference used in ch7 DRAM budget discussion.

## VERDICT
- Crucial updates: no
