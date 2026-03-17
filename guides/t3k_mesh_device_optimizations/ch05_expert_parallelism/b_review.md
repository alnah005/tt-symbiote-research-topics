# B Review — T3K Mesh Chapter 5: Expert Parallelism (Re-review after C1+C2 compression)

## Verdict: APPROVED

Both compression changes (C1 and C2) are correct. The pointer text in `index.md` is accurate, all authoritative content is present and intact in the target files, and no correctness issues were introduced.

---

## Compression Change Verification

### C1 — Removal of "Summary of Expert Placement Strategies" from index.md

**Replacement text (index.md lines 129-130):**
> "For the four placement strategies (naive uniform, load-balanced, locality-aware, expert replication) with full derivations, formulas, and worked examples, see `expert_placement_strategies.md`."

**Pointer accuracy:** Correct. The description matches the actual content of `expert_placement_strategies.md`, which covers all four named strategies across Sections 1-4.

**Content still present in expert_placement_strategies.md — verified:**

| Claim | Location | Verified value |
|---|---|---|
| r_e formula: `max(1, ceil(f_e × E / k)) = max(1, ceil(32 × f_e))` | lines 190-191 | Matches ground truth ✓ |
| Zipf example: f_e=0.17 → `max(1, ceil(0.17×256/8)) = max(1, ceil(5.44)) = 6` | lines 202-208 | r_e=6, matches ground truth ✓ |
| f_e/r_e = 0.17/6 ≈ 0.028 ≈ f_avg | line 208 | Correct (f_avg = 1/32 ≈ 0.031) ✓ |
| T_d = B tokens/device under uniform routing | lines 54-58 | Matches ground truth ✓ |
| Dispatch volume formula: `V = (N-1) × C × E_d × H × 2 bytes` | lines 64-66 | Correct form ✓ |
| B=32, C=2 worked example: `7×2×32×7168×2 = 6,422,528 bytes ≈ 6.4 MB` | line 66 | Matches ground truth exactly ✓ |

Nothing was lost. All strategy derivations, the r_e formula, the Zipf example, and T_d=B are present and correct in `expert_placement_strategies.md`.

---

### C2 — Removal of "Key Quantitative Summary / All-to-All Volume Scaling" from index.md

**Replacement text (index.md line 131):**
> "For dispatch volume calculations, capacity formula, and per-B byte totals, see `token_routing_and_dispatch.md` Section 4."

**Pointer accuracy:** Correct. Section 4 of `token_routing_and_dispatch.md` is titled "Latency Breakdown" and contains the dispatch volume formula, B=1 and B=32 worked examples, and a summary table.

**Content still present in token_routing_and_dispatch.md Section 4 — verified:**

| Claim | Location | Verified value |
|---|---|---|
| Dispatch volume formula: `V = (N-1) × C × E_d × H × 2 bytes` | line 180 | Correct form ✓ |
| B=1 (C=1): `7×1×32×7168×2 = 3,211,264 bytes ≈ 3.2 MB` | lines 182-186 | Arithmetic correct ✓ |
| B=32 (C=2): `7×2×32×7168×2 = 6,422,528 bytes ≈ 6.4 MB` | lines 188-192 | Matches ground truth exactly ✓ |
| Summary table with B=1 (~3.2 MB) and B=32 (~6.4 MB) | lines 196-201 | Present and correct ✓ |

Nothing was lost. The authoritative volume calculations are intact in the target file.

---

## Remaining Content in index.md — Correctness Check

After the removals, the only quantitative statement remaining in the new "Expert Placement and Communication Summary" section (lines 122-131) is in the paragraph above it at line 123 (MoE data flow section):

> "approximately 3.2–6.4 MB per device at B=1–32 decode, and up to 939.5 MB per device at B=32, S=2048 prefill"

- 3.2–6.4 MB range: consistent with the pointer targets (confirmed above). ✓
- 939.5 MB: this is the same ch03 cross-reference value flagged as a non-blocking note in the prior review. It is an attributed value sourced from `ch03_all_to_all_num_links/all_to_all_in_moe.md` and is not derived in this chapter. Status unchanged — not a chapter 5 error.

All other quantitative claims in index.md (model/hardware constants table, C formula, prerequisite chapter descriptions) are unaffected by the removals and were verified correct in the prior review.

---

## Prior Review Findings — Status After C1+C2

| Finding | Prior status | Status now |
|---|---|---|
| Fix 1: dispatch volume formula in expert_placement_strategies.md | Verified correct | Still correct; content preserved through C1 ✓ |
| Fix 2: prefill C calculation in token_routing_and_dispatch.md | Verified correct | Still correct; content preserved through C2 ✓ |
| Cross-reference note: 939.5 MB table from ch03 | Non-blocking, outside ch05 scope | Unchanged; still non-blocking ✓ |
| expert_placement_strategies.md line 214 replication feasibility | Accurate as written | Unchanged ✓ |
| expert_placement_strategies.md line 264 locality hop estimate | Reasonable illustrative estimate | Unchanged ✓ |
| combine_and_accumulation.md BF16 epsilon ≈ 0.004 | Acceptable approximation | Unchanged ✓ |

---

## Summary

C1 and C2 are both correct compressions. The pointer text in `index.md` accurately describes the target files and sections. All authoritative content removed from `index.md` — including the r_e formula, the Zipf f_e=0.17 → r_e=6 example, T_d=B, and the dispatch volume formula with B=1/B=32 worked examples — is fully preserved in `expert_placement_strategies.md` and `token_routing_and_dispatch.md` Section 4 respectively. No correctness issues were introduced.

No issues found — chapter approved.

---

# B Review — Pass 2 (after C3 compression: combine_and_accumulation.md lines 51-53)

## Verdict: APPROVED

### C3 — Removal of re-derived dispatch volume formula from combine_and_accumulation.md

**Change verified:** Lines 51-53 of `combine_and_accumulation.md` now read:

```
V_combine = V_dispatch

The formula and per-batch worked examples (6.4 MB at B=32, 3.2 MB at B=1) are derived in `token_routing_and_dispatch.md` Section 4.
```

The re-derived formula `V_combine = (N-1) × C × E_d × H × 2` and the numeric evaluations at B=32 and B=1 are gone.

**Symmetry assertion still present:** `V_combine = V_dispatch` at line 51. Confirmed.

**Pointer accuracy:** `token_routing_and_dispatch.md` Section 4 is titled "Latency Breakdown" (line 163 of that file). It contains:
- The dispatch volume formula `V = (N-1) × C × E_d × H × 2 bytes` at line 180.
- B=1 (C=1) worked example: 3,211,264 bytes ≈ 3.2 MB at lines 182-186.
- B=32 (C=2) worked example: 6,422,528 bytes ≈ 6.4 MB at lines 188-192.
- Summary table at lines 196-201.

The pointer is accurate. Confirmed.

**No correctness issues introduced:** The symmetry assertion is correct (combine A2A transfers expert outputs of shape [C × E_d, H], identical to dispatch send buffer shape, so volumes are equal). The cited numeric values in the pointer text (6.4 MB at B=32, 3.2 MB at B=1) match the authoritative values in `token_routing_and_dispatch.md` Section 4 exactly.

B: approved
