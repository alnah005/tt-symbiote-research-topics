# B Review — Chapter 6: Performance Benchmarking and Config Selection — Pass 1

## Issues Found

**Issue 1 — benchmarking_methodology.md, line 190: PCC target threshold is wrong**

The file states:

> "The target threshold for bfloat16 MoE outputs is typically 0.9995 (99.95% correlation with a float32 reference)."

The authoritative fact is that the PCC target is typically > 0.999, not 0.9995. This matters because 0.9995 is a stricter threshold, and the document then uses this number in the reporting table and in the prose guidance ("the canonical production configs target the row that achieves PCC >= 0.9995 at the lowest latency"). The correct figure is > 0.999. The stricter 0.9995 threshold appears only in the "Any Projection — Strict PCC Requirement" section of config_decision_matrix.md, where it is correctly labeled as the strict/elevated tier.

---

**Issue 2 — production_config_checklist.md, line 35: arithmetic error in accumulation buffer size example**

The warning box states:

> "For `per_core_N=8` and `out_subblock_h=4`, that is 8 × 4 × 1024 × 4 = 131 KB per core"

The arithmetic is wrong. 8 × 4 = 32; 32 × 1024 = 32,768 elements; 32,768 × 4 bytes = 131,072 bytes = **128 KB**, not 131 KB. 131,072 bytes is exactly 128 KiB (131,072 / 1024 = 128). The stated "131 KB" appears to confuse the byte count (131,072) with kilobytes. The correct statement is 128 KB.

---

## Agent A Change Log — B Feedback Pass 1
- benchmarking_methodology.md: Changed standard PCC threshold from 0.9995 to 0.999 (authoritative figure); kept 0.9995 only for the strict-quality tier
- production_config_checklist.md: Fixed arithmetic "8 × 4 × 1024 × 4 = 131 KB" → "= 131,072 bytes = 128 KB"

---

# B Review — Chapter 6: Performance Benchmarking and Config Selection — Pass 2

## Pass 1 Fix Verification

Both Pass 1 fixes are confirmed present and correct:

- `benchmarking_methodology.md` line 190: Now reads "> 0.999 (99.9% correlation with a float32 reference)." Correct.
- `production_config_checklist.md` line 35: Now reads "= 131,072 bytes = 128 KB per core". Correct.

## Issues Found

**Issue 1 — production_config_checklist.md, line 17: standard PCC gate still uses wrong threshold (0.9995)**

The blocking checklist item reads:

> "PCC >= 0.9995 for all projections, or a lower threshold is explicitly documented and approved."

This inverts the tier structure established by the authoritative facts. The standard production PCC threshold is > 0.999. The 0.9995 figure is the *strict-quality gate*, not the default. Presenting 0.9995 as the default pass/fail criterion means that any deployment using the correct canonical configs (LOFI for gate/up, HIFI2 for down) would appear to fail the checklist even when PCC meets the standard threshold. The line should read: "PCC >= 0.999 for all projections under standard regime, or PCC >= 0.9995 for deployments under the strict-quality gate, or an explicitly documented and approved alternative threshold."

---

**Issue 2 — production_config_checklist.md, lines 142–145 (Checklist Summary Card): PCC threshold wrong in all four summary card rows**

The copy-and-annotate summary card shows:

```
[ ] Gate projection PCC (token-level):   ______   >= 0.9995?
[ ] Up projection PCC (token-level):     ______   >= 0.9995?
[ ] Down projection PCC (token-level):   ______   >= 0.9995?
[ ] Layer-level PCC (full expert FFN):   ______   >= 0.9995?
```

All four rows use `>= 0.9995` as the pass criterion. This is the same error as Issue 1 — the strict-quality threshold is being substituted for the standard production threshold. The summary card is what practitioners fill out at deployment time; getting this wrong will cause valid configs to be rejected or will train practitioners to treat 0.9995 as routine. All four lines should read `>= 0.999?`, with a note that the strict-quality gate uses `>= 0.9995`.

---

**Issue 3 — index.md, lines 41–44 (Decision Flowchart): logical gap between 0.999 and 0.9995 leaves a PCC band unrouted**

The flowchart branches the "What PCC threshold?" decision into exactly two paths:

```
    <= 0.999       > 0.9995
        |             |
      HIFI2        HIFI4 + fp32_dest_acc_en=True
```

The band `0.999 < PCC <= 0.9995` has no branch. A reader whose requirement falls in that band — above the standard threshold but below the strict tier — receives no guidance. The authoritative facts establish that HIFI2 is the correct config for the standard threshold (> 0.999) and HIFI4 with fp32_dest_acc_en=True is required only above 0.9995. The two branches should therefore be `<= 0.9995` → HIFI2 and `> 0.9995` → HIFI4, which are exhaustive and match the tier definitions in config_decision_matrix.md.

## Agent A Change Log — B Feedback Pass 2
- production_config_checklist.md line 17: Changed blocking PCC gate from >= 0.9995 to >= 0.999 (standard threshold)
- production_config_checklist.md lines 142-145: Changed summary card PCC rows from >= 0.9995 to >= 0.999
- index.md lines 41-44: Fixed flowchart PCC branch split from (<= 0.999 / > 0.9995) to (<= 0.9995 → HIFI2 / > 0.9995 → HIFI4)

---

# B Review — Chapter 6: Performance Benchmarking and Config Selection — Pass 3

## Pass 2 Fix Verification

- `production_config_checklist.md` line 17: Now reads "PCC >= 0.999 for all projections, or a lower threshold is explicitly documented and approved." Correct — standard threshold, not the strict 0.9995.
- `production_config_checklist.md` lines 142–145: All four summary card rows now read `>= 0.999?`. Correct.
- `index.md` lines 41–44: Flowchart now branches on `<= 0.9995` → HIFI2 and `> 0.9995` → HIFI4. Correct — exhaustive, no gap.

All three Pass 2 fixes are confirmed present and correctly applied.

## Verdict

**Issue 1 — benchmarking_methodology.md, line 177: packer_l1_acc savings formula does not match the authoritative formula**

The file states:

> "Redundant read reduction = 1 − (b / K_t)"
> where K_t = K/32 and b = `in0_block_w`

The authoritative packer_l1_acc savings formula is `(K_t−1)/K_t`. The file introduces `b` as a free parameter (`in0_block_w`), yielding a generalized expression `1 − (b / K_t)` that equals the authoritative formula only when `b = 1`. For `b > 1` (i.e., `in0_block_w > 1`), the file's formula produces a lower savings figure than the authoritative one. The formula should be `(K_t−1)/K_t`, not `1 − (b / K_t)`.

## Agent A Change Log — B Feedback Pass 3
- benchmarking_methodology.md: Fixed packer_l1_acc savings formula from `1 − (b / K_t)` to `(K_t−1)/K_t`

---

# B Review — Chapter 6: Performance Benchmarking and Config Selection — Pass 4

## Pass 3 Fix Verification

The Pass 3 fix is confirmed present and correctly applied.

`benchmarking_methodology.md` lines 174–183: The formula block now reads:

```
Redundant read reduction = (K_t−1)/K_t
```

The surrounding prose correctly defines `K_t = K/32` without referencing `b` or `in0_block_w`. The two example calculations are correct: K=2048 (K_t=64) → (64−1)/64 = 98.4%; K=7168 (K_t=224) → (224−1)/224 = 99.6%. Both match the authoritative facts exactly.

## Verdict

No feedback — chapter approved.
