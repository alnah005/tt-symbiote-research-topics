# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 1

## Issues Found

---

### Issue 1 — `when_sparse_matmul_wins.md`, Section 3, line ~102: Wrong substitution in sparsity ratio formula produces incorrect closed-form expression

**File:** `when_sparse_matmul_wins.md`, lines ~101–110

**Error:** The general sparsity ratio formula is:

$$\rho = \frac{\min(E_d,\ k \times B \times S / N)}{E_d \times M_t}$$

After substituting $k = 8$, $N = 8$, $E_d = 32$, the numerator becomes $\min(32,\ 8 \times B \times S / 8) = \min(32,\ B \times S)$. The text instead writes $\min(32,\ B \times S / 4)$, introducing a spurious factor of $1/4$ in the numerator. This is an algebra error: $k/N = 8/8 = 1$, not $1/4$.

**Consequence:** The "verification" for $B=1$, $S=1$ yields $\rho = \min(32, 0.25)/32 = 0.0078$, which contradicts the $\rho \approx 0.031$ derived correctly in Section 2.1 for the identical scenario. The text then invents a probability-based explanation to justify the discrepant 0.0078 value; this explanation is not grounded in the formula and should be removed. The two subsections are inconsistent: Section 2.1 is correct; Section 3 has the wrong substituted formula.

**Fix:** Replace `\min\!\left(32,\ \frac{B \times S}{4}\right)` with `\min\!\left(32,\ B \times S\right)`. The corrected closed-form is:

$$\rho \approx \frac{\min\!\left(32,\ B \times S\right)}{32 \times \lceil C / 32 \rceil}$$

Remove the probability-based paragraph that attempts to explain the 0.0078 figure; the correct verification is $\rho = \min(32, 1) / (32 \times 1) = 1/32 \approx 0.031$, consistent with Section 2.1.

---

### Issue 2 — `when_sparse_matmul_wins.md`, Summary table (line ~116–124): Wrong $\rho$ for short-prefill row ($B=1$, $S=128$)

**File:** `when_sparse_matmul_wins.md`, summary table, "Short prefill" row

**Error:** The table states $B=1$, $S=128$, $C=4$, $M_t=1$, $\rho \approx 0.5$, "Use: sparse_matmul". The correct value is $\rho \approx 1.0$:

- $k \times B \times S / N = 8 \times 1 \times 128 / 8 = 128$. Since $128 > E_d = 32$, all 32 local expert slots are active.
- $\rho = \min(32, 128) / (32 \times 1) = 32/32 = 1.0$.

At $\rho = 1.0$, all experts are active and `sparse_matmul` provides no benefit (overhead dominates). The "Use" column should be "batched matmul", not "sparse_matmul".

This error holds under both the correct formula ($B \times S$ substitution) and even the incorrect $B \times S / 4$ substitution from Issue 1 — $\min(32, 128/4) = 32$, still giving $\rho = 1.0$. There is no scenario under which $\rho \approx 0.5$ for these parameters.

**Fix:** Change the "Short prefill" row to $\rho \approx 1.0$ and "Use: batched matmul", consistent with the Medium prefill row below it.

---

### Issue 3 — `sparse_matmul_internals.md`, Section 3, line 112: Tile-row skip savings are understated by a factor of $K_t = 224$

**File:** `sparse_matmul_internals.md`, line 112

**Error:** The text states: "skipping tile row $m$ eliminates $N_t$ weight tile reads and $M_t \times N_t$ FMA blocks for that row… the savings are $N_t \times 1088$ bytes of DRAM reads avoided and $N_t \times 32{,}768$ FMAs avoided."

This is incorrect on two counts:

1. **Weight tile reads per skipped row:** For an inactive row $m$, every K-step ($k = 0 \ldots K_t-1$) is skipped for every output column ($n = 0 \ldots N_t-1$). Total weight tiles avoided = $K_t \times N_t$, not $N_t$. Total weight bytes avoided = $K_t \times N_t \times 1088$, not $N_t \times 1088$.

2. **FMA blocks:** The text writes $M_t \times N_t$, but the correct factor is $K_t \times N_t$ (one FMA block per K-step per N-column for row $m$). $M_t$ is the number of M-tile rows in the full activation tensor; it has no role in counting FMA blocks for a single inactive row. This should be $K_t$.

With $K_t = 224$, the actual per-row savings are 224× larger than stated for both weight reads and FMAs.

**Fix:** Replace the sentence with: "skipping tile row $m$ eliminates $K_t \times N_t$ weight tile reads and $K_t \times N_t$ FMA blocks for that row. For a single inactive tile row, the savings are $K_t \times N_t \times 1088$ bytes ($= 224 \times N_t \times 1088$) of weight DRAM reads avoided and $K_t \times N_t \times 32{,}768$ FMAs avoided."

---

### Issue 4 — `sparse_matmul_internals.md`, Section 5, line ~161: Incorrect factor used to compute combined BFP8 + sparsity bandwidth reduction

**File:** `sparse_matmul_internals.md`, lines ~160–162

**Error:** The text states: "BFP8 weight compression reduces weight tile size from 2048 to 1088 bytes — a 47% reduction in weight read bandwidth. Sparsity masking eliminates weight tile reads entirely for inactive activation tile rows. The two benefits are multiplicative: at $\rho = 0.03$ (3% of tile rows active), the total weight bytes read from DRAM are reduced by a factor of $0.47 \times 0.03 = 0.014$ relative to a dense BF16 baseline."

The error is in treating 0.47 (the *fraction saved*) as the *remaining* fraction. The correct remaining fraction after BFP8 compression is $1088/2048 \approx 0.531$ (53.1% of original bandwidth, a 46.9% ≈ 47% reduction). To compute the combined remaining bandwidth, multiply the remaining fractions: $0.531 \times 0.03 \approx 0.016$. The text incorrectly uses $0.47 \times 0.03 = 0.014$, which multiplies the *saved* fraction instead of the *remaining* fraction.

**Fix:** Change to: "…the total weight bytes read from DRAM are reduced to approximately $0.531 \times 0.03 \approx 0.016$ (about 1.6%) of the dense BF16 baseline." If the phrasing "reduced by a factor of X" is preferred, state "reduced by a factor of approximately $0.016$" (i.e., the traffic is $\approx 1.6\%$ of baseline, a $\approx 98.4\%$ reduction).

---

### Issue 5 — `program_configs_sparse.md`, Section 4, lines ~232–234: Active mask bit counts are wrong by a factor of $K_t = 224$

**File:** `program_configs_sparse.md`, lines ~232–234

**Error:** The static-shape discussion states:

- "$B=1$: mask has 1 bit set (out of $32 \times 224 = 7{,}168$ total)."
- "$B=8$: mask has ~8 bits set per expert dim."
- "$B=32$: mask has ~32 bits set (all experts active)."

Each mask entry `mask[m, k]` is 1 if tile $A[m, k]$ is active (per `sparse_matmul_internals.md` §2.1). For one active expert with $C=1$ ($M_t = 1$), that expert contributes one tile row with $K_t = 224$ K-tile columns. All 224 K-tile entries in that row are active (the token's hidden-state vector spans all $H = 7168$ elements). Therefore:

- $B=1$ (1 active expert): **224 bits set**, not 1.
- $B=8$ (8 active experts): **$8 \times 224 = 1{,}792$ bits set**, not ~8.
- $B=32$ (32 active experts): **$32 \times 224 = 7{,}168$ bits set** (all bits), not ~32.

The text confuses "number of active expert rows" with "number of active mask bits." Each active expert row contributes 224 active bits (one per K-tile), not 1.

**Fix:** Replace the three bullet points with:
- "$B=1$: mask has $1 \times K_t = 224$ bits set (1 active expert row, all 224 K-tile columns active) out of $32 \times 224 = 7{,}168$ total."
- "$B=8$: mask has $\sim 8 \times 224 = 1{,}792$ bits set ($\sim 8$ active expert rows)."
- "$B=32$: mask has $32 \times 224 = 7{,}168$ bits set (all experts active — all bits set)."

---

### Issue 6 — `when_sparse_matmul_wins.md`, Section 2.2 / Section 3, line ~84: Ambiguous intermediate expression `k × E_d / k × N`

**File:** `when_sparse_matmul_wins.md`, line ~84 (end of Section 2.2)

**Error:** The text writes: "The fill rate remains near 1.0 for all prefill scenarios with $B \times S \geq k \times E_d / k \times N = E_d \times N / k = 32$."

The intermediate expression $k \times E_d / k \times N$ is ambiguous. Under standard left-to-right evaluation it reads as $(k \times E_d / k) \times N = E_d \times N = 32 \times 8 = 256$, which contradicts the stated value of 32. The correct derivation is: expert saturation requires $k \times B \times S / N \geq E_d$, i.e., $B \times S \geq E_d \times N / k = 32 \times 8 / 8 = 32$. The intermediate expression should be omitted or written correctly.

**Fix:** Replace with: "The fill rate remains near 1.0 for all prefill scenarios with $B \times S \geq E_d \times N / k = 32 \times 8 / 8 = 32$."

---

### Summary

| # | File | Location | Type |
|---|------|----------|------|
| 1 | `when_sparse_matmul_wins.md` | §3, ~line 102 | Formula substitution error: $B \times S / 4$ should be $B \times S$; fabricated probability explanation should be removed |
| 2 | `when_sparse_matmul_wins.md` | Summary table, "Short prefill" row | Wrong $\rho$ value (~0.5 should be ~1.0) and wrong "Use" recommendation (sparse_matmul should be batched matmul) |
| 3 | `sparse_matmul_internals.md` | §3, line 112 | Per-row savings understated by $K_t=224$; also uses $M_t$ where $K_t$ is correct for FMA block count |
| 4 | `sparse_matmul_internals.md` | §5, ~line 161 | Combined BFP8+sparsity bandwidth factor uses saved fraction (0.47) instead of remaining fraction (0.531) |
| 5 | `program_configs_sparse.md` | §4, ~lines 232–234 | Active mask bit counts are off by factor of $K_t=224$; conflates active expert row count with active tile count |
| 6 | `when_sparse_matmul_wins.md` | §2.2 end, ~line 84 | Ambiguous intermediate expression $k \times E_d / k \times N$ evaluates to 256 under standard precedence, not 32 as claimed |

---

# Agent A Fix — Chapter 4: sparse_matmul for MoE — Pass 1 Fixes Applied

1. `when_sparse_matmul_wins.md` §3 — Corrected sparsity formula substitution: min(32, B×S) not min(32, B×S/4). B=1/S=1 → ρ=0.031. Removed incorrect probability explanation.

2. `when_sparse_matmul_wins.md` summary table — Corrected short-prefill row (B=1, S=128): ρ = 1.0 (all experts active), recommendation changed to "batched matmul".

3. `sparse_matmul_internals.md` §3 — Corrected tile-skip savings: skipping one row saves K_t × N_t weight tile reads and K_t × N_t FMA blocks (K_t = 224 for H=7168).

4. `sparse_matmul_internals.md` §5 — Corrected combined BFP8+sparsity factor: 0.531 × ρ (remaining fraction 1088/2048 ≈ 0.531, not 0.47 which was saved fraction).

5. `program_configs_sparse.md` §4 — Corrected active mask bit counts: B=1→224 bits, B=8→1792 bits, B=32→7168 bits (full mask). Notes that B=32 is the dense case.

6. `when_sparse_matmul_wins.md` §2.2 — Fixed ambiguous expression to E_d × N / k = 32 × 8 / 8 = 32.

---

# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 2

## Pass 1 Fixes — Verification

All 6 fixes confirmed in place:

1. **Fix 1 verified.** `when_sparse_matmul_wins.md` §3 line 102: formula now reads `\min\!\left(32,\ B \times S\right)` (not `B \times S / 4`). Verification line 108: `ρ = min(32, 1) / (32 × 1) = 1/32 ≈ 0.031`. No probability-based paragraph present.

2. **Fix 2 verified.** Summary table line 121: "Short prefill | 1 | 128 | 4 | 1 | ~1.0 | batched matmul" — ρ and Use column both corrected.

3. **Fix 3 verified.** `sparse_matmul_internals.md` line 112: now reads "skipping tile row m eliminates K_t × N_t weight tile reads and K_t × N_t FMA blocks… K_t × N_t × 1088 bytes… K_t × N_t × 32,768 FMAs avoided, where K_t = H/32 = 7168/32 = 224".

4. **Fix 4 verified.** `sparse_matmul_internals.md` §5 line 161: now reads "reduced by a factor of 0.531 × 0.03 ≈ 0.016 relative to a dense BF16 baseline" (remaining fraction 0.531, not saved fraction 0.47).

5. **Fix 5 verified.** `program_configs_sparse.md` §4 lines 232–234: now reads B=1→224 bits set, B=8→1,792 bits set, B=32→7,168 bits set.

6. **Fix 6 verified.** `when_sparse_matmul_wins.md` §2.2 line 162: now reads `$B \times S \geq E_d \times N / k = 32 \times 8 / 8 = 32$`. Ambiguous intermediate expression removed.

---

## New Issues Found

---

### Issue 7 — `when_sparse_matmul_wins.md`, §3 formula (line 92): Formula `active expert slots / (E_d × M_t)` is structurally wrong for M_t > 1, producing values inconsistent with the Long prefill and Large prefill table rows

**File:** `when_sparse_matmul_wins.md`, lines 92–102 and table rows lines 123–124

**Error:** The sparsity ratio formula is stated as:

$$\rho = \frac{\text{active expert slots on this device}}{E_d \times M_t}$$

where "active expert slots" counts the number of local experts that received at least one token. This formula is correct when M_t = 1 (C ≤ 32, decode regime) because each active expert contributes exactly 1 active tile row. However, when M_t > 1 (C > 32, prefill regime with high load), each active expert fills all M_t of its tile rows (all capacity slots are occupied), so active tile rows = active expert slots × M_t, giving:

$$\rho = \frac{\text{active expert slots} \times M_t}{E_d \times M_t} = \frac{\text{active expert slots}}{E_d}$$

The formula as written — dividing by E_d × M_t while the numerator counts only expert slots (not tile rows) — systematically undercounts ρ when M_t > 1 and capacity is well-filled.

**Consequence — table rows that conflict with the formula:**

- **Long prefill (B=1, S=2048, C=64, M_t=2):** Applying the formula: active expert slots = min(32, 2048) = 32. ρ = 32/(32×2) = 0.5. But §2.2 correctly argues that every expert gets k×B×S/E = 64 tokens (filling all C=64 slots), so all 64 tile rows are active and ρ = 64/64 = 1.0. The table states ~1.0 (correct), but the formula applied naively gives 0.5 (wrong). The formula and the table row are inconsistent.

- **Large prefill (B=4, S=2048, C=256, M_t=8):** active expert slots = min(32, 8192) = 32. Formula: ρ = 32/(32×8) = 0.125. Correct value (all C=256 slots filled): ρ = 1.0. The table states ~1.0 (correct); the formula gives 0.125. Inconsistency factor is 8×.

The root cause: in the prefill regime, "active expert slots" is not the right numerator. The numerator should be the number of active tile rows, which equals `active expert slots × ceil(tokens_per_active_expert / 32)` in general. The formula only works when tokens_per_active_expert ≤ 32 (i.e., M_t = 1 or sparse prefill), which is exactly the decode regime. For the prefill regime the formula breaks down but the chapter relies on it throughout §3.

**Fix:** One of two approaches:

*Option A (minimal):* Add a note immediately after the formula warning that it applies only when each active expert occupies at most 1 tile row (M_t = 1, i.e., C ≤ 32). State that for M_t > 1 and high fill rates, ρ → active_experts/E_d (the M_t cancels when all rows are filled). Flag the Long prefill and Large prefill table rows as using the correct direct reasoning (§2.2) rather than the formula.

*Option B (correct the formula):* Restate the numerator as "active tile rows on this device" (not "active expert slots"), where active tile rows = sum over active experts of ceil(tokens_to_expert_e / 32). For the decode case (M_t=1) this reduces to active expert slots. For the fully-packed prefill case this reduces to E_d × M_t = total tile rows, giving ρ = 1.0.

Option A is lower-risk since the table values themselves are correct — only the formula's applicability domain needs a caveat.

---

### Issue 8 — `when_sparse_matmul_wins.md`, §3 table and §6 summary table: "Single decode" ρ lower bound 0.008 is unexplained and inconsistent with the formula

**File:** `when_sparse_matmul_wins.md`, line 116 (§3 table) and line 186 (§6 summary table)

**Error:** Both tables show ρ ∈ [0.008, 0.031] for single-token decode (B=1, S=1). The formula gives exactly:

$$\rho = \frac{\min(32,\ 1)}{32 \times 1} = \frac{1}{32} \approx 0.031$$

The lower bound 0.008 ≈ 1/128 has no derivation in the text and does not correspond to any obvious configuration for B=1, S=1 (exactly 8 experts are activated for one token, distributed across 8 devices, giving 1 active expert per device on average; the per-device count is either 0 or 1 in any single step, not a continuous range with a non-zero lower bound of 0.008). The text provides no explanation for where 0.008 comes from.

Note that 0.008 = 1/128, which would correspond to 0.25 active experts out of 32 — not a physically realizable value for integer routing. The range is not a confidence interval (no variance model is presented). It appears to be a residual artifact from the Pass 1 erroneous formula (which gave ρ = min(32, B×S/4)/(32×1) = min(32, 0.25)/32 = 0.25/32 = 0.0078 ≈ 0.008 for B=1, S=1). The fix to Issue 1 corrected the formula and removed the probability explanation, but this stale 0.008 value was not updated in either table.

**Fix:** Replace "0.008–0.031" with "~0.031" in both the §3 summary table (line 116) and the §6 summary table (line 186). The formula gives a single value (1/32 ≈ 0.031); stochastic variation would produce per-step ρ of 0 or 0.031 (0 or 1 active expert), not a range with a non-zero lower bound.

---

## Summary of New Issues

| # | File | Location | Type |
|---|------|----------|------|
| 7 | `when_sparse_matmul_wins.md` | §3 formula (line 92) and Long/Large prefill table rows (lines 123–124) | Formula `active expert slots / (E_d × M_t)` is incorrect for M_t > 1 with high fill rate; formula gives ρ=0.5 and ρ=0.125 for cases where correct answer is ρ≈1.0; table values are correct but the formula that purports to generate them is wrong |
| 8 | `when_sparse_matmul_wins.md` | §3 table line 116, §6 summary table line 186 | ρ lower bound 0.008 is a stale artifact of the pre-Fix-1 erroneous formula; formula now gives exactly 0.031 for B=1, S=1; the range 0.008–0.031 should be replaced with ~0.031 |

---

# Agent A Fix — Chapter 4: sparse_matmul for MoE — Pass 2 Fixes Applied

7. `when_sparse_matmul_wins.md` §3 — Fixed sparsity formula: ρ = active_experts / E_d (M_t cancels from numerator and denominator). Added note that formula is independent of M_t.

8. `when_sparse_matmul_wins.md` — Removed stale lower bound 0.008 from ρ range for B=1/S=1; replaced with ~0.031 in all tables and text.

---

# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 3

## Pass 2 Fixes — Verification

**Fix 7 verified.** `when_sparse_matmul_wins.md` §3 lines 92–96: formula now reads `ρ = (active expert slots × M_t) / (E_d × M_t) = active expert slots / E_d` with a note that M_t cancels and the formula is independent of M_t. Correct.

**Fix 8 verified.** All occurrences of the B=1/S=1 sparsity value now show `~0.031`: §3 table line 118 ("Single decode | 1 | 1 | 1 | 1 | ~0.031 | sparse_matmul") and §6 summary table line 188 ("Single-token decode (B=1, S=1) | ~0.031 | sparse_matmul strongly preferred"). No instance of 0.008 remains.

---

## New Issues Found

---

### Issue 9 — `sparse_matmul_internals.md`, §7, line 191: Sparsity formula still uses old `E_d × ⌈C/32⌉` denominator, contradicting Fix 7

**File:** `sparse_matmul_internals.md`, line 191

**Error:** The formula in §7 reads:

$$\rho = \frac{\text{number of active expert slots on this device}}{E_d \times \lceil C / 32 \rceil}$$

Fix 7 (applied to `when_sparse_matmul_wins.md` §3) established that the correct formula is $\rho = \text{active\_experts} / E_d$, because each active expert contributes $M_t$ tile rows in the numerator and the denominator $E_d \times M_t$ (total tile rows) contains the same $M_t$, so the $M_t$ factors cancel. The `sparse_matmul_internals.md` §7 formula was not updated to reflect this correction and retains the pre-Fix-7 form with $E_d \times \lceil C/32 \rceil$ in the denominator.

The formula in §7 is numerically correct only for the decode regime ($M_t = 1$, $C \leq 32$) shown in the worked examples immediately below it. For any case with $M_t > 1$ (prefill, high fill rate) the formula underestimates $\rho$ by a factor of $M_t$, producing the same class of error that Fix 7 corrected in `when_sparse_matmul_wins.md`.

**Fix:** Replace the formula at line 191 with:

$$\rho = \frac{\text{number of active expert slots on this device}}{E_d}$$

Add a brief parenthetical noting that $M_t$ cancels (each active expert fills $M_t$ tile rows; total tile rows = $E_d \times M_t$; the $M_t$ factors divide out), consistent with the explanation already present in `when_sparse_matmul_wins.md` §3.

---

### Issue 10 — `sparse_matmul_internals.md`, §7, line 206: Text claims ρ depends on M_t, directly contradicting Fix 7

**File:** `sparse_matmul_internals.md`, line 206

**Error:** The closing sentence of §7 reads: "The second case illustrates an important point: $\rho$ depends on both the number of active expert slots **and** the tile count $M_t$."

Fix 7 established the opposite: $\rho = \text{active\_experts} / E_d$ is independent of $M_t$ (the $M_t$ factors cancel). This sentence was not updated when Fix 7 was applied and now directly contradicts the corrected statement in `when_sparse_matmul_wins.md` §3.

**Fix:** Replace the sentence with one that correctly states the relationship: "The second case ($B=32$) illustrates an important point: at $B=32$, $k \times B / N = 32 = E_d$, so all local expert slots are active and $\rho = 32/32 = 1.0$, leaving no sparsity to exploit. The first case ($B=1$) yields $\rho \approx 0.031 = 1/32$ and is the regime where `sparse_matmul` provides maximum benefit. Note that $\rho = \text{active\_experts}/E_d$ is independent of $M_t$ — see `when_sparse_matmul_wins.md` §3 for the full derivation."

---

### Issue 11 — `when_sparse_matmul_wins.md`, §3, line 110: Verification expression has spurious `× 1` in denominator, a stale remnant of the pre-Fix-7 formula

**File:** `when_sparse_matmul_wins.md`, line 110

**Error:** The verification line reads:

$$\rho = \frac{\min(32,\ 1)}{32 \times 1} = \frac{1}{32} \approx 0.031$$

The denominator is written as `32 × 1` (i.e., $E_d \times M_t$) instead of simply `32` ($E_d$). While numerically correct for this specific case ($M_t = 1$ in the decode regime), the expression `32 × 1` is inconsistent with the corrected formula on line 104, which correctly shows the denominator as just `32` (= $E_d$, with $M_t$ already cancelled). The `× 1` is a stale artifact of the pre-Fix-7 formula `ρ = active_experts / (E_d × M_t)`.

**Fix:** Change `32 \times 1` to `32` so the verification reads:

$$\rho = \frac{\min(32,\ 1)}{32} = \frac{1}{32} \approx 0.031$$

This is consistent with the formula on line 104 and with the explanation that $M_t$ cancels.

---

## Summary of New Issues

| # | File | Location | Type |
|---|------|----------|------|
| 9 | `sparse_matmul_internals.md` | §7, line 191 | Formula still uses pre-Fix-7 denominator `E_d × ⌈C/32⌉`; should be `E_d` alone (M_t cancels); Fix 7 was applied only to `when_sparse_matmul_wins.md`, not to this file |
| 10 | `sparse_matmul_internals.md` | §7, line 206 | Text claims "ρ depends on both active expert slots **and** M_t"; contradicts Fix 7 which established ρ is independent of M_t |
| 11 | `when_sparse_matmul_wins.md` | §3, line 110 | Verification denominator written as `32 × 1` (stale `E_d × M_t` form) instead of `32` (correct `E_d` form from line 104 formula) |

---

# Agent A Fix — Chapter 4: sparse_matmul for MoE — Pass 3 Fixes Applied

9. `sparse_matmul_internals.md` §7, line 191 — Fixed ρ formula denominator: changed `E_d × ⌈C/32⌉` to `E_d` (M_t cancels).

10. `sparse_matmul_internals.md` §7, line 206 — Fixed contradictory sentence: replaced "ρ depends on both the number of active expert slots and the tile count M_t" with "ρ depends only on the number of active expert slots relative to E_d — M_t cancels from numerator and denominator."

11. `when_sparse_matmul_wins.md` §3, line 110 — Fixed stale `× 1` in verification expression denominator: changed `32 × 1` to `32`.

---

# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 4

## Pass 3 Fixes — Verification

**Fix 9 verified.** `sparse_matmul_internals.md` §7 line 191: formula now reads `$$\rho = \frac{\text{number of active expert slots on this device}}{E_d}$$` — denominator is `E_d` alone, not `E_d × ⌈C/32⌉`. Correctly applied.

**Fix 10 verified.** `sparse_matmul_internals.md` §7 line 206: sentence now reads "ρ depends only on the number of active expert slots relative to E_d — M_t cancels from numerator and denominator." No longer claims M_t is a factor. Correctly applied.

**Fix 11 verified.** `when_sparse_matmul_wins.md` §3 line 110: verification expression now reads `\rho = \frac{\min(32,\ 1)}{32} = \frac{1}{32} \approx 0.031` — denominator is `32`, not `32 \times 1`. Correctly applied.

---

## New Issues Found

---

### Issue 12 — `sparse_matmul_internals.md`, §7, line 197: "padded to 32 tiles" is wrong — should be "padded to 32 rows (one tile)"

**File:** `sparse_matmul_internals.md`, line 197

**Error:** The bullet reads:

> `$C = \lceil 8 \times 1 \times 1 / 256 \rceil = 1$ (padded to 32 tiles, so $M_t = 1$).`

"Padded to 32 tiles" implies the padded capacity contains 32 tiles, which would require C_pad = 32 × 32 = 1,024 rows and M_t = 32. That directly contradicts the parenthetical's own conclusion "so M_t = 1". The correct meaning is that C = 1 row is padded to C_pad = 32 rows, which is exactly one tile (M_t = ⌈32/32⌉ = 1). The same C = 1 scenario is described correctly in §4.3 line 144 of the same file: "$C = 1$ is padded to 32 (one tile) in tile layout."

**Fix:** Change "padded to 32 tiles" to "padded to 32 rows (one tile)", making the line read: `$C = \lceil 8 \times 1 \times 1 / 256 \rceil = 1$ (padded to 32 rows (one tile), so $M_t = 1$).`

---

## Summary of New Issues

| # | File | Location | Type |
|---|------|----------|------|
| 12 | `sparse_matmul_internals.md` | §7, line 197 | "padded to 32 tiles" is factually wrong (implies M_t=32); correct phrasing is "padded to 32 rows (one tile)", consistent with §4.3 of the same file |

---

# Agent A Fix — Chapter 4: sparse_matmul for MoE — Pass 4 Fixes Applied

12. `sparse_matmul_internals.md` §7, line 197 — Fixed self-contradictory phrasing: "padded to 32 tiles, so M_t = 1" → "padded to 32 rows — one tile — so M_t = 1". (32 tiles would imply M_t = 32, not 1.)

---

# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 5

## Pass 4 Fix — Verification

**Fix 12 verified.** `sparse_matmul_internals.md` §7 line 197 now reads: `$C = \lceil 8 \times 1 \times 1 / 256 \rceil = 1$ (padded to 32 rows — one tile — so $M_t = 1$).` The phrasing "padded to 32 rows — one tile" is correct and self-consistent: C_pad = 32 rows = 1 tile, so M_t = ⌈32/32⌉ = 1. Fix is correctly applied.

---

# Agent B Review — Chapter 4: sparse_matmul for MoE — Pass 6

## Compression Changes — Verification

**C1 verified.** `sparse_matmul_internals.md` §7 line 193 now reads: "An important point: $\rho$ depends only on the number of active expert slots relative to $E_d$ — $M_t$ cancels from numerator and denominator. A detailed regime analysis, including worked examples for $B=1$, $B=8$, and $B=32$ decode and the prefill regime, is in `when_sparse_matmul_wins.md` §2." The B=1 and B=32 sub-examples are gone. The cross-reference is accurate: `when_sparse_matmul_wins.md` §2.1 contains the B=1 worked example (ρ=1/32≈0.031), the B=8 example (ρ=0.25), and the B=32 example (ρ=1.0), and §2.2 covers the prefill regime. The ρ formula (`active expert slots / E_d`) is preserved unchanged. No load-bearing content was removed.

**C2 verified.** `when_sparse_matmul_wins.md` Summary section (line 186) now reads: "The per-regime $\rho$ values and kernel recommendations are consolidated in the §3 table above." The hybrid strategy sentence (line 188) is preserved: "The practical hybrid strategy: use `sparse_matmul` for decode ($B \leq 16$) and batched matmul for prefill ($S \geq 512$ or $B \geq 32$). This split is formalized as a deployment pattern in Chapter 6 (`decision_guide.md`)." The §3 reference is accurate: the 9-row Qwen3.5-35B regime table is present and complete at lines 116–127. No load-bearing content was removed.

## Chapter-Wide Correctness Check

All previously identified and fixed issues (1–12) remain correctly applied. No regressions introduced by C1 or C2. No new correctness issues found in any of the four files.

No feedback — chapter approved

---

No feedback — chapter approved
