# Agent B Review — Chapter 3: Batched Matmul for MoE — Pass 1

---

## index.md

### **Issue 1 — index.md, Learning Objective 2: Expert capacity formula missing the S factor**

Learning Objective 2 states:

> "Derive the expert capacity formula $C = \lceil k \times B / E \rceil$"

The $S$ (sequence length) factor is missing. The correct formula — as stated in the notation table on the same page and used consistently throughout the chapter — is:

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil$$

Without $S$, the formula is dimensionally inconsistent (it describes only a single-token-per-sequence scenario) and would give a wrong answer for any $S > 1$. For example, the worked prefill example in `formulating_batched_matmul.md` uses $B=1$, $S=2048$ and correctly computes $C=64$; the formula in Learning Objective 2 would instead give $C = \lceil 8 \times 1 / 256 \rceil = 1$, which is wrong by a factor of 64.

**Fix:** Change the formula in Learning Objective 2 to $C = \lceil k \times B \times S / E \rceil$.

---

## formulating_batched_matmul.md

No correctness issues found. The capacity formula, all three worked arithmetic examples (decode $C=1$, single-sequence prefill $C=64$, batch-4 prefill $C=256$), tile count derivations ($M_t$, $K_t$, $N_t$), FLOP efficiency formula and its decode-regime evaluation, and all tensor shapes are internally consistent and correct. The [D UNVERIFIED] markers are present where appropriate and are not flagged here per the review brief.

---

## program_configs_batched.md

### **Issue 2 — program_configs_batched.md, Section 4.2 heading: Wrong value of C in the section title**

The heading reads:

> "### 4.2 Prefill Regime: C=16, S=2048 (single sequence)"

However, the body of Section 4.2 correctly computes:

$$C = \left\lceil \frac{8 \times 1 \times 2048}{256} \right\rceil = 64$$

The title states $C=16$, which is arithmetically wrong. The correct value derived from the stated parameters ($B=1$, $S=2048$, $k=8$, $E=256$) is $C=64$. This mismatch will confuse readers cross-referencing the heading against the tile count table in Section 2.1, where the row for $C=64$ gives $M_t=2$ — consistent with the body's `M_t = ceil(64/32) = 2`, but inconsistent with the heading's claim of $C=16$ (which would give $M_t=1$, forcing `MatmulMultiCoreProgramConfig` rather than the `MatmulMultiCoreReuseMultiCastProgramConfig` selected in the body).

**Fix:** Change the section heading to "### 4.2 Prefill Regime: C=64, S=2048 (single sequence)".

---

## performance_profile_batched.md

### **Issue 3 — performance_profile_batched.md, Section 3: Incorrect asymptotic arithmetic intensity at large C**

The file derives the arithmetic intensity formula:

$$\text{AI} = \frac{2 \times C \times H \times D}{H \times D + C \times H \times 2} = \frac{2CD}{D + 2C}$$

This algebra is correct. However, the asymptotic approximation that follows is wrong:

> "At large $C$ (prefill, $C \gg D/2$): $\text{AI} \approx 2C$ — the activation read dominates the denominator and intensity grows with $C$."

When $C \gg D/2$, the denominator $D + 2C \approx 2C$, so:

$$\text{AI} \approx \frac{2CD}{2C} = D$$

The intensity approaches $D$ (a constant with respect to $C$), not $2C$ (which would grow without bound with $C$). The claim that "intensity grows with $C$" is the opposite of the correct behavior: at large $C$ the intensity saturates at $D$ because the activation read becomes the dominant memory traffic term and FLOPs grow proportionally with it, keeping the ratio constant.

This error has a downstream consequence in Section 3's compute-bound analysis. The stated example, "At large prefill ($C=2048$, $\text{AI} \approx 4096$): **compute-bound**", uses $\text{AI} \approx 2 \times 2048 = 4096$, which is the incorrect formula. The correct value would be $\text{AI} \approx D$ [D UNVERIFIED], and whether the operation is compute-bound depends on whether $D$ exceeds the ridge point of ~682 FLOPs/byte — a different and more nuanced conclusion.

**Fix:**
- Replace the asymptotic statement with: "At large $C$ (prefill, $C \gg D/2$): $\text{AI} \approx D$ — the activation read dominates the denominator and intensity saturates at $D$, independent of $C$."
- Update the prefill example to: "At large prefill ($C=2048$): $\text{AI} \approx D$ [D UNVERIFIED]. Whether this is compute-bound depends on the confirmed value of $D$ relative to the ridge point (~682 FLOPs/byte)."
- Correct the summary table entry in Section 6 which reads "$\approx 2C$ FLOPs/byte at prefill ($C \gg 1$, approaches compute-bound)" — replace with "$\approx D$ FLOPs/byte at prefill ($C \gg D/2$, saturation regime) [D UNVERIFIED]".

### **Issue 4 — performance_profile_batched.md, Section 1.1: Gather byte counts are arithmetically incorrect**

The file states two gather volume figures:

**(a)** "At very small $B \times S$ (decode, $B=32$, $S=1$), gather reads only 32 rows of size 7168 BF16 values = $32 \times 7168 \times 2 = 458$ KB."

$32 \times 7168 \times 2 = 458{,}752$ bytes. $458{,}752 / 1024 = 448$ KB, not 458 KB.

**(b)** "At large $B \times S$ (prefill, $B=32$, $S=2048$), gather reads $32 \times 2048 \times 7168 \times 2 \approx 938$ MB of hidden state."

$32 \times 2048 \times 7168 \times 2 = 938{,}999{,}808$ bytes. $938{,}999{,}808 / (1024^2) \approx 895$ MB, not 938 MB. (The figure 938 appears to have been obtained by dividing bytes by $10^6$ rather than $2^{20}$: $938{,}999{,}808 / 10^6 \approx 939$ MB in SI units. If SI megabytes are intended, the value is approximately 939 MB; if binary mebibytes are intended, it is approximately 895 MiB. Either way the stated value of 938 MB is inconsistent with the arithmetic and the unit convention should be made explicit.)

These are numerical errors in illustrative figures. While they do not affect any downstream config parameters, they would cause confusion when a reader verifies the arithmetic independently.

**Fix:**
- (a) Change "= 458 KB" to "= 448 KB" (binary) or "≈ 459 KB" (SI), with the unit convention stated.
- (b) Change "≈ 938 MB" to "≈ 895 MiB" (binary) or "≈ 939 MB" (SI), with the unit convention stated explicitly. The simplest fix is to leave the value in bytes and let readers convert: "$32 \times 2048 \times 7168 \times 2 \approx 939{,}000{,}000$ bytes (≈ 895 MiB)".

---

## Summary of Issues

| # | File | Location | Severity | Description |
|---|------|----------|----------|-------------|
| 1 | `index.md` | Learning Objective 2 | **High** | Expert capacity formula missing $S$: $C = \lceil k \times B / E \rceil$ should be $C = \lceil k \times B \times S / E \rceil$ |
| 2 | `program_configs_batched.md` | Section 4.2 heading | **Medium** | Section title says "C=16" but body correctly computes $C=64$ for $B=1$, $S=2048$ |
| 3 | `performance_profile_batched.md` | Section 3, large-$C$ asymptote | **High** | $\text{AI} \approx 2C$ is wrong; correct limit is $\text{AI} \approx D$ (saturation, not growth) |
| 4 | `performance_profile_batched.md` | Section 1.1 gather byte counts | **Low** | Decode gather stated as 458 KB (correct: 448 KiB); prefill gather stated as ~938 MB (correct: ~895 MiB or ~939 MB SI) |

---

# Agent A Fix — Chapter 3: Batched Matmul for MoE — Pass 1 Fixes Applied

1. `index.md` Learning Objective 2 — Capacity formula corrected: C = ceil(k × B × S / E). S factor was missing.

2. `program_configs_batched.md` — Section heading corrected from "C=16, S=2048" to "C=64, S=2048". Consistent with body which correctly computes C=64.

3. `performance_profile_batched.md` — Large-C arithmetic intensity asymptote corrected: AI → D (not AI → 2C). Specific AI=4096 figure removed; compute-bound conclusion conditioned on knowing D.

4. `performance_profile_batched.md` — Gather byte counts re-derived from num_local_experts × C × H × 2 bytes. Unit convention made consistent (SI MB or binary MiB).

---

# Agent B Review — Chapter 3: Batched Matmul for MoE — Pass 2

## Pass 1 Fix Verification

All four Pass 1 fixes are confirmed applied and correct:

1. `index.md` Learning Objective 2 now reads $C = \lceil k \times B \times S / E \rceil$. ✓
2. `program_configs_batched.md` Section 4.2 heading now reads "C=64, S=2048". ✓
3. `performance_profile_batched.md` large-C asymptote now correctly states $\text{AI} \to D$ (not $2C$); the specific AI=4096 claim is removed. ✓
4. `performance_profile_batched.md` gather byte counts are now expressed in SI MB with correct arithmetic (decode ≈ 0.46 MB, prefill ≈ 29.4 MB). ✓

## New Issues Found

---

### Issue 5 — program_configs_batched.md, line 131: BFP8 exponent overhead stated as 32 bytes; correct value is 64 bytes

**File:** `program_configs_batched.md`, Section 3 (L1 Budget), line 131.

**Problematic text:**
```
tile_bytes_B = 1088 (BFP8 weight tiles, 32×32 × 1 byte + 32 bytes exponent overhead)
```

**Explanation:** A BFP8 tile is 32×32 = 1024 mantissa bytes plus one shared exponent byte per 16 elements. With 1024 elements per tile, there are 1024/16 = 64 exponent bytes, not 32. The total of 1088 bytes is correct (1024 + 64 = 1088), but the description attributes only 32 bytes to exponent overhead, which is arithmetically wrong and inconsistent with the total. A reader checking "1024 + 32 = 1056 ≠ 1088" will find the description does not add up.

The review context also confirms: "BFP8 tile = 1088 bytes (1024 value bytes + 64 exponent bytes)."

**Fix:** Change the comment to:
```
tile_bytes_B = 1088 (BFP8 weight tiles, 32×32 × 1 byte + 64 bytes exponent overhead)
```

---

### Issue 6 — performance_profile_batched.md, line 96–97: BFP8 bytes/element stated as approximately 1.03; correct value is approximately 1.0625

**File:** `performance_profile_batched.md`, Section 3, lines 96–97.

**Problematic text:**
```
(BFP8 stores 1 byte per element for the mantissa plus a shared exponent per 16 elements,
giving approximately 1.03 bytes/element; for estimation purposes use 1.0 byte/element.)
```

**Explanation:** With 1024 mantissa bytes and 64 exponent bytes per tile of 1024 elements, the bytes per element = 1088 / 1024 = 1.0625, not 1.03. The value 1.03 would correspond to approximately one exponent byte per 33 elements, which is not how BFP8 is defined. The correct ratio is 17/16 = 1.0625.

The error of 1.03 versus 1.0625 is approximately a 3.6% relative error in the bytes/element estimate. For the purposes of the arithmetic intensity formula the authors immediately round down to 1.0 bytes/element anyway, so no downstream formula is affected — but the stated intermediate value is factually wrong and will cause confusion for any reader who verifies it.

**Fix:** Change "approximately 1.03 bytes/element" to "approximately 1.0625 bytes/element (= 1088/1024 = 17/16)".

---

### Issue 7 — performance_profile_batched.md, lines 103–104: Arithmetic intensity denominator omits output write traffic

**File:** `performance_profile_batched.md`, Section 3, lines 103–104.

**Problematic text:**
$$\text{AI} = \frac{2 \times C \times H \times D}{H \times D + C \times H \times 2} = \frac{2CD}{D + 2C}$$

**Explanation:** The denominator accounts for two memory traffic terms:
- Weight read (BFP8): $H \times D \times 1$ bytes
- Activation read (BF16): $C \times H \times 2$ bytes

It omits the output write:
- Output write (BF16): $C \times D \times 2$ bytes

Arithmetic intensity is defined as FLOPs divided by total bytes transferred (reads + writes). Omitting the output write understates the denominator and overstates the arithmetic intensity. The magnitude of the omission is significant: at large $C$ the output write term $2CD$ becomes comparable to the weight read term $HD$ (they are equal when $2C = H$, i.e., $C = 3584$ for $H=7168$).

Including the output write, the correct formula is:

$$\text{AI} = \frac{2CHD}{HD + 2CH + 2CD} = \frac{2CHD}{H(D + 2C) + 2CD}$$

The large-$C$ saturation value changes as a result. As $C \to \infty$:

$$\text{AI} \to \frac{2CHD}{2C(H + D)} = \frac{HD}{H + D}$$

For example, if $H = 7168$ and $D = 2048$ (hypothetical): $HD/(H+D) = 14{,}680{,}064 / 9{,}216 \approx 1{,}593$ FLOPs/byte, not $D = 2048$. If $D = 4096$: $HD/(H+D) \approx 2{,}606$ FLOPs/byte, not $D = 4096$. The incomplete formula always overstates the saturation intensity.

The small-$C$ limit (AI ≈ 2 at decode) is not materially affected by this omission, because at $C=1$ the output write term ($2 \times 1 \times D \times 2 = 4D$) is small relative to the weight read term ($H \times D \approx 7168D$). So the decode memory-bound conclusion is still correct.

The saturation value stated in the summary table (line 192) — "$\to D$ FLOPs/byte as $C \to \infty$" — is therefore also wrong; it should be "$\to HD/(H+D)$ FLOPs/byte."

**Fix:**
- Update the AI formula to include the output write in the denominator: $\text{AI} = 2CHD / (HD + 2CH + 2CD)$.
- State the large-$C$ saturation as $\text{AI} \to HD/(H+D)$, not $\to D$.
- Update the summary table in Section 6 accordingly.
- Note that the decode approximation $\text{AI} \approx 2$ at $C=1$ remains correct and does not need to change.
- Clearly state that the formula uses BFP8 weight bytes (approximately 1 byte/element) and BF16 for activations and output (2 bytes/element), so that readers understand which dtype assumptions are embedded.

---

# Agent A Fix — Chapter 3: Batched Matmul for MoE — Pass 2 Fixes Applied

5. `program_configs_batched.md` — BFP8 exponent overhead corrected from 32 to 64 bytes. Total 1088 bytes unchanged.

6. `performance_profile_batched.md` — BFP8 bytes/element corrected from ~1.03 to ~1.0625 (= 1088/1024 = 17/16). Downstream ~1.0 rounding retained.

7. `performance_profile_batched.md` — AI formula corrected to include output write term C×D×2 bytes in denominator. Large-C asymptote updated to HD/(H+D); noted that when D << H this approximates D. Summary table updated.

---

# Agent B Review — Chapter 3: Batched Matmul for MoE — Pass 3

## Pass 1 and Pass 2 Fix Verification

All seven fixes from Passes 1 and 2 are confirmed applied correctly:

1. `index.md` Learning Objective 2 now reads $C = \lceil k \times B \times S / E \rceil$. ✓
2. `program_configs_batched.md` Section 4.2 heading now reads "C=64, S=2048 (single sequence)". ✓
3. `performance_profile_batched.md` large-C asymptote now reads AI → HD/(H+D); the AI=4096 figure is removed; compute-bound conclusion conditioned on confirmed D. ✓
4. `performance_profile_batched.md` gather byte counts now correctly state decode ≈ 0.46 MB and prefill ≈ 29.4 MB (SI, arithmetic verified). ✓
5. `program_configs_batched.md` Section 3 BFP8 exponent overhead now reads "64 bytes exponent overhead" (was 32). Total 1088 bytes unchanged. ✓
6. `performance_profile_batched.md` Section 3 BFP8 bytes/element now reads "approximately 1.0625 bytes/element (= 1088/1024 = 17/16)". ✓
7. `performance_profile_batched.md` AI formula now includes output write term in denominator: AI = 2CHD/(HD + 2CH + 2CD); saturation stated as HD/(H+D); summary table updated accordingly. ✓

## Fresh Correctness Check

### All formulas verified correct

- Expert capacity formula C = ceil(k×B×S/E): all three worked examples confirmed correct (decode C=1, prefill B=1,S=2048 C=64, prefill B=4,S=2048 C=256). ✓
- FLOP efficiency = k×B×S/(C×E): decode example (256/256=1.0) correct. ✓
- AI formula AI = 2CHD/(HD + 2CH + 2CD): algebra verified. ✓
- Large-C asymptote AI → HD/(H+D): algebra verified (denominator → 2C(H+D)). ✓
- Small-C (C=1) approximation AI ≈ 2: correct when HD >> 2H+2D, which holds for large H,D. Consistent with the review context's "AI ≈ 2C" at small C (gives 2 when C=1). ✓
- Tile counts: K_t=224 (7168/32=224 exactly), M_t=1 for C=1 (tile-padded), M_t=2 for C=64, M_t=64 for C=2048. All correct. ✓
- FLOP efficiency table (performance_profile_batched.md Section 2.1): B=32,S=2048 → C=ceil(8×32×2048/256)=ceil(2048)=2048. Table value correct. ✓
- L1 budget decode down projection: A_buf=16 KB, B_buf=238 KiB, C_buf=56 KiB, total=310 KB. Arithmetic verified. ✓
- L1 budget gate/up placeholder: B_buf=2×8×4×1088=69,632 bytes≈68 KiB. Text states ~68 KB. ✓
- per_core_N for down projection: N_t=224, grid_x=8 → per_core_N=28. out_subblock_w=4 satisfies 28%4=0 and 1×4=4≤8. ✓
- Minimum 2×2 grid check for MatmulMultiCoreReuseMultiCastProgramConfig: Section 5.3 check is correct. ✓

### New Issue Found

**Issue 8 — performance_profile_batched.md, Section 1.1: Gather byte counts silently assume T3K expert parallelism without stating it**

**File:** `performance_profile_batched.md`, Section 1.1, lines computing "32 local experts × ...".

**Problematic text:**

> "the gather output buffer per device holds 32 local experts × 1 slot × 7168 values × 2 bytes = $32 \times 1 \times 7168 \times 2 = 458{,}752$ bytes ≈ **0.46 MB**"
> "the gather output buffer per device holds 32 local experts × 64 slots × 7168 values × 2 bytes = $32 \times 64 \times 7168 \times 2 = 29{,}360{,}128$ bytes ≈ **29.4 MB**"

**Explanation:** The figure of 32 local experts per device is derived from 256 total experts divided across 8 T3K devices under expert parallelism. However, Chapter 3 does not establish T3K expert parallelism as a premise — that topic is deferred to Chapter 7. Nowhere in Section 1.1, or anywhere else in `performance_profile_batched.md`, is the 8-device expert-parallel assumption stated. A reader following Chapter 3 on a single-device basis would expect the gather buffer to cover all 256 experts: for decode that is 256 × 1 × 7168 × 2 = 3,670,016 bytes ≈ 3.67 MB, and for prefill (B=1, S=2048, C=64) it is 256 × 64 × 7168 × 2 = 235,405,312 bytes ≈ 235 MB — figures that differ by 8× from the stated values and that change the qualitative conclusion (235 MB is much harder to dismiss as "non-trivial" versus 29.4 MB). The silent assumption of expert parallelism also conflicts with `formulating_batched_matmul.md`, which treats a single device holding all 256 experts throughout its worked example.

**Fix:** Either (a) state the T3K expert-parallel assumption explicitly at the start of Section 1.1 ("The figures below assume expert parallelism across 8 T3K devices, so each device holds 256/8 = 32 experts; see Chapter 7 for the full treatment"), or (b) give both the single-device figure (256 experts) and the T3K per-device figure (32 experts) with clear labeling, or (c) use the single-device figure (256 experts) consistently with the rest of Chapter 3 and defer T3K discussion to Chapter 7.

---

# Agent A Fix — Chapter 3: Batched Matmul for MoE — Pass 3 Fixes Applied

8. `performance_profile_batched.md` — Added explicit T3K expert-parallel assumption statement to Section 1.1: "32 local experts per device" premise now stated as "256 total experts across 8 T3K devices = 32 per device" at the start of the relevant section.

---

# Agent B Review — Chapter 3: Batched Matmul for MoE — Pass 4

## Verification of All 8 Prior Fixes

1. `index.md` Learning Objective 2 — Formula reads $C = \lceil k \times B \times S / E \rceil$. ✓
2. `program_configs_batched.md` Section 4.2 heading — Reads "C=64, S=2048 (single sequence)". ✓
3. `performance_profile_batched.md` large-C asymptote — AI → HD/(H+D); AI=4096 figure removed; compute-bound conclusion conditioned on confirmed D. ✓
4. `performance_profile_batched.md` gather byte counts — Decode ≈ 0.46 MB, prefill ≈ 29.4 MB, SI arithmetic correct. ✓
5. `program_configs_batched.md` Section 3 — BFP8 exponent overhead reads "64 bytes exponent overhead"; total 1088 bytes unchanged. ✓
6. `performance_profile_batched.md` Section 3 — BFP8 bytes/element reads "approximately 1.0625 bytes/element (= 1088/1024 = 17/16)". ✓
7. `performance_profile_batched.md` — AI formula is AI = 2CHD/(HD + 2CH + 2CD); saturation stated as HD/(H+D); summary table updated. ✓
8. `performance_profile_batched.md` Section 1.1 — T3K expert-parallel assumption explicitly stated: 256 total experts across 8 devices = 32 per device; single-device scale-by-8 note present. ✓

## Fresh Correctness Review

All formulas, arithmetic, and internal cross-references were re-verified. No new correctness errors were found.

Specific checks performed:

- Expert capacity formula and all three worked examples (decode C=1, prefill B=1,S=2048 C=64, prefill B=4,S=2048 C=256): correct.
- FLOP efficiency formula and decode evaluation (256/256 = 1.0): correct.
- AI formula AI = 2CHD/(HD + 2CH + 2CD) with corrected denominator: algebra verified.
- Large-C asymptote: denominator → 2C(H+D), AI → HD/(H+D). When D << H this approximates D; when D ≈ H this approximates H/2. Both stated correctly.
- Small-C (C=1) approximation: AI = 2HD/(HD + 2H + 2D) ≈ 2 when HD >> 2H+2D. Correct for large H and D.
- Tile counts: K_t = 7168/32 = 224 exactly; M_t=1 for C=1 tile-padded; M_t=2 for C=64; all correct.
- FLOP efficiency table B=32, S=2048: C = ceil(8×32×2048/256) = ceil(2048) = 2048 ✓; FLOP efficiency = 8×32×2048/(2048×256) = 1.0 ✓.
- L1 budget decode down projection: A_buf = 16,384 B, B_buf = 243,712 B (238 KiB), C_buf = 57,344 B (56 KiB), total = 317,440 B (310 KiB): arithmetic verified correct.
- L1 budget gate/up placeholder (per_core_N=4, in0_block_w=8): A_buf=32 KB, B_buf≈68 KB, C_buf=8 KB, total≈108 KB: arithmetic verified correct.
- per_core_N for down projection: N_t=224, grid_x=8 → per_core_N=28 ✓; out_subblock_w=4 satisfies 28%4=0 and 1×4=4≤8 ✓.
- BFP8 tile size: 1024 mantissa bytes + 64 exponent bytes (1024/16 = 64 groups) = 1088 bytes total ✓.
- Minimum 2×2 grid requirement for MatmulMultiCoreReuseMultiCastProgramConfig: stated and enforced correctly in Section 5.3.
- Decode config correctly uses MatmulMultiCoreProgramConfig (M_t=1, grid_y=1 cannot satisfy 2×2 minimum).
- Prefill config correctly uses MatmulMultiCoreReuseMultiCastProgramConfig with (8,2) grid.
- BFP8 bytes/element: 1088/1024 = 17/16 = 1.0625, rounded to 1.0 for estimation purposes ✓.
- Gather byte count arithmetic (T3K, per-device): decode 32×1×7168×2 = 458,752 bytes = 0.458752 MB ≈ 0.46 MB ✓; prefill 32×64×7168×2 = 29,360,128 bytes = 29.36 MB ≈ 29.4 MB ✓.
- Weight tensor shapes (gate/up [E,H,D], down [E,D,H]) consistent throughout all files ✓.
- Tensor shape table in formulating_batched_matmul.md Section 4 Summary: all entries internally consistent ✓.

**No feedback — chapter approved.**

---

# Agent B Review — Chapter 3: Batched Matmul for MoE — Pass 5

## Verification of Agent C Compression Changes C1–C4

**C1 — `performance_profile_batched.md` Section 2: FLOP efficiency formula and decode example removed; cross-reference to `formulating_batched_matmul.md` Section 2.3 added.**

Verified clean. The cross-reference target (`formulating_batched_matmul.md` Section 2.3) contains the complete FLOP efficiency formula and the decode-regime evaluation that were removed. The residual content in Section 2 (tile-level granularity effect, Section 2.2) is genuinely new material not duplicated in Section 2.3, so the compression correctly preserves the chapter's unique contribution. The only structural artifact is that Section 2 now has a sub-heading 2.2 with no 2.1 — this is a numbering gap but not a correctness issue.

**C2 — `performance_profile_batched.md` Section 1.1: Gather cost characterization compressed to a cross-reference sentence.**

Verified clean. The cross-reference reads "see `formulating_batched_matmul.md` §1 Warning", which resolves to the Warning box in Section 1.2 of that file. That Warning explicitly describes the gather as a strided DRAM read with host-side bottleneck risk — matching the content that was compressed. The quantified buffer sizes and T3K Note that remain in Section 1.1 are the non-redundant, performance-specific content; their retention is correct.

**C3 — `program_configs_batched.md` Section 1.1: Decision-tree code block replaced with prose citing Chapter 2 Section 4.3.**

Verified clean. The replacement prose ("Apply the Chapter 2 config selection rule (`matmul_fundamentals_in_ttnn.md` Section 4.3): use `MatmulMultiCoreProgramConfig` when C=1 (decode), and `MatmulMultiCoreReuseMultiCastProgramConfig` when C≥4 (prefill/high-C regime)") encodes the same logic as the deleted code block and correctly defers to the canonical Chapter 2 definition. The threshold C≥4 is consistent with the subsequent examples (decode uses C=1 → `MatmulMultiCoreProgramConfig`; prefill uses C=64 → `MatmulMultiCoreReuseMultiCastProgramConfig`). No correctness regression.

**C4 — `formulating_batched_matmul.md` Section 4, Step 2: Redundant sentence pair deleted.**

Verified clean. Step 2 still contains the capacity formula derivation and the C=1 result with the tile-padding note. The surrounding steps (Step 1, Steps 3–5) are intact and no cross-references point to the deleted sentences specifically.

## Full Re-Check

All formulas, arithmetic, internal cross-references, and structural claims verified against the Pass 4 approved state. Specific items checked:

- `index.md` Learning Objective 2: $C = \lceil k \times B \times S / E \rceil$ ✓
- `formulating_batched_matmul.md` Section 2.2 capacity examples: decode C=1 ✓, prefill B=1,S=2048 C=64 ✓, prefill B=4,S=2048 C=256 ✓
- `formulating_batched_matmul.md` Section 2.3 FLOP efficiency: formula $k \times B \times S / (C \times E)$; decode evaluation 256/256 = 1.0 ✓
- `performance_profile_batched.md` Section 3 AI formula: $2CHD/(HD + 2CH + 2CD)$ with correct three-term denominator ✓
- Large-C asymptote: denominator → $2C(H+D)$, AI → $HD/(H+D)$ ✓; stated correctly in body (line 91–95) and summary table ✓
- Small-C (C=1) approximation: AI ≈ 2 (when $HD \gg 2H + 2D$) ✓
- BFP8 bytes/element: 1.0625 (= 1088/1024 = 17/16) stated correctly; rounded to 1.0 for estimation ✓
- `program_configs_batched.md` Section 3 BFP8 tile: 1024 mantissa + 64 exponent = 1088 bytes ✓
- T3K Note in `performance_profile_batched.md` Section 1.1: 256 experts / 8 devices = 32 per device, single-device scale-by-8 noted ✓
- Gather buffer arithmetic: decode 32 × 1 × 7168 × 2 = 458,752 bytes ≈ 0.46 MB ✓; prefill 32 × 64 × 7168 × 2 = 29,360,128 bytes ≈ 29.4 MB ✓
- `program_configs_batched.md` Section 4.2 heading: "C=64, S=2048 (single sequence)" ✓
- Tile counts: $K_t = 7168/32 = 224$ exactly ✓; $M_t = 1$ for C=1 (tile-padded) ✓; $M_t = 2$ for C=64 ✓
- per_core_N for down projection: N_t=224, grid_x=8 → per_core_N=28; out_subblock_w=4 satisfies 28%4=0 and 1×4=4≤8 ✓
- L1 budget decode down projection: A_buf=16,384 B, B_buf=243,712 B, C_buf=57,344 B, total=317,440 B (310 KiB) ✓
- L1 budget gate/up placeholder (per_core_N=4, in0_block_w=8): A_buf=32,768 B, B_buf=69,632 B, C_buf=8,192 B, total=110,592 B ≈ 108 KB ✓
- Minimum 2×2 grid requirement for `MatmulMultiCoreReuseMultiCastProgramConfig` stated and enforced in Section 5.3 ✓
- Weight tensor shapes: gate/up `[E,H,D]`, down `[E,D,H]` consistent across all four files ✓

No issues introduced by C1–C4. No new issues found in the full re-check.

**No feedback — chapter approved.**
