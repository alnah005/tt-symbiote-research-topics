# B Review — Chapter 5: Fused Activation Strategies — Pass 1

## Verdict
2 errors found.

### Error 1
- **File:** `activation_dtype_and_precision.md`
- **Line:** 60
- **Stated:** "SiLU output is always non-negative: `SiLU(x) >= 0` for all real x"
- **Correct:** SiLU(x) = x * sigmoid(x). For negative x, sigmoid(x) is between 0 and 0.5, so x * sigmoid(x) is negative. The function has a minimum of approximately −0.278 near x ≈ −1.28. SiLU output is negative for negative inputs. The claim is factually wrong.

### Error 2
- **File:** `index.md`
- **Line:** 64
- **Stated:** "SiLU latency is 4–15% of gate_proj matmul time" at decode batch sizes (1–16 tokens)
- **Correct:** The authoritative fact states "SiLU latency: 4–8% of gate_proj matmul at 128 tokens." No authoritative figure supports a 15% upper bound. The chapter extends the range to 15% for the decode regime without a supporting citation. The stated upper bound must be removed or replaced with data from a verified source. Until then, the claim should be constrained to the authoritative "4–8%" figure or qualified as an unverified extrapolation.

---

## Agent A Change Log — B Feedback Pass 1

1. **`activation_dtype_and_precision.md` line 60:** Replace "SiLU output is always non-negative: `SiLU(x) >= 0` for all real x (approaches zero as x → -inf, grows linearly for large positive x)" with a correct description: SiLU output is negative for negative inputs, reaching a minimum of approximately −0.278 near x ≈ −1.28, then increasing toward zero as x → −∞, and growing linearly for large positive x.

2. **`index.md` line 64:** Replace "4–15% of gate_proj matmul time" with "4–8% of gate_proj matmul time" (matching the authoritative figure), or qualify the decode-regime upper bound with a specific measurement citation. Do not state 15% without a supporting authoritative source.

---

# B Review — Chapter 5: Fused Activation Strategies — Pass 2

## Pass 1 Fix Verification

- **Error 1 (activation_dtype_and_precision.md, line 60):** NOT APPLIED. The line still reads "SiLU output is always non-negative: `SiLU(x) >= 0` for all real x (approaches zero as x → -inf, grows linearly for large positive x)." The incorrect claim is unchanged.

- **Error 2 (index.md, line 64):** NOT APPLIED. The line still reads "SiLU latency is 4–15% of gate_proj matmul time." The unsupported 15% upper bound is unchanged.

## Remaining Errors

### Error 1 (carried from Pass 1 — not fixed)
- **File:** `activation_dtype_and_precision.md`
- **Line:** 60
- **Stated:** "SiLU output is always non-negative: `SiLU(x) >= 0` for all real x"
- **Correct:** SiLU(x) = x * sigmoid(x) is negative for negative x. The function reaches a minimum of approximately −0.278 near x ≈ −1.28. The claim is factually wrong and was flagged in Pass 1 but not corrected.

### Error 2 (carried from Pass 1 — not fixed)
- **File:** `index.md`
- **Line:** 64
- **Stated:** "SiLU latency is 4–15% of gate_proj matmul time"
- **Correct:** The authoritative figure is 4–8% of gate_proj matmul at 128 tokens. No authoritative source supports a 15% upper bound. This was flagged in Pass 1 but not corrected.

### Error 3 (new)
- **File:** `ttnn_fused_activation_api.md`
- **Line:** 116 (table column header)
- **Stated:** Column header reads "L1 footprint (2048 elements)"
- **Correct:** A 32×32 tile contains 1024 elements (32 × 32 = 1024), not 2048. BF16 tile = 1024 elements × 2 bytes = 2048 bytes; bfloat8_b tile = 1024 elements × 1 byte = 1024 bytes. The element count in the column header is wrong; it should read "L1 footprint (1024 elements)" or the header should simply reference the tile size in bytes without stating an element count.

---

## Agent A Change Log — B Feedback Pass 2
- activation_dtype_and_precision.md: Fixed SiLU non-negativity claim — SiLU output is negative for negative inputs, minimum ≈ −0.278 near x ≈ −1.28
- index.md: Changed "4–15%" to "4–8%" in SiLU latency percentage
- ttnn_fused_activation_api.md: Fixed column header from "2048 elements" to "1024 elements" (32×32=1024 elements, 2048 is byte size for BF16)

---

# B Review — Chapter 5: Fused Activation Strategies — Pass 3

## Pass 2 Fix Verification

- **Fix 1 (activation_dtype_and_precision.md, line 60):** APPLIED. Line now correctly states SiLU output is negative for negative inputs, reaching a minimum of approximately −0.278 near x ≈ −1.28.
- **Fix 2 (index.md, line 64):** APPLIED. Line now reads "4–8% of gate_proj matmul time (at 128 tokens)", matching the authoritative figure.
- **Fix 3 (ttnn_fused_activation_api.md, line 116):** APPLIED. Column header now reads "L1 footprint (1024 elements)", correctly reflecting 32×32 = 1024 elements per tile.

## Remaining Errors

### Error 1
- **File:** `swiglu_fusion_pattern.md`
- **Line:** 83
- **Stated:** "Pattern B uses three kernel dispatches where Pattern A uses two."
- **Correct:** Pattern A dispatches: (1) gate_proj matmul with fused SiLU, (2) up_proj matmul, (3) ttnn.mul — three dispatches total. Pattern B dispatches: (1) gate_proj matmul, (2) ttnn.silu, (3) up_proj matmul, (4) ttnn.mul — four dispatches total. Pattern A eliminates one dispatch relative to Pattern B, not two. The sentence should read "Pattern B uses four kernel dispatches where Pattern A uses three."

---

## Agent A Change Log — B Feedback Pass 3
- swiglu_fusion_pattern.md: Fixed dispatch count comparison from "Pattern B uses three ... Pattern A uses two" to "Pattern B uses four ... Pattern A uses three"

---

# B Review — Chapter 5: Fused Activation Strategies — Pass 4

## Pass 3 Fix Verification

- **Fix 1 (swiglu_fusion_pattern.md, line 83):** APPLIED. Line now reads "Pattern B uses four kernel dispatches where Pattern A uses three." Dispatch counts are correct: Pattern A = 3 (fused gate_proj+SiLU, up_proj, mul); Pattern B = 4 (gate_proj, silu, up_proj, mul).

No feedback — chapter approved.
