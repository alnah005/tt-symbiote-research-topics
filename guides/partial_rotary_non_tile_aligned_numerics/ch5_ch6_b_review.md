# B Review — Chapters 5 and 6: Model Config Audit and Recommendations

## Pass 1

### Issues found: 3

---

**Issue 1 — `inv_freq` formula: factor-of-2 missing in code (`recommended_fix.md`, lines 51–54)**

Quote:
```python
    # inv_freq[i] = 1 / base^(2i / rotary_dim) for i in [0, rotary_half)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_half, dtype=torch.float32) / rotary_dim)
    )  # shape: [rotary_half]
```

What is wrong: The comment correctly states the standard RoPE formula `1 / base^(2i / rotary_dim)`, but the code computes `1 / base^(i / rotary_dim)` — the factor of 2 is missing from the exponent. `torch.arange(0, rotary_half)` yields `[0, 1, 2, ..., rotary_half-1]`, so `arange / rotary_dim` gives exponents `0/rotary_dim, 1/rotary_dim, ..., (rotary_half-1)/rotary_dim`. The correct exponents are `0/rotary_dim, 2/rotary_dim, ..., (rotary_dim-2)/rotary_dim` (i.e., the even indices). This produces frequencies that are off by a factor of 2 across all rotary positions — a systematic numerical error that would cause PCC failure on Test Case 2 even after the strategy C fix is applied.

Correction: Either multiply arange by 2, or divide by `rotary_half` (which is equivalent since `2i / rotary_dim == i / rotary_half` when `rotary_half = rotary_dim // 2`):

```python
    # Option A: explicit factor of 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_half, dtype=torch.float32) * 2.0 / rotary_dim)
    )
    # Option B: divide by rotary_half (equivalent, matches common implementations)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_half, dtype=torch.float32) / rotary_half)
    )
    # Option C: step by 2 over [0, rotary_dim)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
```

All three produce identical results. The current code matches none of them.

---

**Issue 2 — "80 out of 128 output elements" corruption count is wrong (`the_rotary_dim_48_test_case.md`, line 164)**

Quote:
> 80 out of 128 output elements are either zeroed or carry wrong-paired rotation values.

What is wrong: The four corruption categories described immediately above this sentence cover all 128 elements exhaustively with no overlap:
- Positions `[0, 48)`: 48 elements — corrupted (wrong pairing offset)
- Positions `[48, 64)`: 16 elements — zeroed
- Positions `[64, 112)`: 48 elements — corrupted (wrong pairing offset)
- Positions `[112, 128)`: 16 elements — zeroed

48 + 16 + 48 + 16 = 128. Every output element falls into one of these four categories; zero elements are correct. The "80 out of 128" figure is internally inconsistent with the element-level analysis the text itself presents. The correct statement is that all 128 output elements are wrong (either zeroed or produced with the wrong pairing partner).

Correction: Replace "80 out of 128 output elements are either zeroed or carry wrong-paired rotation values" with "All 128 output elements are wrong — 32 are zeroed (positions [48,64) and [112,128)) and the remaining 96 carry wrong-partner rotation values (positions [0,48) and [64,112))."

---

**Issue 3 — Unresolved contradiction about `rotary_dim=64, head_dim=128` in the current (unfixed) code (`is_this_dead_code.md`, lines 17–22)**

Quote:
> For all currently supported models, `rotary_dim=64` and `64 % 32 == 0`. The padding branch is skipped and cos/sin with `shape[-1]=64` is passed directly to `ttnn.experimental.rotary_embedding` with `head_dim=128`.
>
> Wait — if `rotary_dim=64` and `head_dim=128`, then `cos.shape[-1]=64 != head_dim=128`, and the `TT_FATAL` documented in Chapter 2 should still fire. This apparent contradiction is resolved by examining how `TTNNRotaryPositionEmbedding` handles the tile-aligned case: when `rotary_dim % 32 == 0` and `rotary_dim == head_dim / 2`, the class **may** take a different code path or the padding **may** bring cos/sin from 64 to 128.

What is wrong: The text identifies a real contradiction and then resolves it with speculation ("may take a different code path or the padding may bring cos/sin from 64 to 128"). This is not a resolution — it is a hedge. The chapter's stated conclusion is "no currently supported model exercises the non-tile-aligned bug path," but the analysis here cannot establish that, because it does not know whether `rotary_dim=64` itself triggers a TT_FATAL or silent corruption on the same shape-mismatch path. If `cos.shape[-1]=64` is truly passed to the op unchanged (as asserted in the first bullet), then the `TT_FATAL` applies to the tile-aligned case too — which would mean the tile-aligned path is also broken, contradicting the stated finding. If instead the tile-aligned path has additional padding logic that brings cos to `[..., 128]`, then that logic should be identified and described, not speculated about.

The Note that follows ("The exact behavior for `rotary_dim=64, head_dim=128` in the current tt-symbiote code should be verified against `rope.py`") acknowledges the gap but does not fill it. For a chapter whose purpose is to audit whether production models are broken, leaving the production-model code path unresolved is a technical error of omission.

Correction: Verify the actual `rope.py` behavior for `rotary_dim=64, head_dim=128` before publishing. The resolution must be one of: (a) the tile-aligned branch has a separate padding call that brings cos/sin to `head_dim` width (document which one), (b) the tile-aligned path takes a code route that does not invoke `ttnn.experimental.rotary_embedding` at all, or (c) the op accepts `cos.shape[-1] = head_dim / 2` in the tile-aligned case via a different validation branch. Until one of these is confirmed against the actual source, the chapter's key finding ("bug is latent dead code, production models unaffected") is unsubstantiated.

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:

1. **`recommended_fix.md`, `inv_freq` computation (lines 52–54):** Fix the exponent by multiplying `arange` by 2 (or dividing by `rotary_half`). Update the comment to match the corrected code.

2. **`the_rotary_dim_48_test_case.md`, corruption count (line 164):** Replace "80 out of 128" with the accurate figure (128 out of 128, i.e., all elements), and update the breakdown to match: 32 zeroed, 96 wrong-partner-paired.

3. **`is_this_dead_code.md`, `rotary_dim=64` resolution (lines 17–22):** Resolve the stated contradiction by verifying `rope.py` and replacing the speculative "may take a different code path" language with a concrete description of what actually happens in the tile-aligned case. If verification is deferred, mark the section explicitly as unverified and gate the chapter's key finding accordingly.

---

## Pass 2

### Verification of Pass 1 fixes

**Fix 1 — `recommended_fix.md`, `inv_freq` formula: APPLIED**

Current text (lines 52–55):
```python
    # Note: divide by rotary_half (== rotary_dim / 2) to get the factor-of-2 in the exponent.
    # Equivalently: arange * 2.0 / rotary_dim. The two forms are numerically identical.
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_half, dtype=torch.float32) / rotary_half)
    )  # shape: [rotary_half]
```

The divisor is now `rotary_half` (not `rotary_dim`), which correctly encodes `2i / rotary_dim`. The explanatory comment is present. Fix confirmed.

**Fix 2 — `the_rotary_dim_48_test_case.md`, corruption count: APPLIED**

Current text (line 164):
> All 128 output elements are wrong — 32 are zeroed (positions `[48, 64)` and `[112, 128)`) and the remaining 96 carry wrong-partner rotation values (positions `[0, 48)` and `[64, 112)`).

Count verification: zeroed = 16 + 16 = 32; corrupted = 48 + 48 = 96; total = 128. Internally consistent. Fix confirmed.

**Fix 3 — `is_this_dead_code.md`, unresolved contradiction: APPLIED**

The speculative "may take a different code path" language has been removed. An `[UNVERIFIED]` block is present stating that the tile-aligned path has not been verified against `rope.py` and that the "latent dead code" conclusion is an assertion to be verified, not established fact. Fix confirmed.

---

### Issues found: 1

**Issue 1 — `verification_checklist.md`, line 43: `inv_freq` formula in the PyTorch reference function has the same factor-of-2 error that was fixed in `recommended_fix.md`**

File: `/Users/salnahari/dev/tt-symbiote-research-topics/guides/partial_rotary_non_tile_aligned_numerics/ch6_recommendations/verification_checklist.md`

Quote (lines 31–43 of `apply_rotary_partial_reference`):
```python
    assert rotary_dim % 2 == 0
    half = rotary_dim // 2
    ...
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / rotary_dim))
```

What is wrong: `half = rotary_dim // 2`, so `torch.arange(0, half) / rotary_dim` gives exponents `0/rotary_dim, 1/rotary_dim, ..., (rotary_half-1)/rotary_dim`. The correct exponents for the standard RoPE formula are `2i / rotary_dim`, i.e., `0/rotary_dim, 2/rotary_dim, ..., (rotary_dim-2)/rotary_dim`. The factor of 2 is missing — identically the same error that was fixed in `recommended_fix.md` under Pass 1 Issue 1. The variable name `half` here corresponds to `rotary_half` in `recommended_fix.md`; in both cases the divisor must be `rotary_half` (or equivalently `arange * 2.0 / rotary_dim`), not `rotary_dim`.

This is a technical error with a direct consequence: the "golden reference" function used in Test Cases 1–5 computes the wrong frequencies. If used as-is, PCC tests would compare a (correctly fixed) implementation against a reference that has the same wrong frequencies — the tests would pass even if the fix were not applied, and they might produce misleading PCC values that do not reflect actual correctness against the standard RoPE formula.

Correction: Change line 43 from:
```python
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / rotary_dim))
```
to:
```python
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
```
(`/ half` is equivalent to `* 2.0 / rotary_dim` since `half = rotary_dim // 2`, matching the corrected formula in `recommended_fix.md`.)

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. **`verification_checklist.md`, line 43:** Change `/ rotary_dim` to `/ half` in the `inv_freq` line of `apply_rotary_partial_reference` to match the factor-of-2 correction applied to `recommended_fix.md` in Pass 1.

---

## Pass 3

### Verification of Pass 2 fix

**Fix (`verification_checklist.md` — `inv_freq` corrected to `/ half`):** APPLIED

Current text (lines 43–44):
```python
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    # Note: / half == * 2.0 / rotary_dim — the standard formula 1/base^(2i/rotary_dim).
```

The divisor is now `half` (i.e., `rotary_dim // 2`), which correctly encodes the standard `2i / rotary_dim` exponent. The explanatory comment is present. Fix confirmed.

### Issues found: 0

---

No issues found. Chapters 5 and 6 approved.

---
