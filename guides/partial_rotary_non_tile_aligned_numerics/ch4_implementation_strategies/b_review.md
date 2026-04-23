# B Review — Chapter 4: Implementation Strategies

## Pass 1

### Issues found: 3

---

**Issue 1:** `strategy_a_slice_apply_concat.md`, Section 3 "The Padded-Slice Variant", line immediately after the kernel formula display (the bullet beginning "output[i] = x_rot[i] * c_i + x_rot_padded[i + 32] * (-s_i) where...")

**Error:** The zero/real characterization for the two sub-ranges of `i` is reversed.

The padded slice has shape `[..., 64]`: positions `[0, 48)` are real input values; positions `[48, 64)` are zero-padded. With pairing offset 32, the kernel reads `x_rot_padded[i + 32]`:

- For `i in [0, 16)`: `i + 32 in [32, 48)` — these are **real** input values (within the original 48-element slice). Not zero.
- For `i in [16, 32)`: `i + 32 in [48, 64)` — these are **zero-padded** positions. Not real.

The file states the opposite: "x_rot_padded[i+32] is zero for i in [0, 16) (positions 32–47 are real input)" — the parenthetical correctly names the positions as real but the "zero" label contradicts it. And "real values for i in [16, 24) (positions 48–55 are zero-padded)" — again the parenthetical correctly calls them zero-padded but the "real values" label contradicts it.

**Correction:** Replace that bullet with:

> `output[i] = x_rot[i] * c_i + x_rot_padded[i + 32] * (-s_i)` where `x_rot_padded[i+32]` is a **real** input value for `i in [0, 16)` (positions 32–47 are within the original 48-element slice) and **zero** (padded) for `i in [16, 32)` (positions 48–63 are zero-filled). Neither sub-range produces the correct partner `x_rot[i + 24]` required by the `rotary_dim/2=24` pairing; the offset 32 ≠ 24 regardless of whether the partner slot is real or zero.

---

**Issue 2:** `strategy_c_precomputed_full_head_cos_sin.md`, Section 6 "Why Frequency Duplication Is Required", the paragraph at lines 253–254 and the Key Finding block.

**Error:** The file claims that Region 3 cos/sin values (positions `[64, 88)`) "are read by the kernel when it processes the `(j+64)` position as the 'left' of a new pair." This is factually wrong.

The kernel iterates `i` over `[0, head_dim/2) = [0, 64)` only. The formula is:

```
output[i]      = input[i] * cos[i] + input[i+64] * (-sin[i])    for i in [0, 64)
output[i + 64] = input[i] * sin[i] + input[i+64] *   cos[i]     for i in [0, 64)
```

The cos/sin lookup index is always `i`, which ranges only over `[0, 64)`. Positions `[64, 128)` of the cos/sin table are **never read** by the kernel. Setting Region 3 (`[64, 88)`) and Region 4 (`[88, 128)`) to any values has zero effect on the kernel's output.

The Key Finding that states Region 3 duplication "is not redundant" is therefore wrong: it is entirely redundant from the kernel's perspective. The construction is still valid and harmless, but the justification is incorrect.

**Correction:** Replace Section 6 with an explanation that accurately reflects why the duplication is included despite being kernel-irrelevant:

> The kernel reads `cos[i]` and `sin[i]` only for `i in [0, head_dim/2) = [0, 64)`. Positions `[64, 128)` of the cos/sin table are **never accessed** during computation. Setting Region 3 (`[h, h+r) = [64, 88)`) to duplicate Region 1 values, and Region 4 (`[h+r, H) = [88, 128)`) to identity, is therefore redundant for the kernel's output but is included for two reasons: (1) it makes the table self-consistent and documents the intended frequency layout symmetrically; (2) if any future validation or debugging tooling checks cos/sin table structure, the Region 3 values reflect what the correct symmetry should be. The cos/sin table has shape `[max_seq_len, head_dim]` to satisfy the op's shape constraint `cos.shape[-1] == input.shape[-1]`; the upper half of that table is numerically inert.
>
> **Key Finding:** Frequency duplication at Region 3 does not affect the kernel's computation (those entries are never read). It is included for table consistency and documentation, not for correctness. Only Regions 1 and 2 (positions `[0, head_dim/2) = [0, 64)`) affect the output.

---

**Issue 3:** `index.md`, Section "Decision Table" (line ~40), column "Strategy C", row "Numerically correct output".

**Error:** The cell reads "Yes, for any `rotary_dim` where `head_dim % 64 == 0`" with no qualification. Strategy C is only correct under the additional condition that the input head uses the `head_dim/2`-split pairing convention — i.e., `input[j]` and `input[j + head_dim/2]` are the rotation partners for `j in [0, rotary_dim/2)`. If the input uses the standard PyTorch `rotary_dim/2`-split layout (partners at `input[j]` and `input[j + rotary_dim/2]`), Strategy C produces different output from the PyTorch reference at positions `[0, rotary_dim/2)`. This condition is correctly documented in `strategy_c_precomputed_full_head_cos_sin.md` Section 4b, but the index table omits it.

The same omission appears in the index's "Key Finding" box (line 7): "Strategy C ... is the only strategy that is simultaneously correct and trace-compatible" — without stating the input layout precondition.

**Correction:** Update the decision table cell to:

> Yes, if `head_dim % 64 == 0` **and** input head uses `head_dim/2`-split pairing (elements `j` and `j + head_dim/2` are rotation partners); differs from PyTorch `rotary_dim/2`-split reference otherwise.

Update the Key Finding box to add the analogous caveat, e.g.:

> Strategy C — precomputing a full `[max_seq_len, head_dim]` cos/sin table with identity values at passthrough positions and duplicated frequencies at positions `[head_dim/2, head_dim/2 + rotary_dim/2)` — is the only strategy that is simultaneously correct (given the `head_dim/2`-split input layout) and trace-compatible.

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:
1. `strategy_a_slice_apply_concat.md` padded-slice section: corrected zero/real descriptor — i in [0,16) accesses REAL elements at positions [32,48); i in [16,32) accesses ZERO-PADDED elements at positions [48,64)
2. `strategy_c_precomputed_full_head_cos_sin.md` Region 3 section: removed false claim that Region 3 cos/sin values are read by the kernel; kernel only reads positions [0, head_dim/2=64); Region 3 is numerically inert; "frequency duplication" is not required for correctness
3. `index.md` decision table: added pairing convention caveat to Strategy C correctness cell

---

## Pass 2

### Issues found: 3

**Issue 1:** `index.md`, Key Finding box (line 7)

**Error:** Pass 1 Issue 3 identified two locations in `index.md` that omit the pairing-convention caveat for Strategy C: the decision table cell and the Key Finding box. The decision table cell was corrected (Pass 1 Change Log item 3), but the Key Finding box was not updated. It still reads:

> "Strategy C — precomputing a full `[max_seq_len, head_dim]` cos/sin table with identity values at passthrough positions and duplicated frequencies at positions `[head_dim/2, head_dim/2 + rotary_dim/2)` — is the only strategy that is simultaneously correct and trace-compatible."

This omits the precondition that Strategy C is only correct when the input head uses the `head_dim/2`-split pairing convention (elements `j` and `j + head_dim/2` are rotation partners). Without this caveat, the Key Finding box contradicts the decision table cell (which was fixed) and Section 4b of `strategy_c_precomputed_full_head_cos_sin.md`.

**Correction:** Update the Key Finding box to:

> Strategy C — precomputing a full `[max_seq_len, head_dim]` cos/sin table with identity values at passthrough positions and duplicated frequencies at positions `[head_dim/2, head_dim/2 + rotary_dim/2)` — is the only strategy that is simultaneously correct (given the input head uses the `head_dim/2`-split pairing convention, where elements `j` and `j + head_dim/2` are rotation partners for `j in [0, rotary_dim/2)`) and trace-compatible. Strategies A and B address narrower concerns: Strategy A is correct for tile-aligned `rotary_dim` but requires runtime buffer allocation; Strategy B is a fail-fast guard that converts silent corruption into an explicit error.

---

**Issue 2:** `strategy_c_precomputed_full_head_cos_sin.md`, Section 4d "Verifying positions `j in [64, 88)` — rotated second half"

**Error:** Section 4d states: "The construction gives `cos[j] = c_k` and `sin[j] = s_k` (Region 3). The kernel addresses this position as the 'right half' of pair `(k, k+64)` — already covered in 4a."

This directly contradicts Section 6 (the Pass 1 fix), which correctly states: "The kernel reads `cos[i]` and `sin[i]` only for `i in [0, head_dim/2) = [0, 64)`. Positions `[64, 128)` of the cos/sin table are **never accessed** during computation."

The phrase "The kernel addresses this position" implies the kernel reads cos[j] for j in [64, 88). It does not — the kernel's loop variable `i` ranges over `[0, 64)` only; cos/sin at positions [64, 128) are never read. The kernel does use `input[j]` for j in [64, 88) as input data (as the "right half" of a pair), but it does NOT read cos or sin at those indices. Section 4d conflates reading cos/sin at index j with using input data at index j.

The section title "rotated second half" is also misleading: the output at positions [64, 88) is produced by the kernel using `cos[k]` and `sin[k]` for `k in [0, 24)` (Region 1), not by reading cos/sin from [64, 88).

**Correction:** Replace Section 4d with:

> ### 4d. Output at positions `j in [64, 88)` — rotated second half
>
> For `j in [64, 88)`, let `j = 64 + k` where `k in [0, 24)`. The output at position `j` is produced by the kernel as part of the pair `(k, k+64)` already analyzed in 4a:
>
> ```
> output[k + 64] = input[k] * sin[k] + input[k + 64] * cos[k]
>                = input[k] * s_k    + input[k + 64] * c_k
> ```
>
> The kernel reads `cos[k]` and `sin[k]` at positions `k in [0, 24)` (Region 1) — not at positions `[64, 88)`. The values placed in Region 3 (`cos[j]` and `sin[j]` for `j in [64, 88)`) are never read by the kernel. No additional equations arise from Region 3; the outputs at positions [64, 88) are fully determined by Region 1 values via the pair formula in 4a.

---

**Issue 3:** `strategy_c_precomputed_full_head_cos_sin.md`, Section 4a, Note box (line 58)

**Error:** The note states: "What Strategy C achieves is something more subtle: it constructs cos/sin such that the kernel's computation is correct for the 'duplicated-input' interpretation of partial RoPE, where the same rotation is applied to both halves of the `head_dim`-split, **and the result happens to be correct for the positions that matter**."

The phrase "the result happens to be correct for the positions that matter" is contradicted by the Key Finding immediately following in Section 4a (lines 122–123), which explicitly states: "Strategy C produces partial rotation of the `(input[j], input[j+64])` pairs, which is **different** from the reference implementation that rotates `(input[j], input[j+24])` pairs."

The note implies unconditional correctness at certain positions; the Key Finding clarifies that Strategy C is only correct under the `head_dim/2`-split input layout assumption (Section 4b). Without that layout assumption, the output at positions `[0, 24)` and `[64, 88)` is wrong relative to the PyTorch reference. The note should not claim correctness unconditionally.

**Correction:** Replace the final clause of the note with the convention qualifier:

> What Strategy C achieves is something more subtle: it constructs cos/sin such that the kernel's computation correctly rotates the pairs `(input[j], input[j+64])` for `j in [0, 24)` using the real frequencies — which matches the correct partial RoPE output only when the input head is laid out with `head_dim/2`-split pairing (elements `j` and `j+64` are rotation partners). If the input uses `rotary_dim/2`-split pairing (elements `j` and `j+24` are partners), Strategy C produces a different output from the PyTorch reference.

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. `index.md` Key Finding box: added pairing-convention caveat (head_dim/2-split required; PyTorch slice convention produces different output)
2. `strategy_c_precomputed_full_head_cos_sin.md` Section 4d: removed false claim that kernel reads cos/sin at positions [64,88); clarified kernel reads INPUT DATA at [64,88) as right-half partners, but uses cos/sin from indices [0,24) for those rotations
3. `strategy_c_precomputed_full_head_cos_sin.md` Section 4a Note: added head_dim/2-split qualifier to unqualified correctness claim

---

## Pass 3

### Issues found: 1

---

**Issue 1:** `strategy_c_precomputed_full_head_cos_sin.md`, Section 5 "Python Construction Code", code comment inside `build_strategy_c_cos_sin` at the Region 3 block (lines 229–234).

**Error:** The comment explaining why Region 3 values are set reads:

```python
# Region 3: positions [h, h+r) — duplicate Region 1 (frequency duplication)
# why: the kernel pairs element i with element i+h; for the pair (j, j+h)
# where j in [0, r) to be correctly rotated at frequency j, we need
# cos[j] == cos[j+h] and sin[j] == sin[j+h]
```

The phrase "we need cos[j] == cos[j+h] and sin[j] == sin[j+h]" implies that matching the Region 3 values to Region 1 is a correctness requirement — i.e., that the kernel reads `cos[j+h]` and `sin[j+h]` for `j in [0, r)`. This directly contradicts Section 6 of the same file, which was corrected in Pass 1 to state: "The kernel reads `cos[i]` and `sin[i]` only for `i in [0, head_dim/2) = [0, 64)`. Positions `[64, 128)` of the cos/sin table are **never accessed** during computation." Section 4d (corrected in Pass 2) also confirms: "The values placed in Region 3 (`cos[j]` and `sin[j]` for `j in [64, 88)`) are never read by the kernel."

The comment in the code contradicts the corrected prose in Sections 4d and 6 and falsely asserts that Region 3 duplication is required for the kernel to produce correct output. The kernel's loop variable `i` ranges over `[0, h) = [0, 64)` only; it never indexes into `[h, H) = [64, 128)` of the cos/sin table.

**Correction:** Replace the Region 3 comment block with one that accurately reflects why duplication is included despite being kernel-inert:

```python
# Region 3: positions [h, h+r) — duplicate Region 1 (frequency duplication)
# Note: the kernel reads cos/sin only at indices [0, h) = [0, 64);
# these Region 3 values are NEVER read during computation (see Section 6).
# They are set here only for table self-consistency and to document the
# intended symmetric frequency layout; they have zero effect on output.
cos_table[:, h : h + r] = cos_real
sin_table[:, h : h + r] = sin_real
```

---

## Pass 3 Change Log

Changes applied in response to Pass 3 issues:
1. `strategy_c_precomputed_full_head_cos_sin.md` Section 5 Region 3 code comment: replaced "we need cos[j] == cos[j+h]" correctness-claim comment with note that Region 3 values are never read by the kernel; duplication is for table consistency and documentation only, not a correctness requirement — references Section 6

---

## Pass 4

### Issues found: 1

---

**Issue 1:** `strategy_c_precomputed_full_head_cos_sin.md`, Section 4a "Verifying positions `j in [0, 24)` — the rotated first half", line immediately after the two kernel equations (the sentence beginning "With the construction above, `cos[j+64] = c_{j+64-64} = c_j` and `sin[j+64] = s_j` (Region 3). So:").

**Error:** The derivation step invokes Region 3 values (`cos[j+64]` and `sin[j+64]`) as the logical basis for the conclusion `output[j+64] = input[j]*s_j + input[j+64]*c_j`. The phrasing "With the construction above, `cos[j+64] = c_j` and `sin[j+64] = s_j` (Region 3). So: `output[j+64] = ...`" implies that the kernel reads `cos[j+64]` and `sin[j+64]` to produce `output[j+64]`.

This directly contradicts the invariant established and maintained through Passes 1–3: the kernel reads `cos[i]` and `sin[i]` only for `i in [0, head_dim/2) = [0, 64)`. Region 3 indices `[64, 88)` are never read. `output[j+64]` is produced by the kernel formula `output[i+64] = input[i]*sin[i] + input[i+64]*cos[i]` evaluated at `i = j`, using `cos[j]` and `sin[j]` from Region 1 — not from Region 3. The conclusion (the `output[j+64]` formula) is numerically correct, but the stated reason is wrong: `s_j` and `c_j` appear in the formula because `sin[j] = s_j` and `cos[j] = c_j` (Region 1), not because `sin[j+64] = s_j` and `cos[j+64] = c_j` (Region 3).

This is the same class of error that was corrected in Section 4d (Pass 2) and Section 6 (Pass 1): conflating reading cos/sin at index `j+64` with using input data at index `j+64`.

**Correction:** Replace the transitional sentence and the equation block that follows with a derivation that correctly cites Region 1 as the source of `c_j` and `s_j` in `output[j+64]`:

> The kernel computes `output[j+64]` using `cos[j]` and `sin[j]` at index `j` (Region 1), not at index `j+64` (Region 3). With `cos[j] = c_j` and `sin[j] = s_j` (Region 1), the kernel's formula `output[i+64] = input[i]*sin[i] + input[i+64]*cos[i]` at `i = j` gives:
>
> ```
> output[j + 64] = input[j] * s_j + input[j + 64] * c_j
> ```

The remainder of Section 4a (the Key Finding box) is correct and does not need to change.

---

## Pass 4 Change Log

Changes applied in response to Pass 4 issues:
1. `strategy_c_precomputed_full_head_cos_sin.md` Section 4a: replaced transitional sentence that cited Region 3 (`cos[j+64]`, `sin[j+64]`) as the source of c_j/s_j; replaced with sentence explicitly citing Region 1 (`cos[j] = c_j`, `sin[j] = s_j`) as the actual source; Region 3 is never read by the kernel

---

## Pass 5

### Issues found: 0 — APPROVED

**Pass 4 fix verified:** `strategy_c_precomputed_full_head_cos_sin.md` Section 4a now reads (lines 113–114):

> "As shown in the formula above, the kernel uses `cos[j] = c_j` and `sin[j] = s_j` (Region 1, indices `j in [0, 24)`) to compute `output[j+64]` — the same cos/sin indices used for `output[j]`. Region 3 values (`cos[j+64]`, `sin[j+64]`) are never read by the kernel when computing `output[j+64]`; see Section 4d and Section 6 for the derivation."

This correctly attributes `output[j+64]` to Region 1 and explicitly states Region 3 is never read. The fix is accurate and complete.

**All six core invariants verified across all five files:**

1. Region 3 `[64, 88)` never read — confirmed in Section 4a (post-fix), Section 4d, Section 5 code comment, and Section 6 of `strategy_c_precomputed_full_head_cos_sin.md`.
2. `output[j+64]` for `j in [0, 24)` is produced using `cos[j]` and `sin[j]` (Region 1), not `cos[j+64]`/`sin[j+64]` (Region 3) — confirmed in Section 4a and Section 4d; consistent throughout.
3. Strategy C correctness qualifier (head_dim/2-split input convention) — present in Section 4a note, Section 4b, `index.md` Key Finding box, `index.md` Decision Table, `trace_safe_alternatives_to_ttnn_pad.md` Section 5 table and Section 6.
4. Section 4d: kernel reads INPUT DATA at `[64, 88)` as right-half partners but reads cos/sin only from Region 1 indices `[0, 24)` — correctly stated.
5. Section 5 Region 3 code comment: states values are never read by kernel; duplication is for table consistency only — correct.
6. Section 6 Key Finding: Region 3 is numerically inert; only Regions 1 and 2 affect output — correct.

No issues remain. All files are internally consistent and consistent with each other.
