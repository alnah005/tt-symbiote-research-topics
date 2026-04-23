# B Review — Pass 1

1. **[`index.md`, ~line 7, Key Finding overstates the corruption scope]**

   The Key Finding states: "zeros at positions 48–127 of cos/sin corrupt elements 48–127 of the output — **exactly the 80-element passthrough region**."

   This is factually wrong. The trace in `step_by_step_failure_trace.md` (section 3d) itself shows that elements `[0, 48)` — the full 48-element rotation region — are also corrupted: the kernel pairs `input[i]` with `input[i+64]` instead of `input[i+24]`, producing wrong rotations. All 128 output elements are corrupted, not just the 80-element passthrough region. The Key Finding should say "zeros at positions 48–127 of cos/sin, combined with the kernel's fixed pairing offset of 64, corrupt **all 128 output elements**: the 48-element rotation region receives wrong-paired rotations, and the 80-element passthrough region is zeroed or receives incorrect linear combinations."

---

2. **[`step_by_step_failure_trace.md`, ~line 125, `output[24]` correct formula has wrong subscripts]**

   In the element-level trace table, the "Correct partial RoPE output" column for `output[24]` is given as:

   `input[24]*c_24 + input[0]*s_0`

   This is wrong on two counts. Per the rotate-half formula for the right half (`j = 0`, `j+24 = 24`):

   `output[24] = x[0]*s_0 + x[24]*c_0`

   The subscript on `c` should be `c_0`, not `c_24` (the cosine belongs to the pair index `j=0`, not position 24). The element order in the expression is also reversed relative to the canonical formula, but more critically `c_24` is the wrong frequency. This matches the value in `correct_partial_rope_reference.md` line 49 (`x[0]*s_0 + x[24]*c_0`) — the trace table contradicts the reference file.

   Fix: change that cell to `input[0]*s_0 + input[24]*c_0`.

---

3. **[`step_by_step_failure_trace.md`, ~line 125, `output[24]` footnote label is wrong]**

   The footnote (`*`) appended to `output[24]`'s "Correct partial RoPE output" cell says:

   > "the correct formula for `output[24]` in the rotate-half convention is `input[24]*c_24 + input[0]*(-s_0)`"

   This repeats the wrong subscript (`c_24`) and additionally applies a negation sign to `s_0` that is incorrect. For the right-half of rotate-half, the formula is `x[j]*s_j + x[j+24]*c_j` (no negation on `s`). The negation applies only to the left-half pairing term. The correct footnote formula is `input[0]*s_0 + input[24]*c_0`.

---

4. **[`correct_partial_rope_reference.md`, ~line 139, Strategy C analysis misidentifies position 24 as passthrough]**

   Section 5c states: "the kernel computes `output[24] = x[24]*1 + x[88]*0 = x[24]`, which happens to be correct for the **passthrough case**."

   Position 24 is in the rotated region `[0, 48)`, not in the passthrough region `[48, 128)`. Calling this result "correct for the passthrough case" is factually wrong — position 24 should receive a rotated value `x[0]*s_0 + x[24]*c_0`, not the passthrough value `x[24]`. The Strategy C analysis here incorrectly frames the accidental `x[24]` output as "correct"; it is in fact still wrong for position 24. The text should acknowledge that Strategy C with `cos[24:64]=1, sin[24:64]=0` does not produce correct rotation for positions `[24, 48)` of the output — those positions would still be `x[24]` through `x[47]` (passthrough-like) rather than the correct rotated values. The claim that Strategy C "satisfies all these constraints simultaneously" needs to be reconciled with this.

---

5. **[`step_by_step_failure_trace.md`, ~line 93–95, cos/sin index range in Path B description is inconsistent]**

   Section 3a writes:

   ```
   cos[48:63]  = 0.0  (from Step 2 padding)
   cos[64:127] = 0.0  (from the hypothetical second padding)
   ```

   The range `cos[64:127]` uses an inclusive upper bound of 127, but Python slice notation is exclusive — the final element would be index 127, making the range `cos[64:128]`. The range `cos[48:63]` similarly excludes index 63, but Step 2 pads positions 48 through 63 inclusive (16 elements). Both ranges should be written `cos[48:64]` and `cos[64:128]` respectively to match Python conventions and to be consistent with the rest of the file (which correctly uses `cos[48:128]` in section 5b of `correct_partial_rope_reference.md`). As written, these ranges imply 15 zeros (48–62) and 63 zeros (64–126) rather than 16 and 64, which would lead a reader to mis-implement the padding.

---

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1. `index.md` ~line 7: corrected Key Finding — all 128 elements corrupted (not just 80-element passthrough region); rotation region [0, 48) also corrupted by wrong pairing offset
2. `step_by_step_failure_trace.md` ~line 125: corrected output[24] "Correct partial RoPE" cell from `input[24]*c_24 + input[0]*s_0` to `input[0]*s_0 + input[24]*c_0`
3. `step_by_step_failure_trace.md` ~line 130: corrected footnote — removed wrong subscript c_24 and spurious negation on s_0; correct formula is input[0]*s_0 + input[24]*c_0
4. `correct_partial_rope_reference.md` ~line 139: corrected Strategy C analysis — position 24 is in rotated region (not passthrough); output x[24] is still wrong; "satisfies all constraints simultaneously" claim removed
5. `step_by_step_failure_trace.md` ~lines 92–95: corrected slice ranges cos[48:63]→cos[48:64] and cos[64:127]→cos[64:128] (Python slices are exclusive upper bound)

---

# B Review — Pass 2

1. **[`correct_partial_rope_reference.md`, ~line 127, `output[88]` substitution uses wrong zero for `cos[24]`]**

   Section 5b lists the substituted kernel output for five positions. The entry for `output[88]` reads:

   ```
   output[88] = x[24]*s_24 + x[88]*0   — should be x[88] (passthrough)
   ```

   This is factually wrong. Using the authoritative kernel formula `output[i+64] = x[i]*sin[i] + x[i+64]*cos[i]` with `i=24`: `output[88] = x[24]*sin[24] + x[88]*cos[24] = x[24]*s_24 + x[88]*c_24`. `cos[24]` is a **real cosine value** (position 24 is within `[0, 48)`, the real-values region of cos). It is not zero. Zero-padding only begins at position 48.

   Fix: change `x[88]*0` to `x[88]*c_24`. The full corrected line is:
   ```
   output[88] = x[24]*s_24 + x[88]*c_24   — should be x[88] (passthrough)
   ```
   The status annotation remains correct (it is still wrong output), but the formula must use `c_24`, not 0.

2. **[`step_by_step_failure_trace.md`, ~line 139, `[64, 128)` description omits `[112, 128)` from corruption summary]**

   The bullet for "Positions `[64, 128)`" ends with: "corrupting the passthrough elements `[64,112)`." This range is incomplete. The analysis immediately above it states that for `i in [48,64)`, `s_i=0` and `c_i=0`, so `output[i+64] = 0` — covering output positions `[112,128)`. These are also corrupted passthrough elements (zeroed). The sentence as written implies only `[64,112)` is corrupted, omitting the 16 zeroed elements `[112,128)`. A reader implementing a coverage check would miss those positions.

   Fix: change "corrupting the passthrough elements `[64,112)`" to "corrupting the passthrough elements `[64,112)` (wrong linear combination) and zeroing the passthrough elements `[112,128)` (since `c_i=s_i=0` for `i in [48,64)`)."

3. **[`correct_partial_rope_reference.md`, ~line 141, Key Finding contradicts section 5c by calling Strategy C "correct"]**

   The Key Finding box states: "Correct partial RoPE requires Strategy C (Chapter 4): precomputing a full `head_dim`-wide cos/sin table with identity values (`cos=1.0, sin=0.0`) at the positions that should be passthrough."

   This directly contradicts section 5c, which explicitly demonstrates that Strategy C still produces wrong output for positions `[24, 48)` of the output: the kernel computes `output[24] = x[24]` (passthrough-like) when the correct value is `x[0]*s_0 + x[24]*c_0` (a rotation). Section 5c concludes that Strategy C "reduces corruption to only positions `[24, 48)`" — not that it eliminates corruption. Calling it "Correct partial RoPE" in the Key Finding is a factual contradiction with the body of the same section.

   Fix: revise the Key Finding to accurately reflect that Strategy C only partially mitigates the problem. For example: "Correct partial RoPE cannot be produced by any cos/sin value scheme applied to `ttnn.experimental.rotary_embedding` when `rotary_dim < head_dim`; Strategy C (Chapter 4) is the closest viable approach for trace-compatible execution but still requires a supplementary correction for output positions `[24, 48)`. See Chapter 4 for the full analysis."

4. **[`correct_partial_rope_reference.md`, ~lines 60–83, PyTorch reference uses `cos[j+24]`/`sin[j+24]` for right-half outputs but the element-level formula requires `c_j`/`s_j`]**

   The `rotate_half`-based implementation computes `x_embed[j+24]` as:

   ```python
   x_rot[j+24] * cos[j+24] + x_rot[j] * sin[j+24]   # for j in [0, 24)
   ```

   because `cos` is broadcast element-wise over `x_rot` (shape `[B,H,S,48]`). The element-level formula in Section 2 requires:

   ```
   output[j+24] = x[j]*s_j + x[j+24]*c_j            # uses c_j and s_j, not c_{j+24}, s_{j+24}
   ```

   These are equal only if `cos[j+24] = cos[j]` and `sin[j+24] = sin[j]` for all `j in [0,24)` — i.e., the cos/sin table has duplicate (mirrored) frequency values across its two halves. This is the standard RoPE construction, but the file never states it. A reader who constructs a cos/sin table with non-duplicated values (e.g., 48 distinct frequencies monotonically increasing) would get wrong right-half outputs from this code while the left-half would remain correct. The code and the formula will only agree when the table is explicitly constructed with `cos[j+24] = cos[j]`.

   Fix: add a comment in the `apply_partial_rope_reference` function (or in the surrounding text) stating: "This implementation requires that `cos` and `sin` have duplicated frequency values: `cos[j+24] == cos[j]` and `sin[j+24] == sin[j]` for all `j in [0, 24)`. If the table is constructed with 48 distinct monotonically increasing frequencies, the right-half outputs `[24, 48)` will be wrong."

---

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `correct_partial_rope_reference.md` ~line 127: corrected output[88] formula — cos[24] is a real value (not zero); changed x[88]*0 to x[88]*c_24
2. `step_by_step_failure_trace.md` ~line 139: extended [64,128) corruption description — added [112,128) zeroed range (cos=sin=0 for i in [48,64))
3. `correct_partial_rope_reference.md` ~line 141: revised Key Finding — Strategy C does not produce fully correct output; positions [24,48) still wrong; pointed to Ch4 for full analysis
4. `correct_partial_rope_reference.md` ~lines 75–76: added NOTE comment about RoPE frequency-duplication assumption required by the reference implementation

---

# B Review — Pass 3

1. **[`step_by_step_failure_trace.md`, ~line 128, `output[127]` kernel formula uses non-zero notation for zero-padded coefficients, and status description is wrong]**

   The trace table row for `output[127]` shows:

   ```
   Kernel output (Path B): input[63]*s_63 + input[127]*c_63
   Status: Corrupted: mixes input[63] rotation into passthrough
   ```

   This is wrong on two counts.

   First, the notation header (line 120) defines `c_k = cos[k] (real value)` and `s_k = sin[k] (real value)`. But `cos[63]` and `sin[63]` are both zero: position 63 falls in the zero-padded region `[48, 128)`. Using `s_63` and `c_63` — which the header defines as real (nonzero) values — misrepresents the actual substitution. A reader applying the table's notation literally would compute `output[127]` as a nonzero mix of `input[63]` and `input[127]`, when the correct result is 0.

   Second, the status "mixes `input[63]` rotation into passthrough" is factually wrong. Section 3d (line 139) explicitly states that for `i in [48, 64)`, both `s_i=0` and `c_i=0`, so `output[i+64] = 0` — meaning `output[127]` (i=63) is zeroed, not mixed. The status should say "Zeroed: cos=sin=0 at position 63" to match the treatment of `output[48]` and to be consistent with section 3d.

   The `output[48]` row handles this correctly by writing out the substitution explicitly: `input[48]*cos[48] + input[112]*(-sin[48]) = input[48]*0 + input[112]*0 = 0`. The `output[127]` row should use the same explicit form.

   Fix: replace the `output[127]` row with:
   ```
   | `output[127]` | `input[63]*cos[63] + input[127]*(-sin[63])` = `input[63]*0 + input[127]*0` = `0` | `input[127]` (passthrough) | Zeroed: cos=sin=0 at position 63 |
   ```

---

# B Review — Pass 4 (Change Log)

Changes applied in response to Pass 3:
1. `step_by_step_failure_trace.md` ~line 128: corrected output[127] table row — cos[63]=sin[63]=0 (zero-padded region); output is 0 (zeroed), not a mixed corruption; updated notation to show explicit substitution and status to "Zeroed"

---

# B Review — Pass 4

1. **[`correct_partial_rope_reference.md`, ~line 135, `[64, 128)` corruption description conflates zeroed and linear-combination ranges]**

   The closing sentence of section 5b states: "elements in `[48, 64)` are zeroed, and elements in `[64, 128)` receive incorrect linear combinations of input values from the left half."

   This is factually wrong for positions `[112, 128)`. Using the kernel formula `output[i+64] = x[i]*sin[i] + x[i+64]*cos[i]` for i in [48, 64): both `sin[i]=0` and `cos[i]=0`, so `output[i+64] = 0`. Output positions `[112, 128)` are therefore **zeroed**, not "incorrect linear combinations." Only positions `[64, 112)` (corresponding to i in [0, 48)) receive incorrect linear combinations. The same distinction was correctly made in `step_by_step_failure_trace.md` section 3d (Pass 2 fix), but this sentence in `correct_partial_rope_reference.md` was not updated to match.

   Fix: change "elements in `[64, 128)` receive incorrect linear combinations of input values from the left half" to "elements in `[64, 112)` receive incorrect linear combinations of input values from the left half, and elements in `[112, 128)` are zeroed (since `cos[i]=sin[i]=0` for `i in [48, 64)`)."

---

2. **[`correct_partial_rope_reference.md`, ~line 130, `output[24]` annotation is misleading about the nature of the error]**

   The annotation for `output[24]` in section 5b reads: "uses `x[88]`, should use `x[0]`."

   The kernel computes `output[24] = x[24]*c_24 + x[88]*(-s_24)`. The correct output is `x[0]*s_0 + x[24]*c_0`. The annotation identifies only the spurious `x[88]` element, implying the fix is to replace `x[88]` with `x[0]`. But substituting `x[0]` for `x[88]` in the kernel formula yields `x[24]*c_24 + x[0]*(-s_24)`, which is still wrong: the coefficients `c_24` and `s_24` (at frequency index 24) should be `c_0` and `s_0` (at frequency index 0), and the combination rule is additive (not subtractive) for the `x[0]` term. A reader implementing a patch based solely on this annotation would produce an incorrect result. The annotation undercounts the error by one dimension.

   Fix: expand the annotation to reflect both error dimensions, for example: "uses `x[88]*(-s_24)` where the correct term is `x[0]*s_0`; also uses `x[24]*c_24` where the correct coefficient is `c_0` — the kernel applies the wrong frequency index and wrong sign to both terms."

---

3. **[`step_by_step_failure_trace.md`, ~line 137, `[24, 48)` corruption described as "same wrong-pairing issue" — incorrect for right-half positions]**

   Section 3d's bullet for positions `[24, 48)` states: "Kernel uses real `c_i` and `s_i`, but pairs `input[i]` with `input[i+64]`. Same wrong-pairing issue."

   This description is factually imprecise. For positions `[0, 24)`, "wrong pairing offset" is an accurate summary: the kernel pairs `x[i]` with `x[i+64]` when the correct pairing partner is `x[i+24]`, and the combination rule (`x[i]*c_i + x[i+24]*(-s_i)`) still applies — only the index of the partner is wrong.

   For positions `[24, 48)`, the situation is categorically different. The correct output formula for these positions is `output[j+24] = x[j]*s_j + x[j+24]*c_j` for `j in [0, 24)` — a different set of input elements (`x[j]` and `x[j+24]`), different trig subscripts (`j`, not `i`), and no negation. The kernel instead computes `output[i] = x[i]*c_i + x[i+64]*(-s_i)` for `i in [24, 48)`. This is not a "pairing offset" error: the formula structure itself is wrong — the kernel treats these as first-half rotation outputs (with `c_i` on `x[i]`), when they are right-half outputs requiring entirely different inputs and a different combination rule.

   Fix: replace the `[24, 48)` bullet with a description that distinguishes the structural error, for example: "Kernel computes these as first-half rotate-half outputs using `output[i] = x[i]*c_i + x[i+64]*(-s_i)`. But positions `[24, 48)` are the **right half** of the partial RoPE rotation: the correct output is `output[j+24] = x[j]*s_j + x[j+24]*c_j` for `j = i-24 in [0, 24)`. The error is not merely a wrong pairing offset — the input elements, trig subscripts, and combination rule are all different from what the kernel applies."

---

# B Review — Pass 5 (Change Log)

Changes applied in response to Pass 4:
1. `correct_partial_rope_reference.md` ~line 135: corrected [64,128) corruption description — [112,128) are zeroed (cos=sin=0 for i in [48,64)), not "incorrect linear combinations"; only [64,112) receive incorrect linear combinations
2. `correct_partial_rope_reference.md` ~line 130: expanded output[24] annotation — error is in both the wrong input element (x[88] vs x[0]) AND the wrong frequency index (c_24/s_24 vs c_0/s_0) AND the wrong sign convention; full description: "uses x[88]*(-s_24) where correct term is x[0]*s_0; also uses x[24]*c_24 where correct coefficient is c_0"
3. `step_by_step_failure_trace.md` ~line 137: replaced [24,48) "Same wrong-pairing issue" with structural formula error description — kernel applies first-half formula but [24,48) are right-half positions requiring different inputs, trig subscripts, and combination rule

---

# B Review — Pass 5

1. **[`step_by_step_failure_trace.md`, ~line 125, Status column for `output[24]` still understates the error]**

   The Status cell for `output[24]` in the element-level trace table reads: "Wrong pairing: uses `input[88]` instead of `input[0]`."

   The Pass 4 fix expanded the annotation for `output[24]` in `correct_partial_rope_reference.md` section 5b (~line 130) to cover all three error dimensions (wrong input element, wrong frequency index, wrong sign). That fix was not mirrored to the parallel Status cell in `step_by_step_failure_trace.md`. The current Status cell still implies the only error is a wrong input element (`x[88]` vs `x[0]`). A reader who sees only the trace table will not know that the frequency index (`c_24`/`s_24` vs `c_0`/`s_0`) and the combination sign are also wrong.

   Fix: expand the Status cell to match the level of detail in `correct_partial_rope_reference.md`, for example: "Wrong pairing, wrong frequency, wrong sign: uses `input[88]*(-s_24)` where correct term is `input[0]*s_0`; also uses `input[24]*c_24` where correct coefficient is `c_0`."

---

2. **[`step_by_step_failure_trace.md`, ~line 141, Summary conflates [0,24) and [24,48) errors under "wrong pairing"]**

   The Summary sentence reads: "the 48-element rotation region `[0, 48)` has the wrong pairing."

   After the Pass 4 fix, section 3d now correctly distinguishes two different error types for [0,48): positions `[0, 24)` have a wrong pairing offset (structural pairing is right, offset is wrong), while positions `[24, 48)` have a structural formula error (the combination rule, input elements, and trig subscripts are all wrong — not just a pairing offset). Collapsing both sub-regions into "wrong pairing" in the Summary directly contradicts the more precise description established by the Pass 4 fix for `[24, 48)`.

   Fix: replace "the 48-element rotation region `[0, 48)` has the wrong pairing" with a description that preserves the distinction, for example: "the 48-element rotation region `[0, 48)` is entirely corrupted: positions `[0, 24)` receive wrong-paired rotations (pairing offset 64 instead of 24), and positions `[24, 48)` receive structurally wrong outputs (kernel applies first-half formula to right-half positions, using wrong input elements, wrong frequency indices, and wrong combination rule)."

# B Review — Pass 6 (Change Log)

Changes applied in response to Pass 5:
1. `step_by_step_failure_trace.md` ~line 125: expanded output[24] Status cell to cover all three error dimensions — wrong input element (input[88] vs input[0]), wrong frequency index (c_24/s_24 vs c_0/s_0), and wrong sign (subtractive vs additive for cross-term)
2. `step_by_step_failure_trace.md` ~line 141: revised Summary sentence — replaced "has the wrong pairing" for [0,48) with distinction between [0,24) wrong-paired rotations and [24,48) structurally wrong outputs (kernel applies first-half formula to right-half positions)

---

# B Review — Pass 6

1. **[`correct_partial_rope_reference.md`, ~line 135, closing prose for `[0, 48)` collapses structural error into wrong-pairing description]**

   The closing paragraph of section 5b states: "For positions `[0, 48)`, the pairing is across `head_dim/2=64` instead of `rotary_dim/2=24`."

   This is the same understatement that was corrected in `step_by_step_failure_trace.md` section 3d by Pass 4 finding #3, but that fix was not mirrored to this location. The sentence uniformly attributes the `[0, 48)` corruption to a wrong pairing offset. However, positions `[24, 48)` have a categorically different error: the kernel applies the first-half rotate-half formula (`output[i] = x[i]*c_i + x[i+64]*(-s_i)`) to what are actually right-half positions, where the correct formula is `output[j+24] = x[j]*s_j + x[j+24]*c_j` for `j in [0, 24)`. The error for `[24, 48)` is not a wrong offset — the input elements, trig subscripts, and combination rule are all different. Calling this a wrong-pairing offset misrepresents the nature of the error for the `[24, 48)` sub-range.

   Fix: split the `[0, 48)` sentence into two sub-ranges, matching the precision established in `step_by_step_failure_trace.md` line 141. For example: "For positions `[0, 24)`, the pairing is across `head_dim/2=64` instead of `rotary_dim/2=24` (wrong offset, but the first-half formula structure is at least applied). For positions `[24, 48)`, the error is structural: the kernel applies the first-half formula to right-half positions, using the wrong input elements, wrong frequency indices, and the wrong combination rule (subtractive instead of additive for the cross-term)."

---

2. **[`index.md`, ~line 7, Key Finding describes all of `[0, 48)` as "wrong-paired rotations"]**

   The Key Finding states: "the 48-element rotation region `[0, 48)` receives wrong-paired rotations (offset 64 instead of the required 24)."

   This is the same conflation fixed in `step_by_step_failure_trace.md` line 141 by Pass 5 finding #2, but the Key Finding in `index.md` was not updated. Positions `[24, 48)` do not merely receive wrong-paired rotations — they receive structurally wrong outputs because the kernel applies the first-half formula to right-half positions. The description "offset 64 instead of the required 24" is accurate only for `[0, 24)`. For `[24, 48)`, the offset is the least of the problems: the entire formula structure (input indices, frequency indices, and sign) is wrong.

   Fix: revise the Key Finding to preserve the `[0, 24)` / `[24, 48)` distinction, consistent with `step_by_step_failure_trace.md` line 141. For example: "the 48-element rotation region `[0, 48)` is entirely corrupted: positions `[0, 24)` receive wrong-paired rotations (offset 64 instead of 24), and positions `[24, 48)` receive structurally wrong outputs (kernel applies first-half formula to right-half positions, using wrong input elements, wrong frequency indices, and wrong combination rule)."

---

# B Review — Pass 7 (Change Log)

Changes applied in response to Pass 6:
1. `correct_partial_rope_reference.md` ~line 135: split [0,48) "wrong pairing" description to distinguish [0,24) (wrong pairing offset: head_dim/2=64 instead of rotary_dim/2=24) from [24,48) (structural formula error: kernel applies first-half formula to right-half positions requiring different inputs, trig subscripts, and combination rule)
2. `index.md` ~line 7: revised Key Finding — replaced "[0,48) receives wrong-paired rotations (offset 64)" with two-sub-range description: [0,24) wrong-paired rotations; [24,48) structurally wrong outputs

---

# B Review — Pass 7

No further correctness issues found.
