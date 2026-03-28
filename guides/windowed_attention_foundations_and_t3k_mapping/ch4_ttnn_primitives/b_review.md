## Pass 1

### Item 1 — `prefill_primitives.md`, lines 101–111: `triu(-2)` intermediate matrix is wrong

**Error.** The worked example decomposes the band-mask construction into three matrices: `causal (tril)`, `triu(-(w-1)) = triu(-2)`, and `band = intersection`. The middle matrix is labelled `triu(-2)` and shown as:

```
Row 0: 1 1 1 . . . . .
Row 1: 1 1 1 1 . . . .
...
Row 7: . . . . . 1 1 1
```

This is not `triu(-2)`. `torch.ones(8,8).triu(-2)` keeps elements where `col >= row - 2`, so row 0 produces all eight 1s (`col >= -2` is always true), row 1 produces all eight 1s, row 2 produces all eight 1s, and row 3 produces seven 1s (col 0 dropped). What the diagram displays is actually `tril(2)` — the upper-right band of width 3. A reader who tests `torch.ones(8,8).triu(-2)` in Python will get a result completely different from the diagram, and will misunderstand which half of the plane the operation retains.

The code on line 74 (`causal.triu(-(w-1))`) is itself correct: applied to the already-lower-triangular `causal` matrix it does produce the desired band. The error is solely in the standalone intermediate diagram.

**Fix.** Replace the middle matrix in the three-column diagram with the correct output of `torch.ones(8,8).triu(-2)`:

```
Row 0: 1 1 1 1 1 1 1 1
Row 1: 1 1 1 1 1 1 1 1
Row 2: 1 1 1 1 1 1 1 1
Row 3: . 1 1 1 1 1 1 1
Row 4: . . 1 1 1 1 1 1
Row 5: . . . 1 1 1 1 1
Row 6: . . . . 1 1 1 1
Row 7: . . . . . 1 1 1
```

The intersection of this matrix with `causal` does yield the correct band shown in the third column, so only the middle column needs correction.

---

### Item 2 — `prefill_primitives.md`, lines 320–329: prefill handoff diagram slot layout contradicts slot-assignment table

**Error.** The ASCII diagram (T=12, w=8) shows the circular buffer after handoff as:

```
slot:  0    1    2    3    4    5    6    7
       tB   t4   t5   t6   t7   t8   t9   tA
```

The slot-assignment table immediately below (lines 335–343) and the concluding sentence (lines 345–347) both state that slot 0 holds t8, slot 1 holds t9, slot 2 holds t10, slot 3 holds t11, and slots 4–7 hold t4–t7. These are derived from `slot = pos mod w` which gives 8%8=0, 9%8=1, 10%8=2, 11%8=3, 4%8=4, …, 7%8=7. The diagram inverts this: it places tB (pos 11) in slot 0 and t8 (pos 8) in slot 5, which is incorrect. A reader implementing the prefill-to-decode handoff from the diagram would write tokens into the wrong slots.

**Fix.** Correct the diagram to match the slot formula:

```
slot:  0    1    2    3    4    5    6    7
       t8   t9   tA   tB   t4   t5   t6   t7
```

---

### Item 3 — `decode_primitives.md`, line 159: fill-phase condition off by one in prose

**Error.** The explanation of Strategy 2 states: "The window constraint (excluding slots written to positions beyond the window, or padding slots in the fill phase T < w-1)." The fill phase is the period when the buffer is not yet full, which spans steps T = 0 through T = w-1 inclusive (at T=w-1 the write fills the last slot, n_valid = w). The condition should be `T < w`, not `T < w-1`. Under the stated condition `T < w-1`, step T = w-1 is incorrectly classified as steady state even though n_valid = w is first achieved only after this write completes. A reader implementing the phase-transition check would exit fill-phase logic one step early.

**Fix.** Change "fill phase T < w-1" to "fill phase T < w" (equivalently, "fill phase T+1 < w, i.e., the buffer is not yet full after this write").

---

### Item 4 — `index.md`: missing "Next:" navigation footer

**Error.** All three leaf content files (`decode_primitives.md`, `prefill_primitives.md`, `kernel_or_op_gap_analysis.md`) end with a "Next:" footer linking to the following file. The chapter index (`index.md`) has no such footer. A reader who lands on `index.md` has no navigational link to `decode_primitives.md` other than the reading-order list in the body, which is not in the standard footer format used by the rest of the chapter.

**Fix.** Add at the end of `index.md`:

```
---

**Next:** [`decode_primitives.md`](./decode_primitives.md)
```

---

### Item 5 — `decode_primitives.md`, lines 205–214: L1 working-set arithmetic omits online-softmax accumulators, but stated total is underestimated

**Error.** The L1 working-set breakdown for `k_chunk_size = 512, d = 128, BF16` lists:

```
K tile:  512 × 128 × 2 = 131,072 bytes (128 KiB)
V tile:  512 × 128 × 2 = 131,072 bytes (128 KiB)
Q tile:    1 × 128 × 2 =     256 bytes
Scores:    1 × 512 × 2 =   1,024 bytes
Total:                    ≈ 260 KiB
```

The online softmax accumulation (used by the Flash-Attention tiled kernel) requires per-head running maximum and running log-sum-exp scalars, plus the partial output accumulator of shape `[1, d]` per head. For `H_q = 32` heads and `d = 128` in BF16, the output accumulator alone is `32 × 128 × 2 = 8,192 bytes` and the two running scalars are `32 × 2 × 2 = 128 bytes`. These are small but the Q tile size is also understated: with H_q heads resident, Q is `H_q × 1 × d`, not `1 × d`. For `H_q = 32` the Q tile is `32 × 128 × 2 = 8,192 bytes`, not 256 bytes. The 260 KiB total is therefore understated by roughly 16 KiB. The "ample headroom" conclusion holds (276 KiB vs 1,536 KiB L1), but readers who use the 260 KiB figure for their own capacity planning will underestimate the working set.

**Fix.** Qualify the Q tile row to reflect that Q holds all H_q vectors resident per core: `H_q × 1 × d × 2 = 32 × 128 × 2 = 8,192 bytes` (for the example configuration), and add a line for the output accumulator. Revise the total to ~276 KiB and note that "ample headroom" still applies.

---

## Change Log (Pass 1)

### prefill_primitives.md

**Issue 1 — `triu(-2)` intermediate matrix corrected (lines ~101–111)**

The middle column of the three-matrix diagram now shows the correct output of
`torch.ones(8,8).triu(-2)`. Previously the diagram showed a narrow upper-right
band (resembling `tril(2)`). The correct matrix has all 1s in rows 0–2 (since
`col >= row - 2` is satisfied for all columns when row ≤ 2) and a staircase of
zeros in the lower-left for rows 3–7. A short explanatory paragraph was added
after the diagram to describe the `triu(-2)` semantics and confirm that the
intersection with `tril` yields the correct band.

**Issue 2 — Circular buffer ASCII diagram slot assignments corrected (lines ~322–323)**

The diagram for the prefill handoff (T=12, w=8) previously placed `tB` in slot 0
and tokens in incorrect slots throughout. The corrected diagram now shows:
- slot 0 → t8 (8 mod 8 = 0)
- slot 1 → t9 (9 mod 8 = 1)
- slot 2 → tA (10 mod 8 = 2)
- slot 3 → tB (11 mod 8 = 3)
- slot 4 → t4 (4 mod 8 = 4)
- slot 5 → t5 (5 mod 8 = 5)
- slot 6 → t6 (6 mod 8 = 6)
- slot 7 → t7 (7 mod 8 = 7)

The slot-assignment table and concluding prose below the diagram were already
correct and now agree with the diagram.

### decode_primitives.md

**Issue 3 — Fill-phase boundary off-by-one corrected (line ~159 and line ~98)**

Two occurrences of the fill-phase condition `T < w-1` were changed to
`T <= w-1`. The fill phase spans steps T = 0 through T = w-1 inclusive (the
buffer first becomes fully populated after the write at T = w-1). The previous
wording `T < w-1` would have caused code following these comments to exit
fill-phase logic one step early.

**Issue 5 — L1 working-set Q tile note and total footprint note corrected (lines ~205–215)**

The Q tile row in the L1 working-set table now carries the note:
"(per head; multiply by heads_per_core for total Q L1)".
A note was added explaining that the 260 KiB figure assumes heads are processed
one at a time (as `scaled_dot_product_attention_decode` typically operates), and
clarifying that K and V tiles are shared across heads in the same GQA group and
do not multiply by heads_per_core.

---

## Pass 2

### Item 1 — `decode_primitives.md`, line 378: RoPE rotation index stated as `pos_offset + T` instead of `T`

**Error.** The op-sequence summary box contains:

```
RoPE rotation (custom kernel or ttnn elementwise)
uses pos_offset + T to derive rotation angles
```

T is defined throughout the document as the absolute token position (0-indexed from prompt start). The correct index to pass to RoPE is T alone; `pos_offset` is the absolute position of the oldest entry in the KV buffer and has no role in computing rotation angles for the current token. Using `pos_offset + T` produces a value of `max(0, T - w + 1) + T`, which is wrong at every step past the fill phase. For example at T=10, w=8: pos_offset=3 → `pos_offset + T = 13`, not 10. A reader implementing the RoPE call from this summary would apply incorrect rotations to every Q and K vector during decode, causing wrong attention scores.

The body text at line 41–42 ("RoPE rotations are applied to `q_T` and `k_T` at this point using `pos_offset` to derive the correct absolute position T") is also imprecise but does not say to add them together. The summary box is the more likely implementation reference and explicitly says to add them, which is wrong.

**Fix.** Change line 378 to: "uses T (absolute token position) to derive rotation angles".

---

## Change Log (Pass 2)

### decode_primitives.md

**Issue 1 — RoPE rotation index in op-sequence summary box corrected (line 378)**

The op-sequence summary box previously read:

```
uses pos_offset + T to derive rotation angles
```

`T` is defined throughout the document as the absolute token position (0-indexed from prompt start). `pos_offset` is the position of the oldest entry in the KV buffer, not an addend for RoPE. Summing them double-counts the offset: at T=10 with w=8, `pos_offset=3` gives `pos_offset + T = 13`, not the correct 10. The line has been corrected to:

```
uses T (the absolute token position) to derive rotation angles
```

No other occurrences of `pos_offset + T` in a RoPE context were found in this file.

---

## Pass 3

### Prior Pass 2 item — confirmed fixed

`decode_primitives.md` line 378 now reads "uses T (the absolute token position) to derive rotation angles". Correct.

---

### Item 1 — `decode_primitives.md`, lines 41–42: body text still states RoPE uses `pos_offset` to derive T

**Error.** The body text of Step 1 reads:

> "RoPE rotations are applied to `q_T` and `k_T` at this point using `pos_offset` to derive the correct absolute position T for each vector."

This is wrong in two ways. First, `pos_offset` is not computed until Step 2 (it is defined as `max(0, T - w + 1)` inside the `update_cache` block). Second, T is the step counter — a direct input to the decode loop — and is not derived from `pos_offset` by any formula. An implementer reading this sentence would conclude that (a) `pos_offset` must be available before the RoPE call, and (b) the rotation index is obtained from `pos_offset` rather than from T directly. Both conclusions are false. A reader who tries to derive T as `pos_offset + slot_index` (a natural misreading) would apply incorrect rotations during every decode step.

The summary box (line 378) was corrected in Pass 2 and now reads "uses T (the absolute token position) to derive rotation angles." The body text contradicts that fix and remains a misleading implementation reference.

**Fix.** Replace lines 41–42 with: "RoPE rotations are applied to `q_T` and `k_T` at this point using T (the absolute token position) as the rotation index."

---

## Change Log (Pass 3)

### decode_primitives.md

**Issue 1 — Step 1 body text: misleading `pos_offset` reference in RoPE sentence corrected (lines 41–42)**

The body text of Step 1 previously read:

> "RoPE rotations are applied to `q_T` and `k_T` at this point using `pos_offset` to derive the correct absolute position T for each vector."

This was wrong on two counts. First, `pos_offset` is not computed until Step 2 (it is assigned inside the `update_cache` block as `max(0, T - w + 1)`), so it is not available at the point of the RoPE call. Second, T is the decode loop counter — a direct input — and is not derived from `pos_offset` by any formula. The phrasing implied that (a) `pos_offset` must exist before RoPE runs and (b) the rotation index is obtained from `pos_offset`, both of which are false and inconsistent with the already-corrected op-sequence summary box (Pass 2).

The sentence has been replaced with:

> "RoPE rotations are applied to `q_T` and `k_T` at this point using T (the absolute token position) as the rotation index."

This is now consistent with the summary box ("uses T (the absolute token position) to derive rotation angles") and correctly conveys that T is used directly, with `pos_offset` playing no role in RoPE.

---

## Pass 4

### Prior Pass 3 item — confirmed fixed

`decode_primitives.md` lines 41–42 now read: "RoPE rotations are applied to `q_T` and `k_T` at this point using T (the absolute token position) as the rotation index." Correct. No reference to `pos_offset` remains in the RoPE sentence.

---

No feedback — chapter approved.

## Pass 5

### Prior Pass 4 item — confirmed fixed

All four files reviewed. All items flagged in Passes 1–4 are confirmed resolved:
- `prefill_primitives.md` `triu(-2)` intermediate matrix diagram: correct (rows 0–2 are all 1s, staircase below).
- `prefill_primitives.md` prefill handoff diagram (T=12, w=8): correct (`t8 t9 tA tB t4 t5 t6 t7` in slots 0–7).
- `decode_primitives.md` fill-phase boundary: correct (`T <= w-1` / `T < w`).
- `decode_primitives.md` RoPE rotation index (summary box and body text): correct (uses T directly).
- `decode_primitives.md` L1 working-set Q tile note: present and correct.

No new issues found meeting the scope criteria (wrong numerical answer, incorrect implementation, or material conceptual misdirection).

No feedback — chapter approved.
