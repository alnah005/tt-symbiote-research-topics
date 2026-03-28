## Pass 1

### Issue 1 — Inverted / wrong padding range in Strategy A diagram

**File:** `paged_sdpa_and_windowing.md`, approximately line 139

**Error:** The mask annotation in the page-aware windowing example reads:

```
Mask: zero positions 73–200 as valid, -inf for 64–72 (partial first block)
      and -inf for 201–191 (padding at end)
```

The range "201–191" is inverted and therefore nonsensical. The assembled tensor
covers virtual pages 1–3 (3 blocks × 64 tokens = 192 slots), representing token
positions 64–255. Padding falls at positions 201–255 (54 slots), not "201–191".

**Fix:** Change "and -inf for 201–191 (padding at end)" to
"and -inf for 201–255 (padding at end)".

---

### Issue 2 — Strategy A steady-state live block count stated as a fixed "+1" when it is variable

**File:** `eviction_and_page_reuse.md`, approximately line 252

**Error:** The text states that after steady state under Strategy A "the number of
live blocks stabilises at `ceil(w / block_size) + 1` (the extra +1 accounts for
the partially-filled current block)." This is presented as a definitive value,
but it is not fixed. The live count is:

```
floor(T / block_size) - floor((T - w + 1) / block_size) + 1
```

This equals `ceil(w / block_size)` when `(T - w + 1)` and `T` are within the
same block-aligned band, and `ceil(w / block_size) + 1` only when the window
boundary straddles a block boundary. For example, with w=128, block_size=64:

- T=255: p_high=3, p_low=floor(128/64)=2, count = 2 = ceil(128/64) — no "+1"
- T=256: p_high=4, p_low=floor(129/64)=2, count = 3 = ceil(128/64)+1

Stating the count as always `ceil(w/block_size) + 1` will cause a developer
sizing the block pool under Strategy A to over-allocate in the best case and
misunderstand the eviction cadence in the general case.

**Fix:** Replace the fixed claim with the correct characterisation: "the number
of live blocks is `floor(T/block_size) - floor((T-w+1)/block_size) + 1`, which
is `ceil(w/block_size)` when the window boundary is block-aligned and at most
`ceil(w/block_size) + 1` otherwise."

---

### Issue 3 — Broken "Next" link at end of `eviction_and_page_reuse.md`

**File:** `eviction_and_page_reuse.md`, line 332

**Error:** The footer reads:

```
**Next:** [Chapter 6 — T3K Mesh Sharding and CCL Implications](../ch6_t3k_sharding/index.md)
```

The directory `ch6_t3k_sharding/` does not exist. A reader following this link
reaches a 404. Since Ch6 is a dependency that Ch5 explicitly declares
("Chapter 6 uses the tensor shapes established in Chapter 4" — index.md line 68),
a reader who needs to continue the learning path has no working forward reference.

**Fix:** Either create the `ch6_t3k_sharding/` directory with at least a
placeholder `index.md`, or change the footer to a plain-text note such as
"Chapter 6 (T3K Mesh Sharding) — not yet published" until the chapter exists.

---

### Issue 4 — `seq_len` argument semantics in Option 1 host code vs. kernel expectation

**File:** `paged_sdpa_and_windowing.md`, approximately lines 408–419

**Error:** The `reorder_page_table_for_windowing` function sets
`T = seq_lens[b] - 1` (treating `seq_lens[b]` as a token count, so `T` is the
0-indexed position of the most recent token). The caller is then instructed to
pass `seq_len = min(T+1, w)` to `paged_sdpa_decode` (line 419). However, the
standard `paged_sdpa_decode` interface uses `seq_len` as the number of valid
token positions (a count, not a 0-indexed position). Passing `min(T+1, w)` is
correct for the non-windowed fill phase but is wrong once `T >= w`: at steady
state the window holds exactly `w` tokens (positions `T-w+1` through `T`), so
`seq_len` should be `w` — which `min(T+1, w)` does give. The formula is
accidentally correct but the explanation ("set to `min(T+1, w)` so the kernel's
mask logic covers the correct number of valid positions") omits the key
distinction that for `T >= w` the kernel must also be told where the window
starts, not just how many tokens are valid. Without a `start_position` or
equivalent, the kernel will apply its causal mask starting at position 0 instead
of `T-w+1`, producing wrong attention scores for the first partial block in the
gathered set.

**Fix:** Explicitly state that Option 1 works correctly only because the page
table has already been reordered so that the first gathered block begins at
`p_low * block_size`. The `seq_len` argument alone does not encode the window
start position; correctness relies on the reordered page table placing the
oldest valid block first and the attention mask supplied by the caller masking
out the sub-block tokens that precede `t_low` within that first block. Add a
sentence clarifying this dependency so an implementer does not omit the mask.

---

## Pass 2

### Pass 1 item verification

1. **`paged_sdpa_and_windowing.md` ~line 139 — padding range:** Now reads `-inf for 201–255 (padding at end)`. Confirmed correct.
2. **`eviction_and_page_reuse.md` ~line 252 — live block count:** Now reads "varies between `ceil(w/bs)` and `ceil(w/bs)+1`...precisely, `floor(T/bs) - floor((T-w+1)/bs) + 1`". Confirmed correct.
3. **`paged_sdpa_and_windowing.md` ~lines 421–425 — Option 1 mask caveat:** Note block added. Confirmed present.

---

### Issue 1 — Circular-buffer gather diagram incorrectly labels in-window tokens as "stale, outside window"

**File:** `paged_sdpa_and_windowing.md`, line 305

**Error:** The diagram for the circular-buffer gather at T=9, w=8, block_size=4, N_win=2 labels phys_A slots [2,3] as "stale, outside window → -inf". The window at T=9 is `[T-w+1, T] = [2, 9]`. Tokens 2 and 3 lie inside this window. The data at phys_A slots [2,3] was written at T=2 and T=3 and has not been overwritten (slot 2 is not overwritten until T=10, slot 3 until T=11). These positions are valid and must receive `0` in the mask, not `-inf`.

The surrounding prose and the mask formula at line 464 (`[T - min(T, w-1), T] = [2, 9]`) are both correct. Only the diagram label is wrong.

**Impact:** A reader following the diagram to implement the per-slot masking logic will mask valid in-window tokens to `-inf`, corrupting attention scores for the oldest tokens in every partially-filled write block.

**Fix:** Change the label on phys_A slots [2,3] from "stale, outside window → -inf" to "valid (tokens 2–3, inside window [2,9])". The only genuine stale slots in this example are those that would hold data from before the window, which do not exist at T=9 with w=8.

---

### Issue 2 — `bwp` derivation from `seq_len` in the Compatibility section produces a wrong value

**File:** `paged_sdpa_and_windowing.md`, line 331

**Error:** The Compatibility section states: "bwp = (seq_len / block_size) % N_win". Here `seq_len` is described as the total token count (T+1 in 0-indexed notation). This formula does not match the block-write-pointer definition used everywhere else in the chapter, which is `bwp = (T // block_size) % N_win` (T is the 0-indexed position of the most recent token).

Concrete counterexample with w=8, block_size=4, N_win=2 at the end of the initial fill, T=7 (seq_len=8):

- Correct formula: `bwp = (7 // 4) % 2 = 1 % 2 = 1` — writing is in page index 1 (phys_B). Correct.
- Line-331 formula: `bwp = (8 // 4) % 2 = 2 % 2 = 0` — points to page index 0 (phys_A). Wrong.

A developer deriving `p_oldest` from `seq_len` instead of T will get `p_oldest = (bwp + 1) % N_win = 1` when the correct value is `(1 + 1) % 2 = 0`, causing the gather order to be reversed.

**Impact:** Incorrect gather ordering produces a systematically wrong attention output whenever `T % block_size == block_size - 1` (i.e., at the last token of every block).

**Fix:** Replace `bwp = (seq_len / block_size) % N_win` with `bwp = ((seq_len - 1) // block_size) % N_win`, which correctly converts the token count to a 0-indexed position before applying the block pointer formula. Alternatively, write it explicitly as `bwp = (T // block_size) % N_win` where `T = seq_len - 1`.

---

## Change Log (Pass 2)

**Date:** 2026-03-27

The following corrections were applied to `paged_sdpa_and_windowing.md`.

### Fix 1 — Circular-buffer gather diagram: in-window tokens mislabeled as stale

**File:** `paged_sdpa_and_windowing.md`, line 305

**Change:** At T=9, w=8, block_size=4, N_win=2 the window is [T-w+1, T] = [2, 9]. Tokens 2 and 3 are inside this window; phys_A slots [2,3] hold their KV data and must not be masked out. The label previously read:

```
phys_A slots 2–3 = positions 2–3 (stale, outside window → -inf)
```

Corrected to:

```
phys_A slots 2–3 = positions 2–3 (valid, inside window [2,9])
```

Only tokens strictly earlier than `T - w + 1 = 2` would be stale; there are none in this example.

### Fix 2 — Off-by-one in `bwp` derivation from `seq_len`

**File:** `paged_sdpa_and_windowing.md`, line 331

**Change:** The Compatibility section stated `bwp = (seq_len / block_size) % N_win`. Because `seq_len` is the token count (T+1), integer division by `block_size` advances to the next block index at every block boundary, giving a value one block ahead of the actual write pointer. For example, at T=7 (seq_len=8, block_size=4, N_win=2): the old formula gives `(8//4) % 2 = 0` (wrong), while the correct result is `(7//4) % 2 = 1`.

Corrected to: `bwp = ((seq_len - 1) // block_size) % N_win`

## Change Log (Pass 1)

**Date:** 2026-03-27

The following corrections were applied to the Chapter 5 source files.

### Fix 1 — Inverted padding range in Strategy A diagram

**File:** `paged_sdpa_and_windowing.md`, line 139

**Change:** The mask annotation previously read `-inf for 201–191 (padding at end)`, which has an inverted (nonsensical) range. The assembled tensor covers virtual pages 1–3 (3 blocks × 64 tokens), occupying positions 64–255. Padding occupies the 55 slots after the last valid token (T=200), i.e., positions 201–255.

Corrected to: `-inf for 201–255 (padding at end)`

### Fix 2 — Strategy A live block count stated as a fixed value when it is variable

**File:** `eviction_and_page_reuse.md`, ~line 252

**Change:** The original text claimed live blocks "stabilises at `ceil(w / block_size) + 1`" as a fixed value. This is incorrect — the live count is `floor(T/bs) - floor((T-w+1)/bs) + 1`, which equals either `ceil(w/bs)` or `ceil(w/bs) + 1` depending on whether the window boundary falls on a block boundary.

Replaced with: "The number of live blocks varies between `ceil(w/bs)` and `ceil(w/bs) + 1` depending on whether the window boundary falls on a block boundary. Precisely, the live count is `floor(T/bs) - floor((T-w+1)/bs) + 1`, where `bs = block_size`. The worst-case block pool allocation must use `ceil(w/bs) + 1`."

### Fix 3 (Issue 3) — Broken "Next" footer link

**No change made.** Per instructions, the link to `../ch6_t3k_sharding/index.md` is correct per the guide plan — Chapter 6 has not yet been written. The link is intentionally left as-is.

### Fix 4 — Missing mask construction caveat in Option 1 (Strategy B host-side reordering)

**File:** `paged_sdpa_and_windowing.md`, after the `seq_len = min(T+1, w)` explanation (~line 419)

**Change:** The original text did not mention that correctness of Option 1 also depends on an attention mask that zeros out sub-block positions before `t_low` within the first gathered block. Without this mask, stale tokens from the previous window cycle in the partial first block receive non-zero attention scores.

Added note: "Correctness also requires an attention mask that sets scores to −∞ for the sub-block positions before `t_low` within the first gathered block (positions `p_low * block_size` through `t_low − 1`). Without this mask, stale tokens from the previous window cycle in the partial first block will receive non-zero attention."

---

## Pass 3

### Pass 2 item verification

1. **`paged_sdpa_and_windowing.md` ~line 305 — phys_A slots [2,3] label:** Now reads `(valid, inside window [2,9])`. Confirmed corrected.
2. **`paged_sdpa_and_windowing.md` ~line 331 — `bwp` formula:** Now reads `bwp = ((seq_len - 1) // block_size) % N_win`. Confirmed corrected.

---

### Issue 1 — Option 1 mask note uses Strategy A terminology (`p_low`) that is undefined in Strategy B, producing a wrong mask range

**File:** `paged_sdpa_and_windowing.md`, lines 421–425

**Error:** The caveat note added in Pass 1 / Pass 2 reads:

> Correctness also requires an attention mask that sets scores to −∞ for the sub-block positions before `t_low` within the first gathered block (positions `p_low * block_size` through `t_low − 1`).

`p_low` is a Strategy A concept defined as `floor((T - w + 1) / block_size)`. In Strategy B / Option 1, the page table has been reordered so that `p_oldest` (not `p_low`) is placed at assembled-tensor index 0. The first assembled block starts at virtual token address `p_oldest * block_size`, not `p_low * block_size`. Because `p_oldest = (floor(T / block_size) + 1) % N_win` and `p_low = floor((T - w + 1) / block_size)`, these are in general different values.

Concrete example: T=12, w=8, block_size=4, N_win=2.
- t_low = 5
- p_low (Strategy A) = floor(5/4) = 1
- p_oldest (Strategy B) = (floor(12/4) + 1) % 2 = (3+1) % 2 = 0
- After reordering, assembled-tensor position 0 corresponds to virtual token address `p_oldest * block_size = 0`. The stale prefix that must be masked is assembled-tensor slots 0 through `t_low mod block_size - 1` = slots 0 through 0 (i.e., position corresponding to token 0, which is from the previous rotation and outside window [5,12]).
- If an implementer uses the note's formula `[p_low * block_size, t_low - 1]` = [4, 4] as assembled-tensor indices, they mask only slot 4, leaving slot 0 (the actually-stale token from the previous rotation) unmasked.

**Impact:** An implementer constructing the mask per the note's description will leave stale tokens in the oldest assembled block unmasked, producing nonzero attention scores for out-of-window tokens.

**Fix:** Replace the mask range in the note with assembled-tensor coordinates: "mask assembled-tensor positions `0` through `(t_low mod block_size) - 1` in the first gathered block to −∞ — equivalently, the first `(T - w + 1) mod block_size` slots of the assembled tensor. If `(T - w + 1) mod block_size == 0` the window start is block-aligned and no partial-block masking is needed."

---

## Change Log (Pass 3)

**Date:** 2026-03-27

The following correction was applied to `paged_sdpa_and_windowing.md`.

### Fix 1 — Option 1 mask note: wrong coordinates replaced with assembled-tensor coordinates

**File:** `paged_sdpa_and_windowing.md`, lines 421–425

**Change:** The Option 1 mask caveat referenced Strategy A terminology (`p_low * block_size` through `t_low − 1`) as assembled-tensor indices. In Strategy B / Option 1 the page table is reordered so `p_oldest` occupies assembled-tensor index 0, not `p_low`. The stale prefix is therefore the first `(T − w + 1) mod block_size` slots of the assembled tensor, not the range `[p_low * block_size, t_low − 1]`.

Replaced with: "mask assembled-tensor slots `0` through `(t_low mod block_size) − 1` — i.e., the first `(T − w + 1) mod block_size` slots of the assembled tensor. If this value is zero, no masking is needed (the window boundary falls exactly on a block boundary)."

---

## Pass 4

### Pass 3 item verification

- `paged_sdpa_and_windowing.md` lines 421–428: The mask note now reads that the stale prefix occupies assembled-tensor slots `0` through `(t_low mod block_size) − 1`, equivalent to the first `(T − w + 1) mod block_size` slots. Coordinates are correct for Strategy B / Option 1 after page table reordering. Confirmed.

---

### Issue 1 — "Stale positions within the write block" mask bullet unconditionally masks slots that may be in-window

**File:** `paged_sdpa_and_windowing.md`, lines 468–470

**Error:** The "Mask Construction for Windowed Paging" section categorises assembled-tensor slots `[T % block_size + 1, block_size - 1]` of the current write block as "Stale positions … that have not yet been overwritten in this rotation" and implies they should receive `-inf`. This is incorrect when `w > block_size`.

Those slots hold token data from exactly one rotation ago, at absolute token positions `(floor(T / block_size) * block_size + k)` for `k` in `[T % block_size + 1, block_size - 1]`, which simplifies to positions `(floor(T / block_size) - 1) * block_size + (T % block_size + 1)` through `floor(T / block_size) * block_size - 1`. Whether these fall inside the window `[t_low, T] = [T − w + 1, T]` depends on the relationship between `w` and `block_size`. When `w ≥ block_size` (the common case — e.g., `w = 4096`, `block_size = 64`), the tokens from the previous rotation in those slots are typically inside the window and must receive `0`, not `-inf`.

Concrete counterexample using the chapter's own numbers: T=9, w=8, block_size=4, N_win=2.
- Write block = phys_A (bwp=0), T % block_size = 1, so "stale slots" = slots 2–3.
- Those slots hold tokens 2 and 3 (written at T=2,3; not yet overwritten).
- t_low = T − w + 1 = 2. Tokens 2 and 3 are at positions ≥ t_low → inside window [2,9].
- The corrected diagram at line 305 already labels them `(valid, inside window [2,9])`.
- The mask bullet at line 468 contradicts the diagram: following the bullet, an implementer masks slots 2–3 to `-inf`, zeroing out two valid in-window tokens and corrupting the attention output.

The condition for a slot in the write block to be genuinely stale is that its token position is below `t_low`. That is: slot `k` (0-indexed within the write block) is stale if and only if `floor(T / block_size) * block_size + k - block_size < t_low`, i.e., `k < t_low - (floor(T / block_size) - 1) * block_size`. This equals `(T − w + 1) mod block_size` when `(T − w + 1) mod block_size > 0` and `T ≥ w`; otherwise no slots in the write block are stale.

**Fix:** Replace the "Stale positions" bullet with: "Stale positions within the write block: when `(T − w + 1) mod block_size ≠ 0` and `T ≥ w`, assembled-tensor slots `0` through `(T − w + 1) mod block_size − 1` of the first gathered block (which is `p_oldest`'s block, placed first by the reordering) hold token data from outside the window. These are the same slots identified in the Option 1 mask note above. No additional stale slots exist in the write block." (Note: the write block is last in the assembled tensor after reordering; the stale prefix lives in the first assembled block, not the last.)

---

### Issue 2 — Gather diagram at T=9 claims valid "positions 0–9" but tokens 0–1 are absent from the assembled tensor

**File:** `paged_sdpa_and_windowing.md`, line 303

**Error:** The mask annotation reads:

```
Mask: positions 0–9 → phys_B slots 0–3 = positions 4–7 (valid)
                        phys_A slots 0–1 = positions 8–9 (valid)
                        phys_A slots 2–3 = positions 2–3 (valid, inside window [2,9])
```

The claim "positions 0–9" implies the assembled tensor covers tokens 0 through 9. It does not. At T=9, phys_A was overwritten: slots 0–1 now hold tokens 8–9, slots 2–3 hold tokens 2–3 (from the previous rotation, still in-window). The assembled tensor covers tokens {4,5,6,7,8,9,2,3} — i.e., token positions 2–9 — not 0–9. Tokens 0 and 1 do not appear anywhere in the assembled tensor.

**Impact:** A reader verifying the mask range against the expected valid set will expect tokens 0 and 1 to be addressed (and masked to `-inf`), wasting effort and potentially introducing a spurious mask slot that does not correspond to any assembled-tensor position. More seriously, they may conclude that the assembled tensor is `N_win * block_size + 2 = 10` positions wide rather than 8.

**Fix:** Replace "positions 0–9" with "positions 2–9 (the valid window)" and add a note that assembled-tensor slots 0–7 map to token positions [4,5,6,7,8,9,2,3] in that order; no slot maps to tokens 0 or 1.

---

### Issue 3 — `seq_len = min(T+1, w)` passed to `paged_sdpa_decode` is wrong at steady state when the assembled tensor is `N_win * block_size` wide

**File:** `paged_sdpa_and_windowing.md`, line 418–419

**Error:** The Option 1 description instructs setting `seq_len = min(T+1, w)` so "the kernel's mask logic covers the correct number of valid positions." At steady state `min(T+1, w) = w`. However, the assembled tensor after reordering has `N_win * block_size` slots, where `N_win = ceil(w / block_size)`, so the tensor width is `ceil(w / block_size) * block_size ≥ w`. When `w` is not a multiple of `block_size`, `N_win * block_size > w`.

If `paged_sdpa_decode` uses `seq_len` to determine how many trailing slots to mask to `-inf` (positions `seq_len` through `N_win * block_size - 1`), passing `seq_len = w` instructs the kernel to treat `N_win * block_size - w` trailing slots as padding and mask them. But after reordering, those trailing slots are in the write block and contain either valid in-window tokens or the stale prefix described in Issue 1 above — they are not simple padding at the end of the tensor. The kernel's linear padding mask (zeros out the last `N_win*block_size - seq_len` positions) is designed for an append-only tensor where unwritten slots are at the tail. In the reordered circular layout the unwritten/stale region is at the head (assembled-tensor slots 0 through `(T−w+1) mod block_size − 1`), not the tail.

**Impact:** At steady state, the kernel's trailing-padding mask does not mask the stale prefix (Issue 1), and may mask valid tokens in the write block's occupied slots if `block_size` does not divide `w`.

**Fix:** Explicitly state that the `seq_len` argument suppresses trailing-padding masking only and is not sufficient to handle the stale prefix at the head of the assembled tensor. The caller must supply a separate full-width `attn_mask` that (a) sets `-inf` at assembled-tensor slots 0 through `(T−w+1) mod block_size − 1` (the stale prefix, when non-zero) and (b) sets `0` for all remaining in-window slots. The `seq_len` value should be `N_win * block_size` (full assembled width) when a caller-supplied `attn_mask` already encodes all validity, to prevent the kernel from applying a second, conflicting trailing mask.

---

## Change Log (Pass 4)

**Date:** 2026-03-27

The following three corrections were applied to `paged_sdpa_and_windowing.md`.

### Fix 1 — Unconditional stale write-block mask replaced with conditional rule

**File:** `paged_sdpa_and_windowing.md`, Mask Construction section (~lines 468–470 before edits)

**Change:** The "Stale positions within the write block" bullet unconditionally marked
assembled-tensor slots `[T % block_size + 1, block_size - 1]` of the write block as
`−∞`. This is wrong when `w >= block_size`: those slots hold tokens from one full block
rotation ago whose positions are `>= t_low` and therefore inside the current window.

The bullet was replaced with a conditional rule:
- When `w < block_size`: mask write-block slots `[T % block_size + 1, block_size - 1]`
  to `−∞` (those token positions are outside the window).
- When `w >= block_size`: these slots are valid in-window tokens and must NOT be masked.

A separate bullet was also added to explicitly call out the stale prefix in the first
(oldest) assembled block — assembled-tensor slots `0` through
`(T - w + 1) mod block_size - 1` — which is the correct location of out-of-window data
in the reordered tensor and applies regardless of the `w` vs. `block_size` relationship.

### Fix 2 — Diagram annotation "positions 0–9" corrected to "positions 2–9"

**File:** `paged_sdpa_and_windowing.md`, circular-buffer gather diagram at T=9 (~line 303 before edits)

**Change:** The mask annotation previously read "positions 0–9", implying the assembled
tensor covers tokens 0 through 9. This is wrong. At T=9, w=8, block_size=4, N_win=2:
- `t_low = T - w + 1 = 2`
- phys_A slots 0–1 now hold tokens 8–9 (overwritten); phys_A slots 2–3 hold tokens
  2–3 (previous rotation, still in-window).
- The assembled tensor covers token positions {4,5,6,7,8,9,2,3} = positions 2–9.
- Tokens 0 and 1 are absent — their slots were overwritten.

Corrected to "positions 2–9" and a note was added clarifying that assembled-tensor
slots 0–7 map to token positions [4,5,6,7,8,9,2,3] and that tokens 0 and 1 do not
appear in the assembled tensor.

### Fix 3 — `seq_len = min(T+1, w)` described as insufficient alone; caller-supplied `attn_mask` requirement stated explicitly

**File:** `paged_sdpa_and_windowing.md`, Option 1 host-side reordering description (~lines 418–419 before edits)

**Change:** The original text stated `seq_len = min(T+1, w)` "so the kernel's mask logic
covers the correct number of valid positions", implying this is sufficient. It is not
sufficient at steady state when `block_size` does not divide `w`: the assembled tensor
is `N_win * block_size` wide (>= w), and `seq_len` only masks trailing padding — it
does not mask the stale prefix at the head of the reordered tensor (assembled-tensor
slots `0` through `(T - w + 1) mod block_size - 1`).

A block-quoted warning was added stating:
- `seq_len = min(T+1, w)` handles trailing-padding masking only.
- A separate caller-supplied `attn_mask` is required to set stale-prefix slots to `−∞`.
- Without this explicit `attn_mask`, stale tokens at the head remain unmasked even with
  `seq_len` set correctly.
- If `(T - w + 1) mod block_size == 0` the window boundary is block-aligned and no
  stale-prefix mask is needed.

---

## Pass 5

### Pass 4 item verification

1. **Write-block mask conditional rule** — `paged_sdpa_and_windowing.md` lines 477–487: the `w < block_size` / `w >= block_size` split is present and correctly states that write-block slots `[T % block_size + 1, block_size - 1]` are masked to `−∞` only when `w < block_size`. Confirmed.
2. **Diagram annotation "positions 2–9"** — line 303: reads `Mask: positions 2–9 → ...`. Confirmed.
3. **`seq_len = min(T+1, w)` trailing-padding-only caveat with explicit `attn_mask` requirement** — lines 425–435: block-quoted warning present and states both restrictions correctly. Confirmed.

---

### Issue 1 — Stale-prefix length formula `(T − w + 1) mod block_size` is applied to the wrong block and gives a wrong (non-zero) count when `p_oldest` does not contain `t_low`

**File:** `paged_sdpa_and_windowing.md`, lines 489–493 (the "Stale prefix in the oldest (first) assembled block" bullet)

**Error:** The text states: "when `(T - w + 1) mod block_size != 0` and `T >= w`, the first `(T - w + 1) mod block_size` slots of the assembled tensor … hold token data from before `t_low` and must be masked to `−∞`."

The formula `(T - w + 1) mod block_size` gives the token-offset of `t_low` within its containing block. It equals the number of slots at the head of that block that precede `t_low`. It is only the correct stale-prefix length for the first assembled block when `p_oldest` is the same block that contains `t_low`, i.e., when `floor(t_low / block_size) == p_oldest`. This is not guaranteed in general.

Concrete counterexample using the chapter's own numbers: T=9, w=8, block_size=4, N_win=2.

- `t_low = T − w + 1 = 2`
- `p_oldest = (floor(9/4) + 1) % 2 = (2+1) % 2 = 1` → phys_B (holds tokens 4–7)
- Block containing `t_low`: `floor(2/4) = 0` → phys_A (holds tokens 2–3 and 8–9)
- `p_oldest ≠ floor(t_low / block_size)` (1 ≠ 0)
- Formula gives stale prefix = `(9 − 8 + 1) mod 4 = 2 mod 4 = 2` slots.
- Assembled-tensor slots 0–1 are phys_B slots 0–1 = tokens 4 and 5, which are inside window [2,9].
- Correct stale prefix in the first assembled block: 0 slots (phys_B has no tokens before `t_low`).

Following the formula an implementer masks tokens 4 and 5 to `−∞`, corrupting attention scores for two valid in-window tokens.

The formula is correct only when `N_win = 1` (trivially, the single block always contains `t_low`) or when the window is exactly block-aligned in a way that forces `p_oldest` to equal `floor(t_low / block_size)`. In all other cases the stale prefix of the first assembled block is 0 (because `p_oldest` contains tokens from at most one full rotation ago whose earliest token position equals `p_oldest * block_size`, which is `>= t_low` at steady state when `p_oldest != floor(t_low / block_size)`).

The actual stale-prefix masking for Strategy B / Option 1 is correctly described in the Option 1 block-quoted warning (lines 425–435) and in the "Stale prefix in the oldest (first) assembled block" bullet from the Pass 3 mask-coordinate fix: assembled-tensor slots `0` through `(T − w + 1) mod block_size − 1` of the *first gathered block* must be masked only if that first gathered block is `p_oldest = floor(t_low / block_size)`. For a circular reordering, `p_oldest` is the block with the oldest tokens, which in general is not the block containing `t_low`. The stale sub-block prefix lives at the head of the block that contains `t_low`; after reordering that block is placed at assembled-tensor position `(floor(t_low / block_size) − p_oldest + N_win) % N_win * block_size`, not at position 0.

**Impact:** An implementer following the bullet will mask the wrong assembled-tensor slots to `−∞`, producing nonzero attention scores for out-of-window tokens or zero attention scores for valid in-window tokens, depending on the specific T, w, and block_size values.

**Fix:** Rewrite the "Stale prefix in the oldest (first) assembled block" bullet to clarify that the stale sub-block prefix lives in the assembled block corresponding to `floor(t_low / block_size)`, not necessarily in assembled-tensor slot 0. After page-table reordering, that block sits at assembled-tensor index `delta * block_size` where `delta = (floor(t_low / block_size) − p_oldest + N_win) % N_win`. The first `t_low mod block_size` slots of that block (assembled-tensor slots `delta * block_size` through `delta * block_size + (t_low mod block_size) − 1`) must be masked to `−∞`. When `t_low mod block_size == 0` no masking is needed regardless of `delta`.

---

## Change Log (Pass 5)

**Date:** 2026-03-27

The following correction was applied to `paged_sdpa_and_windowing.md`.

### Fix 1 — Stale-prefix bullet rewritten with correct assembled-tensor block index

**File:** `paged_sdpa_and_windowing.md`, lines 488–493 (before edit)

**Change:** The "Stale prefix in the oldest (first) assembled block" bullet incorrectly applied the formula `(T - w + 1) mod block_size` as a stale-slot count starting at assembled-tensor slot 0. This is wrong whenever `p_oldest` (the physically oldest block, placed at assembled-tensor index 0 after reordering) is not the same block that contains `t_low = T - w + 1`. In that case assembled-tensor slots 0 through `stale_count - 1` correspond to fully in-window tokens (e.g., tokens 4 and 5 in the T=9, w=8, block_size=4 example), and masking them to `−∞` corrupts attention scores.

The correct logic identifies the block containing `t_low` (`b_tlow = t_low // block_size`), maps it to its assembled-tensor index (`assembled_idx = (b_tlow - p_oldest + N_win) % N_win`), and masks only the first `stale_count = t_low % block_size` slots of that assembled block (assembled-tensor slots `assembled_idx * block_size` through `assembled_idx * block_size + stale_count - 1`). Blocks before and after `assembled_idx` are fully valid.

The bullet was replaced with a numbered procedure plus a worked example using the chapter's own numbers (T=9, w=8, block_size=4, N_win=2, p_oldest=1):
- `t_low = 2`, `b_tlow = 0` (phys_A), `assembled_idx = 1`, `stale_count = 2`
- Masking is applied to assembled-tensor slots 4–5 (phys_A slots 0–1 = tokens 0, 1), not slots 0–1.

## Pass 6

### Pass 5 item verification

- `paged_sdpa_and_windowing.md` lines 488–511: The stale-prefix bullet now uses the `assembled_idx = (b_tlow - p_oldest + N_win) % N_win` formula and includes a worked example claiming `stale_count = 2` and masking assembled-tensor slots 4–5. Verification of the worked example reveals an error — see Issue 1 below.

---

### Issue 1 — Worked example at lines 505–511 claims "phys_A slots 0–1 = tokens 0,1" and masks them, but those slots hold valid tokens 8 and 9 at T=9

**File:** `paged_sdpa_and_windowing.md`, lines 505–511

**Error:** The worked example computes `stale_count = 2` and instructs masking assembled-tensor slots 4–5, labelled "phys_A slots 0–1 = tokens 0,1". This is wrong. At T=9, write operations T=8 and T=9 have already overwritten phys_A slots 0 and 1 with tokens 8 and 9 respectively (confirmed by the trace at lines 259–270 and by the diagram at lines 302–308 which correctly labels "phys_A slots 0–1 = positions 8–9 (valid)"). Masking assembled slots 4–5 would suppress the two most recently written, fully in-window tokens.

Moreover, `stale_count = 2` is wrong for this example. The assembled tensor at T=9 covers exactly tokens {4,5,6,7,8,9,2,3} = positions 2–9 (the complete window [2,9]); there are zero stale slots. The diagram at line 303 states this directly: "The assembled tensor covers exactly token positions 2–9". The procedure gives a non-zero stale_count and a wrong masking target that directly contradicts the diagram on the same page.

**Root cause:** The formula `stale_count = t_low % block_size` counts how many absolute token positions within the virtual block precede `t_low`. But in the circular buffer those positions have been overwritten with newer tokens. Stale slots only exist in the assembled tensor when the block containing `t_low` is `p_oldest` itself (the physically oldest block, which has not yet been partially overwritten in the current rotation). In the T=9 example `b_tlow % N_win = 0 = bwp`, not `p_oldest = 1`. The block containing `t_low` is the currently-being-written block; its early slots have already been updated and are valid.

**Concrete check:** At T=9, phys_A slot 0 was written at T=8 (token 8), slot 1 at T=9 (token 9). Token 8 and token 9 are the newest tokens and must not be masked. The correct stale_count for this example is 0.

**Impact:** An implementer following the numbered procedure will mask tokens 8 and 9 to `−∞`, producing zero attention weight for the two most recent context tokens. The attention output is incorrect.

**Fix:** Add a guard: masking of the first `stale_count` slots of the block at `assembled_idx` is only required when `b_tlow % N_win == p_oldest` (i.e., the block containing `t_low` is the oldest, not-yet-overwritten block). When `b_tlow % N_win == bwp` (the write block) or any other block that has already been partially overwritten in the current rotation, `stale_count` stale tokens no longer exist in the assembled tensor and no masking is needed. Equivalently, stale-prefix masking is only warranted when `assembled_idx == 0` after page-table reordering (since `p_oldest` is always placed at assembled position 0). Update the worked example to reflect `stale_count = 0` and no masking for the T=9 case.

---

### Issue 2 — Previous-rotation token position formula at line 476 is wrong by a factor of `N_win`

**File:** `paged_sdpa_and_windowing.md`, line 476

**Error:** The "Stale positions within the write block" bullet states that slots `[T % block_size + 1, block_size - 1]` of the write block hold "token data from one full block rotation ago (positions `floor(T / block_size) * block_size + k - block_size` for each slot index `k`)."

The formula `floor(T/block_size) * block_size + k - block_size` subtracts only `block_size`, as if the circular period were 1 block. The correct circular period is `N_win` blocks. The previous-rotation token in write-block slot `k` was written when the write pointer last visited the same virtual page, which was `N_win` block-writes ago, not 1. The correct formula is:

```
(floor(T / block_size) - N_win) * block_size + k
```

Concrete check at T=9, block_size=4, N_win=2, write block = phys_A (bwp=0): the chapter's formula gives `2*4+k-4 = 4+k`, producing tokens 6,7 for k=2,3. The correct formula gives `(2-2)*4+k = k`, producing tokens 2,3 for k=2,3. The actual phys_A slots 2–3 hold tokens 2 and 3 (confirmed by the trace at lines 267–270), matching the correct formula, not the chapter's formula.

**Impact:** A developer using this formula to reason about which previous-rotation tokens occupy write-block slots will compute incorrect token indices, leading to wrong masking decisions and potentially wrong staleness analysis for any diagnostic or custom masking code they build.

**Fix:** Replace `floor(T / block_size) * block_size + k - block_size` with `(floor(T / block_size) - N_win) * block_size + k`.

---

### Issue 3 — Staleness condition for write-block tail slots is "w < block_size" but the correct condition is "block_size does not divide w"

**File:** `paged_sdpa_and_windowing.md`, lines 480–487

**Error:** The "Stale positions within the write block" bullet concludes "these slots are stale **only when `w < block_size`**." This is overly restrictive. Using the corrected formula from Issue 2, the minimum previous-rotation token in write-block tail slots `[T % block_size + 1, block_size - 1]` is at position `(floor(T/block_size) - N_win) * block_size + T % block_size + 1 = T - N_win * block_size + 1`. This is less than `t_low = T - w + 1` iff:

```
T - N_win * block_size + 1 < T - w + 1
↔  N_win * block_size > w
↔  ceil(w / block_size) * block_size > w
↔  block_size does not divide w
```

So the correct condition is "stale when `block_size` does not divide `w`" (i.e., `w % block_size != 0`), regardless of the relative sizes of `w` and `block_size`. The chapter's condition `w < block_size` is a strict subset: it implies `block_size` doesn't divide `w` (since w ≥ 1 and if w < block_size then w cannot be a positive multiple of block_size), but it excludes cases like `w = 100`, `block_size = 64` where `w > block_size`, `block_size` does not divide `w`, and the tail slots are stale.

**Concrete counterexample:** T=200, w=100, block_size=64, N_win=2. t_low=101. bwp=(200//64)%2=3%2=1 (write block = page 1). Write-block tail slots `[200%64+1, 63] = [9, 63]`. Previous-rotation token at slot k: `(3-2)*64 + k = 64 + k`. For k=9: token 73 < t_low=101 → stale. The chapter says `w=100 >= block_size=64` → "do NOT mask them". This is incorrect; tokens 64–100 (tail slots 0–36) are outside the window and must be masked.

**Impact:** For any configuration where `w >= block_size` but `block_size` does not divide `w`, write-block tail slots that are genuinely stale will be left unmasked, producing nonzero attention scores for out-of-window tokens.

**Fix:** Replace the condition "`w < block_size`" / "`w >= block_size`" with "`w % block_size != 0`" / "`w % block_size == 0`" respectively.

---

## Change Log (Pass 6)

**Date:** 2026-03-27

The mask construction subsection of `paged_sdpa_and_windowing.md` was rewritten
from scratch to eliminate all accumulated correctness issues identified in Passes
1–5.

### Summary of prior issues resolved

The section had gone through five rounds of incremental patches, each fixing one
error while introducing or leaving others. The accumulated problems were:

1. Stale write-block slots unconditionally masked (wrong when `w >= block_size`).
2. Stale-prefix formula `(T - w + 1) mod block_size` applied to assembled-tensor
   slot 0 (wrong when `p_oldest` does not contain `t_low`).
3. Worked example (`stale_count = 2`, masking slots 4–5 = tokens 8,9) contradicted
   the block-trace diagram on the same page; tokens 8 and 9 are the two most recently
   written tokens and must not be masked.
4. Previous-rotation token position formula `floor(T/bs)*bs + k - bs` off by a factor
   of `N_win` (correct formula: `(floor(T/bs) - N_win)*bs + k`).
5. Staleness condition stated as "`w < block_size`" but the correct condition is
   "`block_size` does not divide `w`" (`w % block_size != 0`).

Rather than patch a fifth time, the section was rewritten to cover only the common
deployable case where `block_size` divides `w` exactly, where the analysis is clean
and verifiably correct.

### What was rewritten

**File:** `paged_sdpa_and_windowing.md`, section "Mask Construction for Windowed Paging"

**Change:** The entire subsection was replaced with a self-contained, verified
derivation structured as follows:

1. **Scope declaration** — explicitly states the section applies only when
   `N_win * block_size == w` (i.e., `block_size` divides `w`). Non-divisible cases
   are flagged as out-of-scope with a note directing implementers to choose
   `block_size` to divide `w` in production.

2. **Definitions** — `t_low`, `phys_block(t)`, `bwp`, `p_oldest`,
   `oldest_block_start` defined in one place.

3. **Step 1: Count stale slots** — derives `n_stale = max(0, t_low - oldest_block_start)`
   and proves algebraically that when `block_size` divides `w` this equals 0 at every
   steady-state step. Proof uses `T = q*bs + r`, `N_win = w/bs`, and shows
   `t_low - oldest_block_start = r - bs + 1 <= 0`.

4. **Step 2: Mask application** — states that with `n_stale = 0` no stale-prefix
   masking is needed; only fill-phase trailing-padding masking (via `seq_len = T + 1`)
   is required. At steady state `seq_len = w` suffices.

5. **Worked example (T=9, w=8, block_size=4, N_win=2)** — verifies `n_stale = 0`,
   shows the assembled tensor covers exactly tokens 2–9 (the full window), and confirms
   no masking is needed beyond the kernel's default trailing-padding path.

6. **Non-divisible-case note** — advises that when `block_size` does not divide `w`,
   partial-block masking analysis is required and is outside this guide's scope.

### Key correctness properties of the new text

- `n_stale = 0` is proved for all steady-state `T`, not assumed.
- The worked example is consistent with the block-trace diagram at lines 259–270
  and the gather diagram at lines 293–309. All three sources agree that at T=9
  the assembled tensor holds tokens 2–9 with no stale slots.
- No conditional branches on `w < block_size` or `w >= block_size` remain; the
  old (incorrect) conditional was removed entirely.
- The token-position formula for previous-rotation write-block slots (which was
  wrong by a factor of `N_win`) does not appear in the rewritten section, since
  the common case requires no write-block tail masking.

## Pass 7

### Pass 6 item verification

The rewritten mask construction section (`paged_sdpa_and_windowing.md` lines 469–576) was reviewed in full.

- Algebraic proof that `n_stale = 0` when `block_size | w`: verified correct. Substituting `T = q*bs + r` and `N_win = w/bs` gives `t_low - oldest_block_start = r - bs + 1 <= 0` for all `0 <= r < bs`.
- Worked example T=9, w=8, block_size=4, N_win=2: all intermediate values (`bwp`, `p_oldest`, `oldest_block_start`, `t_low`, `n_stale`) verified numerically correct and consistent with the block-trace diagram at lines 259–270.
- Scope restriction to divisible case and the non-divisible-case note: present and correctly placed.

---

### Issue 1 — Fill-phase masking fails for T < block_size because page-table reordering places an unwritten block at the head of the assembled tensor

**File:** `paged_sdpa_and_windowing.md`, lines 402–415 (host reordering function) and lines 533–535 (fill-phase mask claim)

**Error:** The host function `reorder_page_table_for_windowing` applies the `p_oldest = (T // block_size + 1) % N_win` formula unconditionally for all T. During the first `block_size` steps (T = 0 .. block_size - 1), `bwp = 0` and `p_oldest = 1`, so the function places physical block 1 (phys_B) first in the reordered page table, followed by physical block 0 (phys_A).

At this point in the fill phase, phys_B has never been written. After reordering, the assembled tensor is:

```
slots 0 .. block_size-1   : phys_B  (uninitialized — never written)
slots block_size .. T      : phys_A  (tokens 0 .. T, valid)
slots T+1 .. N_win*block_size-1 : phys_A (uninitialized remainder)
```

The section then says: "trailing slots `[T + 1, N_win * block_size - 1]` that have never been written must be set to `−∞`. This is handled by passing `seq_len = T + 1` to `paged_sdpa_decode`; the kernel's trailing-padding mask covers these positions automatically."

The kernel's trailing-padding mask suppresses positions `seq_len` through `N_win*block_size - 1`, i.e., positions `T+1` onward. But phys_B occupies assembled-tensor positions `0 .. block_size-1`, which are entirely **below** `T+1`. These positions are not masked by the kernel and receive non-zero attention scores from uninitialized DRAM contents.

**Concrete example — T=2, block_size=4, N_win=2:**

- `p_oldest = (2//4 + 1) % 2 = 1` → phys_B first (never written)
- Assembled: [phys_B(uninit), phys_B(uninit), phys_B(uninit), phys_B(uninit), t0, t1, t2, phys_A(uninit)]
- `seq_len = 3` → kernel masks positions 3–7 to `−∞`
- Positions 0–2 (phys_B slots 0–2, uninitialized) are unmasked and contribute to attention output

An implementer following the chapter's fill-phase recipe will produce wrong attention scores for all T in `[0, block_size - 1]`.

**The non-reordered layout is correct at fill phase.** Without reordering, phys_A occupies positions 0..block_size-1 of the assembled tensor, phys_B occupies block_size..N_win*block_size-1. With `seq_len = T+1`, the kernel correctly masks positions T+1 onward (the unwritten tail of phys_A and all of phys_B). The reordering is only correct and necessary once all N_win blocks have been written at least once (i.e., T >= N_win * block_size - 1 = w - 1 in the divisible case).

**Fix:** Add a guard to `reorder_page_table_for_windowing`: apply the circular reordering only when `T >= N_win * block_size - 1` (i.e., after the first full rotation has completed). For `T < N_win * block_size - 1`, pass the page table in its natural order `[0, 1, ..., N_win - 1]` so that the unwritten blocks remain in the tail where `seq_len = T + 1` correctly suppresses them. Equivalently, note in the text that the fill-phase claim "`seq_len = T + 1` handles all unwritten slots" is only valid when the page table is NOT reordered during fill phase.

---

No further issues found in the rewritten section or in `eviction_and_page_reuse.md` and `index.md`.

## Change Log (Pass 7)

**Date:** 2026-03-27

The following correction was applied to `paged_sdpa_and_windowing.md`.

### Fix 1 — Fill-phase guard added to `reorder_page_table_for_windowing`

**File:** `paged_sdpa_and_windowing.md`, Option 1 host-side reordering code block
(lines 401–416 before edit)

**Problem:** The function applied `p_oldest = (T // block_size + 1) % N_win`
unconditionally for all T. For T in `[0, block_size - 1]` (the first block of
the fill phase), this gives `p_oldest = 1`, placing phys_B — which has never
been written — at assembled-tensor index 0. The kernel's trailing-padding mask
suppresses positions `T+1` onward; because phys_B occupies positions
`0 .. block_size-1` (all below `T+1`), its uninitialized DRAM contents are
unmasked and contribute nonzero attention scores. For example, at T=2 with
block_size=4, N_win=2: the reordered assembled tensor places three uninitialized
phys_B slots at positions 0–2, none of which are suppressed by `seq_len = 3`.

**Fix:** Added a fill-phase guard inside the per-sequence loop:

```python
if T < N_win * block_size:
    # Fill phase: keep natural order; seq_len = T+1 masks unwritten tail
    pass
else:
    # Steady state: apply circular reorder so oldest block is at index 0
    p_oldest = (T // block_size + 1) % N_win
    indices = [(p_oldest + i) % N_win for i in range(N_win)]
    reordered[b] = page_table[b, indices]
```

During the fill phase (`T < N_win * block_size`), blocks are written in natural
order (phys_A first, then phys_B, ...), so the natural page table ordering is
already "oldest first" and no reordering is needed. Keeping the natural order
ensures all unwritten blocks remain in the assembled-tensor tail, where
`seq_len = T + 1` correctly suppresses them. Circular reordering is applied
only once the buffer has been fully written at least once
(`T >= N_win * block_size`), when every block holds valid data and the
oldest-first ordering must be established explicitly.

A block-quoted explanation of the fill-phase guard and its rationale was also
added immediately after the code block, so implementers understand why the
guard is required and when reordering becomes safe.

## Pass 8

### Issue 1 — `check_window_invariants` references undefined `N_blocks`

**File:** `eviction_and_page_reuse.md`, lines 144–158

**Error:** The defensive checker asserts `page_table[b, p] < N_blocks`, but `N_blocks` is never defined as a function parameter, a local variable, or a documented global. Any implementer copying this snippet will get a `NameError` at runtime, silently disabling the entire check. The function signature needs `N_blocks` as a required parameter alongside `page_table`, `seq_lens`, `block_size`, and `N_win`.

**Criterion:** (b) incorrect implementation.

---

### Issue 2 — Strategy A page-count upper bound stated as "exactly `ceil(w / block_size)`" then immediately contradicted

**File:** `paged_sdpa_and_windowing.md`, lines 107–127

**Error:** Line 107 states: "the op loads only the `ceil(w / block_size)` most recent pages." Lines 125–127 then correctly note that the count `p_high - p_low + 1` equals "`ceil(w / block_size)` (plus at most 1 extra page if the window boundary does not align to a block boundary)." The opener is therefore wrong for the misaligned case, where up to `ceil(w / block_size) + 1` pages are loaded. An implementer sizing the `start_page`-to-`end_page` gather range or a gather buffer from the line-107 figure would under-allocate by one block in the misaligned case.

**Criterion:** (a)/(b) wrong numerical answer / incorrect implementation.

---

### Issue 3 — `bwp`-from-`seq_len` derivation claim is false when Option 1 caps `seq_len` at `w`

**File:** `paged_sdpa_and_windowing.md`, lines 332–337

**Error:** The "Compatibility with the Existing Interface" section (which describes the host-side reordering path, Option 1) claims: "`seq_len` encodes both the total token count and the write pointer: `bwp = ((seq_len - 1) // block_size) % N_win`." However, Option 1 specifies `seq_len = min(T+1, w)` (line 435). At steady state (`T >= w`), `seq_len` is pinned at `w` regardless of `T`, so the formula always yields `bwp = ((w-1) // block_size) % N_win` — a constant that does not track the actual write-block pointer. The claim that the kernel "can derive `p_oldest` from `seq_len` and `N_win` without any new argument" is only true for Option 2 (kernel-native circular gather, where `seq_len = T+1`). Under Option 1 the pre-reordered page table makes `bwp` derivation unnecessary, but the text presents the derivation as working under the current (`seq_len = min(T+1, w)`) interface, which it does not.

**Criterion:** (b)/(c) implementer of a kernel extension following this description would produce incorrect gather ordering for all steady-state steps after the first full rotation.

---

No further issues found. Items 1–3 are the only correctness-affecting errors identified in this pass.

## Change Log (Pass 8)

**Date:** 2026-03-27

The following three corrections were applied to the Chapter 5 source files.

### Fix 1 — `check_window_invariants`: `N_blocks` added as explicit parameter

**File:** `eviction_and_page_reuse.md`, lines 144–158

**Change:** The function `check_window_invariants` asserted `page_table[b, p] < N_blocks`
but `N_blocks` was not present in the function signature or defined anywhere in scope,
causing a `NameError` at runtime and silently disabling the check.

`N_blocks` was added as a required parameter to the function signature:

```python
def check_window_invariants(page_table, seq_lens, block_size, N_win, N_blocks):
```

Callers must now pass the total number of physical blocks in the pool (i.e.,
`block_pool.shape[0]`). The assertion body is unchanged.

---

### Fix 2 — Strategy A opening claim corrected to worst-case page count

**File:** `paged_sdpa_and_windowing.md`, lines 107–109 (Core Idea paragraph)

**Change:** The original text stated the op loads only "the `ceil(w / block_size)` most
recent pages." Lines 125–127 correctly note that the count is `p_high - p_low + 1`,
which can equal `ceil(w / block_size) + 1` when the window boundary is not
block-aligned. The opening claim was therefore wrong for the misaligned case and would
cause implementers to under-size gather buffers by one block.

Corrected to: "between `ceil(w / block_size)` and `ceil(w / block_size) + 1` pages
depending on block-boundary alignment." This is now consistent with the derivation
in lines 125–127.

---

### Fix 3 — `bwp`-from-`seq_len` derivation scoped to Option 2 only

**File:** `paged_sdpa_and_windowing.md`, lines 333–338 (Compatibility with the Existing Interface)

**Change:** The Compatibility section claimed that `bwp = ((seq_len - 1) // block_size) % N_win`
allows the kernel to derive the write pointer from `seq_len` "without any new argument."
This is only valid under Option 2 (kernel-native circular gather), where `seq_len = T+1`.

Under Option 1, `seq_len = min(T+1, w)` is pinned at `w` at steady state, making
the formula return a constant that does not track the actual write-block pointer.

The paragraph was updated to:
- Explicitly scope the `bwp`-from-`seq_len` derivation to Option 2.
- Add a block-quoted warning explaining that under Option 1, `seq_len` is pinned at
  `w` at steady state, so the kernel cannot derive `bwp` from it. The host must
  either pass `bwp` as a new kernel argument, or (as Option 1 actually does) absorb
  the circular reordering into the page table directly so that the kernel never needs
  to know `bwp`.

## Pass 9

**Verified Pass 8 fixes:**
1. `check_window_invariants` signature now includes `N_blocks` — confirmed (eviction_and_page_reuse.md line 145).
2. Strategy A page count now reads "between `ceil(w/bs)` and `ceil(w/bs)+1`" — confirmed (paged_sdpa_and_windowing.md lines 108–109).
3. `bwp`-from-`seq_len` derivation scoped to Option 2 with Option 1 warning block — confirmed (paged_sdpa_and_windowing.md lines 333–347).

---

**Issue 1 — `check_window_invariants`: fill-phase guard skips too little and comment is actively misleading**

File: `eviction_and_page_reuse.md`, lines 148–150.

```python
if T < block_size:
    continue  # fill phase, invariants not yet fully applicable
```

Strategy B pre-allocates all `N_win` physical blocks at sequence start, so all `N_win` page-table entries are valid block indices from `T = 0` onward. The comment "invariants not yet fully applicable" is wrong: Invariant 1 (page-table validity) applies immediately. The guard `T < block_size` skips the check for only the first `block_size` steps, but the comment implies a broader fill-phase exemption. An implementer reading this will incorrectly believe that during the fill phase some page-table entries are legitimately unallocated, which is false under Strategy B. If they adapt this pattern to a lazy-allocation Strategy A variant (where pages really are allocated on demand), they will remove the guard too early and miss sentinel-entry violations. The guard should either be removed entirely (for Strategy B as written) or expanded to `T < N_win * block_size - 1` with an accurate comment. As written, it neither guards correctly for Strategy A nor documents the true invariant for Strategy B.

**Category: (c) — materially misleads the reader about when the page-table invariant holds.**

---

**Issue 2 — `paged_sdpa_and_windowing.md` Option 1 stale-prefix mask formula uses wrong modulus argument**

File: `paged_sdpa_and_windowing.md`, lines 468–472.

```
the stale prefix occupies assembled-tensor slots 0 through
(T - w + 1) mod block_size - 1
```

This formula is `t_low mod block_size - 1`, where `t_low = T - w + 1`. This gives the number of stale tokens as `t_low mod block_size`, which equals `(T - w + 1) mod block_size`. That is only correct when the reordering places the block containing `t_low` at assembled-tensor slot 0 — i.e., the formula silently assumes that `oldest_block_start = floor(t_low / block_size) * block_size`. Under Option 1 (host-side page-table reordering), `p_oldest = (T // block_size + 1) % N_win`, so the first assembled block starts at `oldest_block_start = ((T // block_size) - N_win + 1) * block_size`, which equals `floor(t_low / block_size) * block_size` only when `block_size` divides `w` exactly. The text immediately above this passage explicitly conditions on the divisible case (`N_win * block_size == w`), but the formula appears in the *non-divisible* context of the warning two sections later (lines 463–473 cover the general case). An implementer working in the non-divisible case who applies this formula will compute the wrong stale-prefix length and produce incorrect attention masks.

Strictly speaking, in the non-divisible case `n_stale = max(0, t_low - oldest_block_start)` (the general formula at line 536) should be used. The shorthand `(T - w + 1) mod block_size` is valid only when `block_size | w`; the text must state this restriction explicitly at the point of use.

**Category: (b) — an implementer in the non-divisible case will compute the wrong mask boundary.**

---

No further items meeting the scope threshold.

---

## Change Log (Pass 9)

**Date:** 2026-03-28

### Fix 1 — `eviction_and_page_reuse.md` ~line 148 — Fill-phase guard comment corrected

The comment `# fill phase, invariants not yet fully applicable` on the guard `if T < block_size: continue` was misleading. Under Strategy B all `N_win` physical blocks are pre-allocated at sequence start, so Invariant 1 (page-table validity) holds from `T = 0`; no fill phase exists where entries are uninitialized.

**Change:** Replaced the single-line comment with a multi-line comment that accurately states:
- Strategy B pre-allocates all `N_win` blocks so invariants hold from `T = 0`.
- This guard is only needed for lazy-allocation variants (Strategy A-style) where blocks are allocated on demand.
- Under Strategy B as written the guard is unnecessary and can be removed.

The `continue` statement itself was preserved so the runtime behavior is unchanged; only the documentation of its purpose was corrected.

### Fix 2 — `paged_sdpa_and_windowing.md` ~lines 468–472 — Stale-prefix shorthand restricted to divisible case

The shorthand `(T - w + 1) mod block_size - 1` for the end of the stale-prefix range was presented without qualification in a passage that also discusses the non-divisible case (`block_size` does not divide `w`). The shorthand is only valid when `w % block_size == 0`; in the non-divisible case it can yield a wrong mask boundary.

**Change:** Appended an inline restriction immediately after the shorthand formula:

> **(valid only when `block_size` divides `w` exactly, i.e. `w % block_size == 0`; for the non-divisible case use the general formula `max(0, t_low - oldest_block_start)` where `t_low = T - w + 1` and `oldest_block_start = ((T // block_size) - N_win + 1) * block_size`)**

No other prose was altered.

## Pass 10

### Pass 9 item verification

1. **`eviction_and_page_reuse.md` ~line 148 — fill-phase guard comment:** Lines 148–154 now read a multi-line comment stating that Strategy B pre-allocates all `N_win` blocks so invariants hold from T=0, that the guard is only needed for lazy-allocation variants, and that under Strategy B as written it is unnecessary. Confirmed correct.
2. **`paged_sdpa_and_windowing.md` ~lines 468–472 — stale-prefix shorthand annotated:** Lines 467–472 carry the inline restriction "(valid only when `block_size` divides `w` exactly, i.e. `w % block_size == 0`; for the non-divisible case use the general formula `max(0, t_low - oldest_block_start)` where `t_low = T - w + 1` and `oldest_block_start = ((T // block_size) - N_win + 1) * block_size`)". Confirmed present.

---

**Issue 1 — Fill-phase block-count formula `ceil(T / block_size)` is wrong at T=0 and at every multiple of `block_size`**

Files: `eviction_and_page_reuse.md` line 253; `paged_sdpa_and_windowing.md` lines 365–367.

Both locations state that a sequence at step T has `ceil(T / block_size)` allocated physical blocks during the fill phase. The correct count is `floor(T / block_size) + 1`: writing token T requires the block that contains it, which is block `floor(T / block_size)`, so blocks 0 through `floor(T / block_size)` — a total of `floor(T / block_size) + 1` — must exist.

`ceil(T / block_size)` equals `floor(T / block_size) + 1` only when `block_size` does not divide `T`. At T=0 it gives 0 (correct is 1); at T=64 with `block_size=64` it gives 1 (correct is 2). An implementer sizing the block pool or an allocator using this formula will under-count allocated blocks by one at every block boundary, producing allocation failures or pool-sizing errors.

**Criterion:** (a) wrong numerical answer, (b) incorrect implementation.

**Fix:** Replace `ceil(T / block_size)` with `floor(T / block_size) + 1` at both locations.

---

**Issue 2 — Memory Accounting section states steady-state in-use block count as a fixed `ceil(w / block_size)` — contradicts the corrected claim in `eviction_and_page_reuse.md` and is wrong when the window boundary is not block-aligned**

File: `paged_sdpa_and_windowing.md`, line 370.

The text reads: "After the window stabilises, the host frees `floor((T - w) / block_size)` blocks, leaving `ceil(w / block_size)` blocks in use — matching Strategy B's permanent allocation."

The claim of a fixed `ceil(w / block_size)` in-use count contradicts the corrected characterisation in `eviction_and_page_reuse.md` (Pass 1 Fix 2), which states the count is `floor(T/bs) - floor((T-w+1)/bs) + 1` and varies between `ceil(w/bs)` and `ceil(w/bs) + 1`. Concretely: at T=128, w=128, block_size=64, `p_low = floor(1/64) = 0`, `p_high = floor(128/64) = 2`, in-use = 3 = `ceil(128/64) + 1 = 3`. The text says 2. An implementer sizing the Strategy A block pool reservation from this line would under-allocate by one block in the misaligned case, causing allocation failures.

**Criterion:** (a) wrong numerical answer, (b) incorrect implementation.

**Fix:** Replace "leaving `ceil(w / block_size)` blocks in use" with "leaving between `ceil(w / block_size)` and `ceil(w / block_size) + 1` blocks in use (precisely `floor(T/bs) - floor((T-w+1)/bs) + 1`; worst-case pool reservation must use `ceil(w/bs) + 1`)". Remove the parenthetical "— matching Strategy B's permanent allocation" since the counts differ in the misaligned case.
## Change Log (Pass 10)

**Date:** 2026-03-28

The following corrections were applied to `eviction_and_page_reuse.md` and `paged_sdpa_and_windowing.md`.

### Fix 1 — Fill-phase in-use block count formula corrected from `ceil(T / block_size)` to `floor(T / block_size) + 1`

**Files:** `eviction_and_page_reuse.md` line 253; `paged_sdpa_and_windowing.md` lines 51 and 107 (Physical Layout and Strategy A Core Idea) and the Memory Accounting section.

**Change:** The formula `ceil(T / block_size)` was used to describe how many physical blocks a sequence occupies after generating `T` tokens. This formula is wrong at T=0 (gives 0; correct is 1) and at every positive multiple of `block_size` (gives one fewer than required). The correct count is `floor(T / block_size) + 1`: token T resides in block `floor(T / block_size)`, so blocks 0 through `floor(T / block_size)` — a total of `floor(T / block_size) + 1` — must be allocated.

Verification:
- T=0 → `floor(0/block_size) + 1 = 1` block. Correct (one block needed to write the first token).
- T=3, block_size=4 → `floor(3/4) + 1 = 1` block. Correct (tokens 0–3 fit in block 0).
- T=4, block_size=4 → `floor(4/4) + 1 = 2` blocks. Correct (token 4 requires block 1).

All occurrences of `ceil(T / block_size)` used as a fill-phase in-use block count were replaced with `floor(T / block_size) + 1`.

### Fix 2 — Memory Accounting section: steady-state in-use block count corrected from fixed `ceil(w / block_size)` to a range

**File:** `paged_sdpa_and_windowing.md`, Memory Accounting section (previously around line 370).

**Change:** The text previously stated that after the window stabilises the host frees older blocks "leaving `ceil(w / block_size)` blocks in use — matching Strategy B's permanent allocation." This is wrong when the window boundary is not block-aligned: the in-use count is `floor(T/bs) - floor((T-w+1)/bs) + 1`, which equals `ceil(w/bs)` when the window boundary falls on a block boundary but can equal `ceil(w/bs) + 1` otherwise. Concretely, at T=128, w=128, block_size=64 the formula gives 3 = `ceil(128/64) + 1`, not 2.

Corrected to: "leaving between `ceil(w / block_size)` and `ceil(w / block_size) + 1` blocks in use depending on block-boundary alignment (precisely `floor(T/bs) - floor((T-w+1)/bs) + 1` where `bs = block_size`). Worst-case pool reservation must use `ceil(w / block_size) + 1`." The parenthetical "— matching Strategy B's permanent allocation" was removed because the counts differ in the misaligned case.

## Pass 11

### Pass 10 item verification

1. **`floor(T / block_size) + 1` in all occurrences:** Confirmed at `paged_sdpa_and_windowing.md` lines 51, 107, and 365; `eviction_and_page_reuse.md` line 253. No remaining instance of `ceil(T / block_size)` as a fill-phase block count.
2. **Steady-state in-use count range:** `paged_sdpa_and_windowing.md` lines 369–372 reads "leaving between `ceil(w / block_size)` and `ceil(w / block_size) + 1` blocks in use … Worst-case pool reservation must use `ceil(w / block_size) + 1`". `eviction_and_page_reuse.md` lines 256–260 states the same range with the exact formula and the worst-case ceiling. Both confirmed.

---

### Issue 1 — Stale-prefix shorthand in lines 469–474 gives a nonzero count for the very case it claims to be valid for, contradicting the proof two pages later

**File:** `paged_sdpa_and_windowing.md`, lines 469–474 (the `Important` block-quoted warning inside Option 1).

**Error:** The warning states:

> "the stale prefix occupies assembled-tensor slots `0` through `(T - w + 1) mod block_size - 1` **(valid only when `block_size` divides `w` exactly, i.e. `w % block_size == 0`; ...)**"

The annotation "(valid only when `block_size` divides `w` exactly)" was added by Pass 9 Fix 2 to restrict the formula to the divisible case. However, in the divisible case the proof at lines 548–569 shows algebraically that `n_stale = max(0, r - bs + 1) = 0` for all steady-state `T`, where `r = T mod block_size`. The formula `(T - w + 1) mod block_size` is therefore nonzero whenever `(T - w + 1)` is not divisible by `block_size`, which is the common situation even when `block_size | w`.

Concrete counterexample using the chapter's own numbers (T=9, w=8, block_size=4, divisible):
- `(T - w + 1) mod block_size = (9 - 8 + 1) mod 4 = 2 mod 4 = 2`
- Formula claims stale prefix at assembled-tensor slots 0–1 (2 stale tokens)
- Lines 548–569 prove `n_stale = 0`; the worked example at lines 587–611 confirms "No stale-prefix masking needed"
- The assembled tensor at T=9 holds tokens {4,5,6,7,8,9,2,3}; slots 0–1 are tokens 4 and 5, which are valid in-window tokens

The formula in lines 469–474 thus directly contradicts the proof and the worked example in the same section for the exact case it claims to be valid for.

**Impact:** An implementer reading the `Important` warning and applying the formula for the divisible case will mask assembled-tensor slots 0 through `(T-w+1) mod block_size - 1` to `-inf`, suppressing valid in-window tokens (e.g., tokens 4 and 5 in the T=9 example). Attention output is wrong whenever `(T - w + 1) mod block_size != 0`, which occurs at the majority of decode steps.

**Fix:** Remove the stale-prefix formula from lines 469–474 entirely, or replace it with the correct statement: "When `block_size` divides `w`, `n_stale = 0` at all steady-state steps (proved in the Mask Construction section below); no stale-prefix masking is required. For the non-divisible case use the general formula `max(0, t_low - oldest_block_start)` where `t_low = T - w + 1` and `oldest_block_start = ((T // block_size) - N_win + 1) * block_size`."

---

## Change Log (Pass 11)

**File:** `paged_sdpa_and_windowing.md`, lines 469–474

**Issue fixed:** The stale-prefix shorthand formula `(T - w + 1) mod block_size - 1`, annotated as "valid only when `block_size` divides `w`", was incorrect for the very case it claimed to handle. Using the chapter's own numbers (T=9, w=8, block_size=4), the formula yields `2 mod 4 - 1 = 1` nonzero stale slots, while the algebraic proof at lines 548–569 and the worked example both confirm `n_stale = 0`. An implementer applying the formula would mask valid in-window tokens (e.g., tokens 4 and 5 in the T=9 example) to `-inf`, producing wrong attention output.

**Change made:** Removed the erroneous shorthand formula from the `Important` blockquote. Replaced it with the statement: "In the divisible case (`w % block_size == 0`), `n_stale = 0` at all steady-state steps — see the algebraic proof below. No stale-prefix masking is required." The general non-divisible formula `max(0, t_low - oldest_block_start)` is retained for the non-divisible case. The dangling ", and" connector left by the removal was also corrected to a period to restore grammatical flow.

## Pass 12

### Pass 11 item verification

**`paged_sdpa_and_windowing.md` lines 469–475:** Confirmed. The stale-prefix shorthand formula has been removed. The text now reads: "In the divisible case (`w % block_size == 0`), `n_stale = 0` at all steady-state steps — see the algebraic proof below. No stale-prefix masking is required." The general non-divisible formula `max(0, t_low - oldest_block_start)` follows. Fix is correct and complete.

---

### Issue 1 — Lines 312–316 claim stale slots in the write block must be masked to -inf; this is wrong at steady state in the divisible case and directly contradicts the worked example immediately above

**File:** `paged_sdpa_and_windowing.md`, lines 312–316.

**Error:** The paragraph states: "The stale slots in the currently-being-written block must be masked to `-inf` until they are overwritten."

In the divisible case (`block_size | w`) at steady state, the "not-yet-overwritten" slots of the currently-being-written block hold tokens from the previous rotation that fall *inside* the current window. The algebraic proof at lines 548–569 shows `n_stale = max(0, r - bs + 1) = 0` always, and the worked example at lines 588–611 confirms that for T=9, w=8, block_size=4: phys_A slots 2–3 contain tokens 2 and 3, which are valid in window [2, 9]. The example (lines 304–306) itself labels those slots "valid, inside window [2,9]."

The sentence at line 312 directly contradicts both: it says those same slots must be masked to -inf until overwritten, which would suppress tokens 2 and 3 from the attention computation and produce wrong output. A reader implementing from lines 312–316 without reading the Mask Construction section would mask valid in-window tokens at the majority of steady-state decode steps.

Concrete check: T=9, w=8, block_size=4. phys_A (bwp=0) slots 2,3 = tokens 2,3. Window = [2,9]. Masking slots 2,3 to -inf removes tokens 2 and 3 from attention. Wrong.

**Fix:** Replace lines 312–316 with a statement that distinguishes fill phase from steady state: during the fill phase (T < N_win * block_size), unwritten trailing slots must be masked via `seq_len = T+1`; at steady state in the divisible case, no stale-prefix or partial-block masking is needed because the unwritten slots of the write block contain valid previous-rotation tokens inside the window (n_stale = 0, proved in the Mask Construction section).

## Change Log (Pass 12)

**Date:** 2026-03-28

**File revised:** `ch5_paged_kv_cache/paged_sdpa_and_windowing.md`

**Lines affected:** 312–316 (formerly incorrect masking instruction for write-block unwritten slots)

**Change summary:**

Removed the incorrect blanket instruction "The stale slots in the currently-being-written block must be masked to `-inf` until they are overwritten." This statement was wrong for the divisible case (`w % block_size == 0`) at steady state, where the unwritten slots of the write block hold tokens from exactly one full window rotation ago — tokens that are provably within the current window (`n_stale = 0`, as shown in the algebraic proof). Masking them would suppress valid in-window tokens and corrupt attention scores.

Replaced with two scoped statements:

1. **Divisible case** (`w % block_size == 0`): Unwritten slots of the write block contain valid previous-rotation tokens and must NOT be masked. A reference to the algebraic `n_stale = 0` proof is included.
2. **Non-divisible case only** (`w % block_size ≠ 0`): Some tail slots of the write block may fall outside the window and require masking to `-inf`.

This correction ensures that readers implementing from this section will not incorrectly mask valid tokens in the common divisible case.

## Pass 13

### Pass 12 item verification

**`paged_sdpa_and_windowing.md` lines 312–316:** The write-block masking instruction is now correctly scoped. The divisible case (`w % block_size == 0`) states that unwritten slots must NOT be masked; the non-divisible case states that some tail slots may require masking to `-inf`. Confirmed correct per the Pass 12 fix description.

---

### Issue 1 — Comparison table row for Strategy A "Physical blocks per sequence" understates the steady-state maximum by one block

**File:** `paged_sdpa_and_windowing.md`, line 360 (Interface Compatibility table).

**Error:** The table entry reads:

```
| Physical blocks per sequence | Grows then stabilises at `ceil(w/block_size)` | Fixed at `N_win` from the start |
```

The steady-state in-use count under Strategy A is not a fixed `ceil(w/block_size)`; it is `floor(T/bs) - floor((T-w+1)/bs) + 1`, which equals either `ceil(w/bs)` or `ceil(w/bs) + 1` depending on whether the window boundary falls on a block boundary. This was corrected in the Memory Accounting prose (lines 369–375) and in `eviction_and_page_reuse.md` (lines 256–260) in Pass 10, but the comparison table was not updated and still shows the lower (and sometimes-wrong) fixed value.

Concrete counterexample: w=128, block_size=64. At T=128, p_low=floor(1/64)=0, p_high=floor(128/64)=2, live count = 3 = `ceil(128/64) + 1 = 3`. The table says 2.

An implementer sizing the block pool for Strategy A from the table will reserve `N_win = ceil(w/block_size)` blocks per sequence and encounter allocation failures at every step where the window boundary is not block-aligned.

**Fix:** Replace "Grows then stabilises at `ceil(w/block_size)`" with "Grows then stabilises at `ceil(w/block_size)` to `ceil(w/block_size) + 1` (worst-case: `ceil(w/block_size) + 1`)".

---

No further issues found. With the table correction above applied, the chapter is internally consistent across all three files.

---

## Change Log (Pass 13)

**File:** `paged_sdpa_and_windowing.md`, line 360 (Interface Compatibility table, Strategy A row "Physical blocks per sequence").

**Change:** The cell previously read:

```
Grows then stabilises at `ceil(w/block_size)`
```

This contradicts the corrected Memory Accounting prose (lines 373–376) which correctly states the steady-state live block count ranges between `ceil(w/block_size)` and `ceil(w/block_size) + 1`, with the worst-case being `ceil(w/block_size) + 1`.

**Fix applied:** Updated the cell to read:

```
Grows to `ceil(w/block_size)` – `ceil(w/block_size) + 1`; worst-case `ceil(w/block_size) + 1`
```

The table is now consistent with the prose in the Memory Accounting section and with the corrections made in prior passes to `eviction_and_page_reuse.md`.

## Pass 14

### Pass 13 item verification

**`paged_sdpa_and_windowing.md` line 360, Strategy A "Physical blocks per sequence":** The cell now reads "Grows to `ceil(w/block_size)` – `ceil(w/block_size) + 1`; worst-case `ceil(w/block_size) + 1`". Confirmed correct per the Pass 13 fix.

---

### Issue 1 — Physical Layout section gives a wrong block-count formula when T is a multiple of block_size

**File:** `paged_sdpa_and_windowing.md`, line 51.

**Error:** The sentence reads:

> "A sequence currently holding `T` tokens occupies `floor(T / block_size) + 1` physical blocks."

Here T is used as a token count (the surrounding sentence says "holding `T` tokens"). For a token count T, the correct number of physical blocks is `ceil(T / block_size)`. The formula `floor(T / block_size) + 1` equals `ceil(T / block_size)` only when T is not a multiple of block_size. When T IS a multiple of block_size the formula overcounts by 1:

- T = 64, block_size = 64: formula gives floor(64/64)+1 = 2, but 64 tokens fill exactly 1 block → correct answer is 1.
- T = 128, block_size = 64: formula gives floor(128/64)+1 = 3, correct answer is ceil(128/64) = 2.

A reader who implements memory sizing from this sentence will reserve one extra block per sequence at every step where T is a multiple of block_size, producing an inflated pool-size estimate. Combined with the pool-sizing guidance later in `eviction_and_page_reuse.md`, this could also cause confusion with the `floor(T / block_size) + 1` formula used correctly elsewhere in the file (where T is a 0-indexed token position, not a count).

**Fix:** Replace "A sequence currently holding `T` tokens occupies `floor(T / block_size) + 1` physical blocks." with "A sequence currently holding `T` tokens occupies `ceil(T / block_size)` physical blocks." (Equivalently: `floor((T - 1) / block_size) + 1` for T ≥ 1, but `ceil(T / block_size)` is cleaner and unambiguous.)


## Change Log (Pass 14)

**File:** `paged_sdpa_and_windowing.md`, line 51 (Physical Layout section).

**Issue:** The sentence "A sequence currently holding `T` tokens occupies `floor(T / block_size) + 1` physical blocks" used T ambiguously. The word "holding `T` tokens" implied T was a count, which would make the correct formula `ceil(T / block_size)`. The Pass 14 review entry in this file argued for that change. However, that analysis was incorrect.

**T convention determination:** T is used as a **0-based position index** (the position of the current decode step / token being written) throughout the entire file:
- Line 427 in the Python snippet: `T = seq_lens[b] - 1  # 0-indexed position of most recent token`
- `seq_len = T+1` usage (lines 339, 345) confirms T is 0-based (T+1 = total tokens written)
- All decode examples: T=0 writes the first token, T=4 writes the fifth token
- Formulas `bwp(T) = floor(T/block_size) mod N_win` and `offset_in_block(T) = T mod block_size` only make sense with 0-based T

**Correct formula for 0-based T:** `floor(T / block_size) + 1`
- T=0 → floor(0/bs)+1 = 1 block (correct: 1 token written, occupies 1 block)
- T=block_size-1 → floor((bs-1)/bs)+1 = 1 block (correct: bs tokens, fills 1 block exactly)
- T=block_size → floor(bs/bs)+1 = 2 blocks (correct: bs+1 tokens, spills into second block)

**Fix applied:** Kept `floor(T / block_size) + 1` (it was already correct) and rewrote the surrounding prose to make the 0-based convention explicit: "At decode step `T` (0-based position index of the token being written, so `T+1` tokens have been stored), the sequence occupies `floor(T / block_size) + 1` physical blocks."

The Prior Pass 14 entry proposing `ceil(T / block_size)` was based on misreading T as a count; that change would have introduced an off-by-one error at every T that is a multiple of block_size (undercounting by 1 block).

## Pass 15

### Issue 1 — `T_rounded` formula wrong for T that are exact multiples of `block_size`

**File:** `paged_sdpa_and_windowing.md`, line 82.

**Error:** The formula given is:

```
T_rounded = ceil(T / block_size) * block_size
```

When T is an exact multiple of `block_size` (e.g. T=64, block_size=64), `ceil(T / block_size) = T / block_size`, so `T_rounded = T`. This is one block short of the correct value. At T=64 there are 65 tokens stored spanning 2 complete blocks, so `T_rounded` should be 128, not 64. The kernel would assemble a KV tensor of shape `[1, H_kv, 64, d]` instead of `[1, H_kv, 128, d]`, silently dropping the entire current block from the attention computation.

The correct formula consistent with the `floor(T / block_size) + 1` block count established on line 52 is:

```
T_rounded = (floor(T / block_size) + 1) * block_size
```

Verification: T=200 → (3+1)*64 = 256 (unchanged, matches the diagram). T=64 → (1+1)*64 = 128 (correct). T=0 → (0+1)*64 = 64 (correct; `ceil(0/64)*64 = 0` is wrong).

**Impact:** An implementer deriving `T_rounded` from the text formula would produce a short gather at every T that is a multiple of `block_size`, omitting the KV vectors for the current block. The masking description ("masks out positions 201–255") is computed from the example T=200 which is not a multiple of 64, so the surrounding prose does not expose the error.

**Fix:** Replace `ceil(T / block_size) * block_size` with `(floor(T / block_size) + 1) * block_size` on line 82.

---

No further issues. Chapter approved subject to the fix above.

## Change Log (Pass 15)

**File:** `paged_sdpa_and_windowing.md`, line 81

**Change:** Replaced incorrect `T_rounded` formula.

- Before: `T_rounded = ceil(T / block_size) * block_size`
- After: `T_rounded = (floor(T / block_size) + 1) * block_size`

**Rationale:** The `ceil` form underestimates `T_rounded` when T is an exact multiple of `block_size` (e.g. T=64, block_size=64 → `ceil` gives 64, but the correct gather size is 128). The corrected formula is consistent with the block count `floor(T / block_size) + 1` established at line 52, ensuring the full current block is always included in the gather.

## Pass 16

### Item 1 — Pool example says "one more sequence" when two more can fit

**File:** `eviction_and_page_reuse.md`, approximately line 192

**Error:** The pool state example has N_pool=20, N_win=4, with three active sequences occupying blocks 0–11. The comment reads:

```
Blocks 12–19: free  (can accommodate one more sequence: 20 - 12 = 8 ≥ 4)
```

8 free blocks with N_win=4 can accommodate floor(8/4) = **2** more sequences, not 1. The formula `B_max = floor(N_pool / N_win) = floor(20/4) = 5` stated one line above the example is correct, and 5 - 3 = 2 remaining slots confirms the parenthetical is wrong. A reader using this annotated example to reason about pool headroom would conclude half the available capacity is unavailable.

**Fix:** Change "one more sequence" to "two more sequences": `(can accommodate two more sequences: floor(8/4) = 2)`.

---

### Item 2 — Pass 15 fix confirmed

**File:** `paged_sdpa_and_windowing.md`, line 81

`T_rounded = (floor(T / block_size) + 1) * block_size` is present and correct. Pass 15 item verified.

---

No further issues beyond Item 1. Chapter approved subject to the fix above.

## Change Log (Pass 16)

**File:** `eviction_and_page_reuse.md`, line 192

Corrected pool state example: changed "can accommodate one more sequence" to "can accommodate two more sequences". With 8 free blocks and N_win=4, `floor(8/4) = 2` sequences can be accommodated, not 1.

## Pass 17

Pass 16 item verified: `eviction_and_page_reuse.md` line 192 reads "two more sequences" (8 free blocks / N_win=4 = 2). Confirmed correct.

All numerical examples, formulas, and derivations were checked:

- Strategy A diagram (T=200, w=128, block_size=64): t_low=73, p_low=1, p_high=3, assembled 192 slots, padding masked 201–255. All correct.
- Block-boundary misalignment example (w=100, block_size=64, T=200): t_low=101, 37 wasted tokens. Correct.
- Strategy B circular example (w=8, block_size=4, N_win=2, T=9): bwp=0, p_oldest=1, oldest_block_start=4, t_low=2, n_stale=0. All correct.
- Memory per block and B_max calculations in eviction_and_page_reuse.md: arithmetic verified correct.
- Fill-phase guard condition in reordering code (`T < N_win * block_size`) is consistent with the steady-state condition (`T >= N_win * block_size`) described in the accompanying prose.

No feedback — chapter approved.

## Pass 18 (post-compression)

### Verification of all prior fixes

All corrections from Passes 1–17 were re-checked against the current post-compression source files:

1. **`T_rounded` formula** (`paged_sdpa_and_windowing.md` line 79): `(floor(T / block_size) + 1) * block_size`. Present and correct.
2. **Strategy A page count** (lines 107–108): "between `ceil(w / block_size)` and `ceil(w / block_size) + 1` pages". Present and correct.
3. **Memory Accounting steady-state in-use count** (lines 358–362): range formula and worst-case `ceil(w/bs) + 1` pool reservation. Present and correct.
4. **Comparison table Physical blocks (Strategy A)** (line 345): "Grows to `ceil(w/block_size)` – `ceil(w/block_size) + 1`; worst-case `ceil(w/block_size) + 1`". Present and correct.
5. **Pool example two-sequence capacity** (`eviction_and_page_reuse.md` line 186): "two more sequences". Present and correct.
6. **`check_window_invariants` fill-phase guard comment** (lines 144–149): multi-line comment accurately scoping the guard to lazy-allocation variants. Present and correct.
7. **Write-block masking scoped by divisibility** (`paged_sdpa_and_windowing.md` lines 298–306): divisible case states "must NOT be masked"; non-divisible case gates masking on tail slots. Present and correct.
8. **`bwp`-from-`seq_len` derivation scoped to Option 2** (lines 323–336): block-quoted warning for Option 1 steady-state pin. Present and correct.
9. **Physical Layout T convention** (line 51): "0-based position index of the token being written, so `T+1` tokens have been stored". Present and correct.
10. **`bwp`-from-`seq_len` Option 1 warning block** (lines 330–336): correctly explains that under Option 1, `seq_len` is pinned at `w` at steady state and the formula is inapplicable. Present and correct.

---

No new issues found after compression. No content was removed or altered in a way that reintroduces previously corrected errors, and no new numerical, implementation, or conceptual errors were introduced.

No feedback — chapter approved.
