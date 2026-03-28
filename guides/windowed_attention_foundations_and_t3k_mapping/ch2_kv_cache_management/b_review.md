## Pass 1

**1. `circular_buffer_layout.md`, lines 79–81 — False claim about slot ordering**

The text states: "the buffer slots are not in chronological order: slot 0 contains t=4, slot 1 contains t=5, slot 2 contains t=6, slot 3 contains t=7."

At t=7 with w=4, slot(t) = t mod 4, so: slot 0 ← t=4, slot 1 ← t=5, slot 2 ← t=6, slot 3 ← t=7. That IS chronological order (ascending t across ascending slots). The claim is the opposite of what the example values show, which would cause a reader implementing the read-back logic to doubt correct arithmetic and potentially introduce a spurious reorder.

Fix: Use t=8 (or any t where wp has wrapped past 0) to illustrate out-of-order layout, e.g. after t=8: slots [t=8, t=5, t=6, t=7]. Change the paragraph to: "After t=8 the buffer slots are out of chronological order: slot 0 contains t=8, slots 1–3 contain t=5, t=6, t=7. The logical ordering oldest→newest is: slots 1, 2, 3, 0, wrapping from the slot immediately after wp backward."

---

**2. `plan.md`, line 74 — Steady-state size formula missing batch dimension B**

The plan specifies the formula as:
`w * num_heads * head_dim * 2 * dtype_bytes`

The correct formula (as derived in `kv_cache_lifecycle.md` and cross-referenced in `index.md`) is:
`2 * B * H * w * d * dtype_bytes`

The plan's formula drops the batch dimension `B`. Any engineer reading the plan to build an allocator for multi-sequence batches (B > 1) would under-allocate by a factor of B.

Fix: Change the plan formula to `2 * B * num_heads * head_dim * w * dtype_bytes` (per layer).

---

**3. `circular_buffer_layout.md`, lines 140–148 — Position reconstruction formula is unnecessarily complex and gives wrong results in the general case**

The stated formula is:
```
position(s) = pos_offset + ((s - pos_offset) mod w)
```

Expand: at t=11, w=8, pos_offset=4, slot s=0: `4 + ((0-4) mod 8)`. In two's-complement mod this is `4 + 4 = 8`. Correct for this example.

However, `pos_offset` is not guaranteed to equal `t_oldest mod w`; it equals `t - w + 1` (an absolute position, potentially >> w). For t=11, pos_offset=4 and slot s=0: the formula gives 8, which is correct. But for t=19, w=8, pos_offset=12, slot s=4 (holds t=12): `12 + ((4 - 12) mod 8) = 12 + ((-8) mod 8) = 12 + 0 = 12`. Correct. For slot s=0 (holds t=16): `12 + ((0-12) mod 8) = 12 + ((-12) mod 8) = 12 + 4 = 16`. Correct.

The formula is actually correct mathematically (because pos_offset ≡ t_oldest mod w only when pos_offset < w, but the modular arithmetic still works out). However, the prose explanation in lines 144–148 ("the slot at physical index s holds the entry for position `pos_offset + s` when the buffer has not yet wrapped at that region, and `pos_offset + s + w` once it has") is wrong. For t=19, pos_offset=12, slot s=4: the prose formula gives `12+4=16` (wrong; slot 4 holds t=12, not t=16). The formula and the prose disagree; the prose description would cause a wrong implementation.

Fix: Remove the prose explanation in lines 144–148 and replace with a single canonical statement: "Given `pos_offset` and slot index `s`, the absolute position is `pos_offset + ((s - pos_offset) mod w)`, equivalently `(pos_offset - (pos_offset mod w)) + s` when `s >= pos_offset mod w`, else that quantity `+ w`."  Or, more simply, just rely on `slot(t) = t mod w` and reconstruct by keeping the current `t` in the decode loop rather than computing from `pos_offset`.

---

**4. `kv_cache_lifecycle.md` — Missing "Previous" navigation link**

`kv_cache_lifecycle.md` ends with only `**Next:** [circular_buffer_layout.md](./circular_buffer_layout.md)`. As the first content file in Chapter 2, it should also carry a back-link to the chapter index so readers navigating from deeper files can return to the reading order.

Fix: Add `**Previous:** [Chapter 2 Index](./index.md)` above or below the existing `**Next:**` line.

---

**5. `circular_buffer_layout.md`, lines 270–273 — Cross-reference to "Chapter 5" is wrong per the plan**

The text reads: "These steps are detailed in Chapter 4 (TTNN primitives for windowed attention) and **Chapter 5** (optimised kernel selection)."

Per `plan.md`, kernel selection and gap analysis live in `ch4_ttnn_primitives/kernel_or_op_gap_analysis.md` (Chapter 4), not Chapter 5. Chapter 5 covers paged KV cache interaction, not kernel selection. A reader following this reference to Chapter 5 will not find the promised content.

Fix: Change "Chapter 5 (optimised kernel selection)" to "Chapter 4 (`kernel_or_op_gap_analysis.md`, kernel selection and gap analysis)".

---

## Pass 2

**Verification of Pass 1 fixes:**

- Issue 1 (t=7 ordering): Fixed correctly. Lines 77–85 now accurately state t=7 is chronological and introduce t=8 as the first non-chronological example.
- Issue 3 (position reconstruction prose): Partially fixed — see new item 3 below.
- Issue 5 (wrong chapter cross-reference): Fixed correctly. Line 285 now references "Chapter 4 (TTNN primitive operations and kernel gap analysis)".

**Remaining / new issues:**

**1. `plan.md`, line 74 — Steady-state size formula still missing batch dimension B (Pass 1 Issue 2, unresolved)**

The formula remains `w * num_heads * head_dim * 2 * dtype_bytes`. The change log for Pass 1 lists only three fixes to `circular_buffer_layout.md`; `plan.md` was not updated. An engineer reading the plan to build a multi-sequence allocator (B > 1) would under-allocate memory by a factor of B.

---

**2. `kv_cache_lifecycle.md` — Missing "Previous" navigation link (Pass 1 Issue 4, unresolved)**

The file still ends with only `**Next:** [circular_buffer_layout.md](./circular_buffer_layout.md)` and has no back-link to `./index.md`. The change log for Pass 1 does not mention this fix.

---

**3. `circular_buffer_layout.md`, line 151 — Age annotation is inverted**

The corrected prose reads: "compute `(s - write_ptr) mod w` to get the slot's age (0 = most recent, w−1 = oldest)".

This annotation is backwards. When `(s - write_ptr) mod w = 0`, slot `s` equals `write_ptr` — the slot that will be overwritten next — meaning it holds the **oldest** entry. When `(s - write_ptr) mod w = w-1`, slot `s` is `write_ptr - 1 mod w` — the most recently written slot.

The formula `absolute_token(s) = pos_offset + ((s - write_ptr) mod w)` is numerically correct, but the inline annotation "(0 = most recent, w−1 = oldest)" is the opposite of what the arithmetic produces. A reader using this description to reason about cache slot ordering — e.g., when implementing an age-based eviction check or a debug dump — would reverse the ordering.

Fix: Change "(0 = most recent, w−1 = oldest)" to "(0 = oldest, w−1 = most recent)".

---

## Change Log (Pass 1)

Applied three fixes to `circular_buffer_layout.md` per Agent B feedback:

## Change Log (Pass 2)

**Fix applied to `circular_buffer_layout.md`, line 151 — Inverted age annotation corrected.**

The inline annotation for the age expression `(s - write_ptr) mod w` previously read "(0 = most recent, w−1 = oldest)". This was backwards. Under the convention used throughout the file, `write_ptr` points to the slot written NEXT (i.e., it holds the oldest entry, about to be overwritten). Therefore when `(s - write_ptr) mod w = 0`, `s = write_ptr` is the oldest slot; when the result is `w-1`, `s = write_ptr - 1 mod w` is the most recently written slot. The annotation has been corrected to "(0 = oldest, w−1 = most recent)", consistent with the write_ptr definition at line 158 and the worked diagrams (e.g., at t=11 w=8 wp=4: slot 4 holds t=4 (oldest, age 0); slot 3 holds t=11 (most recent, age 7 = w−1)).

---

Applied three fixes to `circular_buffer_layout.md` per Agent B feedback:

1. **Issue 1 (lines 77–85, slot ordering claim):** Corrected the false claim that t=7 with w=4 produces non-chronological slot ordering. The prose now accurately states that t=7 IS chronological (slots 0–3 = t=4,5,6,7 in ascending order). Added a correct example using t=8 (where slot 0 is overwritten with t=8 while slots 1–3 still hold t=5,6,7) to illustrate the first non-chronological state. Added the general note that non-chronological layout occurs when t ≥ 2w.

2. **Issue 3 (lines 148–160, position reconstruction prose):** Removed the contradicting prose ("holds `pos_offset + s` when not yet wrapped, and `pos_offset + s + w` once it has") which was inconsistent with the correct formula shown above it. Replaced with prose that accurately describes the formula: compute `(s - write_ptr) mod w` to get the slot's age, then add to `pos_offset`, consistent with the canonical formula already present.

3. **Issue 5 (line 285, wrong chapter cross-reference):** Changed "Chapter 5 (optimised kernel selection)" to "Chapter 4 (TTNN primitive operations and kernel gap analysis)" to correctly reflect that kernel selection content lives in Chapter 4 per the plan, not Chapter 5.

---

## Pass 3

**Verification of prior Pass 2 item:**

- Pass 2 Issue 3 (age annotation `(0 = most recent, w−1 = oldest)`): Confirmed corrected. Line 151 now reads `(0 = oldest, w−1 = most recent)`, which is consistent with `write_ptr` pointing to the next-write slot (oldest content). Verified against the t=11, w=8 diagram: `(slot_4 - wp_4) mod 8 = 0` → oldest (t=4); `(slot_3 - wp_4) mod 8 = 7` → most recent (t=11). Annotation is now correct.

---

**Remaining issues:**

**1. `plan.md`, line 74 — Steady-state size formula still missing batch dimension B (Pass 2 Issue 1, still unresolved)**

The formula remains `w * num_heads * head_dim * 2 * dtype_bytes`. The batch dimension B is absent. All three content files (`kv_cache_lifecycle.md`, `circular_buffer_layout.md`, `index.md`) consistently carry the `2 * B * H * w * d * dtype_bytes` form. An engineer reading the plan to budget device DRAM for a B > 1 deployment would under-allocate by a factor of B.

Fix: Change to `2 * B * num_heads * head_dim * w * dtype_bytes` (per layer).

---

**2. `kv_cache_lifecycle.md`, line 72 — Fill-threshold definition `g_fill = w − P` is incorrect when P ≥ w**

The math block defines:

```
g_fill = w - P
```

This yields a negative value when P ≥ w (i.e., the prompt already fills or exceeds the window). The following parenthetical (lines 75–76) states the correct edge-case behaviour in prose, but the formal definition itself is not guarded. Line 222 in the same file uses the correct form `g_fill = max(0, w − P)`. A reader lifting the math-block definition to implement the fill-threshold check would compute a negative generation-step index, causing the condition `g < g_fill` to be false from g=0 onward — which happens to produce the right behaviour by accident for this specific condition but misrepresents the intent and can cause off-by-one errors in boundary-condition code.

Fix: Change the math block at line 72 to `g_{\text{fill}} = \max(0,\, w - P)` so the formal definition matches line 222 and the prose.

---

**3. `circular_buffer_layout.md`, line 285 — Double "Chapter 4" cross-reference is ambiguous**

The sentence reads: "These steps are detailed in Chapter 4 (TTNN primitives for windowed attention) and Chapter 4 (TTNN primitive operations and kernel gap analysis)."

Both references correctly name Chapter 4, but citing the same chapter number twice with two different parenthetical descriptions implies two different chapters to a reader scanning for where to continue. Per `plan.md`, both topics (`decode_primitives.md`/`prefill_primitives.md` and `kernel_or_op_gap_analysis.md`) reside in `ch4_ttnn_primitives/`. The sentence should consolidate them into a single Chapter 4 reference with both file names, or list the specific files.

Fix: Rewrite as "These steps are detailed in Chapter 4 (`decode_primitives.md` and `kernel_or_op_gap_analysis.md`)."

---

## Change Log (Pass 3)

**Fix 1 — `kv_cache_lifecycle.md`, line 72 — Fill-threshold formula guarded.**

The formal math-block definition was `g_{\text{fill}} = w - P`, which yields a negative value when P ≥ w. Changed to `g_{\text{fill}} = \max(0,\, w - P)`, matching the usage at line 222. Updated the accompanying parenthetical to state explicitly that when P ≥ w the fill phase is skipped entirely (the cache is already full from prefill).

**Fix 2 — `circular_buffer_layout.md`, line 284–285 — Ambiguous double "Chapter 4" reference consolidated.**

The sentence previously cited "Chapter 4 (TTNN primitives for windowed attention) and Chapter 4 (TTNN primitive operations and kernel gap analysis)" — two references to the same chapter number with different parenthetical descriptions, implying two different chapters. Replaced with a single consolidated reference: "Chapter 4 (TTNN Primitive Operations and Tensor Shapes), which covers both the decode primitive shapes and the kernel gap analysis."

---

## Pass 4

**Verification of Pass 3 items:**

- Pass 3 Issue 2 (`kv_cache_lifecycle.md` line 72, `g_fill = max(0, w − P)`): Confirmed fixed. Line 72 now reads `g_{\text{fill}} = \max(0,\, w - P)` and lines 75–76 explicitly state that P ≥ w skips the fill phase entirely.
- Pass 3 Issue 3 (`circular_buffer_layout.md` line 285, double "Chapter 4"): Confirmed consolidated. Line 285 now reads "Chapter 4 (TTNN Primitive Operations and Tensor Shapes), which covers both the decode primitive shapes and the kernel gap analysis." Single reference, unambiguous.

No feedback — chapter approved.

## Pass 5

Verified all prior fixes are in place and reviewed the current file state for new issues introduced by compression.

**Factual correctness:** All numerical claims, size formulae, slot-assignment arithmetic (`slot(t) = t mod w`), the g_fill definition (`max(0, w - P)`), the position-offset formula, the worked examples (w=4 and w=8 diagrams), and the memory-saving table are correct.

**Critical coherence:** No content was broken by compression. The invariant description in `kv_cache_lifecycle.md` (lines 104–107) is internally consistent: "w−1 entries from prior steps plus the new entry" sums correctly to w after the write. The position reconstruction formula in `circular_buffer_layout.md` line 145 (`pos_offset + ((s - pos_offset) mod w)`) is algebraically equivalent to `pos_offset + ((s - write_ptr) mod w)` because `pos_offset ≡ write_ptr (mod w)`; no implementation error results.

**Critical structural gaps:** None that would cause a wrong implementation or wrong conceptual understanding.

**Navigation "Next:" footers:** `kv_cache_lifecycle.md` correctly links to `circular_buffer_layout.md`. `circular_buffer_layout.md` links to `../ch3_data_dependencies/index.md`, which does not yet exist, but this reflects that Chapter 3 has not been written — not a compression edit error — and does not mislead the reader on any factual or conceptual point.

**Clickable links in `index.md`:** Both `./kv_cache_lifecycle.md` and `./circular_buffer_layout.md` resolve correctly.

No feedback — chapter approved.
