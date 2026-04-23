# Compression Analysis: Chapter 2 — Op Shape Contract — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: 440 lines (index.md: 88, shape_validation_in_invoke.md: 139, kernel_rotate_half_pairing.md: 109, what_the_golden_function_reveals.md: 104)
- Estimated post-compression line count: ~350 lines
- Estimated reduction: ~20%

---

## CRUCIAL Suggestions

### Partial RoPE math block in index.md duplicates Ch1 verbatim
**Issue:** `index.md` lines 64–70 ("Recap of Chapter 1 Prerequisites") reproduce the three-case piecewise formula for partial RoPE and the `nearest_32` definition in full. This is a near-verbatim copy of content from `ch1_rope_fundamentals/partial_rope_math.md` (worked example section) and `ch1_rope_fundamentals/tile_alignment_in_ttnn.md`. The "mismatch" bullet at line 70 also pre-states the conclusion of `shape_validation_in_invoke.md` §4, spoiling the finding before the reader reaches it.
**Suggestion:** Collapse the entire "Recap of Chapter 1 Prerequisites" section to a two-sentence forward reference and a link. Keep only the symbolic names (`rotary_dim`, `head_dim`, `nearest_32`) so a skimming reader can orient themselves; remove the formula, the tile-alignment definition, and the pre-stated conclusion.
**Passage to cut/replace:**
```
## Recap of Chapter 1 Prerequisites

Chapter 1 established the following facts that this chapter builds on directly:

- **Partial RoPE math:** Standard RoPE rotates all `head_dim` elements. Partial RoPE rotates only the first `rotary_dim` elements and copies the remaining `head_dim - rotary_dim` elements unchanged. This is expressed as:

  $$x'_i = \begin{cases} ... \end{cases}$$

- **Tile alignment:** Tenstorrent hardware operates on $32 \times 32$ tiles. ... For `rotary_dim=48`, $\text{nearest\_32}(48) = 64$.

- **The mismatch:** When `rotary_dim=48` and `head_dim=128`, naively padding cos/sin to shape `[..., 64]` and passing them to `ttnn.experimental.rotary_embedding` will be rejected by a `TT_FATAL` ...
```
Replace with a single recap sentence, e.g.:  
> "This chapter assumes familiarity with partial RoPE math, tile alignment, and `nearest_32` from Chapter 1. The key starting point: for `rotary_dim=48, head_dim=128`, any cos/sin with `shape[-1] != 128` will be rejected — Chapter 2 explains exactly why."

---

### `validate` function described twice: index.md and shape_validation_in_invoke.md
**Issue:** `index.md` lines 43–45 (data-flow diagram annotation) explicitly name `RotaryEmbeddingDeviceOperation::validate` and annotate it as "repeated gate." `shape_validation_in_invoke.md` §2 then provides the full C++ code block and a prose explanation of the same fact. There is no information in the index.md diagram annotation that is not fully developed in the downstream file — but the annotation in the diagram pre-reads the punchline ("repeated gate") as a label before the reader opens the file.
**Suggestion:** In the data-flow diagram, change the `validate` annotation from `TT_FATAL: cos.padded_shape()[-1] == X  ← repeated gate` to just `TT_FATAL: cos.padded_shape()[-1] == X` (drop the parenthetical editoralization). The file `shape_validation_in_invoke.md` correctly explains that it is a repetition — the diagram need not pre-editorialize.
**Passage to cut/replace:**
```
  │  TT_FATAL: cos.padded_shape()[-1] == X  ← repeated gate
```
Replace with:
```
  │  TT_FATAL: cos.padded_shape()[-1] == X
```

---

## MINOR Suggestions

### shape_validation_in_invoke.md, §4 Concrete Failure Scenario (~lines 112–136): closing note restates §1b conclusion
**Issue:** The closing `> Note:` at lines 134–136 ("The `rotary_dim` parameter accepted by the Python API is passed through to the kernel as a tile count...") introduces new nuance (kernel tile-count usage), but the sentence "The shape of cos/sin must still be `head_dim` wide regardless of `rotary_dim`" is the fourth time this statement appears in this one file (intro, §1b Key Finding, §3 Warning, now §4 Note). The new information (tile-count forwarding, tile-alignment caveat) should stay; the restatement of the shape requirement should be trimmed.
**Suggestion:** Cut the trailing clause "The shape of cos/sin must still be `head_dim` wide regardless of `rotary_dim`." from the §4 closing Note — by that point in the file the reader has seen it three times.

---

### kernel_rotate_half_pairing.md, §4 (~lines 90–105): Strategy C forward reference duplicated
**Issue:** The phrase "Under Strategy C (identity values at `[rotary_dim:]` — see Chapter 4), the same kernel produces correct output." appears at line 96, and then a full description of Strategy C semantics is re-given at lines 100–105 ("Positions `[0, rotary_dim)`: hold the actual cos/sin values... Positions `[rotary_dim, head_dim)`: hold identity-compatible values (cos=1, sin=0)...") ending with "This is **Strategy C (identity-filled cos/sin)**, which is analyzed in Chapter 4." The description at 100–105 adds genuine detail but the bold label repeats.
**Suggestion:** Remove the sentence at line 96 entirely ("Under Strategy C … the same kernel produces correct output.") since it is immediately superseded by the fuller explanation at lines 100–105. Alternatively, keep the line 96 sentence only and cut lines 100–105, but the full description is more useful so cutting 96 is the better edit.

---

### what_the_golden_function_reveals.md, §3 table (~lines 75–85): validate row restates invoke rows
**Issue:** The summary table in §3 includes a row `cos.padded_shape()[-1] == X | validate (C++) | same as invoke, enforced again`. The "same as invoke" annotation means the row carries no new information for the reader — it just confirms that `validate` repeats `invoke`. This fact is already the main point of `shape_validation_in_invoke.md` §2 and is stated in the table's own cell as "same as invoke." The row should either be cut or its annotation should be made substantively different (e.g., noting that `validate` fires even if `invoke` is patched).
**Suggestion:** Either remove the `validate` row from the table (four constraint rows without it are complete), or change the annotation to "enforced independently; fires even if `invoke` is patched" to justify the row's existence.

---

### what_the_golden_function_reveals.md, §1 (~lines 9–28): `rotate_half` Python code duplicated from Ch1
**Issue:** The `rotate_half` helper definition in §1 (lines 13–17) is a near-identical copy of the same helper in `ch1_rope_fundamentals/partial_rope_math.md` (lines 22–27). The only difference is the docstring `"""Rotate the last dimension by half."""`. Showing the function again is not harmful to understanding, but it is a cross-chapter duplicate.
**Suggestion:** Remove the code block and replace with a single sentence: "The golden's `rotate_half(x)` is identical to the helper in Ch1's `partial_rope_math.md`: it splits `x.shape[-1] // 2` and returns `cat((-x2, x1), dim=-1)`." This saves ~7 lines and avoids the impression that the golden defines something novel.

---

### index.md, §Data-Flow Diagram (~lines 30–56): autoformat annotation duplicates shape_validation_in_invoke.md §3
**Issue:** The diagram entry `(transformer.py) — golden slices input to [:rotary_dim], applies cos/sin, concatenates passthrough unchanged` (line 30) and the `AutoFormat::pad_to_tile_shape applied to cos/sin` annotation (line 33) are also fully described with code traces in `shape_validation_in_invoke.md` §3. The diagram is useful as an overview, so this is a minor overlap, but the `run_with_autoformat` box in the diagram (lines 32–35) could be condensed to just the key result rather than listing both the transform and the intermediate shape.
**Suggestion:** Condense the `run_with_autoformat` diagram block to: `AutoFormat pads cos/sin to tile boundary (shape[-1]=48 → 64)` — a single-line annotation is sufficient here since §3 of shape_validation_in_invoke.md gives the full trace.

---

## Load-Bearing Evidence

- [shape_validation_in_invoke.md, lines 18–31]: The exact `TT_FATAL` code for the `% 64 == 0` check and the `X = input_tensor.padded_shape()[-1]` assignment are stated here and nowhere else in Ch2. Must not be cut.
- [shape_validation_in_invoke.md, lines 36–53]: The `cos_cache.padded_shape()[-1] == X` check with both `invoke` and the `> Note:` scoping its verification status. The caveat "Only the `% 64 == 0` and `padded_shape()[-1]` checks have been verified from the source" is a research-honesty marker that must not be cut.
- [shape_validation_in_invoke.md, lines 66–87]: The full `validate` C++ code block. Even though the constraint is restated, the code itself is the primary evidence that `validate` is an independent enforcement gate.
- [kernel_rotate_half_pairing.md, lines 16–20]: The `Wt = input.padded_shape()[-1] / TILE_WIDTH; half_Wt = Wt / 2` derivation. This is the only place in Ch2 that shows where `half_Wt` comes from in code, as opposed to in prose description.
- [kernel_rotate_half_pairing.md, lines 63–84]: The reader kernel `rotated_input_curr_id = start_id + half_Wt` line and the two-pass fetch pattern. This is the only place this dataflow detail appears.
- [kernel_rotate_half_pairing.md, lines 97–105]: The identity-value requirement (cos=1, sin=0) for passthrough positions is stated here as a concrete correctness condition. This is load-bearing for understanding Strategy C even before Chapter 4.
- [what_the_golden_function_reveals.md, lines 44–55]: The `golden_rotary_embedding` function body, particularly `cos_slice = cos_cached[:, :, token_index : token_index + 1, :rotary_dim]`. This is the only place in Ch2 that shows the golden pre-slices to `rotary_dim`, which is the key divergence from the C++ op's `head_dim` requirement.
- [what_the_golden_function_reveals.md, lines 93–100]: The zero-padding PCC ~0.71 mechanism walkthrough ($x'_i = x_i \cdot 0 - x_{i+64} \cdot 0 = 0$). This is the only place in Ch2 where the actual numerical corruption is derived element-wise. Must not be cut.

---

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 2 — Partial RoPE Fundamentals and Tile Alignment Requirements — Pass 2

## Summary

- Files re-read: 1 (`index.md`)
- Pass 1 targeted 2 CRUCIAL items. Both items are confirmed resolved.

## CRUCIAL Suggestions

None — Pass 1 items resolved.

- Item 1 (Ch1 recap duplication): `index.md` lines 64–70 now collapsed to 2 sentences + link. The full piecewise formula, the `nearest_32` derivation, and the pre-stated mismatch conclusion have been removed. The replacement sentence names the key premise and links to `../ch1_rope_fundamentals/index.md`.
- Item 2 (`← repeated gate` annotation): annotation removed from the `validate` step in the data-flow diagram. The step now reads `TT_FATAL: cos.padded_shape()[-1] == X` without editorialization.

## Load-Bearing Evidence

- All 5 learning objectives (index.md lines 11–15) are intact and unchanged.
- All 3 reading-order file links (index.md lines 70–72) are intact: `shape_validation_in_invoke.md`, `kernel_rotate_half_pairing.md`, `what_the_golden_function_reveals.md`.
- The `What's Next` link to Chapter 3 (index.md line 80) is intact.
- The new recap sentence (index.md line 62) includes a live link to `../ch1_rope_fundamentals/index.md`, satisfying the requirement for readers who need to review Ch1.
- The `← shape gate` annotation on the `invoke` step (index.md line 40) is preserved, correctly distinguishing the first enforcement gate from the now-unannotated `validate` step.

## VERDICT
- Crucial updates: no
