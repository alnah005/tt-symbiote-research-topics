# Compression Analysis: Chapter 2 — M-RoPE in Qwen3.6-35B-A3B — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~310 lines
- Estimated post-compression line count: ~255 lines
- Estimated reduction: ~18%

---

## CRUCIAL Suggestions

### `index.md` ~lines 9–21
**Issue:** The "Quick-Reference: M-RoPE Config Fields" table duplicates the opening JSON block and field-by-field bullet explanations in `qwen36_rope_config.md` sections 1–3. Every value in the table (`rope_theta`, `partial_rotary_factor`, `head_dim`, `rotary_dim`, `rope_scaling.type`, `rope_scaling.mrope_section`) is already stated and explained in full in the config file. The `index.md` table adds no new information and requires maintenance when values change.
**Suggestion:** Remove the entire "Quick-Reference: M-RoPE Config Fields" section from `index.md`. The Contents table already tells readers where to find config details; the quick-reference table just restates what they will immediately see on opening `qwen36_rope_config.md`.

### `hf_reference_implementation.md` ~lines 78–93
**Issue:** The "full simplified function" code block in Step d is a near-verbatim repeat of what was already shown piecemeal in Steps a–d. It restates the unsqueeze, the `q_rot`/`q_pass` split, the `q_embedded` formula, and the `torch.cat`, all of which appear with identical variable names and comments 15–30 lines above. The only delta is the `# Similarly for k` comment, which is not new information.
**Suggestion:** Delete the "full simplified function" code block (lines ~80–93). The step-by-step trace is the canonical walkthrough; a re-collapsed version adds length without adding clarity. If a consolidated view is genuinely wanted, replace it with a single sentence pointing to the actual HuggingFace source.

### `position_id_construction.md` ~lines 111–113
**Issue:** Section 5 ("Text-Only Degenerate Case") is a near-verbatim restatement of the last two sentences of section 2 ("Text-Only Construction", lines ~27–28). Both sections state: all three rows are identical to sequential 1D positions; the assembled cos/sin equals standard 1D RoPE; Chapter 3 proves this. The only addition in section 5 is the `TTNNRotaryPositionEmbedding` sentence, which belongs in a forward-references section, not a standalone section.
**Suggestion:** Delete section 5 entirely. Move the single new sentence ("The existing `TTNNRotaryPositionEmbedding` text-only path does not need modification for text-only Qwen3.6 inference") into the section 6 Forward References bullet that already points to Chapter 6.

### `hf_reference_implementation.md` ~lines 131–143
**Issue:** Section 4 ("Position ID Shapes and Dtypes") partially duplicates `position_id_construction.md`. The shape `[3, batch_size, seq_len]`, the axis semantics (0=temporal, 1=height, 2=width), and the text-only vs. image-input descriptions are all covered in full in `position_id_construction.md` sections 1–3. Having a mini-summary here creates a second source of truth for the same facts.
**Suggestion:** Replace section 4 with a single cross-reference sentence: "Position ID shapes, dtypes, and construction for text-only and vision inputs are covered in [`position_id_construction.md`](./position_id_construction.md)." This removes ~10 lines while preserving navigability.

---

## MINOR Suggestions

### `hf_reference_implementation.md` ~line 10
**Issue:** "These two functions together constitute the numerical ground truth for TTNN validation." This sentence restates the obvious consequence of "traces both functions exactly so that the TTNN implementation in Chapter 4 can replicate their numerical behavior" said in the same sentence. Two clauses saying the same thing.
**Suggestion:** Delete the second sentence ("These two functions together constitute..."). The preceding sentence already covers the purpose.

### `qwen36_rope_config.md` ~line 63
**Issue:** "The `partial_rotary_factor` at the top level is a consistency hint for tooling and documentation; the section sum is the authoritative source inside the model code." This is the third time the relationship between `partial_rotary_factor` and the section sum is stated (also in field explanations at line ~23 and the Key Finding callout at line ~67). The inline prose at line 63 is the weakest of the three — it is sandwiched between the derivation and the callout that already makes the point more forcefully.
**Suggestion:** Delete the sentence at line 63. The Key Finding callout immediately below it makes the same point with more precision.

### `position_id_construction.md` ~lines 76–79
**Issue:** The three "Key points" bullets after the image construction code block hedge with "always" and restate facts already visible in the code comments above them. The first bullet ("Text tokens always have identical values across all three rows") was stated in section 2. The third bullet (suffix text starts from `n_text_pre + max(H, W)`) repeats the `post_start` line in the code.
**Suggestion:** Collapse the three bullets to one: "Suffix text positions start from `n_text_pre + max(H, W)` to avoid colliding with any image coordinate." The other two points are already evident from the code or stated elsewhere.

### `position_id_construction.md` ~lines 119–121
**Issue:** Section 6 ("Forward References") contains two bullets. The first bullet ("The position ID construction above follows the HuggingFace reference described in `hf_reference_implementation.md`") is a backward reference to the file immediately preceding this one in the reading order. It states the obvious and adds no navigational value.
**Suggestion:** Delete the first bullet of section 6. Keep only the Chapter 6 forward reference.

### `hf_reference_implementation.md` ~line 37
**Issue:** "This broadcasts cos/sin across all attention heads when multiplied against `q` and `k`." This prose explanation of broadcasting is standard PyTorch and is fully evident from the shape comment (`[B, 1, S, rotary_dim]`) on the same line. It adds 16 words of padding.
**Suggestion:** Delete the explanatory sentence following Step a. The shape annotation is self-documenting.

---

## Load-Bearing Evidence

- `qwen36_rope_config.md` line ~67: "> **Key Finding:** The canonical source of `rotary_dim` for an M-RoPE model is `2 × sum(rope_scaling.mrope_section)`, not `partial_rotary_factor`. A TTNN implementation must read `rope_scaling.mrope_section` from the config, not rely on `partial_rotary_factor` alone." — Load-bearing because this is the primary implementation constraint this chapter exists to communicate; removing or weakening it would leave implementers without the critical config-reading rule.

- `qwen36_rope_config.md` lines ~49–51 (the head-dimension breakdown table): The explicit mapping of head-dimension ranges (`[0,11)`, `[32,43)`, etc.) to coordinate sections is not derivable from the section widths alone without knowing the interleaved layout. This table is the reference a TTNN implementer needs to get index arithmetic right.

- `hf_reference_implementation.md` lines ~101–115 (the three-gather + duplication code block): The exact sequence of column-slice indices (`[:s_t]`, `[s_t:s_t+s_h]`, `[s_t+s_h:]`) and the `cat([cos_half, cos_half])` duplication are the numerical ground truth that Chapter 4 must replicate. This cannot be cut.

- `hf_reference_implementation.md` lines ~118–127 (the duplication explanation with the rotation matrix): The prose and math explaining *why* `cos_half` is duplicated — that rotate-half pairs dimension $i$ with $i+32$ and both need the same $\cos\theta_i$ — is the non-obvious step. Without this, the duplication looks like a bug. Load-bearing.

- `position_id_construction.md` lines ~46–71 (the image construction code block): The specific use of `repeat_interleave` for height vs. `repeat` for width, the `n_text_pre` offset applied to `h_image` and `w_image` but not to `t_image`, and the `max(H, W)` gap for suffix tokens are all non-obvious and not duplicated anywhere else. Load-bearing.

- `position_id_construction.md` lines ~90–108 (the video construction code block): The pattern of holding spatial coordinates constant across frames while incrementing temporal is stated here and nowhere else. Load-bearing for the video use case.

---

## VERDICT
- Crucial updates: yes

## C Compression Application Log — Pass 1

- C1: Removed the "Quick-Reference: M-RoPE Config Fields" table from index.md (duplicated qwen36_rope_config.md)
- C2: Deleted the re-collapsed "full simplified function" code block from hf_reference_implementation.md; step-by-step trace retained
- C3: Deleted Section 5 ("Text-Only Degenerate Case") from position_id_construction.md; TTNNRotaryPositionEmbedding sentence moved to Section 6 Forward References
- C4: Replaced Section 4 ("Position ID Shapes and Dtypes") in hf_reference_implementation.md with a single cross-reference sentence to position_id_construction.md

---

# Compression Analysis: Chapter 2 — M-RoPE in Qwen3.6-35B-A3B — Pass 2

## Summary
- Files re-analyzed: 4
- Current line count: ~354 lines (15 + 88 + 133 + 118)
- Estimated post-compression: ~347 lines
- Estimated reduction this pass: ~7 lines (~2%)

## CRUCIAL Suggestions

None. All four Pass 1 CRUCIAL items were correctly applied:
- C1: `index.md` no longer contains a "Quick-Reference: M-RoPE Config Fields" table. File is now 15 lines (prerequisites + contents table only).
- C2: `hf_reference_implementation.md` no longer contains the re-collapsed "full simplified function" block. Step d ends at line 76 with no repeat.
- C3: `position_id_construction.md` has no Section 5. The `TTNNRotaryPositionEmbedding` sentence now appears correctly in Section 6 Forward References (line 114).
- C4: `hf_reference_implementation.md` Section 4 is replaced by the two-line cross-reference block at lines 114–116 pointing to `position_id_construction.md`.

No new redundancies of ≥5 lines were found.

## MINOR Suggestions

All five Pass 1 MINOR items remain unapplied and still valid:

### M1 (carry-over) — `hf_reference_implementation.md` line 10
**Issue:** "These two functions together constitute the numerical ground truth for TTNN validation." is a restatement of the preceding clause ("traces both functions exactly so that the TTNN implementation in Chapter 4 can replicate their numerical behavior"). Two consecutive phrases saying the same thing.
**Suggestion:** Delete the second sentence. Save 1 line.

### M2 (carry-over) — `qwen36_rope_config.md` line 63
**Issue:** "The `partial_rotary_factor` at the top level is a consistency hint for tooling and documentation; the section sum is the authoritative source inside the model code." The `partial_rotary_factor`-vs-section-sum relationship is stated three times in this file: in the field bullet at line 23, here at line 63, and in the Key Finding callout at line 67. The line 63 occurrence is the weakest — it is sandwiched between the derivation and the callout.
**Suggestion:** Delete the sentence at line 63. The Key Finding callout makes the same point with more precision. Save 1 line.

### M3 (carry-over) — `position_id_construction.md` lines 75–79
**Issue:** The three "Key points" bullets restate facts already visible in the code directly above. Bullet 1 ("Text tokens always have identical values across all three rows") duplicates the Section 2 statement at line 11. Bullet 3 ("Suffix text positions continue from `n_text_pre + max(H, W)` so that post-image text position IDs do not collide...") repeats the `post_start` variable assignment in the code block at line 63.
**Suggestion:** Collapse to one bullet: "Suffix text positions start from `n_text_pre + max(H, W)` to avoid colliding with any image coordinate." Save ~2 lines.

### M4 (carry-over) — `position_id_construction.md` line 113
**Issue:** First bullet of Section 6 Forward References: "The position ID construction above follows the HuggingFace reference described in `hf_reference_implementation.md`." This is a backward pointer to the file immediately preceding this one in reading order. It states the obvious and adds no navigational value.
**Suggestion:** Delete this bullet. Keep only the Chapter 6 forward reference. Save 1 line.

### M5 (carry-over) — `hf_reference_implementation.md` line 37
**Issue:** "This broadcasts cos/sin across all attention heads when multiplied against `q` and `k`." The broadcasting behavior is already fully expressed by the shape annotation `[B, 1, S, rotary_dim]` on the same line; the dimension-1 slot makes the broadcast self-evident to any PyTorch reader.
**Suggestion:** Delete this explanatory sentence. Save 1 line.

### M6 (new) — cross-file: `position_id_construction.md` line 27 vs `hf_reference_implementation.md` line 129
**Issue:** The claim "Chapter 3 proves [text-only M-RoPE equals standard 1D RoPE]" appears in both files. `position_id_construction.md` line 27: "The assembled cos/sin is numerically equal to standard 1D RoPE. Chapter 3 proves this formally." `hf_reference_implementation.md` line 129 (Section 6 Forward References): "Chapter 3 … proves that when all three position ID rows are identical, the assembled cos/sin is numerically equal to the standard 1D RoPE output." The statement is short (1 line each) so this is minor, but one of the two occurrences is redundant. The `position_id_construction.md` instance is the more appropriate location because that file is where the text-only construction is shown.
**Suggestion:** In `hf_reference_implementation.md` Section 6 Forward References (line 129), shorten to: "Chapter 3 (`../ch3_text_only_reduction/mathematical_equivalence_proof.md`) proves the text-only M-RoPE equivalence formally." This removes the re-statement of the conclusion while preserving the navigational pointer. Save a few words but not a line; treat as style-level.

## Load-Bearing Evidence

- `index.md` (entire file, 15 lines): The prerequisites pointer and contents table are the only navigational scaffolding for this chapter. Every line is structural; nothing is redundant prose.

- `qwen36_rope_config.md` line 67 (Key Finding callout): "The canonical source of `rotary_dim` for an M-RoPE model is `2 × sum(rope_scaling.mrope_section)`, not `partial_rotary_factor`. A TTNN implementation must read `rope_scaling.mrope_section` from the config, not rely on `partial_rotary_factor` alone." This is the primary implementation constraint the chapter exists to communicate. Removing it would leave implementers without the critical config-reading rule.

- `hf_reference_implementation.md` lines 84–98 (three-gather + cos_full duplication code block): The column-slice indices (`[:s_t]`, `[s_t:s_t+s_h]`, `[s_t+s_h:]`) and the `cat([cos_half, cos_half])` pattern are the exact numerical operations Chapter 4 must replicate. Not duplicated anywhere else.

- `position_id_construction.md` lines 46–72 (image construction code block): The `repeat_interleave` vs `repeat` distinction for height/width, the asymmetric offset (temporal not offset, spatial offset by `n_text_pre`), and the `max(H, W)` gap formula are non-obvious and have no equivalent in any other file.

## VERDICT
- Crucial updates: no
