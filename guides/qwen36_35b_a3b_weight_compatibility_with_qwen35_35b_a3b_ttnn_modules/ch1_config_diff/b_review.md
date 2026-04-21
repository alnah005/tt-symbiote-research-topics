# B Review — Pass 1

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

---

1. **`new_and_modified_fields.md`, lines 295–299 — Parameter count arithmetic is wrong**

   The document states the MTP head adds "≈ 205M parameters". Adding up the terms
   in the displayed equation gives:

   - q + o projections: 2 × 8192 × 7168 = 117,440,512
   - k + v projections: 2 × 512 × 7168 = 7,340,032
   - gate + up + down: 3 × 2048 × 7168 = 44,040,192
   - norms: 4 × 7168 = 28,672

   Total ≈ 168.8M, not 205M. The stated figure overstates the count by ~36M
   (roughly 21%). A reader using this number to estimate checkpoint size or memory
   budget will get a wrong answer.

   **Fix:** Change "≈ 205M parameters" to "≈ 169M parameters".

2. **`new_and_modified_fields.md`, lines 280–288 — MTP head FFN size assumes `moe_intermediate_size` (2048) but a single-layer dense MTP head would use `intermediate_size` (14336)**

   The weight key listing assigns `gate_proj`/`up_proj`/`down_proj` a shape of
   `[2048, 7168]` / `[7168, 2048]`, referencing `moe_intermediate_size = 2048`.
   However, the MTP head is described as "architecturally a copy of one backbone
   decoder layer." In this hybrid model the dense (non-MoE) layers use
   `intermediate_size = 14336`, not `2048`. If the MTP layer is dense (a single
   FFN, not a mixture-of-experts layer), its FFN weight shapes are
   `[14336, 7168]` and `[7168, 14336]`, making the per-weight-key shapes and the
   parameter count in item 1 both wrong. The document should either confirm that
   the MTP FFN is a sparse/MoE FFN (justifying `2048`) or correct the shapes to
   use `intermediate_size = 14336`.

   **Fix:** Add a clarifying note that the MTP FFN uses `moe_intermediate_size`
   only if the MTP layer is itself a MoE layer; otherwise replace `2048` with
   `14336` throughout the MTP weight key table and recalculate the parameter count.

3. **`structural_fields.md`, lines 171–187 — Section titled "Fields That Differ Numerically" contains no fields that actually differ**

   Every row in this table shows the same value in the "Qwen3.5 value" and
   "Qwen3.6 value" columns (e.g., `0.02` vs `0.02`, `1e-6` vs `1e-6`,
   `"silu"` vs `"silu"`, `false` vs `false`, `131072` vs `131072`). The section
   heading promises numerically different values; the content contradicts it. A
   reader trusting the heading will believe differences exist and waste time
   investigating them, or will doubt whether the structural-identity claim in the
   chapter is accurate.

   **Fix:** Rename the section to "Fields Confirmed Identical — Listed for
   Completeness" (or similar) to match what the table actually shows.

---

# B Review — Pass 2

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Item 1 (205M → 169M): The text at line 302 now reads "≈ 169M parameters". Text change confirmed.
- Item 2 (FFN shapes 2048 → 14336): The MTP weight key table at lines 280–284 now uses `[14336, 7168]` / `[7168, 14336]`, and the prose at lines 291–293 explicitly states `intermediate_size = 14336`. Shape fix confirmed.
- Item 3 (section heading): Line 171 now reads "Fields Confirmed Identical — Listed for Completeness". Heading fix confirmed.

---

1. **`new_and_modified_fields.md`, line 302 — Parameter count "≈ 169M" is now incorrect after the FFN shape fix**

   Pass 1 fixed the FFN shapes from `moe_intermediate_size = 2048` to
   `intermediate_size = 14336`, which is correct. However, the stated total "≈ 169M
   parameters" was not updated and now does not match the formula displayed
   immediately above it. Summing the formula terms as they currently stand gives:

   - q + o projections: 2 × 8192 × 7168 = 117,440,512
   - k + v projections: 2 × 512 × 7168 = 7,340,032
   - gate + up + down: 3 × 14336 × 7168 = 308,281,344
   - norms: 4 × 7168 = 28,672

   Total = 433,090,560 ≈ **433M parameters**, not 169M. The 169M figure was
   arithmetically consistent with the old (now-removed) `2048` FFN shapes. A
   reader using this number to estimate checkpoint size or memory budget will be
   off by a factor of roughly 2.6.

   **Fix:** Change "≈ 169M parameters" to "≈ 433M parameters" at line 302.

---

# B Review — Pass 3

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Line 302 now reads "≈ 433M parameters". Arithmetic verified independently: 2×8192×7168 + 2×512×7168 + 3×14336×7168 + 4×7168 = 433,090,560 ≈ 433M. Confirmed correct.

---

1. **`new_and_modified_fields.md`, lines 137–138 — `bos_token_id = 248044` is incorrectly identified as `<|im_start|>`**

   The document states that `248044` "is the ID of the explicit `<|im_start|>` or BOS special token" in the extended Qwen3.6 vocabulary. In standard Qwen2/3 tiktoken vocabularies, `<|im_start|>` is token ID `151644` (one above the 151643 end of the base vocabulary). `248044` differs from `151644` by ~96 400 and does not correspond to any well-documented Qwen special token. A developer verifying the BOS token identity will look up `<|im_start|>` in the tokenizer, find `151644`, and be materially misled about which token the config is referring to. The practical out-of-bounds warning on line 133 is still correct (248044 > 151935), but the identity claim is wrong.

   **Fix:** Replace the `<|im_start|>` identification with a neutral statement, e.g.: "The precise identity of token `248044` requires inspecting the Qwen3.6 tiktoken vocabulary file directly; it is a special token in the extended vocabulary that lies outside the base embedding table. Do not assume it corresponds to `<|im_start|>` (which is `151644` in standard Qwen2/3 tokenizers)."

---

**No further correctness issues found.** All equations use `$$...$$` display math. Navigation footers are present and correct on both content files. All `index.md` file references use clickable markdown links. The `compression_analysis.md` file present in the directory is an internal change log and does not need to be linked from `index.md`.

---

# B Review — Pass 4

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Confirmed fixed in Pass 3.
- Pass 3 item 1 (incorrect `<|im_start|>` identification): Lines 137–145 of `new_and_modified_fields.md` now contain no claim that `248044` maps to `<|im_start|>`; the text instead instructs the reader to inspect the tiktoken file directly and notes the ~96,400 difference from `151644`. Fix confirmed.

**No feedback — chapter approved.**

# B Review — Pass 5 (post-compression)

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Confirmed fixed in Pass 3.
- Pass 3 item 1 (incorrect `<|im_start|>` identification): Confirmed fixed in Pass 4.
- Pass 4: No issues found.

**Post-compression checks:**

1. Factual correctness — all numerical values and derivations verified: `d_rot = floor(128 * 0.25) = 32` (correct); cos/sin table shape `[max_position_embeddings, 16]` where 16 = d_rot/2 (correct); `q_proj = [8192, 7168]`, `k/v_proj = [512, 7168]` (correct); MTP parameter sum 2×8192×7168 + 2×512×7168 + 3×14336×7168 + 4×7168 = 433,090,560 ≈ 433M (correct); `bos_token_id = 248044` described as outside `[0, 151935]` (correct).
2. LaTeX equations — all display equations use `$$...$$` blocks; none in plain text or code blocks.
3. Navigation footers — `structural_fields.md` ends with correct `Next` link to `new_and_modified_fields.md`; `new_and_modified_fields.md` ends with correct `Next` link to Chapter 2.
4. Clickable links — all file references in `index.md` use `[text](./path)` markdown link syntax.
5. Critical structural gaps — none; all cross-chapter references (ch3, ch4, ch5) are correctly marked as future analysis chapters.

**No feedback — chapter approved.**

# B Review — Pass 6

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Confirmed fixed in Pass 3.
- Pass 3 item 1 (incorrect `<|im_start|>` identification): Confirmed fixed in Pass 4.
- Pass 4: No issues found.
- Pass 5: No issues found (post-compression verification).

---

1. **`index.md` — Missing "Prerequisites" section**

   The chapter index has no "Prerequisites" section. The checklist explicitly requires a "Prerequisites" section in index files. Readers arriving at this chapter from a top-level guide have no indication of what background knowledge (e.g., familiarity with HuggingFace config resolution, TTNN weight loading, Qwen architecture basics) is expected before proceeding.

   **Fix:** Add a "Prerequisites" section to `index.md` listing the assumed background knowledge, placed before the Reading Order section.

# B Review — Pass 7

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Confirmed fixed in Pass 3.
- Pass 3 item 1 (incorrect `<|im_start|>` identification): Confirmed fixed in Pass 4.
- Pass 4: No issues found.
- Pass 5: No issues found (post-compression verification).
- Pass 6 item 1 (missing "Prerequisites" section in `index.md`): `index.md` now contains a "Prerequisites" section (table with three rows: HuggingFace `config.json` format, Qwen3.5-35B-A3B architecture, TTNN weight loading) placed before the Reading Order section. Fix confirmed.

No feedback — chapter approved.

# B Review — Pass 8

**Reviewed files:** `index.md`, `structural_fields.md`, `new_and_modified_fields.md`

**Prior feedback re-check:**

- Pass 1 items 1–3: Confirmed fixed in Pass 2.
- Pass 2 item 1 (433M parameter count): Confirmed fixed in Pass 3.
- Pass 3 item 1 (incorrect `<|im_start|>` identification): Confirmed fixed in Pass 4.
- Pass 4: No issues found.
- Pass 5: No issues found (post-compression verification).
- Pass 6 item 1 (missing "Prerequisites" section in `index.md`): Confirmed fixed in Pass 7.
- Pass 7: No issues found.

**Deletion coherence check (MTP speculative subsection removed):**

The deleted "MTP During Standard Autoregressive Decoding" subsection has been removed cleanly. No dangling cross-references to it remain in any of the three files. The promises made in `index.md` (lines 28–29 — "whether the MTP head is inference-active" and "whether its weight keys can interfere with the existing TTNN loading path") are still fully honoured by the remaining MTP section in `new_and_modified_fields.md`, which covers both points directly. No coherence gap was introduced by the deletion.

All prior factual fixes remain intact: `bos_token_id` described neutrally with the ~96,400 difference from `151644`; MTP FFN shapes using `intermediate_size = 14336`; total MTP parameter count stated as ≈ 433M and verified by the displayed formula.

No feedback — chapter approved.
