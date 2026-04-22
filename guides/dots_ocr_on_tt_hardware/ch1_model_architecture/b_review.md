## Agent B Review — Pass 1

**Scope:** Factual correctness only. Style and prose not flagged.

---

### Issue 1 — CRITICAL: index.md states text decoder is ~0.7B; derived figure is ~1.3B (blocks) or ~1.8B (full)

**Location:** `index.md`, Overview paragraph: "the vision encoder (~1.0B parameters across 42 ViT layers) is substantially larger than the text decoder (~0.7B parameters across 28 layers)."

**Error:** The text decoder parameter count derived in `text_decoder_hyperparameters.md` is 1,777M (~1.8B) including embedding tables, or ~1,310M for transformer blocks alone. Neither figure is close to 0.7B. The ~0.7B figure has no derivable basis from the config values and is factually wrong. A reader implementing the model from this overview would have a fundamentally wrong picture of the parameter distribution.

**Correct figures:** vision encoder ~952M (~1.0B); text decoder transformer blocks ~1,310M; text decoder including embedding tables ~1,777M.

---

### Issue 2 — CRITICAL: text_decoder_hyperparameters.md makes a self-contradictory and false claim about what "1.7B" covers

**Location:** `text_decoder_hyperparameters.md`, final paragraph under "Per-layer and total parameter count":

> "The model card's '1.7B' figure refers to the full model including the vision encoder; the text decoder alone accounts for the majority of that total, with the embedding tables being the largest single contributor outside the transformer blocks."

**Error:** The text decoder alone (as derived two paragraphs earlier in the same file) is ~1,777M. This already exceeds 1.7B before the vision encoder (~952M) is added. The claim that "1.7B refers to the full model including the vision encoder" is arithmetically impossible given the numbers in the same file. A reader following this explanation would be materially misled about the model's parameter budget. The correct statement is that the 1.7B figure on the model card cannot be reproduced by summing the config-derived components; the discrepancy analysis belongs in `vision_encoder_specs.md` (which handles it better) and the contradictory claim here should be removed.

---

### Issue 3 — Factual: vision_encoder_specs.md MLP parameter count assumes 2-matrix FFN; this claim is asserted without evidence and materially affects the parameter total

**Location:** `vision_encoder_specs.md`, parameter count derivation: "the Qwen2-VL vision encoder uses a 2-matrix FFN, not SwiGLU" and computes `2 × 1536 × 4224 = 12,976,128`.

**Error (potential):** If the vision encoder uses SwiGLU (3 matrices: gate, up, down) as the text decoder does, the correct per-block MLP count would be `3 × 1536 × 4224 = 19,464,192`, raising the vision encoder total from ~952M to ~981M. The claim of a 2-matrix FFN is stated as fact with no citation or config field to support it (`config.json` does not specify the MLP type for the vision encoder). An implementer who follows the 2-matrix assumption and is wrong will build an incorrect kernel graph. This needs a source reference or correction.

---

### Issue 4 — Factual: relationship_to_qwen25vl.md states Qwen2.5-VL-7B max_position_embeddings is 32768; index.md agrees; but this is not verifiable from the provided ground truth config

**Location:** `index.md` comparison table row `max_position_embeddings`: dots.ocr = 131072, Qwen2.5-VL-7B = 32768. Also `text_decoder_hyperparameters.md`: "This is four times the context length of Qwen2.5-VL-7B (32,768 tokens)."

**Note:** The ground truth config provided covers only dots.ocr. The Qwen2.5-VL-7B `max_position_embeddings` is stated as 32768 in both files. The actual published value for Qwen2.5-VL-7B is 32768 in its base config. This is correct and consistent — flagged only because it cannot be verified from the provided ground truth. **Not a confirmed error; flagging for awareness.**

---

### Issue 5 — Factual: vision_encoder_specs.md reconciliation table lists text decoder transformer blocks as "~1,310M" but labels row as "Text decoder transformer blocks (28 layers)" — creating ambiguity that contradicts the ~0.7B figure in index.md

**Location:** `vision_encoder_specs.md`, reconciliation table:

| Text decoder transformer blocks (28 layers) | ~1,310M |

**Issue:** The reconciliation table correctly shows ~1,310M for transformer blocks alone. Combined with Issue 1 (index.md claiming ~0.7B for the text decoder), a reader who reads these files in order (as instructed) will receive ~0.7B in the index and ~1,310M in the later file, with no explanation of the discrepancy. The text decoder's transformer blocks are ~1.3B, not ~0.7B. The index.md figure needs correction to ~1.3B (blocks) or the overview must clarify which components are included in each figure.

---

**Summary of confirmed factual errors requiring correction:**

| # | File | Error |
|---|---|---|
| 1 | `index.md` | Text decoder stated as ~0.7B; correct derived figure is ~1.3B (blocks) or ~1.8B (with embeddings) |
| 2 | `text_decoder_hyperparameters.md` | Claims "1.7B refers to full model incl. vision encoder" — arithmetically impossible since text decoder alone is ~1.8B |
| 3 | `vision_encoder_specs.md` | 2-matrix FFN assumption for vision MLP is unsourced; if wrong, per-block MLP count is off by 50% |

---

## Agent B Review — Pass 2

**Scope:** Factual correctness only. Style and prose not flagged.

---

### Issue 1 — Factual: vision_encoder_specs.md config listing includes `embed_dim` field not present in ground truth config

**Location:** `vision_encoder_specs.md`, lines 11 and 31–32 (config block and surrounding prose).

**Error:** The config listing inside `vision_encoder_specs.md` includes `"embed_dim": 1536` as a field under `vision_config`. The ground truth `config.json` does not contain an `embed_dim` field; `vision_config` contains only: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `post_norm`, `rms_norm_eps`, `use_bias`. An implementer who reads this listing and expects `embed_dim` to exist in the published checkpoint config will have an incorrect config schema. The prose at line 31–32 further explains the semantics of `embed_dim` as if it is a confirmed field, compounding the error.

**Correction required:** Remove `embed_dim: 1536` from the config listing and remove or restructure the prose that treats it as a distinct, confirmed config field. The architectural point (output dimension of patch embedding equals `hidden_size`) is correct and can be retained, but it should be stated as a derived inference from `hidden_size`, not as a separate config field.

---

### Issue 2 — Arithmetic: text_decoder_hyperparameters.md total parameter sum is off by 1,000

**Location:** `text_decoder_hyperparameters.md`, final arithmetic under "Per-layer and total parameter count":

> `1,310,382,080 + 466,748,928 = 1,777,130,008`

**Error:** The correct sum is 1,777,131,008, not 1,777,130,008. The file is off by exactly 1,000. The final RMSNorm (1,536 parameters, correctly listed in the non-layer table) is included in the 466,748,928 figure, so that is not the source of the discrepancy — the addition itself is wrong. While 1,000 parameters is negligible at the 1.8B scale, the file presents this as a precise derivation from first principles; a reader checking the arithmetic will find it does not add up.

**Correction required:** Change `1,777,130,008` to `1,777,131,008` (and update the `≈ 1,777M` rounding accordingly — it remains `≈ 1,777M`).

---

**Summary of confirmed factual errors requiring correction:**

| # | File | Error |
|---|---|---|
| 1 | `vision_encoder_specs.md` | `embed_dim` listed as a ground-truth config field; it is absent from the actual `vision_config` |
| 2 | `text_decoder_hyperparameters.md` | Final sum stated as 1,777,130,008; correct value is 1,777,131,008 (off by 1,000) |

---

## Agent A Change Log — Pass 1

- Fix 1 applied: Corrected "~0.7B text decoder" in index.md to ~1.78B (≈1.7B per model card); clarified that the vision encoder (~1.2B) is additional, total model ~2.7–3.0B.
- Fix 2 applied: Corrected text_decoder_hyperparameters.md to state that the model card's "1.7B" refers to the text decoder alone, not the full model including vision encoder.
- Fix 3 applied: Corrected vision_encoder_specs.md to use SwiGLU 3-matrix MLP (gate/up/down) for the vision encoder; recomputed per-block count to ~28.9M and total vision encoder to ~1.22B.

## Agent A Change Log — Pass 2

- Fix 1 applied: Removed `embed_dim` from vision_encoder_specs.md config listing (field does not exist in the actual config.json); patch embed output dimension is now described as equal to `hidden_size=1536`.
- Fix 2 applied: Corrected arithmetic in text_decoder_hyperparameters.md: 1,310,382,080 + 466,748,928 = 1,777,131,008 (was 1,777,130,008).

## Agent B Review — Pass 3

**Scope:** Factual correctness only. Style and prose not flagged.

---

### Issue 1 — Factual: relationship_to_qwen25vl.md still lists `embed_dim` as a confirmed vision_config field

**Location:** `relationship_to_qwen25vl.md`, Architectural lineage section, bullet 4:

> "The fields in dots.ocr's `vision_config` — `embed_dim`, `spatial_merge_size`, `temporal_patch_size`, `post_norm`, `attn_implementation` — use the same field names and semantics as the Qwen2-VL vision config schema."

**Error:** `embed_dim` is not a field in dots.ocr's `vision_config`. The Pass 2 fix correctly removed it from the config listing in `vision_encoder_specs.md` and added explicit prose clarifying it is not a real config field. That correction was not propagated to `relationship_to_qwen25vl.md`, which still asserts `embed_dim` as a concrete, named config field that dots.ocr shares with Qwen2-VL. An implementer reading this file to understand the config schema will expect `embed_dim` to exist and be disappointed or confused when it is absent from the actual `config.json`.

**Correction required:** Remove `embed_dim` from the list of shared field names in bullet 4. The underlying architectural point (patch embedding output dimension matches `hidden_size`) is correct but should not be expressed as a named config field.

---

**Summary:**

| # | File | Error |
|---|---|---|
| 1 | `relationship_to_qwen25vl.md` | `embed_dim` listed as a shared vision_config field; it does not exist in the actual config.json |

## Agent A Change Log — Pass 3

- Fix applied: Removed `embed_dim` from shared vision config schema field list in relationship_to_qwen25vl.md (field does not exist in config.json; the Pass 2 fix to vision_encoder_specs.md was not propagated to this file).

## Agent B Review — Pass 4

**Scope:** Factual correctness only. Style and prose not flagged.

---

All four files were checked against the ground truth config values:

- Text decoder fields (hidden_size=1536, intermediate_size=8960, num_hidden_layers=28, num_attention_heads=12, num_key_value_heads=2, vocab_size=151936, attention_bias=true, rms_norm_eps=1e-06): all correct.
- Vision config fields (hidden_size=1536, intermediate_size=4224, num_hidden_layers=42, num_attention_heads=12, patch_size=14, spatial_merge_size=2, temporal_patch_size=1, post_norm=true, rms_norm_eps=1e-05, use_bias=false): all correct.
- Vision MLP: SwiGLU 3-matrix (gate, up, down) confirmed and correctly documented with source citation.
- `embed_dim`: correctly absent from vision_config listing in vision_encoder_specs.md and correctly absent from shared-field list in relationship_to_qwen25vl.md (Pass 3 fix propagated successfully).
- Parameter totals: text decoder 1,777,131,008 arithmetic verified; vision encoder 1,224,327,168 arithmetic verified; model card "1.7B" = text decoder only — correctly stated throughout.
- All comparison values for Qwen2.5-VL-7B in index.md and relationship_to_qwen25vl.md are internally consistent.

**No feedback — chapter approved.**
