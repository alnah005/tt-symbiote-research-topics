# Compression Analysis: Chapter 2 — Weight Shapes — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~382 lines
- Estimated post-compression line count: ~341 lines
- Estimated reduction: ~11%

## CRUCIAL Suggestions

### [shared_weight_shapes.md] ~lines 13–15 (attention section intro + Note callout)
**Issue:** The section intro (line 13) and the immediately following `> **Note:**` callout (lines 15–16) both state the same scope restriction: that the attention weights documented here belong to full-attention layers (indices 30–39), handled by `TTNNQwen3FullAttention`. The callout then adds the contrast with linear attention layers (indices 0–29), which is new information — but it restates the full-attention scope a second time verbatim before doing so.
**Suggestion:** Remove the intro sentence (line 13) that pre-states the scope. Keep only the Note callout, which covers the full scope including the linear attention contrast. This eliminates the redundant first statement.

### [shared_weight_shapes.md] ~lines 5–7 (Purpose section) and ~line 43 (MoE FFN section intro)
**Issue:** The Purpose section (line 7) states that `decoder_sparse_step` determines which layers are MoE vs dense, and the caveat block (lines 7–8) says this value should be verified from `config.json`. The MoE FFN section intro (line 43) then repeats this same explanation: "For layers designated as MoE layers (determined by `decoder_sparse_step`)". The dense FFN section intro (line 83) does the same: "For layers designated as dense layers (determined by `decoder_sparse_step`)". The `decoder_sparse_step` caveat explanation is thus stated three times.
**Suggestion:** Remove the parenthetical `(determined by \`decoder_sparse_step\`)` from the MoE FFN section intro (line 43) and from the dense FFN section intro (line 83). The full explanation with the verification caveat already lives in the Purpose section's callout block. Back-references are not needed; the reader already has the context.

### [extra_weight_keys.md] ~lines 13–16 (MTP table dimension derivation column)
**Issue:** The `Dimension derivation` column for `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.v_proj.weight`, and `self_attn.o_proj.weight` in the MTP keys table (lines 13–16) reproduces the identical derivation text already present in `shared_weight_shapes.md` lines 19–22 — including the verbose `num_attention_heads × head_dim` = 64 × 128 = 8192` arithmetic. Similarly for `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `input_layernorm`, and `post_attention_layernorm` in lines 17–21. These shapes are identical to backbone shapes; the table note already says "The MTP head reuses the full `hidden_size` and `head_dim` of the backbone."
**Suggestion:** Shorten the `Dimension derivation` cells in the MTP table for all keys that directly mirror backbone shapes to simply reference the governing hyperparameter without re-expanding the arithmetic (e.g., change `num_attention_heads × head_dim` = 64 × 128 = 8192 rows; `hidden_size` = 7168 cols` to `same as backbone \`self_attn.q_proj\``). For `enorm.weight` and `hnorm.weight` (which are new), keep the full derivation. This removes approximately 9 lines of repeated arithmetic spread across the derivation column.

## MINOR Suggestions

### [shared_weight_shapes.md] ~line 26 (q_norm/k_norm prose after attention table)
**Issue:** The paragraph after the attention table (line 26) restates what the table already shows: that `q_norm` and `k_norm` are scalar weight vectors of length `head_dim`. The table already has a `Dimension derivation` column that says "Per-head RMSNorm; length = `head_dim` = 128". The prose adds "not full projection matrices" which is the only novel phrase.
**Suggestion:** Trim to a single sentence: "`q_norm` and `k_norm` are per-head RMSNorm scalars (length `head_dim`), not projection matrices; they are applied after projecting, before RoPE." Saves ~1 line.

### [shared_weight_shapes.md] ~line 91 (dense FFN note)
**Issue:** The note "`intermediate_size = 14336` is approximately twice `hidden_size = 7168`, a conventional 2× ratio for dense FFN layers in this architecture family" is low-value elaboration. The values are already in the table; the 2× ratio is observable arithmetic, not a constraint needed for weight loading.
**Suggestion:** Delete this note entirely. Saves 1 line.

### [extra_weight_keys.md] ~line 50 (MTP parameter count prose)
**Issue:** "This is approximately 1.3% of the total Qwen3.6-35B-A3B parameter count" — the total parameter count is not stated anywhere in Chapter 2, so this percentage is unverifiable in context and adds hedging rather than precision.
**Suggestion:** Remove "approximately 1.3% of the total Qwen3.6-35B-A3B parameter count and" — keep only "does not affect the backbone's inference-time compute graph when MTP is disabled." Saves ~half a line.

### [extra_weight_keys.md] ~lines 56–70 (from_pretrained section)
**Issue:** The phrase "Default behavior (effective `strict=False` for top-level loading):" is followed by four bullets, the first of which begins "HuggingFace's `from_pretrained` does **not** raise an error for unexpected keys." This restates what the section heading already implies by saying it is "Default behavior." The fourth bullet (lines 68–69) summarizes that "the final model is functionally identical to a model loaded from a Qwen3.5 checkpoint" — this is stated as a consequence when it is already the chapter's top-level finding from `index.md`.
**Suggestion:** Remove the fourth bullet (lines 68–69) that re-asserts the cross-model equivalence. Saves ~2 lines.

## Load-Bearing Evidence
- `shared_weight_shapes.md` line ~111: "The following table makes the guarantee explicit. All shapes are governed solely by hyperparameters that are identical across both checkpoints (verified in Chapter 1)." — load-bearing because it frames the Shape Identity Confirmation table as the definitive guarantee rather than a summary, which is the structural claim the chapter is built around. The table itself (lines 113–135) is the only place where Qwen3.5 and Qwen3.6 shapes are shown side-by-side and cannot be cut.
- `extra_weight_keys.md` line ~7: "> **Caveat:** The key prefix `model.future_prediction.0.*` matches the pattern observed in DeepSeek-V3..." — load-bearing because it alerts readers that the documented prefix may not be authoritative for all checkpoint revisions, which is an actionable warning for weight-loading code authors.
- `index.md` line ~7: "Understanding both sets is required before writing or modifying any TTNN weight-loading code." — load-bearing framing that establishes why the chapter covers both the shared backbone AND the extra MTP keys; removing it would make the reading order section seem unmotivated.

## VERDICT
- Crucial updates: yes

## C Compression Application Log — Pass 1
- C1: [shared_weight_shapes.md] Removed redundant intro sentence from "Attention Projection Weights" section. The Note callout immediately below already states the same scope (full-attention layers 30–39) and adds the linear attention contrast.
- C2: [shared_weight_shapes.md] Removed parenthetical `(determined by \`decoder_sparse_step\`)` from MoE FFN section intro (line ~43). The Purpose section caveat block already covers this.
- C3: [shared_weight_shapes.md] Removed parenthetical `(determined by \`decoder_sparse_step\`)` from dense FFN section intro (line ~83). Same reason as C2.
- C4: [extra_weight_keys.md] Shortened `Dimension derivation` cells in the MTP keys table for the 9 keys whose shapes mirror backbone shapes. Replaced verbose arithmetic with concise hyperparameter references matching the backbone tables.

---

# Compression Analysis: Chapter 2 — Weight Shapes — Pass 2

## Summary
- Files re-analyzed: 3
- Current line count: ~383 lines (`index.md` ~33, `shared_weight_shapes.md` ~188, `extra_weight_keys.md` ~162)
- Estimated post-compression: ~378 lines (MINOR items only; no new CRUCIAL items)
- Estimated reduction this pass: ~5 lines (~1%)

## CRUCIAL Suggestions

None. All Pass 1 CRUCIAL items were correctly applied.

- C1 confirmed: `shared_weight_shapes.md` "Attention Projection Weights" section opens directly with the `> **Note:**` callout; no preceding redundant intro sentence present.
- C2 confirmed: MoE FFN section intro (line 41) reads "For layers designated as MoE layers, the FFN is replaced by a mixture-of-experts block." — no `(determined by \`decoder_sparse_step\`)` parenthetical present.
- C3 confirmed: Dense FFN section intro (line 81) reads "For layers designated as dense layers, the FFN follows a standard gated-linear-unit layout." — no `(determined by \`decoder_sparse_step\`)` parenthetical present.
- C4 confirmed: All 9 MTP table rows whose shapes mirror backbone shapes now carry concise "same as backbone `<key>`" derivation text rather than verbose arithmetic.

## MINOR Suggestions

All four MINOR items from Pass 1 carry over unchanged; none were applied.

### M1 (carry-over) — [shared_weight_shapes.md] ~line 24 (q_norm/k_norm prose)
**Issue:** The paragraph after the attention table ("q_norm and k_norm are applied per attention head after projecting, before RoPE. They are scalar weight vectors of length `head_dim`, not full projection matrices.") restates what the table's `Dimension derivation` column already shows ("Per-head RMSNorm; length = `head_dim` = 128"). The only unique content is "not full projection matrices" and the RoPE timing note.
**Suggestion:** Trim to one sentence: "`q_norm` and `k_norm` are per-head RMSNorm scalars (length `head_dim`), not projection matrices; they are applied after projecting, before RoPE." Saves ~1 line.

### M2 (carry-over) — [shared_weight_shapes.md] ~line 89 (dense FFN 2× ratio note)
**Issue:** "Note: `intermediate_size = 14336` is approximately twice `hidden_size = 7168`, a conventional 2× ratio for dense FFN layers in this architecture family." This is observable arithmetic from the table values, not a constraint needed for weight loading.
**Suggestion:** Delete this note entirely. Saves 1 line.

### M3 (carry-over) — [extra_weight_keys.md] ~line 50 (1.3% parameter percentage)
**Issue:** "This is approximately 1.3% of the total Qwen3.6-35B-A3B parameter count" — the total parameter count is not stated in Chapter 2, making this percentage unverifiable in context. It adds hedging rather than precision.
**Suggestion:** Remove "approximately 1.3% of the total Qwen3.6-35B-A3B parameter count and" — retain only "does not affect the backbone's inference-time compute graph when MTP is disabled." Saves ~half a line.

### M4 (carry-over) — [extra_weight_keys.md] ~lines 68–70 (`from_pretrained` fourth bullet)
**Issue:** The fourth bullet under "Default behavior" ("The backbone weights all load normally; the final model is functionally identical to a model loaded from a Qwen3.5 checkpoint...") re-asserts the chapter's top-level finding already stated in `index.md`'s Summary Finding block.
**Suggestion:** Remove the fourth bullet. Saves ~2 lines.

## Load-Bearing Evidence
- `shared_weight_shapes.md` line ~109: "The following table makes the guarantee explicit. All shapes are governed solely by hyperparameters that are identical across both checkpoints (verified in Chapter 1)." — load-bearing because it frames the Shape Identity Confirmation table as the definitive cross-model guarantee; the side-by-side comparison table (lines 113–134) is the structural core of the chapter and cannot be cut.
- `extra_weight_keys.md` line ~7: "> **Caveat:** The key prefix `model.future_prediction.0.*` matches the pattern observed in DeepSeek-V3 and related implementations..." — load-bearing because it explicitly warns weight-loading code authors that the documented prefix must be verified against the actual checkpoint, making it an actionable safety note rather than elaboration.
- `index.md` line ~7: "Understanding both sets is required before writing or modifying any TTNN weight-loading code." — load-bearing because it establishes the functional purpose of reading both `shared_weight_shapes.md` and `extra_weight_keys.md`, motivating the two-document structure of the chapter.

## VERDICT
- Crucial updates: no
