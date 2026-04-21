## B Feedback Application Log — Pass 1

- Item 1: Changed `enorm(h_t)` to `hnorm(h_t)` in ASCII diagram in `mtp_head_architecture.md`
- Item 2: Fixed LaTeX equation subscripts to use `hnorm` for h_t and `enorm` for embedding in `mtp_head_architecture.md`
- Item 3: Removed spurious `/128` from MoE FFN parameter count formula in `qwen36_mtp_config.md`
- Item 4: Fixed malformed LaTeX range — changed `t+N+1` to `x_{t+N+1}` in `mtp_training_objective.md`
- Item 5: Renamed "Active FFN parameters per token" row to "Effective intermediate width per token" in `qwen36_mtp_config.md`

## B Feedback Application Log — Pass 2

- Item 1: Fixed active-parameters-per-token formula in `qwen36_mtp_config.md` — changed `$3 \times 2048 \times 7168 / 8$` to `$8 \times 3 \times 2048 \times 7168$`

## B Feedback Application Log — Pass 3

- Item 1: Updated prose parenthetical in `mtp_head_architecture.md` to name both `hnorm` and `enorm` layer norm weights

## B Feedback Application Log — Pass 4

- Item 1: Replaced "concatenated" with "combined via element-wise addition (after independent layer normalization)" in `mtp_training_objective.md` line ~33

## B Feedback Application Log — Pass 5

- Item 1: Split Stern et al. 2018 and Cai et al. 2024 descriptions in comparison table in `mtp_training_objective.md`; corrected Stern's description to "full transformer decoder blocks" instead of "linear projections"

---

# Compression Analysis: Chapter 1 — MTP Foundations — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~452 lines
- Estimated post-compression line count: ~360 lines
- Estimated reduction: ~20%

---

## CRUCIAL Suggestions

### [mtp_training_objective.md] ~lines 79–93
**Issue:** The comparison table (lines 79–84) concisely encodes all four technique comparisons in four rows. The four prose paragraphs that follow (lines 86–93) almost entirely restate what the table already shows. The only net-new information is the Medusa-vs-MTP quality argument (line 92–93), which is also restated in `mtp_head_architecture.md` line 67.
**Suggestion:** Delete the prose paragraphs for knowledge distillation, consistency regularization, and blockwise parallel decoding (they add no information beyond the table). Retain the Medusa paragraph only for its acceptance-rate claim, but condense it to one sentence appended as a table footnote. Net saving: ~7 lines.

### [mtp_head_architecture.md + mtp_training_objective.md] ~lines 108–122 (arch) and ~lines 53–59 (objective)
**Issue:** The multi-block chained-input structure (`block k receives output of block k-1 combined with embed(x_{t+k})`) is explained in full prose in `mtp_training_objective.md` lines 59 and again in `mtp_head_architecture.md` lines 49 and 115–122. Three separate explanations of the same mechanic.
**Suggestion:** In `mtp_training_objective.md`, reduce line 59 to a one-sentence forward pointer to `mtp_head_architecture.md`. Keep the full description only in the architecture file. Net saving: ~3 lines.

### [qwen36_mtp_config.md] ~lines 62–76
**Issue:** The "MTP Head Hyperparameters vs. Backbone Hyperparameters" table (lines 65–75) re-lists `hidden_size` (7168), attention heads (64 Q / 8 KV), `head_dim` (112), `layer norm type`, `activation function`, and `RoPE` — all of which are already listed in the "Relevant Configuration Fields" table (lines 11–27) with values. The only genuinely new information added by the second table is the FFN-type comparison (Dense vs. Sparse MoE) and the effective-width row.
**Suggestion:** Remove the redundant hyperparameter rows (`hidden_size`, attention heads, head_dim, layer norm, activation, RoPE) from the second table, leaving only the FFN-related rows that are not in the first table. Add a single sentence above the trimmed table noting the shared attention/norm hyperparameters. Net saving: ~6 lines.

### [mtp_head_architecture.md] ~lines 128–132
**Issue:** The final two sentences of the "Relationship to the Backbone's Layer Stack" section state: "This placement has an important implication: the MTP head cannot begin computing until the full backbone forward pass is complete. There is no opportunity to pipeline the MTP head with early backbone layers during a single forward pass. The compute graph is strictly sequential: backbone → MTP head." All three sentences convey the same point with progressive redundancy.
**Suggestion:** Collapse to one sentence, e.g., "Because the MTP head uses only the final backbone hidden state, it cannot begin computing until the full backbone pass completes — there is no intra-pass pipelining opportunity." Net saving: ~2 lines.

---

## MINOR Suggestions

### [index.md] ~line 59
**Issue:** The sentence "Full definitions appear in the guide conventions and are restated on first use in each file" conveys no actionable information for the reader — the key terms table immediately above already provides the definitions, and the reader will encounter the terms again when reading the referenced files.
**Suggestion:** Delete this sentence entirely. Net saving: 1 line.

### [mtp_training_objective.md] ~lines 65–71
**Issue:** The "Loss Weighting" section restates the combined-loss equation that was already typeset three lines earlier (line 29–30). The equation is re-displayed purely to accompany the prose about λ.
**Suggestion:** Remove the re-displayed equation block; replace with inline prose ("The scalar λ in the combined objective above..."). Net saving: ~4 lines.

### [mtp_head_architecture.md] ~lines 71–79
**Issue:** "Output of the MTP Head" section includes the equation `logits_aux = h_t^mtp W_lm_head^T` and immediately describes its shape `[B, S, V]`. This is a trivial linear projection that was already fully described in the diagram and in the input section's prose. The equation adds marginal value.
**Suggestion:** Remove the equation; keep only the prose description of the shared `lm_head` and the output shape. Net saving: ~3 lines.

### [qwen36_mtp_config.md] ~lines 50–52
**Issue:** The note "Note that `enorm` and `hnorm` are the dedicated layer norm weights applied to the shifted token embedding and backbone hidden state respectively, prior to their combination" repeats verbatim what `mtp_head_architecture.md` lines 61–65 already establishes in detail.
**Suggestion:** Replace with a forward pointer: "See `mtp_head_architecture.md` §Input for `enorm`/`hnorm` definitions." Net saving: ~2 lines.

### [qwen36_mtp_config.md] ~lines 100–102
**Issue:** The final paragraph of the lineage comparison ("The absence of `mtp_num_hidden_layers` in Qwen3.5 is not a default-zero situation…") uses ~5 lines to describe the behavior of cross-loading mismatched checkpoints. The missing-key and unexpected-key warning scenarios are implementation details not needed to understand the lineage context.
**Suggestion:** Trim to one sentence noting the field is absent (not defaulting to zero) and that MTP weights are entirely missing from Qwen3.5 checkpoints. Net saving: ~3 lines.

---

## Load-Bearing Evidence

- `index.md` line ~13: "MTP is an auxiliary objective added during model training. It teaches the model to predict not just the immediately next token but several future tokens simultaneously." — Load-bearing because this is the orienting framing for the entire chapter; removing or paraphrasing it weakens the entry-point clarity for readers arriving cold.
- `mtp_training_objective.md` line ~33: "For depth k=1 (the common single-block case), a single MTP head block processes the backbone's final hidden state combined via element-wise addition (after independent layer normalization) with the embedding of x_{t+1} to produce a distribution over x_{t+2}." — Load-bearing because this is the precise description of the element-wise addition input combination mechanic, which is the key architectural distinction from independent-head designs; it appears once in the objective file and should not be cut here even though the architecture file elaborates on it.
- `mtp_head_architecture.md` line ~62: "These two tensors are each independently layer-normalized (using dedicated `hnorm` and `enorm` layer norm weights, each separate from the backbone's layer norms), then added element-wise to produce the combined input c_t" — Load-bearing because this is the primary definition of `hnorm`/`enorm` and their separation from backbone norms; referenced by the config file.
- `qwen36_mtp_config.md` line ~26: "The MTP head does not use a MoE FFN. The MTP head block uses a dense FFN with intermediate dimension equal to `intermediate_size` (2048). This is a critical distinction: the backbone's transformer layers...use sparse MoE FFNs with 128 experts, whereas the single MTP head block uses a standard dense FFN." — Load-bearing because this is the single most important practical difference between an MTP head block and a backbone layer in this model, and it directly affects weight-loading and memory estimates in Chapter 2.

---

## VERDICT
- Crucial updates: yes

## C Compression Application Log — Pass 1

- C1: Deleted redundant prose paragraphs after comparison table in `mtp_training_objective.md`; kept Medusa acceptance-rate note only
- C2: Reduced chained-block description in `mtp_training_objective.md` to one-sentence forward pointer; kept full description in `mtp_head_architecture.md`
- C3: Removed redundant hyperparameter rows from second table in `qwen36_mtp_config.md`; kept only FFN-related new rows
- C4: Collapsed 3 sequential sentences to 1 in `mtp_head_architecture.md` lines ~128-132

## B Feedback Application Log — Pass 7

- Item 1: Renamed "Input normalization (enorm)" row to "Input normalization (hnorm, enorm)" in `mtp_head_architecture.md`
- Item 2: Converted all `$$...$$` display equations to ` ```math ` fenced blocks in `mtp_training_objective.md` (4 equations) and `mtp_head_architecture.md` (2 equations)

## B Feedback Application Log — Pass 8

- Item 1: Changed "Per-head output heads" to "Full transformer decoder blocks (one per future position)" in the Blockwise Parallel Decoding row of the comparison table in `mtp_training_objective.md`

## B Feedback Application Log — Pass 9

- Item 1: Revised line 96 of `mtp_training_objective.md` to clarify MTP's differentiator — vs. Medusa: shifted-token-embedding conditioning (Medusa has no transformer blocks); vs. Stern et al. 2018: same conditioning distinguishes MTP since both use full decoder blocks

## B Feedback Application Log — Pass 10

- Item 1: Changed "Future tokens via independent prediction heads" to "Future tokens via sequential full decoder blocks" in the Blockwise Parallel Decoding row of the comparison table in `mtp_training_objective.md`

---

# Compression Analysis: Chapter 1 — MTP Foundations — Pass 2

## Summary
- Files re-analyzed: 4
- Current line count: `index.md` ~68, `mtp_training_objective.md` ~111, `mtp_head_architecture.md` ~165, `qwen36_mtp_config.md` ~110; total ~454 lines
- Estimated post-compression: ~440 lines
- Estimated reduction this pass: ~14 lines (~3%)

## CRUCIAL Suggestions

None. All four Pass 1 CRUCIAL suggestions (C1–C4) have been applied. The remaining items are minor in scope (1–4 lines each). No new ≥5-line redundancy blocks were found in the current text.

## MINOR Suggestions

### [index.md] line ~59 — carry-over from Pass 1 Minor 1 (never applied)
**Issue:** The sentence "Full definitions appear in the guide conventions and are restated on first use in each file." is filler. The key terms table immediately above already supplies the definitions; the trailing sentence adds nothing the reader can act on.
**Suggestion:** Delete this sentence. Net saving: 1 line.

### [mtp_training_objective.md] lines ~73–77 — carry-over from Pass 1 Minor 2 (never applied)
**Issue:** The "Loss Weighting" section re-displays the combined-loss equation (`L_total = L_AR + λ · L_aux`) inside a fenced math block, even though the identical equation was already typeset in the "MTP Formulation" section six lines earlier. The only new content in this section is the prose about λ values; the equation adds no new information.
**Suggestion:** Remove the re-displayed equation block; replace with inline prose anchored to the earlier equation ("The scalar λ in the combined objective above…"). Net saving: ~4 lines.

### [mtp_head_architecture.md] lines ~75–79 — carry-over from Pass 1 Minor 3 (never applied)
**Issue:** The "Output of the MTP Head" section contains the equation `logits_aux = h_t^mtp W_lm_head^T` along with a shape annotation. This projection is already fully captured in the ASCII diagram (which labels `lm_head: linear H → V` and its output shape `[B, S, V]`) and restated in plain prose on the following lines. The equation is marginally redundant.
**Suggestion:** Remove the fenced equation block; keep the prose description of the shared `lm_head` and output shape. Net saving: ~3 lines.

### [qwen36_mtp_config.md] lines ~52–53 — carry-over from Pass 1 Minor 4 (never applied)
**Issue:** The paragraph beginning "Note that `enorm` and `hnorm` are the dedicated layer norm weights…" restates the definition and separation of these norms from backbone layer norms, which is already established in full in `mtp_head_architecture.md` §Input to the MTP Head.
**Suggestion:** Replace with a one-line forward pointer: "See `mtp_head_architecture.md` §Input for `enorm`/`hnorm` definitions." Net saving: ~2 lines.

### [qwen36_mtp_config.md] lines ~96–97 — carry-over from Pass 1 Minor 5 (never applied)
**Issue:** The final paragraph of the lineage comparison section describes in detail the HuggingFace missing-key and unexpected-key warning behavior when cross-loading Qwen3.5/Qwen3.6 checkpoints into the wrong model class. This is a secondary implementation note that dilutes the lineage narrative.
**Suggestion:** Trim to one sentence: "The field is absent from Qwen3.5 entirely (not defaulting to zero), so all `model.future_prediction.*` weights are missing from Qwen3.5 checkpoints and will produce key-mismatch warnings if loaded into the Qwen3.6 model class." Net saving: ~3 lines.

## Load-Bearing Evidence

- `index.md` line ~13: "MTP is an auxiliary objective added during model training. It teaches the model to predict not just the immediately next token but several future tokens simultaneously." — Core orienting framing; entry point for readers arriving cold.
- `mtp_training_objective.md` line ~41: "For depth k=1 (the common single-block case), a single MTP head block processes the backbone's final hidden state combined via element-wise addition (after independent layer normalization) with the embedding of x_{t+1} to produce a distribution over x_{t+2}." — Precise definition of the element-wise addition input combination; the key architectural distinction from independent-head designs.
- `mtp_head_architecture.md` line ~62: "These two tensors are each independently layer-normalized (using dedicated `hnorm` and `enorm` layer norm weights, each separate from the backbone's layer norms), then added element-wise to produce the combined input c_t" — Primary definition of `hnorm`/`enorm`; referenced by the config file and used in weight-loading in Chapter 2.
- `qwen36_mtp_config.md` line ~26: "The MTP head does not use a MoE FFN. The MTP head block uses a dense FFN with intermediate dimension equal to `intermediate_size` (2048). This is a critical distinction…" — The single most important practical difference between an MTP head block and a backbone layer in this model; directly drives memory estimates in Chapter 2.

## VERDICT
- Crucial updates: no
