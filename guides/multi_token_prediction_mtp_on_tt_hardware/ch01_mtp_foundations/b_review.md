# B Review — Pass 1

1. **`mtp_head_architecture.md`, line 27 (diagram) — wrong norm name for hidden state**
   The ASCII diagram shows `enorm(h_t) + enorm(embed(x_{t+1}))`, using `enorm` for both inputs. `qwen36_mtp_config.md` (line 52) states explicitly that `hnorm` normalizes the backbone hidden state and `enorm` normalizes the shifted token embedding. A reader writing weight-loading code who follows this diagram will look for `model.future_prediction.0.enorm.weight` to normalize `h_t`, but the correct key is `model.future_prediction.0.hnorm.weight`. Fix: change `enorm(h_t)` to `hnorm(h_t)` in the diagram.

2. **`mtp_head_architecture.md`, line 63 (LaTeX equation) — equation subscripts conflict with actual weight key names**
   The equation uses `\text{LayerNorm}_{\text{enorm\_h}}(h_t) + \text{LayerNorm}_{\text{enorm\_emb}}(\text{embed}(x_{t+1}))`. The subscript names `enorm_h` and `enorm_emb` are not the actual weight keys. Per `qwen36_mtp_config.md`, the keys are `hnorm` (for `h_t`) and `enorm` (for the embedding). This is the same root error as item 1, appearing in the formal equation. An implementer mapping this equation to a weight-loading routine will use the wrong key for `h_t`. Fix: use `\text{LayerNorm}_{\text{hnorm}}(h_t) + \text{LayerNorm}_{\text{enorm}}(\text{embed}(x_{t+1}))`.

3. **`qwen36_mtp_config.md`, line 76 — MoE FFN total parameter count formula is arithmetically wrong**
   The text states the MoE FFN has `$3 \times 2048 \times 128 \times 7168 / 128 = 3 \times 2048 \times 7168$ parameters across all experts`. Dividing by 128 is unjustified: the total parameter count across all 128 experts is `128 × 3 × 2048 × 7168`, which is 128× larger than a single dense layer, not equal to it. The formula as written algebraically cancels the 128 experts against a spurious division, making the MoE FFN appear to have the same parameter count as the dense MTP head FFN. This directly contradicts the file's own claim that the MTP head is "substantially smaller." Fix: remove the `/ 128` and state the total as `$128 \times 3 \times 2048 \times 7168$`.

4. **`mtp_training_objective.md`, line 81 (comparison table) — malformed LaTeX range missing `x_` on upper bound**
   The "What is predicted" cell for MTP reads `Future tokens $x_{t+2}, \ldots, t+N+1$`. The upper-bound term `t+N+1` is a bare arithmetic expression, not a token variable. It should be `$x_{t+N+1}$`. As written, the expression `$x_{t+2}, \ldots, t+N+1$` is syntactically inconsistent and a reader cannot determine the predicted range from the formula alone. Fix: change `t+N+1` to `x_{t+N+1}` so the cell reads `$x_{t+2}, \ldots, x_{t+N+1}$`.

5. **`mtp_training_objective.md`, no navigation footer — structural gap**
   `mtp_training_objective.md` ends at line 104 with `**Next:** [\`mtp_head_architecture.md\`](./mtp_head_architecture.md)`. On inspection this footer is present and uses a clickable link, so navigation is correct. Disregard — no issue here.

   **Replacing item 5 with the actual fifth issue:**

5. **`qwen36_mtp_config.md`, line 71 (hyperparameter table) — "Active FFN parameters per token" row reports intermediate dimensions, not parameter counts**
   The MoE cell says `2048 per expert × 8 experts = 16384 effective` and the dense/MTP cell says `2048`. These are intermediate-dimension values (scalars), not parameter counts. A parameter count for active MoE FFN weights per token would be `3 × 2048 × 7168 × 8 = ~353 M`, while the dense/MTP count is `3 × 2048 × 7168 = ~44 M`. The row label says "parameters" but the values are "intermediate size × active experts." A reader computing memory or FLOP estimates from this row will get numbers that are off by a factor of `3 × 7168 = 21504`. Fix: either rename the row to "Effective intermediate width per token" or replace the values with actual parameter counts.

---

# B Review — Pass 2

1. **`qwen36_mtp_config.md`, line 76 — active-parameters-per-token formula uses division instead of multiplication**
   The sentence reads: "which has $128 \times 3 \times 2048 \times 7168$ parameters across all experts but $3 \times 2048 \times 7168 / 8$ active parameters per token at the MoE routing level." With `num_experts_per_tok = 8`, active parameters per token equals `8 × 3 × 2048 × 7168` (approximately 353 M). The formula `$3 \times 2048 \times 7168 / 8$` instead divides by 8, giving a value roughly 64× too small and contradicting the table on line 71, which correctly shows `16384 effective` intermediate width (i.e., `2048 × 8`). A reader using this formula to estimate active-weight memory or MoE FLOP cost will be off by a factor of 64. Fix: replace `$3 \times 2048 \times 7168 / 8$` with `$8 \times 3 \times 2048 \times 7168$`.

**Pass 1 items — re-check status:**
- Item 1 (`mtp_head_architecture.md` line 27, `enorm(h_t)` → `hnorm(h_t)`): Fixed.
- Item 2 (`mtp_head_architecture.md` line 63, LaTeX subscripts): Fixed.
- Item 3 (`qwen36_mtp_config.md` line 76, spurious `/128` in total parameter formula): Fixed. (A new error was introduced in the same sentence — see item 1 above.)
- Item 4 (`mtp_training_objective.md` line 81, `t+N+1` → `x_{t+N+1}`): Fixed.
- Item 5 (`qwen36_mtp_config.md` line 71, row label renamed to "Effective intermediate width per token"): Fixed.

---

# B Review — Pass 3

1. **`mtp_head_architecture.md`, line 61 — prose names only `enorm` when two separate norm layers exist**
   The sentence reads: "These two tensors are each independently layer-normalized (using dedicated `enorm` layer norm weights separate from the backbone's layer norms), then added element-wise." Only `enorm` is named, omitting `hnorm`. An implementer reading this sentence in isolation would allocate one normalization layer for both inputs rather than two, and would look for only `model.future_prediction.0.enorm.weight` when the checkpoint also requires `model.future_prediction.0.hnorm.weight`. The equation on line 63 immediately below is correct (`LayerNorm_hnorm` and `LayerNorm_enorm`), but the prose directly contradicts it. Fix: replace the parenthetical with "(using dedicated `hnorm` and `enorm` layer norm weights, each separate from the backbone's layer norms)" so the prose matches the equation and the checkpoint key list in `qwen36_mtp_config.md`.

---

# B Review — Pass 4

1. **`mtp_training_objective.md`, line 33 — "concatenated" contradicts the element-wise addition described in `mtp_head_architecture.md`**
   The sentence reads: "a single MTP head block processes the backbone's final hidden state **concatenated** with the embedding of $x_{t+1}$ to produce a distribution over $x_{t+2}$." However, `mtp_head_architecture.md` (lines 61–65) consistently describes the operation as an element-wise addition after independent layer normalization: $c_t = \text{LayerNorm}_{\text{hnorm}}(h_t) + \text{LayerNorm}_{\text{enorm}}(\text{embed}(x_{t+1}))$, yielding shape `[B, S, H]`. Concatenation would produce shape `[B, S, 2H]` and would require an additional projection back to `H` that is not described anywhere in the guide. An implementer who follows `mtp_training_objective.md` line 33 will build a wrong input-combination module. Fix: replace "concatenated" with "combined via element-wise addition (after independent layer normalization)" so the description matches the equation in `mtp_head_architecture.md` and the checkpoint weight keys (`hnorm`, `enorm`) in `qwen36_mtp_config.md`.

---

# B Review — Pass 5

1. **`mtp_training_objective.md`, line 90 — Blockwise Parallel Decoding (Stern et al., 2018) mischaracterized as simple linear projections**
   The sentence reads: "Blockwise parallel decoding attaches independent prediction heads directly to the backbone's final hidden state, one head per future position. Each head is a separate linear projection with no transformer blocks." This description accurately characterizes Medusa (Cai et al., 2024) but is factually wrong for Stern et al., 2018. Stern's Blockwise Parallel Decoding uses a sequence of full transformer decoder blocks as predictors — one block per future position — not simple linear projections. The same sentence is used to justify the claim that MTP heads achieve higher acceptance rates due to their "richer parameterization," but if Stern's blocks are also full transformer decoder blocks, this contrast does not hold for that citation. A reader who looks up Stern et al. and finds full decoder blocks will be confused and may distrust the surrounding analysis. Fix: separate the two citations. Describe Stern et al., 2018 as using full transformer decoder blocks (one per future position, each attending to the backbone hidden state), and Medusa (Cai et al., 2024) as using simple feedforward heads. The contrast with MTP's richer parameterization should then apply only to Medusa-style heads, not to Stern's method.

---

# B Review — Pass 6

No feedback — chapter approved.

---

# B Review — Pass 7

1. **`mtp_head_architecture.md`, line 102 (shared-parameters table) — row label omits `hnorm`, lists only `enorm`**
   The table row reads "Input normalization (enorm) | No — dedicated MTP head enorm weights." There are two dedicated input-normalization layers (`hnorm` for the backbone hidden state and `enorm` for the shifted token embedding), but the row label and description name only `enorm`. A reader consulting this table to enumerate MTP weight keys will allocate one normalization layer instead of two and will miss `model.future_prediction.0.hnorm.weight` entirely. Pass 3 corrected the prose on line 61 to name both `hnorm` and `enorm`, but this table row was not updated. Fix: change the row to "Input normalization (hnorm, enorm) | No — dedicated MTP head hnorm and enorm weights".

2. **All content files — display equations use `$$...$$` LaTeX delimiters instead of ` ```math ``` ` fenced code blocks**
   `mtp_training_objective.md` (lines 11, 22, 25, 29, 67), `mtp_head_architecture.md` (lines 63, 75), and `qwen36_mtp_config.md` (line 70) all use `$$...$$` display-math delimiters. The review criteria require display equations to appear in ` ```math ``` ` fenced blocks. `$$` delimiters render correctly only in environments that enable LaTeX math (e.g., Jupyter, some Markdown renderers); plain GitHub Markdown and many documentation viewers will display the raw LaTeX source as literal text. Fix: convert every `$$...$$` display equation block in all three content files to a ` ```math ``` ` fenced code block.

---

# B Review — Pass 8

1. **`mtp_training_objective.md`, line 94 (comparison table) — "Gradient flows through" cell for Blockwise Parallel Decoding still says "Per-head output heads", which mischaracterizes Stern et al. 2018**
   The table row reads "Blockwise parallel decoding | Future tokens via independent prediction heads | Per-head output heads | Can be used directly for draft tokens." Stern et al. 2018 uses a sequence of full transformer decoder blocks as predictors — gradient flows through full decoder blocks, not through simple "per-head output heads." "Per-head output heads" accurately describes Medusa-style linear projections but is factually wrong for Stern's method. Pass 5 corrected the prose sentence on line 96 to no longer conflate Stern and Medusa, but the table itself was not updated and still carries the inaccurate description. A reader checking the table to understand architectural differences will get a false picture of Stern 2018 and may incorrectly conclude that all non-MTP multi-step methods use shallow heads. Fix: change the "Gradient flows through" cell for the Blockwise parallel decoding row from "Per-head output heads" to "Full transformer decoder blocks (one per future position)".

---

# B Review — Pass 9

1. **`mtp_training_objective.md`, line 96 — coherence gap introduced by the Pass 8 fix: the footnote sentence attributes MTP's acceptance-rate advantage to "full transformer decoder blocks," but the now-corrected table shows Blockwise Parallel Decoding also uses full transformer decoder blocks**
   Line 96 reads: "Medusa heads (Cai et al., 2024) are simple linear projections with no transformer blocks; MTP heads achieve higher draft acceptance rates in practice due to their full transformer decoder blocks and shifted-token-embedding conditioning." The Pass 8 correction changed the Blockwise parallel decoding row's "Gradient flows through" cell to "Full transformer decoder blocks (one per future position)." As a result, "full transformer decoder blocks" is now a property shared by both MTP and Blockwise Parallel Decoding. A reader who consults the table and then reads line 96 will find the justification incomplete: the sentence only contrasts MTP with Medusa-style linear projections and does not explain why MTP is expected to outperform Blockwise Parallel Decoding despite both using full transformer decoder blocks. The actual distinguishing factor for MTP — absent in Blockwise Parallel Decoding — is the shifted-token-embedding conditioning (the `hnorm`/`enorm` combination input). Fix: revise line 96 to make clear that the "full transformer decoder blocks" contrast is against Medusa only, and that the differentiating factor for MTP vs. Blockwise Parallel Decoding is the shifted-token-embedding conditioning of the input combination step.

---

# B Review — Pass 10

1. **`mtp_training_objective.md`, line 96 — Pass 9 fix is coherent; no residual incoherence in the revised sentence**
   The revised sentence now reads: "Compared to Medusa heads (Cai et al., 2024), which are simple linear projections with no transformer blocks, MTP heads achieve higher draft acceptance rates due to their shifted-token-embedding conditioning (`hnorm`/`enorm` input combination). Compared to Blockwise Parallel Decoding (Stern et al., 2018), which also uses full transformer decoder blocks, MTP's key differentiator is the same shifted-token-embedding conditioning — the MTP head receives the ground-truth next token as an explicit input to each prediction depth." The two comparisons are now structurally parallel and non-contradictory: the Medusa contrast is on block depth, the Stern contrast is on input conditioning. The table row for Blockwise Parallel Decoding ("Full transformer decoder blocks (one per future position)") is consistent with the prose. The Pass 9 fix is coherent.

2. **`mtp_training_objective.md`, comparison table (line 94) — "What is predicted" cell for Blockwise Parallel Decoding uses the term "independent prediction heads," which is now inconsistent with the same row's "Gradient flows through" cell**
   The row reads: "Blockwise parallel decoding | Future tokens via independent prediction heads | Full transformer decoder blocks (one per future position) | Can be used directly for draft tokens." The Pass 8 fix correctly updated "Gradient flows through" to "Full transformer decoder blocks," but "independent prediction heads" in the "What is predicted" column was not updated. "Heads" carries a strong connotation of lightweight linear projection layers (as in attention heads or Medusa heads), directly at odds with the now-correct characterization of Stern's method as using full transformer decoder blocks. A reader scanning the table will find the two cells in the same row contradictory: the "What is predicted" cell implies shallow heads, the "Gradient flows through" cell correctly states full decoder blocks. Fix: change the "What is predicted" cell for Blockwise parallel decoding from "Future tokens via independent prediction heads" to "Future tokens via sequential full decoder blocks."

---

# B Review — Pass 11

No feedback — chapter approved.
