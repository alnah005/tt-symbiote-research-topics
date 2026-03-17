# Compression Analysis: MoE Architecture Fundamentals — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~460 lines
- Estimated post-compression line count: ~385 lines
- Estimated reduction: ~16%

---

## CRUCIAL Suggestions

### [moe_overview.md] ~lines 39–54
**Issue:** The inline code comment at line 39 is extremely long and contains a parenthetical explanation of softmax-vs-sigmoid ordering differences that duplicates the discussion already present in `routing_and_sparsity.md` lines 23–25. The comment at line 49 then immediately re-states that scores require re-normalization — a point already made at line 40 and again in the function `compute_routing` in `routing_and_sparsity.md` lines 54–55. Both notes therefore appear three times across the two files.
**Suggestion:** In `moe_overview.md`, trim the comment at line 39 to `# softmax first, then top-k (Mixtral style); see routing_and_sparsity.md for sigmoid variant` and delete the duplicate note at line 49 entirely. The full treatment already lives in `routing_and_sparsity.md`.

### [moe_overview.md] ~lines 77–122 (Switch / Mixtral / DeepSeek sections)
**Issue:** The "Key parameters" summary tables at lines 81–84, 93–96, and 118–121 restate values (`num_experts`, `top_k`, `d_model`, `d_ff`) that were just given verbatim in the preceding prose sentences in the same paragraphs. The prose at line 89 ("uses `top_k = 2` with `num_experts = 8`") and the table at lines 93–96 ("num_experts = 8, top_k = 2") are exact duplicates. The same pattern holds for Switch Transformer (lines 77–84) and DeepSeek-MoE (lines 105–111).
**Suggestion:** Remove the "Key parameters" bullet lists / tables for all three model families. The values are already stated in the prose immediately above each table. If a quick-reference table is desired, consolidate all three into a single cross-model comparison table and drop the per-model tables.

### [moe_overview.md] ~lines 113–116
**Issue:** The note about per-token active expert counts ("the true per-token active expert count for DeepSeek-V2 is 8 (6 routed + 2 shared)... the true per-token active expert count for DeepSeek-MoE 16B is also 8") is a doubled clarification — the same correction is made twice within two adjacent sentences, once for DeepSeek-V2 and once for DeepSeek-MoE, using almost identical phrasing.
**Suggestion:** Merge into a single sentence: "Note: both DeepSeek-MoE 16B and DeepSeek-V2 have 8 active experts per token (6 routed + 2 shared); the percentages above count only routed experts."

### [routing_and_sparsity.md] ~lines 153–171 (Dropped Tokens section)
**Issue:** The "zero-contribution convention" is explained three times in close succession: (1) the definition paragraph at lines 153–161, (2) the "Canonical behavior" block at lines 161–163, and (3) the inline summary table at line 276. The canonical-behavior block also re-phrases the definition paragraph using different vocabulary but adds no new information beyond the Chapter 8 cross-reference.
**Suggestion:** Collapse the definition paragraph and the "Canonical behavior" block into a single paragraph. The distinction between "residual stream" and "MoE contribution" only needs to be stated once; the Chapter 8 cross-reference can be appended as a single sentence.

### [routing_and_sparsity.md] ~lines 243–267 (Sparsity Ratio derivation)
**Issue:** The algebraic derivation of `sparsity_ratio = 1 - 1/C` is repeated three times: once symbolically in the formula block (lines 243–257), once in the "Note" at lines 249–251 qualifying the no-drop condition, and once again with a fresh prose re-statement at lines 261–265 (cases (a) and (b)). Case (a) re-derives the same `1 - 1/C` result already established in the formula block, adding only the tail-latency aside. The two Notes at lines 249 and 259 both qualify the no-drop condition in nearly identical terms.
**Suggestion:** Remove case (a)'s derivation prose — it restates what the formula block already proved. Merge the two "Note" qualifications into one. Case (a) can be kept as a single sentence naming the tail-latency consequence without re-deriving the ratio.

### [moe_on_hardware.md] ~lines 58–62 (Kernel Launch Overhead)
**Issue:** The parenthetical at line 60 is a 130-word embedded aside explaining SwiGLU weight-matrix count, fused vs. unfused dispatch totals, and elementwise op fusion — all inside a single bullet point. The same SwiGLU structure (`W_gate`, `W_up`, `W_down`) was already stated in `moe_overview.md` line 91. The aside inflates a straightforward arithmetic example into a near-paragraph embedded in a bullet.
**Suggestion:** Pull the dispatch-count arithmetic out of the parenthetical into its own short sentence or a separate note. Delete the cross-reference back to `moe_overview.md` for SwiGLU structure — it is sufficient to say "3-matrix SwiGLU experts." Cut the elementwise fusion caveat; it belongs in Chapter 3 where kernel configuration is covered.

---

## MINOR Suggestions

### [index.md] ~line 35
**Issue:** The Note block contains the phrase "This is formally introduced in Chapter 2. The only tile-relevant detail in this chapter is..." followed by a parenthetical re-statement of the same constraint in different words. The parenthetical "(defined in Chapter 2; in this chapter, you need only know that `expert_capacity` must be a multiple of 32)" restates the opening sentence.
**Suggestion:** Delete the parenthetical; the opening sentence already scopes what the reader needs to know.

### [index.md] ~lines 69
**Issue:** The paragraph below the Key Terms table re-states the distinction between `sparsity pattern` and `sparsity tensor` that is already captured in the table rows directly above it (lines 67–68).
**Suggestion:** Remove the standalone paragraph; the table entries are self-explanatory with their "Introduced in" column.

### [moe_overview.md] ~line 7
**Issue:** "This allows a model to have a very large total parameter count while keeping the per-token FLOPs roughly constant — the model is large in parameter space but sparse in computation space." The em-dash clause restates the preceding clause in different words without adding precision.
**Suggestion:** Delete "— the model is large in parameter space but sparse in computation space."

### [moe_overview.md] ~lines 63–69 (Sparse vs. Dense section)
**Issue:** "The trade-off is straightforward:" followed by a sentence that re-states the trade-off already described in the preceding two paragraphs (large parameters / low FLOPs versus routing complexity). The word "straightforward" hedges without adding content.
**Suggestion:** Delete "The trade-off is straightforward:" and begin the sentence with "Sparse activation means...".

### [routing_and_sparsity.md] ~line 59
**Issue:** "The flat token dimension `T = batch * seq` is important." This sentence is redundant — the definition `T = batch * seq` is already established in the code comment at line 44 (`# [T, d_model], T = batch * seq`) and in the capacity-factor formula section.
**Suggestion:** Delete the sentence; the code already communicates this.

### [routing_and_sparsity.md] ~lines 203–205 (code comment)
**Issue:** The comment block at lines 203–205 re-explains the purpose of `reverse_mapping` using the same wording as the comment at lines 185–188, just one screen above.
**Suggestion:** Replace lines 203–205 with a single line: `# see Chapter 3 for full combine implementation`.

### [moe_on_hardware.md] ~lines 63–66
**Issue:** Two consecutive Notes at lines 64 and 66 both disclaim the accuracy of dispatch-count and latency figures with nearly identical hedging: "always re-profile on the target firmware." The second Note only adds the Mixtral 8x22B layer count, which could be a parenthetical in the preceding bullet.
**Suggestion:** Merge into one Note: "Mixtral 8x22B uses 56 layers (1,344 matmul dispatches). Dispatch latency varies with firmware and program cache state; always re-profile on the target hardware."

### [moe_on_hardware.md] ~line 74
**Issue:** "Dense accelerator throughput is achieved when matrix dimensions are large enough to tile across the full grid." This sentence repeats the arithmetic-intensity explanation given in the paragraph immediately above it (lines 72–73).
**Suggestion:** Delete this sentence; the preceding paragraph makes the same point with more precision.

---

## Load-Bearing Evidence

Not applicable — VERDICT is "Crucial updates: yes."

---

## VERDICT
- Crucial updates: yes

---

## Change Log — Agent A Pass 1 (2026-03-16)

All 6 CRUCIAL suggestions from Agent C were applied.

**Item 1 — [moe_overview.md] ~line 39 comment + line 49 duplicate note**
Trimmed the inline comment on the `probs = logits.softmax(-1)` line to `# softmax first, then top-k (Mixtral style); see routing_and_sparsity.md for sigmoid variant`. Deleted the duplicate `# Note: scores here are raw top-k softmax probs; in practice, re-normalize before combining…` comment that immediately preceded the Combine step comment.

**Item 2 — [moe_overview.md] ~lines 81–84, 93–96, 118–121 per-model Key parameters tables**
Removed the "Key parameters" bullet lists for Switch Transformer (lines 81–84), Mixtral 8x7B (lines 93–96), and DeepSeek-V2 (lines 118–121). Removed the separate "Key parameters for DeepSeek-MoE 16B" block as well. Added a single cross-model comparison table (Model / num_experts / num_shared_experts / top_k / d_model / d_ff per expert) placed after the three model description sections.

**Item 3 — [moe_overview.md] ~lines 113–116 duplicate DeepSeek active-expert-count notes**
Merged the two adjacent "Note:" sentences ("the true per-token active expert count for DeepSeek-V2 is 8…" and "the true per-token active expert count for DeepSeek-MoE 16B is also 8…") into a single sentence: "Note: both DeepSeek-MoE 16B and DeepSeek-V2 have 8 active experts per token (6 routed + 2 shared); the percentages above count only routed experts."

**Item 4 — [routing_and_sparsity.md] ~lines 153–171 Dropped Tokens definition + Canonical behavior block**
Collapsed the two-paragraph structure (definition paragraph + "Canonical behavior" block) into a single paragraph. The residual-stream vs. MoE-contribution distinction is stated once; the Chapter 8 correctness-validation cross-reference is appended as the final sentence of that paragraph. The separate `> Note on alternatives` block was folded in as well.

**Item 5 — [routing_and_sparsity.md] ~lines 243–267 Sparsity Ratio derivation**
Removed case (a)'s full re-derivation prose; case (a) is now a single sentence naming the tail-latency consequence without re-proving the ratio. Merged the two "Note" qualifications (lines 239 and 249) into one combined note covering both the no-drop condition and the tile-ceiling rounding caveat.

**Item 6 — [moe_on_hardware.md] ~line 60 Kernel Launch Overhead parenthetical**
Extracted the dispatch-count arithmetic from the 130-word parenthetical into a plain sentence: "each MoE layer requires up to 24 matmul dispatches in the unfused case: 3 weight matrices per 3-matrix SwiGLU expert × 8 experts." Replaced the cross-reference to `moe_overview.md` for SwiGLU structure with the inline phrase "3-matrix SwiGLU experts". Deleted the elementwise-op fusion caveat (belongs in Chapter 3).

---

## Change Log — Agent A Pass 2 (2026-03-16)

Both mandatory fixes from Agent B (Pass 2) were applied.

**Item 1 — [routing_and_sparsity.md] Broken `router_weight.T` call in `compute_routing`**
Changed `flat_x @ router_weight.T` to `router_weight(flat_x)` on the logits-computation line inside `compute_routing`. The original expression would raise `AttributeError` at runtime because `router_weight` is an `nn.Linear` module, not a tensor. The idiomatic PyTorch approach calls the module directly.

**Item 2 — [moe_overview.md] Incorrect Switch Transformer CF default**
Replaced the note asserting "Switch Transformer typically uses `CF = 1.0`" with a corrected statement: Switch Transformer uses `CF = 1.25` as its operational default (arXiv:2101.03961, Table 5), and `CF = 1.0` is the theoretical minimum used in the paper to analyze token-drop behavior — not the recommended setting.

---

# Compression Analysis: MoE Architecture Fundamentals — Pass 2

## Summary

- Files re-examined: 4 (`index.md`, `moe_overview.md`, `routing_and_sparsity.md`, `moe_on_hardware.md`)
- All 6 prior CRUCIAL items: **resolved** (verified against current file content)
- New CRUCIAL items: none
- MINOR items remaining from pass 1 (not yet applied): 7 of 8 from pass 1 are still present; one new MINOR item identified

---

## CRUCIAL Suggestions

Re-check results for all 6 prior CRUCIAL items:

1. **[moe_overview.md] ~line 39 — softmax-vs-sigmoid duplicate comment**: RESOLVED. Line 39 now reads `# softmax first, then top-k (Mixtral style); see routing_and_sparsity.md for sigmoid variant`. The separate duplicate re-normalization note is gone.

2. **[moe_overview.md] ~lines 77–122 — per-model "Key parameters" tables**: RESOLVED. Per-model tables removed; a single cross-model comparison table now appears at lines 105–110.

3. **[moe_overview.md] ~lines 113–116 — doubled DeepSeek active-expert-count note**: RESOLVED. Line 99 now reads the merged single sentence: "Note: both DeepSeek-MoE 16B and DeepSeek-V2 have 8 active experts per token (6 routed + 2 shared); the percentages above count only routed experts."

4. **[routing_and_sparsity.md] ~lines 153–171 — zero-contribution convention explained 3 times**: RESOLVED. The "Dropped Tokens" section is now a single paragraph covering the convention once.

5. **[routing_and_sparsity.md] ~lines 243–267 — sparsity ratio derivation repeated 3 times**: RESOLVED. Case (a) is now a single sentence; the two Note qualifications are merged into one.

6. **[moe_on_hardware.md] ~lines 58–62 — 130-word SwiGLU parenthetical**: RESOLVED. The dispatch-count arithmetic is now a plain standalone sentence; the cross-reference to `moe_overview.md` for SwiGLU structure and the elementwise-fusion caveat are removed.

**No new CRUCIAL items found.**

---

## MINOR Suggestions

The following MINOR items from pass 1 were not applied and remain in the current files. They are re-stated here for completeness, plus one newly identified item.

### [index.md] ~line 35 — parenthetical restates opening sentence (carry-over from pass 1)
The Note block opens "This is formally introduced in Chapter 2" then immediately adds "(defined in Chapter 2; in this chapter, you need only know that `expert_capacity` must be a multiple of 32)" — the parenthetical says the same thing as the sentence that precedes it.
**Suggestion:** Delete the parenthetical `(defined in Chapter 2; in this chapter, you need only know that \`expert_capacity\` must be a multiple of 32)`.

### [index.md] ~line 69 — paragraph below Key Terms table restates table content (carry-over from pass 1)
The sentence "`sparsity pattern` (introduced in this chapter) is an operational concept… The `sparsity tensor` is the TTNN data structure encoding this pattern; its format is defined in Chapter 5." duplicates the information already captured in the two table rows immediately above it.
**Suggestion:** Delete the standalone paragraph; the table is self-explanatory.

### [moe_overview.md] ~line 7 — em-dash clause restates preceding clause (carry-over from pass 1)
"— the model is large in parameter space but sparse in computation space" adds no information beyond the clause it follows.
**Suggestion:** Delete the em-dash clause.

### [routing_and_sparsity.md] ~line 59 — sentence restates code comment (carry-over from pass 1)
"The flat token dimension `T = batch * seq` is important." is redundant given the comment `# [T, d_model], T = batch * seq` at line 44 in the same function.
**Suggestion:** Delete the sentence.

### [routing_and_sparsity.md] ~lines 193–195 — `reverse_mapping` comment duplicates earlier comment (carry-over from pass 1)
The comment block at lines 193–195 (`# Note: the combine step requires the reverse mapping...`) re-explains `reverse_mapping` using nearly identical wording to the comment at lines 185–188 (`# This is required to implement the combine step: after expert FFNs run...`).
**Suggestion:** Replace lines 193–195 with `# see Chapter 3 for full combine implementation`.

### [moe_on_hardware.md] ~lines 63–66 — two consecutive Notes disclaim accuracy with near-identical hedging (carry-over from pass 1)
Lines 64 and 66 are two separate `> Note:` blocks. The first states the Mixtral 8x22B layer count; the second disclaims the 10 µs figure. Both end with a variant of "always re-profile on the target hardware/firmware." The closing hedge is repeated in full in both notes.
**Suggestion:** Merge into one Note: "Mixtral 8x22B uses 56 layers (1,344 matmul dispatches under the naive loop). Dispatch latency varies with firmware version and program cache state; the 10 µs figure is illustrative — always re-profile on the target hardware."

### [moe_on_hardware.md] ~line 74 — sentence repeats preceding paragraph's conclusion (carry-over from pass 1)
"Dense accelerator throughput is achieved when matrix dimensions are large enough to tile across the full grid." repeats the arithmetic-intensity argument made in the paragraph immediately above it.
**Suggestion:** Delete this sentence.

### [moe_on_hardware.md] ~line 101 — table cells duplicate near-identical boilerplate (new item)
In the "Comparison at a Glance" table, both the "Batched Matmul" and "`sparse_matmul`" cells for "Kernel launch count" read "Low (1–2 per MoE layer; dependent on kernel fusion configuration; see Chapter 3/4)". The qualifier phrase "dependent on kernel fusion configuration" is identical in both cells and the only difference is the chapter reference.
**Suggestion:** Collapse both cells to "Low (1–2 per MoE layer)" and add a single shared footnote or trailing sentence: "Exact count depends on kernel fusion configuration; see Chapters 3 and 4."

---

## Load-Bearing Evidence

The following passages were examined and confirmed non-redundant — they cannot be cut without removing information used downstream in this guide or needed for reader comprehension.

- **`moe_overview.md` line 5** ("A Mixture of Experts (MoE) layer is a neural network component…"): This is the primary definitional sentence for the entire guide. Every subsequent chapter uses "MoE layer", "expert", and "gating network" without re-definition. Cutting any part of this sentence removes the anchor for all forward references.

- **`moe_overview.md` lines 21–32** (Dispatch-Combine Pattern phases 1–3): The three-phase breakdown is referenced by name in `moe_on_hardware.md` ("the dispatch-combine pattern requires synchronization") and in `routing_and_sparsity.md` ("the dispatch step"). The phase numbering (steps 1–5) is used in the code comment labels at lines 37, 43, 46, 49. Cutting or condensing this block would break those cross-references.

- **`routing_and_sparsity.md` lines 76–84** (Capacity Factor definition and formula): The formula `expert_capacity = C * T * top_k / num_experts` is the foundational identity for all sparsity ratio derivations in lines 239–255 and for the capacity compute function at lines 91–99. It is also referenced by name in `moe_on_hardware.md` line 88. This block cannot be shortened without removing the load-bearing formula.

- **`routing_and_sparsity.md` lines 217–255** (Sparsity Ratio section): The derivation `sparsity_ratio = 1 - 1/C` and the two regime cases (no-drop and drop) are the factual basis for the "Comparison at a Glance" table in `moe_on_hardware.md` and the Chapter 6 decision framework. The Tip at lines 255–256 introducing the decode-mode example (`T=1, top_k=2, sparsity ≈ 0.992`) is load-bearing for Chapter 6's crossover analysis and is not a duplicate of anything else.

- **`moe_on_hardware.md` lines 86–103** (Preview: Two TTNN Strategies and comparison table): This section is explicitly cross-referenced in `index.md` learning objective 6 and provides the only reader-facing preview of Chapters 3–4. The comparison table at lines 96–103 is not a restatement of any earlier content — it introduces the "Requires sparsity tensor", "Best regime", and "Config complexity" dimensions for the first time.

---

## VERDICT

Crucial updates: no
