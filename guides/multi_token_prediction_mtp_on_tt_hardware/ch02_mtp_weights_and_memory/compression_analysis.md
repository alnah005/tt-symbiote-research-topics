# Compression Analysis: Chapter 2 — MTP Weights and Memory — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~524 lines (pre-edit)
- Estimated post-compression line count: ~499 lines (post-edit)
- Estimated reduction: ~5% (crucial fixes only; minor suggestions not yet applied)

---

## CRUCIAL Suggestions

### [mtp_weight_inventory.md] ~lines 133–149
**Issue:** The "MTP Head vs. One Backbone Block" comparison table (weight shapes and parameter counts) is a 13-line table with 150+ words of explanatory prose. `mtp_memory_footprint.md` lines 47–54 restates the same comparison as a 6-line "Summary Comparison" table with the same parameter and MiB figures. Both tables make the same point (MTP head ≈ 2.8% of one backbone MoE block). The inventory table is the more detailed version; the memory file's table is a redundant re-summary of it.
**Suggestion:** Remove the "Summary Comparison" table in `mtp_memory_footprint.md` (lines 45–54 including header) and replace with a single sentence cross-referencing `mtp_weight_inventory.md`. The detailed comparison table in the inventory file is the authoritative source.

### [mtp_memory_footprint.md] ~lines 28–33 AND [mtp_vs_backbone_compute_cost.md] ~lines 66–68
**Issue:** Both files re-explain that the MTP head's attention sub-layer uses the same GQA configuration as the backbone ("64 query heads, 8 KV heads, `head_dim` = 112, `hidden_size` = 7168"). This fact is established in `mtp_weight_inventory.md` (the weight table itself plus the config block at lines 15–31) and restated verbatim in two subsequent files. Combined these re-explanations span ~7 lines.
**Suggestion:** In `mtp_memory_footprint.md` lines 30–33, replace the two-sentence GQA re-explanation with a single clause ("identical GQA configuration — see `mtp_weight_inventory.md`"). In `mtp_vs_backbone_compute_cost.md` line 67, trim the parenthetical restatement of the full GQA config to just "same GQA configuration as the MTP head".

### [mtp_weight_inventory.md] ~lines 82–83 AND [mtp_vs_backbone_compute_cost.md] ~lines 41–42
**Issue:** The SwiGLU FFN mechanic (gate_proj and up_proj produce parallel intermediate projections; element-wise product after SiLU; passed through down_proj) is described at full length in both files. The compute-cost file duplicates the description verbatim.
**Suggestion:** In `mtp_vs_backbone_compute_cost.md` lines 41–42, cut the SwiGLU re-explanation entirely. Replace with "The MTP head's dense SwiGLU FFN (see `mtp_weight_inventory.md`) contributes:".

### [mtp_memory_footprint.md] ~line 54 AND [mtp_weight_inventory.md] ~line 149 AND [mtp_vs_backbone_compute_cost.md] ~line 120
**Issue:** The ratio "~2.8% the size of one full backbone MoE block" (or "2.8% of the per-block DRAM bandwidth cost") is computed and stated independently in all three content files. Each file re-derives or re-asserts the same 2.8% figure in different contexts (parameter count, BF16 MiB, DRAM streaming time) — these are numerically identical because they all flow from the same ratio of weight sizes. The per-file context is different enough that some repetition is justified, but the re-derivation prose is redundant.
**Suggestion:** This is partially load-bearing (each file's context differs), so only trim the prose re-derivation in `mtp_memory_footprint.md` line 54 — the sentence "It is approximately $304.6 / 10{,}973 \approx 2.8\%$ the size of one full backbone MoE block" can be cut since the table above it makes this obvious from the numbers, and the footnote about T3K sharding is the more useful information.

---

## MINOR Suggestions

### [mtp_weight_inventory.md] ~lines 5–12
**Issue:** The `## Introduction` lists three purposes for the weight inventory (memory placement, KV cache sizing, weight-loading filter design). Each bullet is 3–5 sentences of self-evident elaboration. Bullet 1 re-explains what "memory placement" means; bullet 3 re-explains what a weight-loading filter does. These are verbose beyond what readers need.
**Suggestion:** Trim each bullet to its first sentence only, cutting the explanatory sub-sentences (e.g. "the per-tensor breakdown also allows finer-grained placement decisions…" and "The inventory here defines the complete set of keys to either load into TTNN tensors or discard…").

### [mtp_weight_inventory.md] ~line 29
**Issue:** The sentence "Note that `head_dim = 112`, not 128. This follows directly from `7168 / 64 = 112`. This distinction matters for deriving the correct attention projection shapes." repeats information already expressed in the config block comment one line above (`head_dim = 112 # H / num_attention_heads = 7168 / 64 = 112`). The prose restatement adds nothing.
**Suggestion:** Cut the three-sentence note entirely; the code comment already carries this information.

### [mtp_memory_footprint.md] ~lines 99–100
**Issue:** "These estimates assume that MTP weights are not pipelined with backbone computation. If MTP weights are pre-fetched during the final backbone block's compute phase, the MTP DRAM streaming latency can be fully hidden, reducing the practical per-step overhead to zero." This same pipelining observation is made again in `mtp_vs_backbone_compute_cost.md` line 134.
**Suggestion:** Keep the observation in `mtp_vs_backbone_compute_cost.md` (the compute-cost file is the right home for latency-reduction strategies). Remove the duplicate two-sentence paragraph from `mtp_memory_footprint.md`.

### [mtp_vs_backbone_compute_cost.md] ~lines 102–104
**Issue:** The cross-check paragraph ("As a rough cross-check: the MTP head is equivalent to 319/936 ≈ 0.34 backbone blocks…") followed by the dense-FFN counterfactual paragraph (lines 104–105) are hedging elaborations. The cross-check restates the ratio already expressed in the equation above it. The dense-FFN counterfactual is speculative and adds no actionable information.
**Suggestion:** Cut lines 102–105 (the cross-check sentence and the dense-FFN counterfactual paragraph) entirely, or compress to: "Equivalently, ~0.34 backbone blocks, or ~0.36% of backbone compute."

### [index.md] ~lines 7
**Issue:** The single-sentence prerequisites paragraph restates five specific hyperparameter values already in `mtp_weight_inventory.md`. Readers who have read Chapter 1 don't need the recap; readers who haven't are directed to Chapter 1 via the link on line 5.
**Suggestion:** Cut the hyperparameter restatement from the prerequisites description. Keep only: "Chapter 1 established the MTP head architecture, the weight key naming convention, and the dense vs. MoE FFN distinction."

---

## Load-Bearing Evidence

- `mtp_weight_inventory.md` line ~50: `"Note that the output dimension of q_proj equals 64 × 112 = 7168 = hidden_size, so q_proj and o_proj are square matrices."` — load-bearing because this explains the non-obvious symmetry of q_proj and o_proj being square, which directly affects memory layout decisions downstream.
- `mtp_memory_footprint.md` line ~89: `"Even before accounting for the backbone's own L1 allocations…the MTP head weights alone exceed the per-chip L1 capacity by a factor of approximately 304.6 / 108 ≈ 2.8×."` — load-bearing because this is the primary conclusion about L1 infeasibility; the 2.8× figure here means something different from the 2.8% figure elsewhere (L1 overflow ratio vs. backbone-block fraction).
- `mtp_vs_backbone_compute_cost.md` line ~122: `"At AI ≈ 1.0 FLOPs/byte, the MTP head forward pass at batch=1 is approximately 889× below the ridge point — firmly in the memory-bandwidth-bound regime."` — load-bearing because the 889× figure and the ridge-point derivation are the quantitative basis for all downstream claims that MTP latency is DRAM-bound, not compute-bound.
- `index.md` line ~33: `"At decode (batch=1), it is entirely memory-bandwidth-bound with an arithmetic intensity near 1.0 FLOPs/byte."` — load-bearing summary sentence; accurately distills the chapter's central finding and orients readers before they enter the detail files.

---

## VERDICT
- Crucial updates: yes

---

## C Compression Application Log — Pass 1

- C1: **[mtp_memory_footprint.md]** Removed redundant "Summary Comparison" table (lines 45–54) that restated backbone vs. MTP parameter/MiB figures already fully presented in `mtp_weight_inventory.md`. Replaced with a single cross-reference sentence.
- C2: **[mtp_memory_footprint.md]** Trimmed the GQA re-explanation in the "Backbone Attention Weights" section (formerly lines 30–33) to a single cross-reference clause instead of re-listing all four GQA parameters.
- C3: **[mtp_vs_backbone_compute_cost.md]** Removed the SwiGLU re-explanation in the "Dense FFN FLOPs" section intro (formerly lines 41–42). Replaced with a cross-reference to `mtp_weight_inventory.md`.
- C4: **[mtp_vs_backbone_compute_cost.md]** Trimmed the GQA re-explanation in "Backbone Attention FLOPs" (formerly line 67) to a single cross-reference clause.

---

# Compression Analysis: Chapter 2 — MTP Weights and Memory — Pass 2

## Summary
- Files re-analyzed: 4
- Current line count: ~503 lines (`index.md` 44 + `mtp_weight_inventory.md` 178 + `mtp_memory_footprint.md` 124 + `mtp_vs_backbone_compute_cost.md` 157)
- Estimated post-compression: ~479 lines
- Estimated reduction this pass: ~24 lines (~5%)

## CRUCIAL Suggestions

None. All Pass 1 CRUCIAL items were correctly applied.

- C1 confirmed: The "Summary Comparison" table was removed from `mtp_memory_footprint.md`; a single cross-reference sentence now points to `mtp_weight_inventory.md § "MTP Head vs. One Backbone Block"`.
- C2 confirmed: The GQA re-explanation in `mtp_memory_footprint.md` "Comparison to One Backbone Block" is trimmed to "same GQA configuration as the backbone (see `mtp_weight_inventory.md`)".
- C3 confirmed: The SwiGLU re-explanation in `mtp_vs_backbone_compute_cost.md` "Dense FFN FLOPs" is replaced with a parenthetical cross-reference to `mtp_weight_inventory.md`.
- C4 confirmed: The GQA re-explanation in `mtp_vs_backbone_compute_cost.md` "Backbone Attention FLOPs" is trimmed to "same GQA configuration as the MTP head".

## MINOR Suggestions

### M1 (carry-over) — [mtp_weight_inventory.md] ~lines 5–12: Introduction bullets over-explain
Each of the three `## Introduction` bullets elaborates its own purpose beyond the first sentence. Bullet 1 adds "for example, placing the small layer norm weights in L1 while leaving the large projection matrices in DRAM" — a detail that belongs in Chapter 5, not here. Bullet 3 adds a full sentence restating what weight-loading filter design means. Cut each bullet to its opening sentence; the remaining content is self-evident or belongs downstream.

### M2 (carry-over) — [mtp_weight_inventory.md] ~line 29: Triple-sentence note on `head_dim = 112` duplicates the inline code comment
The note "Note that `head_dim = 112`, not 128. This follows directly from `7168 / 64 = 112`. This distinction matters for deriving the correct attention projection shapes." repeats exactly what the code comment on the preceding line (`# H / num_attention_heads = 7168 / 64 = 112`) already conveys. Cut all three sentences.

### M3 (carry-over) — [mtp_memory_footprint.md] ~lines 99–100: Pipelining observation duplicated in compute-cost file
The two-sentence paragraph "These estimates assume that MTP weights are not pipelined with backbone computation. If MTP weights are pre-fetched during the final backbone block's compute phase, the MTP DRAM streaming latency can be fully hidden, reducing the practical per-step overhead to zero." is repeated nearly verbatim at `mtp_vs_backbone_compute_cost.md` line 134. The compute-cost file is the correct home for latency-reduction strategies; remove the duplicate from `mtp_memory_footprint.md`.

### M4 (carry-over) — [mtp_vs_backbone_compute_cost.md] ~lines 102–105: Cross-check paragraph and dense-FFN counterfactual are hedging elaborations
The cross-check sentence ("As a rough cross-check: the MTP head is equivalent to 319/936 ≈ 0.34 backbone blocks…") restates the ratio already expressed by the equation immediately above it. The dense-FFN counterfactual paragraph ("Since the MTP head uses a single dense FFN while the backbone uses MoE…if the backbone used dense FFNs throughout…") is speculative and adds no actionable information. Both can be cut or compressed to: "Equivalently, ~0.34 backbone blocks, or ~0.36% of backbone compute."

### M5 (carry-over) — [index.md] ~line 7: Prerequisites paragraph restates five hyperparameter values
The single-sentence prerequisites description lists specific values (`head_dim = 112`, `hidden_size = 7168`, `intermediate_size = 2048`, GQA 64/8) that are immediately enumerated in full in `mtp_weight_inventory.md`. Readers directed to Chapter 1 via the link on line 5 do not need the recap here. Trim to: "Chapter 1 established the MTP head architecture, the weight key naming convention, and the dense vs. MoE FFN distinction."

## Load-Bearing Evidence
- `index.md` line ~33: `"At decode (batch=1), it is entirely memory-bandwidth-bound with an arithmetic intensity near 1.0 FLOPs/byte."` — load-bearing because this is the chapter's primary orienting claim; all downstream placement and latency arguments rest on the memory-bandwidth-bound conclusion.
- `mtp_weight_inventory.md` line ~50: `"Note that the output dimension of q_proj equals 64 × 112 = 7168 = hidden_size, so q_proj and o_proj are square matrices."` — load-bearing because the square-matrix shape is non-obvious and directly affects memory layout and sharding decisions in Chapter 5.
- `mtp_memory_footprint.md` line ~65: `"the MTP head weights alone exceed the per-chip L1 capacity by a factor of approximately 304.6 / 108 ≈ 2.8×"` — load-bearing because this 2.8× L1-overflow ratio is the primary quantitative basis for the conclusion that full MTP weight L1 residency is not feasible; it is distinct from the 2.8% backbone-block fraction used elsewhere.
- `mtp_vs_backbone_compute_cost.md` line ~122: `"At AI ≈ 1.0 FLOPs/byte, the MTP head forward pass at batch=1 is approximately 889× below the ridge point — firmly in the memory-bandwidth-bound regime."` — load-bearing because the 889× figure and ridge-point derivation are the quantitative foundation for all downstream claims that MTP latency is DRAM-bound, not compute-bound.

## VERDICT
- Crucial updates: no
