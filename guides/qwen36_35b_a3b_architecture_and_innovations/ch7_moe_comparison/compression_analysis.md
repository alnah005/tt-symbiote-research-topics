# Compression Analysis: Chapter 7 — MoE Architecture and Cross-Model Comparison — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~357 lines (index.md ~58, qwen36_moe_architecture.md ~159, cross_model_moe_comparison.md ~150)
- Estimated post-compression line count: ~300 lines
- Estimated reduction: ~16%

---

## CRUCIAL Suggestions

### [qwen36_moe_architecture.md] ~lines 82–100
**Issue:** The FLOP analysis is stated three times in succession with different conventions. Lines 84–86 compute the full breakdown from scratch (gate, up, down individually). Line 88 then re-derives the same total using a slightly different grouping ("Equivalently, per expert…"). Lines 96–100 repeat it a third time with yet another framing, before settling on a final answer. The three derivations arrive at numbers that are actually inconsistent (37.7M vs 56.6M), and the prose attempts to reconcile them — making the section longer, not clearer.
**Suggestion:** Pick one convention (the 56.6M = 9 × 3 × 2 × 2048 × 512 exact count is more defensible) and explain the 37.7M figure in a single bracketed parenthetical: "(37.7M if gate and up projections are counted jointly as a single matmul; 56.6M counting all three matrices independently — the model plan uses 37.7M)." Remove lines 88 and 96–100 in their current long form; one paragraph is sufficient.

### [cross_model_moe_comparison.md] ~lines 109–119 vs qwen36_moe_architecture.md ~lines 104–132
**Issue:** The DRAM bandwidth calculation — loading 9 × 3 expert weight matrices, each [2048, 512], at bfp16 = 6.3 MB per expert, 56.7 MB per layer, 2.2 GB per forward pass — is worked out in full in `cross_model_moe_comparison.md` (lines 109–119). The per-expert parameter count (9 × 3 × 2048 × 512 = 28.3M) is also computed in full in `qwen36_moe_architecture.md` (lines 121–126). The cross-model file then silently re-derives the same weight-size arithmetic, which a reader who has read the architecture file will have already seen. The 6.3 MB / 56.7 MB / 2.2 GB chain is not summarized — it is fully rederived.
**Suggestion:** In `cross_model_moe_comparison.md`, replace the element-by-element derivation (lines 111–119) with a single sentence referencing the established per-expert size: "Each expert weight set is 6.3 MB at bfp16 (established in `qwen36_moe_architecture.md`); 9 active experts per layer reads 56.7 MB, and across 40 layers ~2.2 GB per token per forward pass, dropping to ~0.55 GB at bfp4."

### [cross_model_moe_comparison.md] ~lines 91–95 vs qwen36_moe_architecture.md ~line 154
**Issue:** The count "30,720 routed expert weight matrices (256 experts × 3 matrices × 40 layers)" is computed and stated verbatim in both files. In `qwen36_moe_architecture.md` line 154 it appears as "30,720 routed expert weight matrices (256 experts × 3 matrices × 40 layers)". In `cross_model_moe_comparison.md` lines 91–92 it is computed again with the same parenthetical expansion, and then again at line 141 with a third instance of the identical formula. Three occurrences of the same arithmetic in two files is pure redundancy.
**Suggestion:** State the figure once (in `qwen36_moe_architecture.md`, where it is introduced in context of quantization). In `cross_model_moe_comparison.md`, replace both instances with the bare number "30,720" and omit the parenthetical re-derivation (readers already have it).

---

## MINOR Suggestions

### [index.md] ~lines 36–44
**Issue:** The cross-reference block names three external guides and then closes with a sentence explaining that "the present chapter focuses on architectural properties" while the guides provide "TTNN implementation details." This framing sentence is obvious from context (the reader is in a chapter called "MoE Architecture and Cross-Model Comparison," not an implementation guide) and adds no information.
**Suggestion:** Delete the closing explanatory sentence ("The present chapter focuses on the architectural properties…"). The guide list is sufficient.

### [index.md] ~lines 50–52
**Issue:** The "Relationship to Other Chapters" entries for Chapter 1 and Chapter 3 restate facts already given in the Overview (lines 6–7). The overview already says "architecturally identical to Qwen3.5 (see Chapter 3)" and "the MoE configuration examined here applies equally to both." Chapter 1's introduction of the MoE hyperparameter table is also already implied by calling this chapter a "deep dive."
**Suggestion:** Condense the Chapter 1 and Chapter 3 bullets to a single line: "**Chapter 1** and **Chapter 3** provide the hyperparameter table and confirm MoE identity between Qwen3.5 and Qwen3.6 respectively; both are superseded in detail by this chapter."

### [qwen36_moe_architecture.md] ~lines 108–112
**Issue:** The parenthetical "(Approximately 805M when rounding 257 × 3.1M)" immediately after the exact formula is redundant. The formula already gives the exact value (~808M); the approximate version is less accurate and adds no clarity.
**Suggestion:** Delete the parenthetical rounding note.

### [qwen36_moe_architecture.md] ~lines 36–42 (Shared Expert subsection)
**Issue:** "It is always active: every token passes through the shared expert regardless of which routed experts are selected. The shared expert contributes a fixed, non-routed component to the FFN output." The second sentence restates the first in different words. "Always active" and "fixed, non-routed component" describe the same property.
**Suggestion:** Keep the first sentence only; delete the second.

### [cross_model_moe_comparison.md] ~lines 36–42 (DeepSeek-V3 Shared Design Philosophy)
**Issue:** The three bullet points under "Shared Design Philosophy" (256 experts, top-8 routing, auxiliary-loss-free balancing) restate facts already given in the Summary Table and in the DeepSeek-V3 Configuration section immediately above. They add no new information — they are a compressed re-listing of what was just said.
**Suggestion:** Delete the three bullets. Keep only the synthesis sentence: "The key difference is scale: DeepSeek-V3 applies this philosophy to a model roughly 20× larger, which changes the compute and memory profile substantially but not the structural logic."

### [cross_model_moe_comparison.md] ~lines 66–71 (Gemma4 Routing Philosophy)
**Issue:** "This is the same routing scheme used by the original Mixture-of-Experts papers (Switch Transformer, GShard)." This is background trivia unrelated to the Tenstorrent deployment analysis and not referenced again anywhere in the chapter.
**Suggestion:** Delete this sentence.

### [cross_model_moe_comparison.md] ~lines 125–131
**Issue:** The expert parallelism paragraph ends with: "The all-to-all volume is proportional to B × 8 × H = B × 8 × 2048 per layer, which at small batch sizes (B=1 or B=32) is manageable." The word "manageable" is a hedge that asserts a conclusion without data. "See guides/expert_parallelism_strategies/ for T3K-specific all-to-all tuning" already defers to the authoritative source.
**Suggestion:** Delete "which at small batch sizes (B=1 or B=32) is manageable." The guide reference is sufficient.

---

## Load-Bearing Evidence

- `index.md` line ~7: "Because Qwen3.6 is architecturally identical to Qwen3.5 (see Chapter 3), the MoE configuration examined here applies equally to both. No changes to the MoE routing or expert forward pass are required for the TTNN implementation to support Qwen3.6 weights." — load-bearing because this is the chapter's primary practical conclusion for the TTNN implementer; it cannot be cut without removing actionable guidance.
- `qwen36_moe_architecture.md` line ~127: "Across 40 layers, active expert parameters total approximately 28.3M × 40 = 1.13B. Adding non-expert parameters (attention, embeddings, norms), the total active parameter count per token is approximately 3B, consistent with the model name 'A3B' (3B active)." — load-bearing because this is the derivation that justifies the "A3B" in the model name; it closes the loop between the hyperparameter table and the model's public identifier.
- `cross_model_moe_comparison.md` line ~121: "DRAM bandwidth, not compute, is the bottleneck for Qwen3.6 MoE inference. This is the expected regime for large-expert-count MoE models with small expert dimensions." — load-bearing because this is the central hardware conclusion of the chapter; all quantization and batching recommendations in surrounding sections follow from this claim.

---

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — Pass 1

- **CRUCIAL 1:** Collapsed the triple-stated FLOP derivation in `qwen36_moe_architecture.md` (lines 82–100) into a single clear derivation using three bullet lines for gate/up/down projections, with 56.6M as the primary result and 37.7M explained in a single parenthetical, removing the two redundant re-derivations.
- **CRUCIAL 2:** Replaced the element-by-element DRAM bandwidth re-derivation block in `cross_model_moe_comparison.md` (the three W_gate/W_up/W_down bullet lines and the two-step equation chain) with a single sentence stating the 6.3 MB / 56.7 MB / 2.2 GB result and cross-referencing `qwen36_moe_architecture.md` for the full derivation, while preserving the "DRAM bandwidth, not compute, is the bottleneck" conclusion sentence.
- **CRUCIAL 3:** Removed both parenthetical re-derivations of "30,720" in `cross_model_moe_comparison.md` — the display-equation derivation block in the "More expert weight tensors in DRAM" paragraph was collapsed to a bare prose count, and the inline "(256 experts × 3 matrices × 40 layers)" parenthetical in the bfp4 quantization section was dropped — leaving only the bare number "30,720" in both locations.

---

# Compression Analysis: Chapter 7 — MoE Architecture and Cross-Model Comparison — Pass 2

## Summary

- `index.md`: ~58 lines (unchanged from Pass 1 estimate)
- `qwen36_moe_architecture.md`: ~151 lines (reduced from ~159; FLOP section collapsed from ~12 lines to ~5)
- `cross_model_moe_comparison.md`: ~136 lines (reduced from ~150; DRAM derivation block collapsed to 1 line; two parenthetical re-derivations of "30,720" removed)
- Combined: ~345 lines (Pass 1 estimated post-compression ~300; actual post-compression is ~345, reflecting that Agent A applied targeted cuts without wholesale restructuring)

---

## Crucial Item Verification

**CRUCIAL 1 — Triple-stated FLOP derivation (`qwen36_moe_architecture.md` ~lines 82–100): RESOLVED.**
The section now contains a single derivation: three bullet lines giving per-matrix FLOPs, one formula computing the 56.6M total, and a one-line parenthetical "(The model plan uses 37.7M, counting gate and up projections jointly as a single matmul: 9 × 2 × 2 × 2048 × 512 = 37.7M. Both figures are internally consistent; 56.6M counts all three weight matrices independently.)" This is exactly the structure requested. No second or third re-derivation remains.

**CRUCIAL 2 — DRAM bandwidth re-derivation in `cross_model_moe_comparison.md` ~lines 109–119: RESOLVED.**
The full element-by-element chain has been replaced. Line 105 of the current file reads as a single sentence stating the 6.3 MB / 56.7 MB / 2.2 GB / 0.55 GB chain and cross-references `qwen36_moe_architecture.md` for the full derivation. The "DRAM bandwidth, not compute, is the bottleneck" conclusion is preserved on the following line. No re-derivation remains.

**CRUCIAL 3 — "30,720 expert weight matrices (256 × 3 × 40)" parenthetical appearing twice: RESOLVED.**
Both occurrences in `cross_model_moe_comparison.md` now use the bare number only. Line 91 reads "30,720 (plus 120 shared expert weight sets, for 30,840 total)" — the count of shared sets is new context, not a re-derivation of 30,720 itself. Line 127 reads "The 30,720 expert weight matrices at bfp16 require approximately…" — bare number, no parenthetical expansion. The parenthetical "256 experts × 3 matrices × 40 layers" is gone from both locations.

---

## Load-Bearing Evidence

- `index.md` line 7: "Because Qwen3.6 is architecturally identical to Qwen3.5 (see Chapter 3), the MoE configuration examined here applies equally to both. No changes to the MoE routing or expert forward pass are required for the TTNN implementation to support Qwen3.6 weights." — Cannot be cut: this is the primary actionable conclusion for an implementer deciding whether to port new code for Qwen3.6. Removing it turns the chapter into analysis-only with no engineering directive.
- `qwen36_moe_architecture.md` lines 90–92: The 56.6M FLOP formula and its 37.7M parenthetical together. Cannot be cut: the discrepancy between 56.6M and 37.7M appears in external documentation (model plan) and must be reconciled here; removing either number leaves readers unable to cross-check against published figures.
- `cross_model_moe_comparison.md` line 107: "DRAM bandwidth, not compute, is the bottleneck for Qwen3.6 MoE inference." — Cannot be cut: this sentence is the hardware conclusion that motivates every downstream recommendation in the chapter (bfp4 quantization, expert batching, all-to-all tuning). Every surrounding paragraph is a consequence of or qualification to this claim.

---

## MINOR Suggestions

### [`cross_model_moe_comparison.md`] line 105 — inline arithmetic still present in the cross-reference sentence
The replacement sentence for the DRAM derivation reads: "Each active expert weight set is 6.3 MB at bfp16 (3 matrices × 1,048,576 elements × 2 bytes)". The parenthetical "3 matrices × 1,048,576 elements × 2 bytes" is a partial re-derivation of the 6.3 MB figure that was removed from the body. It is small (one line) and does not reproduce the full chain, so it does not rise to a crucial issue; but trimming it to "(3 × ~1M params × 2 bytes at bfp16)" or simply removing the parenthetical entirely would be fully consistent with the stated goal of cross-referencing the architecture file for derivation details.

### [`qwen36_moe_architecture.md`] line 104 — rounding note still present
"(Approximately 805M when rounding 257 × 3.1M.)" immediately follows the exact formula giving ~808M. Pass 1 flagged this as a MINOR issue; it remains in the file. The approximate figure is less accurate than the formula it accompanies and adds no clarity. One-line deletion.

### [`cross_model_moe_comparison.md`] lines 36–42 — DeepSeek-V3 "Shared Design Philosophy" bullets
Three bullets restate facts from the Summary Table and the Configuration section directly above them (256 experts, top-8 routing, auxiliary-loss-free balancing). Pass 1 flagged this; the bullets remain. Removing them and keeping only the synthesis sentence ("The key difference is scale…") would save 5–6 lines without any information loss.

---

## VERDICT
- Crucial updates: no
