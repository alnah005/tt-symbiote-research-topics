# Compression Analysis — Chapter 5: Fused Activation Strategies — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~450 lines
- Estimated post-compression line count: ~330 lines
- Estimated reduction: ~27%

---

## CRUCIAL Suggestions

### [ttnn_fused_activation_api.md] ~lines 127–141
**Issue:** The "Wormhole B0 Hardware Context" table (L1 per core: 1.5 MB, 80 Tensix cores, BF16 tile size 2,048 bytes, bfloat8_b tile size 1,024 bytes, SiLU arithmetic intensity ~0.5 FLOP/byte, SiLU latency fraction 4–8%) is a verbatim re-statement of hardware constants already established in `ch02_silu_on_wormhole_hardware/tensix_compute_engine.md` (L1, Tensix pipeline table, tile sizes, SFPU description) and the SiLU arithmetic intensity figure already derived and placed on the roofline in `ch04_silu_vs_matmul_comparison/roofline_analysis.md` (Section 2, explicitly 0.5 FLOP/byte). The "4–8% of gate_proj at 128 tokens" figure is the direct output of `ch04_silu_vs_matmul_comparison/compute_vs_memory_bound_regimes.md` Table 3/Summary. This entire section restates previously established facts without adding new context.
**Suggestion:** Delete the full "Wormhole B0 Hardware Context" section (lines 127–141). Replace it with a single cross-reference sentence: "For the hardware constants that determine when these savings are meaningful — L1 per core, tile sizes, SiLU arithmetic intensity, and the 4–8% latency fraction at 128 tokens — see Chapter 2, `tensix_compute_engine.md` and Chapter 4, `roofline_analysis.md`."

### [swiglu_fusion_pattern.md] ~lines 7–24
**Issue:** The "SwiGLU Computation Graph" section restates the SwiGLU formula (`SwiGLU(x) = silu(gate_proj(x)) * up_proj(x)`), expands it into the four-op sequence (gate matmul → silu → up matmul → element-wise mul), and confirms it operates on the same shape. This is a near-verbatim repetition of content already canonically defined in `ch01_silu_in_moe_architecture/swiglu_variant.md`: the SwiGLU math (lines 46–64 of that file), the four-op expansion, and the standalone vs. fused code paths (lines 75–103). Chapter 1 is the declared canonical SwiGLU definition source.
**Suggestion:** Replace the "SwiGLU Computation Graph" section with a one-sentence pointer: "SwiGLU is defined in Chapter 1, `swiglu_variant.md` as `SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x)`, which expands to four ops: gate matmul, SiLU, up matmul, element-wise multiply on the same shape." This preserves local context (the sentence is needed to set up the Fusion Challenge) without duplicating the canonical source.

### [index.md] ~lines 62–67 (the "How This Chapter Fits in the Guide" section)
**Issue:** This section re-narrates the conclusions of Chapters 2 and 4 in full declarative sentences — "Chapter 2 established that SiLU runs on the SFPU as a sequential pass…" and "Chapter 4 showed that SiLU latency is 4–8% of gate_proj matmul time (at 128 tokens)" — material already present verbatim in the Prerequisites table (lines 22–25) of the same `index.md` file. The Prerequisites table already covers both cross-chapter dependencies with identical content. This makes "How This Chapter Fits" a third re-statement of the same two points in the same file.
**Suggestion:** Delete the "How This Chapter Fits in the Guide" section in full (6 lines). The Prerequisites table already provides the backward cross-references; the forward pointer to Chapter 6 (also in this section) can be folded into the "Next Steps" section at the bottom of the file, which already exists and points forward.

---

## MINOR Suggestions

### [ttnn_fused_activation_api.md] ~lines 103–108
**Issue:** The sentence "For `num_tokens=8`, `hidden_dim=2048`, BF16: the activation tensor is `8 * 2048 * 2 = 32,768 bytes` — 32 KB that must be written to L1 by the matmul kernel, then read back and rewritten by the SiLU kernel" is repeated almost word-for-word in `swiglu_fusion_pattern.md` at lines 65–66: "At `num_tokens=8`, `d_ff=2048`, BF16, this means 32 KB of L1 traffic is eliminated: the 8 * 2048 * 2 = 32,768-byte activation tensor does not need to be written by the matmul kernel and read back by the SiLU kernel." The 32 KB example calculation appears a third time in `swiglu_fusion_pattern.md` line 184.
**Suggestion:** Consolidate to one occurrence in `ttnn_fused_activation_api.md` (the API reference file where the kernel semantics are defined). In `swiglu_fusion_pattern.md`, replace the repeated calculation with "…eliminates the 32 KB intermediate tensor described in `ttnn_fused_activation_api.md`."

### [activation_dtype_and_precision.md] ~lines 66–78
**Issue:** The L1 Footprint Reduction section restates Wormhole B0 hardware specs (1.5 MB L1 per core, 80 cores, 8×10 grid) that are canonical in Chapter 2. While the application to `activation_dtype` is new context, the hardware spec sentence at line 68 is pure restatement: "Wormhole B0 provides 1.5 MB of L1 per Tensix core across 80 cores (8×10 grid)."
**Suggestion:** Remove the hardware-spec preamble sentence from line 68 and begin the section directly with the application: "For sharded SwiGLU computations with large hidden dims, the activation tensor is split across cores…" A footnote cross-reference to Chapter 2 is sufficient for readers who need the spec.

### [swiglu_fusion_pattern.md] ~lines 97–103
**Issue:** The "Practical Recommendation" section (7 lines) largely restates what is already implied by the Pattern A vs. Pattern B comparison directly above it, and the cross-reference to Chapter 4 (`compute_vs_memory_bound_regimes.md`) for the token-count threshold is already present in the `index.md` Prerequisites table.
**Suggestion:** Condense to three lines: retain the "Use Pattern A as the default" statement and the reference to Chapter 4 for threshold validation, and remove the sentence that explains what Pattern A is implementable through (already explained in the Pattern A section above).

### [index.md] ~lines 40–47 ("Summary of Available Fusion Mechanisms" table)
**Issue:** This three-row table summarizing `activation="silu"`, `fused_activation`, and `activation_dtype` is nearly a verbatim subset of the detailed content in `ttnn_fused_activation_api.md`. While a chapter index may reasonably include a brief orientation table, this one includes the `activation_dtype` field with a specific recommendation (`ttnn.bfloat8_b` to halve L1 footprint), pre-empting the full treatment in `activation_dtype_and_precision.md`.
**Suggestion:** Remove the `activation_dtype` row from the table (it belongs in the dtype file, not the index), and add a note that dtype details are in `activation_dtype_and_precision.md`. This keeps the table as orientation-only without duplicating the downstream file's content.

---

## Load-Bearing Evidence

Not required — VERDICT is "Crucial updates: yes."

---

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- ttnn_fused_activation_api.md: Replaced "Wormhole B0 Hardware Context" table with Chapter 2 + Chapter 4 cross-reference
- swiglu_fusion_pattern.md: Collapsed "SwiGLU Computation Graph" section to one-sentence pointer to Chapter 1; kept Fusion Challenge setup
- index.md: Removed redundant "How This Chapter Fits in the Guide" subsection; folded Chapter 6 pointer into Next Steps

---

## Pass 2 Verification

### Fix Verification

**Fix 1 — `ttnn_fused_activation_api.md`: "Wormhole B0 Hardware Context" table → Chapter 2 + Chapter 4 cross-reference**
CONFIRMED. The table is gone. Line 127 now reads a single sentence cross-reference: "For Wormhole B0 hardware constants and SiLU arithmetic intensity context, see Chapter 2 (`tensix_compute_engine.md`) and Chapter 4 (`roofline_analysis.md`)." The fix matches the Pass 1 suggestion precisely.

**Fix 2 — `swiglu_fusion_pattern.md`: "SwiGLU Computation Graph" section collapsed to one-sentence pointer**
CONFIRMED. Lines 7–8 now read: "SwiGLU computes `silu(gate_proj(x)) * up_proj(x)` (see Chapter 1, `swiglu_variant.md` for the full definition); only the `gate_proj` SiLU is fusible in a single `ttnn.matmul` call." The multi-line formula expansion and four-op walkthrough are gone; the Fusion Challenge section that follows remains intact and functional.

**Fix 3 — `index.md`: "How This Chapter Fits in the Guide" removed; Chapter 6 pointer folded into Next Steps**
CONFIRMED. No "How This Chapter Fits in the Guide" heading or content is present. The Next Steps section (line 63) now includes: "After completing this chapter, Chapter 6 (`configuration_recommendations.md`) synthesizes the fusion recommendations into a decision table for production MoE configurations on Wormhole B0 and T3K." Both the removal and the forward-pointer fold were applied correctly.

---

### Remaining Duplication Scan

No new crucial duplications were found. The remaining items from Pass 1's MINOR list are still present and unaddressed (expected — they were minor):

- The 32 KB L1 calculation (`8 * 2048 * 2 = 32,768 bytes`) appears in `ttnn_fused_activation_api.md` line 108, `swiglu_fusion_pattern.md` line 48, and `swiglu_fusion_pattern.md` line 167. Not crucial: each occurrence adds local narrative context and the duplication was already triaged as minor.
- `activation_dtype_and_precision.md` line 68 still opens with the hardware-spec preamble ("Wormhole B0 provides 1.5 MB of L1 per Tensix core across 80 cores (8×10 grid)."). Not crucial: the sentence is one line and the surrounding content applies it to a new context (sharded activation footprint), which differs from the canonical Chapter 2 treatment.
- `index.md` lines 40–47 still include the `activation_dtype` row in the "Summary of Available Fusion Mechanisms" table. Not crucial: the row is orientation-only and does not replicate the full analysis in `activation_dtype_and_precision.md`.

---

## Crucial updates: no

## Load-Bearing Evidence

- **`swiglu_fusion_pattern.md` line 7:** "SwiGLU computes `silu(gate_proj(x)) * up_proj(x)` (see Chapter 1, `swiglu_variant.md` for the full definition); only the `gate_proj` SiLU is fusible in a single `ttnn.matmul` call." This sentence cannot be cut: it is the setup premise for the entire Fusion Challenge section that follows. Without it, the distinction between what is fusible and what is not has no grounding in this file.
- **`ttnn_fused_activation_api.md` line 127:** "For Wormhole B0 hardware constants and SiLU arithmetic intensity context, see Chapter 2 (`tensix_compute_engine.md`) and Chapter 4 (`roofline_analysis.md`)." This cross-reference sentence cannot be cut: it is the replacement for the deleted table and is the only pointer in this file directing readers to the hardware context that governs when the fusion savings are meaningful.
- **`index.md` lines 61–63 (Next Steps):** The Chapter 6 forward pointer ("After completing this chapter, Chapter 6 synthesizes the fusion recommendations…") cannot be cut: it was the surviving content from the deleted "How This Chapter Fits in the Guide" section, and it is now the only forward navigation in the entire chapter index.
- **`activation_dtype_and_precision.md` line 68:** "Wormhole B0 provides 1.5 MB of L1 per Tensix core across 80 cores (8×10 grid)." This is the one instance flagged as minor duplication. It cannot be cut without also revising the following calculation (lines 70–78) that references the per-core L1 budget. The sentence is load-bearing for the arithmetic in its own paragraph, even though the spec is canonical in Chapter 2.

## MINOR Suggestions

- **`swiglu_fusion_pattern.md` lines 48 and 167:** The 32 KB L1 elimination calculation (`8 * 2048 * 2 = 32,768 bytes`) appears twice in this file alone (lines 48 and 167) in addition to once in `ttnn_fused_activation_api.md` line 108. The line 167 occurrence ("At these dimensions with `num_tokens=8`, Pattern A eliminates a 32 KB L1 round-trip per expert per forward pass") could be shortened to "Pattern A eliminates the 32 KB L1 round-trip per expert described in `ttnn_fused_activation_api.md`," removing the redundant calculation while keeping the per-expert framing that is unique to the model applicability context.
- **`activation_dtype_and_precision.md` line 68:** The hardware-spec preamble sentence remains a minor cleanup opportunity. Replacing it with a parenthetical cross-reference — e.g., "For sharded SwiGLU computations with large hidden dims (Wormhole B0: 1.5 MB L1 per core, see Chapter 2), the activation tensor is split across cores." — would preserve the L1 budget number inline without a standalone spec sentence.
