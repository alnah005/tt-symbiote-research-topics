# Critic Review (Pass 1) -- Chapter 8: Porting Strategy

## Issue 1: Qwen-Image attention file reference is fabricated

**File:** `model_prioritization.md`, line 117
**Claim:** "The transformer uses a **custom attention variant** (`attention_qwenimage.py`) that diverges from the standard DiT joint attention"
**Actual:** No file named `attention_qwenimage.py` exists anywhere in the TT-DiT codebase. `QwenImageTransformerBlock` (in `transformer_qwenimage.py`) directly extends the shared `TransformerBlock` from `blocks/transformer_block.py` and imports the standard attention mechanism. The Qwen-Image model reuses the same joint attention as SD3.5/Flux1/Motif.

**Impact:** This error overstates Qwen-Image's porting complexity. The model_prioritization text characterizes Qwen-Image as needing a "custom attention pattern" (ranking criterion), which inflates its difficulty relative to Motif. In reality, the transformer blocks are structurally the same as SD3.5's blocks. The true differentiator for Qwen-Image is only the encoder (Qwen2.5-VL), not the attention.

**Fix:** Remove the `attention_qwenimage.py` reference. Replace "custom attention variant" with a note that `QwenImageTransformerBlock` extends the standard `TransformerBlock` with minor modifications (e.g., text normalization, different embedding). Adjust the "Attention complexity" assessment accordingly.

---

## Verdict

1 material issue found. All other numerical claims (SD3.5 = 38 layers for the large variant, Flux1 = 19 joint + 38 single blocks, Mochi = 48 layers default, six pipeline directories, TT-Symbiote class names) verified against source code. The guide-level `index.md` has correct clickable links to all eight chapter `index.md` files. Tier distribution percentages (30/40/30) are approximate but within acceptable rounding of the actual counts (26/42/32).

---

# Critic Review (Final Pass) -- Cross-Chapter Consistency

## Issue 1: Broken cross-chapter reference in Chapter 6

**File:** `ch6_weight_loading/index.md`, line 29
**Text:** `[Chapter 1](../ch1_introduction/index.md)`
**Problem:** The link target `ch1_introduction` does not exist. The actual directory is `ch1_architecture_overview`. This is the only cross-chapter link in the entire guide that uses an incorrect directory name. The same file's link to Chapter 7 on line 30 (`../ch7_tracing_and_performance/index.md`) is correct.

**Fix:** Change `../ch1_introduction/index.md` to `../ch1_architecture_overview/index.md`.

---

## Issue 2: Guide-level index is missing sections specified in the review checklist

**File:** `index.md` (guide root)
**Problem:** The guide-level index contains only a description paragraph, a Chapter Index (with correct clickable links to all 8 chapters), and an Additional Resources link. It is missing the following sections that the review checklist requires: "How to Use This Guide", "Quick Reference", "Prerequisites", and "Source Code Pointers". The plan does specify the audience and prerequisites information (in `plan.md`), but this information is not surfaced in the guide's entry point where readers would first encounter it.

**Fix:** Add a brief "Prerequisites" section (summarizing the audience from `plan.md`), a "How to Use This Guide" note (e.g., read sequentially or jump to Ch8 for the synthesis), and a "Quick Reference" section with the key terminology from the plan's Conventions section.

---

## Checks That Passed

- **Terminology consistency:** `Module` (TT-DiT), `TTNNModule` (TT-Symbiote), `CCLManager`, `DiTParallelConfig`, and `DistributedConfig` are used consistently across all 8 chapters per the plan's terminology conventions. The Conv2d wrapper name is `TTNNConv2dNHWC` consistently in Ch3 and Ch8 (the plan uses the shorter `TTNNConv2d`, but no cross-chapter inconsistency exists within the guide itself).
- **Navigation chain:** Every chapter's last content file links to the next chapter's `index.md`. Ch1 last file -> Ch2, Ch2 -> Ch3, Ch3 -> Ch4, Ch4 -> Ch5, Ch5 -> Ch6, Ch6 -> Ch7, Ch7 -> Ch8. Ch8's last file links back to the guide index. All verified.
- **Cross-chapter references:** All forward and backward references between chapters resolve to correct file paths, with the single exception of Issue 1 above.
- **Contradictions:** No contradictions found. Ch3's description of components aligns with Ch8's tier classification. Ch8's "requires new infrastructure" items (CCLManager, DistributedLayerNorm, joint attention, Conv3d, pipeline orchestration) are exactly the items that Ch2, Ch3, Ch4, and Ch5 identify as having no TT-Symbiote equivalent. The plan's component assessment matches the implemented chapter content.
- **Concepts used before defined:** All concepts are introduced in their designated chapters before being referenced in later chapters, consistent with the cross-chapter dependency graph in the plan.
