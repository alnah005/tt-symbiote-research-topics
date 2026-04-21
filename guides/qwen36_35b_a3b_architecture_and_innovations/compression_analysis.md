# Cross-Chapter Compression Analysis — Pass 1

## Summary

- Total files analyzed: 12 (guide index.md + ch5 × 3 + ch6 × 2 + ch7 × 3 + ch8 × 3)
- Estimated current line count: ~1,110 lines (chapters 5–8 + guide index)
- Estimated post-compression line count: ~980 lines
- Estimated reduction: ~12%

---

## CRUCIAL Suggestions

### C1 — MTP "training-vs-inference" key point stated three times at identical depth

The statement that MTP is primarily a training mechanism and is fully optional at inference is stated three separate times, at the same level of prose detail, in three distinct locations:

- `ch5_multi_token_prediction/index.md` lines 6–8 (Overview paragraph) and lines 36–38 (dedicated "Key Concept" section)
- `ch5_multi_token_prediction/speculative_decoding_inference.md` lines 96–104 ("Accuracy Guarantee" section), which again restates the lossless/zero-degradation guarantee

The chapter index (ch5/index.md) repeats the full concept in two separate places within itself (Overview paragraph AND the "Key Concept" block). The detail-level is the same in both — neither adds new information relative to the other. One of those two in-chapter locations must go; the Key Concept block at lines 35–39 of `ch5/index.md` is the more egregious repeat because the Overview two paragraphs above already covers it.

Additionally, the guide-level `index.md` Quick Reference row for MTP (line 44) already records the key fact (`mtp_num_hidden_layers=1`; speculative decoding is optional at inference). That is appropriate as a reference entry. The full prose restatements inside the chapter index are the bloat.

**Action for Agent A:** Remove the "Key Concept: Training Objective vs. Inference Tool" section from `ch5_multi_token_prediction/index.md` (lines 35–39). The Overview paragraph above it covers the same ground. The Accuracy Guarantee section in `speculative_decoding_inference.md` is load-bearing (it provides the lossless correctness guarantee, which is a distinct point from "MTP is optional") — retain it.

---

### C2 — Hybrid layer layout (30 GDN + 10 GA, 3:1 pattern) re-explained at length in Ch6 after definition in Ch1

`ch6_thinking_preservation/thinking_preservation_mechanism.md` lines 65–87 contain a full re-derivation of the hybrid architecture: "40 total decoder layers arranged in a repeating 3:1 pattern — three Gated DeltaNet layers followed by one Gated Attention layer, repeated 10 times. This produces 30 Gated DeltaNet layers and 10 Gated Attention layers."

This derivation exists because Ch6 needs to establish which layer types are affected by Thinking Preservation's increased token count. However, the layer count and pattern were already fully defined in Ch1 (referenced in the guide index Quick Reference table, line 41). Ch6 is not the authoritative source; Ch1 is. The two-sentence re-statement in Ch6 is acceptable as orienting context, but what follows — the paragraph explaining "what 3:1 means" in derivation form — is cross-chapter redundancy.

**Action for Agent A:** In `ch6/thinking_preservation_mechanism.md`, reduce the hybrid-layout explanation at lines 65–66 to a single cross-reference sentence (e.g., "Qwen3.6 uses a hybrid architecture (30 Gated DeltaNet + 10 Gated Attention layers; see Chapter 1) whose two layer types respond very differently to increased token counts."). Delete the derivational explanation that follows. The per-layer-type consequence paragraphs (lines 69–87) are load-bearing for Ch6's own argument and must be kept.

---

### C3 — Vision encoder identity (Qwen3.5 = Qwen3.6, zero TTNN changes) stated identically in both Ch8 files

`ch8_vision_encoder/index.md` lines 7–8 state: "the vision encoder is identical between Qwen3.5 and Qwen3.6 ... any TTNN implementation of the Qwen3.5 vision encoder ports directly to Qwen3.6 with no architectural changes."

`ch8_vision_encoder/vision_encoder_comparison.md` lines 3–5 state: "The vision encoder is identical between Qwen3.5 and Qwen3.6. All six configuration fields ... are unchanged ... A TTNN implementation of the Qwen3.5 vision encoder requires zero architectural changes to run the Qwen3.6 vision encoder."

These are near-verbatim repetitions of the same finding. The chapter index (ch8/index.md) is a navigation file and should carry a brief pointer, not the full statement. The comparison file is where the reader goes for the detailed finding; it is appropriate there. The chapter index is repeating at the same length what the comparison file already says.

**Action for Agent A:** Shorten the identity statement in `ch8_vision_encoder/index.md` to one sentence: "A key finding: the vision encoder is architecturally identical to Qwen3.5 — see [`vision_encoder_comparison.md`](./vision_encoder_comparison.md) for details." Remove the parenthetical elaboration in lines 7–8 that restates the "identical config / only weights differ / zero TTNN changes" points at length.

---

## MINOR Suggestions

### m1 — MTP configuration parameter table reproduced in both ch5/index.md learning objectives and mtp_architecture_and_training.md

`ch5/index.md` learning objective #1 (line 18) names `mtp_num_hidden_layers=1` and `mtp_use_dedicated_embeddings=false` inline in prose. `mtp_architecture_and_training.md` opens with a two-column table of these same parameters at lines 6–11. This is an expected and low-volume cross-file mention; the chapter index naming the config key in a learning objective is useful orientation. No change required unless Agent A wants to tighten the learning objective prose.

### m2 — DeepSeek-V3 MTP design comparison appears twice: summary in ch5/index.md and full table in mtp_architecture_and_training.md

`ch5/index.md` line 39 says "For a detailed side-by-side comparison with DeepSeek-V3's MTP design, see `mtp_architecture_and_training.md`." The comparison table itself is in `mtp_architecture_and_training.md` lines 128–137. This is correct layering — the index points to the detail file. No redundancy concern.

### m3 — "Zero TTNN changes" finding appears in guide index Quick Reference, ch3 chapter text, ch7/index.md overview, and ch8/index.md

The guide-level `index.md` Quick Reference row (line 49) records "Zero — existing Qwen3.5 TTNN implementation runs Qwen3.6 weights without modification." `ch7/index.md` lines 7–8 restates this for the MoE chapter context: "Because Qwen3.6 is architecturally identical to Qwen3.5 (see Chapter 3), the MoE configuration examined here applies equally to both." The ch7 version is brief and contextually necessary — it prevents readers who jump directly to Ch7 from wondering whether the analysis applies to their model. This is acceptable low-volume orienting context.

### m4 — Expert parallelism (32 experts per T3K device) mentioned in both qwen36_moe_architecture.md (line 145) and cross_model_moe_comparison.md (lines 113–117)

`qwen36_moe_architecture.md` line 145 states "with 256 experts across 8 T3K devices, 32 experts reside per device" as a single bullet. `cross_model_moe_comparison.md` lines 113–117 repeats this calculation in prose with the division shown explicitly. The repetition is minor: one is a bullet in a summary, the other is the main analysis section. Agent A may optionally compress the bullet in `qwen36_moe_architecture.md` to a pure cross-reference ("see `cross_model_moe_comparison.md`") to avoid the duplication, but this is not required.

---

## Load-Bearing Evidence

- **`ch5/speculative_decoding_inference.md` — Accuracy Guarantee section (lines 96–104):** Distinct from the "MTP is optional" point. This section provides the lossless-output correctness proof for the accept/reject loop — a separate, necessary claim. Must be retained.

- **`ch6/thinking_preservation_mechanism.md` — Per-layer-type KV cache consequence paragraphs (lines 69–87):** These paragraphs derive the actual memory impact (Gated DeltaNet: zero cost; Gated Attention: linear growth) and the KV scaling formula. This is the unique analytical content of Ch6 and cannot be removed even if the upstream hybrid layout re-derivation is cut.

- **`ch7/cross_model_moe_comparison.md` — Summary table and per-model analysis:** The full comparison table (lines 8–19) and the per-model sections are the primary deliverable of Ch7's second file. None of this appears elsewhere in the guide at this level of detail. No cut warranted.

- **`ch8/vision_encoder_specs.md` — Full image and video pipeline (lines 22–92):** The step-by-step processing pipeline, token count formulas, and shape summaries are unique to this file. Nothing in the guide index or chapter index approximates this depth. Must be retained in full.

- **`ch8/vision_encoder_comparison.md` — Gemma4 and LLaVA comparison tables and TTNN deployment considerations:** The Gemma4 parameter table, the LLaVA comparison, and the prefill-only / text-only-omission analysis are unique content not present in any other chapter. Fully load-bearing.

- **`ch5/mtp_architecture_and_training.md` — Parameter overhead derivation (lines 103–120):** The calculation showing the MTP dense layer is only ~0.24% of total parameters (not the naive 2.5%) is unique content. The derivation is justified by the cross-chapter context that the main layers are MoE, not dense — a point made clearly in Ch7 but applied here for the first time. Retain.

---

## VERDICT

- Crucial updates: yes

## Agent A Change Log — Pass 1

- C1 applied: Removed the "Key Concept: Training Objective vs. Inference Tool" section (lines 35–40) from `ch5_multi_token_prediction/index.md`; the Overview paragraph above it covers the same ground. The DeepSeek pointer sentence was dropped (it duplicated content already available via the Chapter Contents table).
- C2 applied: Reduced the hybrid-layout re-derivation in `ch6_thinking_preservation/thinking_preservation_mechanism.md` (lines 65–66) to a single cross-reference sentence: "Qwen3.6 uses a hybrid architecture (30 Gated DeltaNet + 10 Gated Attention layers; see Chapter 1) whose two layer types respond very differently…". The per-layer-type consequence paragraphs are preserved.
- C3 applied: Shortened the vision encoder identity statement in `ch8_vision_encoder/index.md` to one sentence pointing to `vision_encoder_comparison.md`, removing the three-sentence elaboration that duplicated the comparison file.

---

# Cross-Chapter Compression Analysis — Pass 2

## VERDICT

Crucial updates: no

---

## Summary

- `ch5_multi_token_prediction/index.md` — ~40 lines
- `ch6_thinking_preservation/thinking_preservation_mechanism.md` — ~112 lines
- `ch8_vision_encoder/index.md` — ~29 lines

---

## Load-Bearing Evidence

- **`ch5/index.md` line 6–8:** "A critical point to keep in mind throughout this chapter: **MTP is primarily a training mechanism.** Its inference-time role—enabling speculative decoding—is optional." — The Key Concept section is absent; this Overview sentence is the sole location of the training/inference distinction in the chapter index. C1 resolved.

- **`ch6/thinking_preservation_mechanism.md` line 65:** "Qwen3.6 uses a hybrid architecture (30 Gated DeltaNet + 10 Gated Attention layers; see [Chapter 1](../ch1_architecture_overview/index.md)) whose two layer types respond very differently to increased token counts from preserved reasoning." — Exactly the prescribed single cross-reference sentence; the multi-paragraph re-derivation is gone. C2 resolved.

- **`ch8/index.md` line 7:** "A key finding: the vision encoder is architecturally identical to Qwen3.5 — see [`vision_encoder_comparison.md`](./vision_encoder_comparison.md) for details." — One sentence, pointer only; the three-sentence elaboration is absent. C3 resolved.

---

## MINOR Suggestions

### m5 — ch5/index.md Learning Objective #5 could tighten phrasing

Learning Objective #5 (line 21) reads: "Walk through the speculative decoding accept/reject loop enabled by MTP at inference time." The phrase "enabled by MTP at inference time" is redundant within a chapter about MTP; "Walk through the speculative decoding accept/reject loop" is sufficient and saves three words.

### m6 — ch6/thinking_preservation_mechanism.md closing navigation link has no preceding blank line before the horizontal rule

Line 109 is `---` immediately followed by a blank line and the Next link. There is no blank line between the last body paragraph (line 107) and the `---`. Most other chapters in this guide place a blank line before the horizontal rule separator. This is cosmetic but inconsistent with the style used in ch5 and ch8.

### m7 — ch8/index.md Contents table lists only two files but the chapter Learning Objectives reference three capabilities (prefill-only encoding, text-only omission, ~300M parameter budget) that are covered across both files without indicating which file covers which capability

The table maps file-to-description, but the TTNN deployment objectives in the Learning Objectives bullet (line 17) bundle three distinct points under one dash-separated clause. Splitting them into three separate learning objectives (one per TTNN implication) would let a reader jump directly to the right file without reading both.
