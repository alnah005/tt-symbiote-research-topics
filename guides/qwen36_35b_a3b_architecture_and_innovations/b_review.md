## Cross-Chapter Pass 1

Reviewed: guide `index.md` and all content files in chapters 5–8 (ch5_multi_token_prediction, ch6_thinking_preservation, ch7_moe_comparison, ch8_vision_encoder). Chapters 1–4 were reviewed in a prior session.

---

1. **Broken cross-chapter reference — wrong chapter title in navigation footer**
   - **File:** `ch3_qwen35_vs_qwen36_differences/index.md`, line 50
   - **Issue:** The Previous-chapter link is labeled "Chapter 2 — Hybrid Attention: DeltaNet and Full Attention". The canonical title of Chapter 2 — as established in `ch2_gated_deltanet/index.md` line 1 and the guide-level `index.md` line 25 — is "Gated DeltaNet Deep Dive". The label "Hybrid Attention: DeltaNet and Full Attention" does not appear anywhere else in the guide and does not match any chapter.
   - **Fix:** Change line 50 to:
     ```
     | Previous chapter | [Chapter 2 — Gated DeltaNet Deep Dive](../ch2_gated_deltanet/index.md) |
     ```

2. **Broken cross-chapter reference — wrong chapter title in navigation footer**
   - **File:** `ch8_vision_encoder/index.md`, line 28
   - **Issue:** The Previous-chapter link is labeled "Chapter 7 — MoE Comparison". The canonical title of Chapter 7 — as established in `ch7_moe_comparison/index.md` line 1, the guide-level `index.md` line 30, and all other in-guide references (`ch6_thinking_preservation/index.md` line 30, `ch6_thinking_preservation/thinking_preservation_mechanism.md` line 113) — is "MoE Architecture and Cross-Model Comparison". The shortened label "MoE Comparison" is inconsistent.
   - **Fix:** Change line 28 to:
     ```
     **Previous:** [Chapter 7 — MoE Architecture and Cross-Model Comparison](../ch7_moe_comparison/index.md)
     ```

## Cross-Chapter Pass 2

Re-checked both issues flagged in pass 1.

1. **Issue 1 — FIXED**
   - **File:** `ch3_qwen35_vs_qwen36_differences/index.md`, line 50
   - **Status:** The previous-chapter link label now correctly reads "Chapter 2 — Gated DeltaNet Deep Dive". This matches the canonical title.

2. **Issue 2 — FIXED**
   - **File:** `ch8_vision_encoder/index.md`, line 28
   - **Status:** The previous-chapter link label now correctly reads "Chapter 7 — MoE Architecture and Cross-Model Comparison". This matches the canonical title.

No new cross-chapter issues found.

No feedback — guide approved.
