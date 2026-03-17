# Compression Analysis — Chapter 6: Performance Impact and Recommendations

## Crucial updates: no

No crucial duplications found.

---

## Load-Bearing Evidence

Every passage in Chapter 6 that echoes earlier chapters is either an attributed synthesis summary, a chapter-specific adaptation, or a single-sentence paraphrase serving as a cross-reference anchor. Specific assessments:

**`index.md` — "Key Findings from Chapters 3–5" (lines 33–52)**
Each bullet is an abbreviated, explicitly sourced summary of conclusions from the named chapter. The wording differs from the source chapters throughout; no multi-line block is verbatim. This section is load-bearing: it provides the sole location where a reader can verify all upstream quantitative dependencies before acting on the Chapter 6 recommendations without re-reading five prior chapters.

**`when_fusion_helps.md` — Decision Framework table (lines 19–27) and decode arithmetic intensity derivation (lines 39–41)**
The table structure resembles `ch04/compute_vs_memory_bound_regimes.md` Section 6 but differs in content: Chapter 6 adds a `dtype` column, splits the 1–8 token row by `d_ff` value, and changes the "Action" framing. The arithmetic intensity derivation uses a 2048×2048 weight shape specific to ch06's decode scenario, not the 4096×8192 shape used in ch04. Both sections are load-bearing because the chapter's decision framework must be self-contained for production use without requiring the reader to cross-reference ch04 tables during configuration work.

**`when_fusion_helps.md` — T3K Multi-Chip Context (lines 74–82)**
The observation that expert parallelism keeps per-chip token count in the decode regime is stated in one paragraph in ch06. Chapter 4 and Chapter 5 do not contain a T3K-specific subsection. This is original to Chapter 6.

**`when_fusion_helps.md` — Anti-Pattern warning (lines 86–98)**
The warning that `activation="silu"` silently produces incorrect results when the gate_proj output has multiple consumers appears in no prior chapter. Original to Chapter 6.

**`configuration_recommendations.md` — Scenario 1 code block (lines 15–41)**
`ch05/swiglu_fusion_pattern.md` Pattern A shows a minimal 3-line version (`gate = ttnn.matmul(x, w1, activation="silu")`). The Chapter 6 version adds `dtype`, `memory_config`, variable annotations, and the full `up`/`hidden` sequence. It is an extended, production-oriented adaptation, not a copy.

**`configuration_recommendations.md` — Scenario 2 BFP8 program config (lines 64–115)**
The `ch05/activation_dtype_and_precision.md` validation section uses absolute/relative error metrics. Chapter 6 introduces a separate, simpler PCC-based validation function and threshold (`PCC > 0.999`) that does not appear in ch05. Both the code and threshold are new. The PCC tip about underrepresentation of the SiLU negative region in random distributions is also original to Chapter 6.

**`configuration_recommendations.md` — "Fused Activation in Sharded Matmul Program Configs" rule and "Correct vs Incorrect" code block (lines 171–192)**
`ch05/ttnn_fused_activation_api.md` line 67 contains one sentence on the same rule. Chapter 6 expands this into a full subsection with a code comparison. The "Incorrect" code comment (`# WRONG`) and the warning about silent failure are reinforcement content required in the recommendations chapter for an engineer who may not have read ch05 in detail. Duplication is intentional and minimal (a single rule restated with more context).

**`configuration_recommendations.md` — Tracy Profiler Verification Checklist (lines 196–236)**
`ch05/ttnn_fused_activation_api.md` line 84 contains a single parenthetical cross-reference to this checklist ("See Chapter 6, `configuration_recommendations.md` for the verification checklist"). The checklist itself does not exist in ch05; it is original to Chapter 6.

**`measurement_summary_and_next_steps.md` — Guide Summary bullet list (lines 102–115)**
A terminal summary of all six chapters. This is a standard capstone pattern for multi-chapter guides and does not reproduce extended verbatim text from any individual chapter. The per-chapter bullets are shorter than the corresponding chapter introductions.

---

## MINOR Suggestions

1. **`index.md` "Key Findings" vs. `when_fusion_helps.md` Decision Framework** — The `num_tokens = 16` threshold appears in both `index.md` (line 44) and `when_fusion_helps.md` (line 48) with nearly identical phrasing ("5% end-to-end FFN speedup" / "more than 5% end-to-end FFN speedup"). Consider consolidating the threshold definition to one location and cross-referencing from the other.

2. **`configuration_recommendations.md` Scenario 2 BFP8 tip (lines 117)** — The parenthetical description of SiLU's negative region behavior `(minimum ≈ −0.278 near x ≈ −1.28)` is also covered in `ch05/activation_dtype_and_precision.md` line 60–61. The tip in Chapter 6 is a single sentence; acceptable as-is, but it could simply cite ch05 rather than restate the values.

3. **`measurement_summary_and_next_steps.md` Interpretation Guide** — The "If your measured SiLU latency is higher than 20 µs for small tensor shapes" check (lines 44–47) refers to `TILE_LAYOUT` and BF16/BFP8 dtype requirements. These points are covered in ch02 and ch03 but are useful as a quick-reference diagnostic. No change required; worth keeping for standalone usability of the document.

4. **Latency table placeholder notation** — `measurement_summary_and_next_steps.md` uses `[MEASURED: ~X µs]` placeholders throughout its latency table. This is consistent with the guide's "fill in from hardware" pattern established in ch04, but the notation is slightly inconsistent: ch04 uses `[expected]` while ch06 uses `[MEASURED: ...]`. Aligning notation would reduce reader confusion, though this is cosmetic.
