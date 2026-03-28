# Compression Analysis: Chapter 7 — Kernel Gaps and Roadmap — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~475 lines
- Estimated post-compression line count: ~425 lines
- Estimated reduction: ~10%

## CRUCIAL Suggestions

### [index.md] ~line 12 vs [existing_ttnn_primitives_survey.md] ~line 105
**Issue:** Index.md listed "four remaining gaps" as `recurrent_gated_delta_rule`, `causal_conv1d_update`, `FusedRMSNormSwishGate`, `chunk_gated_delta_rule` — omitting `causal_conv1d_fn` (prefill) and wrongly including `FusedRMSNormSwishGate` as a required gap. The survey's authoritative list is `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule` (four required), with `FusedRMSNormSwishGate` as an optional composable fifth.
**Suggestion:** Update index.md to match the survey's authoritative classification.

## Agent A Change Log — Compression Pass 1

**File edited:** `index.md`

- Updated the description of the four required custom-kernel gaps to: `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, and `chunk_gated_delta_rule`.
- Clarified `FusedRMSNormSwishGate` as "a fifth gap, optional — composable from existing `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` primitives but lacking a single fused kernel form."
- Updated the Key Take-Away paragraph to use "four required custom-kernel gaps" consistently.

---

# Compression Analysis: Chapter 7 — Kernel Gaps and Roadmap — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~439 lines
- Estimated post-compression line count: ~410 lines
- Estimated reduction: ~6%

## CRUCIAL Suggestions

None — Pass 1 crucial resolved.

Pass 1 fix verified: index.md now correctly lists `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule` as four required gaps, with `FusedRMSNormSwishGate` as an optional composable fifth. All four files are consistent on this classification.

## MINOR Suggestions

### [development_roadmap.md] ~line 46
**Issue:** "Prerequisite for: Priority 2 (conv state must also be on-device to avoid round-trip), Priority 3" — Priority 3 (FusedRMSNormSwishGate) is independent of Priority 1; the dependency claim is misleading.
**Suggestion:** Remove Priority 3 from the prerequisite list: "Prerequisite for: Priority 2 (conv state must also be on-device to avoid round-trip)."

### [index.md] ~line 16
**Issue:** Section description says "not yet implemented" which is imprecise — the survey uses the tag `[GAP — requires custom kernel]`, not "not yet implemented."
**Suggestion:** Replace "not yet implemented" with "requiring new custom kernel development."

### [tt_transformers_review.md] ~line 3
**Issue:** "All file paths are relative to the tt-metal repository root" — file paths do appear below (`models/experimental/tt_symbiote/modules/qwen_attention.py`) but the preamble adds a sentence before revealing this.
**Suggestion:** Move the file-path note to appear inline with the first path reference; remove the standalone preamble sentence.

## Load-Bearing Evidence

- `index.md` line 12: "The four required custom-kernel gaps — `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, and `chunk_gated_delta_rule`" — the authoritative naming used by all downstream references in the roadmap; must match the survey exactly.
- `development_roadmap.md` line 35: "4 × 128 × 128 × 2 = 131,072 bytes = 128 KiB — fits in a single Tensix core's 1.5 MiB L1" — the feasibility argument for Priority 1 L1-resident state; cannot be removed.
- `tt_transformers_review.md` line 55: "No off-device fallback exists for Gated Attention." — the critical distinction separating the complete Gated Attention path from the incomplete DeltaNet path.
- `development_roadmap.md` Priority rankings 1–5 with their Impact/Dependency fields — unique to this file and load-bearing for the entire roadmap; cannot be condensed.
- `existing_ttnn_primitives_survey.md` Section 8 summary count: "[AVAILABLE] 14, [AVAILABLE — not yet connected] 9, [GAP — requires custom kernel] 4" — the quantitative audit summary; must be retained exactly.

## VERDICT
- Crucial updates: no
