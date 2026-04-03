# Agent B Review: Chapter 1 — Architecture Overview

## Pass 1

1. **File:** `layer_organization.md`, lines 8–11 and throughout; also `index.md` line 43; `heterogeneous_attention_configs.md` line 21.
   **Error:** The chapter states there are **48 sliding-window layers and 12 global layers**. The actual `config.json` for `google/gemma-4-31B-it` contains a `layer_types` array with **50 sliding_attention entries and 10 full_attention entries**. The full_attention indices are 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 — that is 10 layers, not 12. This error propagates to every mention of "48 sliding" and "12 global" across all four files (including the plan's line 43 description and the K=V savings calculation in `novel_components.md` line 30 which says "Across 12 global layers this is a saving of roughly 264 MB" — it should be 10 layers, ~220 MB).
   **Fix:** Replace all instances of "48 sliding" with "50 sliding" and "12 global" with "10 global." Update the K=V memory savings in `novel_components.md` from "264 MB" to "~220 MB" (10 × ~22 MB).

2. **File:** `heterogeneous_attention_configs.md`, lines 62–63.
   **Error:** The global attention parameter count is stated as **186,646,528** but the arithmetic `5376 × 16384 + 5376 × 2048 + 16384 × 5376` yields **187,170,816** (= 88,080,384 + 11,010,048 + 88,080,384). The stated number is wrong by 524,288.
   **Fix:** Replace `186,646,528` with `187,170,816`.

## Pass 2

**No feedback — chapter approved.**

Verification summary:
- Pass 1 fix #1 (layer counts): Confirmed applied. All four chapter files now correctly state 50 sliding / 10 global layers. The global layer indices (5, 11, 17, 23, 29, 35, 41, 47, 53, 59) are correct. The K=V savings in `novel_components.md` correctly reads "10 global layers" and "~220 MB."
- Pass 1 fix #2 (parameter count): Confirmed applied. `heterogeneous_attention_configs.md` now states 187,170,816, which matches the arithmetic (88,080,384 + 11,010,048 + 88,080,384).
- All config.json parameter values in the quick-reference table (`index.md`) were verified against the official `google/gemma-4-31B` config and are correct.
- RoPE parameters (theta, partial_rotary_factor, rope_type) for both layer types are correct.
- Projection weight shapes and per-layer parameter counts are arithmetically correct.
- The K=V sharing dataflow description accurately reflects the code behavior (shared k_proj, divergent k_norm+RoPE vs v_norm paths).
- No factual errors, no implementation-misleading statements, no critical structural gaps found.
