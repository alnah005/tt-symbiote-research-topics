# Compression Analysis — Chapter 5: Per-Projection Quantization Strategy

## Crucial updates: yes

### Duplication 1

**Source (original):** `ch01_quantization_formats/bfloat4_b_format.md`, lines 113–126 — "Expert weight example" block, including the three-projection per-expert byte calculation (w1 = 7.34 MB, w3 = 7.34 MB, w2 = 14.7 MB, total ~29.4 MB, ~3× reduction) for d_model=7168, d_ff=2048.

**Duplicate (ch05 location):** `mixed_precision_memory_layout.md`, lines 50–84 — "DRAM Footprint per Expert" section. The same three-line per-projection arithmetic, the same element counts, the same 3.0× reduction conclusion, and the same general formula (`d_model × d_ff × 2.0` vs. `d_model × d_ff × 6.0`) are all reproduced. The ch05 version adds a self-corrected recomputation paragraph and a Tip callout, but the core content is not new.

**Recommended action:** Replace lines 50–84 of `mixed_precision_memory_layout.md` with a two-sentence summary that states the per-expert mixed-precision total (28.00 MB) and the 3.0× reduction ratio, followed by a cross-reference: "See `ch01_quantization_formats/bfloat4_b_format.md` § Memory Footprint for the full derivation." Retain the Tip callout if the 2.67× vs. 3.0× clarification is considered load-bearing for Qwen specifically (it is not in ch01). The per-chip and all-chips T3K totals (lines 93–116) are Qwen/T3K-specific and should be kept in full.

---

### Duplication 2

**Source (original):** `ch04_throughput_impact/decode_memory_bandwidth.md`, lines 60–64 — "Quantized weights load in fewer cache lines per tile" table, which lists bfloat16 = 2,048 bytes / 32 cache lines, bfloat8_b = 1,024 bytes / 16 lines, bfloat4_b = 512 bytes / 8 lines per 32×32 tile.

**Duplicate (ch05 location):** `mixed_precision_memory_layout.md`, lines 39–44 — "Tile Memory Sizes" table. The three rows and byte values are verbatim (bfloat16 = 2,048, bfloat8_b = 1,024, bfloat4_b = 512 bytes per 32×32 tile). Only the column headings differ slightly.

**Recommended action:** Replace the three-row tile size table with a one-line statement giving the byte values inline (e.g., "bfloat16 tiles are 2,048 bytes, bfloat8_b 1,024 bytes, and bfloat4_b 512 bytes per 32×32 tile") and a cross-reference to `ch04_throughput_impact/decode_memory_bandwidth.md` § L1 vs DRAM Placement, and to `ch01_quantization_formats/index.md` § Format Comparison Summary. The sentence "These tile sizes are the foundation for all footprint calculations below" can be kept to preserve the local narrative thread.

---

## Load-Bearing Evidence

The content NOT flagged above cannot be removed without losing critical ch05-specific reasoning:

- **`gate_and_up_projection_strategy.md`:** The SwiGLU data flow pseudocode, SiLU error compression argument, element-wise product noise dilution math, and the 0.96 PCC validation threshold are the core mechanistic content of ch05. They are presented in ch03 `projection_sensitivity.md` from an empirical angle; ch05 provides the causal/mechanistic angle that ch03 references back to. Removing these would hollow out ch05's stated purpose.
- **`down_projection_strategy.md`:** The LayerNorm amplification argument and the per-layer error accumulation reasoning are presented nowhere else in the guide. The fallback options table (PCC range → action) is original to ch05.
- **`mixed_precision_memory_layout.md` T3K totals (lines 93–116):** The per-chip (448 MB) and system-wide (3.50 GB vs. 10.50 GB) calculations are Qwen 235B-A22B / 8-chip T3K-specific and do not appear in ch01 or ch04.
- **Compute kernel config code blocks across all ch05 files:** Although ch02 `compute_kernel_config.md` defines LoFi and HiFi2 standard configs, the ch05 configs deliberately differ on `fp32_dest_acc_en` (ch02 HiFi2: `True`; ch05 HiFi2: `False`) and on `math_approx_mode` (ch02 LoFi: `True`; ch05 LoFi: `False`). These are the validated MoE-specific settings. Replacing them with cross-references to ch02 would mislead readers into using the wrong field values. They must remain verbatim in ch05.
- **`qwen_adaptation_guide.md`:** The full five-step workflow — parameter name regex, per-projection conversion functions, PCC validation harness, caching pattern, and complete `convert_qwen_moe_checkpoint` orchestration — is entirely original to ch05. None of it appears in prior chapters.

---

## MINOR Suggestions

1. **Intra-ch05 SwiGLU pseudocode duplication:** `gate_and_up_projection_strategy.md` (lines 8–13) and `down_projection_strategy.md` (lines 8–13) both open with a near-identical 4-line SwiGLU pseudocode block. The down projection file only needs the last two lines (`inter = gate_out * up_out` and `w2_out = inter @ w2.T`) to establish context; the full repeat of the gate and up lines is unnecessary. Consider trimming `down_projection_strategy.md` to show only the relevant excerpt, with a note "...continuing from `gate_and_up_projection_strategy.md`".

2. **`COMPUTE_KERNEL_CONFIG_LOFI` defined twice in `gate_and_up_projection_strategy.md`:** The config object is defined in the "Recommended Configuration" section (lines 89–98) and again identically in the "Code Pattern" section (lines 137–142). The second definition in the code pattern is correct for copy-paste completeness, but the first standalone block is redundant. A note saying "see the Code Pattern below for the full definition" in the Recommended Configuration section would remove the repeat without losing anything.

3. **`qwen_adaptation_guide.md` Step 4 defines both configs then `qwen_expert_forward` uses them inline:** This is clean and correct. Minor note only: consider extracting the two `WormholeComputeKernelConfig` definitions to a module-level constants block (referenced in ch05 `index.md` Tip) so readers loading only the guide page have a single authoritative location per config.

4. **PCC threshold values repeated across files:** `gate_and_up_projection_strategy.md` (≥ 0.96), `down_projection_strategy.md` (≥ 0.975), and `qwen_adaptation_guide.md` (both thresholds in docstring and assert statements) all state the same thresholds. This repetition is appropriate for a procedure guide where each file may be read in isolation, so this is not flagged as crucial. If a single source-of-truth table is ever introduced (e.g., in a future ch07), the assert values in `qwen_adaptation_guide.md` are the best single place to keep live threshold values.

## Agent A Change Log — C Feedback Pass 1
- mixed_precision_memory_layout.md: Replaced step-by-step per-expert footprint derivation with 2-sentence summary + Chapter 1 cross-reference; kept Qwen/T3K system totals
- mixed_precision_memory_layout.md: Replaced tile sizes table with inline sentence + Chapter 1 cross-reference

## Pass 2 Verification

**Fix 1 — per-expert DRAM footprint derivation:** Confirmed. Lines 49–91 of the original file (the step-by-step bfloat4_b/bfloat8_b arithmetic, the self-corrected recomputation block, the general-formula code blocks, and the Tip callout) have been removed. The "DRAM Footprint per Expert" section now contains exactly the two-sentence summary: the 28.0 MB mixed-precision figure with `2 × d_model × d_ff` formula, the 84.0 MB bfloat16 baseline, the 3× reduction ratio, and the cross-reference to Chapter 1, `bfloat4_b_format.md`. The T3K system totals section ("Total Expert Weight Memory on a T3K System") is fully intact — per-chip (448 MB), all-chips mixed precision (3.50 GB), all-chips bfloat16 (10.50 GB), and DRAM headroom commentary.

**Fix 2 — tile memory sizes table:** Confirmed. The three-row table (bfloat16=2048 B, bfloat8_b=1024 B, bfloat4_b=512 B) has been removed. The "Tile Memory Sizes" section now contains a single paragraph: the introductory sentence about TTNN's 32×32 tile unit, the inline byte values for all three formats, the cross-reference to Chapter 1, `bfloat16_format.md`, and the original closing sentence "These tile sizes are the foundation for all footprint calculations below." — preserving the local narrative thread as specified.

Cross-reference accuracy: `ch01/bfloat4_b_format.md` § Memory Footprint (lines 105–135) contains the byte-level derivation for the three-projection mixed-precision expert (w1=7.34 MB, w3=7.34 MB, w2=14.7 MB, total ~29.4 MB) — the cited source is correct. `ch01/bfloat16_format.md` line 132 establishes the 2048-byte per-tile figure for bfloat16; `ch04/decode_memory_bandwidth.md` lines 60–64 is the source with all three dtype rows, but `bfloat16_format.md` is a valid ch01 anchor for the tile size concept.

### Remaining Crucial Duplications Check

Spot-checked `gate_and_up_projection_strategy.md`, `down_projection_strategy.md`, and `qwen_adaptation_guide.md` against ch01–ch04 content. No new crucial duplications found beyond those already identified as MINOR in Pass 1 (the intra-ch05 SwiGLU pseudocode repeat between gate_and_up and down_projection files, and the double definition of `COMPUTE_KERNEL_CONFIG_LOFI` within `gate_and_up_projection_strategy.md`). These remain classified as minor because the content is procedurally necessary in context and does not duplicate cross-chapter source material. No remaining crucial duplications found.

## Crucial updates: no

## Load-Bearing Evidence

The two fixes were executed exactly as specified. No load-bearing content was removed: the T3K per-chip and system totals are intact, the closing narrative sentence in the tile sizes section is intact, and all other sections of `mixed_precision_memory_layout.md` (Tensor Shapes and Dtypes table, Tile Alignment Constraint, Code Pattern, Summary Table, Next Steps) are unchanged. The two-sentence footprint summary preserves all three numbers readers need (28.0 MB, 84.0 MB, 3×) with a cross-reference for the derivation. The inline tile size sentence preserves all three byte values (2048, 1024, 512) with a cross-reference for format context.

## MINOR Suggestions

1. **Tile sizes cross-reference completeness:** The current cross-reference points to `bfloat16_format.md` for all three dtype tile sizes, but that file only explicitly derives the bfloat16 tile size (2048 B). Readers wanting to verify the 1024 B and 512 B figures would need to look in `ch04_throughput_impact/decode_memory_bandwidth.md` (lines 60–64), which has the three-format table. Consider adding a second cross-reference: "...per 32×32 tile (see Chapter 1, `bfloat16_format.md`; all three formats compared in Chapter 4, `decode_memory_bandwidth.md`)."

2. **Summary Table at end of `mixed_precision_memory_layout.md`:** The Summary Table (lines 148–155) still shows "Total (bfloat16): 1,344 MB" for per-device size. This is the per-device bfloat16 total (16 experts × 84 MB = 1,344 MB), which is correct and original to ch05. No change needed — noting only for completeness that this figure is consistent with and derived from the now-summarized per-expert footprint.
