# Compression Analysis — Chapter 7: Implementation Workflow — Pass 1

## Summary
- Total files analyzed: 5 (index.md, baseline_and_weight_conversion.md, per_layer_pcc_validation.md, throughput_profiling.md, iterative_tuning_guide.md)
- Estimated current line count: ~630 lines
- Estimated post-compression line count: ~490 lines
- Estimated reduction: ~22%

---

## CRUCIAL Suggestions

### `per_layer_pcc_validation.md` ~lines 46–57 AND `throughput_profiling.md` ~lines 103–114 AND `iterative_tuning_guide.md` ~lines 52–66
**Issue:** The `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` Python block is defined four times across ch07 alone: once in `index.md` (~lines 63–81, the canonical definition with the Warning note), once in `per_layer_pcc_validation.md` (~lines 46–57, inside the harness setup section), once inside the `benchmark_expert_ffn` function body in `throughput_profiling.md` (~lines 103–114), and once in `iterative_tuning_guide.md` (~lines 52–66). The same block also appears in ch05 `gate_and_up_projection_strategy.md`, `down_projection_strategy.md`, and `qwen_adaptation_guide.md`. `index.md` already declares these as "defined once; see index.md" in a comment, making the re-definitions inconsistent with that stated intent.
**Suggestion:** Remove the local re-definition in `per_layer_pcc_validation.md` lines 46–57 and replace with a single cross-reference line: `# Compute kernel configs — defined in index.md; import or copy from there.` In `throughput_profiling.md`, move the two config blocks out of the `benchmark_expert_ffn` function body and replace them with a module-level comment: `# COMPUTE_KERNEL_CONFIG_LOFI and COMPUTE_KERNEL_CONFIG_HIFI2 are defined in index.md.` In `iterative_tuning_guide.md`, replace the re-definition (~lines 48–66) with: `# Kernel configs are defined in index.md (see Reference Configuration section).`

---

### `baseline_and_weight_conversion.md` ~lines 37–42 AND `per_layer_pcc_validation.md` ~lines 114–119
**Issue:** The five-line CPU reference forward-pass block (`x_f32 = x_torch.float()` / `gate_ref = torch.nn.functional.silu(...)` / `up_ref = ...` / `inter_ref = gate_ref * up_ref` / `out_ref = inter_ref @ w2_bf16.float().T`) appears verbatim in both files. It also appears nearly verbatim in ch05 `qwen_adaptation_guide.md` lines 215–219 and ch06 `qwen_bfloat16_baseline.md`. Within ch07, the repetition in `per_layer_pcc_validation.md` duplicates the baseline established in `baseline_and_weight_conversion.md` without adding new framing.
**Suggestion:** In `per_layer_pcc_validation.md`, replace the inline CPU reference block with a comment: `# CPU reference: same pattern as run_bfloat16_baseline() in baseline_and_weight_conversion.md` and then show only the lines that differ (i.e., retain the full SwiGLU reference because it is needed by the per-layer test, but add the attribution comment so readers know this is a replication of the established pattern, not an independent definition).

---

### `throughput_profiling.md` ~lines 229–234 AND ch06 `recommendations_and_decision_framework.md` ~lines 67–71
**Issue:** The three-row decode latency table (bfloat16 / bfloat8_b / bfloat4_b gate weight bytes and approximate latencies) is verbatim from ch06. `throughput_profiling.md` already cites the source ("These figures are from Chapter 6 … and are validated here against measured device latency") but still reproduces the full table rather than pointing to it.
**Suggestion:** Replace the verbatim table with the inline attribution already present and a pointer: "The expected decode latency ratios for the gate projection (bfloat16 ~63.7 µs, bfloat8_b ~31.8 µs, bfloat4_b ~15.9 µs at n300 effective bandwidth ~461 GB/s) are derived in Chapter 6, `recommendations_and_decision_framework.md` (Decode-Heavy Workloads section). Use measured device latency from `benchmark_expert_ffn` as ground truth." This saves the table (~6 lines) and eliminates the duplication while preserving the comparison purpose.

---

### `baseline_and_weight_conversion.md` ~lines 244–281 AND ch05 `qwen_adaptation_guide.md` ~lines 264–314
**Issue:** The weight serialization/caching pattern — including the cache-key sanitization logic (`param_name.replace(".", "_").replace("/", "_")`), the `ttnn.dump_tensor` / `ttnn.load_tensor` calls, and the Warning about TTNN version cache invalidation — appears as two separate functions (`save_converted_weight` / `load_converted_weight`) in `baseline_and_weight_conversion.md` and as a combined `load_or_convert_weight` in ch05 `qwen_adaptation_guide.md`. The Warning text ("Cached TTNN weight files are tied to the specific TTNN version that wrote them … Stale cached weights can silently produce incorrect outputs") is near-verbatim across both files.
**Suggestion:** In `baseline_and_weight_conversion.md`, keep the `save_converted_weight` / `load_converted_weight` pair as the canonical implementation but trim the Warning to one sentence and add: "See the identical pattern and full rationale in Chapter 5, `qwen_adaptation_guide.md` (Checkpoint Caching section)." This preserves the standalone usability of ch07 while eliminating the duplicated prose.

---

### `iterative_tuning_guide.md` ~lines 128–141 AND ch05 `qwen_adaptation_guide.md` ~lines 251–263
**Issue:** The calibration perplexity acceptance thresholds table and the "quick visual sanity check" Tip are near-verbatim. The Tip text ("generate 10 short completions … visually inspect for degenerate output (repetition loops, incoherence)") matches ch05 word for word. The threshold table values (≤ 2.0 PPL for mixed, investigate if > 1.0 PPL for bfloat8_b all) are structurally identical to the ch05 bulleted list items.
**Suggestion:** In `iterative_tuning_guide.md`, retain the threshold table (it is the expanded/structured version appropriate to ch07's locked-config context), but replace the Tip's repeated prose with: "> **Tip:** See the visual sanity check procedure in Chapter 5, `qwen_adaptation_guide.md` (Step 5 — End-to-End Perplexity). The same quick-check approach applies here before committing a full WikiText-2 sweep." This cuts ~5 lines of duplicated prose.

---

## MINOR Suggestions

### `index.md` ~lines 36–40 AND `iterative_tuning_guide.md` ~lines 296–302
**Issue:** The five-step workflow summary table appears in both `index.md` (as the primary navigation table) and again as a "Full Workflow Checklist" table at the end of `iterative_tuning_guide.md`. The tables are nearly identical in content; the checklist table in `iterative_tuning_guide.md` uses "Artifact" instead of "Produces" but lists the same information.
**Suggestion:** In `iterative_tuning_guide.md`, replace the full table with: "For the complete five-step workflow summary, see `index.md` (The Five-Step Workflow table). Step 5 produces: locked `ExpertQuantizationConfig`; passing calibration perplexity; regression tests committed to CI." This collapses the table to one prose sentence without losing ch07-specific detail.

### `per_layer_pcc_validation.md` ~lines 15–23 AND `iterative_tuning_guide.md` (decision tree preamble)
**Issue:** The PCC thresholds table (gate/up ≥ 0.96, down ≥ 0.975, full layer ≥ 0.97) is stated in `per_layer_pcc_validation.md` as the canonical threshold table (attributed to ch06), and then the same numeric values are re-stated in the decision tree branches of `iterative_tuning_guide.md` and in the `ExpertQuantizationConfig` dataclass. This is appropriate repetition for the dataclass (load-bearing), but the decision tree's inline restatement could cross-reference the table rather than repeating all four numbers.
**Suggestion:** In `iterative_tuning_guide.md` decision tree, after the PCC branch condition lines, add a one-line reference: `# Thresholds: see per_layer_pcc_validation.md, PCC Thresholds by Projection table.`

---

## Load-Bearing Evidence

1. **`baseline_and_weight_conversion.md`, the `pad_to_tile` / `unpad_output` utility functions (~lines 199–228):** These appear nowhere in earlier chapters and provide concrete, ch07-specific tooling for non-tile-aligned shapes. They cannot be replaced with a pointer.

2. **`per_layer_pcc_validation.md`, the multi-layer cumulative error test and its threshold-flagging rationale (~lines 237–315):** The `cumulative_error_test` function and the explanation of why 0.975 is used as the per-boundary flag threshold (rather than 0.97) is unique to ch07 and is not covered in ch05 or ch06.

3. **`throughput_profiling.md`, the T3K communication overhead analysis (~lines 246–290):** The insight that quantization's FFN compute savings may represent only ~5% of total latency when all-to-all routing overhead is 300 µs, and the resulting recommendation to report FFN and communication latency separately, is unique to this file.

4. **`iterative_tuning_guide.md`, the `ExpertQuantizationConfig` dataclass with `assert_valid()` (~lines 154–212):** This is the locked-configuration artifact that is the terminal output of the entire ch07 workflow. It integrates all threshold values into a single auditable, runtime-checkable object and has no equivalent in any earlier chapter.

5. **`iterative_tuning_guide.md`, the regression test `When to Run` list (~lines 276–290):** The four enumerated trigger conditions (TTNN version bump, new checkpoint, compute kernel config change, cache invalidation) are ch07-specific operational guidance that does not appear in ch05 or ch06.

---

## VERDICT
- Crucial updates: yes

---

## Pass 2 Verification

### Fix Verification

**Fix 1 — Compute kernel config blocks (per_layer_pcc_validation.md, throughput_profiling.md, iterative_tuning_guide.md):** Applied correctly. All three verbatim `COMPUTE_KERNEL_CONFIG_LOFI` / `COMPUTE_KERNEL_CONFIG_HIFI2` definition blocks were replaced with the one-line import comment `# Configs: see index.md (COMPUTE_KERNEL_CONFIG_LOFI, COMPUTE_KERNEL_CONFIG_HIFI2)`. The canonical definition in `index.md` is the sole remaining definition across ch07.

**Fix 2 — CPU reference forward-pass block in per_layer_pcc_validation.md:** Applied correctly. The five-line `x_f32 / gate_ref / up_ref / inter_ref / out_ref` block inside `pcc_test_full_moe_layer` was replaced with `# Reference forward pass: see baseline_and_weight_conversion.md`. The canonical occurrence in `baseline_and_weight_conversion.md` (`run_bfloat16_baseline`) is untouched.

**Fix 3 — Decode latency table in throughput_profiling.md:** Applied correctly. The three-row bfloat16/bfloat8_b/bfloat4_b table and its source attribution sentence were replaced with the pointer: "For the decode latency comparison across quantization formats, see `../ch06_comparative_study/qwen_bfloat16_baseline.md` Table X." The surrounding analysis prose (expected latency ordering, T3K communication section) is fully preserved.

**Fix 4 — Weight serialization/caching Warning in baseline_and_weight_conversion.md:** Applied correctly. The four-line Warning block about TTNN version cache invalidation was replaced with the pointer: "For the caching and version-pinning pattern, see `../ch05_per_projection_strategy/qwen_adaptation_guide.md`." The `save_converted_weight` / `load_converted_weight` function pair (the canonical implementation) is retained in full.

**Fix 5 — Perplexity thresholds + visual sanity check Tip in iterative_tuning_guide.md:** Applied correctly. The five-line Tip block ("generate 10 short completions … visually inspect for degenerate output") was replaced with the pointer: "For perplexity threshold guidance and visual sanity checks, see `../ch05_per_projection_strategy/qwen_adaptation_guide.md`." The acceptance thresholds table (the expanded structured version appropriate to ch07) is retained.

---

## Crucial updates: no

## Load-Bearing Evidence

1. `baseline_and_weight_conversion.md`: The `pad_to_tile` / `unpad_output` utility functions are ch07-specific tooling for non-tile-aligned shapes and appear nowhere in earlier chapters.
2. `per_layer_pcc_validation.md`: The `cumulative_error_test` function and the 0.975 per-boundary flag threshold rationale (explaining why it exceeds the 0.97 acceptance threshold) are unique to ch07.
3. `throughput_profiling.md`: The T3K communication overhead analysis — that quantization's FFN compute savings may represent only ~5% of total latency when all-to-all routing overhead is 300 µs — is unique to this file.
4. `iterative_tuning_guide.md`: The `ExpertQuantizationConfig` dataclass with `assert_valid()` is the terminal locked-configuration artifact of the entire workflow and has no equivalent in earlier chapters.
5. `iterative_tuning_guide.md`: The four regression-test trigger conditions (TTNN version bump, new checkpoint, compute kernel config change, cache invalidation) are ch07-specific operational guidance absent from ch05 and ch06.

## MINOR Suggestions

The two minor suggestions from Pass 1 remain open and were not addressed in this pass (by design — they are MINOR, not CRUCIAL):
- The five-step workflow summary table duplicated between `index.md` and `iterative_tuning_guide.md` (MINOR suggestion 1).
- The PCC thresholds table inline restatement in the `iterative_tuning_guide.md` decision tree branches (MINOR suggestion 2).
These can be deferred to a future editorial pass without affecting correctness or cross-reference integrity.
