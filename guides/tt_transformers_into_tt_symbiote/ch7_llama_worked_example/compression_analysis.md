# Compression Analysis: Chapter 7 — LLaMA Worked Example — Pass 1

## Summary
- Total files: 4
- Current lines: ~509
- Post-compression lines: ~493
- Reduction: ~3% (~16 lines removed)

---

## CRUCIAL Suggestions

### C1 — step1_module_map.md lines 66–70: Prose restates visible code block content

**Location:** `step1_module_map.md`, section 2 "How It Works", the paragraph immediately after the `LlamaMLP` class code block (lines 66–70).

**What is redundant:** The bullet list "What it exposes to the replacement walker: `gate_proj`, `up_proj`, `down_proj` — each an `nn.Linear`, visible in `_modules`; `act_fn` — an `nn.SiLU`, now a proper child module" is a verbatim restatement of `self.gate_proj`, `self.up_proj`, `self.down_proj`, and `self.act_fn = nn.SiLU()` that are already visible in the code block four lines above. The sentence that follows ("The `from_torch` classmethod satisfies the `register_module_replacement_dict` convention…") re-describes what the two-line `from_torch` body already shows and what "Why It Is Needed" already states.

**Suggestion:** Remove lines 66–70 entirely (the "What it exposes…" bullet list and the `from_torch` convention sentence). The code speaks for itself; the surrounding prose in "Why It Is Needed" already explains the purpose.

**Estimated saving:** ~5 lines.

---

### C2 — step1_module_map.md lines 93–96: Post-code prose restates code comments

**Location:** `step1_module_map.md`, section 2 "How the Two-Pass Replacement Works", the two sentences after the second code block (lines 93–96).

**What is redundant:** "After pass 1 all 16 `LlamaMLP` instances have been replaced by the wrapper class. The `gate_proj`, `up_proj`, and `down_proj` linears inside them are now accessible as registered children, ready for pass 2." This information is already communicated by the `# Pass 1` and `# Pass 2` comments inside the code block and by the "Why It Is Needed" section that precedes it. The sentence about `test_llama_intelligent` using `SmartTTNNLinear` ("The `test_llama_intelligent` variant follows the same two-pass pattern but substitutes `SmartTTNNLinear` for `TTNNLinear` in `nn_to_ttnn`.") duplicates what the mapping table in section 3 already shows in the `Symbiote replacement (test_llama_intelligent)` column.

**Suggestion:** Remove the two sentences at lines 93–96.

**Estimated saving:** ~4 lines.

---

### C3 — step2_precision_config.md lines 3–5: Section-opening prose purely restates the table below it

**Location:** `step2_precision_config.md`, section 1, lines 3–5 (before the `preprocess_weights_impl` code block).

**What is redundant:** "`test_llama` uses `TTNNLinear` as the replacement for all `nn.Linear` layers that the walker visits (i.e., `gate_proj`, `up_proj`, `down_proj` inside `LlamaMLP`). The attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj` are wrapped by `LlamaAttention.from_torch` into `TTNNLinear` instances regardless of the outer mapping." Both facts are visible in the summary table in this same section (rows "MLP gate / up / down proj" and "Attention Q / K / V / O proj" both show `TTNNLinear`). The same information is also covered in detail in step1 section 3 (the complete mapping table) and step1 section 2 (the two-pass replacement).

**Suggestion:** Remove these two sentences (lines 3–5) and let the section begin directly with the `preprocess_weights_impl` code block and its preceding one-liner introduction if needed. The section heading "Current Symbiote Precision in `test_llama.py`" already orients the reader.

**Estimated saving:** ~3 lines.

---

### C4 — step2_precision_config.md lines 101–103: Post-table prose restates table "Notes" column

**Location:** `step2_precision_config.md`, section 3, lines 101–103 (the two prose sentences immediately after the recommended mapping table).

**What is redundant:** "`TTNNLinearLLama` differs from `TTNNLinear` in two ways: it stores weights at `ttnn.bfloat8_b` and its `forward` is decorated with `@deallocate_weights_after`, which frees the device weight tensor immediately after each forward pass. `TTNNLinearLLamaBFloat16` applies only `@deallocate_weights_after` without changing the dtype." These two sentences are a prose expansion of the "Notes" column of the table immediately above them, which already states "stores weights at `ttnn.bfloat8_b`" and "`TTNNLinearLLamaBFloat16` applies only `@deallocate_weights_after` without changing the dtype." The sentence about `SmartTTNNLinearLLama` (line 103) is similarly covered by its own "Notes" cell.

**Suggestion:** Remove lines 101–103 entirely. The table "Notes" column carries the same information more concisely.

**Estimated saving:** ~4 lines.

---

## MINOR Suggestions

### M1 — step3_validation_and_benchmarking.md lines 29–30: Slightly verbose test description

**Location:** `step3_validation_and_benchmarking.md`, section 1, "Intelligent test" sub-section, the sentence "This test uses `SmartTTNNLinear` instead of `TTNNLinear` for the `nn.Linear` entries, excludes `lm_head` from replacement, and writes its timing data to…"

**Note:** The first part of this sentence ("uses `SmartTTNNLinear` instead of `TTNNLinear`, excludes `lm_head`") restates information from step1 and step2, but in the context of step3 it serves as a useful orientation before the reader runs the test. The CSV filenames (`llama_intelligent_timing_stats.csv`) are documented only here and are load-bearing. Keep as-is.

---

### M2 — step2_precision_config.md lines 119–126: DPL section has a mildly redundant 5-step enumeration

**Location:** `step2_precision_config.md`, section 4, "What DPL Does at Each Module", steps 1–5.

**Note:** Steps 1–5 describe DPL behavior in detail. Step 5 ("Returns the Torch output so downstream layers receive the reference value") is the only non-obvious constraint (it is what prevents error compounding), so the list as a whole is not purely redundant. Keep all five steps.

---

### M3 — index.md lines 32–35: "What This Chapter Delivers" list

**Location:** `index.md`, lines 32–35.

**Note:** This four-item list previews the chapter contents. It overlaps with the navigation table above it (lines 9–13) but adds qualitative framing ("honest inventory of current limitations") rather than just file names. Minor overlap; keep for reader orientation.

---

## Load-Bearing Evidence

The following facts are documented in only one place or constitute non-obvious operational constraints; they must not be removed:

1. **`LlamaMLP.act_fn` is a plain attribute, not a registered child module** — the sole explanation for why the wrapper is required. (`step1_module_map.md` section 2, "Why It Is Needed")
2. **`qkv_same_shape` check and GQA branch** — LLaMA 3.2-1B uses GQA (`num_attention_heads=32`, `num_key_value_heads=8`), causing `init_parameters` (not `init_fused_parameters`) to be called, producing four separate `TTNNLinear` instances. Documented only in step1 section 3 note. (`step1_module_map.md` lines 112)
3. **`SmartTTNNLinear._get_prefill_pc` early-exit for `lm_head`** — behavioral constraint not visible from the class name alone. (`step1_module_map.md` lines 134)
4. **Three reasons for excluding `lm_head`** (argmax sensitivity, memory pressure, correctness baseline). (`step1_module_map.md` section 4 and `step2_precision_config.md` section 5)
5. **`TTNNLinearLLama` is `@trace_disabled`** because `@deallocate_weights_after` is incompatible with trace capture. Documented only in step3. (`step3_validation_and_benchmarking.md` lines 106–111)
6. **`SmartTTNNLinearLLama` lacks `@trace_disabled`**, creating a potential trace-capture issue. Documented only in step3 lines 113.
7. **`@deallocate_weights_after` causes re-upload on every token** — operational consequence documented only in step3 lines 129.
8. **`LlamaAttention.forward` silently drops `attention_mask`** for batch sizes > 1. Documented only in step3 section 4. (`step3_validation_and_benchmarking.md` lines 132–140)
9. **DPL mode warning**: "Modules created via `from_parameters` (without a backing PyTorch layer) cannot be used under DPL mode." (`step2_precision_config.md` lines 150–151)
10. **N300 enablement requires 4 discrete steps** (MeshDevice, sharded linears, `TTNNDistributedRMSNorm`, updated `LlamaAttention.init_*`). (`step3_validation_and_benchmarking.md` lines 159–163)
11. **CSV filenames**: `llama_timing_stats.csv`, `llama_timing_stats_pivot.csv`, `llama_intelligent_timing_stats.csv`, `llama_intelligent_timing_stats_pivot.csv` — documented only in step3.
12. **PCC thresholds**: >= 0.99 for intermediate activations, >= 0.999 for normalization layers, < 0.95 warrants investigation. Documented only in step2 section 4 and step3 section 3.
13. **TT Transformers comment on bfloat4**: "bfp4 normally ok here but sub .99 pcc for llama 3.1 weights" — explains why `ttnn.bfloat8_b` is the practical default for `FF1_FF3`. (`step2_precision_config.md` line 55)

---

## VERDICT
- Crucial updates: yes

---

## Change Log — Pass 1 CRUCIAL fixes applied

### Fix C1 — step1_module_map.md: Removed post-code prose restating wrapper exposures and `from_torch` convention

Removed the bullet list beginning "What it exposes to the replacement walker:" and the sentence beginning "The `from_torch` classmethod satisfies…" (original lines 66–70). The code block immediately above shows the same information; the "Why It Is Needed" section already explains the purpose of the wrapper.

### Fix C2 — step1_module_map.md: Removed post-code prose restating two-pass sequence and `test_llama_intelligent` variant note

Removed the sentence "After pass 1 all 16 `LlamaMLP` instances have been replaced by the wrapper class. The `gate_proj`, `up_proj`, and `down_proj` linears inside them are now accessible as registered children, ready for pass 2." and the sentence "The `test_llama_intelligent` variant follows the same two-pass pattern but substitutes `SmartTTNNLinear` for `TTNNLinear` in `nn_to_ttnn`." (original lines 93–96). These restate code comments and the mapping table.

### Fix C3 — step2_precision_config.md: Removed section-opening sentences that restate the summary table

Removed the two sentences opening section 1 that describe `TTNNLinear` being used for MLP and attention projections (original lines 3–5). The summary table in the same section carries this information; step1 section 3 covers it in full.

### Fix C4 — step2_precision_config.md: Removed post-table prose restating the table "Notes" column

Removed the two sentences after the recommended mapping table describing `TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`, and `SmartTTNNLinearLLama` (original lines 101–103). The table "Notes" column already states these facts.

---

# Compression Analysis: Chapter 7 — LLaMA Worked Example — Pass 2

## Summary

All 4 files were re-read in full after Pass 1 edits. No new CRUCIAL redundancies were found. The Pass 1 fixes are correctly applied and the remaining prose in all four files is either load-bearing or already catalogued as MINOR in Pass 1.

## Pass 1 Verification

- **C1 (step1_module_map.md)**: Confirmed removed. The "What it exposes to the replacement walker" bullet list and the `from_torch` convention sentence are absent. Section 2 now flows directly from the code block to "### How the Two-Pass Replacement Works".
- **C2 (step1_module_map.md)**: Confirmed removed. The post-code sentences about "After pass 1 all 16 `LlamaMLP` instances…" and "The `test_llama_intelligent` variant follows the same two-pass pattern…" are absent. The second code block in section 2 is followed immediately by the section 3 heading.
- **C3 (step2_precision_config.md)**: Confirmed removed. Section 1 now opens directly with "`TTNNLinear.preprocess_weights_impl` stores weights in `ttnn.bfloat16`:" followed by the code block, without the two sentences that restated the summary table.
- **C4 (step2_precision_config.md)**: Confirmed removed. Section 3 ends cleanly at the table (line 98). The two prose sentences describing `TTNNLinearLLama`/`TTNNLinearLLamaBFloat16`/`SmartTTNNLinearLLama` differences are absent.

## CRUCIAL Suggestions

None found.

All remaining prose was evaluated:

- **index.md**: Navigation table and "What This Chapter Delivers" list both retained. The list adds qualitative framing not present in the table (already noted as MINOR M3).
- **step1_module_map.md**: The "Why It Is Needed" prose, the Notes paragraph on the attention row (GQA shape analysis, `qkv_same_shape` check, `init_parameters` vs `init_fused_parameters` branching), the `exclude_replacement` reasoning, and the "Modules Not Currently Replaced" paragraph are all load-bearing — each contains non-obvious operational constraints or facts present only in this file.
- **step2_precision_config.md**: The intro paragraph to section 3 ("distinguished primarily by weight dtype and the `@deallocate_weights_after` decorator") provides framing context not stated in the table heading; minor overlap only. The `test_llama_intelligent` sentence at the end of section 1 documents the 32-token dispatch threshold and `prefill_forward`/`decode_forward` distinction, which is not present in the table above it — load-bearing.
- **step3_validation_and_benchmarking.md**: All subsections (trace capture analysis, LM head exclusion consequences, `@deallocate_weights_after` re-upload behavior, attention mask batch-size constraint, N300 steps) contain facts documented only here — load-bearing throughout.

## MINOR Suggestions

No new MINOR suggestions beyond those already catalogued in Pass 1 (M1, M2, M3).

## VERDICT
- Crucial updates: no

## Change Log — Pass 2 fixes applied

None.
