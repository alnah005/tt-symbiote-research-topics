## Change Log — B Feedback Pass 1

**math_fidelity_evaluation.md:** Corrected the accuracy harness reference implementation from a two-weight simplified matmul (ref_gate_up = x @ w1; ref_down = silu(...) @ w2) to the correct three-weight GLU formula: ref_gate = x @ w1; ref_up = x @ w3; ref_inter = silu(ref_gate) * ref_up; ref_out = ref_inter @ w2.

# Compression Analysis: Ch4 Matmul Config and Math Fidelity — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~748 lines (index.md: ~85, program_config_tuning.md: ~353, math_fidelity_evaluation.md: ~310)
- Estimated post-compression line count: ~620 lines
- Estimated reduction: ~17%

## CRUCIAL Suggestions

### [program_config_tuning.md + math_fidelity_evaluation.md] lines 15–40 / 125–143
**Issue:** The full body of `_make_sparse_matmul_program_config` is reproduced in `program_config_tuning.md` (lines 15–40), and then a cosmetically renamed but functionally identical `make_prog_cfg` helper is re-implemented in full in `math_fidelity_evaluation.md` (lines 125–143). These 18 lines of harness code are identical in structure and parameter logic; only the function name and one default argument differ.
**Suggestion:** In `math_fidelity_evaluation.md`, replace the full `make_prog_cfg` implementation with a one-line comment pointing to `program_config_tuning.md`: `# make_prog_cfg mirrors make_config in program_config_tuning.md — see that file for the full implementation.` Then show only the call site. This removes ~16 lines of duplicated code and makes the cross-file dependency explicit.

### [program_config_tuning.md + math_fidelity_evaluation.md] lines 257–262 / 116–122
**Issue:** The `WormholeComputeKernelConfig` four-field block (`math_fidelity`, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`) is written out in full in both harnesses. It also appears a third time in `math_fidelity_evaluation.md` as the "current expert compute config" code block (lines 40–47).
**Suggestion:** Define the config once in the evaluation harness and reference it by name in the benchmarking harness. Eliminate one of the three standalone reproductions. Net saving: ~8 lines.

### [program_config_tuning.md] line 336
**Issue:** The "Expected pattern" paragraph (line 336) is 7 sentences. The first five sentences restate the memory-bound-at-batch=1 rationale that is already established in `index.md` (lines 45–47) and in `math_fidelity_evaluation.md` lines 21–22 and 264–268 — the same point appears in three separate locations.
**Suggestion:** Trim the "Expected pattern" paragraph to its unique content: the observation that `in0_block_w=11` is the notable candidate for the down projection because it cuts K-loop iterations from 11 to 4. Cut the first four sentences and the closing generalization. Net saving: ~5 lines.

### [math_fidelity_evaluation.md] lines 21–22 and 264–268
**Issue:** The memory-bound-at-batch=1 point ("sparse matmul is largely memory-bound; FPU is not the bottleneck; fidelity benefit most visible in compute-bound conditions") appears verbatim twice within the same file — once in the fidelity level table prose (lines 21–22) and again in the "Expected outcome at batch=1 decode" section (lines 264–268), which also repeats the same point from `program_config_tuning.md` line 336.
**Suggestion:** Keep the full explanation in one place — the "Expected outcome" section is the more appropriate location since it delivers the actionable inference. In the fidelity table prose, replace lines 21–22 with a forward reference: "See 'Expected outcome at batch=1 decode' below for the throughput implication at this batch size." Net saving: ~3 lines.

---

## MINOR Suggestions

### [program_config_tuning.md] lines 58–62
**Issue:** The `per_core_N` derivation for GLM-4-MoE gate/up and down is worked out again in prose (lines 61–62), reproducing the same numbers already in the `index.md` tile arithmetic table (lines 67–69) with no added explanation.
**Suggestion:** Replace the two worked-example lines with a cross-reference: "See the tile dimension arithmetic in `index.md`." Retain only the formula line (line 60).

### [program_config_tuning.md] lines 87 and 111
**Issue:** Two nearly identical sentences state that the `min(4, ...)` cap is "not binding" for GLM-4-MoE — one for gate/up (line 87) and one for down (line 111). The reasoning is the same; only the size values differ.
**Suggestion:** State the general principle once ("The `min(4, ...)` cap is only binding when `K_tiles < 4`, which requires `dim_size < 128`. Neither projection satisfies this condition."), then drop the second restatement.

### [math_fidelity_evaluation.md] lines 88–94
**Issue:** Metric 3 (routing-agreement rate) ends with the parenthetical "they always should be, since the gate is not changed" (line 94), which undercuts the metric's purpose and adds a hedging sentence that reduces rather than increases the reader's confidence in the protocol.
**Suggestion:** Delete the parenthetical clause "If routing decisions are identical (they always should be, since the gate is not changed) but..." and rephrase to simply state the condition being checked: "If the final residual output cosine similarity falls below 0.9999, precision loss in expert outputs is propagating."

### [index.md] lines 82–84
**Issue:** The "Reading Order" section restates facts already implied by the chapter structure (Q3 before Q4, shapes established in the index) and adds a sentence ("Both files build on the shape analysis in this index") that is self-evident from the file structure.
**Suggestion:** Reduce the reading order section to one sentence: "Read `program_config_tuning.md` first; `math_fidelity_evaluation.md` extends its benchmarking harness with accuracy tracking."

### [math_fidelity_evaluation.md] lines 111–113
**Issue:** `hidden_size = 4096`, `intermediate_size = 1408`, `padded_tokens = 32` are redefined at the top of the evaluation harness even though they were defined identically in the `program_config_tuning.md` harness (lines 216–218). A comment attributing the values would suffice.
**Suggestion:** Add a comment `# Same constants as program_config_tuning.md harness` and keep the definitions (they are needed for the code to run), but note the duplication in a docstring so a future reader does not maintain two separate copies.

---

## Load-Bearing Evidence

- `index.md` line ~45: "**`in0_block_w`** controls how many K-dimension tiles are fetched into L1 per outer loop iteration." — load-bearing because it is the primary conceptual definition; everything in `program_config_tuning.md`'s field reference section derives from this.
- `index.md` line ~51: "For GLM-4-MoE's gate/up projection (`n_tiles = 44`, `num_cores = 64`), only 44 of 64 cores are active — 68.75% utilization." — load-bearing because this structural inefficiency conclusion is unique to the index and not restated elsewhere.
- `program_config_tuning.md` line ~122: "The jump from 4 to 11 is notable: there is no valid value between 4 and 11 for this dimension." — load-bearing because it is the key insight motivating the `in0_block_w=11` candidate and must be preserved exactly.
- `program_config_tuning.md` line ~348: "Do not change production values without a confirmed latency improvement of at least 5% on device — within-noise changes should not be committed." — load-bearing because it states the actionable decision threshold that governs the entire Q3 investigation.
- `math_fidelity_evaluation.md` line ~49: "`fp32_dest_acc_en=True` means the output accumulation register is 32-bit even though the inputs are bf16. This partially compensates for the reduced mantissa product precision" — load-bearing because it explains why `HiFi2 + fp32_dest_acc_en` is more precise than bare `HiFi2`, a nuance that directly affects the accuracy interpretation.
- `math_fidelity_evaluation.md` line ~229: "**Accept LoFi only if:** token agreement rate ≥ 99.5% AND perplexity delta ≤ 0.5%." — load-bearing because it is the quantitative go/no-go threshold that must not be paraphrased away.

## VERDICT
- Crucial updates: yes

## Change Log — C Compression Pass 1

**Suggestion 1 — `math_fidelity_evaluation.md` duplicate `make_prog_cfg` helper removed.**
Replaced the full 18-line `make_prog_cfg` implementation (~lines 125–143) with a three-line cross-reference comment directing readers to `make_config` in `program_config_tuning.md`. Updated the call site at the former `make_prog_cfg(device, out_features)` call to use `make_config(device, out_features)` with an inline comment. Removed the now-unused `import math` from the harness imports.

**Suggestion 2 — `WormholeComputeKernelConfig` block deduplicated.**
The canonical four-field block definition is retained in `math_fidelity_evaluation.md` (the "Current expert compute config" section). In `program_config_tuning.md`'s `bench()` harness, replaced the standalone repeated inline block with a one-line reference comment `# WormholeComputeKernelConfig(HiFi2) — see math_fidelity_evaluation.md for the canonical definition` followed by a single-line constructor call. The `make_expert_compute_cfg(fidelity)` factory in `math_fidelity_evaluation.md` is preserved as-is (it is parameterized, not a bare duplicate).

**Suggestion 3 — `program_config_tuning.md` "Expected pattern" paragraph trimmed.**
Reduced the 7-sentence paragraph at ~line 336 to 3 sentences: (1) memory-bound conclusion at batch=1, (2) the unique `in0_block_w=11` inference for the down projection including the "no valid value between 4 and 11" fact, (3) forward reference to when larger values become relevant (batch ≥ 8). Five redundant restatements of the memory-bound rationale removed.

**Suggestion 4 — `math_fidelity_evaluation.md` duplicate memory-bound explanation collapsed.**
Kept the definitive memory-bound statement in the introduction (~lines 21–22). Replaced the later "Expected outcome at batch=1 decode" paragraph (~lines 264–268) with a back-reference sentence `(See introduction: batch=1 is memory-bound; LoFi's compute savings are irrelevant until batch≥8.)` followed by the two actionable bullet points. Redundant prose eliminated while the go/no-go bullets are preserved.

# Compression Analysis: Ch4 Matmul Config and Math Fidelity — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~726 lines (index.md: 84, program_config_tuning.md: 351, math_fidelity_evaluation.md: 291)
- Estimated post-compression line count: ~706 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions

None remaining.

## MINOR Suggestions

### [program_config_tuning.md] ~lines 58–62
**Issue:** The `per_core_N` derivation for GLM-4-MoE gate/up and down is worked out in prose a second time (lines 61–62: "For GLM-4-MoE gate/up on T3K: N_tiles = ceil(1408/32) = 44…"). The identical numbers already appear in `index.md` lines 67–69 with no additional explanation added here beyond restating the formula result.
**Suggestion:** Replace the two worked-example lines (61–62) with "See tile dimension arithmetic in `index.md`." Retain only the formula line. Net saving: ~2 lines.

### [program_config_tuning.md] ~lines 87 and 111
**Issue:** Two sentences with identical reasoning state that the `min(4, …)` cap is "not binding" — once for gate/up (line 87) and once for down (line 111). The logic is the same: `K_tiles >> 4`, so the cap is a no-op. The second instance adds only the dimension-specific number, not new reasoning.
**Suggestion:** State the general principle once under gate/up, then for down write only: "The cap is again non-binding (44 > 4)." Net saving: ~2 lines.

### [math_fidelity_evaluation.md] ~line 94
**Issue:** The sentence "If routing decisions are identical (they always should be, since the gate is not changed) but the final residual output cosine similarity falls below 0.9999, the precision loss in expert outputs is propagating" contains a parenthetical that contradicts the purpose of the check — if they always should be identical, the check reads as unnecessary, undermining the protocol.
**Suggestion:** Delete the parenthetical "(they always should be, since the gate is not changed)" and rephrase to: "Compare the final residual output cosine similarity; if it falls below 0.9999, precision loss in expert outputs is propagating." Net saving: ~1 line.

### [index.md] ~lines 82–84
**Issue:** The "Reading Order" section (3 sentences) restates what is already evident from the file table directly above it. The closing sentence "Both files build on the shape analysis in this index" is self-evident from the document structure.
**Suggestion:** Reduce to one sentence: "Read `program_config_tuning.md` first; `math_fidelity_evaluation.md` extends its benchmarking harness with accuracy tracking." Net saving: ~2 lines.

### [math_fidelity_evaluation.md] ~lines 111–113
**Issue:** `hidden_size = 4096`, `intermediate_size = 1408`, `padded_tokens = 32` are defined identically in `program_config_tuning.md` lines 216–218. No comment links the two definitions, leaving a silent maintenance risk if sizes change.
**Suggestion:** Add a one-line comment above the constants: `# Same constants as program_config_tuning.md harness — update both if model sizes change.` No lines added if it replaces a blank line; net change: 0 lines, but eliminates the silent duplication risk.

---

## Load-Bearing Evidence

- `program_config_tuning.md` line ~335: "Expected pattern: At batch=1 the sparse matmul is memory-bound, so `in0_block_w=4` is likely near-optimal for both projections. The `in0_block_w=11` candidate for the down projection is the one exception worth measuring: the 4→11 jump reduces K-loop iterations from 11 to 4 (a 2.75× reduction) and there is no valid value between 4 and 11 for `K=44`. When batch ≥ 8 and the kernel becomes compute-bound, larger `in0_block_w` values and `per_core_M > 1` become relevant tuning knobs." — load-bearing because this is the trimmed-to-3-sentences form required by Pass 1 CRUCIAL item 3; it contains the unique `in0_block_w=11` insight and must not be trimmed further.
- `math_fidelity_evaluation.md` line ~247: "(See introduction: batch=1 is memory-bound; LoFi's compute savings are irrelevant until batch≥8.)" — load-bearing because this is the back-reference that replaced the duplicate memory-bound explanation per Pass 1 CRUCIAL item 4; removing it would re-introduce the information gap.
- `math_fidelity_evaluation.md` lines ~124–127: comment directing to `make_config` in `program_config_tuning.md` — load-bearing because this is the cross-reference that replaced the duplicate `make_prog_cfg` implementation per Pass 1 CRUCIAL item 1; it must be preserved verbatim.
- `program_config_tuning.md` line ~257: "# WormholeComputeKernelConfig(HiFi2) — see math_fidelity_evaluation.md for the canonical definition" — load-bearing because this is the deduplication reference per Pass 1 CRUCIAL item 2; it keeps the bench harness functional while eliminating the duplicate block.

## VERDICT
- Crucial updates: no
