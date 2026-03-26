## Change Log — B Feedback Pass 1

**token_padding_and_dispatch.md:** Corrected expert-to-device routing example: expert 37 maps to device floor(37/16)=2, not device 4.

## Change Log — B Feedback Pass 2

---

# Compression Analysis: Ch3 Expert Dispatch Pipeline Profiling — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~700 lines
- Estimated post-compression line count: ~560 lines
- Estimated reduction: ~20%

## CRUCIAL Suggestions

### [token_padding_and_dispatch.md] ~lines 39–57
**Issue:** The section "Why SPARSITY_BLOCK_SIZE=32 was chosen" and "Is reducing SPARSITY_BLOCK_SIZE safe?" overlap heavily. The three numbered safety reasons (lines 52–55) restate content already implied by the "why 32 was chosen" explanation directly above. The conclusion sentence (line 57) then restates both sections in a third pass.
**Suggestion:** Merge into a single subsection "SPARSITY_BLOCK_SIZE=32: rationale and constraints" (~10 lines). Keep the three safety bullets but drop the paragraph that sets them up (lines 51–52 top) and collapse the conclusion sentence into the final bullet.

### [token_padding_and_dispatch.md] ~lines 59–68
**Issue:** "Effective batch after padding" section re-explains that batch=1 pads to 32 tokens (already stated at line 19, line 29, line 35, and in the Stage 1 opening) and re-explains the expert-sparsity mitigation (already introduced in `index.md` lines 5–7 and Stage 2 context). The final sentence about "2 distinct experts across 8 devices" duplicates content from lines 109–116.
**Suggestion:** Cut this subsection entirely or reduce it to two sentences appended to "Why SPARSITY_BLOCK_SIZE=32 was chosen." The FLOPs-are-32× point is the only non-duplicate; keep that line only.

### [sparse_matmul_profiling.md] ~lines 96–125
**Issue:** The "What each parameter controls" subsection contains five parameter explanations. The `per_core_M` block (lines 111–115) devotes three sentences to the batch=1 case and two sentences to the prefill case, then explicitly flags the prefill content as out of scope ("This is a prefill-regime concern, not a decode-regime concern"). The `fuse_batch=False` explanation (lines 124–125) restates what `fuse_batch` means in a way derivable from the parameter name and the sparse block layout already shown above.
**Suggestion:** Trim the `per_core_M` paragraph to the two batch=1 sentences only; cut the prefill sentences. Cut the `fuse_batch=False` explanation entirely (one line saved there, ~3 saved from per_core_M).

### [weight_application_overhead.md] ~lines 223–252
**Issue:** The "Benchmarking the alternatives" code block (lines 226–252) re-implements the warmup+timed loop pattern verbatim for a third time in this chapter (same pattern already appears in `token_padding_and_dispatch.md` lines 152–167 and `sparse_matmul_profiling.md` lines 204–225). The only novel content is the two `print` statements labeling the two variants.
**Suggestion:** Replace the full code block with a prose note: "Use the standard warmup+timed loop (20 warmup, 100 timed iterations with `ttnn.synchronize_device`) to time both `apply_weights_current` and `apply_weights_alternative`; record results in the table below." Save ~20 lines.

### [bottleneck_summary.md] ~lines 36–54
**Issue:** The Bailing per-stage latency table (lines 40–54) is structurally identical to the GLM-4-MoE table (lines 22–34): same stages, same line references, same column headers. The only differences are the header and a single italicized note. Since both models share `hidden_size=4096`, `intermediate_size~1408`, and `topk=2`, and `index.md` already states that both configs are identical in `in0_block_w`, a duplicate table adds no structural information.
**Suggestion:** Collapse into one table with two measurement columns ("GLM-4-MoE mean (µs)" and "Bailing mean (µs)") sharing the same stage/description/lines columns. Saves ~18 lines.

## MINOR Suggestions

### [index.md] ~lines 61–65
**Issue:** The final paragraph ("Findings feed into Chapter 7's optimization priority matrix…") restates routing logic already stated at lines 21–22 ("If Stage 2 or Stage 6 dominate…"). The sentence is hedging ("the stage identified as the bottleneck… determines which subsequent chapter… deserves immediate attention") without adding actionable information beyond what the prior paragraph says.
**Suggestion:** Cut this paragraph; the cross-chapter pointer is already carried by the link in line 53 (`bottleneck_summary.md`).

### [sparse_matmul_profiling.md] ~lines 138–142
**Issue:** The note block explaining divisibility of `in0_block_w` into `K_tiles` (lines 141–142) enumerates all three valid values and their iteration counts for `K_tiles=44`. This is mechanical arithmetic that a practitioner running the sweep will verify naturally; spelling out all three cases adds length without insight.
**Suggestion:** Shorten to one sentence: "`in0_block_w` must divide `K_tiles` evenly; for `K_tiles=44`, values {1, 2, 4} are all valid."

### [weight_application_overhead.md] ~lines 69–82
**Issue:** The "Tensor sizes involved" table lists six rows computing byte sizes for tensors that are all trivially small (4–16 KB). The table header note "All tensors are small" (line 82) then explicitly states the conclusion the table was built to support. The table itself is scaffolding for a point that can be made in one sentence.
**Suggestion:** Replace the table and the following prose with: "All intermediate tensors in Stage 7 are small (4–16 KB at batch=1); the cost is kernel-launch-overhead-dominated, not bandwidth-dominated."

### [bottleneck_summary.md] ~lines 62–80
**Issue:** The "Pre-Measurement Predictions" prose paragraphs for dispatch, matmuls, and weight application each conclude with an "Expected range" sentence that is then restated in the prediction table at lines 92–104. The prose adds qualitative rationale, but several sentences in each paragraph (e.g., lines 66–67 repeating the 64 KB message size already computed in `token_padding_and_dispatch.md`, lines 75–77 re-quoting the FLOPs figure from `sparse_matmul_profiling.md`) duplicate content rather than add interpretation.
**Suggestion:** Cut the back-references to per-file computations (the lines that say "as computed in X.md") and the repeated numeric derivations. Keep only the qualitative reasoning unique to this summary view. Saves ~6 lines.

## Load-Bearing Evidence

- `token_padding_and_dispatch.md` line ~47: "The constant `TOPK_MIN_WIDTH=64` (`moe.py:L51`) sets a related floor: it is twice `SPARSITY_BLOCK_SIZE` and controls the minimum number of dispatched tokens that triggers the sparse path" — load-bearing because this coupling between `TOPK_MIN_WIDTH` and `SPARSITY_BLOCK_SIZE` is not stated anywhere else in the chapter and explains a non-obvious branching behavior.
- `sparse_matmul_profiling.md` line ~119: "Only `44` of `64` cores are active; `20` cores are idle. This represents 31% underutilization." — load-bearing because the core underutilization figure for the gate/up projections is a quantitative bottleneck fact cited by `bottleneck_summary.md` and required for the optimization priority assessment.
- `weight_application_overhead.md` line ~194: "If the TTNN mul kernel does not support implicit broadcast on `hidden_size` dimension and instead requires equal shapes, it will trigger a shape mismatch error." — load-bearing because this is the critical risk caveat for the alternative implementation; removing it would leave the alternative appearing unconditionally safe.
- `bottleneck_summary.md` line ~137: The full "Source Line Map for Optimization Targets" table — load-bearing because it is the only place in the chapter that maps each bottleneck directly to both `moe.py` line ranges and the specific responsible construct, serving as the actionable handoff to Chapter 4.

## VERDICT
- Crucial updates: yes

## Change Log — C Compression Pass 1

**token_padding_and_dispatch.md:** Merged "Why SPARSITY_BLOCK_SIZE=32 was chosen" and "Is reducing SPARSITY_BLOCK_SIZE safe?" into a single subsection "SPARSITY_BLOCK_SIZE=32: rationale and constraints". Removed the "Effective batch after padding" subsection, retaining only its essential fact as the closing sentence of the merged subsection: "At batch=1 decode, the single real token is padded to 32 tokens (`SPARSITY_BLOCK_SIZE`), so every expert matmul processes 32 rows regardless of actual occupancy."

**sparse_matmul_profiling.md:** Removed the prefill-specific sentence from the `per_core_M` explanation (the `batch=16` / `per_core_M=2` example flagged as out-of-scope). Trimmed the `fuse_batch=False` explanation from two sentences to one line.

**weight_application_overhead.md:** Replaced the duplicate warmup+timed benchmark loop code block (~27 lines) in "Benchmarking the alternatives" with a single prose note directing readers to use the same harness from `token_padding_and_dispatch.md` and `sparse_matmul_profiling.md`, substituting the weight-application implementations as the timed block.

**bottleneck_summary.md:** Collapsed the structurally identical GLM-4-MoE and Bailing per-stage latency tables into one combined table with "GLM-4-MoE mean (µs)" and "Bailing mean (µs)" measurement columns, saving ~18 lines. All rows and stage descriptions preserved.

---

# Compression Analysis: Ch3 Expert Dispatch Pipeline Profiling — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~985 lines
- Estimated post-compression line count: ~950 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
None remaining.

## MINOR Suggestions

### [sparse_matmul_profiling.md] ~lines 317–330
**Issue:** The "Core Utilization Analysis" section (lines 317–330) has two subsections. The "Gate/up projection" subsection adds new conclusions (31% underutilization figure, "cannot be fixed by tuning" judgment). The "Down projection" subsection (lines 327–330) is purely repetitive: "all 64 cores are active" and "per-core workload is slightly higher" were both stated in full at lines 118–119 within the `per_core_N` parameter explanation above.
**Suggestion:** Remove the "Down projection: full core utilization" subsection (~4 lines). The gate/up subsection is worth keeping for its unique conclusions.

### [sparse_matmul_profiling.md] ~lines 138–142
**Issue:** (Carried from Pass 1, not yet resolved.) The divisibility note enumerates all three valid `in0_block_w` values for `K_tiles=44` with their iteration counts. The arithmetic is mechanical and will be verified naturally by anyone running the sweep.
**Suggestion:** Shorten to one sentence: "`in0_block_w` must divide `K_tiles` evenly; for `K_tiles=44`, values {1, 2, 4} are all valid."

### [weight_application_overhead.md] ~lines 69–82
**Issue:** (Carried from Pass 1, not yet resolved.) The "Tensor sizes involved" table computes byte sizes for six tensors all in the 4–16 KB range, then the prose at line 82 immediately states "All tensors are small" — the conclusion the table was built to support.
**Suggestion:** Replace the table and the sentence that follows it with: "All intermediate tensors in Stage 7 are small (4–16 KB at batch=1); the cost is kernel-launch-overhead-dominated, not bandwidth-dominated."

### [bottleneck_summary.md] ~lines 62–82
**Issue:** (Carried from Pass 1, not yet resolved.) The "Pre-Measurement Predictions" prose paragraphs each conclude with an expected range (e.g., "Expected range for each all-to-all op: 50–150 µs") that is then restated verbatim as the "Predicted range" column of the table at lines 69–82. The message-size and FLOPs figures referenced in the prose ("~64 KB computed in token_padding_and_dispatch.md", "~369 MFLOPs ... as computed in sparse_matmul_profiling.md") duplicate content from those files rather than synthesising it.
**Suggestion:** Cut the back-reference sentences containing already-computed numerics; keep only the qualitative reasoning unique to this summary. The table already carries the ranges. Saves ~6 lines.

### [index.md] ~lines 61–65
**Issue:** (Carried from Pass 1, not yet resolved.) The closing paragraph re-routes the reader to chapters based on which stage is the bottleneck. This routing logic was already stated at lines 21–22 ("If Stage 2 or Stage 6 dominate... If Stages 4 and 5 dominate..."). The link to `bottleneck_summary.md` is already present in the files table at line 53.
**Suggestion:** Cut this paragraph. The cross-chapter pointer is redundant.

## Load-Bearing Evidence

- `token_padding_and_dispatch.md` line ~46: "`TOPK_MIN_WIDTH=64` (`moe.py:L51`) is twice `SPARSITY_BLOCK_SIZE` by convention and controls the token-count threshold that activates the sparse path. Halving one without halving the other breaks the dispatch-vs-sparse branching logic." — load-bearing because this coupling is not stated anywhere else in the chapter and explains a non-obvious safety constraint on reducing the constant.
- `sparse_matmul_profiling.md` line ~321: "Only `44` of `64` cores are active; `20` cores are idle. This represents 31% underutilization. ... a structural inefficiency inherent to the model's intermediate dimension and cannot be fixed by tuning the program config." — load-bearing because the 31% figure and the "cannot be fixed" conclusion are the only place this quantitative judgment is stated; removing it would leave the bottleneck summary without its supporting derivation.
- `weight_application_overhead.md` line ~194: "If the TTNN mul kernel does not support implicit broadcast on `hidden_size` dimension and instead requires equal shapes, it will trigger a shape mismatch error." — load-bearing because this is the only risk caveat for the alternative implementation; removing it would leave the alternative appearing unconditionally safe.
- `bottleneck_summary.md` lines ~115–124: The "Source Line Map for Optimization Targets" table — load-bearing because it is the only place that maps each bottleneck to both exact `moe.py` line ranges and the responsible construct, providing the actionable handoff to Chapter 4.

## VERDICT
- Crucial updates: no
