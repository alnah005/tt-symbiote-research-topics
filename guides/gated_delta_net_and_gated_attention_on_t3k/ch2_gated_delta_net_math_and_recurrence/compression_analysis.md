# Compression Analysis: Chapter 2 — Gated Delta Net: Mathematical Formulation and Recurrence Structure — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~692 lines
- Estimated post-compression line count: ~615 lines
- Estimated reduction: ~11%

## CRUCIAL Suggestions

### [state_vs_kv_cache_memory.md] ~lines 72–89
**Issue:** The combined KV calculation is presented three times in succession. Lines 72–73 state the simplified result ("B × 2,048 × T bytes per Gated Attention layer"), then lines 75–79 repeat the full derivation, then lines 81–89 open a "Wait — let us be precise" aside that re-derives the same figure a third time and cross-references a plan document. The aside reads as a live author correction that was never cleaned up, not as final prose, and it duplicates the already-correct derivation directly above it.
**Suggestion:** Delete lines 75–89 entirely (the redundant re-derivation block and "Wait" aside). The derivation at lines 62–70 is already correct and complete. The single-sentence clarification "The plan's stated value of `B × 1,024 × T` refers to one cache (K or V only). The combined KV is `B × 2,048 × T bytes`." can be appended as a one-line note after the first derivation if the K-only/K+V distinction must be called out.

### [gated_delta_rule_formulation.md] ~lines 37–46
**Issue:** The "Dimensional consistency check" block (lines 37–46) annotates every single arithmetic step in the recurrence that was already written out on lines 30–34 with full shapes. Each bullet restates a shape already visible in the equation block directly above it. A reader who can read the recurrence does not need a separate item-by-item restatement confirming `[d_k, d_v] + [d_k, d_v] → [d_k, d_v]`.
**Suggestion:** Reduce the dimensional check to a two-line prose statement (e.g., "All operations are dimensionally consistent: the outer product `k̃_t (correction)^T` produces `[d_k, d_v]`, matching `g_t · S_{t-1}`, and the output `S_t^T q̃_t` produces `[d_v]`."). This removes 7 of the 9 lines while keeping the load-bearing confirmation that the shapes close correctly.

## MINOR Suggestions

### [parallelism_and_scan.md] ~lines 40–54
**Issue:** The explanation of the `D` diagonal matrix in step 3 of the WY-decomposition (lines 50–54) gives the formula, then restates in prose what `D[τ,τ] = Γ_τ` means, then adds a full sentence explaining what happens "without D". The "without D" counterfactual sentence restates the purpose of D, which was just explained.
**Suggestion:** Delete the final sentence beginning "Without D, every query position would incorrectly retrieve..." (approximately 35 words). The role of D is already clear from the two preceding sentences.

### [state_vs_kv_cache_memory.md] ~lines 105–119
**Issue:** The crossover section runs three separate code blocks (lines 106–109, 113–116) and then a prose summary (lines 119) that repeats all three numerical results: "~512–544 tokens ... K-only or K+V ... including conv state". The two subsidiary code blocks cover edge-case derivations that are already summarised in the final sentence; the reader gains little from seeing both the intermediate blocks and their prose recap.
**Suggestion:** Collapse the two secondary code blocks (lines 106–116) into a single note sentence below the primary crossover result, e.g. "Using K-only cache or omitting conv state shifts the crossover to ~512 tokens; including conv state gives ~544 tokens." Delete the long parenthetical at the end of line 119.

### [gated_delta_rule_formulation.md] ~lines 95–99
**Issue:** Section 5 (GLA comparison), lines 95–99, contains three numbered "Key differences" bullets. Bullet 1 largely restates what was already said in the preceding prose paragraph on lines 89–94 (row-wise vs. scalar gating), and bullet 2 repeats the write-mechanism contrast that is explicit in the `S_t` equation on line 92.
**Suggestion:** Collapse bullets 1 and 2 into a single bullet ("GLA applies row-wise per-key-dimension gating and writes `k̃_t v_t^T` unconditionally; Gated Delta Net uses a scalar gate and writes only the error residual."), keeping bullet 3 (error correction) unchanged. Saves ~2 lines while preserving all distinctions.

### [parallelism_and_scan.md] ~lines 78–88
**Issue:** Section 2.3 calculates the parallel scan workspace cost in a code block, then re-states the result in the following prose ("With 32 V-heads per layer, that is 32 × 128 MB = 4,096 MB ≈ 4 GB") — the multiplication is already visible in the code block one line above.
**Suggestion:** Remove the prose restatement of the arithmetic. Keep only the conclusion sentence: "In practice, `chunk_gated_delta_rule` scans the T/C chunks sequentially, benefiting from within-chunk parallelism without paying the associative scan workspace cost."

### [index.md] ~lines 19–20
**Issue:** Lines 19–20 introduce a "Questions Answered" section that names Q1 and Q3. These questions are not defined anywhere in this chapter (their definitions presumably live in the guide spec), so the reference is opaque to a reader who only has this chapter. The Learning Objectives section (lines 11–16) already covers the same scope in concrete, actionable terms.
**Suggestion:** Remove the "Questions Answered" section (lines 19–20 and header). The Learning Objectives section is a complete and more informative substitute.

## Load-Bearing Evidence
- `gated_delta_rule_formulation.md` line ~10: "**Gated Delta Net** combines both mechanisms: scalar coarse forgetting from GLA and error-correcting writes from DeltaNet." — load-bearing because it is the single-sentence architectural thesis of the whole chapter; every subsequent derivation references it.
- `gated_delta_rule_formulation.md` line ~33: "`S_t  = g_t · S_{t-1}  +  k̃_t (β_t · (v_t − g_t · S_{t-1}^T k̃_t))^T`" — load-bearing because it is the master recurrence equation; all parallelism and memory analysis downstream depends on this exact form.
- `parallelism_and_scan.md` line ~68: "`(a_2, B_2) ∘ (a_1, B_1)  =  (a_2 · a_1,  a_2 · B_1 + B_2)`" — load-bearing because it is the associativity proof for the inter-chunk operator, which justifies why parallel scan is theoretically possible at the chunk level.
- `state_vs_kv_cache_memory.md` line ~103: "`T  =  1,114,112 / 2,048  =  544 tokens`" — load-bearing because it is the quantitative crossover result that motivates the entire memory comparison argument.
- `state_vs_kv_cache_memory.md` line ~148: "`KV cache / DeltaNet state  =  512 MB / 1.0625 MB  ≈  482×`" — load-bearing because it is the headline numerical conclusion of the chapter's memory analysis.

## VERDICT
- Crucial updates: yes

# Compression Analysis: Chapter 2 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~673 lines
- Estimated post-compression line count: ~650 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions
None — both Pass 1 crucials resolved.

Pass 1 crucial 1 (`state_vs_kv_cache_memory.md` duplicate KV derivation + "Wait" aside): confirmed deleted; lines 72–73 now flow directly into the crossover section with no re-derivation block.

Pass 1 crucial 2 (`gated_delta_rule_formulation.md` 7-bullet dimensional consistency check): confirmed collapsed to the prescribed single prose sentence at lines 37–39.

## MINOR Suggestions

### [state_vs_kv_cache_memory.md] ~lines 80–102 — Crossover triple-code-block + prose recap (Pass 1 minor, unapplied)
The crossover section still runs three code blocks and a prose summary sentence that repeats all three numerical results. The subsidiary blocks (lines 88–101) recalculate the same T = 512–544 range already visible in the primary block. Collapse the two subsidiary code blocks into a single note sentence: "Using K-only cache or omitting conv state shifts the crossover to ~512 tokens; including conv state gives ~544 tokens." Delete the long parenthetical at the end of the summary sentence. Saves ~12 lines.

### [parallelism_and_scan.md] ~line 54 — "Without D" counterfactual sentence (Pass 1 minor, unapplied)
The sentence beginning "Without D, every query position would incorrectly retrieve..." restates the purpose of D, which the two preceding sentences already established. Delete it (~35 words).

### [gated_delta_rule_formulation.md] ~lines 91–93 — GLA bullets 1 and 2 overlap prose above (Pass 1 minor, unapplied)
Bullets 1 and 2 in the "Key differences" list restate the row-wise/scalar gating contrast and the write-mechanism contrast, both of which are explicit in the preceding paragraph and `S_t` equation. Merge bullets 1 and 2 into one: "GLA applies row-wise per-key-dimension gating and writes `k̃_t v_t^T` unconditionally; Gated Delta Net uses a scalar gate and writes only the error residual." Saves ~2 lines.

### [parallelism_and_scan.md] ~lines 85–86 — Prose restatement of scan workspace arithmetic (Pass 1 minor, unapplied)
The sentence "With 32 V-heads per layer, that is 32 × 128 MB = 4,096 MB ≈ 4 GB of scan workspace per layer" repeats multiplication that is already visible in the code block immediately above. Remove the arithmetic restatement; keep only the conclusion about the sequential scan being used in practice.

### [state_vs_kv_cache_memory.md] ~lines 123–126 — Fourth repetition of B × 1,114,112 figure
The value "B × 1,114,112 bytes ≈ B × 1.0625 MB" appears at lines 22, 44–50, and 81, and again in Section 4.2 (lines 124–125). By Section 4.2 the reader has seen it three times. Replace the standalone code block in §4.2 with a cross-reference: "As computed in §1.3, B × ~1.0625 MB per layer." Saves 3 lines.

## Load-Bearing Evidence
- `index.md` line 5: "This chapter derives the Gated Delta Net state-update rule from first principles, defines every tensor and its shape for the Qwen3.5-35B-A3B configuration" — anchors the chapter's scope; must not be truncated.
- `gated_delta_rule_formulation.md` line 33: "`S_t  = g_t · S_{t-1}  +  k̃_t (β_t · (v_t − g_t · S_{t-1}^T k̃_t))^T`" — the master recurrence; all downstream content depends on it.
- `parallelism_and_scan.md` line 68: "`(a_2, B_2) ∘ (a_1, B_1)  =  (a_2 · a_1,  a_2 · B_1 + B_2)`" — the associativity proof for the inter-chunk operator.
- `state_vs_kv_cache_memory.md` line 102: "for any sequence longer than ~512–544 tokens, the Gated Attention KV cache is more expensive per layer than the DeltaNet recurrent state" — the headline conclusion of the memory comparison.

## VERDICT
- Crucial updates: no

---

## Agent A Change Log — Compression Pass 1

**Applied by:** Agent A (Generator)
**Date:** 2026-03-27

### Item 1 — `state_vs_kv_cache_memory.md` (~lines 72–89): Remove duplicate KV derivation blocks

Removed the redundant re-derivation block ("For clarity in crossover analysis...") and the "Wait — let us be precise" live-editing artifact that followed it. Both blocks re-derived the same `B × 2,048 × T` result already established in lines 62–70. Replaced with a single concise sentence confirming the combined KV figure. Approximately 17 lines removed.

### Item 2 — `gated_delta_rule_formulation.md` (~lines 37–46): Collapse 7-bullet dimensional consistency check to prose

Replaced the 7-bullet enumeration under "Dimensional consistency check" — each bullet restating a shape already explicit in the equation block directly above — with a single prose sentence confirming that all terms are ∈ R^{d_k × d_v} and the readout ∈ R^{d_v}. Approximately 6 lines removed.
