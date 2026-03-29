# Compression Analysis: Prefill TTFT Optimization — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~400 lines
- Estimated post-compression line count: ~380 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

### 1. [Multiple files] "5.3x" headline repeated
**Issue:** The "5.3x TTFT improvement" figure appears in: `index.md` line 5, `index.md` lines 15-18 (performance table), `batched_projections.md` line 68, `gdn_prefill_strategy.md` line 144.
**Suggestion:** The index owns the headline number. The two sub-file mentions could be trimmed to "the overall TTFT improvement" with a back-reference to the index table.

### 2. [gdn_prefill_strategy.md] ~lines 67-78
**Issue:** The conv1d shift register copy chain is shown in full, duplicating the same pattern described in Chapter 3. The file acknowledges on line 67: "The same 4-tap causal conv1d used in decode (see Chapter 3)."
**Suggestion:** A one-line note ("identical to decode; see Chapter 3") plus the B=1 callout would suffice since the only difference is B=1 vs B=32.

### 3. [batched_projections.md] ~lines 3-4 and 62-68
**Issue:** "Dispatch overhead" explained in two places: lines 3-4 give a soft description, lines 62-68 give the concrete numbers (285 dispatches).
**Suggestion:** The first instance could be shortened to just "dispatch overhead" since the second instance quantifies it with load-bearing numbers.

### 4. [batched_projections.md] ~lines 23 and 45
**Issue:** `x_dram: [1, 1, seq_len, dim=5120]` shape annotation appears twice in the same file for attention and GDN sections.
**Suggestion:** The second occurrence could simply say "same `x_dram`" since the shape was defined 20 lines earlier.

## Load-Bearing Evidence
- `index.md` line ~9: "The three-category decomposition (batched projections, GDN prefill strategy, state replication)" — load-bearing as the structural spine of the chapter; every sub-file maps to exactly one category
- `batched_projections.md` line ~62: "288 baseline vs. 3 optimized per GDN layer" dispatch-count arithmetic — load-bearing as the only quantification of the per-layer mechanism behind the 5.3x headline
- `gdn_prefill_strategy.md` line ~82: "num_pairs = 1 * 12 = 12 (vs. decode's 384)" — load-bearing as the essential detail tying Chapters 4 and 5 together
- `state_replication.md` line ~35: Recurrence state replication code with `torch.repeat(B, 1, 1)` — load-bearing because it explains why `repeat` is used instead of `expand`

## VERDICT
- Crucial updates: no
