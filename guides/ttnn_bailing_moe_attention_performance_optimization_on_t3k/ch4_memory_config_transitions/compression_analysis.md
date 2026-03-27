## Agent A Change Log — B Feedback Pass 1

- Fix 1 (`transition_cost_model.md`): Corrected `v_raw` tile-padded shape in tensor size table from `(1, 4, 32, 512)` to `(1, 1, 32, 512)`, consistent with the 16,384-element count in the same row.
- Fix 2 (`decode_tensor_lifecycle.md`): Corrected T3b transition table V source shape from `(1, 4, 32, 512)` to `(1, 1, 32, 512)`, aligning the table with Stage 1 of the same file.
- Fix 3 (`optimization_opportunities.md`): Replaced "V-slice" in the T_norm Strategy A code-change annotation with "Q-slice (and K-slice, if use_qk_norm applies to K as well)", as V does not pass through QK norm.

## Agent A Change Log — B Feedback Pass 2

- Fix 1 (`decode_tensor_lifecycle.md`): Added a required `ttnn.reshape(v_raw, (1, 4, 32, 128))` step immediately before the `ttnn.to_memory_config` call in the Stage 6 T3b code block, with prose clarifying that `to_memory_config` cannot perform shape changes and the reshape is a non-optional prerequisite.
- Fix 2 (`transition_cost_model.md`): Corrected the Re-Tiling Cost section to note that the no-re-tiling claim for T3b is only valid after the required `ttnn.reshape` from `(1, 1, 32, 512)` to `(1, 4, 32, 128)` has been applied; without this reshape, `(1, 1, 32, 512)` does not match the `(32, 128)` shard boundary of `kv_update_mem`.
- Fix 3 (`optimization_opportunities.md`): Updated the T3b Priority 4 description to clarify that directing the V slice to `kv_update_mem` eliminates only the `to_memory_config` kernel dispatch, not the reshape from `(1, 1, 32, 512)` to `(1, 4, 32, 128)`, which remains a required step; adjusted the estimated saving from ~11 µs to ~8 µs accordingly, and updated the T3b summary table row and section prose to match.

## Agent A Change Log — B Feedback Pass 3

- Fix 1 (`decode_tensor_lifecycle.md`): Corrected Stage 4 V shape from `(1, 4, 32, 512)` to `(1, 1, 32, 512)` in the prose note explaining that V is not passed through RoPE; V is never reshaped at Stage 2, so the 4-head dimension was erroneous.
- Fix 2 (`optimization_opportunities.md`): Changed "approximately 65 µs" to "approximately 62 µs" to match the explicit arithmetic sum (21 + 11 + 11 + 11 + 8 = 62 µs) stated in the formula immediately above that sentence.
- Fix 3 (`transition_cost_model.md`): Removed the stray `×4` suffix from the V reshaped shape annotation in the tensor size table; the element count (16,384) confirms the shape is `(1,4,32,128)` with no multiplier.

## Agent A Change Log — B Feedback Pass 4

- Fix 1 (`optimization_opportunities.md`): Corrected total recoverable overhead from "129 µs" to "126 µs" to match the document's own arithmetic (Priority 1 upper bound 64 µs + Priorities 2–4 combined 62 µs = 126 µs).

## Agent A Change Log — B Feedback Pass 5

- Fix 1 (`decode_tensor_lifecycle.md`, `index.md`): Corrected the out-of-bounds `rope_shard_mem_q` core grid in the `__init__` pattern — replaced `CoreCoord(self.num_heads - 1, 0)` (which resolves to `CoreCoord(15, 0)`, exceeding the Wormhole chip's 8-column limit) with `CoreCoord(7, 1)`, giving a valid 8 cols × 2 rows = 16-core rectangle; updated all associated prose, diagram, transition table, and `index.md` glossary entry to reflect the corrected 2-row Q-head layout, and split the single generic `rope_shard_mem` definition into separate `rope_shard_mem_q` (2-row) and `rope_shard_mem_k` (1-row, unchanged) blocks.

## Agent A Change Log — B Feedback Pass 6

- Fix 1 (`optimization_opportunities.md`): Removed T2b (~11 µs) from the Priorities 2–4 combined saving formula (reducing the sum from 62 µs to 51 µs) to eliminate double-counting with Priority 1, updated Priority 3's claimed saving from ~22 µs to ~11 µs (T3a only, since T2b is already attributed to Priority 1), and updated the combined total from 126 µs to 115 µs throughout the file.
- Fix 2 (`decode_tensor_lifecycle.md`): Corrected Stage 5's opening sentence to state that T2a (Q eviction to DRAM) is driven by `TTNNRMSNorm` when `use_qk_norm=True`, not by `paged_sdpa_decode`; added a note that when `use_qk_norm=False` the T2a transition may not be required or may be driven by a different consumer.

## Agent A Change Log — B Feedback Pass 7

- Fix 1 (`optimization_opportunities.md`): Corrected T3b saving from ~8 µs to ~11 µs, removing the incorrect "metadata operation cost absorbs the transfer component" reasoning — `ttnn.reshape` is zero-copy when tile boundaries are compatible (holds for `(1,1,32,512)→(1,4,32,128)`), so the full dispatch overhead + transfer is saved; updated Priorities 2–4 combined sum from 51 µs to 54 µs (21 + 11 + 11 + 11) and combined total from 115 µs to 118 µs throughout the file.

## Agent A Change Log — B Feedback Pass 8

- Fix 1 (`optimization_opportunities.md`): Corrected Priority 4 code and description — replaced the invalid sequence of `ttnn.slice(..., memory_config=kv_update_mem)` followed by `ttnn.reshape` (which would require redistributing tiles across L1 banks on a HEIGHT_SHARDED tensor, not a zero-copy operation) with the correct three-step sequence: `ttnn.slice` → DRAM INTERLEAVED, `ttnn.reshape` → zero-copy on contiguous DRAM, `ttnn.to_memory_config` → L1; clarified that the optimization eliminates the redundant intermediate `to_memory_config` staging step from the baseline four-step path (slice→DRAM, reshape→DRAM, to_memory_config→DRAM, to_memory_config→L1), saving one dispatch overhead (~8–11 µs), while the final `to_memory_config` to `kv_update_mem` still executes.

## Agent A Change Log — B Feedback Pass 9

- Fix 1 (`optimization_opportunities.md`): Corrected T3b summary table row — status changed from "partially eliminable" to "not eliminated — still executes" (the final DRAM→L1 shard copy to `kv_update_mem` remains); updated the table's recommended strategy to accurately state that Priority 4 removes the redundant intermediate staging call, not T3b itself; relabelled the fourth term in the combined saving formula from "T3b" to "intermediate V staging call" and updated the combined range to ~51–54 µs with a clarifying note that T3b still executes; updated the summary prose figure from "54 µs" to "51–54 µs" for internal consistency.

---

# Compression Analysis: Chapter 4 — Memory-Config Transitions — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~821 lines
- Estimated post-compression line count: ~720 lines
- Estimated reduction: ~12%

## CRUCIAL Suggestions

### [decode_tensor_lifecycle.md] ~lines 86–88
**Issue:** The section header "Definition of `rope_shard_mem_q` and `rope_shard_mem_k`" is immediately followed by the prose sentence "Q and K require separate `MemoryConfig` objects because they have different head counts (16 vs 4), which means different core grids." This sentence is a direct restatement of the header — both say the same thing in different words.
**Suggestion:** Delete lines 87–88 (the prose sentence). The header alone is sufficient; the code blocks that follow make the difference concrete.

### [transition_cost_model.md] ~lines 84–142
**Issue:** The six per-transition cost estimate blocks (T1a through T3b) each apply the identical three-line template (`Tensor`, `Path`, `t_transfer`, `t_overhead`, `t_Tx`) with only the numbers varying. Lines 84–142 span 59 lines to convey what a 7-row table communicates in 10 lines. The formula is stated once earlier in the file and then silently re-applied six times.
**Suggestion:** Replace the six code blocks with a single compact table: columns = Transition, Tensor size, t_transfer, t_overhead, Total. Retain the existing totals table at lines 170–182, which already summarises the final figures. This collapses ~59 lines to ~12 lines while preserving all numeric content.

### [decode_tensor_lifecycle.md] ~lines 130–145 and lines 269–313
**Issue:** The source/destination MemoryConfig pair for each transition is stated in three separate places: (1) inline `# Source / # Dest` comments inside the T1a/T1b code block (lines 135–144), (2) the ASCII lifecycle diagram (lines 282–300), and (3) the summary table (lines 305–313). The code-block comments and the summary table carry nearly identical information.
**Suggestion:** Remove the `# Source MemoryConfig: ...` / `# Dest MemoryConfig: ...` comment pairs from the T1a/T1b code block (lines 135–144), since the diagram and table already capture this. Apply the same removal to the analogous comment pairs in the T2 (lines 180–187) and T3 (lines 217–238) code blocks. The diagram and table remain the authoritative reference.

## MINOR Suggestions

### [optimization_opportunities.md] ~lines 5–7
**Issue:** The Overview restates the total transition overhead figure (150–171 µs) and per-transition cost range (11–21 µs) already given at the end of `transition_cost_model.md`. One cross-reference sentence replaces the repeated figures.
**Suggestion:** Shorten to: "See `transition_cost_model.md` for cost estimates (~150–171 µs total); the analysis here classifies transitions as eliminable or kernel-mandated and ranks opportunities by impact."

### [transition_cost_model.md] ~lines 29–34
**Issue:** The "Kernel Dispatch Overhead" enumeration lists three sub-steps (Python dispatch, op enqueue, kernel launch) each with their own `[ESTIMATE]` range, then immediately sums them in a bold sentence. The sub-step breakdown adds marginal value; the sum is what the rest of the file uses.
**Suggestion:** Collapse to two sentences: one naming the three components, one stating the ~5–13 µs aggregate. Saves ~6 lines.

### [optimization_opportunities.md] ~lines 244–252
**Issue:** The "Combined potential saving" block restates per-priority saving figures (T1a ~21 µs, T1b ~11 µs, T3a ~11 µs) already broken down in each priority's own prose and at the end of each code block. The rollup arithmetic is useful, but the full itemised restatement is redundant.
**Suggestion:** Keep the formula line and the two-sentence summary; remove the intermediate parenthetical restatement of each individual saving.

### [decode_tensor_lifecycle.md] ~lines 64 and ~80
**Issue:** Lines 64 and 80 each contain a parenthetical forward reference to `transition_cost_model.md` (`"cost discussed in transition_cost_model.md"` / `"no data movement occurs"`). These parentheticals interrupt the lifecycle walkthrough for readers who have already read the cost model, and the cost model itself is the next file in reading order.
**Suggestion:** Remove the parenthetical `(discussed in transition_cost_model.md)` at line 64 and tighten line 80's prose; the "Next:" footer at line 350 already points to the cost model.

## Load-Bearing Evidence
- `decode_tensor_lifecycle.md` line ~7: `"num_heads=16, num_kv_heads=4, head_dim=128, hidden_size=4096, partial_rotary_factor < 1.0, use_qk_norm=True"` — load-bearing because these are the specific model parameters that drive every shard shape, core-grid, and byte-count calculation in all three files.
- `transition_cost_model.md` line ~34: `"Total fixed overhead per transition: approximately 5–13 µs [ESTIMATE]."` — load-bearing because it establishes the overhead floor that dominates over bandwidth for all small-tensor transitions in the decode path.
- `optimization_opportunities.md` line ~252: `"Approximately 51–54 µs can be recovered per decode step using only Python-level code changes in TTNNBailingMoEAttention, without any modifications to TTNN kernels."` — load-bearing because it is the document's primary quantified conclusion and the basis for the combined-saving claim.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 1

- Change 1 (`decode_tensor_lifecycle.md` line 88): Deleted the redundant prose sentence "Q and K require separate `MemoryConfig` objects because they have different head counts (16 vs 4), which means different core grids." immediately following the section header "Definition of `rope_shard_mem_q` and `rope_shard_mem_k`" — the header and sentence were verbatim restatements; the header alone is kept.
- Change 2 (`transition_cost_model.md` lines 84–142): Replaced the six structurally identical per-transition cost code blocks (T1a through T3b, ~59 lines) with a single 8-row compact table (columns: Transition, Description, Tensor, Size, Path, t_transfer, t_overhead, Total), preserving all numeric values; the formula explanation in the preceding prose is retained as the single definition.
- Change 3 (`decode_tensor_lifecycle.md` lines ~135–238): Removed all `# Source MemoryConfig:` / `# Dest MemoryConfig:` comment pairs from the T1a/T1b, T2a/T2b, and T3a/T3b code blocks; retained comments that explain WHY a transition occurs or flag a required prerequisite (the reshape warning in T3b), as the summary table and ASCII diagram remain the authoritative source/destination reference.

## Agent A Change Log — B Feedback Pass 11

- Fix 1 (`optimization_opportunities.md`): Updated T3a/T3b early analysis section (formerly lines 84–109) to align with Priority 4 — replaced the instruction to pass `memory_config=kv_update_mem` to `ttnn.slice` with the correct three-step sequence (slice→DRAM, reshape zero-copy on DRAM, `to_memory_config`→L1); removed the incorrect "zero-copy" claim for reshape when input is HEIGHT_SHARDED; corrected "T3b is eliminated" to "T3b still executes — what is eliminated is the redundant intermediate staging call"; updated the saving claim from "~11 µs for T3b" to "~8–11 µs for the eliminated intermediate staging call"; and updated the Code location block to show the correct three-step implementation matching Priority 4.

## Agent A Change Log — B Feedback Pass 12

- Fix 1 (`optimization_opportunities.md`): Added mandatory `ttnn.reshape(q_slice, (1, 16, 32, 128))` between `ttnn.slice` and `ttnn.to_memory_config(rope_shard_mem_q)` in both the T1a/T1b early-analysis section and the Priority 2 section, with comments explaining that `q_raw` exits the slice as `(1,1,32,2048)` (flat, 32 tile-rows) which is geometrically incompatible with the `rope_shard_mem_q` HEIGHT_SHARDED spec requiring `(1,16,32,128)` (16 cores × 32 rows = 512 tile-rows), and that the reshape must occur on the contiguous DRAM tensor to be zero-copy; reclassified T1a and T1b as not eliminable without a RoPE kernel change; updated the Priority 2 heading from "~32 µs saving" to a correctness note; corrected the summary table T1a/T1b rows; updated the combined saving formula (removing T1a + T1b = 32 µs, reducing Priorities 3–4 combined from ~51–54 µs to ~19–22 µs and the combined total from ~118 µs to ~83–86 µs); and added T1a/T1b to the Non-Eliminable Transitions section.

## Agent A Change Log — B Feedback Pass 13

- Fix 1 (`optimization_opportunities.md`): Renumbered the Non-Eliminable Transitions list from the duplicate sequence 1, 2, 2, 3 to the correct sequential numbering 1, 2, 3, 4 — the T4 entry was relabelled from "2." to "3." and the DRAM-to-L1 within `paged_update_on_device` entry was relabelled from "3." to "4.".
- Fix 2 (`optimization_opportunities.md`): Added a double-count warning to the T2b row's saving cell in the Summary Table, changing "~11 µs" to "~11 µs (included in Priority 1 — do not add separately)", making clear that T2b's saving is already counted within Priority 1's ~64 µs total and must not be summed again.

## Agent A Change Log — B Feedback Pass 14

- Fix 1 (`optimization_opportunities.md`): Corrected the combined saving range at line 275 from "83–86 µs" to "69–86 µs" — the lower bound was wrong (50+19 = 69, not 83); the upper bound 86 µs was already correct (64+22 = 86).

## Agent A Change Log — B Feedback Pass 15

- Fix 1 (`optimization_opportunities.md`): Corrected Priority 1 saving from "approximately 50–64 µs" to "approximately **64 µs**" (exact sum: T2a + T2b + T_norm_in_Q + T_norm_in_K = 21 + 11 + 21 + 11 = 64 µs; the 50 µs lower bound was spurious and unexplained); restored the combined recoverable total from "69–86 µs" to "83–86 µs" (Priority 1 exact 64 µs + Priorities 3–4 range 19–22 µs = 83–86 µs; the "69–86 µs" introduced in Pass 14 was based on the now-removed spurious 50 µs lower bound).

---

# Compression Analysis: Chapter 4 — Memory-Config Transitions — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~776 lines (`decode_tensor_lifecycle.md` 330 + `transition_cost_model.md` 154 + `optimization_opportunities.md` 292)
- Estimated post-compression line count: ~746 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions

### [optimization_opportunities.md] lines 212–232: Priority 2 block duplicates T1a/T1b classification section
**Issue:** The Priority 2 action block (lines 212–232, ~21 lines) fully duplicates the T1a/T1b classification section at lines 11–49. Both contain: (1) the mandatory three-step sequence prose, (2) the same code block showing `slice → reshape → to_memory_config`, and (3) the same conclusion that T1a/T1b are kernel-mandated and cannot be bypassed without a RoPE kernel change. The Priority 2 section was added to correct an earlier numbering error (Pass 12 change log), but its prose and code are verbatim restatements of the earlier section. The only unique contribution of Priority 2 is the priority label and the heading — the substantive content is entirely covered by lines 11–49.
**Suggestion:** Collapse the Priority 2 block to a heading + one sentence cross-referencing the T1a/T1b section. For example:

```
**Priority 2 — Ensure correct Q/K reshape before HEIGHT_SHARDED `to_memory_config` (correctness, not a saving):**

T1a and T1b are kernel-mandated; see the T1a/T1b classification above for the mandatory three-step sequence and rationale. No latency saving is available here without a RoPE kernel change.
```

This removes ~17 lines while preserving the priority label and its correctness framing.

## MINOR Suggestions

### [transition_cost_model.md] lines 95–115: T4 and T_norm still use old code-block format
**Issue:** T4 (lines 95–103, 9 lines) and T_norm (lines 105–115, 11 lines) remain in the pre-Pass-1 prose code-block format (`Tensor:`, `Path:`, `t_transfer:`, `t_overhead:`, `t_Tx:`) while T1a–T3b were collapsed into a table in Pass 1. Each block conveys 3 numeric values that would occupy 1 table row. The inconsistency is minor since both are edge-case entries (conditional / cross-reference), but it leaves ~20 lines where ~6 would suffice.
**Suggestion:** Append T4 (marked conditional) and T_norm (marked cross-reference) as additional rows in the existing per-transition table, and remove the prose code blocks. Saves ~14 lines.

### [optimization_opportunities.md] lines 248–252: "What this optimization eliminates / does not eliminate" paragraphs restate code-block comments
**Issue:** Lines 248–252 consist of two short paragraphs ("What this optimization eliminates: ..." / "What this optimization does not eliminate: ...") that repeat what the inline code-block comments on lines 259–264 already state. The code block immediately following them (lines 254–265) makes the same point with actual syntax.
**Suggestion:** Delete lines 248–252 (the two "What this optimization..." paragraphs). The code block and its comments are sufficient. Saves ~5 lines.

### [optimization_opportunities.md] line 281: Non-Eliminable item 1 restates T1a/T1b conclusion a fourth time
**Issue:** Non-Eliminable Transitions item 1 (line 281, ~2 sentences) restates the mandatory-reshape constraint for T1a/T1b — a point already made at lines 36–49 (Feasibility paragraph), lines 212–213 (Priority 2 heading), and the summary table rows for T1a/T1b. The Non-Eliminable list has structural value as a post-optimisation residual inventory, but the sentence can be shortened to its unique content: that elimination requires a RoPE kernel change.
**Suggestion:** Shorten item 1 to one sentence: "**T1a, T1b (Q and K DRAM→L1 for RoPE):** Elimination requires modifying the RoPE kernel to accept DRAM INTERLEAVED input; the mandatory reshape from flat slice shape to per-head shape must precede the HEIGHT_SHARDED shard copy and cannot be folded into it." Saves ~1–2 lines.

## Load-Bearing Evidence
- `optimization_opportunities.md` line ~17: `"The mandatory steps leading up to T1a are: ttnn.slice produces q_slice in shape (1,1,32,2048) (DRAM INTERLEAVED), followed by ttnn.reshape to (1,16,32,128) (zero-copy on contiguous DRAM), followed by ttnn.to_memory_config into rope_shard_mem_q (T1a — the actual HEIGHT_SHARDED shard copy)."` — load-bearing because it is the definitive statement of the mandatory three-step sequence; the Priority 2 duplicate at lines 212–232 adds nothing beyond what this passage already establishes.
- `optimization_opportunities.md` line ~275: `"Approximately 19–22 µs can be recovered per decode step using only Python-level code changes in TTNNBailingMoEAttention, without any modifications to TTNN kernels."` — load-bearing as the primary no-kernel-change saving claim; must not be removed or weakened.
- `transition_cost_model.md` line ~84: table header `"Table: Per-transition cost estimates applying t_trans = t_overhead + (B / BW) [all ESTIMATE]"` — load-bearing as the authoritative numeric reference for all per-transition cost claims throughout all three files.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 2

- Change 1 (`optimization_opportunities.md` lines 212–232): Collapsed the Priority 2 block (~21 lines) to a heading + one cross-reference sentence, removing the duplicated three-step sequence prose, code block, and kernel-mandated conclusion that were already fully covered by the T1a/T1b classification section at lines 11–49.

## Agent A Change Log — B Feedback Pass 17

- Fix 1 (`optimization_opportunities.md` line 201): Added enumeration line immediately below the Priority 1 heading making the 64 µs figure directly derivable: "*Comprises: T2a (21 µs) + T2b (11 µs) + T_norm_in_Q (21 µs) + T_norm_in_K (11 µs) = 64 µs total.*"

---

# Compression Analysis: Chapter 4 — Memory-Config Transitions — Pass 3

## Summary
- Total files analyzed: 3
- Estimated current line count: ~760 lines (`decode_tensor_lifecycle.md` ~330 + `transition_cost_model.md` ~154 + `optimization_opportunities.md` ~276)
- Estimated post-compression line count: ~741 lines
- Estimated reduction: ~2.5%

## CRUCIAL Suggestions

### [transition_cost_model.md] lines 95–115: T4 and T_norm blocks remain in pre-Pass-1 verbose format
**Issue:** The Pass 1 CRUCIAL collapsed T1a–T3b from ~59 lines of repetitive code blocks into a single compact table. T4 (lines 95–103, 9 lines) and T_norm (lines 105–115, 11 lines) were not collapsed at that time and still use the old `Tensor: / Path: / t_transfer: / t_overhead: / t_Tx:` prose format. Each block conveys at most 4 numeric values — exactly what one table row holds. The existing per-transition table at lines 84–93 is the authoritative cost reference; these two blocks are structural duplicates of the table format sitting outside it.
**Suggestion:** Append T4 (marked "conditional") and T_norm (marked "see Chapter 6") as additional rows in the existing per-transition table at lines 84–93, then delete the two prose code blocks. Saves ~14 lines and resolves the format inconsistency introduced by Pass 1.

## MINOR Suggestions

### [optimization_opportunities.md] lines 232–236: "What this optimization eliminates / does not eliminate" paragraphs restate adjacent code-block comments
**Issue:** The two short paragraphs at lines 232–236 ("What this optimization eliminates: the redundant intermediate `to_memory_config` call..." / "What this optimization does not eliminate: the `ttnn.slice`, the `ttnn.reshape`, or the final `ttnn.to_memory_config` call...") repeat what the inline comments in the code block immediately following them (lines 243–248) already state. The code block is the normative reference; the paragraphs add no information beyond what the comments provide.
**Suggestion:** Delete lines 232–236 (the two "What this optimization..." paragraphs). The code block and its comments are sufficient. Saves ~5 lines.

### [optimization_opportunities.md] line ~281: Non-Eliminable item 1 restates T1a/T1b conclusion a fourth time (carried from Pass 2 MINOR — not yet applied)
**Issue:** Non-Eliminable Transitions item 1 (~2 sentences) restates the mandatory-reshape constraint for T1a/T1b — already covered in the T1a/T1b Feasibility paragraph, in the Priority 2 cross-reference, and in the summary table rows. The list entry can be shortened to its unique claim: elimination requires a RoPE kernel change.
**Suggestion:** Shorten item 1 to one sentence: "**T1a, T1b (Q and K DRAM→L1 for RoPE):** Elimination requires modifying the RoPE kernel to accept DRAM INTERLEAVED input; the mandatory reshape from flat slice shape to per-head shape must precede the HEIGHT_SHARDED shard copy." Saves ~1–2 lines.

## Load-Bearing Evidence
- `transition_cost_model.md` line ~95: `"t_T4: ≈ 21 µs (if this transition is required) [ESTIMATE]"` — load-bearing because this conditional cost estimate is referenced in the totals table and in `optimization_opportunities.md` T4 analysis; must be preserved (in table-row form) after the code block is removed.
- `transition_cost_model.md` line ~110: `"T_norm_Q (in + out): ≈ 2 × 21 µs = 42 µs / T_norm_K (in + out): ≈ 2 × 11 µs = 22 µs / T_norm_total: ≈ 64 µs [ESTIMATE]"` — load-bearing because the 64 µs figure anchors Priority 1 in `optimization_opportunities.md` and the totals table; must be preserved (in table-row form) after the code block is removed.
- `optimization_opportunities.md` line ~259: `"Approximately 19–22 µs can be recovered per decode step using only Python-level code changes in TTNNBailingMoEAttention, without any modifications to TTNN kernels."` — load-bearing as the primary no-kernel-change saving claim for Priorities 3–4.

## VERDICT
- Crucial updates: yes

## Agent A Change Log — C Compression Pass 3

- Change 1 (`transition_cost_model.md` lines 95–115): Added T4 and T_norm as rows to the existing per-transition cost table (T1a–T3b), then removed the ~20-line verbose prose code blocks; preserved unique explanatory notes (T4 conditionality, T_norm's 4-transition composition and kernel-incompatibility reason, Chapter 6 cross-reference) as a short note immediately below the table.

## Agent A Change Log — B Feedback Pass 19

- Fix 1 (`transition_cost_model.md`): Split the single T_norm row (64 µs) into two sub-rows — T_norm_in (Q-in 21 µs + K-in 11 µs = 32 µs, DRAM→L1, eliminable via Priority 1) and T_norm_out (Q-out 21 µs + K-out 11 µs = 32 µs, L1→DRAM, non-eliminable) — in both the per-transition cost table and the aggregated totals table; updated the "Dominant Transition and Why" prose to name the four sub-transitions individually (T_norm_in_Q, T_norm_in_K, T_norm_out_Q, T_norm_out_K) and explain the in/out split and its optimisation significance; table totals (150/171 µs) unchanged.
- Fix 2 (`optimization_opportunities.md`): Split the single T_norm (Q+K) summary table row into T_norm_in (Q+K) (~32 µs, partially eliminable via Priority 1) and T_norm_out (Q+K) (~32 µs, non-eliminable without norm output config change); verified Priority 1 enumeration line already correctly reads `T2a (21 µs) + T2b (11 µs) + T_norm_in_Q (21 µs) + T_norm_in_K (11 µs) = 64 µs total` and Non-Eliminable item 2 already correctly references T_norm_out — no changes needed to those two items.

---

# Compression Analysis: Chapter 4 — Memory-Config Transitions — Pass 4

## Summary
- Total files analyzed: 3
- Estimated current line count: ~745 lines (`decode_tensor_lifecycle.md` ~330 + `transition_cost_model.md` ~138 + `optimization_opportunities.md` ~277)
- Estimated post-compression line count: ~744 lines
- Estimated reduction: ~0.1%

## CRUCIAL Suggestions

**Pass 3 CRUCIAL re-check: confirmed applied correctly.**

`transition_cost_model.md` lines 86–96 now contain T4, T_norm_in, and T_norm_out as proper table rows in the per-transition cost table. The verbose prose code blocks for T4 and T_norm have been removed. The T_norm split (Pass 19) further divided the single T_norm row into T_norm_in and T_norm_out sub-rows in both the per-transition table (lines 95–96) and the aggregated totals table (lines 113–114). Table totals (150/171 µs) are unchanged. No CRUCIAL issues remain from the Pass 3 item.

No new CRUCIAL-level redundancy was introduced by the T_norm split.

## MINOR Suggestions

### [transition_cost_model.md] line 98: table note restates arithmetic already in table rows and Dominant section
**Issue:** The note below the per-transition table (line 98) includes the arithmetic "Q-in 21 µs + K-in 11 µs = 32 µs" and "Q-out 21 µs + K-out 11 µs = 32 µs" introduced by the T_norm split. This arithmetic is already visible directly in the T_norm_in and T_norm_out table rows (column Total = ≈ 32 µs each), and is restated again in full at line 129 of the Dominant Transition section. The note's arithmetic is the unique redundancy introduced by the split; the rest of the note (T4 conditionality, `use_qk_norm=True` applicability, Chapter 6 cross-reference, eliminable/non-eliminable distinction) is not duplicated.
**Suggestion:** Remove the two parenthetical arithmetic clauses from the note, retaining only the unique content: "T4 cost applies only when `paged_sdpa_decode` requires a fresh DRAM→L1 load of `q_post_rope`. T_norm_in and T_norm_out each apply only when `use_qk_norm=True`; they are listed separately because Priority 1 eliminates T_norm_in while T_norm_out remains non-eliminable without a kernel output config change. Full T_norm breakdown is in Chapter 6, `qk_norm_latency.md`." Saves ~1 line.

## Load-Bearing Evidence
- `transition_cost_model.md` line ~95–96: T_norm_in row `"≈ 32 µs"` and T_norm_out row `"≈ 32 µs"` — load-bearing because these are the post-split per-row totals that feed into the aggregated 64 µs figure anchoring Priority 1 in `optimization_opportunities.md`; must not be removed or merged back.
- `transition_cost_model.md` line ~129: `"T_norm_in (Q-in 21 µs + K-in 11 µs = 32 µs)... T_norm_out (Q-out 21 µs + K-out 11 µs = 32 µs). This distinction matters for optimisation: T_norm_in is eliminable (Priority 1 targets it together with T2a and T2b), while T_norm_out is non-eliminable without changing the norm kernel's output memory config."` — load-bearing because it is the definitive prose explanation of the in/out split significance; the table note at line 98 partially duplicates it.
- `optimization_opportunities.md` line ~260: `"Approximately 19–22 µs can be recovered per decode step using only Python-level code changes in TTNNBailingMoEAttention, without any modifications to TTNN kernels."` — load-bearing as the primary no-kernel-change saving claim; unchanged by T_norm split.

## VERDICT
- Crucial updates: no
