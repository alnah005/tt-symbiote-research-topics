# Compression Analysis — Chapter 6: Sequence Length Scaling Analysis

## Crucial updates: yes

---

### Duplication 1: CCL All-to-All Formula and Worked Example

**Source (original):**
`ch5_identifying_gap/gap_attribution.md`, lines 148–170 — "Method 3: Check If the Gap Aligns with a CCL Collective," Step 1. Contains the CCL per-chip message size formula (`message_bytes = (seq_len * top_k * d_model * bytes_per_element) / num_chips`), identical constants (seq_len=1024, top_k=8, d_model=7168, bytes_per_element=2, num_chips=8, effective_bw=7e9), and the same computed result (~2.1 ms).

**Duplicate:**
`ch6_sequence_length_scaling/scaling_theory.md`, lines 86–119 — "CCL All-to-All" subsection. Presents the same formula split into `total_bytes` and `per_chip_bytes` steps, the same Python worked example with the same five constants, and the same 2.10 ms output. The math is identical; only variable names and inline comment wording differ slightly (`message_bytes` vs. `per_chip_bytes`, `ccl_latency_s` used in both).

**Recommended action:**
Collapse the ch6 worked example to a two-line pointer. Keep the `total_bytes = seq_len × top_k × d_model × 2` formula line and the scaling-law conclusion ("CCL latency scales as O(seq_len)") because those serve ch6's theoretical argument. Remove the duplicate Python code block and replace with: *"For the full parameter values and worked calculation at seq_len=1024 (result: ~2.10 ms), see [Method 3 in `ch5_identifying_gap/gap_attribution.md`](../ch5_identifying_gap/gap_attribution.md#method-3-check-if-the-gap-aligns-with-a-ccl-collective)."* The note at lines 125–131 (`/num_chips` factor clarification) is load-bearing for ch6's scaling-law interpretation and should be retained.

---

### Duplication 2: Key Model Configuration Table

**Source (original):**
`ch4_moe_op_breakdown/index.md`, lines 95–107 — "Key Model Configuration" table. Contains: `d_model`=7168, `d_ff (per expert)`=2048, `num_experts`=128, `top_k`=8, Hardware (Wormhole B0; 80 Tensix cores, ~131 TFLOP/s BF16, ~300 GB/s DRAM), Dtype BF16. Note: ch5 `index.md` line 137 already handles this correctly by cross-referencing ch4 ("All other dimensions are defined in [Chapter 4's configuration table]") rather than re-printing it.

**Duplicate:**
`ch6_sequence_length_scaling/index.md`, lines 155–170 — "Key Model Configuration (Used Throughout This Chapter)" table. Seven rows overlap directly with ch4's table (d_model, d_ff, num_experts, top_k, dtype, hardware). Ch6 adds two rows not in ch4's version: `Model` (DeepSeek-V3 / Qwen 235B-A22B) and `Ethernet bandwidth` (~7 GB/s effective per inter-chip link). The six shared rows are verbatim or near-verbatim duplicates.

**Recommended action:**
Follow ch5's pattern. Replace the six overlapping rows with a single cross-reference line: *"Architecture constants (d_model, d_ff, num_experts, top_k, dtype, hardware) are the same as [Chapter 4's configuration table](../ch4_moe_op_breakdown/index.md#key-model-configuration-used-throughout-this-chapter)."* Retain only the two ch6-specific rows (Model name and Ethernet bandwidth) in a small supplementary table, since the ethernet bandwidth value is directly load-bearing for the CCL latency calculations throughout the chapter and the model name disambiguates DeepSeek-V3 vs. Qwen 235B-A22B equivalence for first-time ch6 readers.

---

## Load-Bearing Evidence

Not applicable (crucial duplications found above). The retained content in each case is specified inline in the Recommended action fields.

---

## MINOR Suggestions

1. **`compute_stats` defined twice in `experiment_design.md`** — The function appears at lines 123–131 (inside a docstring-annotated standalone version for illustration) and again at lines 251–258 (inside the pytest block) and lines 325–332 (inside the standalone script). The standalone illustration copy at lines 123–131 is redundant with the two script-embedded copies. Consider removing the illustration copy and adding a comment to the pytest and script copies pointing readers to the "Sample Size" section for the interpretation context.

2. **Pattern behavior descriptions in `index.md` vs. `common_gap_patterns.md`** — The four "Behavior" sections in `ch6/index.md` (lines 63–126) are thematically parallel to ch5's Pattern A–D descriptions but reframed around scaling predictions rather than diagnostic steps. This parallel structure is intentional and load-bearing for ch6 (readers need the prediction before designing the sweep), so no removal is recommended. A minor improvement would be to add an explicit sentence at the start of each Behavior block such as: *"This is the predicted scaling of [Pattern B / Pattern C / etc.] from Chapter 5."* This would tie the ch6 prediction framework back to the ch5 pattern catalog without duplicating prose.

3. **`interpreting_scaling_results.md` "Full Attribution Summary" table** — The table at lines 307–312 anticipates ch7 content. The note "Chapter 7 provides the full template" is present. No compression needed, but consider whether the table should live in ch7 instead, with a forward pointer from ch6. As written it is a useful ch6 deliverable checkpoint and is not duplicated elsewhere.

4. **`experiment_design.md` Tracy export command** — The `tracy-csvexport -u output.tracy > zones_seq${SEQ_LEN}.csv` command at line 103 is identical in form to commands introduced in ch2 and ch5. This is appropriate repetition (readers need the exact command in context) and does not warrant removal, but a parenthetical cross-reference to `ch2_tracy_setup/output_format.md` would reduce the risk of the two diverging if the flag syntax changes.

## Agent A Change Log — C Feedback Pass 1
- scaling_theory.md: Replaced CCL worked example Python block with Chapter 5 cross-reference; kept abstract formula and scaling conclusion
- index.md: Replaced model configuration table with Chapter 4 cross-reference; retained T3K ethernet bandwidth value

## Pass 2 Verification

**Fix 1 — scaling_theory.md CCL worked example:**
Verified. The 20-line Python code block (seq_len=1024, top_k=8, d_model=7168, bytes_per_element=2, num_chips=8, effective_bw=7e9 → 2.10 ms) has been removed. In its place is the single sentence: "See Chapter 5, `gap_attribution.md`, Method 3 for the Python calculation confirming ~2.1 ms at seq_len=1024." The abstract `total_bytes` and `per_chip_bytes` formula blocks (lines 88–98) are retained, as is the scaling-law conclusion paragraph and the load-bearing `/num_chips` note. The cross-reference target (`## Method 3: Check If the Gap Aligns with a CCL Collective`) exists in `ch5_identifying_gap/gap_attribution.md` at line 129.

**Fix 2 — index.md model configuration table:**
Verified. The six overlapping rows (d_model, d_ff, num_experts, top_k, dtype, hardware) have been removed. The section now opens with the prose cross-reference: "Model configuration constants (d_model, d_ff, top_k, num_experts for Qwen/DeepSeek): see Chapter 4, `index.md`." A two-row supplementary table retains the ch6-specific `Model` name row and the `Ethernet bandwidth` (~7 GB/s) row, which is load-bearing for CCL latency calculations throughout the chapter. The cross-reference target (`## Key Model Configuration (Used Throughout This Chapter)`) exists in `ch4_moe_op_breakdown/index.md` at line 94.

### Remaining Crucial Duplications Check

No remaining crucial duplications found. The abstract formula blocks retained in `scaling_theory.md` (total_bytes and per_chip_bytes expressions) serve ch6's own scaling-law derivation and are not duplicates of the ch5 Python code — they present the algebraic form, not the numeric calculation. The `Model` row retained in `index.md` is not present in ch4's table (ch4 does not include a model-name row) and is therefore not a duplication.

## Crucial updates: no

## Load-Bearing Evidence
Both fixes reduced content without removing any load-bearing material. In Fix 1, the retained abstract formula lines are necessary for the scaling-law argument that CCL is O(seq_len) — removing them would break the logical chain in the CCL All-to-All subsection. The `/num_chips` note is load-bearing because it clarifies a common misreading of the per-chip formula. In Fix 2, the retained `Ethernet bandwidth` row (~7 GB/s) is directly referenced in CCL latency estimates throughout the chapter; without it, readers would have no local source for the bandwidth constant used in the "~2.1 ms at seq_len=1024" claim.

## MINOR Suggestions
1. **`scaling_theory.md` cross-reference wording** — The replacement sentence "See Chapter 5, `gap_attribution.md`, Method 3 for the Python calculation confirming ~2.1 ms at seq_len=1024" is functional but could be made a relative Markdown link for readers using a rendered documentation viewer: `See Chapter 5, [Method 3 in \`gap_attribution.md\`](../ch5_identifying_gap/gap_attribution.md#method-3-check-if-the-gap-aligns-with-a-ccl-collective) for the Python calculation confirming ~2.1 ms at seq_len=1024.` This matches the link style already used elsewhere in the guide.
2. **`index.md` cross-reference wording** — Similarly, "see Chapter 4, `index.md`" could be a relative link: `see [Chapter 4, \`index.md\`](../ch4_moe_op_breakdown/index.md#key-model-configuration-used-throughout-this-chapter)` to keep navigation consistent with the ch5 cross-reference pattern noted in the original compression_analysis.md.
