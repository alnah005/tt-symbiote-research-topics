## Agent A Change Log — B Feedback Pass 1

- Fix 1 (Step 2 variable name): renamed `q_norm_in_raw_k` to `k_norm_in_raw` so that Step 3's reference to `k_norm_in_raw` resolves correctly, eliminating the NameError.
- Fix 2 (Step 4 Q flops total): corrected the product 16 × 257 from "4,128" to "4,112".

## Agent A Change Log — B Feedback Pass 2

- Fix 1 (`_build_rope_tables` shape annotation): corrected the comment from `(1, 1, max_seq_len, head_dim)` to `(1, 1, max_seq_len, head_dim // 2)`, matching the actual tensor shape produced after `unsqueeze` of the 64-column cos/sin tables.
- Fix 2 (Memory Cost of Padding formula): corrected the per-table width from `head_dim=128` to `head_dim // 2 = 64`; updated the formula to `S_max × 64 × 2 bytes` per table and `S_max × 256 bytes` for both; updated the S_max=4096 example from "2 MB" to "1 MB" (4096 × 256 = 1,048,576 bytes) and replaced the S_max=8192 example with S_max=32768 ≈ 8 MB.

## Agent A Change Log — B Feedback Pass 3

- Fix 1 (padded cos/sin equivalence claim): replaced unconditional "mathematically equivalent" claim with a conditional statement; equivalence holds only if `TTNNDistributedRotaryPositionEmbedding` supports a configurable rotation offset set to `rotary_dim//2=32` rather than `head_dim//2=64`; added a prominent verification note warning that element pairing will be wrong without this parameter and that the approach must not be used until confirmed.
- Fix 2 (`_apply_rope` decode-mode cos/sin slice width): changed `[:self.rotary_dim]` to `[:self.rotary_dim // 2]` in both cos_slice and sin_slice lines for the decode branch, correcting the slice to pass exactly 32 values (one per rotation pair) instead of 64 (which included all 32 identity-padding columns as a no-op); added a comment explaining the kernel expects `rotary_dim // 2` values.

---

# Compression Analysis: Chapter 6 — RoPE and QK Norm — Pass 1

## Summary
- Total files analyzed: 2
- Estimated current line count: ~628 lines (293 + 335)
- Estimated post-compression line count: ~490 lines
- Estimated reduction: ~22%

## CRUCIAL Suggestions

None. No single block of redundancy is severe enough to constitute a structural error, mislead a reader, or produce a material audit failure. All redundancy is repetition-for-emphasis or summary restatement that, while compressible, does not rise to must-fix status.

## MINOR Suggestions

1. **`qk_norm_latency.md` line 7 — overview paragraph front-loads section structure.** The sentence "This file establishes the complete code path, derives the per-component latency breakdown, compares the aggregate cost against the fused QKV matmul for perspective, determines whether the L1 move is required by the norm kernel's input constraints or is merely a precaution, and evaluates options for reducing or eliminating the overhead." lists five things that are already section headings. Cut it; leave only the factual setup sentence ending with "≈ 64 µs of DRAM↔L1 transition overhead per step [ESTIMATE], not counting the reshape dispatch cost." Estimated saving: 3 lines.

2. **`qk_norm_latency.md` lines 27–27 — Step 1 explanatory sentence is tautological.** "T2a and T2b are defined and costed in Chapter 4. They are listed here because they are the proximate cause of why Q and K must make another DRAM→L1 trip for the norm: by the time the norm is reached, both tensors are in DRAM." The second sentence is self-evident from the code block immediately above it. Keep only the first sentence; the reader can see the tensors land in DRAM. Estimated saving: 1 line.

3. **`qk_norm_latency.md` line 109 — Step 5 repeats Step 3 verbatim.** "Same logic as Step 3: this is a metadata-only operation on a contiguous L1 INTERLEAVED tensor. No data copy." This repeats the argument given in full at lines 65–65. A back-reference suffices: "Metadata-only, same reasoning as Step 3." Estimated saving: 1 line.

4. **`qk_norm_latency.md` lines 183–190 — "Reshape Kernel Cost vs. Actual Measurement" sub-section restates Step 3.** Conditions 1 and 2 and the tile-alignment argument were already given at lines 65–65 (Step 3). The only new information is the "1–3 µs per call" dispatch overhead and "4–12 µs aggregate" estimate. The rest can be cut and the new estimate folded into a single paragraph. Estimated saving: 6 lines.

5. **`qk_norm_latency.md` line 167 — paragraph restates the immediately preceding table.** "The cost model from Chapter 4 reported T_norm_in = 32 µs and T_norm_out = 32 µs as aggregate figures. Including the norm kernel dispatch overhead (3–8 µs each for Q and K, executed sequentially), the true total lands in the 70–80 µs range per decode step [ESTIMATE]. The 64 µs figure cited in Chapter 4 covers only the DRAM↔L1 transitions; the full accounting including norm dispatch is higher." These three sentences repeat the table directly above them. The reconciliation note about 64 µs can be kept; the rest can be cut. Estimated saving: 2 lines.

6. **`qk_norm_latency.md` lines 215–221 — "Whether the Move Could Be Absorbed" subsection pre-empts Option A.** The two bullet points (kernel-fusion strategy, HEIGHT_SHARDED bypass) restate exactly what Option A then describes in detail. The subsection should be reduced to a one-sentence forward reference: "Eliminating the preceding T2a/T2b evictions would also eliminate T_norm_in; see Option A." Estimated saving: 5 lines.

7. **`qk_norm_latency.md` line 276 — conclusion restates Option A/C descriptions.** "The recommended path is Option A for short- to medium-context deployments, combined with Option C if both downstream kernels can be confirmed to accept L1 input." This duplicates the option prose. It can be cut entirely since the comparison table and the Options paragraphs already establish this clearly. Estimated saving: 1 line.

8. **`partial_rotary_rope.md` lines 5–8 — overview preview duplicates section headings.** The "This file establishes: (1)... (2)... (3)... (4)..." enumeration repeats the four section titles. Cut to two sentences: the factual setup sentence and one sentence noting the cost analysis. Estimated saving: 3 lines.

9. **`partial_rotary_rope.md` line 92 + line 105 — conclusion stated twice in close proximity.** Lines 90–92 state "the reason for choosing the non-distributed variant is not performance — it is correctness." Lines 105 restate: "The performance concern from using non-distributed RoPE is therefore not a direct latency penalty; it is an opportunity cost." These two passages make the same point four sentences apart. Merge into one. Estimated saving: 3 lines.

10. **`partial_rotary_rope.md` lines 179–186 — "Would This Enable Faster Execution?" subsection restates the table above it.** The table at lines 94–103 already shows the distributed kernel is marginally slower at batch=1. The subsection then re-explains this conclusion in prose plus two bullets. The two bullets (batch > 1 / prefill benefit) are new; the rest is redundant. Keep only those two bullets under a shortened heading. Estimated saving: 4 lines.

11. **`partial_rotary_rope.md` lines 192–209 — code block for a ruled-out strategy is bloat.** The post-blend `ttnn.where` strategy is ruled out in lines 224–224 as "strictly worse." Presenting a full implementation code block for an approach that is immediately discarded inflates the document without aiding comprehension. The concept can be described in two sentences; the code block should be cut. Estimated saving: 15 lines.

12. **`partial_rotary_rope.md` line 228 — fourth restatement of the same conclusion.** "For single-token decode, the non-distributed TTNNRotaryPositionEmbedding is the correct and performant choice." This is stated at lines 92, 105, 180, and again here. This fourth instance adds no information. Cut. Estimated saving: 1 line.

13. **`partial_rotary_rope.md` lines 326–330 — "Expected Outcome" restates the Recommendation section.** Three bullets restate content from lines 234–240 and from the earlier correctness/performance conclusion. The entire "Expected Outcome" block can be cut or collapsed to one sentence. Estimated saving: 6 lines.

## Load-Bearing Evidence

- `qk_norm_latency.md` line ~93: "At Wormhole's peak FP throughput, these operations complete in well under 1 µs [ESTIMATE]. The norm kernel cost is dominated by its dispatch overhead, not its arithmetic." — load-bearing because it establishes the core analytical finding: arithmetic is not the bottleneck, dispatch is. This distinguishes the cost model from naive FLOPs analysis and drives all optimization conclusions.
- `partial_rotary_rope.md` line ~90: "The distributed kernel offers no advantage at batch=1 decode — the dominant cost is the DRAM↔L1 data movement, not the kernel arithmetic." — load-bearing because it explains why the analysis section exists at all: the reader might expect the non-distributed path to be a performance regression, and this sentence closes that question with a quantified comparison.

## VERDICT
- Crucial updates: no

---

## Agent A Change Log — B Feedback Pass 4

- Fix 1 (`qk_norm_latency.md` ~line 274, Options A+C saving): replaced the bare "96 µs / 60%" sentence with an explicit breakdown table showing the four eliminable transition components (T2a+T2b=32 µs, T_norm_in=32 µs, T_norm_out=32 µs, reshape dispatch=4–12 µs); clarified that the 96 µs figure refers to the four DRAM↔L1 transition costs alone (T2a+T2b+T_norm_in+T_norm_out) and that including reshape elimination raises the total to 100–108 µs; removed the undefined "60%" claim.
- Fix 2 (`qk_norm_latency.md` ~line 163, first table 70–80 µs vs second table 74–92 µs): added a note immediately below the first table explicitly stating it excludes reshape dispatch overhead (≈ 4–12 µs across four `ttnn.reshape` calls) and directing the reader to the second table for the complete estimate.
- Fix 3 (`partial_rotary_rope.md` ~line 296, cos_slice/sin_slice shape comment): corrected the shape annotation from `(1, 1, seq_len, head_dim)` to `(1, 1, seq_len, head_dim // 2)` on both the cos_slice and sin_slice lines, matching the actual 64-column table width stored in `self.cos_cached` / `self.sin_cached`.
