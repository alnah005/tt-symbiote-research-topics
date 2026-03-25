# B Review — Chapter 7: Llama3 Walkthrough — Pass 1

## Issues Found

1. **`multi_device_decode.md` Step 13 — Vocab-parallel LM head collective explanation was internally inconsistent.**
   The file correctly stated that each device holds a weight shard of shape `[8192, vocab_size/8]` and produces output `[batch, vocab_size/8]` — a distinct vocabulary shard per device. It then explained that `ttnn.all_reduce` is used because "partial outputs must be summed (reduced), not concatenated." This reasoning is backwards for the stated weight decomposition: if each device holds a non-overlapping vocabulary shard, the correct assembly operation to produce the full `[batch, vocab_size]` tensor would be concatenation (all_gather), not summation (all_reduce). The collective itself (`ttnn.all_reduce`) is confirmed correct; the explanation was wrong.
   **Fix applied:** Replaced the erroneous explanation with a neutral statement that `ttnn.all_reduce` produces the complete replicated logit vector needed for next-token sampling on every device, without asserting a false reason about why summation applies.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Llama 3.1 8B: hidden=4096, n_heads=32, n_kv_heads=8, head_dim=128, 32 layers | CORRECT — `single_device_decode.md` model config table matches exactly |
| N150: single Wormhole chip | CORRECT — stated in `index.md` and `single_device_decode.md` |
| Accuracy mode: BFP8 MLP, BFP8 attn → ~23 t/s/u | CORRECT — `single_device_decode.md` precision tables and throughput figures |
| Performance mode: BFP4 MLP, BFP8 attn → ~28 t/s/u | CORRECT — `single_device_decode.md` precision tables and throughput figures |
| LM head and embedding: always BF16/HiFi4 | CORRECT — confirmed in both `single_device_decode.md` and `multi_device_decode.md` |
| Math fidelity: BFP4→LoFi, BFP8→HiFi2, BF16→HiFi4 | CORRECT — all three pairings stated correctly throughout both files |
| BFP4 vs BF16 bandwidth: 3.56x | CORRECT — `single_device_decode.md` states 3.56x |
| BFP8 vs BF16 bandwidth: 1.88x | CORRECT — `single_device_decode.md` states 1.88x |
| Weight layout: pre-transposed [K, N] = [in_features, out_features] | CORRECT — stated in `single_device_decode.md` weight loading section |
| DRAM-sharded matmul for decode linear layers | CORRECT — confirmed at every linear layer in both files |
| Paged KV cache: [1, 8, n_blocks*block_size, 128] for Llama 3.1 8B | CORRECT — `single_device_decode.md` Step 4 |
| current_pos: scalar int | CORRECT — confirmed in both files |
| Warm-up: JIT compilation only, no trace capture | CORRECT — `single_device_decode.md` correctly separates warm-up (JIT) from subsequent trace capture as a distinct step |
| Llama 3.1 70B: 64 Q heads, 8 KV heads, 80 layers, TP=8 on T3K | CORRECT — `multi_device_decode.md` model config table |
| T3K: 8-chip Wormhole Ethernet ring (NOT mesh) | CORRECT — explicitly stated as ring topology in `multi_device_decode.md` |
| TP=8: each device → 8 Q heads, 1 KV head | CORRECT — `multi_device_decode.md` model config and Step 2/4 |
| GQA G=8 for 70B (64 Q / 8 KV) | CORRECT — `multi_device_decode.md` model config table |
| Column-parallel: ttnn.all_gather; Row-parallel: ttnn.reduce_scatter (NOT all_reduce) | CORRECT — `multi_device_decode.md` Steps 5 and 11; collective summary table |
| Vocab-parallel LM head: ttnn.all_reduce (NOT all_gather) | CORRECT — collective used is confirmed; explanation corrected in this pass |
| GQA TP=8=n_kv_heads: no KV replication needed | CORRECT — stated explicitly in `multi_device_decode.md` |
| Performance mode ~22% faster than accuracy mode | CORRECT — stated in `index.md` key concepts table and `single_device_decode.md` |

VERDICT: CHANGES NEEDED — 1 issue found and fixed. The collective operation choice (`ttnn.all_reduce` for the vocab-parallel LM head) was correct and confirmed. The prose explanation for why that collective is used was internally inconsistent with the stated weight decomposition; the explanation has been corrected in `multi_device_decode.md`.

---

# B Review — Chapter 7: Llama3 Walkthrough — Pass 2

## Issues Found

1. **`multi_device_decode.md` Collective Communication Summary table — residual incorrect rationale for `ttnn.all_reduce` at LM head.**
   The table's "Why" column for the LM head row read "Sum vocab-parallel partial logits; every device needs full logits." This repeated the same flawed summation rationale that Pass 1 corrected in the Step 13 prose: for a vocab-parallel split, each device produces logits for a non-overlapping vocabulary shard, so the assembly operation is not a sum of partial contributions. The confirmed collective (`ttnn.all_reduce`) was already correct. Only the stated rationale was wrong.
   **Fix applied:** Replaced "Sum vocab-parallel partial logits; every device needs full logits" with "Assemble full logit tensor on every device; every device needs full logits for next-token sampling" in the collective summary table of `multi_device_decode.md`.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Column-parallel: ttnn.all_gather; Row-parallel: ttnn.reduce_scatter (NOT all_reduce) | CORRECT — Steps 5 and 11 and collective summary table all use reduce_scatter for row-parallel |
| Vocab-parallel LM head: ttnn.all_reduce (NOT all_gather) | CORRECT — collective used is confirmed in Step 13 and summary table |
| T3K: 8-chip Ethernet ring (NOT mesh) | CORRECT — explicitly stated as ring in model config section |
| 70B: 64 Q heads, 8 KV heads; 8 Q + 1 KV per device, no replication needed | CORRECT — model config table and Step 2/4 confirm |
| BFP4→LoFi, BFP8→HiFi2, BF16→HiFi4 | CORRECT — all three pairings consistent throughout both files |
| BFP4 vs BF16: 3.56x; BFP8 vs BF16: 1.88x | CORRECT — single_device_decode.md states both figures |
| Llama 3.1 8B: ~23 t/s/u accuracy, ~28 t/s/u performance; ~22% faster | CORRECT — single_device_decode.md precision sections and key takeaways |
| current_pos: scalar int; warm-up: JIT only, no trace capture | CORRECT — confirmed in both files |

VERDICT: CHANGES NEEDED — 1 issue found and fixed. The `ttnn.all_reduce` collective at the LM head was and remains correct. The "Why" entry in the collective summary table still described the operation as summing partial logits (the same flawed rationale corrected in Pass 1 Step 13 prose but missed in the table); the table entry has been corrected to neutral assembly language.

---

# B Review — Chapter 7: Llama3 Walkthrough — Pass 3

## Issues Found

1. **`multi_device_decode.md` Step 13 — Residual incorrect summation rationale persisted in `all_reduce` vs `all_gather` contrast paragraph (line 290 pre-fix).**
   The paragraph read: "The distinction matters: `all_gather` concatenates shards (producing `[batch, vocab_size]` by joining the `vocab_size/8` slices), while `all_reduce` sums partial contributions across all devices and replicates the result." This still characterized `ttnn.all_reduce` as the operation that "sums partial contributions" — the same flawed summation framing corrected in Pass 1 (Step 13 main prose) and Pass 2 (collective summary table "Why" column). For a vocab-parallel split, each device holds a non-overlapping vocabulary shard; the confirmed rationale (per key facts) is that `ttnn.all_reduce` is used to assemble the full logit tensor on every device for next-token sampling, not because summation of partial contributions is the correct mathematical description of the assembly. The paragraph was also internally contradictory: it correctly noted that `all_gather` would concatenate shards to produce the full `[batch, vocab_size]` tensor, yet asserted `all_reduce` (summation) is preferable — without justifying why summation is correct for non-overlapping vocab shards.
   **Fix applied:** Removed the erroneous contrast sentence. The corrected paragraph now simply states that `ttnn.all_reduce` is used so that every device ends up with the complete logit vector needed for next-token sampling, consistent with the confirmed key fact rationale and the fixes already applied in Passes 1 and 2.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Vocab-parallel LM head: ttnn.all_reduce; rationale is assembling full logit tensor on every device for sampling (NOT "sum partial logits") | CORRECT — collective summary table "Why" (Pass 2 fix) confirmed present; erroneous summation contrast sentence removed in this pass |
| Column-parallel: ttnn.all_gather; Row-parallel: ttnn.reduce_scatter | CORRECT — Steps 5 and 11 and collective summary table consistent throughout |
| T3K: 8-chip Ethernet ring (NOT mesh) | CORRECT — explicitly stated as ring in model config section |
| 70B TP=8: 8 Q + 1 KV per device | CORRECT — model config table and Steps 2/4 confirm |
| BFP4→LoFi; BFP8→HiFi2; BF16→HiFi4 | CORRECT — all three pairings consistent throughout both files |
| current_pos: scalar int; warm-up: JIT only | CORRECT — confirmed in both files |

VERDICT: CHANGES NEEDED — 1 issue found and fixed. The Pass 2 fix to the collective summary table "Why" column is confirmed in place. A residual instance of the summation rationale was found in the Step 13 `all_reduce` vs `all_gather` contrast paragraph of `multi_device_decode.md` and has been removed. All collective assignments, math fidelity pairings, T3K topology, and head-distribution facts are correct across both files.

---

# B Review — Chapter 7: Llama3 Walkthrough — Pass 4

## Issues Found

None found.

Pass 3 mandate verified: the `all_reduce` rationale is consistently "every device gets the complete logit vector needed for next-token sampling" throughout both files. No residual summation framing ("sums partial contributions," "sums partial logits") was found in any location — Step 13 prose, the collective summary table "Why" column, or the Key Takeaways section of `multi_device_decode.md`. `single_device_decode.md` does not use `all_reduce` and required no check on this point.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Vocab-parallel LM head: ttnn.all_reduce; rationale is "every device gets complete logit vector for sampling" (NOT "sums partial contributions" or "sums partial logits") | CORRECT — Step 13 prose (lines 285–290), collective summary table "Why" column, and Key Takeaways all use assembly/sampling language only; no summation framing present |
| Column-parallel: ttnn.all_gather; Row-parallel: ttnn.reduce_scatter (NOT all_reduce) | CORRECT — multi_device_decode.md Steps 5 and 11, collective summary table, and Key Takeaways all consistent |
| T3K: Ethernet ring (NOT mesh); TP=8; 70B: 8 Q + 1 KV per device | CORRECT — model config section explicitly states ring; Steps 2 and 4 confirm head distribution |
| BFP4→LoFi; BFP8→HiFi2; BF16→HiFi4; BFP4 3.56x; BFP8 1.88x | CORRECT — all pairings and bandwidth multipliers consistent throughout both files |
| Llama 3.1 8B: ~23 t/s/u accuracy, ~28 t/s/u performance | CORRECT — single_device_decode.md precision sections confirm both figures |
| current_pos: scalar int; warm-up: JIT only | CORRECT — confirmed in both files |

VERDICT: APPROVED

---

# B Review — Chapter 7: Llama3 Walkthrough — Pass 5

## Issues Found

None found.

C Pass 1 edits verified:

1. `single_device_decode.md` Step 13: LM head rationale is now a clean pointer to the dedicated "LM Head and Embedding: Always BF16" subsection (lines 299–300: "The LM head is never quantized to BFP8 or BFP4 in either precision mode. See 'LM Head and Embedding: Always BF16' above for the rationale."). The subsection itself (lines 65–70) is factually correct: BF16 weight, HiFi4 fidelity, unquantized in both modes — consistent with confirmed key facts.

2. `multi_device_decode.md` Step 11: reduce_scatter rationale now reads "[confirmed] `ttnn.reduce_scatter` is used, not `ttnn.all_reduce` — same rationale as Step 5." Step 5 (lines 157–163) carries the full confirmed rationale: reduce_scatter sums partial row-parallel contributions and distributes the result in sharded form, preferred over all_reduce because the result feeds directly into a subsequent all_gather or sharded op. The back-reference is accurate and sufficient.

## Verification of Confirmed Facts

| Confirmed Fact | Status |
|---|---|
| Vocab-parallel LM head: BF16, HiFi4; always unquantized | CORRECT — single_device_decode.md Step 13 and "LM Head and Embedding: Always BF16" subsection; multi_device_decode.md Step 13 and model config table all consistent |
| Row-parallel: ttnn.reduce_scatter (NOT all_reduce) | CORRECT — multi_device_decode.md Steps 5 and 11; collective summary table; Key Takeaways all use reduce_scatter; "same rationale as Step 5" back-reference is accurate |
| T3K: Ethernet ring; 70B: TP=8, 8 Q + 1 KV per device; no KV replication | CORRECT — multi_device_decode.md model config, topology section, Steps 2/4, and Key Takeaways all consistent |
| BFP4→LoFi; BFP8→HiFi2; BF16→HiFi4 | CORRECT — all three pairings consistent throughout both files |
| Llama 3.1 8B: ~23 t/s/u accuracy, ~28 t/s/u performance | CORRECT — single_device_decode.md precision sections confirm both figures |

VERDICT: APPROVED
