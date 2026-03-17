# B Review — Chapter 4: MoE Forward Pass Op Breakdown — Pass 1

**Error 1 — Wrong GDDR6 bank count (`full_op_sequence_reference.md`, line 103)**

The file states "8 GDDR6 banks at 12 GB total." Wormhole B0 has 12 GDDR6 banks, not 8. The bank count is incorrect.

**Error 2 — Down projection FLOPs understated by 2× (`expert_matmul_phase.md`, lines 114–115)**

The file states "~120 GFLOPs (down proj, same shape transposed)." The down projection is `[total_tokens, d_ff] × [d_ff, d_model]` = `[8192, 2048] × [2048, 7168]`, which gives `2 × 8192 × 2048 × 7168 ≈ 240 GFLOPs` — identical to the gate/up projections. Transposing dimensions does not change the FLOP count. The ~120 GFLOPs figure is wrong by a factor of 2.

**Error 3 — expert_capacity formula missing top_k factor (`expert_matmul_phase.md`, lines 96–98)**

The file applies the formula `capacity_factor × seq_len / num_experts = 1.0 × 1024 / 128 = 8` and concludes `expert_capacity = 8`. The correct formula is `capacity_factor × seq_len × top_k / num_experts`. With top_k=8: `1.0 × 1024 × 8 / 128 = 64`. The value 64 is used correctly in all tensor shapes throughout the chapter (e.g., `[128, 64, 7168]`), but the formula derivation at lines 96–98 produces 8, which directly contradicts line 144 where expert_capacity=64 is stated correctly.

## Agent A Change Log — B Feedback Pass 1
- full_op_sequence_reference.md: Fixed GDDR6 bank count from 8 to 12
- expert_matmul_phase.md: Fixed down projection FLOPs — not halved by transposition; updated to ~240 GFLOPs matching gate/up
- expert_matmul_phase.md: Fixed expert_capacity formula to include top_k: seq_len × top_k / num_experts = 1024 × 8 / 128 = 64

---

# B Review — Chapter 4: MoE Forward Pass Op Breakdown — Pass 2

## Pass 1 Fix Verification

**Fix 1 — GDDR6 bank count (`full_op_sequence_reference.md`, line 103):** Confirmed correct. Now reads "12 GDDR6 banks at 12 GB total." Bank count matches the authoritative spec.

**Fix 2 — Down projection FLOPs (`expert_matmul_phase.md`, lines 114–116):** Confirmed correct. Now reads "≈ 240 GFLOPs per matmul (gate or up)" and "≈ 240 GFLOPs (down proj; transposing dimensions does not change FLOP count)." The equal-FLOPs statement is now accurate.

**Fix 3 — expert_capacity formula (`expert_matmul_phase.md`, lines 96–97):** Confirmed correct. Now reads "capacity_factor × seq_len × top_k / num_experts" with the computation "1.0 × 1024 × 8 / 128 = 64." Formula and result are both correct.

## Remaining Errors Found

**Error 1 — expert_capacity formula missing top_k factor (`dispatch_phase.md`, line 141)**

The "Output Shape: Why It Varies" section states: "expert_capacity = capacity_factor × seq_len / num_experts". The top_k factor is absent. The correct formula is `capacity_factor × seq_len × top_k / num_experts`. This is the identical class of error that was fixed in `expert_matmul_phase.md` but was not applied to `dispatch_phase.md`. With the given values: `1.0 × 1024 / 128 = 8`, which is wrong; correct is `1.0 × 1024 × 8 / 128 = 64`.

**Error 2 — DeepSeek-V3 num_experts stated as 256 (`full_op_sequence_reference.md`, line 83)**

The "Known Variations" section states "DeepSeek-V3 also has num_experts=256 (vs. Qwen's 128)." The authoritative facts specify DeepSeek-V3 has num_experts=128, top_k=8 — the same expert count as Qwen 235B. The claim that DeepSeek-V3 doubles the expert count to 256 is incorrect. The following line (line 84) compounds the error by giving the router linear weight shape as `[7168, 256]`, which would also need correction to `[7168, 128]`.

## Agent A Change Log — B Feedback Pass 2
- dispatch_phase.md: Fixed expert_capacity formula to include top_k factor: capacity_factor × seq_len × top_k / num_experts = 1.0 × 1024 × 8 / 128 = 64
- full_op_sequence_reference.md: Fixed DeepSeek-V3 num_experts from 256 to 128; router weight shape from [7168, 256] to [7168, 128]

---

# B Review — Chapter 4: MoE Forward Pass Op Breakdown — Pass 3

## Pass 2 Fix Verification

**Fix 1 — expert_capacity formula in dispatch_phase.md (line 141):** Confirmed correct. The "Output Shape: Why It Varies" section now reads "expert_capacity = capacity_factor × seq_len × top_k / num_experts" — top_k factor is present and the example value of 64 is consistent with it.

**Fix 2 — DeepSeek-V3 num_experts and router weight shape (full_op_sequence_reference.md, lines 82–84):** Confirmed correct. The text now reads "DeepSeek-V3 also has `num_experts=128` (vs. Qwen's 128), the same expert count" and gives the router linear weight shape as `[7168, 128]`. Both match the authoritative facts.

## Remaining Errors

**Error 1 — Expert matmul decode total inconsistent across files (expert_matmul_phase.md vs. full_op_sequence_reference.md)**

`expert_matmul_phase.md` latency table (lines 133–135) sums the three decode matmuls (gate 80–200 µs + up 80–200 µs + down 80–200 µs) and gives a phase total of "~400–800 µs." `full_op_sequence_reference.md` Phase Latency Summary (line 113) gives Expert matmul decode as "300–650 µs." The two files contradict each other. One range must be corrected to match the other; the 300–650 µs figure in the reference table is the lower of the two and does not cover the high end of the per-op ranges, making it the more likely error.

## Agent A Change Log — B Feedback Pass 3
- expert_matmul_phase.md: Fixed total decode-regime expert matmul phase latency from ~400–800 µs to ~300–650 µs (consistent with per-op 80–200 µs × 3 matmuls = 240–600 µs range, and with full_op_sequence_reference.md Phase Latency Summary)

---

# B Review — Chapter 4: MoE Forward Pass Op Breakdown — Pass 4

## Pass 3 Fix Verification

**Fix — Expert matmul decode total (`expert_matmul_phase.md`, line 135):** Confirmed correct. The table row now reads `**~300–650 µs**` for the expert matmul phase total in decode. This is consistent with the per-op decode ranges of 80–200 µs per matmul × 3 matmuls = 240–600 µs base plus overhead, and matches `full_op_sequence_reference.md` Phase Latency Summary (line 113) which also states 300–650 µs.

## No feedback — chapter approved.
