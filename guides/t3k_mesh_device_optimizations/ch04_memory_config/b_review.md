# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 1

1. **decode_memory_strategy.md, "Current-Token Hidden States and Query Projections" section (lines 56–58) — incorrect per-core calculation.**
   The text states: "When sharded across 80 Tensix cores using HEIGHT_SHARDED: Per-core bytes = 448 KB / 80 = 5.6 KB/core." This is wrong. HEIGHT_SHARDED assigns whole rows, not fractional bytes. With B=32 rows and 80 cores, each core receives ceil(32/80) = 1 row. That row has H=7168 elements at 2 bytes each, giving 1 × 7168 × 2 = 14,336 bytes = **14.0 KB/core**. The naive division 448 KB / 80 is not meaningful because rows cannot be split across cores. The chapter's own worked example (lines 207–209) and its own summary table (line 302) both correctly state 14.0 KB/core, making this a self-contradiction within the same file.

2. **index.md, summary table (line 52) — propagates the same incorrect per-core value.**
   The row for "Current-token hidden states / query" states "~5.7 KB/core at B=32, H=7168." This figure originates from the flawed naive division in issue 1. The correct value is 14.0 KB/core (confirmed by decode_memory_strategy.md's worked example and summary table). The index table must be corrected to match.

3. **prefill_memory_strategy.md, attention intermediate threshold — stated cutoff (S ≤ 512) contradicts the algebra shown (S ≤ 8720 for B=1).**
   The section heading and surrounding text declare "Short sequences (S ≤ 512): L1 HEIGHT_SHARDED / Long sequences (S > 512): DRAM interleaved." However, the derivation immediately following shows the per-core shard fits in 1.5 MB as long as B·S ≤ 80 × 109 = 8,720 — i.e., for B=1, the L1 shard fits up to S=8,720, and for B=4 up to S=2,180. Neither bound is S=512. The 512 threshold appears nowhere in the algebra. The text needs either (a) a corrected derivation that produces S=512 as the bound, or (b) a corrected threshold consistent with the math shown (with an explicit note if 512 is intentionally more conservative than the math requires, and an explanation of why).

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 2

**Pass 1 fixes verified:** All 4 fixes from Pass 1 are correctly applied.

1. `decode_memory_strategy.md` hidden states section (lines 56–58): correctly states ceil(32/80) = 1 row/core, 14.0 KB/core. Confirmed.
2. `decode_memory_strategy.md` dispatch buffer section (lines 140–142): correctly states 14.0 KB/core with same HEIGHT_SHARDED derivation. Confirmed.
3. `index.md` summary table (lines 52, 54): both rows now read "14.0 KB/core" with "1 row/core HEIGHT_SHARDED" annotation. Confirmed.
4. `prefill_memory_strategy.md`: threshold correctly changed to "B·S ≤ 2,880" with correct derivation (3 × ceil(B·S/80) × 14,336 ≤ 1,572,864 → B·S ≤ 2,880). All S ≤ 512 references removed. Confirmed.

**Remaining issues found:**

5. **`prefill_memory_strategy.md`, line 106 — incorrect per-core comment for B=1, S=2048.**
   The code comment reads: `# Example: B=1, S=2048 -> B*S=2048 <= 2880 -> L1 HEIGHT_SHARDED (~98 KB/core single tensor)`.
   The correct per-core bytes for a single Q (or K or V) tensor at B=1, S=2048 with HEIGHT_SHARDED across 80 cores is: ceil(2048/80) × H × 2 = 26 × 7168 × 2 = 372,736 bytes ≈ 364 KB/core per tensor.
   The value ~98 KB/core corresponds instead to B·S=512 (ceil(512/80) = 7 rows/core; 7 × 7168 × 2 = 100,352 bytes ≈ 98 KB), not B·S=2048. The comment is carried over from the old S ≤ 512 threshold and was not updated when the threshold changed to B·S ≤ 2,880. The correct comment should state ~364 KB/core per tensor at B=1, S=2048 (with Q+K+V combined ≈ 1.07 MB/core, which is consistent with the prose on line 67 that correctly states 1.07 MB/core for Q+K+V at this configuration).

6. **`wormhole_memory_hierarchy.md`, lines 199–203 — function output annotation is inconsistent with the function's own logic.**
   The function `per_core_bytes_height_sharded` is called with `M=512, N=7168, n_cores=80`, and the annotation claims "Per-core shard: 91.0 KB". Tracing the function: M_t = ceil(512/32) = 16 tile-rows; N_t = ceil(7168/32) = 224 tile-cols; per_core_M_t = ceil(16/80) = 1; bytes_per_tile = 32 × 32 × 2 = 2,048; return = 1 × 224 × 2,048 = 458,752 bytes = 448 KB. The annotated output of 91.0 KB is inconsistent with the function as written. (For reference, a naive element-level approach gives ceil(512/80) × 7168 × 2 = 7 × 14,336 = 100,352 bytes ≈ 98 KB, still not 91 KB.) The annotation must be corrected to 448 KB to match the function's actual output given the tile-aligned formula used.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 1 Fixes Applied

1. `decode_memory_strategy.md` lines 56–58 (hidden states section) — Fixed per-core calculation: changed naive 448 KB / 80 = 5.6 KB/core to correct HEIGHT_SHARDED calculation: ceil(32/80) = 1 row × H × 2 = 14,336 bytes = 14.0 KB/core.

2. `decode_memory_strategy.md` line 142 (dispatch buffer section) — Same fix applied: 5.6 KB/core → 14.0 KB/core.

3. `index.md` lines 52, 54 — Updated both "~5.7 KB/core" and "~5.6 KB/core" to "14.0 KB/core" with clarification "1 row/core HEIGHT_SHARDED."

4. `prefill_memory_strategy.md` — Changed Q/K/V placement threshold from "S ≤ 512" (unsupported by the inline math) to "B·S ≤ 2,880" (derived from the actual crossover formula accounting for Q+K+V simultaneously in L1). Added derivation: 3 × ceil(B·S/80) × H × 2 ≤ 1,572,864 → B·S ≤ 80 × 36 = 2,880. Updated all references consistently.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 2 Fixes Applied

5. `prefill_memory_strategy.md` line 106 comment — Corrected stale per-core annotation: "~98 KB/core" → "~364 KB/core" for B=1, S=2048 (correct: ceil(2048/80) = 26 rows × 7168 × 2 = 372,736 bytes ≈ 364 KB/core per tensor).

6. `wormhole_memory_hierarchy.md` function output annotation — Corrected "Per-core shard: 91.0 KB" → "Per-core shard: 448.0 KB (1 tile-row × 224 tile-cols × 2048 bytes/tile)" for `per_core_bytes_height_sharded(M=512, N=7168, n_cores=80)`: M_t=ceil(512/32)=16, N_t=224, per_core_M_t=ceil(16/80)=1, return=1×224×2048=458,752 bytes=448 KB.

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 3

**Verification of Pass 2 fixes:**

Fix 5 (`prefill_memory_strategy.md` line 106 comment): now reads `~364 KB/core single tensor: ceil(2048/80)=26 rows × 7168 × 2 = 372736 bytes`. Correct. Confirmed.

Fix 6 (`wormhole_memory_hierarchy.md` annotation lines 202–203): now reads `# Per-core shard: 448.0 KB  (1 tile-row × 224 tile-cols × 2048 bytes/tile)`. Correct. Confirmed.

**Remaining issues found:**

7. **`prefill_memory_strategy.md`, summary table line 322 — wrong shape and wrong size for the "B·S > 2880" Q/K/V row.**

   The table header column says "Shape (B=1, S=2048)" but the row for "Q/K/V projections (B·S>2880)" gives example `[2048, H]` at B=4. These are contradictory: the B·S>2880 case does not apply when B=1, S=2048 (since 2048 ≤ 2880), so the table column heading is wrong for this row. Additionally, the shape `[2048, H]` is the shape for B=1 (2048 rows), but the size shown, `117.9 MB`, corresponds to B=4 (4 × 2048 × 7168 × 2 = 117,440,512 bytes ≈ 117.4 MB decimal). The shape and size within the row are internally inconsistent with each other: `[2048, H]` × BF16 = 29.4 MB (not 117.9 MB), while `[8192, H]` × BF16 = 117.4 MB (not 117.9 MB). The row needs to pick a consistent example configuration. For B=4, S=2048 (the intended illustration): shape = `[8192, H]`, size = 117.4 MB. The size 117.9 MB does not match any consistent calculation and should be corrected to 117.4 MB (or 117.5 MB to match the activation table's rounding at line 27). The table column note "Shape (B=1, S=2048)" should be updated to reflect that this row is illustrating a different configuration (e.g., B=4, S=2048).

8. **`decode_memory_strategy.md`, lines 196–209 — dead variable `hidden_state_per_core` (line 204) implements the wrong formula.**

   The function `sharded_per_core` (lines 196–200) takes `total_elements` as the row-dimension element count, applies tile alignment (rounding to nearest multiple of 32), and returns `rows_per_core * bytes_per_element`. When called as `sharded_per_core(B, n_cores) * H` (line 204) with B=32, n_cores=80: `rows_per_core = ceil(32/80/32) × 32 = ceil(0.0125) × 32 = 32`, return = 32 × 2.0 = 64, then ×H = 64 × 7168 = 458,752 bytes = 448 KB. This is not the 14 KB per-core figure the comment on line 205 describes. The variable `hidden_state_per_core` is never used in the final sum (which correctly uses `hidden_state_per_core_bytes` from line 207), so the printed output values are correct. However, the computed-but-unused variable at line 204 returns an incorrect value and will mislead any reader who traces through the code. Either the variable should be removed, or the `sharded_per_core` call should be replaced with the correct formula already at line 207 (`math.ceil(B / n_cores) * H * int(bytes_per_element)`).

9. **`index.md`, summary table line 56 — "~1.1 MB/core total at limit" does not correspond to the limit B·S = 2,880.**

   The annotation reads "Q+K+V fit simultaneously (~1.1 MB/core total at limit)." At the actual threshold limit B·S = 2,880: `3 × ceil(2880/80) × 7168 × 2 = 3 × 36 × 14,336 = 1,548,288 bytes ≈ 1.48 MB/core`. The value ~1.1 MB/core corresponds instead to B=1, S=2048 (3 × 26 × 14,336 = 1,118,208 bytes ≈ 1.07 MB ≈ 1.1 MB), which is a point well below the limit. The phrase "at limit" should either be replaced with "at B=1, S=2048" (to accurately identify the configuration) or with the correct at-limit value of ~1.48 MB/core.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 3 Fixes Applied

7. `prefill_memory_strategy.md` summary table — Fixed B·S>2880 row: shape `[2048, H] at B=4` → `[8192, H] (B=4, S=2048)` (correct: B=4,S=2048 gives B·S=8192 rows); size `117.9 MB` → `117.4 MB` (correct: 8192 × 7168 × 2 = 117,440,512 bytes = 117.4 MB decimal).

8. `decode_memory_strategy.md` — Removed dead variable `hidden_state_per_core` (which called `sharded_per_core(B, n_cores) * H` and returned an incorrect 448 KB) since it was never used in the printed output. Kept the correct `hidden_state_per_core_bytes = math.ceil(B / n_cores) * H * bytes_per_element`. Replaced the misleading intermediate variable and stale comment with a single correct explanatory comment.

9. `index.md` summary table — Fixed annotation for Q/K/V Prefill row: changed "~1.1 MB/core total at limit" to "~1.1 MB/core at B=1,S=2048; ~1.48 MB/core at threshold B·S=2,880" (at-limit correct value: 3 × 36 × 14,336 = 1,548,288 bytes ≈ 1.48 MB/core).

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 4

**Verification of Pass 3 fixes:**

Fix 7 (`prefill_memory_strategy.md` summary table, Q/K/V B·S>2880 row): shape now reads `[8192, H] (B=4, S=2048)`, size reads `117.4 MB`. Correct. Confirmed.

Fix 8 (`decode_memory_strategy.md` dead variable removal): the `hidden_state_per_core` variable that called `sharded_per_core(B, n_cores) * H` is gone. The correct `hidden_state_per_core_bytes = math.ceil(B / n_cores) * H * int(bytes_per_element)` remains and is used in the final sum. Note: the `sharded_per_core` function definition (lines 196–200) is still present but is now dead code — it is defined but never called anywhere in the block. This is not a numeric error (the printed outputs are all correct) but is noted for completeness.

Fix 9 (`index.md` summary table Q/K/V row annotation): now reads "~1.1 MB/core at B=1,S=2048; ~1.48 MB/core at threshold B·S=2,880". Correct. Confirmed.

**Remaining issues found:**

10. **`prefill_memory_strategy.md`, activation table lines 26–27 — wrong sizes for B=4, S=2048 and B=1, S=8192.**

    Both rows show `117.5 MB`. The correct calculation is `4 × 2048 × 7168 × 2 = 117,440,512 bytes = 117.44 MB ≈ 117.4 MB` (decimal), as established by the key fact. The same number of elements applies to B=1, S=8192: `8192 × 7168 × 2 = 117,440,512 bytes = 117.4 MB`. Both rows should read `117.4 MB`, not `117.5 MB`.

11. **`prefill_memory_strategy.md`, all-to-all buffer table line 134 — wrong size for B=4, S=2048.**

    The table row reads `256 × 32 × 7168 × 2 = 117.5 MB`. The expression evaluates to `117,440,512 bytes = 117.4 MB` (decimal). This contradicts the key fact (117.4 MB) and is inconsistent with the corrected summary table (line 322) which already shows 117.4 MB for the same quantity. The correct value is `117.4 MB`.

12. **`prefill_memory_strategy.md`, prose line 137 — wrong size repeats 117.5 MB.**

    The sentence "The B=4, S=2048 case alone requires 117.5 MB per direction" must be corrected to `117.4 MB` to match the key fact and the corrected summary table.

13. **`prefill_memory_strategy.md`, code comment line 158 — wrong annotated output for B=4.**

    The comment `# B= 4 S=2048: C=  256, buffer=117.5 MB/device/direction` contains two errors. First, the value is wrong: 117.4 MB (decimal), not 117.5 MB. Second, the code computes `vol_mb = alltoall_buffer_bytes(B, S) / (1024 ** 2)`, which divides by 1,048,576 (binary MiB), not 1,000,000 (decimal MB). For B=4: `117,440,512 / 1,048,576 = 112.0 MiB`, not 117.4 MB or 117.5 MB. The same binary/decimal mismatch applies to B=1 (code produces 28.0 MiB, comment says 29.4 MB) and B=32 (code produces 896.0 MiB, comment says 939.5 MB). The code's divisor `(1024 ** 2)` must be changed to `1_000_000` to make the computed output match the decimal MB values shown in the comments, or all three comment lines must be updated to reflect MiB values (28.0, 112.0, 896.0).

14. **`wormhole_memory_hierarchy.md`, HEIGHT_SHARDED formula line 153 — missing ceiling operator.**

    The displayed formula is:
    $$\text{per-core shard shape} = \left[\frac{M_t}{n_{\text{cores}}}, N_t\right] \times \text{bytes\_per\_tile}$$
    This uses plain (floor) division. The entire chapter's key principle is that rows must be whole tile-rows — they cannot be fractionally split — requiring `ceil`. With M_t=16, n_cores=80, plain division gives 0.2 (floors to 0), while the correct value is ceil(16/80) = 1. The formula contradicts the function immediately below it (which correctly uses `math.ceil`) and contradicts all the worked examples. The formula must be corrected to use `\lceil M_t / n_{\text{cores}} \rceil`. The same issue exists in the WIDTH_SHARDED formula on line 161 (`N_t / n_cores` should be `\lceil N_t / n_{\text{cores}} \rceil`) and in the BLOCK_SHARDED formulas on line 169.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 4 Fixes Applied

10+11+12. `prefill_memory_strategy.md` — Changed all occurrences of `117.5 MB` to `117.4 MB` (correct: 256 × 32 × 7168 × 2 = 117,440,512 bytes / 1,000,000 = 117.44 MB; applies to activation table rows for B=4,S=2048 and B=1,S=8192; all-to-all buffer table; and the prose sentence "The B=4, S=2048 case alone requires 117.5 MB per direction").

13. `prefill_memory_strategy.md` code — Changed divisor in `alltoall_buffer_bytes` output loop from `(1024 ** 2)` (binary MiB, gives 112.0) to `1_000_000` (decimal MB, gives 117.4), so the computed output now matches the decimal MB values in the comments and table.

14. `wormhole_memory_hierarchy.md` — Added `\lceil ... \rceil` ceiling operators to all three shard layout formulas: HEIGHT_SHARDED `[ceil(M_t/n_cores), N_t]`, WIDTH_SHARDED `[M_t, ceil(N_t/n_cores)]`, BLOCK_SHARDED `[ceil(M_t/r), ceil(N_t/c)]`. Without ceiling, plain division gives fractional tile-rows which is physically impossible.

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 5

**Verification of Pass 4 fixes:**

Fix 10+11+12 (`prefill_memory_strategy.md`, all `117.5 MB` → `117.4 MB`): Partially applied. The activation table row for B=4, S=2048 (line 26) and B=1, S=8192 (line 27) now correctly read `117.4 MB`. The prose sentence on line 137 now correctly reads `117.4 MB`. The code output comment on line 158 now correctly reads `117.4 MB`. However, the all-to-all buffer sizing table on line 134 was missed — it still reads `256 × 32 × 7168 × 2 = 117.5 MB`. This is a remaining error (see issue 15 below).

Fix 13 (`prefill_memory_strategy.md` code divisor): `1_000_000` is present on line 152. The code output comments now read 29.4, 117.4, and 939.5 MB respectively — consistent with decimal MB. Confirmed.

Fix 14 (`wormhole_memory_hierarchy.md` ceiling operators): All three shard layout formulas now use `\lceil ... \rceil`: HEIGHT_SHARDED line 153, WIDTH_SHARDED line 161, BLOCK_SHARDED line 169. Confirmed.

**Remaining issues found:**

15. **`prefill_memory_strategy.md`, all-to-all buffer sizing table line 134 — `117.5 MB` was not corrected.**

    The table row still reads: `$256 \times 32 \times 7168 \times 2 = 117.5\,\text{MB}$`. The correct decimal MB value is 256 × 32 × 7168 × 2 = 117,440,512 bytes / 1,000,000 = 117.4 MB. This is the one occurrence missed in the Pass 4 fix sweep. Every other instance of this value in the same file (prose line 137, code comment line 158, summary table line 337) has been corrected to 117.4 MB. This table cell must be changed from `117.5\,\text{MB}` to `117.4\,\text{MB}` to achieve internal consistency.

16. **`prefill_memory_strategy.md`, activation tensor table line 28 — `940 MB` is inconsistent with the established value `939.5 MB`.**

    The activation size table row for B=32, S=2048 reads `940 MB`. The correct calculation is 32 × 2048 × 7168 × 2 = 939,524,096 bytes / 1,000,000 = 939.5 MB. The value 939.5 MB appears correctly in the all-to-all buffer sizing table (line 135), the code output comment (line 159), and the summary table (line 338). The `940 MB` in the activation table is an inconsistent rounding that contradicts the established key fact and three other occurrences in the same file. It must be changed to `939.5 MB`.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 5 Fixes Applied

15. `prefill_memory_strategy.md` all-to-all buffer table — Fixed missed occurrence: `$256 \times 32 \times 7168 \times 2 = 117.5\,\text{MB}$` → `117.4\,\text{MB}` (this used LaTeX `\,\text{MB}` format so was not matched by the earlier replace_all for `117.5 MB`).

16. `prefill_memory_strategy.md` activation table — Fixed `940 MB` → `939.5 MB` for B=32, S=2048 row (correct: 32 × 2048 × 7168 × 2 = 939,524,096 bytes / 1,000,000 = 939.5 MB). Updated percentage 783% → 782.9%.

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 6

**Verification of Pass 5 fixes:**

Fix 15 (`prefill_memory_strategy.md` all-to-all buffer table, line 134): now reads `$256 \times 32 \times 7168 \times 2 = 117.4\,\text{MB}$`. Correct. Confirmed.

Fix 16 (`prefill_memory_strategy.md` activation table, line 28): now reads `939.5 MB` and `782.9%`. Correct. Confirmed.

**Remaining issues found:**

17. **`decode_memory_strategy.md`, line 28 — KV cache calculation result is wrong by a factor of 2.**

    The formula shown is: $64 \times 2 \times 32 \times 4096 \times 8 \times 128 \times 2 = 17.2\,\text{GB}$

    Computing that product: 64 × 2 = 128; × 32 = 4,096; × 4,096 = 16,777,216; × 8 = 134,217,728; × 128 = 17,179,869,184; × 2 = 34,359,738,368 bytes = 34.4 GB (decimal). The result stated, 17.2 GB, equals 17,179,869,184 bytes — which is the product of all factors except the final `× 2 bytes`. In other words, the last factor of 2 (for BF16 byte width) was not applied in the arithmetic. The correct result is **34.4 GB**, not 17.2 GB. The summary table on line 298 compounds this by stating `~17 GB`; it should read `~34 GB`. Neither value changes the qualitative conclusion (far exceeds L1), but the numbers are wrong.

18. **`prefill_memory_strategy.md`, lines 83–102 — `attention_memory_config` function checks single-tensor L1 fit but the stated threshold (B·S ≤ 2,880) is derived from triple-tensor (Q+K+V) fit.**

    The function computes `shard_bytes = rows_per_core * H_Q * 2` (one tensor only) and checks `shard_bytes <= L1_PER_CORE * 0.8`. This passes the L1 check when a single tensor uses ≤ 80% of L1 per core. Solving: `ceil(B·S/80) * 14,336 <= 1,258,496` gives `ceil(B·S/80) <= 87.8`, i.e., the function returns L1 for all B·S ≤ 6,960.

    However, the chapter's stated threshold is B·S ≤ 2,880, derived from Q+K+V all fitting simultaneously: `3 × ceil(B·S/80) × 14,336 ≤ 1,572,864`. For any B·S in the range (2,880, 6,960], the function returns L1, but three tensors of that size will together exceed 1.5 MB per core. For example, at B·S = 3,072: single shard = 39 × 14,336 = 559,104 bytes (≤ 1,258,496, so function returns L1), but Q+K+V = 3 × 559,104 = 1,677,312 bytes > 1,572,864 — exceeds L1.

    The function must check `3 * shard_bytes <= L1_PER_CORE` (for the triple-tensor constraint, no safety margin needed since the text threshold is already tight) or equivalently replace the condition with `rows_per_core * H_Q * 2 * 3 <= L1_PER_CORE`. The comment on line 106 noting "B=4, S=2048 → DRAM" is coincidentally correct because 8,192 > 6,960, but the function's logic does not enforce the chapter's documented 2,880 cutoff.

---

# Agent A Fix — Chapter 4: Memory Configuration — Pass 6 Fixes Applied

---

# Agent A Fix — Chapter 4: Memory Configuration — Compression Pass 1 Fixes Applied

C1. `prefill_memory_strategy.md` Summary section — Removed duplicate all-to-all buffer sizing formula block and 3-row table (B=1/4/32, C values, MB sizes) that was an exact repeat of the content in the "All-to-All Dispatch and Combine Buffers" section above. Replaced with a single reference sentence pointing to the authoritative location.

C2. `wormhole_memory_hierarchy.md` DRAM Interleaving section — Removed the 10-line Python code block showing `ttnn.MemoryConfig(INTERLEAVED, DRAM)` equivalence with `ttnn.DRAM_MEMORY_CONFIG`. This block is fully covered in `memory_config_api.md`. Replaced with a one-sentence prose pointer.

---

17. `decode_memory_strategy.md` line 28 — Fixed KV cache arithmetic: `17.2 GB` → `34.4 GB` (correct: 64 × 2 × 32 × 4096 × 8 × 128 × 2 = 34,359,738,368 bytes = 34.4 GB; the previous answer 17.2 GB omitted the final ×2 BF16 factor). Also updated summary table line 298: `~17 GB` → `~34 GB`.

18. `prefill_memory_strategy.md` line 88 — Fixed `attention_memory_config` function condition from `shard_bytes <= L1_PER_CORE * 0.8` (single-tensor, 80% margin, allows up to B·S ≈ 6,960) to `3 * shard_bytes <= L1_PER_CORE` (triple-tensor Q+K+V constraint, matches the documented B·S ≤ 2,880 threshold). Updated comment from "80% safety margin" to "Q, K, and V must all fit simultaneously".

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 7

**Verification of Pass 6 fixes:**

Fix 17 (`decode_memory_strategy.md` KV cache result and summary table): The formula result now reads `34.4 GB` (correct: 64 × 2 × 32 × 4096 × 8 × 128 × 2 = 34,359,738,368 bytes = 34.4 GB) and the summary table row now reads `~34 GB`. Confirmed.

Fix 18 (`prefill_memory_strategy.md` `attention_memory_config` function condition): The condition is now `if 3 * shard_bytes <= L1_PER_CORE:` with comment "Q, K, and V must all fit simultaneously". Boundary check: at B·S=2880, rows_per_core=36, shard_bytes=516,096, 3×516,096=1,548,288 ≤ 1,572,864 — correctly returns L1. At B·S=2881, rows_per_core=37, shard_bytes=530,432, 3×530,432=1,591,296 > 1,572,864 — correctly returns DRAM. The function now enforces the documented B·S ≤ 2,880 threshold. Confirmed.

No remaining technical errors found.

No feedback — chapter approved

---

# Agent B Review — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 8

**Verification of Compression Pass 1 fixes C1 and C2:**

C1 (`prefill_memory_strategy.md` Summary section — duplicate all-to-all formula and table): The Summary section no longer contains the duplicate `V_prefill` formula block or the three-row B=1/4/32 table. Line 328 now reads: "For reference, see the all-to-all buffer sizing formula and table in the All-to-All Dispatch and Combine Buffers section above." This is the correct one-sentence reference. C1 confirmed applied.

C2 (`wormhole_memory_hierarchy.md` DRAM Interleaving section — 10-line Python code block): The Python code block constructing `ttnn.MemoryConfig(INTERLEAVED, DRAM)` and asserting equivalence with `ttnn.DRAM_MEMORY_CONFIG` is gone. Lines 103–104 now read: "TTNN exposes this as `ttnn.DRAM_MEMORY_CONFIG`; the constructor equivalent and predefined constant are shown in `memory_config_api.md`." This is the correct prose pointer. C2 confirmed applied.

**Full correctness scan:**

All ground-truth values verified against the five chapter files:

- 80 cores, 1.5 MB L1/core (`wormhole_memory_hierarchy.md` line 18): correct.
- H=7168 throughout all files: correct.
- BF16 tile = 2048 bytes (`wormhole_memory_hierarchy.md` line 51): correct.
- KV cache formula result = 34.4 GB (`decode_memory_strategy.md` line 28, summary table line 298 ~34 GB): correct.
- `attention_memory_config` condition `3 * shard_bytes <= L1_PER_CORE` (`prefill_memory_strategy.md` line 88): correct; enforces the documented B·S ≤ 2,880 threshold.
- Divisor `1_000_000` for decimal MB (`prefill_memory_strategy.md` line 152): correct.
- Prefill all-to-all buffer sizes B=1→29.4 MB, B=4→117.4 MB, B=32→939.5 MB (table lines 133–135, code comments lines 156–159, summary table lines 324–325): all consistent and correct.
- Activation tensor sizes B=4,S=2048→117.4 MB; B=32,S=2048→939.5 MB (activation table lines 26–28): correct.
- Ceiling operators in all three shard layout formulas (`wormhole_memory_hierarchy.md` HEIGHT_SHARDED line 142, WIDTH_SHARDED line 150, BLOCK_SHARDED line 158): correct.
- `per_core_bytes_height_sharded(M=512, N=7168, n_cores=80)` annotation = 448.0 KB (`wormhole_memory_hierarchy.md` lines 191–192): correct (1 tile-row × 224 tile-cols × 2048 bytes/tile = 458,752 bytes = 448 KB).
- Index summary table per-core values (14.0 KB/core for hidden state and dispatch buffer, lines 52 and 54): correct.
- Index Q/K/V annotation "~1.1 MB/core at B=1,S=2048; ~1.48 MB/core at threshold B·S=2,880" (line 56): correct.
- `decode_memory_strategy.md` worked example outputs (14.0 KB hidden, 0.5 KB routing, 14.0 KB dispatch, ~14.0 KB FFN, total ~42.5 KB, 2.8%): correct.
- No new errors introduced by the C1 or C2 compression edits.

No remaining technical errors found.

No feedback — chapter approved
