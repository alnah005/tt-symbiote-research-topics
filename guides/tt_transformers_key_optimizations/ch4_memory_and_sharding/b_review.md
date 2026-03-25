# B Review — Chapter 4: Memory and Sharding — Pass 1

## Issues Found

1. **FlashAttention buffer formula incorrect (tensix_memory_hierarchy.md)**

   The formula as written was:
   ```
   Q + K + V + O = 4 × Br × D × 2 bytes
   ```
   This is only correct when Br = Bc. Q and O have shape `Br × D`, but K and V have shape `Bc × D`. The general formula must be:
   ```
   Q + K + V + O = (2 × Br + 2 × Bc) × D × 2 bytes
   ```
   The formula error also made the table row for Br=128, Bc=64 arithmetically inconsistent: the expression shown, `4 × 128 × 128 × 2`, evaluates to 128 KB, not the 96 KB the table claimed. The 96 KB figure is correct; the formula used to produce it was wrong.

   - Br=64, Bc=64: `(128+128)×128×2 = 64 KB` — correct
   - Br=128, Bc=64: `(256+128)×128×2 = 96 KB` — correct
   - Br=128, Bc=128: `(512)×128×2 = 128 KB` — correct

## Verification of Confirmed Facts

| Fact | Source location | Status |
|---|---|---|
| L1 SRAM = 120 KB per Tensix core | tensix_memory_hierarchy.md line 9 | CORRECT |
| `packer_l1_acc` is throughput-only, not a correctness requirement | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| `packer_l1_acc` only valid when output buffer is in L1 | tensix_memory_hierarchy.md line 23 | CORRECT |
| BF16 = 2 bytes per element | tensix_memory_hierarchy.md line 27 | CORRECT |
| FlashAttention buffer formula | tensix_memory_hierarchy.md line 30 | INCORRECT (fixed) |
| Table value 96 KB for Br=128, Bc=64 | tensix_memory_hierarchy.md line 36 | CORRECT (value was right; formula was wrong) |
| Table value 128 KB for Br=128, Bc=128 | tensix_memory_hierarchy.md line 37 | CORRECT |
| KV cache shape: `[1, n_kv_heads, n_blocks × block_size, head_dim]` | tensix_memory_hierarchy.md line 56 | CORRECT |
| Decode activation size M=32, K=4096, BF16 = ~256 KB | sharding_patterns.md line 46 | CORRECT |
| N300: 2-chip, on-board Ethernet | tensix_memory_hierarchy.md / index.md | CORRECT |
| T3K: 8-chip Ethernet ring | tensix_memory_hierarchy.md / index.md | CORRECT |
| `MatmulMultiCoreReuseMultiCast1DProgramConfig` for height-sharded activation | sharding_patterns.md line 13 | CORRECT |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for DRAM-sharded decode | sharding_patterns.md line 50 | CORRECT |
| `MatmulMultiCoreReuseMultiCastProgramConfig` for NoC multicast (one DRAM read, broadcast to all) | tensix_memory_hierarchy.md line 77 | CORRECT |

VERDICT: CHANGES NEEDED

## Agent A Change Log — Pass 1 Fix

**File modified:** `tensix_memory_hierarchy.md`

**Change 1 — FlashAttention buffer formula (lines 27–37)**

- Replaced incorrect formula `Q + K + V + O = 4 × Br × D × 2 bytes` with the correct general formula `Q + K + V + O = (2 × Br + 2 × Bc) × D × 2 bytes`.
- Added a clarifying sentence noting that Q and O have shape `Br × D` while K and V have shape `Bc × D`.
- Updated all three table rows to use the expanded form `(2×Br + 2×Bc) × D × 2` so the arithmetic is transparent and correct for the Br ≠ Bc case (Br=128, Bc=64).
- The table values (64 KB, 96 KB, 128 KB) were already correct and were not changed.

No changes were required to `index.md` or `sharding_patterns.md`.

---

# B Review — Chapter 4: Memory and Sharding — Pass 2

## Issues Found

1. **Row-parallel linear collective operation wrong (sharding_patterns.md)**

   The 1D Ring-Sharded section stated that row-parallel linear layers aggregate partial results via `all-reduce` / `ttnn.all_reduce` (lines 64 and 68, and Key Takeaways line 106). This is incorrect. Per the key facts, row-parallel linear output requires `ttnn.reduce_scatter`, not `ttnn.all_reduce`. `ttnn.all_reduce` is the operation for vocab-parallel LM head. The same section in `tensix_memory_hierarchy.md` (lines 92–93) correctly distinguishes the two: `ttnn.all_gather` after column-parallel, `ttnn.reduce_scatter` after row-parallel.

   Three locations fixed in `sharding_patterns.md`:
   - Line 64: "the all-reduce (implemented via `ttnn.all_reduce`...)" changed to "a reduce-scatter (implemented via `ttnn.reduce_scatter`...)"
   - Line 68: "the all-reduce combines results" changed to "`ttnn.reduce_scatter` combines and distributes the results"
   - Key Takeaways line 106: "aggregates via all-reduce" changed to "aggregates via reduce-scatter (`ttnn.reduce_scatter`)"

## Verification of Confirmed Facts

| Fact | Source location | Status |
|---|---|---|
| L1 SRAM = 120 KB per Tensix core | tensix_memory_hierarchy.md line 9 | CORRECT |
| `packer_l1_acc` is throughput-only, not a correctness requirement | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| `packer_l1_acc` only valid when output buffer is in L1 | tensix_memory_hierarchy.md line 23 | CORRECT |
| BF16 = 2 bytes per element | tensix_memory_hierarchy.md line 27 | CORRECT |
| FlashAttention buffer formula `(2×Br + 2×Bc) × D × 2` | tensix_memory_hierarchy.md line 30 | CORRECT (fixed in Pass 1) |
| KV cache shape: `[1, n_kv_heads, n_blocks × block_size, head_dim]` | tensix_memory_hierarchy.md line 56 | CORRECT |
| Decode activation size M=32, K=4096, BF16 = ~256 KB | sharding_patterns.md line 46 | CORRECT |
| N300: 2-chip, on-board Ethernet | tensix_memory_hierarchy.md line 87 / index.md line 15 | CORRECT |
| T3K: 8-chip Ethernet ring | tensix_memory_hierarchy.md line 88 / index.md line 15 | CORRECT |
| `ttnn.all_gather` after column-parallel projections | tensix_memory_hierarchy.md line 92 | CORRECT |
| `ttnn.reduce_scatter` after row-parallel projections | tensix_memory_hierarchy.md line 93 | CORRECT |
| `ttnn.all_reduce` for All-Reduce TP variants (not row-parallel) | tensix_memory_hierarchy.md line 94 | CORRECT |
| Row-parallel linear uses `ttnn.reduce_scatter` (not `ttnn.all_reduce`) | sharding_patterns.md lines 64, 68, 106 | INCORRECT (fixed) |
| `MatmulMultiCoreReuseProgramConfig` for independent per-core DRAM fetch / 2D | sharding_patterns.md line 30 / tensix_memory_hierarchy.md line 77 | CORRECT |
| `MatmulMultiCoreReuseMultiCastProgramConfig` for 2D NoC multicast | tensix_memory_hierarchy.md line 77 | CORRECT |
| `MatmulMultiCoreReuseMultiCast1DProgramConfig` for height-sharded / 1D | sharding_patterns.md line 13 | CORRECT |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for DRAM-sharded | sharding_patterns.md line 50 | CORRECT |

VERDICT: CHANGES NEEDED

## Agent B Change Log — Pass 2 Fix

**File modified:** `sharding_patterns.md`

**Change 1 — Row-parallel collective operation: `ttnn.all_reduce` → `ttnn.reduce_scatter` (three locations)**

- 1D Ring-Sharded "What it enables" paragraph: replaced "the all-reduce (implemented via `ttnn.all_reduce`...)" with "a reduce-scatter (implemented via `ttnn.reduce_scatter`...)".
- 1D Ring-Sharded "Typical layers" paragraph: replaced "the all-reduce combines results" with "`ttnn.reduce_scatter` combines and distributes the results"; also removed the stale reference to a "fused All-Reduce variant" in the preceding sentence.
- Key Takeaways bullet for 1D ring-sharded: replaced "aggregates via all-reduce" with "aggregates via reduce-scatter (`ttnn.reduce_scatter`)".

No changes were required to `index.md` or `tensix_memory_hierarchy.md`.

---

# B Review — Chapter 4: Memory and Sharding — Pass 3

## Issues Found

1. **Summary table still says "all-reduce" for 1D ring-sharded (sharding_patterns.md line 97)**

   The Summary Table's Config class column for the 1D ring-sharded row read:
   ```
   [INFERRED] K-parallel matmul + all-reduce
   ```
   Pass 2 corrected three locations in the prose body of `sharding_patterns.md` where `ttnn.all_reduce` was wrongly used for row-parallel / 1D ring-sharded reduce, but missed this fourth occurrence in the summary table. Per key facts, row-parallel linear output uses `ttnn.reduce_scatter`, not `ttnn.all_reduce` (which is reserved for vocab-parallel LM head). Fixed to:
   ```
   [INFERRED] K-parallel matmul + reduce-scatter
   ```

## Verification of Confirmed Facts

| Fact | Source location | Status |
|---|---|---|
| L1 SRAM = 120 KB per Tensix core | tensix_memory_hierarchy.md line 9 | CORRECT |
| `packer_l1_acc` is throughput-only, not a correctness requirement | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| `packer_l1_acc` only valid when output buffer is in L1 | tensix_memory_hierarchy.md line 23 | CORRECT |
| FlashAttention buffer formula `(2×Br + 2×Bc) × D × 2` | tensix_memory_hierarchy.md line 30 | CORRECT (fixed Pass 1) |
| Table values 64 KB / 96 KB / 128 KB for Br/Bc combinations | tensix_memory_hierarchy.md lines 35–37 | CORRECT |
| KV cache shape `[1, n_kv_heads, n_blocks × block_size, head_dim]` | tensix_memory_hierarchy.md line 56 | CORRECT |
| Decode activation size M=32, K=4096, BF16 = ~256 KB | sharding_patterns.md line 46 | CORRECT |
| N300: 2-chip, on-board Ethernet | tensix_memory_hierarchy.md line 87 | CORRECT |
| T3K: 8-chip Ethernet ring | tensix_memory_hierarchy.md line 88 | CORRECT |
| `ttnn.all_gather` after column-parallel projections | tensix_memory_hierarchy.md line 92 | CORRECT |
| `ttnn.reduce_scatter` after row-parallel projections | tensix_memory_hierarchy.md line 93 | CORRECT |
| `ttnn.all_reduce` for All-Reduce TP variants (not row-parallel) | tensix_memory_hierarchy.md line 94 | CORRECT |
| `MatmulMultiCoreReuseMultiCast1DProgramConfig` for 1D / height-sharded | sharding_patterns.md line 13 | CORRECT |
| `MatmulMultiCoreReuseProgramConfig` for independent per-core DRAM fetch | sharding_patterns.md line 30 | CORRECT |
| `MatmulMultiCoreReuseMultiCastProgramConfig` for 2D NoC multicast | tensix_memory_hierarchy.md line 77 | CORRECT |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` for DRAM-sharded | sharding_patterns.md line 50 | CORRECT |
| 1D ring-sharded uses reduce-scatter (not all-reduce) — prose body | sharding_patterns.md lines 64, 68, 106 | CORRECT (fixed Pass 2) |
| 1D ring-sharded summary table uses reduce-scatter (not all-reduce) | sharding_patterns.md line 97 | INCORRECT (fixed Pass 3) |

VERDICT: CHANGES NEEDED

## Agent B Change Log — Pass 3 Fix

**File modified:** `sharding_patterns.md`

**Change 1 — Summary table 1D ring-sharded config description: "all-reduce" → "reduce-scatter" (line 97)**

- Summary Table Config class cell for 1D ring-sharded: replaced `[INFERRED] K-parallel matmul + all-reduce` with `[INFERRED] K-parallel matmul + reduce-scatter`.
- This was the only remaining occurrence of `all-reduce` in the row-parallel / 1D ring-sharded context, missed by Pass 2 which fixed the three prose-body occurrences.

No changes were required to `tensix_memory_hierarchy.md`.

---

# B Review — Chapter 4: Memory and Sharding — Pass 4

## Issues Found

None found.

## Verification of Confirmed Facts

| Fact | Source location | Status |
|---|---|---|
| L1 SRAM = 120 KB per Tensix core | tensix_memory_hierarchy.md line 9 | CORRECT |
| `packer_l1_acc` is throughput-only, not a correctness requirement | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| `packer_l1_acc` only valid when output buffer is in L1 | tensix_memory_hierarchy.md line 23 | CORRECT |
| FlashAttention buffer formula `(2×Br + 2×Bc) × D × 2` | tensix_memory_hierarchy.md line 30 | CORRECT (fixed Pass 1) |
| KV cache shape `[1, n_kv_heads, n_blocks × block_size, head_dim]` | tensix_memory_hierarchy.md line 56 | CORRECT |
| `ttnn.all_gather` after column-parallel projections | tensix_memory_hierarchy.md line 92 | CORRECT |
| `ttnn.reduce_scatter` after row-parallel projections | tensix_memory_hierarchy.md line 93 | CORRECT |
| `ttnn.all_reduce` for All-Reduce TP variants only (not row-parallel) | tensix_memory_hierarchy.md line 94 | CORRECT |
| `MatmulMultiCoreReuseMultiCastProgramConfig` = 2D NoC multicast | tensix_memory_hierarchy.md line 77 | CORRECT |
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` = DRAM-sharded bank parallelism | tensix_memory_hierarchy.md line 73 | CORRECT |
| `MatmulMultiCoreReuseMultiCast1DProgramConfig` for height-sharded / 1D ring | sharding_patterns.md line 13 | CORRECT |
| `MatmulMultiCoreReuseProgramConfig` for independent per-core DRAM fetch (block-sharded) | sharding_patterns.md line 30 | CORRECT |
| 1D ring-sharded uses `ttnn.reduce_scatter` — all prose and summary table locations | sharding_patterns.md lines 64, 68, 97, 106 | CORRECT (fixed Pass 2 and Pass 3) |

VERDICT: APPROVED

---

# B Review — Chapter 4: Memory and Sharding — Pass 5

## Issues Found

None found.

## Verification of Confirmed Facts

| Fact | Source location | Status |
|---|---|---|
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`: dedicated per-core DRAM bank ownership; core k owns its bank; all banks active simultaneously; aggregate bandwidth scales linearly | tensix_memory_hierarchy.md line 73 (forwarding sentence); sharding_patterns.md lines 44–46 | CORRECT |
| Condensed section accurately describes DRAM-sharded mechanism even in brief form | tensix_memory_hierarchy.md lines 71–73 | CORRECT — forwarding sentence preserves "dedicated per-core DRAM bank ownership", "maximum aggregate DRAM bandwidth", and "no inter-bank contention"; sharding_patterns.md carries the full layout detail |
| `packer_l1_acc`: throughput only; requires L1 output buffer | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| L1 = 120 KB per core | tensix_memory_hierarchy.md line 9 | CORRECT |
| FlashAttention buffers: `(2×Br + 2×Bc) × D × 2 bytes` | tensix_memory_hierarchy.md line 30 | CORRECT (fixed Pass 1) |
| row-parallel: `ttnn.reduce_scatter`; vocab-parallel: `ttnn.all_reduce`; column-parallel: `ttnn.all_gather` | tensix_memory_hierarchy.md lines 92–94 | CORRECT |
| 1D ring-sharded K-reduction in row-parallel context uses `ttnn.reduce_scatter` | sharding_patterns.md lines 64, 68, 97, 106 | CORRECT (fixed Pass 2 and Pass 3) |

VERDICT: APPROVED

---

# B Review — Chapter 4: Memory and Sharding — Pass 6

## Issues Found

None found.

## Verification of Confirmed Facts

**C Pass 2 change verified:** The DRAM-Sharded Key Takeaways bullet in `tensix_memory_hierarchy.md` has been replaced with a forward-pointer sentence (line 105): "DRAM-sharded matmul and NoC multicast are described in [sharding_patterns.md](sharding_patterns.md#width-sharded--dram-sharded----weight-tensors-for-decode) and [the Multicast subsection above](#multicast-one-dram-read-many-cores) respectively; together they maximise aggregate DRAM bandwidth for decode and minimise redundant DRAM reads for prefill."

No information gap was created. The forward pointer correctly routes to the full DRAM-sharded mechanism in `sharding_patterns.md`, which carries the complete detail: dedicated per-core DRAM bank ownership, all banks active simultaneously, and maximum aggregate bandwidth (lines 44–46). The condensed subsection in `tensix_memory_hierarchy.md` (lines 71–73) also retains its own brief summary sentence covering the same core facts.

| Fact | Source location | Status |
|---|---|---|
| `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`: dedicated per-core DRAM bank, all banks active simultaneously, maximum aggregate bandwidth | sharding_patterns.md lines 44–46; tensix_memory_hierarchy.md line 73 | CORRECT |
| Forward-pointer bullet in Key Takeaways accurately characterises the effect ("maximise aggregate DRAM bandwidth for decode") | tensix_memory_hierarchy.md line 105 | CORRECT |
| No DRAM-sharded facts formerly in the old bullet are now missing from the combined file set | tensix_memory_hierarchy.md line 73 + sharding_patterns.md lines 44–50 | CORRECT — no gap |
| `packer_l1_acc`: throughput only; requires L1 output buffer | tensix_memory_hierarchy.md lines 21–23 | CORRECT |
| FlashAttention buffers: `(2×Br + 2×Bc) × D × 2 bytes` | tensix_memory_hierarchy.md line 30 | CORRECT |

VERDICT: APPROVED
