# Compression Analysis — Chapter 4: Memory and Sharding
Reviewer: Agent C (compression reviewer)
Date: 2026-03-24
Files reviewed:
- index.md
- tensix_memory_hierarchy.md
- sharding_patterns.md

## Summary

The three files are largely non-redundant. `index.md` serves a pure orientation role; `tensix_memory_hierarchy.md` and `sharding_patterns.md` cover distinct concerns with appropriate brief cross-references between them. One near-verbatim duplication was found: the DRAM-sharded bank parallelism mechanism (weight columns distributed contiguously across banks, core k owns its bank, all banks active simultaneously, aggregate bandwidth scales linearly) is stated in full in both `tensix_memory_hierarchy.md` (NoC section) and `sharding_patterns.md` (DRAM-sharded section). The canonical home for this mechanism is `sharding_patterns.md`; the NoC section in `tensix_memory_hierarchy.md` should be trimmed to a forward pointer rather than re-stating the full explanation. No other cross-file verbatim or near-verbatim duplication was found.

## CRUCIAL Suggestions

1. **`tensix_memory_hierarchy.md` — NoC / DRAM-Sharded Matmul subsection (lines 72–73):**
   The paragraph beginning "When a matmul uses `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`..." through "...the maximum possible DRAM throughput for this access pattern" restates nearly verbatim the mechanism explained in `sharding_patterns.md` lines 44–45 ("Weight columns are distributed contiguously across DRAM banks — not interleaved round-robin... Core k owns the column tiles... All banks are active simultaneously; aggregate bandwidth scales linearly with bank count."). The full mechanism belongs in `sharding_patterns.md`, which is its natural home.

   **Cut from `tensix_memory_hierarchy.md`:** The body of the "DRAM-Sharded Matmul: Bank Parallelism" subsection — the sentence describing the distribution and bank ownership mechanism and the aggregate bandwidth claim. Replace with a one-sentence pointer to `sharding_patterns.md`.

## MINOR Suggestions

None found.

## Load-Bearing Evidence

The following content must NOT be cut:

- `tensix_memory_hierarchy.md` — the `packer_l1_acc` subsection (lines 17–23): full mechanism, validity condition, and throughput vs. correctness distinction. Not duplicated anywhere else.
- `tensix_memory_hierarchy.md` — L1 Sizing Constraints table and arithmetic (lines 25–39): the 120 KB ceiling calculation for Br/Bc combinations. Not duplicated elsewhere.
- `tensix_memory_hierarchy.md` — KV cache DRAM layout shape (lines 53–59): the `[1, n_kv_heads, n_blocks × block_size, head_dim]` layout with leading-1 explanation. Not duplicated elsewhere.
- `tensix_memory_hierarchy.md` — Ethernet CCL table and three-collective descriptions (lines 84–94): `all_gather`, `reduce_scatter`, `all_reduce` with functional descriptions. `sharding_patterns.md` only makes brief mentions.
- `sharding_patterns.md` — "Keeping Activations in L1 Across Layers" section (lines 72–86): the decode MLP block walkthrough steps 1–4. Not duplicated elsewhere.
- `index.md` — Key Concepts table: serves as orientation, not a restatement of detail content.

## VERDICT
- Crucial updates: yes (edit applied to `tensix_memory_hierarchy.md`)

# Compression Pass 2 — Chapter 4: Memory and Sharding
Date: 2026-03-24

## CRUCIAL Suggestions

1. **`tensix_memory_hierarchy.md` — Key Takeaways, DRAM-sharded bullet (line 105 before edit):** After Pass 1 replaced the "DRAM-Sharded Matmul: Bank Parallelism" subsection body with a forwarding sentence, the Key Takeaways bullet in `tensix_memory_hierarchy.md` continued to state the full mechanism ("placing each core's weight columns in a dedicated DRAM bank, eliminating inter-bank contention") — near-verbatim with `sharding_patterns.md` Key Takeaways line 104 ("each core reads from its own dedicated DRAM bank in parallel, achieving maximum aggregate DRAM bandwidth with no bank contention"). The bullet now summarises content that no longer lives in `tensix_memory_hierarchy.md`, making it a cross-file duplicate. **Edit applied:** replaced the bullet with a forward pointer to `sharding_patterns.md` and the Multicast subsection.

## MINOR Suggestions

1. **`index.md` line 12 vs `tensix_memory_hierarchy.md` Key Takeaways line 103:** The `packer_l1_acc` gloss in the index table ("Packer accumulates into L1 output buffer instead of read-modify-write to DRAM... eliminates one DRAM read per K-block") closely mirrors the Key Takeaways bullet wording. This is expected index-vs-chapter overlap and does not warrant a cut, but the phrase "eliminates one DRAM read per K-block" is shared verbatim. Acceptable as-is given the navigational role of `index.md`.

## VERDICT
- Crucial updates: yes

# Compression Pass 3 — Chapter 4: Memory and Sharding
Date: 2026-03-24

## CRUCIAL Suggestions

None found.

## MINOR Suggestions

1. **`tensix_memory_hierarchy.md` Multicast subsection (lines 75–77) vs. `sharding_patterns.md` height-sharding section (line 13):** Both state that weight tiles are read once from DRAM and multicast via NoC to all cores. tensix_memory_hierarchy.md's treatment is the dedicated mechanism explanation (explicitly contrasting with O(n_cores) redundant reads); sharding_patterns.md's mention is a single incidental clause in the height-sharding config description. The overlap is a single clause, not a paragraph, and the two occurrences serve different roles. No cut is warranted, but if further compression is desired, sharding_patterns.md line 13 could be shortened to "Weight tiles are multicast via NoC (see [tensix_memory_hierarchy.md](tensix_memory_hierarchy.md#multicast-one-dram-read-many-cores))." This is low priority.

## VERDICT
- Crucial updates: no
