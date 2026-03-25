# B Review — Chapter 5: Multi-Device Scaling — Pass 1

## Issues Found

None found.

## Verification of Confirmed Facts

| Fact | Status |
|---|---|
| N300: 2-chip Wormhole, on-board Ethernet (not PCIe), max TP=2 | CORRECT — stated accurately in index.md, tensor_parallelism.md, and ccl_and_ethernet.md |
| T3K: 8-chip Wormhole Ethernet ring (not mesh), max TP=8 | CORRECT — ring topology explicitly noted; "not a mesh" stated in ccl_and_ethernet.md |
| Column-parallel linear: weight shard [K, N/TP]; activation broadcast; post-op ttnn.all_gather | CORRECT |
| Row-parallel linear: weight shard [K/TP, N]; activation pre-sharded along K; post-op ttnn.reduce_scatter (not all_reduce) | CORRECT — the distinction from all_reduce is explicitly called out in both tensor_parallelism.md and ccl_and_ethernet.md |
| Vocab-parallel LM head: weight shard [vocab_size/TP, hidden]; post-op ttnn.all_reduce | CORRECT — correctly identified as the only use of all_reduce in the standard transformer TP pattern |
| GQA: TP <= n_kv_heads → KV sharded; TP > n_kv_heads → KV replicated | CORRECT |
| Llama 3.1 70B: 64 Q heads, 8 KV heads, TP=8 → 8 Q heads and 1 KV head per device | CORRECT |
| G = n_q_heads / n_kv_heads = 8 for Llama 3.1 70B | CORRECT |
| ttnn.all_gather: every device receives full tensor | CORRECT |
| ttnn.reduce_scatter: partial sums aggregated; each device gets one shard | CORRECT |
| ttnn.all_reduce: all_gather + reduce; every device gets full reduced result; only correct for vocab-parallel LM head | CORRECT |
| BFP4 vs BF16 bandwidth: 3.56x | CORRECT |
| BFP8 vs BF16 bandwidth: 1.88x | CORRECT |

VERDICT: APPROVED

---

# B Review — Chapter 5: Multi-Device Scaling — Pass 2

## Issues Found

None found.

All C Pass 1 edits applied correctly:
- `index.md` Hardware Context section now contains only pointer sentences (no raw specs duplicated there); confirmed.
- `tensor_parallelism.md` Summary of Collectives table removed; reduce_scatter/all_reduce rationale trimmed to cross-references pointing to `ccl_and_ethernet.md`; confirmed.
- `ccl_and_ethernet.md` is unchanged and retains the authoritative Summary table and full rationale; confirmed.

No new factual claims introduced by the C Pass 1 edits. No claims removed that would leave a key fact unverified.

## Verification of Confirmed Facts

| Fact | Status |
|---|---|
| N300: 2-chip Wormhole, on-board Ethernet, max TP=2 | CORRECT — stated accurately in index.md Key Concepts table, tensor_parallelism.md Hardware Configurations, and ccl_and_ethernet.md |
| T3K: 8-chip Wormhole Ethernet ring (not mesh), max TP=8 | CORRECT — "not a mesh" stated explicitly in both tensor_parallelism.md and ccl_and_ethernet.md |
| Column-parallel: ttnn.all_gather; Row-parallel: ttnn.reduce_scatter (not all_reduce); Vocab-parallel LM head: ttnn.all_reduce | CORRECT — all three collectives assigned correctly throughout; the reduce_scatter/all_reduce distinction is cross-referenced to ccl_and_ethernet.md as required |
| GQA: TP <= n_kv_heads → shard; TP > n_kv_heads → replicate | CORRECT |
| Llama 3.1 70B: 64 Q heads, 8 KV heads, TP=8 → 8 Q heads and 1 KV head per device | CORRECT |
| BFP4 vs BF16: 3.56x; BFP8 vs BF16: 1.88x | CORRECT |

VERDICT: APPROVED
