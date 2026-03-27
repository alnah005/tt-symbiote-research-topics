# Chapter 2 — Fused QKV Projection

## Scope

This chapter dissects the fused QKV projection layer used during the decode step of the Ling MoE attention block on T3K. The central object of study is `TTNNLinearIColShardedWAllReduced`, an operator that collapses what would otherwise be three independent matmuls (one each for Q, K, and V) plus multiple collective communication operations into a single fused matmul followed by a single all-reduce.

The chapter answers **Question 1** of the guide: *Is `num_links=1` an optimal choice for the CCL all-reduce that follows the fused QKV matmul, and if not, what value should be used?*

## Prerequisites

Readers should be comfortable with the following concepts introduced in Chapter 1:

- **T3K mesh topology** — 8 Wormhole n300 chips arranged on a 1×8 logical mesh, connected by 16×100 Gb/s Ethernet ports per chip (see Chapter 1, `t3k_topology_primer.md`)
- **Tensor sharding strategies** — specifically WIDTH_SHARDED and HEIGHT_SHARDED layouts, and how a shard is placed on a Tensix core's L1 cache
- **Ling model dimensions** — `hidden_size=4096`, `num_heads=16`, `num_kv_heads=4`, `head_dim=128`
- **Fused QKV weight shape** — the concatenated weight matrix has 3072 output columns (`16×128 (Q) + 4×128 (K) + 4×128 (V) = 3,072`), as derived in Chapter 1

No TTNN Python API experience is required, though familiarity with the `ttnn.linear` call signature will help when reading code fragments.

## Reading Order

Work through the files in this order:

1. [`fusion_mechanics.md`](./fusion_mechanics.md) — How the fusion is implemented, which CCL primitive is chosen for the all-reduce, and a theoretical bandwidth model comparing compute time against communication time.
2. [`latency_savings_analysis.md`](./latency_savings_analysis.md) — Quantitative comparison of the unfused baseline (3 matmuls + 5 CCL ops) against the fused path (1 matmul + 1 all-reduce) at Ling's hidden dimension.
3. [`num_links_tuning.md`](./num_links_tuning.md) — Physical meaning of `num_links`, sensitivity analysis across `num_links` ∈ {1, 2, 4}, and the recommended value.

## Key Symbols Used in This Chapter

| Symbol | Meaning |
|---|---|
| `H` | Hidden size (4096 for Ling) |
| `D` | Head dimension (128) |
| `Nq` | Number of Q heads (16) |
| `Nkv` | Number of KV heads (4) |
| `C` | Total output columns of fused weight = `(Nq + 2·Nkv)·D` = 3072 |
| `P` | Number of chips = 8 |
| `L` | `num_links` for CCL operations |

**Start reading:** [Fusion Mechanics](fusion_mechanics.md)
