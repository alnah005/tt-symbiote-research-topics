# Chapter 4: Matmul Config and Math Fidelity

## What This Chapter Answers

Chapter 3 identified which stages of `TTNNExperts.forward` dominate latency at batch=1 decode. Chapter 4 moves from observation to action: given that Stages 4 and 5 (the three `sparse_matmul` calls) are the primary compute bottleneck, how should `_make_sparse_matmul_program_config` (`moe.py:L62–L91`) be tuned, and does the current `HiFi2` math fidelity setting leave performance on the table?

Two research questions govern this chapter:

- **Q3** — Are `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1` (set at `moe.py:L1138–L1157`) optimal for GLM-4-MoE and Bailing on T3K?
- **Q4** — Is `HiFi2` sufficient for expert matmuls, or would `LoFi` improve throughput without meaningful accuracy loss?

---

## Scope

This chapter covers **expert matmuls only**: the w1 (gate projection), w3 (up projection), and w2 (down projection) `sparse_matmul` calls inside `TTNNExperts.forward`. The gate linear in `TTNNMoE.forward` (`moe.py:L1449–L1454`) uses `HiFi4` and is held fixed — it is a precision-critical routing decision and is not a tuning target here.

---

## How Program Config Parameters Connect to Hardware

`MatmulMultiCoreReuseMultiCast1DProgramConfig` maps directly onto the Wormhole Tensix core execution model. Understanding the mapping is a prerequisite for knowing which parameters to sweep and why.

```
Activation tensor (in0)              Weight tensor (in1)
shape: (M, K)                        shape: (K, N)
                                        │
                     mcast_in0=True     │
  Source core ──────────────────────►  All cores simultaneously hold the same
  broadcasts one M-tile row           M-tile; each core owns a slice of N
                                        │
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Core c  (c ∈ {0 … num_cores-1})                                       │
  │                                                                         │
  │  L1 buffer: in0_block_w activation tiles  (K-direction)                │
  │           + per_core_N weight tiles        (N-direction)               │
  │           + per_core_M × per_core_N output tiles                       │
  │                                                                         │
  │  Outer loop: K_tiles / in0_block_w iterations                          │
  │    Inner step: FMA accumulation over in0_block_w × per_core_N tiles    │
  └─────────────────────────────────────────────────────────────────────────┘
```

**`in0_block_w`** controls how many K-dimension tiles are fetched into L1 per outer loop iteration. Larger values reduce the number of DRAM round-trips for the activation matrix and increase arithmetic intensity per L1 load. The current cap of `min(4, K_tiles)` is an L1 capacity constraint, not an arbitrary choice.

**`per_core_M`** controls how many M-tiles (output row tiles) each core owns. At batch=1 decode, the padded activation has exactly `SPARSITY_BLOCK_SIZE / TILE_SIZE = 32 / 32 = 1` M-tile. Setting `per_core_M=1` is therefore the only valid value at this batch size, and all cores share the same single M-tile via the `mcast_in0=True` broadcast.

**`per_core_N`** is derived — not tunable — computed from `out_features` and the number of active cores. It determines the column-slice of the weight each core owns.

**Throughput consequence:** The ratio `(active_cores × per_core_N) / n_tiles` determines core utilization. For GLM-4-MoE's gate/up projection (`n_tiles = 44`, `num_cores = 64`), only 44 of 64 cores are active — 68.75% utilization. This structural inefficiency cannot be resolved by config tuning alone.

---

## Tile Dimension Arithmetic for GLM-4-MoE

All tile counts use `ttnn.TILE_SIZE = 32`.

| Quantity | Value | Tile count |
|---|---|---|
| `hidden_size` | 4096 | 128 tiles |
| `moe_intermediate_size` | 1408 | 44 tiles |
| `n_routed_experts` | 128 | — |
| Padded tokens (`SPARSITY_BLOCK_SIZE`) | 32 | 1 M-tile |
| T3K cores (`core_x × core_y`) | 8 × 8 | 64 |

**Gate/up projection (w1, w3):** K = 128 tiles, N = 44 tiles. `per_core_N = ceil(44/64) = 1`. 44 cores active.

**Down projection (w2):** K = 44 tiles, N = 128 tiles. `per_core_N = ceil(128/64) = 2`. 64 cores active.

---

## Files in This Chapter

| File | Research question | Contents |
|---|---|---|
| [`program_config_tuning.md`](./program_config_tuning.md) | Q3 | Complete field-by-field guide to `MatmulMultiCoreReuseMultiCast1DProgramConfig` as constructed by `_make_sparse_matmul_program_config`. Derives valid `in0_block_w` sweep sets for GLM-4-MoE and Bailing. Provides a benchmarking harness and Pareto evaluation method. |
| [`math_fidelity_evaluation.md`](./math_fidelity_evaluation.md) | Q4 | Compares `HiFi2` (current) vs `LoFi` for expert matmuls. Accuracy metric definitions (cosine similarity, max absolute error). Measurement protocol. Go/no-go criterion for adopting `LoFi`. |

---

## Reading Order

Read `program_config_tuning.md` first: it establishes the parameter space and the benchmarking methodology that `math_fidelity_evaluation.md` then extends with accuracy tracking. Both files build on the shape analysis in this index.
