# Chapter 4: sparse_matmul for MoE — When and How to Skip Zero Tiles

## Overview

Chapter 3 (`ch03_batched_matmul_for_moe/`) established the batched matmul strategy: gather tokens by expert assignment, stack them into a padded `[E, C, H]` activation tensor, and compute all expert FFNs in a single batched kernel call. That approach is efficient when expert capacity $C$ is large (prefill regime) but wastes hardware resources when $C$ is small — at decode with $B=32$, $E=256$, $k=8$, we get $C=1$, which means 31 of 32 rows in every tile are padding zeros. The FPU executes FMAs against zeros, consuming the same cycles it would spend on real tokens.

This chapter introduces `ttnn.sparse_matmul`, a kernel variant that accepts a tile-level sparsity mask and skips any tile whose mask bit is zero. For MoE workloads, the mask encodes which token slots in the activation tensor contain real (non-padding) tokens. When routing leaves many expert slots empty — as it does at low batch sizes — the skip savings are substantial.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Explain the tile-skip mechanism in `sparse_matmul`: how the sparsity mask is structured, how the kernel checks it before loading a tile pair, and why the mask shape must be static even though mask values vary.
2. State the canonical dtype configuration for MoE sparse matmul (BFP8 weights, BF16 activations) and identify which tensor provides the sparse tiles (activations, not weights).
3. Derive the sparsity ratio $\rho$ from expert routing parameters and compute it for the decode and prefill regimes of Qwen3.5-35B on a T3K device.
4. Identify the crossover sparsity ratio $\rho^*$ below which `sparse_matmul` outperforms batched matmul, and qualitatively explain how it depends on hidden dimension $H$ and per-expert intermediate dimension $D$ [D UNVERIFIED].
5. Select a `SparsityConfig` and appropriate program config parameters (`per_core_M`, `grid_y`) for the decode regime, and contrast them with the Chapter 3 prefill configs.
6. State the static-shape requirement for program caching and explain why padded capacity $C$ must be fixed even when the number of active tokens varies.

---

## Chapter Notation

New symbols introduced in this chapter. All symbols from Chapter 3 (`index.md` notation table) remain in effect.

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\rho$ | Sparsity ratio — fraction of activation tiles that are **active** (non-zero) | $\rho = 1$ means fully dense; $\rho = 0$ means all tiles inactive |
| $\rho^*$ | Crossover sparsity ratio: value of $\rho$ at which `sparse_matmul` and batched matmul have equal latency | Approximately 0.7 empirically |
| $E_d$ | Local expert count per device in expert-parallel configuration | $E_d = E / N = 256 / 8 = 32$ for T3K with $N=8$ devices |
| $M_t$ | Tile count along M (capacity) dimension: $\lceil C / 32 \rceil$ | From Ch. 3; repeated here for reference |
| $T_\text{act}$ | Number of active (non-zero) activation tiles out of total $M_t \times K_t$ tiles | $T_\text{act} = \rho \times M_t \times K_t$ |
| $\alpha$ | Capacity oversubscription factor ($\alpha \geq 1$) | From Ch. 3; default $\alpha = 1.0$ |

---

## Summary Comparison: sparse_matmul vs. Batched Matmul

| Dimension | Batched Matmul | sparse_matmul |
|-----------|---------------|---------------|
| **Core mechanism** | Computes all tiles, including zero-padding | Checks mask per tile; skips zero (inactive) tiles |
| **Activation dtype** | BF16 | BF16 (sparse tiles in BF16) |
| **Weight dtype** | BFP8 | BFP8 (weights always read for active tiles) |
| **Sparsity bookkeeping** | None | Must construct and maintain sparsity mask tensor |
| **Mask static-shape constraint** | N/A | Mask shape $[C/32, H/32]$ must be fixed per program cache key |
| **FLOP efficiency at decode ($\rho \approx 0.03$)** | ~3% (31/32 tiles are padding) | ~3% of FLOPs executed — matches actual active fraction |
| **FLOP efficiency at prefill ($\rho \approx 1$)** | Near 1.0 | Slightly below 1.0 (mask overhead on every tile) |
| **Grid utilization at decode** | Launches full grid; most cores compute zeros | Small grid (e.g., 2×8) sufficient; cores do real work |
| **Program caching** | Cache key: $(B, S, C)$ shapes | Cache key: $(B, S, C)$ shapes + mask shape |
| **Pro** | Simple; no metadata; optimal at high $C$ | Skips zeros explicitly; latency scales with $\rho$ at decode |
| **Con** | Wastes compute at low $C$; high padding waste at decode | Mask construction overhead; break-even at $\rho \approx 0.7$ |
| **Recommended use** | Prefill ($\rho > 0.7$); balanced routing; high $C$ | Decode ($\rho < 0.7$); imbalanced routing; low $C$ |
| **Not recommended** | Decode with $C=1$ and many empty experts | Prefill at full capacity; irregular sparsity patterns |

---

## Files in This Chapter

| File | Contents |
|------|----------|
| [sparse_matmul_internals.md](sparse_matmul_internals.md) | Tile-skip mechanism, sparsity mask structure and dtype, static-shape constraint and its rationale, FLOP cost model, relationship between routing and sparsity |
| [when_sparse_matmul_wins.md](when_sparse_matmul_wins.md) | Crossover analysis, two-regime treatment (decode vs. prefill), sparsity ratio derivation for Qwen3.5-35B on T3K, sequence length effects, when not to use sparse_matmul |
| [program_configs_sparse.md](program_configs_sparse.md) | SparsityConfig parameter, grid sizing for sparse decode, three worked decode configs, static-shape requirement for program caching, contrast with Ch. 3 prefill configs |

Read the files in order. `when_sparse_matmul_wins.md` assumes familiarity with the tile-skip mechanism from `sparse_matmul_internals.md`. `program_configs_sparse.md` assumes both.

---

## Prerequisites

- **Chapter 1** — MoE Architecture Fundamentals: top-K routing, expert capacity, and the dispatch-combine pattern are assumed throughout without re-introduction.
- **Chapter 2** — TTNN and Wormhole Primer: tile layout, BF16/BFP8 dtypes, L1 budget formulas, and program config vocabulary (`per_core_M`, `per_core_N`, `out_subblock_h`, `out_subblock_w`, `in0_block_w`) are assumed known.
- **Chapter 3** — Batched Matmul for MoE (`ch03_batched_matmul_for_moe/`): the gather-pad-matmul pattern, expert capacity formula, and the program configs for decode and prefill regimes established in `program_configs_batched.md` are directly compared throughout this chapter. Specifically:
  - The decode config (`MatmulMultiCoreProgramConfig`, grid `(8, 1)`, `per_core_M=1`) is the baseline for comparison.
  - The prefill config (`MatmulMultiCoreReuseMultiCastProgramConfig`, grid `(8, 2)`) represents the high-$C$ regime where batched matmul is preferred.
  - The observation in `performance_profile_batched.md` §2.2 — that tile-level FLOP efficiency at decode is approximately $1/32 \approx 3\%$ — motivates the switch to `sparse_matmul`.

---

## Next Steps

Begin with [sparse_matmul_internals.md](sparse_matmul_internals.md) to understand how the tile-skip mechanism works at the kernel level, then proceed to [when_sparse_matmul_wins.md](when_sparse_matmul_wins.md) for the quantitative crossover analysis, and finish with [program_configs_sparse.md](program_configs_sparse.md) for configuration examples.

After completing this chapter, Chapter 5 covers memory layout optimization — how to arrange expert weight tensors and activation buffers in DRAM to maximize bandwidth utilization regardless of which matmul kernel is used.

---

## References

- Chapter 3, `performance_profile_batched.md` — FLOP efficiency at decode, arithmetic intensity, motivation for sparse approach.
- Chapter 3, `program_configs_batched.md` — Baseline program configs that Chapter 4 contrasts against.
- Chapter 2, `matmul_fundamentals_in_ttnn.md` — Program config vocabulary and L1 budget formulas.
- Chapter 1, `routing_and_sparsity.md` — Top-K routing, expert capacity, and sparsity in MoE layers.
