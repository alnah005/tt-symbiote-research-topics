# Chapter 6: Comparative Analysis — Choosing the Right Approach

## Overview

Chapters 3 through 5 developed the two primary matmul strategies for Mixture-of-Experts (MoE) inference on Tenstorrent Wormhole B0 hardware:

- **Chapter 3** profiled batched matmul for MoE: high throughput at high expert utilization, padded gather layout, DRAM-bound at low utilization.
- **Chapter 4** analyzed sparse matmul: tile-skipping via a sparsity tensor, strong advantage when the active fraction $\rho$ is low, non-monotonic latency as $\rho$ increases.
- **Chapter 5** covered sparsity tensor construction: $O(B \times k)$ build cost, per-step update requirements, placement in L1.

This chapter synthesizes those findings into a concrete decision framework. The central result is straightforward:

> **Tip:** For Qwen3.5-35B on T3K (Tenstorrent 3000 series, 8-device configuration), the recommended default is a **hybrid strategy**: use batched matmul during the prefill phase and sparse matmul during the decode phase.

---

## Prerequisites

- Chapter 1: MoE Architecture Fundamentals
- Chapter 2: TTNN Wormhole Primer
- Chapter 3: Batched Matmul for MoE
- Chapter 4: Sparse Matmul for MoE
- Chapter 5: Sparsity Tensor Construction

---

## Key Model Parameters (Qwen3.5-35B on T3K)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Total experts | $E$ | 256 |
| Top-k per token | $k$ | 8 |
| Hidden dimension | $H$ | 7168 |
| Number of devices | $N$ | 8 |
| Experts per device | $E_d$ | 32 |
| Sparsity ratio (decode, B=1) | $\rho$ | $k/E = 8/256 \approx 3.1\%$ |

The sparsity ratio $\rho$ is defined as the fraction of expert slots that are active in the activation tensor. At decode with batch size $B=1$, only 8 out of 256 experts receive a token, so 96.9% of tiles can be skipped.

---

## Decision Flowchart

The following flowchart guides selection between batched matmul and sparse matmul for a given inference request:

```
Start: MoE layer forward pass
        |
        v
Is this a PREFILL step?
(seq_len > 1 or large batch)
        |
   YES  |  NO (decode step)
        |       |
        v       v
Batched     Estimate ρ = k / E
matmul       (or measure from router)
(default)         |
                  v
           ρ < 0.1?
           /        \
         YES         NO
          |           |
          v           v
       Sparse     0.1 ≤ ρ ≤ 0.5?
       matmul      /        \
       (strong)  YES         NO (ρ > 0.5)
                  |           |
                  v           v
              Profile     Batched
              both;       matmul
              sparse
              usually
              wins
```

For Qwen3.5-35B: $\rho = 3.1\%$ at decode — this is well below the 0.1 threshold, so sparse matmul is always preferred during decode without profiling.

---

## Chapter Navigation

| File | Description |
|------|-------------|
| `performance_comparison_matrix.md` | Quantitative comparison across four canonical scenarios (prefill/decode × large/small batch); model-dimension interaction analysis; non-monotonic latency curve |
| `memory_and_bandwidth_tradeoffs.md` | DRAM bandwidth pressure, L1 footprint analysis, T3K multi-chip considerations, sparsity tensor construction overhead |
| `decision_guide.md` | Structured decision rules, runtime sparsity measurement, hybrid strategy implementation, anti-patterns to avoid |

---

## Summary of Key Takeaways

**Hybrid strategy is the recommended default.** The two matmul approaches are not interchangeable at all operating points; they occupy complementary regimes:

| Phase | Batch | Expert utilization | Recommended approach |
|-------|-------|--------------------|----------------------|
| Prefill | Any | High ($\rho \to 1.0$) | Batched matmul |
| Decode | Large ($B \geq 32$) | High ($\rho \approx 1.0$) | Batched matmul (borderline) |
| Decode | Small ($B = 1$) | Very low ($\rho = 3.1\%$) | Sparse matmul |

**Why the hybrid works:**

1. During prefill, every expert receives many tokens (expert capacity $C$ is large), so the padded gather layout wastes little bandwidth and the matmul compute is dense. Batched matmul's simplicity and high throughput dominate.

2. During decode, batch size is small and each token selects only 8 of 256 experts. With 96.9% of experts idle, sparse matmul skips 96.9% of tile reads and tile computations — a direct throughput gain.

3. Phase detection (prefill vs. decode) is a static property of the inference loop; no per-step runtime measurement is required to apply the hybrid strategy.

**The sparsity ratio $\rho$ unifies the comparison.** All performance differences between the two approaches reduce to a single quantity: how many tiles are active. Chapters 3 and 4 developed the performance models in terms of $\rho$; this chapter converts those models into actionable thresholds.

---

## References

- `ch03_batched_matmul_for_moe/performance_profile_batched.md` — batched matmul throughput and gather cost analysis
- `ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md` — sparsity ratio threshold derivation ($\rho < 0.5$)
- `ch04_sparse_matmul_for_moe/sparse_matmul_internals.md` — tile-skip mechanics and metadata overhead
- `ch05_sparsity_tensor_construction/sparsity_tensor_placement.md` — L1 placement and sizing
- `ch05_sparsity_tensor_construction/constructing_from_router_output.md` — per-step construction requirements

## Next Steps

Proceed to `performance_comparison_matrix.md` for a quantitative breakdown of all four canonical inference scenarios, including worked expert-capacity calculations and the non-monotonic latency analysis for sparse matmul.
