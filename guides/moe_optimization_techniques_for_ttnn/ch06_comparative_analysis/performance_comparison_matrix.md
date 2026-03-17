# Performance Comparison Matrix

This file provides a quantitative comparison of batched matmul and sparse matmul (SM) across four canonical inference scenarios for Qwen3.5-35B on Tenstorrent Wormhole B0. All calculations use the model constants $E = 256$ experts, $k = 8$ top-k, $H = 7168$ hidden dimension, and capacity factor $CF = 1.0$ (exact capacity, for analysis clarity; production uses $CF = 1.25$).

---

## 1. Four Canonical Scenarios

### Expert Capacity Formula

For a given batch size $B$ and sequence length $S$, expert capacity $C$ is:

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil$$

Using $CF = 1.0$, $k = 8$, $E = 256$:

$$C = \left\lceil \frac{8 \times B \times S}{256} \right\rceil = \left\lceil \frac{B \times S}{32} \right\rceil$$

The sparsity ratio $\rho$ at the expert-slot level is the fraction of expert slots receiving at least one token:

$$\rho = \frac{\text{active expert slots}}{E \times C}$$

Under uniform routing, each expert receives approximately $k \times B \times S / E$ tokens. When this number is $\geq 1$, essentially all 256 experts are active ($\rho \approx 1.0$). When $B \times S$ is very small relative to $E/k = 32$, most experts are idle.

### Scenario Table

| # | Scenario | $B \times S$ | $C$ (tokens/expert) | Expert utilization | Recommended approach | Primary reason |
|---|----------|-------------|--------------------|--------------------|----------------------|----------------|
| 1 | Prefill, large batch | $32 \times 2048 = 65{,}536$ | $\lceil 65{,}536/32 \rceil = 2{,}048$ | All 256 experts active, $\rho \approx 1.0$ | **Batched matmul** | Dense compute; padding waste negligible |
| 2 | Prefill, small batch | $1 \times 2048 = 2{,}048$ | $\lceil 2{,}048/32 \rceil = 64$ | All 256 experts receive tokens under uniform routing, $\rho \approx 1.0$ | **Batched matmul** | High expert coverage; batched matmul metadata-free |
| 3 | Decode, large batch | $32 \times 1 = 32$ | $\lceil 32/32 \rceil = 1$ | 32 tokens × 8 experts = 256 expert-assignments; nearly all 256 experts receive 1 token; $\rho \approx 1.0$ | **Batched matmul** (preferred) | High expert utilization; SM metadata overhead not worth it |
| 4 | Decode, small batch | $1 \times 1 = 1$ | $\lceil 1/32 \rceil = 1$ | Only 8 of 256 experts receive the 1 token; 248 experts (96.9%) idle; $\rho \approx 3.1\%$ | **Sparse matmul** | 96.9% tiles skipped; SM advantage is maximal |

### Scenario 1: Prefill, Large Batch ($B=32$, $S=2048$)

Total tokens: $32 \times 2048 = 65{,}536$. Expert assignments: $65{,}536 \times 8 = 524{,}288$. Average assignments per expert: $524{,}288 / 256 = 2{,}048$. Every expert receives exactly $C = 2{,}048$ tokens (under uniform routing). The expert-grouped activation tensor $[\,E,\, C,\, H\,] = [256,\, 2{,}048,\, 7{,}168]$ has zero empty slots. The gather phase moves real data into every slot — no padding waste. Sparse matmul's sparsity tensor would encode all tiles as active, contributing metadata overhead with no tile-skip savings. **Batched matmul is the clear winner.**

### Scenario 2: Prefill, Small Batch ($B=1$, $S=2048$)

Total tokens: $2{,}048$. Expert assignments: $2{,}048 \times 8 = 16{,}384$. Average per expert: $64$. Under uniform routing, every expert receives approximately 64 tokens: $C = 64$ tokens/expert, all 256 experts active. The expert-grouped tensor is $[256,\, 64,\, 7{,}168]$. While smaller than scenario 1, utilization is still high ($\rho \approx 1.0$). The sparsity tensor for sparse matmul would again mark nearly all tiles as active. **Batched matmul preferred** — no benefit from sparsity machinery.

### Scenario 3: Decode, Large Batch ($B=32$, $S=1$)

Total tokens: $32$. Expert assignments: $32 \times 8 = 256$. With $E = 256$ experts, this means on average exactly 1 assignment per expert under uniform routing. The expert-grouped capacity is $C = \lceil 32/32 \rceil = 1$ token/expert. All 256 experts receive approximately 1 token — expert utilization is high despite the small total token count. The sparsity tensor would mark nearly all expert rows as active. **Batched matmul is preferred**: the gather phase is cheap (only 32 tokens to move), and sparse matmul's per-tile metadata check provides minimal savings when nearly all tiles are active. This scenario is borderline and worth profiling under actual routing distributions.

### Scenario 4: Decode, Small Batch ($B=1$, $S=1$)

Total tokens: $1$. Expert assignments: $8$. Exactly 8 of 256 experts receive this token. The remaining $256 - 8 = 248$ experts (96.9%) are idle. With $C = 1$, the expert-grouped tensor shape is $[256,\, 1,\, 7{,}168]$, but 248 of those 256 rows are zero-padded. Batched matmul would process all 256 rows (with 248 wasted), spending DRAM bandwidth reading zero-padding. Sparse matmul reads only the 8 active rows and skips the rest. **Sparse matmul wins strongly**: $\rho = 8/256 = 3.1\%$, which is well below the $\rho < 0.1$ threshold derived in Chapter 4.

---

## 2. How Model Dimensions Interact

### Effect of Hidden Dimension $H$

Both approaches' compute scales with $H$. The sparse matmul advantage per skipped tile scales with tile compute cost, which is $O(T^2)$ for a tile of dimension $T = 32$. A larger $H$ means more tiles along the $H$ dimension:

$$\text{tiles along } H = \left\lceil \frac{H}{32} \right\rceil = \left\lceil \frac{7168}{32} \right\rceil = 224$$

For each skipped expert (row), sparse matmul avoids computing 224 tiles. Larger $H$ amplifies the per-skipped-row savings. At $H = 7{,}168$, the 248 skipped experts in scenario 4 represent $248 \times 224 = 55{,}552$ skipped tile operations.

### Effect of Expert FFN Intermediate Dimension $d_{ff}$

The expert feed-forward network (FFN) intermediate dimension $d_{ff}$ sets the weight matrix size: $[H, d_{ff}]$. A larger $d_{ff}$ increases per-tile work. This amplifies the benefit of skipping tiles: each skipped tile now represents more compute. As $d_{ff}$ grows, the break-even $\rho$ threshold (above which batched matmul is preferred) shifts slightly upward, extending the regime where sparse matmul is advantageous.

### Effect of $E$ (Number of Experts) and $k$ (Top-k)

The sparsity ratio is:

$$\rho = \frac{k}{E}$$

(at decode, $B=1$, $S=1$, under the assumption that each selected expert is distinct, which holds when $k \ll E$).

- **Increasing $E$** (e.g., from 256 to 512): $\rho$ drops from $3.1\%$ to $1.6\%$. More sparsity — sparse matmul becomes even more advantageous.
- **Decreasing $E$** (e.g., to 64): $\rho$ rises to $12.5\%$ — still below $0.5$, but approaching the profiling regime.
- **Increasing $k$** (e.g., from 8 to 16): $\rho$ doubles from $3.1\%$ to $6.25\%$ — still strongly in the sparse matmul regime.
- **Decreasing $k$** (e.g., to 2): $\rho$ drops to $0.78\%$ — even stronger sparse matmul advantage.

For Qwen3.5-35B, $\rho = 8/256 = 3.1\%$ represents one of the lowest practical sparsity ratios in production MoE models. This places Qwen3.5-35B decode firmly in the regime where sparse matmul is optimal.

---

## 3. Non-Monotonic Latency of Sparse Matmul

### Why Latency Is Non-Monotonic

Sparse matmul latency is not simply proportional to $\rho$. There are two cost components:

1. **Metadata read cost**: For every tile position (active or not), the sparse matmul kernel must read the sparsity tensor entry to decide whether to skip the tile. This is an $O(E_d \times M_t \times K_t)$ operation regardless of $\rho$, where $M_t$ and $K_t$ are the number of tile rows and columns.

2. **Compute cost**: Only active tiles contribute compute. At fraction $\rho$ active, this is $O(\rho \times E_d \times M_t \times K_t \times T^2)$ multiply-accumulate operations.

As $\rho$ increases from 0 to 1:

- At $\rho \to 0$: compute cost is negligible; metadata cost is manageable; total latency is low.
- At moderate $\rho$ (0.1–0.3): compute cost grows; metadata cost is constant; total latency rises roughly linearly with $\rho$.
- At $\rho \to 0.5$: sparse matmul latency approximately matches batched matmul, which has no metadata overhead.
- At $\rho > 0.5$: sparse matmul latency exceeds batched matmul because metadata overhead is paid for tiles that mostly would have been computed anyway.

> **Warning:** The crossover is NOT at $\rho = 1.0$. Sparse matmul becomes slower than batched matmul at approximately $\rho \approx 0.5$, not when fully dense. The exact crossover depends on tile size (32×32), $d_{model}$, and Wormhole B0 memory hierarchy characteristics.

### Latency Crossover Concept

The following ASCII diagram illustrates the qualitative relationship between sparse matmul latency ($L_{SM}$) and batched matmul latency ($L_{BM}$) as a function of $\rho$:

```
Latency
  |
  |                                         /  L_SM (sparse matmul)
  |                                       /
  |                                     /
  |                                   /
  |                         _________/___________
  |  L_BM (batched matmul) /
  |                       X  <-- crossover (~ρ ≈ 0.5)
  |                     /
  |                   /
  |                 /
  |    ____________/
  |___/
  |
  +-----|---------|---------|---------|---------|---> ρ (active fraction)
       0.0       0.1       0.2       0.3       0.5
       ^
       |
  Qwen3.5-35B decode B=1
  ρ = 3.1% → sparse matmul
  has large latency advantage
```

Key observations from the diagram:

- At $\rho = 3.1\%$ (Qwen3.5-35B decode, $B=1$): $L_{SM} \ll L_{BM}$ — large sparse matmul advantage.
- At $\rho \approx 0.5$: the two curves intersect; neither approach has a clear advantage.
- At $\rho > 0.5$: $L_{SM} > L_{BM}$ — batched matmul preferred.
- $L_{SM}$ is not strictly linear in $\rho$ because the metadata read cost creates a non-zero floor even at $\rho = 0$.

The non-monotonic property also implies that profiling at one $\rho$ value does not predict performance at a different $\rho$. If the routing distribution changes (e.g., load imbalance raises $\rho$ above the expected value), sparse matmul latency can increase faster than expected.

---

## References

- `ch03_batched_matmul_for_moe/performance_profile_batched.md` — expert-capacity and gather cost analysis
- `ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md` — $\rho < 0.5$ threshold derivation and crossover analysis
- `ch04_sparse_matmul_for_moe/sparse_matmul_internals.md` — tile-skip mechanics, metadata read cost, non-monotonic latency detail
- `ch05_sparsity_tensor_construction/sparsity_tensor_placement.md` — sparsity tensor shape and sizing

## Next Steps

Proceed to `memory_and_bandwidth_tradeoffs.md` for an analysis of DRAM bandwidth pressure, L1 footprint, and T3K multi-chip considerations for both approaches.
