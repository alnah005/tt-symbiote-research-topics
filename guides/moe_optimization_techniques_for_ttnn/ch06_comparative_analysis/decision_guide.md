# Decision Guide: Selecting the Right Matmul Strategy

This file provides structured decision rules for selecting between batched matmul and sparse matmul (SM) in a Mixture-of-Experts (MoE) deployment on Tenstorrent Wormhole B0. The rules are grounded in the quantitative analysis from `performance_comparison_matrix.md` and `memory_and_bandwidth_tradeoffs.md`. All rules use Qwen3.5-35B constants: $E = 256$, $k = 8$, $H = 7{,}168$, $N = 8$ devices (T3K).

---

## 1. Structured Decision Rules

### Rule 1: Phase-Based Default

The inference phase (prefill vs. decode) is the most reliable predictor of expert utilization:

| Phase | Default strategy | Rationale |
|-------|-----------------|-----------|
| Prefill | **Batched matmul** | Expert capacity $C$ is large; all 256 experts receive tokens; $\rho \approx 1.0$; sparsity machinery adds overhead with no tile-skip savings |
| Decode | **Sparse matmul** | Expert capacity $C = 1$; at $B=1$, only 8 of 256 experts are active; $\rho = 3.1\%$; 96.9% of tiles can be skipped |

Phase detection is a static property of the inference loop — a single `is_decode` boolean flag is sufficient. No per-step runtime measurement is required to apply this rule.

**Rationale for prefill:** During prefill with $B=32$, $S=2048$, the expert-grouped activation tensor $[256,\, 2{,}048,\, 7{,}168]$ is fully populated. All 256 experts receive approximately 2,048 tokens each. The sparsity ratio $\rho \approx 1.0$, so sparse matmul would read the entire sparsity tensor (448 KB at this scale) and find no tiles to skip — pure overhead.

**Rationale for decode:** During decode with $B=1$, $S=1$, exactly 8 experts out of 256 receive the single input token. The sparsity ratio $\rho = 8/256 = 3.1\%$. Sparse matmul skips $248/256 = 96.9\%$ of tile reads and computations, providing a near-linear throughput improvement for the activation load.

### Rule 2: Sparsity Ratio Threshold

When phase alone is insufficient (e.g., non-standard models, unusual batch sizes), apply the sparsity ratio threshold:

$$\rho = \frac{k}{E} \quad \text{(at decode, B=1, uniform routing)}$$

More generally, measure $\rho$ as described in Section 2.

| Sparsity ratio | Recommended strategy |
|---------------|----------------------|
| $\rho < 0.1$ | Sparse matmul strongly preferred |
| $0.1 \leq \rho \leq 0.5$ | Profile to determine crossover; sparse matmul usually still preferred |
| $\rho > 0.5$ | Batched matmul preferred |

For Qwen3.5-35B: $\rho = 3.1\%$ at decode, $B=1$ — well below the 0.1 threshold. Sparse matmul is the correct choice without profiling.

> **Warning:** The $\rho < 0.5$ threshold is approximate. The exact crossover depends on tile size (32×32), $H$, and Wormhole B0 memory hierarchy behavior. See `ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md` for the derivation. When $\rho$ falls in the $[0.1, 0.5]$ range, profiling (Section 4) is required.

### Rule 3: Expert Capacity Threshold

Expert capacity $C$ is a more directly observable quantity than $\rho$ in some deployment contexts:

| Expert capacity per device | Recommended strategy |
|---------------------------|----------------------|
| $C > 64$ tokens/expert | Batched matmul preferred (high utilization) |
| $8 < C \leq 64$ | Profile; expert utilization is moderate |
| $C \leq 8$ tokens/expert | Sparse matmul preferred |

At $C = 1$ (decode, $B=1$): sparse matmul is strongly preferred. At $C = 2{,}048$ (prefill, $B=32$): batched matmul is strongly preferred.

The capacity threshold $C \leq 8$ corresponds to approximately $\rho \leq k \times 8 / (E \times C)$ — a rough alignment with the $\rho < 0.1$ threshold from Rule 2.

---

## 2. Measuring Sparsity Ratio at Runtime

When the routing distribution is unknown or non-uniform, measure $\rho$ directly from the router output:

```python
def measure_sparsity_ratio(router_indices: torch.Tensor, E: int, C: int) -> float:
    """
    Measure actual sparsity ratio ρ from router output.

    router_indices: [B, k] tensor of expert indices selected per token
    E: total number of experts
    C: expert capacity (max tokens per expert)
    Returns: fraction of expert slots that are active (1 - skip fraction)
    """
    B, k = router_indices.shape
    # Count unique experts receiving at least one token
    active_experts = len(router_indices.unique())
    total_expert_slots = E  # one slot per expert in the capacity buffer
    rho = active_experts / total_expert_slots
    return rho
```

**Usage notes:**

- `router_indices` is the $[B, k]$ tensor of expert indices output by the router, where $B$ is batch size and $k = 8$ is top-k.
- This function measures the fraction of experts that receive at least one token — a proxy for how many expert rows in the activation tensor are non-zero.
- For Qwen3.5-35B at decode $B=1$: `router_indices.unique()` returns 8 indices; `rho = 8/256 = 0.031`.
- At decode $B=32$: up to 256 unique experts may be selected; `rho` approaches 1.0.
- The returned $\rho$ can be passed directly to `get_matmul_strategy` (Section 3).

> **Tip:** Runtime measurement of $\rho$ adds negligible overhead ($O(B \times k)$ work) and provides an exact sparsity estimate rather than relying on the uniform-routing approximation $\rho = k/E$. For production deployments with non-uniform or load-imbalanced routing, runtime measurement is preferred.

---

## 3. Hybrid Strategy Implementation

The hybrid strategy applies batched matmul for prefill and sparse matmul for decode. The selection logic is encoded in a single function:

```python
def get_matmul_strategy(is_decode: bool, rho: float) -> str:
    """
    Select matmul strategy based on phase and sparsity ratio.

    is_decode: True if in decode (token generation) phase
    rho: measured or estimated sparsity ratio (fraction of active expert slots)
    Returns: 'sparse' or 'batched'
    """
    if rho > 0.5:
        # High utilization: batched matmul always wins
        return 'batched'
    if not is_decode:
        # Prefill: expert capacity is large, batched matmul preferred
        # unless rho is somehow very low (unusual for prefill)
        return 'batched'
    # Decode with low sparsity ratio: sparse_matmul preferred
    return 'sparse'
```

**Integration pattern:**

```python
class MoELayer:
    def forward(self, x: torch.Tensor, router_indices: torch.Tensor,
                is_decode: bool) -> torch.Tensor:
        # Measure actual sparsity ratio from router output
        rho = measure_sparsity_ratio(router_indices, E=256, C=1 if is_decode else None)

        strategy = get_matmul_strategy(is_decode=is_decode, rho=rho)

        if strategy == 'sparse':
            return self._forward_sparse(x, router_indices)
        else:
            return self._forward_batched(x, router_indices)
```

**Phase detection:** The `is_decode` flag is passed from the outer inference loop. It is set to `False` during the initial prefill pass (processing the full prompt) and `True` during all subsequent decode steps (generating one token at a time). This is a static property — it does not require inspecting the input tensor shape at each step.

**Sparsity tensor lifecycle with the hybrid strategy:**

- Prefill: no sparsity tensor is constructed; batched matmul uses a standard gather layout.
- First decode step: construct sparsity tensor from the router output; pass to sparse matmul kernel.
- Each subsequent decode step: reconstruct sparsity tensor (routing changes each step; see Section 5, Anti-pattern 3).

---

## 4. When to Profile Rather Than Guess

The phase-based default (Rule 1) and the $\rho$ thresholds (Rule 2) cover the vast majority of Qwen3.5-35B deployment scenarios. Profile explicitly when:

1. **Non-standard $k$ or $E$**: If the model uses $k \neq 8$ or $E \neq 256$, recompute $\rho = k/E$ and verify which threshold regime applies. A model with $k=16$, $E=64$ has $\rho = 25\%$ at decode $B=1$ — this falls in the $[0.1, 0.5]$ profiling regime.

2. **Batch size in the borderline regime**: Decode with $B=32$ on Qwen3.5-35B has approximately $\rho \approx 1.0$ (all experts active), which favors batched matmul. Batch sizes between $B=4$ and $B=32$ can produce $\rho$ values in $[0.1, 0.5]$ — profile to find the crossover.

3. **After a TTNN firmware update**: Kernel implementations for both batched matmul and sparse matmul may change between TTNN releases, shifting the crossover point. Re-profile after firmware updates.

4. **Significant deviation from Qwen3.5-35B dimensions**: If $H$ or $d_{ff}$ differs substantially, the tile count and per-tile compute change, which shifts the crossover $\rho$.

**Profiling procedure:**

Run both batched matmul and sparse matmul on the same input shape. Use 100 warmup iterations (to prime L1 and DRAM caches) followed by 1,000 measurement iterations. Compare median latency (not mean — outliers from DRAM cache misses can skew the mean). The approach with lower median latency at the target batch size and sequence length is preferred.

---

## 5. Anti-Patterns to Avoid

### Anti-Pattern 1: Sparse Matmul for Prefill at Large Sequence Lengths

Using sparse matmul during prefill when $B \geq 1$ and $S \geq 2{,}048$ results in $\rho \approx 1.0$: every expert receives tokens, so nearly all tiles are active. The sparsity tensor metadata overhead is paid in full with no tile-skip benefit. At $S = 2{,}048$, the sparsity tensor itself reaches 448 KB per device — non-negligible L1 pressure — and encodes almost entirely "active" entries.

> **Warning:** Applying sparse matmul during prefill with $B=32$, $S=2{,}048$ is strictly slower than batched matmul due to metadata overhead. The performance regression can be 20–50% depending on sequence length and batch size.

### Anti-Pattern 2: Batched Matmul for Decode at $B=1$

Using batched matmul for single-token decode means gathering 1 input token into a $[256,\, 1,\, 7{,}168]$ expert-grouped tensor where 248 of 256 rows are zero-padding. The downstream matmul reads all 256 rows from DRAM, but 96.9% of the data contributes nothing to the output. This wastes DRAM bandwidth and L1 space loading zero-padded slots.

> **Warning:** Batched matmul at decode $B=1$ performs approximately $32\times$ more DRAM reads for the activation tensor than necessary, compared to sparse matmul. On a bandwidth-bound operation, this translates directly to a latency penalty.

### Anti-Pattern 3: Reusing a Stale Sparsity Tensor Across Decode Steps

The router selects different experts for each generated token. The sparsity tensor encodes which expert rows are active for a specific decode step. If the tensor is not rebuilt at each step, the kernel will skip rows that are now active (producing zeros in the output) or process rows that are now inactive (wasting compute). Both errors are silent — no exception is raised.

> **Warning:** A stale sparsity tensor produces silent correctness errors. The model output will be wrong, but no runtime error will indicate the cause. Always rebuild the sparsity tensor at every decode step. The rebuild cost is $O(B \times k)$ — negligible compared to the matmul.

### Anti-Pattern 4: Not Updating the Sparsity Tensor When Batch Size Changes

The sparsity tensor shape depends on $B$ (through expert capacity $C = \lceil k \times B / E \rceil$). If the batch size changes between requests (e.g., dynamic batching) without rebuilding the sparsity tensor, the tensor shape may be stale. This causes either a shape mismatch error (if the kernel validates shapes) or incorrect tile-skip behavior (if it does not).

> **Warning:** In dynamic batching deployments, treat the sparsity tensor as a per-request, per-step artifact. Never cache it across requests or across decode steps of different sequence lengths.

---

## References

- `ch03_batched_matmul_for_moe/performance_profile_batched.md` — gather cost and bandwidth utilization at varying batch sizes
- `ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md` — $\rho < 0.5$ threshold derivation and profiling methodology
- `ch04_sparse_matmul_for_moe/sparse_matmul_internals.md` — metadata overhead and non-monotonic latency analysis
- `ch05_sparsity_tensor_construction/sparsity_tensor_placement.md` — sparsity tensor L1 placement and lifecycle
- `ch05_sparsity_tensor_construction/constructing_from_router_output.md` — per-step construction and update requirements

---

**Next:** [Chapter 7 — T3K Multi-Chip MoE](../ch07_t3k_multi_chip_moe/index.md)
