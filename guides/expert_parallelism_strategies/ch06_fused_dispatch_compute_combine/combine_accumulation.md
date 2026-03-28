# Combine Accumulation

## Section 1: Input to Accumulation

After the combine all-to-all (Stage 4) completes, each device holds the expert
outputs for all $B$ tokens that originated on that device. Logically, this is a
three-dimensional buffer:

$$\text{expert out}[B, k, H] \quad \text{(BF16)}$$

where:
- $B$ is the batch size of tokens originating on this device.
- $k = 8$ is the number of experts selected per token.
- $H = 7168$ is the hidden dimension.

The $k = 8$ expert outputs for token $b$ arrived from up to $k$ different devices
during the combine all-to-all. From the router (Stage 1), this device also retained
the raw sigmoid routing scores:

$$\text{scores}[B, k] \quad \text{(BF16 or FP32)}$$

These scores were produced locally and never transmitted over the network; they have
been in local memory since Stage 1 completed. Stage 5 uses them to weight-combine the
expert outputs.

The two inputs to Stage 5 are therefore:

| Tensor | Shape | Size at $B=32$ |
|---|---|---|
| `expert_out` | $[B, k, H]$ | $32 \times 8 \times 7168 \times 2 = 3.67$ MB |
| `scores` | $[B, k]$ | $32 \times 8 \times 2 = 512$ bytes |
| Output buffer | $[B, H]$ | $32 \times 7168 \times 2 = 448$ KB |

## Section 2: Weighted Scatter-Add

### Formula

The accumulation computes a weighted sum of the $k = 8$ expert outputs for each token.
Let $w_{b,j}^{\text{raw}} = \text{scores}[b, j]$ be the raw sigmoid score for token
$b$, expert slot $j$. The normalized weight is:

$$w_{b,j}^{\text{norm}} = \frac{w_{b,j}^{\text{raw}}}{\sum_{l=1}^{k} w_{b,l}^{\text{raw}}}$$

The output for token $b$ is:

$$\text{output}[b, :] = \sum_{j=1}^{k=8} w_{b,j}^{\text{norm}} \cdot \text{expert out}[b, j, :]$$

This is a weighted sum over $k = 8$ vectors of length $H = 7168$.

### Parallelization Strategy

**Key design rule:** Assign each token $b$ to a dedicated core. That core handles all
$k = 8$ accumulations for token $b$ entirely within its own L1.

This assignment eliminates race conditions. Because each token $b$ has exactly one
writer for its output slot `output[b, :]`, there is never a conflict between cores.
In contrast, if the parallelization were over the $k = 8$ expert slots for a given
token (multiple cores writing to the same `output[b, :]`), a reduction or atomic
operation would be required.

**Core assignment at $B = 32$:**

With 80 Tensix cores and 32 tokens:

$$\text{tokens per core} = \lceil 32 / 80 \rceil = 1 \quad \text{(32 cores active, 48 idle)}$$

Or equivalently, assign 1–2 tokens per core using a round-robin assignment to keep
the active core set small and L1-resident:

```python
# Conceptual assignment (actual TTNN dispatch uses sharding)
tokens_per_core = math.ceil(B / num_active_cores)
for core_id in range(num_active_cores):
    token_start = core_id * tokens_per_core
    token_end = min(token_start + tokens_per_core, B)
    # This core accumulates output[token_start:token_end, :]
```

**Core assignment at $B = 1$ (single-token decode):**

Only 1 core is needed. It loads `expert_out[0, 0:8, :]` (8 vectors of length 7168)
and computes the weighted sum in L1. Total data: $8 \times 7168 \times 2 = 114{,}688$
bytes = 112 KB — fits comfortably within the 1.5 MB per-core L1.

## Section 3: L1 vs. DRAM Accumulation

The output buffer size scales with $B$:

$$|\text{output}[B, H]| = B \times 7168 \times 2 \text{ bytes (BF16)}$$

| $B$ | Output buffer size | Recommendation |
|---|---|---|
| 1 | 14,336 bytes = 14 KB | L1 accumulation (trivially fits) |
| 32 | 458,752 bytes = 448 KB | L1 accumulation with `HEIGHT_SHARDED` |
| 64 | 917,504 bytes ≈ 896 KB | L1 accumulation across 2 cores (HEIGHT_SHARDED) |
| 256 | 3,670,016 bytes ≈ 3.5 MB | DRAM accumulation; L1 too small |
| 2048 | 28,311,552 bytes ≈ 27 MB | DRAM required |

For decode batch sizes ($B \leq 32$), the output buffer fits entirely in L1 under
HEIGHT_SHARDED layout. L1 accumulation avoids DRAM write-back latency and keeps the
residual add (Stage 6) in L1 as well.

For prefill ($B \geq 256$), use DRAM-backed output buffers. The accumulation is
compute-bound at those batch sizes, so the extra DRAM latency is amortized.

> **Tip:** At $B = 32$ with FP32 accumulators (see Section 5), the output buffer
> doubles to 896 KB, which still fits within the aggregate L1 of a HEIGHT_SHARDED
> allocation across 2 cores.

## Section 4: Overlap with Residual Add

Stage 6 computes:

$$\text{output}[b, :] \mathrel{+}= \text{residual}[b, :]$$

where `residual` is the input embedding passed unchanged around the MoE sublayer.
This addition can be fused with the final pass of Stage 5 to eliminate a separate
kernel launch and an intermediate tensor write.

**Fused kernel (Stage 5 + Stage 6 combined):**

For each token $b$ assigned to this core:

```
for h in range(H):
    acc = 0.0  # FP32 accumulator
    for j in range(k):  # k = 8
        acc += w_norm[b, j] * expert_out[b, j, h]
    output[b, h] = acc + residual[b, h]  # fused residual add
```

The fusion saves one read of the residual tensor and one write of an intermediate
accumulation buffer. Both tensors (`expert_out[b, :, h]` and `residual[b, h]`) can
be kept in L1 registers during the inner loop.

> **Warning:** The fused kernel requires `residual` to be available in L1 or at
> least in DRAM with low-latency read access when Stage 5 begins. Ensure the residual
> tensor is not evicted from L1 by the combine all-to-all buffer traffic (Stage 4)
> before Stage 5 starts. Pre-stage the residual in a dedicated L1 region if needed.

## Section 5: Numerical Precision

BF16 has a unit roundoff of $\varepsilon_{\text{BF16}} = 2^{-7} \approx 0.0078$.
When accumulating $k = 8$ terms in BF16, the accumulated relative error is bounded by:

$$\text{relative error} \leq k \times \varepsilon_{\text{BF16}} = 8 \times 2^{-7} = 2^{-4} = 0.0625 \approx 6.25\%$$

This is a very conservative (loose) upper bound. In practice, cancellations between
positive and negative intermediate errors mean actual relative errors are far smaller,
typically well below 0.1% for routing weight accumulations where all weights are
positive and sum to 1.

**FP32 accumulator option:**

For applications where numerical precision is critical (e.g., quantization-sensitive
inference or when matching exact FP32 reference outputs), use a FP32 intermediate
accumulator:

```python
# Pseudocode: FP32 accumulation, BF16 output
for b in range(B):
    acc = torch.zeros(H, dtype=torch.float32)  # FP32 accumulator in L1
    for j in range(k):
        acc += w_norm[b, j].float() * expert_out[b, j, :].float()
    output[b, :] = acc.bfloat16()  # cast back to BF16 for downstream layers
```

FP32 accumulator memory at $B = 32$:

$$32 \times 7168 \times 4 = 917{,}504 \text{ bytes} \approx 896 \text{ KB}$$

This still fits within L1 under a HEIGHT_SHARDED allocation across 2 cores (each
holding 448 KB of the accumulator), making FP32 accumulation practical even at decode.

| Accumulator type | Error bound | Output buffer at $B=32$ | L1-resident? |
|---|---|---|---|
| BF16 | $\leq 6.25\%$ (loose) | 448 KB | Yes (single core) |
| FP32 | $\leq 0.05\%$ (estimate) | 896 KB | Yes (2 cores HEIGHT_SHARDED) |

> **Note:** The 6.25% error bound is a mathematical worst case. Empirical measurements
> on Qwen-class MoE models show BF16 accumulation errors below 0.2% relative to FP32
> reference for typical routing weight distributions, making BF16 accumulation
> acceptable for standard inference.

## References

- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_combine.md`
  (combine all-to-all output buffer layout).
- Chapter 5 of this guide: `ch05_routing_weight_optimization/router_kernel_fusion.md`
  (routing score normalization and storage).
- TT-Metalium TTNN documentation: `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, BF16
  precision characteristics.
- Wormhole B0 Architecture Guide: per-core L1 (1.5 MB), aggregate L1 (120 MB).

---

**Next:** [end_to_end_latency_model.md](./end_to_end_latency_model.md)
