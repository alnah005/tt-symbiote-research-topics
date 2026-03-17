# Performance Profile: Batched Matmul for MoE

This file characterizes the runtime behavior of the batched matmul strategy: where time is spent, how utilization scales with expert capacity, when the approach is compute-bound versus memory-bound, and what the primary bottlenecks are. It concludes with a brief pointer to Chapter 4's `sparse_matmul` for regimes where batched matmul is not the right choice.

---

## 1. Latency Breakdown

A single MoE layer using batched matmul decomposes into three sequential phases:

```
[1. Gather]  →  [2. Expert FFN matmul]  →  [3. Scatter + combine]
```

### 1.1 Phase 1: Gather

Gather is a strided DRAM read — see `formulating_batched_matmul.md` §1 Warning — with quantified buffer sizes:

> **Note:** The following examples assume a T3K expert-parallel configuration with 256 total experts distributed across 8 devices (32 local experts per device); on a single device with all 256 experts, figures scale by 8×.

At decode ($B=32$, $S=1$, $C=1$): gather output buffer per device = $32 \times 1 \times 7168 \times 2 = 458{,}752$ bytes ≈ **0.46 MB**. At prefill ($B=1$, $S=2048$, $C=64$): gather output buffer per device = $32 \times 64 \times 7168 \times 2 = 29{,}360{,}128$ bytes ≈ **29.4 MB**, and can become a significant fraction of total layer latency.

### 1.2 Phase 2: Expert FFN Matmul

The matmul time covers three batched GEMM calls (gate, up, down projections) plus the element-wise SwiGLU between them. For a single expert's gate projection:

- FLOP count: $2 \times C \times H \times D$ [D UNVERIFIED — verify against Qwen3 Technical Report]
- Weight bytes read from DRAM: $H \times D \times \text{bytes\_per\_BFP8}$ [D UNVERIFIED]
- Activation bytes read: $C \times H \times 2$ (BF16)
- Output bytes written: $C \times D \times 2$ (BF16) [D UNVERIFIED]

Summed across all $E=256$ experts and all three projections, the matmul phase accounts for the majority of total layer compute at high $C$ (prefill regime).

### 1.3 Phase 3: Scatter and Combine

Scatter reads from the `[E, C, H]` output buffer and writes to the `[B×S, H]` hidden state buffer, weighted by router scores. The access pattern is the inverse of gather: strided reads from the expert dimension, non-contiguous writes to the token dimension.

Scatter cost scales similarly to gather: $O(k \times B \times S \times H)$ bytes moved, with write scatter determined by routing distribution.

### 1.4 Relative Magnitude by Regime

| Regime | Gather+Scatter fraction | Matmul fraction | Notes |
|--------|------------------------|-----------------|-------|
| Decode ($B=32$, $S=1$) | Moderate (few KB, but DRAM latency) | Low absolute time; efficiency limited by $C=1$ padding | Padding waste dominates utilization |
| Prefill ($B=1$, $S=2048$) | Small relative fraction | Large (high $C$, most compute time here) | Gather of 2048-token sequence is non-trivial |
| Prefill ($B=32$, $S=2048$) | Moderate to large | Dominant | 65536-token aggregate gather can rival matmul time |

> **Tip:** Profile gather and scatter separately from the matmul using `ttnn.device.profiler` or Tracy traces. When gather+scatter time exceeds 20% of total layer latency, investigate whether the gather can be fused with the routing op or overlapped with DRAM prefetching of weight tensors.

---

## 2. FLOP Efficiency vs. Expert Capacity

For the FLOP efficiency formula and decode worked example, see `formulating_batched_matmul.md` Section 2.3; the key new observation here is the tile-level granularity effect:

### 2.2 Why Decode is Particularly Inefficient

At decode ($B=32$, $S=1$, $k=8$, $E=256$): $C=1$, and the 256 (token, expert) assignments cover each expert exactly once on average. Under uniform routing, efficiency = 1.0. However:

1. Since $C=1 < 32$, the actual padded capacity in tile layout is $C_\text{pad}=32$. The tile-level FLOP efficiency drops to $1/32 \approx 3\%$ — 31 of 32 rows in every expert's tile are padding zeros.
2. Under non-uniform routing (which is typical), some experts receive 0 tokens and others receive 2+, but all are padded to $C_\text{pad}=32$. The effective utilization can be below 3%.

This is the primary reason `sparse_matmul` (Chapter 4) is preferred for decode: it skips zero tiles explicitly rather than computing them.

---

## 3. Arithmetic Intensity and Compute vs. Memory Bound

The arithmetic intensity of the expert FFN determines whether the matmul is compute-bound (FPU is the bottleneck) or memory-bound (DRAM bandwidth is the bottleneck).

For a single expert's gate or up projection ($[C, H] \times [H, D]$):

$$\text{FLOPs} = 2 \times C \times H \times D \quad \text{[D UNVERIFIED — verify against Qwen3 Technical Report]}$$

Weight bytes read (BFP8, one-time DRAM read per matmul):

$$\text{weight bytes} = H \times D \times \text{bytes\_per\_BFP8} \approx H \times D \times 1.0 \quad \text{[D UNVERIFIED]}$$

(BFP8 stores 1 byte per element for the mantissa plus a shared exponent per 16 elements, giving approximately 1.0625 bytes/element (= 1088/1024 = 17/16); for estimation purposes use 1.0 byte/element.)

Activation bytes read:

$$\text{activation bytes} = C \times H \times 2 \quad \text{(BF16)}$$

Arithmetic intensity (FLOPs per byte of memory traffic), including the output write $C \times D \times 2$ bytes (BF16):

$$\text{AI} = \frac{2 \times C \times H \times D}{H \times D + C \times H \times 2 + C \times D \times 2} = \frac{2CHD}{HD + 2CH + 2CD} \quad \text{[D UNVERIFIED — verify against Qwen3 Technical Report]}$$

(Denominator: BFP8 weight read $H \times D \times 1$ bytes + BF16 activation read $C \times H \times 2$ bytes + BF16 output write $C \times D \times 2$ bytes.)

At large $C$ (prefill, $C \gg HD/(H+D)$): the denominator $HD + 2CH + 2CD \approx 2C(H+D)$, so:

$$\text{AI} \to \frac{2CHD}{2C(H+D)} = \frac{HD}{H+D} \quad \text{[D UNVERIFIED]}$$

The intensity saturates at $HD/(H+D)$, independent of $C$. When $D \ll H$, this approximates $D$; when $D \approx H$, it approximates $H/2$.

At small $C$ (decode, $C=1$): $\text{AI} \approx \frac{2HD}{HD + 2H + 2D} \approx \frac{2HD}{HD} = 2$ [D UNVERIFIED] — the weight read dominates ($HD \gg 2H + 2D$ for large $H$, $D$) and intensity approaches 2 FLOPs/byte regardless of $D$.

**Memory-bound threshold:** The Wormhole B0 peak DRAM bandwidth is approximately 384 GB/s and peak FPU throughput is approximately 262 TFLOPS (BFP8). The machine balance point is roughly $\text{AI}_\text{ridge} = 262 \times 10^{12} / (384 \times 10^9) \approx 682$ FLOPs/byte. Operations below this threshold are memory-bound; above it, compute-bound.

At decode ($C=1$, $\text{AI} \approx 2$): heavily **memory-bound** — the weight tensor must be read from DRAM but the FPU can consume it far faster than DRAM can supply it.

At large prefill, as $C \to \infty$, $\text{AI} \to HD/(H+D)$ [D UNVERIFIED]. Whether this regime is compute-bound depends on verifying $D$ against the hardware ridge point ($\approx 682$ FLOPs/byte). The specific numeric claim AI ≈ 4096 at C=2048 has been removed pending confirmation of $D$.

> **Tip:** The memory-bound decode regime is precisely where BFP8 weight compression provides its greatest benefit: halving weight bytes (versus BF16) approximately doubles effective memory bandwidth for the weight-read-limited regime. Always use BFP8 for expert weights in decode.

---

## 4. Known Bottlenecks

### 4.1 Padding Waste at Low Load (Decode: C=1)

As discussed in Section 2.2, $C=1$ with tile padding to 32 gives tile-level FLOP efficiency of approximately $1/32$. The FPU spends most of its time processing zero tiles. The kernel still dispatches, runs the full K-reduction loop, and writes the full output buffer — all for one real token per expert. This is the dominant limitation of batched matmul at decode.

**Mitigation:** Use `sparse_matmul` (Chapter 4) for decode. Batched matmul is not the right kernel strategy when $C < 32$ per expert.

### 4.2 Gather Cost on DRAM

Gather is a random-access DRAM read pattern: each token in `hidden_states` must be read and written to a non-contiguous slot in the `[E, C, H]` expert buffer. This produces poor DRAM burst utilization.

**Mitigation:** At small total tokens (decode, $C=1$, 32 local experts), the gather output buffer is ≈ 0.46 MB (32 × 1 × 7168 × 2 bytes) and absolute gather time is low. At large total tokens (prefill, $C=64$, 32 local experts), the gather output buffer is ≈ 29.4 MB (32 × 64 × 7168 × 2 bytes) and must be carefully profiled. A fused router + gather kernel that writes directly to the expert buffer in a single pass over the hidden state tensor can eliminate the intermediate random-write pass.

### 4.3 Recompilation on Shape Change

Any change in $B$, $S$, or $C$ invalidates the program cache and triggers TTNN recompilation. Recompilation adds hundreds of milliseconds on first call. In dynamic batching scenarios (variable-length sequences), this can cause latency spikes.

**Mitigation:** Maintain a config cache indexed by $(B, S, C)$ and pre-compile for the expected working set of shapes at deployment time. For serving systems with variable batch sizes, define a small set of canonical shapes (e.g., powers of two for $B$) and pad inputs to the nearest canonical shape before dispatch.

### 4.4 Expert Imbalance and Capacity Overflow

When routing is imbalanced, hot experts overflow their capacity $C$ and tokens are dropped; cold experts waste their allocated capacity. Both reduce effective FLOP efficiency.

**Mitigation:** Increase $C$ by raising the capacity factor $\alpha > 1.0$ (e.g., $\alpha = 1.25$) to absorb routing imbalance without token drops. This increases memory footprint and total FLOPs proportionally, trading compute efficiency for robustness.

---

## 5. When Batched Matmul is Preferred

Batched matmul is the correct strategy when the following conditions hold:

| Condition | Reason |
|-----------|--------|
| High batch size ($B \geq 16$) | Large total tokens → higher $C$ → better FLOP efficiency |
| Long sequence length ($S \geq 128$) | $C$ scales with $S$; prefill with long sequences gives $C \geq 4$ |
| Balanced routing | Uniform routing maximizes fill rate; imbalanced routing wastes capacity |
| Static shapes or small shape working set | Program cache reuse across forward passes; no recompilation overhead |
| Expert weight matrix fits in DRAM | No per-expert weight eviction needed; weights are cold-read from DRAM per forward pass |

Batched matmul is **not** the preferred strategy when:

| Condition | Better Alternative |
|-----------|-------------------|
| Decode ($C=1$, high padding waste) | `sparse_matmul` (Chapter 4) — skips zero tiles explicitly |
| Highly imbalanced routing | `sparse_matmul` — processes only filled tiles; immune to capacity waste |
| Very small $B \times S$ with dynamic shapes | Per-token matmul loop with tracing per unique shape (niche) |

---

## 6. Contrast with Chapter 4: Sparse Matmul

Chapter 4 introduces `ttnn.sparse_matmul`, a kernel variant that accepts a tile-level sparsity mask and skips tiles that are zero. For MoE workloads, the sparsity mask encodes which token slots in the `[E, C, H]` activation tensor contain real (non-padded) tokens.

The high-level trade-off is:

- **Batched matmul:** Simple, no sparsity bookkeeping, excellent at high $C$. Performance degrades gracefully with $C$ but wastes compute on zero tiles when $C$ is small.
- **Sparse matmul:** Avoids zero-tile compute, well-suited to decode ($C=1$) and imbalanced routing. Adds sparsity tensor construction and management overhead; performance is non-monotonic in sparsity ratio (metadata overhead at low sparsity).

A detailed comparison — including the sparsity ratio threshold at which `sparse_matmul` overtakes batched matmul — is provided in Chapter 4 and synthesized in Chapter 6.

> **Tip:** A practical hybrid strategy uses batched matmul for prefill (high $C$, high fill rate) and switches to `sparse_matmul` for decode (low $C$, high padding waste). This is discussed as a concrete deployment pattern in Chapter 6 (`decision_guide.md`).

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Latency phases** | Gather (DRAM strided read) → expert FFN matmul × 3 → scatter+combine |
| **FLOP efficiency** | Scales with $k \times B \times S / (C \times E)$; approaches 1.0 for uniform routing at prefill scale |
| **Tile efficiency** | Drops to $\sim 1/32$ at decode ($C=1$) due to tile-boundary padding |
| **Arithmetic intensity** | $\approx 2$ FLOPs/byte at decode (memory-bound); $\to HD/(H+D)$ FLOPs/byte as $C \to \infty$ (constant, not growing with $C$; when $D \ll H$ this approximates $D$) — [D UNVERIFIED — verify against Qwen3 Technical Report; compute-bound conclusion requires confirming $HD/(H+D)$ > ridge point] |
| **Primary decode bottleneck** | Padding waste ($C=1$ → 31 zero tile rows per expert per tile) |
| **Primary prefill bottleneck** | Gather cost at large $B \times S$; and DRAM weight reads at moderate $C$ |
| **Preferred regime** | High batch size, long sequence length, balanced routing (prefill) |
| **Not preferred** | Decode with small $C$; highly imbalanced routing — use Chapter 4's `sparse_matmul` |

---

## Next Steps

You have completed Chapter 3. The batched matmul strategy, its tensor shapes, program config selection, and performance profile are now fully characterized.

Proceed to **Chapter 4 — sparse_matmul for MoE** to learn about the alternative kernel strategy that outperforms batched matmul in the decode regime by explicitly skipping zero tiles in the activation tensor.
