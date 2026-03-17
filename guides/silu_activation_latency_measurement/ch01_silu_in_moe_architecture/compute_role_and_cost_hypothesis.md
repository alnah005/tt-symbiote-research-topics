# Compute Role and Cost Hypothesis

This file explains why activation functions are traditionally treated as free compute in ML performance analysis, why that assumption breaks down on SFPU-based architectures like Tenstorrent Wormhole, and frames the preliminary cost hypothesis for SiLU in MoE inference.

---

## Why Activations Are Traditionally "Free"

In GPU-based transformer inference, the FFN bottleneck is almost always the matrix multiplications (`W_gate`, `W_up`, `W_down`). Activations like ReLU and SiLU are element-wise operations that:

1. Operate on a single pass over one tensor (no second operand tensor to load).
2. Run at memory bandwidth speeds — their throughput is limited by how fast data can be loaded from and stored to memory, not by compute.
3. Are often fused into the preceding matmul kernel by the compiler or hand-written CUDA kernel, so they add zero additional kernel launch overhead.

On an NVIDIA A100 with 2TB/s HBM bandwidth, applying SiLU to a `[1, 14336]` row takes approximately 14336 * 4 bytes / 2e12 ≈ 29 nanoseconds (~28.7 ns) — negligibly small compared to the matmul it follows. This is why most transformer performance models ignore activation time entirely.

---

## SFPU: A Different Execution Model

Tenstorrent Wormhole uses a fundamentally different compute architecture. Each Tensix core contains two distinct functional units:

| Unit | Purpose | Instructions | Data path |
|---|---|---|---|
| FPU / Math engine | Matrix multiply, element-wise add/mul | FMAC, dot products | Operates on full tiles (32x32) in parallel |
| SFPU | Non-linear functions: exp, log, sqrt, reciprocal, sigmoid, tanh | Sequential per-element SFPU instructions | Operates on one datum at a time within a tile |

The SFPU is not a vectorized SIMD unit in the traditional sense. It processes elements within a tile sequentially, and the throughput for non-linear functions is substantially lower than the FPU throughput for linear operations.

### SFPU Instruction Cost for SiLU

SiLU requires evaluating `x * sigmoid(x)` (see [`swiglu_variant.md`](swiglu_variant.md) for the full definition). On Wormhole, evaluating `sigmoid(x)` requires the following SFPU operations per element:
1. Negate: `-x`
2. Exp: `exp(-x)` — typically implemented via SFPU lookup table + polynomial correction
3. Add: `1 + exp(-x)`
4. Reciprocal: `1 / (1 + exp(-x))` — another SFPU non-linear op
5. Multiply: `x * sigmoid(x)`

Compare with ReLU:
```
ReLU(x) = max(0, x)
```
ReLU requires a single SFPU clamp (or conditional select) instruction. No exp, no reciprocal.

Approximate relative SFPU cycle cost (order-of-magnitude estimates based on Wormhole instruction timing):

| Activation | Approx. SFPU ops per element | Relative cost vs ReLU |
|---|---|---|
| ReLU | 1 (clamp) | 1× |
| SiLU | 5 (neg, exp, add, recip, mul) | ~5–8× |
| GELU (tanh approx) | 4–6 (mul, tanh, add, mul, mul) | ~5–8× |

> **Warning:** These are rough estimates. Actual SFPU cycle counts depend on the specific Wormhole firmware version, tile format (bfloat16 vs float32), and SFPU instruction scheduling within the Tensix core's compute pipeline. Chapter 3 measurements will replace these estimates with empirical data.

---

## The Activation-Matmul Overlap Question

Even if SiLU is more expensive per element than ReLU, it may still be "free" if the SFPU work can be hidden behind concurrent computation. There are two relevant scenarios:

### Scenario A: Fused SiLU inside matmul kernel

When `ttnn.matmul` is configured with `activation="silu"`, the SFPU instructions execute after each partial sum accumulation, interleaved with the math engine pipeline. In this case:

- If the matmul is FPU-bound (large M, N, K), the SFPU work for SiLU may be fully hidden.
- If the matmul is memory-bandwidth-bound (small M, as in single-token decode or small expert dispatch), the pipeline stall from SFPU instructions can become visible.

### Scenario B: Standalone SiLU after matmul

When `ttnn.silu` runs as a separate kernel:

- The matmul result must be written to L1 SRAM and then re-read by the SiLU kernel.
- The SiLU kernel dispatches SFPU instructions with no concurrent FPU work.
- Wall-clock cost = SFPU execution time + two L1 read/write passes over the tensor.

For large tensors (prefill with T=2048, d_ffn=14336), the L1 working set may exceed per-core SRAM and require off-chip DRAM access, adding memory latency.

---

## Cost Hypothesis for MoE SiLU

Based on the above analysis, the preliminary cost hypothesis is:

**Hypothesis H1 (Standalone SiLU cost is non-negligible in MoE decode):**

During MoE decode (T=1 or small T), the per-expert matmuls are small (e.g., `[1, 4096] × [4096, 14336]`), making them bandwidth-bound rather than compute-bound. In this regime, the standalone `ttnn.silu` kernel may represent a measurable fraction of per-expert FFN latency — potentially 10–30% — because the SFPU throughput advantage is lost and kernel dispatch overhead is not amortized over large work.

**Hypothesis H2 (Fused SiLU recovers overlap during prefill):**

During MoE prefill (T=512 or larger), the per-expert matmuls are large enough to be FPU-bound. Fused SiLU is expected to add near-zero wall-clock overhead in this regime. Standalone SiLU will add a measurable but sub-dominant cost.

**Hypothesis H3 (Fine-grained MoE amplifies SiLU overhead):**

Models like Qwen2-MoE (64 experts, top-8) and DeepSeek-V2 (160 experts, top-6) have smaller per-expert `d_ffn_expert` values (e.g., 2048 vs 14336 in Mixtral). Each expert matmul is smaller and more bandwidth-bound, making the SFPU overhead fraction larger per call. The increased number of SiLU invocations (8× vs 2×) further amplifies the absolute SiLU contribution to layer latency.

---

## Framing the Measurement Goal

The hypotheses above define the measurement objectives for this guide:

| Measurement | Addresses | Expected outcome |
|---|---|---|
| Standalone `ttnn.silu` latency vs tensor size | H1 | Non-linear scaling; overhead visible at small M |
| Fused vs standalone `ttnn.matmul(..., activation="silu")` | H1, H2 | Fused faster for large T; comparable for small T |
| SiLU vs ReLU latency at matched tensor sizes | Calibration | SiLU ~5–8× slower per element on SFPU |
| Per-expert SiLU latency across expert sizes | H3 | Small experts show higher SiLU fraction |
| Total SiLU share of MoE layer latency | H1, H3 | Quantified as % of end-to-end FFN time |

*See Chapter 2, `measurement_setup.md` for the benchmark harness design and how these measurements are structured.*

---

---

**Next:** [Chapter 2 — SiLU on Wormhole Hardware](../ch02_silu_on_wormhole_hardware/index.md)
