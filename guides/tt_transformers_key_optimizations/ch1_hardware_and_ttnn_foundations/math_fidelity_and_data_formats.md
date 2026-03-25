# Math Fidelity and Data Formats

This file explains Tenstorrent's math fidelity system — the hardware mechanism that trades numeric precision for throughput in matrix operations — and the `WormholeComputeKernelConfig` object that controls it. By the end of this file you will understand why tt-transformers uses different fidelity settings for QKV projections vs MLP layers, what `packer_l1_acc` does and when it helps, and how to make the LoFi / HiFi2 / HiFi4 decision for your own model layers.

---

## The Hardware Multiplier: 5-bit × 7-bit

At the lowest level, the Wormhole matrix FPU implements a **5-bit × 7-bit integer multiplier** in its matrix arithmetic unit. Floating-point multiplication is implemented by feeding mantissa bits through this integer multiplier in one or more passes, then reconstructing the floating-point result.

For BF16 operands (7-bit mantissa each side), the mantissa product is 7 × 7 = 49 bits of combined significance. The hardware multiplier uses **5-bit × 7-bit operands, yielding a 12-bit product per pass** (5 + 7 = 12 output bits). To cover the full 7-bit × 7-bit mantissa multiplication at full precision, the hardware performs multiple passes, each contributing partial products at successively lower significance. The exact pass count per fidelity level is defined by the Tenstorrent hardware: LoFi = 1 pass, HiFi2 = 2 passes, HiFi3 = 3 passes, HiFi4 = 4 passes. The higher the pass count, the more of the mantissa significance range is covered, and the closer the result is to a full BF16 multiply. Fewer passes = fewer mantissa bits covered = lower precision, but **proportionally higher throughput**.

This multi-pass mechanism is the basis of the **math fidelity** system.

---

## Math Fidelity Levels

TTNN defines four fidelity levels for matmul operations. Each level corresponds to a fixed number of multiplier passes per output element:

| Fidelity | Passes | Mantissa bits covered | Relative throughput (matmul) | Notes |
|---|---|---|---|---|
| `LoFi` | 1 | Highest-significance ~3–4 bits | ~3.5× vs HiFi4 | Fastest; significant precision loss for small values |
| `HiFi2` | 2 | Upper mantissa range; empirically sufficient for BFP8 × BF16 (see Key Pairing Rule below) | ~2× vs HiFi4 | Best throughput/accuracy tradeoff for BFP8/BFP4 weight layers |
| `HiFi3` | 3 | ~5–6 bits with additional partial-product coverage | ~1.5× vs HiFi4 | Rarely used in practice |
| `HiFi4` | 4 | All 7 mantissa bits (full BF16) | 1× (baseline) | Full BF16 precision; needed for attention scores in prefill |

> The Wormhole peak throughput numbers from [tensix_architecture.md](./tensix_architecture.md) — LoFi ~262 TOPS, HiFi2 ~148 TOPS, HiFi4 ~74 TOPS — directly reflect this 1×/2×/3.5× throughput scaling.

### What "Passes" Mean for Accuracy

In LoFi mode, the 5-bit sub-multiplier operates on a single pass: one operand contributes its top 5 mantissa bits (the bottom 2 bits of its 7-bit mantissa are dropped), while the other operand retains its full 7-bit mantissa. The effective multiply is therefore 5-bit × 7-bit — not 5×5 — consistent with the hardware multiplier definition above. The specific assignment of which operand maps to the 5-bit side versus the 7-bit side is not definitively confirmed in public documentation; the practical guideline is that LoFi is safe when the weight has low precision (BFP4, which carries only a 3-bit mantissa), regardless of activation precision — the weight's quantization already constrains the representable values far below what the 5-bit truncation affects. For large, well-scaled values this causes negligible error. For values that differ significantly in magnitude within the same tile (common in attention score computation), the error can accumulate.

In HiFi4, all 7 mantissa bits participate across four passes, yielding results that match standard BF16 multiplication within normal floating-point rounding behavior.

HiFi2 covers approximately the top half of the mantissa bits across its two passes and is empirically validated as sufficient for BFP8 × BF16 workloads — for the full rationale, see the Key Pairing Rule section below.

---

## `WormholeComputeKernelConfig`

TTNN provides `ttnn.WormholeComputeKernelConfig` (and its alias `ttnn.GrayskullComputeKernelConfig` for older hardware) to configure the compute kernel behavior for a specific op invocation. The four most important fields are:

### Fields

```python
ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,   # LoFi, HiFi2, HiFi3, or HiFi4
    math_approx_mode=False,                   # Use polynomial approx for SFPU transcendentals
    fp32_dest_acc_en=False,                   # Use FP32 accumulation in Dst register
    packer_l1_acc=False,                      # Accumulate Packer outputs in L1 (L1 += Dst)
)
```

#### `math_fidelity`

Controls the number of multiplier passes as described above. This is the primary throughput dial.

#### `math_approx_mode`

When `True`, SFPU operations (`exp`, `rsqrt`, `softmax`, `tanh`) use polynomial approximations instead of iterative hardware algorithms. This reduces SFPU latency at the cost of slightly reduced accuracy.

- **Use `True` for**: RMSNorm (`rsqrt`), layer norm, GELU/SiLU activation functions where the accuracy delta is within the model's tolerance
- **Use `False` for**: softmax in attention score normalization when full precision is required

#### `fp32_dest_acc_en`

When `True`, the Dst register accumulates in FP32 rather than BF16. This prevents accumulated rounding error across many tile additions (important when summing many partial products in a large matmul).

**Trade-off**: enabling FP32 accumulation halves Dst capacity from 8 tiles to 4 tiles (see [tensix_architecture.md](./tensix_architecture.md) — Dst Register File), which may require reducing the output subblock dimensions.

- **Use `True` for**: high-precision accumulation in attention score computation, or matmuls where the input dynamic range is large
- **Use `False` for**: most inference scenarios where BF16 accumulation error is absorbed by the model's robustness

#### `packer_l1_acc`

This flag enables **L1-accumulation mode** in the Packer unit. It applies when the output buffer is in L1. Without `packer_l1_acc`, the Packer performs a read-modify-write each time it updates the output L1 buffer: it reads the existing partial sum from L1, adds the Dst tile, and writes the result back — an extra L1 read per K-block write event. With `packer_l1_acc=True`, the Packer instead **adds** the current Dst tile directly onto the value already at the output L1 address in-place: `L1[output_addr] += Dst[tile]`, eliminating that read-modify-write overhead.

This is a **throughput optimization**, not a correctness requirement — TTNN kernels correctly accumulate K-blocks without this flag (using the Dst register), but the read-modify-write overhead becomes a significant bottleneck for large-K matmuls. With `packer_l1_acc=True`, the Packer accumulates in-place and the full accumulated result is built in L1 across iterations.

---

## `packer_l1_acc` in Depth: Multi-Tile Accumulation

Note: `in0_block_w` (the K-dimension tile block size in the program config) is constrained by L1 storage capacity for input tiles in the SrcA/SrcB circular buffers, not by Dst capacity; Dst capacity separately constrains `out_subblock_h × out_subblock_w`. The `packer_l1_acc` flag has no effect on either of those constraints.

**When `packer_l1_acc` helps most:**

- Large K-dimension matmuls (MLP FF1/FF2 with hidden dim 14336 in Llama 3 70B)
- `in0_block_w` set to a larger value than Dst can hold in a single pass
- The output buffer is in L1 (not DRAM) — the accumulation requires the output address to be writable at L1 speeds

---

## Using `WormholeComputeKernelConfig` with TTNN Ops

Most TTNN matmul ops accept a `compute_kernel_config` argument:

```python
import ttnn

# HiFi2 config for MLP weight matmul with BFP8 weights
mlp_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

output = ttnn.matmul(
    activation,
    weight,
    program_config=matmul_program_config,   # see Chapter 3
    memory_config=output_memory_config,
    dtype=ttnn.bfloat16,
    compute_kernel_config=mlp_compute_config,
)

# LoFi config for BFP4 weight matmul in FF1/FF3
ff1_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# HiFi4 config for attention score accumulation in prefill
attention_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,   # full FP32 accumulation for long-context accuracy
    packer_l1_acc=False,     # attention output not multi-step accumulated
)
```

---

## Decision Guide: Which Fidelity for Which Layer?

The following table summarizes the recommended starting configuration for each layer type in a typical transformer LLM:

| Layer | Weight dtype | Recommended fidelity | `fp32_dest_acc_en` | `packer_l1_acc` | Rationale |
|---|---|---|---|---|---|
| QKV projection (decode) | BFP8_B | HiFi2 | False | True | BFP8 weight has 7-bit mantissa; HiFi2 covers the meaningful bits; packer_l1_acc helps large K |
| QKV projection (prefill) | BFP8_B | HiFi2 | False | True | Same as decode; prefill is compute-bound so throughput gain matters more |
| Output projection `wo` (decode) | BFP8_B | HiFi2 | False | True | Same reasoning as QKV |
| Attention score QK^T (prefill) | BF16 activations | HiFi4 | True | False | Long-context softmax accuracy sensitive; FP32 accumulation avoids softmax exp overflow |
| Attention score QK^T (decode) | BF16 activations | HiFi2 | False | False | Single-token, sequence axis short; HiFi2 sufficient |
| MLP FF1 / FF3 (SwiGLU gate) | BFP4_B | LoFi | False | True | BFP4 has only 3-bit mantissa; LoFi is matched to this precision; LoFi gives ~3.5× throughput |
| MLP FF2 | BFP4_B or BFP8_B | LoFi or HiFi2 | False | True | Match fidelity to weight dtype |
| RMSNorm / LayerNorm | BF16 activations | Any (fidelity does not affect SFPU ops) | False | False | Element-wise only (SFPU — rsqrt, exp, etc.); math fidelity controls the matmul multiplier pass count and has no effect on SFPU operations. The only meaningful performance knob for RMSNorm/LayerNorm is `math_approx_mode=True`, which enables polynomial approximations for rsqrt. |

### Key Pairing Rule

> **The fidelity level should be matched to the number of meaningful mantissa bits in the lower-precision operand.**
>
> - BFP4 weight (3-bit mantissa) → LoFi (1 pass, 5 bits) is more than sufficient
> - BFP8 weight (7-bit mantissa) paired with BF16 activation → HiFi2 (2 passes) is empirically validated as sufficient for model accuracy (per Tenstorrent's PERF.md). Note: BFP8_B has the same per-value mantissa width as BF16 (7 bits each) — the distinction is that BFP8_B uses a shared block exponent across 16 values. The shared exponent means the effective dynamic range per block is already constrained, reducing worst-case rounding impact and making HiFi4's extra passes provide diminishing returns. The exact hardware mechanism is not publicly confirmed in full detail; the sufficiency of HiFi2 for BFP8 × BF16 should be treated as an empirically established guideline rather than a derivation from mantissa bit-width alone.
> - BF16 × BF16 (7-bit × 7-bit mantissa) → HiFi4 (4 passes) for full precision; here **both** operands carry full 7-bit mantissas, and the complete cross-product requires more passes than HiFi2 can provide — this is why HiFi4 is required for attention score QK^T in prefill where both Q and K are BF16

Using HiFi4 with BFP4 weights is pure throughput waste: you are running 4 multiplier passes to cover mantissa bits that were already discarded during quantization.

---

## Fidelity × Throughput Tradeoff Summary

At model level, the combined effect of data type and math fidelity on decode throughput for Llama 3.1 8B on a single N150 device:

| Configuration | Approximate decode throughput | Source |
|---|---|---|
| BF16 weights, HiFi4 | ~14 t/s/u (baseline) | (estimated from PERF.md data) |
| BFP8 weights, HiFi2 | ~23 t/s/u (~1.6× baseline) | (PERF.md accuracy mode) |
| BFP4 MLP + BFP8 attn, LoFi/HiFi2 | ~28 t/s/u (~2× baseline) | (PERF.md performance mode) |

The jump from BFP8/HiFi2 to BFP4/LoFi on MLP weights accounts for approximately +22% on its own (23 → 28 t/s/u), entirely from the reduced weight bandwidth and the matched LoFi computation.

---

## Key Takeaways

- Math fidelity controls the number of 5-bit × 7-bit multiplier passes per matmul element: LoFi = 1 pass (~3.5× faster than HiFi4), HiFi2 = 2 passes (~2× faster), HiFi4 = 4 passes (full BF16 precision).
- The fidelity should be matched to the weight dtype: BFP4 weights → LoFi, BFP8 weights → HiFi2, BF16 weights → HiFi4. Using HiFi4 with quantized weights wastes throughput without improving accuracy.
- `fp32_dest_acc_en=True` enables full FP32 accumulation in the Dst register at the cost of halving Dst capacity (8 tiles → 4 tiles); use it for attention score computation in long-context prefill.
- `packer_l1_acc=True` enables L1 += Dst in-place accumulation when the output buffer is in L1, eliminating the read-modify-write overhead (read L1 partial sum, add Dst, write back) that the Packer otherwise performs at each K-block write event. This is a throughput optimization (not a correctness requirement — accumulation works without it, but incurs extra L1 read overhead), and is particularly valuable for large-K matmuls (MLP layers).
- `math_approx_mode=True` speeds up SFPU transcendentals (rsqrt, exp); use it for RMSNorm and SiLU but benchmark accuracy for attention softmax.

---

## Further Reading

- Tenstorrent compute kernel configuration API reference: `ttnn.WormholeComputeKernelConfig` (github.com/tenstorrent/tt-metal — `ttnn/cpp/ttnn/operations/compute_throttle/`)
- PERF.md in tt-metal repository (`models/tt_transformers/PERF.md`) — per-model fidelity and dtype configurations used in production
- [tensix_architecture.md](./tensix_architecture.md) — Dst register capacity constraints that interact with `fp32_dest_acc_en`
- [ttnn_tensor_model.md](./ttnn_tensor_model.md) — data type definitions (BFP8_B, BFP4_B) that pair with the fidelity levels described here
- Chapter 3 ([matmul_program_configs.md](../ch3_matmul_optimizations/matmul_program_configs.md)) — how `compute_kernel_config` integrates with the full matmul program config

---

**Next:** [Chapter 2 — Attention Optimizations](../ch2_attention_optimizations/index.md)
