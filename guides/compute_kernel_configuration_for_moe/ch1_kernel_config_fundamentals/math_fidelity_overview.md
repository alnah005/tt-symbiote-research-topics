# math_fidelity: Overview

## Overview

`math_fidelity` is the `ttnn.MathFidelity` enum field in `WormholeComputeKernelConfig` that controls how many mantissa bits from each bfloat16 operand are actually presented to the multiplier hardware during dot-product accumulation. It is the primary lever for trading arithmetic throughput against numerical accuracy in TTNN matmul operations.

This document covers:

- What math fidelity controls at the hardware level
- The four enum values and their integer codes
- The key throughput/accuracy intuition
- Why the TTNN default is `LoFi` and when that default is suboptimal
- A forward pointer to Chapter 2 for the full quantitative comparison

---

## What Math Fidelity Controls

Each input tile to the Tensix FPU consists of bfloat16 values: 1 sign bit, 8 exponent bits, and 7 mantissa bits. During a dot-product accumulation (the inner loop of a matrix multiply), each pair of bfloat16 values from the A and B tiles must be multiplied together.

A full bfloat16 multiplication would use all 7 mantissa bits from each operand, producing a 14-bit mantissa product before rounding. However, using all bits is not always necessary to produce a useful result — and it costs cycles.

`math_fidelity` controls how many of those 7 mantissa bits are actually used per multiplication step. This is sometimes described as "mantissa bit truncation before the multiplier": the hardware presents only the top N mantissa bits of each operand to the multiplier, discarding the rest.

- **Lower fidelity** = fewer mantissa bits used = smaller effective multiplier input = faster per-multiply, lower accuracy
- **Higher fidelity** = more mantissa bits used = closer to true bfloat16 product = slower per-multiply, higher accuracy

The effect compounds across the K dimension: each K tile's contribution to the output is computed with the configured fidelity, so deeper K loops amplify the aggregate precision difference between fidelity levels.

> **Tip:** Think of math fidelity as a precision dial on the multiplier, not on the accumulator. `fp32_dest_acc_en` controls accumulator precision; `math_fidelity` controls per-multiply precision. These two mechanisms are independent.

---

## Enum Values

```python
import ttnn

ttnn.MathFidelity.LoFi   # Integer code 0 — lowest fidelity, highest throughput
ttnn.MathFidelity.HiFi2  # Integer code 2 — medium-high fidelity
ttnn.MathFidelity.HiFi3  # Integer code 3 — intermediate (rarely used directly)
ttnn.MathFidelity.HiFi4  # Integer code 4 — highest fidelity, closest to full bfloat16 product
```

Note that integer code 1 is not assigned — the enum jumps from LoFi (0) to HiFi2 (2). HiFi3 exists but is rarely the right choice in practice: HiFi2 is fast enough for most precision-sensitive ops, and HiFi4 is reserved for cases where PCC must be maximized regardless of cost.

| Enum Value | Integer Code | Effective Mantissa Bits Used | Relative Throughput | Typical Use Case |
|---|---|---|---|---|
| `LoFi` | 0 | Reduced (approximate top bits only) | Highest (~2x over HiFi4 for matmul-bound ops) | Gate/up projections, bandwidth-bound decode |
| `HiFi2` | 2 | Moderate | High | Down projections, accuracy-sensitive paths |
| `HiFi3` | 3 | Near-full | Moderate | Rarely used; niche intermediate level |
| `HiFi4` | 4 | Full (all 7 mantissa bits) | Lowest | Maximum PCC requirements, reference validation |

> **Note:** The exact mantissa bit counts for each level are hardware-microcode-specific and are not exposed in the TTNN Python API. The table above reflects the practical observable effect: LoFi introduces the most mantissa error, HiFi4 the least. Chapter 2 provides measured PCC data and throughput multipliers for representative MoE matmul shapes.

---

## The Key Intuition: Fewer Bits, More Throughput

The Tensix FPU multiplier pipeline executes multiply-accumulate operations on tiles. For a given tile size (32x32 for bfloat16), the number of cycles required to complete the multiply-accumulate across all 1024 element pairs is proportional to how many mantissa bits must be processed per pair.

LoFi achieves its throughput advantage by reducing the effective multiplier input width. The hardware completes each per-element multiply faster, meaning the FPU can process more tiles per unit time. For a matmul that is compute-bound (large M, large N, large K), this translates directly into higher tile throughput and lower latency.

For a matmul that is bandwidth-bound (the typical case for MoE decode with M = batch size = 1 to 32), the compute savings from LoFi may be partially masked by memory access latency. Even so, reducing compute time per tile frees the pipeline sooner, reducing stall time and contributing to overall throughput.

```python
# Example: comparing fidelity levels on a gate projection matmul
# [batch=1, seq=1, d_model=7168] x [d_model=7168, d_ff=2048]
# This is bandwidth-bound at decode batch=1

import ttnn

# Config sweep for benchmarking fidelity impact
configs = {
    "lofi": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ),
    "hifi2": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ),
    "hifi4": ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ),
}

# In practice: run each config, measure latency and PCC vs PyTorch reference
# See Chapter 2 for the full sweep methodology and expected PCC ranges
```

---

## Why the Default is `LoFi`

The default `math_fidelity` for `WormholeComputeKernelConfig` is `ttnn.MathFidelity.LoFi`. This reflects a deliberate throughput-first default for Wormhole B0.

The reasoning:

1. **Most inference workloads are bandwidth-bound at production batch sizes.** At decode-mode batch sizes (M=1 to 32), weight reads dominate execution time. Reducing compute cycles via LoFi does not slow down the op as long as it remains bandwidth-bound.

2. **bfloat16 already has limited precision.** bfloat16's 7-bit mantissa is a significant reduction from float32's 23 bits. The gap between LoFi and HiFi4 precision is thus measured against a 7-bit baseline, not a 23-bit one. For many practical workloads, the PCC degradation at LoFi is within the acceptable range of bfloat16 compute in general.

3. **LoFi is the safe failure mode.** If a developer forgets to specify `math_fidelity`, they get a fast config rather than an accidentally slow one. Accuracy regressions at LoFi are detectable via PCC testing before deployment.

The downside of this default: operations that feed directly into sensitive aggregations (residual stream accumulation, softmax over large sequence lengths) may see PCC degradation at LoFi. For those operations, the developer must explicitly choose HiFi2 or HiFi4.

---

## When LoFi is Not Sufficient

LoFi becomes problematic in two scenarios:

1. **K-deep accumulations that feed directly into the residual stream.** The down projection in MoE FFN blocks (`[batch, d_ff] x [d_ff, d_model]`) accumulates across `K_t = d_ff / 32` tiles. For `d_ff = 2048`, `K_t = 64`. Each LoFi multiply introduces a small mantissa error; these errors, when summed across K_t tiles and added to the residual stream, can shift hidden state values enough to degrade end-to-end model PCC below the 0.999 threshold.

2. **Operations involving softmax over long sequences.** The exp approximation and reciprocal in softmax are sensitive to fidelity when the sequence length is large (K >= 16K), because small errors in the logit matmul amplify via the exponential. These cases are addressed in Chapter 4 (`math_approx_mode`) rather than here.

For MoE expert matmuls specifically:

```
Gate (w1):  LoFi acceptable  — feeds SiLU, error does not accumulate into residual
Up (w3):    LoFi acceptable  — feeds SiLU, same reasoning
Down (w2):  HiFi2 required   — feeds residual stream directly; LoFi PCC ~0.99, below threshold
```

---

## Relationship to `fp32_dest_acc_en`

For how `fp32_dest_acc_en` interacts with fidelity choice, see `fp32_dest_acc_en.md`. Down-projection typically uses HiFi2 + `fp32_dest_acc_en=True` because residual stream accumulation is sensitive to rounding.

---

## Forward Reference: Chapter 2

This overview deliberately omits quantitative throughput multipliers and measured PCC values. Those numbers require characterization against specific matmul shapes and are the subject of Chapter 2 (`ch2_math_fidelity_levels/`), which covers:

- Measured relative throughput for LoFi / HiFi2 / HiFi4 on representative MoE shapes
- PCC vs PyTorch reference for gate, up, and down projections at each fidelity level
- The fidelity selection workflow: how to step down from HiFi4 to LoFi and identify the lowest safe fidelity for a given model

---

## Next Steps

This completes Chapter 1. The four primary fields of `WormholeComputeKernelConfig` have been introduced:

- `math_fidelity` — per-multiply mantissa precision (this file)
- `fp32_dest_acc_en` — destination register accumulation precision (`fp32_dest_acc_en.md`)
- `packer_l1_acc` — packer L1 vs. DRAM accumulation path (`wormhole_compute_kernel_config_api.md`)
- `math_approx_mode` — SFPU transcendental approximation (`wormhole_compute_kernel_config_api.md`)

Proceed to **Chapter 2** (`ch2_math_fidelity_levels/index.md`) for a full quantitative comparison of LoFi, HiFi2, and HiFi4 on MoE expert matmul shapes, including PCC data and the fidelity selection workflow.
