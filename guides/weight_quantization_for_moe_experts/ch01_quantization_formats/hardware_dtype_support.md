# Hardware and dtype Support on Wormhole

## Overview

This section covers the operational details that connect the three quantization formats to actual Wormhole hardware behavior: the TTNN `DataType` enum, tile layout constraints, `MathFidelity` compute kernel levels, the `fp32_dest_acc_en` flag, and the DRAM bandwidth math that explains why lower-precision formats accelerate expert loading in MoE workloads.

---

## TTNN DataType Enum

TTNN exposes all supported numeric formats through the `ttnn.DataType` enum. The relevant values for weight quantization are:

| TTNN Constant | Enum Name | Bits/elem | Family |
|---|---|---|---|
| `ttnn.bfloat16` | `DataType.BFLOAT16` | 16 | IEEE-like |
| `ttnn.bfloat8_b` | `DataType.BFLOAT8_B` | 8 | Block FP |
| `ttnn.bfloat4_b` | `DataType.BFLOAT4_B` | 4 | Block FP |
| `ttnn.float32` | `DataType.FLOAT32` | 32 | IEEE 754 |
| `ttnn.uint32` | `DataType.UINT32` | 32 | Integer |
| `ttnn.uint16` | `DataType.UINT16` | 16 | Integer |

For MoE expert weights, only `bfloat16`, `bfloat8_b`, and `bfloat4_b` are used in practice. `float32` is too large for on-device weight storage in production.

### Querying a tensor's dtype

```python
import ttnn

w = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
print(w.dtype)            # DataType.BFLOAT8_B
print(w.dtype == ttnn.bfloat8_b)  # True
```

---

## Tile Layout Constraint

`ttnn.TILE_LAYOUT` is mandatory for both `bfloat8_b` and `bfloat4_b`. This constraint is not accidental — it is a fundamental consequence of block floating-point encoding:

- The shared exponent is computed per **32×32 tile**.
- Without tile boundaries, there is no defined group over which to compute the shared scale.
- The hardware's compute units expect block-FP operands to arrive in tile-aligned memory regions.

```python
# This will raise a runtime error:
w_bad = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
# RuntimeError: bfloat8_b requires TILE_LAYOUT

# This is correct:
w_ok = ttnn.from_torch(w_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
```

`bfloat16` is the only weight dtype that supports both `TILE_LAYOUT` and `ROW_MAJOR_LAYOUT`. In practice, even bfloat16 weights are typically kept in `TILE_LAYOUT` for matmul ops because the hardware compute path requires tile-aligned inputs.

---

## MathFidelity Levels

`MathFidelity` controls the precision of intermediate arithmetic within a compute kernel — specifically, how the hardware accumulates partial products in the FPU. It is set in the `WormholeComputeKernelConfig` passed to ops like `ttnn.linear` and `ttnn.matmul`.

| MathFidelity Constant | Accumulation precision | Recommended for |
|---|---|---|
| `ttnn.MathFidelity.LoFi` | Low — fast accumulation, fewer passes | `bfloat4_b` weights |
| `ttnn.MathFidelity.HiFi2` | Medium — 2 accumulation passes | `bfloat8_b` weights |
| `ttnn.MathFidelity.HiFi4` | High — 4 accumulation passes | `bfloat16` weights, full precision |

The pairing matters: using `HiFi4` with `bfloat4_b` wastes compute without improving output quality (the weight precision is the limiting factor). Using `LoFi` with `bfloat8_b` discards precision that `bfloat8_b` could otherwise deliver.

### DeepSeek-V3 expert config

```python
# Gate projection (w1) and up projection (w3) — bfloat4_b + LoFi
gate_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Down projection (w2) — bfloat8_b + HiFi2
down_compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

> **Tip:** `packer_l1_acc=True` enables L1 accumulation in the output packer, which improves accuracy for long reduction chains (deep K-dimension matmuls) without the full cost of HiFi4. It is generally recommended for both bfloat4_b and bfloat8_b expert projections.

---

## The `fp32_dest_acc_en` Flag

`fp32_dest_acc_en` controls whether the destination accumulation register uses 32-bit float precision or 16-bit precision. When set to `True`, partial sums accumulate in float32 before being written to the output tensor.

| Setting | Effect | Use when |
|---|---|---|
| `fp32_dest_acc_en=False` | Accum in bfloat16 | Most MoE expert projections; lower overhead |
| `fp32_dest_acc_en=True` | Accum in float32 | High-precision ops, attention softmax, LM head |

For `bfloat4_b` and `bfloat8_b` weight ops in MoE, `fp32_dest_acc_en=False` is standard. The weight precision is the dominant source of error; elevating the accumulator to float32 adds memory bandwidth overhead without meaningfully improving final output quality.

---

## DRAM Bandwidth Impact of Quantized Weights

For MoE models, the key bottleneck during expert computation is **loading expert weights from DRAM** into L1/SRAM for each batch of routed tokens. The fewer bytes per element, the faster each expert can be loaded.

### Bandwidth model

Wormhole n150 DRAM bandwidth: approximately **288 GB/s** (theoretical peak).

For a single expert with w1, w3, w2 projections loaded per token batch:

| Weight config | Total expert bytes | Load time @ 288 GB/s |
|---|---|---|
| All bfloat16 | 88.1 MB | ~306 µs |
| All bfloat8_b | 44.1 MB | ~153 µs |
| Mixed (bfloat4_b + bfloat8_b) | 29.4 MB | ~102 µs |

> **Note:** These are idealized figures assuming sequential loading with no overlap. Real systems overlap weight loading with compute from other experts, and actual effective bandwidth is lower due to DRAM timing, routing, and L1 capacity constraints.

Even as rough estimates, the numbers illustrate the core principle: for MoE models with sparse expert routing, **DRAM bandwidth is often the binding constraint**, and weight precision has a direct, linear impact on latency.

### L1 working set considerations

Wormhole's L1 (SRAM) per Tensix core is limited. For expert parallelism across multiple cores, the weight tiles must be streamed in from DRAM per-inference rather than kept resident. A 29.4 MB expert weight set (mixed precision) fits more easily in the L1 streaming pipeline than an 88.1 MB full-precision set.

---

## PCC Thresholds from the Test Suite

The Tenstorrent test suite validates MoE model correctness using Pearson Correlation Coefficient (PCC) between quantized and full-precision outputs. Published thresholds for acceptable deployment:

| Configuration | Layer | Min PCC |
|---|---|---|
| bfloat16 | Any | ~1.000 (baseline) |
| bfloat8_b | MLP/FFN layer output | ~0.975 |
| bfloat4_b (gate/up) + bfloat8_b (down) | Full MoE expert output | ~0.970 |
| All bfloat4_b | Full MoE expert output | ~0.950–0.960 |

> **Warning:** PCC thresholds are model- and task-dependent. The values above are derived from DeepSeek-V3 MoE testing. Different architectures, activation functions, or weight distributions may yield different quality curves. Always measure PCC on your specific model before deploying quantized weights to production.

---

## DeepSeek-V3 Expert Quantization Summary

DeepSeek-V3 uses a deliberate mixed-precision strategy for its 256 experts:

| Projection | Weight dtype | MathFidelity | Rationale |
|---|---|---|---|
| Gate (w1) | `ttnn.bfloat4_b` | `LoFi` | Pre-activation; errors suppressed by SiLU gate |
| Up (w3) | `ttnn.bfloat4_b` | `LoFi` | Pre-activation; errors suppressed by SiLU gate |
| Down (w2) | `ttnn.bfloat8_b` | `HiFi2` | Post-activation; feeds residual stream directly |

This configuration achieves approximately 3× expert weight compression versus all-bfloat16, enabling the full 256-expert model to fit within the DRAM budget of a single n150 or n300 board while maintaining output PCC above 0.97.

---

## Quick Reference: Choosing a dtype

Use this decision tree when selecting a weight dtype for a new layer:

```
Is the layer a MoE expert projection?
├── Yes
│   ├── Is it gate (w1) or up (w3) projection?
│   │   └── Use bfloat4_b + LoFi   (4× compression, 4× throughput)
│   └── Is it down (w2) projection?
│       └── Use bfloat8_b + HiFi2  (2× compression, 2× throughput)
└── No
    ├── Is it attention (QKV, output)?
    │   └── Use bfloat16 + HiFi4
    ├── Is it embedding / LM head?
    │   └── Use bfloat16
    └── Is it a shared FFN (non-MoE)?
        └── Consider bfloat8_b if memory constrained
```

---

---

**Next:** [Chapter 2 — TTNN Quantization API](../ch02_ttnn_quantization_api/index.md)
