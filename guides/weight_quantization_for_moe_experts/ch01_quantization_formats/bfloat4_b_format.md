# bfloat4_b — Block Floating-Point 4-Bit Format

## What Is bfloat4_b?

`bfloat4_b` is Tenstorrent's most aggressive weight compression format on Wormhole hardware. It stores **4 bits per element** — one quarter the cost of bfloat16 and half the cost of bfloat8_b — using the same block floating-point tile structure as bfloat8_b but with a narrower mantissa per element.

Like `bfloat8_b`, the `_b` suffix denotes **block** floating-point with a **shared exponent per 32×32 tile**. Within a tile, each element contributes only ~3 bits of mantissa, with the tile's shared exponent providing the common scale. This is a significant precision reduction, and it is appropriate only for specific ops and weight roles — but when those conditions are met, it delivers up to 4× throughput over bfloat16 and cuts expert weight memory to 25% of baseline.

---

## Binary Layout

For a 32×32 tile in bfloat4_b:

- **Shared exponent:** Stored once per tile in the tile header, setting the scale for all 1,024 elements.
- **Per-element storage:** 4 bits each, encoding the mantissa relative to the shared exponent.

Bit layout per element:

```
[S MMM]
 │ └─┘
 │  └── 3 mantissa bits (value relative to shared exponent)
 └───── 1 sign bit
```

**1 sign bit + 3 mantissa bits = 4 bits per element.** The shared exponent is stored once in the tile header and amortized across all 1,024 elements — it contributes zero overhead to the per-element storage cost.

With 3 mantissa bits and an implicit leading 1, the effective significand has 8 levels (2³) of precision between consecutive powers of 2. This is analogous to having roughly **1.2 decimal digits** of relative precision per element — much coarser than bfloat16's 2.4 digits, but sufficient for gate and up projections whose outputs pass through nonlinear activations.

### Packing

Two elements are packed into each byte:

```
Byte N: [elem_2k+1 (4 bits) | elem_2k (4 bits)]
```

The hardware unpacks elements during the tile load and compute phases. From the programmer's perspective, the tensor shape and indexing are unchanged — only the dtype and memory footprint differ.

---

For a side-by-side comparison of all three formats, see the format comparison table in `index.md`.

---

## TTNN Usage

The TTNN dtype constant is `ttnn.bfloat4_b`. Like `bfloat8_b`, it **requires `ttnn.TILE_LAYOUT`** — using ROW_MAJOR_LAYOUT will raise a runtime error.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

# Gate projection (w1) and up projection (w3) — candidates for bfloat4_b
w1_torch = torch.randn(7168, 2048, dtype=torch.bfloat16)
w3_torch = torch.randn(7168, 2048, dtype=torch.bfloat16)

w1_tt = ttnn.from_torch(
    w1_torch,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

w3_tt = ttnn.from_torch(
    w3_torch,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

print(w1_tt.dtype)   # DataType.BFLOAT4_B
```

> **Warning:** `ttnn.bfloat4_b` should not be used for activations or for the down projection (w2) in standard FFN/MoE blocks without careful accuracy testing. The reduced mantissa precision amplifies errors when values span a wide range within a single tile, as is common in intermediate activation tensors.

### MathFidelity pairing

When using `bfloat4_b` weights, the recommended compute kernel config is `MathFidelity.LoFi` (Low Fidelity). This aligns the compute precision with the weight precision and maximizes throughput:

```python
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

output = ttnn.linear(
    activation,
    w1_tt,
    compute_kernel_config=compute_kernel_config,
)
```

See [hardware_dtype_support.md](./hardware_dtype_support.md) for the full MathFidelity reference.

---

## Memory Footprint

bfloat4_b stores exactly **0.5 bytes (4 bits) per element** — a **4× reduction** compared to bfloat16.

### Expert weight example

The bfloat16 baseline is ~88.1 MB per expert (see `bfloat16_format.md` for the full derivation).

For the gate projection `7168×2048`, bfloat4_b costs:

```
bfloat4_b: 7168×2048 × 0.5 bytes = 7.34 MB   ← 4× reduction from bfloat16
```

For a full three-projection expert with DeepSeek-V3's mixed precision (w1/w3 in bfloat4_b, w2 in bfloat8_b):

```
w1 (gate):  7168×2048 × 0.5 bytes = 7.34 MB
w3 (up):    7168×2048 × 0.5 bytes = 7.34 MB
w2 (down):  2048×7168 × 1.0 byte  = 14.7 MB
Total: ~29.4 MB per expert   ← ~3× total reduction from ~88.1 MB bfloat16 baseline
```

Across 256 experts:

```
All bfloat16:    256 × 88.1 MB = ~22.5 GB
Mixed precision: 256 × 29.4 MB = ~7.5 GB   ← fits on n150 (12 GB DRAM)
```

This is the concrete reason DeepSeek-V3's expert quantization strategy makes single-chip deployment viable.

---

## Wormhole Throughput for bfloat4_b

bfloat4_b packs four elements per compute unit access compared to bfloat16's one, delivering approximately 4× throughput (~296 TFLOPS on n150, ~524 TFLOPS on n300); see the format comparison table in `index.md` for the full comparison.

> **Tip:** The 4× throughput gain is most fully realized for **weight-stationary** operations with large matrices where the bottleneck is compute rather than data movement. For small batch sizes with narrow token routing (e.g., 8 tokens across 256 experts), the actual bottleneck may still be DRAM load latency, in which case the 4× memory reduction is the dominant benefit.

---

## When bfloat4_b Is Viable

bfloat4_b is appropriate when all of the following conditions hold:

1. **Weight-stationary operation.** The weight matrix is loaded once and used for multiple token activations, amortizing the precision cost over many multiply-accumulate steps.
2. **Large matrices.** Larger tiles and deeper accumulation allow small per-element errors to average out statistically. Shapes like `7168×2048` are well-suited; very small projections may show more degradation.
3. **Pre-nonlinearity.** Gate and up projections feed into SiLU or similar nonlinear activations before combining. The nonlinearity acts as a natural noise reducer for small precision errors.
4. **Validated PCC.** The full model pipeline has been measured against a bfloat16 reference and achieves acceptable PCC. For MoE with mixed precision (bfloat4_b gate/up + bfloat8_b down), the target is approximately **PCC ≥ 0.97** at the full-model level.

### Cases where bfloat4_b is not recommended

- **Down projections (w2):** These combine the gated outputs and feed directly into the residual stream. Precision errors here accumulate across layers. Use `bfloat8_b` instead.
- **Attention QKV projections:** Attention scores are sensitive to small variations in Q and K, especially at low temperatures.
- **Embeddings and LM head:** These operate at the vocabulary boundary and benefit from full precision.
- **Small matrices:** Tensors with fewer than ~1,024 elements per tile dimension may not amortize the shared-exponent error effectively.

---

## Precision Analysis: What Happens Inside a Tile

Consider a 32×32 tile containing weight values. The shared exponent is set to represent the maximum-magnitude value in the tile. Any element whose magnitude is much smaller than the maximum will have its lower-precision bits wiped out — effectively rounded to zero or to a coarse step.

Example: if a tile has a maximum weight of 0.5 and a minimum of 0.001, the step size for bfloat4_b at the minimum's magnitude is:

```
0.5 / 2^3 = 0.0625   (step size ≈ 6% of the tile maximum)
0.001 rounds to 0.0   (below the minimum representable relative value)

0.001 rounds to zero because it is less than half the step size (0.0625 / 2 = 0.03125), not because it is smaller than the step size itself.
```

For trained transformer weights, the distribution within a single tile is typically narrow enough that this is not a significant problem. However, if a model uses weight initialization or fine-tuning strategies that produce high-variance tiles, bfloat4_b may show visible quality degradation.

---

## Next Steps

- [hardware_dtype_support.md](./hardware_dtype_support.md) — TTNN DataType enum, MathFidelity levels, `fp32_dest_acc_en`, and DRAM bandwidth analysis
- Return to [index.md](./index.md) for the full format comparison table
