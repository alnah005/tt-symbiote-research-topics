# bfloat16 — The Standard Tenstorrent Weight Format

## What Is bfloat16?

`bfloat16` (Brain Float 16) is a 16-bit floating-point format developed at Google Brain and now widely adopted across AI accelerators including Tenstorrent Wormhole. It is the default precision for weights and activations in most TTNN workloads, and it is the baseline against which `bfloat8_b` and `bfloat4_b` are compared.

The key design choice in bfloat16 is to preserve the **exponent width of float32** (8 bits) while truncating the mantissa from 23 bits down to 7 bits. This gives bfloat16 the same dynamic range as float32, making it easy to cast between the two without overflow or underflow — unlike float16, which has only 5 exponent bits and frequently overflows during training.

---

## Binary Layout

A bfloat16 value occupies exactly 16 bits arranged as follows:

```
Bit 15   Bits 14–7     Bits 6–0
  S      EEEEEEEE       MMMMMMM
  1 bit  8 bits         7 bits
  sign   exponent       mantissa
```

- **Sign (1 bit):** 0 = positive, 1 = negative.
- **Exponent (8 bits):** Biased exponent with bias 127, identical to float32. This gives a range of approximately 2⁻¹²⁶ to 2¹²⁷.
- **Mantissa (7 bits):** The fractional part of the significand. An implicit leading 1 is assumed for normal numbers (same as IEEE 754), so the effective significand has 8 bits of precision.

### Comparison to float32

| Property | float32 | bfloat16 |
|---|---|---|
| Total bits | 32 | 16 |
| Sign bits | 1 | 1 |
| Exponent bits | 8 | 8 |
| Mantissa bits | 23 | 7 |
| Max value | ~3.4×10³⁸ | ~3.4×10³⁸ |
| Min normal | ~1.18×10⁻³⁸ | ~1.18×10⁻³⁸ |
| Decimal precision | ~7.2 digits | ~2.4 digits |

The shared exponent range means a `float32 → bfloat16` cast is lossless in terms of range; only mantissa precision is reduced (the lower 16 bits of the float32 mantissa are truncated).

---

## Why bfloat16 Is Standard on Tenstorrent

Wormhole's compute units are natively optimized for bfloat16 matrix operations. The FPU tiles in Wormhole execute bfloat16 matrix-multiply-accumulate (MMA) instructions at peak throughput using TILE_LAYOUT 32×32 tiles. Key reasons bfloat16 is the default:

1. **Broad model compatibility.** Most pretrained LLMs and MoE checkpoints publish weights in float32 or bfloat16. A direct cast to bfloat16 incurs negligible quality loss.
2. **Numerically safe activations.** Activations in transformer attention and FFN layers span a wide dynamic range. bfloat16's 8-bit exponent handles outliers that would overflow float16.
3. **Simple dtype path.** No per-tensor scale factors, no tile-level metadata — every element is self-contained and independently readable.

---

## TTNN Usage

The TTNN dtype constant is `ttnn.bfloat16`. It works with both `ttnn.TILE_LAYOUT` and `ttnn.ROW_MAJOR_LAYOUT`, though TILE_LAYOUT is required for matmul and most compute ops.

```python
import ttnn
import torch

# Load a weight tensor from a PyTorch checkpoint
w_torch = torch.randn(7168, 2048, dtype=torch.bfloat16)

# Move to Tenstorrent device in bfloat16, tile layout
device = ttnn.open_device(device_id=0)

w_tt = ttnn.from_torch(
    w_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

print(w_tt.shape)   # [7168, 2048]
print(w_tt.dtype)   # DataType.BFLOAT16
```

> **Tip:** `ttnn.from_torch` accepts `torch.bfloat16` tensors directly. When the source dtype matches `ttnn.bfloat16`, no numeric conversion occurs — only a layout transformation if `TILE_LAYOUT` is requested.

---

## Memory Footprint

bfloat16 stores exactly **2 bytes per element**.

### Expert weight example

Consider a single MoE expert's gate projection (w1) with shape `7168 × 2048`:

```
Elements = 7168 × 2048 = 14,680,064
Bytes    = 14,680,064 × 2 = 29,360,128 bytes ≈ 28.0 MiB ≈ 29.4 MB
```

For a three-projection expert (gate w1, up w3, down w2) where w1 and w3 are `7168×2048` and w2 is `2048×7168`:

```
w1 (gate):  7168×2048 × 2 bytes = 29.4 MB
w3 (up):    7168×2048 × 2 bytes = 29.4 MB
w2 (down):  2048×7168 × 2 bytes = 29.4 MB
Total per expert: ~88.1 MB
```

Across a 256-expert MoE model:

```
256 experts × 88.1 MB = ~22.5 GB
```

This exceeds the DRAM capacity of a single Wormhole n150 card (12 GB), which is the primary motivation for lower-precision formats.

---

## Wormhole Throughput for bfloat16

| Hardware | Peak bfloat16 TFLOPS |
|---|---|
| Wormhole n150 (1 chip) | 74 TFLOPS |
| Wormhole n300 (2 chips) | 131 TFLOPS |

These figures represent the hardware ceiling for bfloat16 matrix multiply. Real workloads achieve a fraction of peak depending on matrix shapes, memory bandwidth, and routing overhead in MoE models.

> **Warning:** These are peak theoretical numbers. For small expert batch sizes (low token routing), the bottleneck is often DRAM bandwidth loading weights, not compute throughput. bfloat16's 2 bytes/element makes it the most bandwidth-hungry of the three formats.

---

## Tile Layout and Packing

In TILE_LAYOUT, tensors are stored as 32×32 element tiles in row-major tile order. For bfloat16, each tile occupies:

```
32 × 32 × 2 bytes = 2048 bytes per tile
```

A `7168×2048` weight matrix requires:

```
(7168/32) × (2048/32) = 224 × 64 = 14,336 tiles
14,336 × 2048 bytes = ~28.0 MiB
```

Dimensions that are not multiples of 32 are zero-padded to the next tile boundary before being sent to device.

---

---

**Next:** [`bfloat8_b_format.md`](./bfloat8_b_format.md)
