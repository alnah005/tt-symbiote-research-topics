# bfloat8_b — Block Floating-Point 8-Bit Format

## What Is bfloat8_b?

`bfloat8_b` is Tenstorrent's 8-bit block floating-point format for weight storage and matrix-multiply compute on Wormhole hardware. It stores **1 byte per element** — half the cost of bfloat16 — and enables a corresponding doubling of throughput on Wormhole's BlockFP8 compute path.

The `_b` suffix is critical: it signals **block** floating-point. Unlike standard 8-bit floats (e.g., FP8 E4M3/E5M2), where every element independently stores its own exponent and mantissa, bfloat8_b uses a **shared exponent** per 32×32 tile. This design is specific to Tenstorrent hardware and is distinct from the OCP FP8 standard.

---

## Block Floating-Point: The Core Concept

In a standard floating-point number, each value independently encodes its own scale via its exponent field. This is flexible but expensive: the exponent bits contribute overhead to every single element.

Block floating-point takes a different approach: **a group of elements shares one exponent**, and each element stores only its mantissa (significand) relative to that shared scale. The group in bfloat8_b is a **32×32 tile** (1,024 elements).

```
Standard FP (per element):    [S | EEEEEEEE | MMMMMMM]  ← 16 bits bfloat16
                                                           each element self-contained

Block FP (per tile group):
  Shared exponent: 1 value stored once for the entire 32×32 tile
  Per element:     [S MMMMMMM]  ← 1 sign bit + 7 mantissa bits
                                   (shared exponent stored once per 32×32 tile, not per element)
```

Because the exponent overhead is amortized across 1,024 elements, the effective per-element cost drops to 8 bits while preserving meaningful relative precision within each tile.

---

## Binary Layout

For a 32×32 tile in bfloat8_b:

- **Shared exponent:** Stored in the tile header — 1 value that sets the scale for all 1,024 elements in the tile.
- **Per-element storage:** 7 mantissa bits each (plus 1 sign bit), totaling 8 bits per element, interpreted relative to the shared exponent.

The total storage per tile is:

```
1,024 elements × 1 byte/element = 1,024 bytes
+ tile header (shared exponent + metadata, hardware-managed)
```

From the software perspective, you observe **1 byte per element** when calculating memory.

### Precision characteristics

Within a tile, all elements are scaled relative to the maximum-magnitude element (which determines the shared exponent). Elements that are much smaller than the tile maximum lose relative precision. This is acceptable for trained neural network weights because:

- Weight values within a single projection matrix tend to cluster in a narrow magnitude range.
- The shared exponent naturally adapts to each tile's data distribution.
- Occasional precision loss in low-magnitude elements has minimal effect on output quality.

---

## TTNN Usage

The TTNN dtype constant is `ttnn.bfloat8_b`. It **requires `ttnn.TILE_LAYOUT`** — bfloat8_b cannot be used with `ttnn.ROW_MAJOR_LAYOUT` because the block floating-point encoding is fundamentally tied to the 32×32 tile structure.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

# Load a down-projection weight (w2) from a pretrained checkpoint
w2_torch = torch.randn(2048, 7168, dtype=torch.bfloat16)

# Cast to bfloat8_b on device — tile layout is mandatory
w2_tt = ttnn.from_torch(
    w2_torch,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

print(w2_tt.dtype)   # DataType.BFLOAT8_B
print(w2_tt.shape)   # [2048, 7168]
```

> **Warning:** Passing `layout=ttnn.ROW_MAJOR_LAYOUT` with `dtype=ttnn.bfloat8_b` will raise a runtime error. The block floating-point encoding requires tile boundaries to be defined before quantization can proceed.

### Verifying the cast quality

Before deploying, measure the Pearson Correlation Coefficient (PCC) between bfloat8_b output and a bfloat16 reference:

```python
import ttnn

# Run matmul with bfloat8_b weights
output_bf8 = ttnn.linear(activation, w2_tt)

# Convert back to torch for comparison
output_bf8_torch = ttnn.to_torch(output_bf8)
output_ref_torch  = ttnn.to_torch(output_ref)   # bfloat16 reference

# Compute PCC
def pcc(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()

score = pcc(output_bf8_torch, output_ref_torch)
print(f"PCC: {score:.4f}")   # Expect ~0.975 for down projection
```

> **Tip:** The Tenstorrent test suite uses a PCC threshold of **~0.975** for MLP layers quantized to bfloat8_b. This is a practical passing bar for production deployment of down-projection (w2) layers.

---

## Memory Footprint

bfloat8_b stores exactly **1 byte per element** — a **2× reduction** compared to bfloat16.

### Expert weight example

Using the same `7168×2048` expert gate projection:

```
bfloat16:  7168×2048 × 2 bytes = 29.4 MB
bfloat8_b: 7168×2048 × 1 byte  = 14.7 MB   ← 2× reduction
```

For the down projection (w2) at shape `2048×7168`:

```
bfloat16:  2048×7168 × 2 bytes = 29.4 MB
bfloat8_b: 2048×7168 × 1 byte  = 14.7 MB
```

DeepSeek-V3 uses bfloat8_b specifically for the down projection (w2) because it handles the accumulation path where higher precision matters more than gate/up paths.

---

## Wormhole Throughput for BlockFP8

bfloat8_b delivers 2× throughput over bfloat16 on Wormhole hardware (148 TFLOPS on n150, 262 TFLOPS on n300); see the format comparison table in `index.md` for the full comparison. The 2× throughput gain reflects the hardware's ability to pack twice as many elements per compute cycle when using BlockFP8. This is not just a memory bandwidth effect — the MAC units themselves operate faster on 8-bit operands.

> **Tip:** In memory-bandwidth-limited regimes (small batch sizes, large expert weights), the 2× memory reduction of bfloat8_b often matters more than the raw TFLOPS increase. Loading a 14.7 MB weight from DRAM instead of 29.4 MB halves the load latency for each expert invocation.

---

## Packing Behavior in Tiles

In TILE_LAYOUT with bfloat8_b, each 32×32 tile occupies:

```
32 × 32 × 1 byte = 1,024 bytes per tile
```

Compare to bfloat16:

```
32 × 32 × 2 bytes = 2,048 bytes per tile
```

A `7168×2048` weight matrix in bfloat8_b:

```
(7168/32) × (2048/32) = 224 × 64 = 14,336 tiles
14,336 tiles × 1,024 bytes = ~14.0 MiB
```

The tile header containing the shared exponent is managed by hardware and is not reflected in the software-visible tensor size.

---

## DeepSeek-V3 Usage

DeepSeek-V3 uses bfloat8_b for the down projection (w2) because it feeds the residual stream directly, requiring higher precision. See `hardware_dtype_support.md` for the full mixed-precision configuration table.

---

## Next Steps

- [bfloat4_b_format.md](./bfloat4_b_format.md) — The 4-bit block FP format, 4× compression, and when it is viable
- [hardware_dtype_support.md](./hardware_dtype_support.md) — MathFidelity levels and TTNN DataType enum details
