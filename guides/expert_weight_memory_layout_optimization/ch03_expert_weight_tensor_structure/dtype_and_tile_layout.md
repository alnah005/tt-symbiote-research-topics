# Dtype and Tile Layout Requirements

## Overview

Before an expert weight tensor can be used in a `ttnn.matmul` call, it must satisfy two requirements: it must use a supported dtype, and it must be in TILE_LAYOUT. This file covers the standard dtypes and their byte costs, the tile-layout requirement and its 32-element alignment constraint, the memory overhead introduced by padding non-aligned shapes, and worked calculations for total per-expert weight memory.

---

## Standard Dtypes

TTNN supports three quantized floating-point dtypes commonly used for MoE weight storage. All three can represent the same logical values but differ in bit width and, therefore, memory cost.

| Dtype | Bits per element | Bytes per element | Relative size vs BF16 |
|---|---|---|---|
| `ttnn.bfloat16` (BF16) | 16 | 2.0 | 1× (baseline) |
| `ttnn.bfloat8_b` (BF8) | 8 | 1.0 | 0.5× |
| `ttnn.bfloat4_b` (BF4) | 4 | 0.5 | 0.25× |

> **Tip:** `bfloat8_b` is the most common production dtype for MoE expert weights on Tenstorrent hardware. It halves memory relative to BF16 with acceptable accuracy loss on most transformer models. `bfloat4_b` is used when DRAM capacity is the binding constraint, but requires careful accuracy validation.

---

## TILE_LAYOUT Requirement

`ttnn.matmul` requires both operands to be in `TILE_LAYOUT`. If a tensor is in `ROW_MAJOR_LAYOUT` when passed to `ttnn.matmul`, TTNN inserts an automatic layout conversion at runtime. This conversion:

- Copies the entire tensor from ROW_MAJOR to TILE format.
- Allocates temporary L1 (Level 1 on-chip SRAM) buffer space for the converted tensor.
- Adds latency proportional to tensor size.

> **Warning:** For large expert weight matrices (tens of megabytes per expert), a ROW_MAJOR-to-TILE conversion at inference time can add several milliseconds per matmul dispatch. Always convert weights to TILE_LAYOUT during model loading, not during inference.

### Converting to TILE_LAYOUT

```python
import ttnn
import torch

# Load weight from checkpoint (typically float32 or bfloat16 torch tensor)
w1_torch = checkpoint["expert_0.w1"]   # torch.Tensor, shape [d_model, d_ff]

# Convert to TTNN tensor with TILE_LAYOUT at load time
w1_ttnn = ttnn.from_torch(
    w1_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,   # required for ttnn.matmul
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## Tile Size and Alignment Constraint

TTNN's tile is a 32×32 element block. In TILE_LAYOUT, the tensor is partitioned into 32×32 tiles; each tile is the minimum addressable unit.

| Property | Value |
|---|---|
| Tile height | 32 elements |
| Tile width | 32 elements |
| Elements per tile | 1,024 |

**Constraint:** Both the height and width of every shard must be multiples of 32.

```
shard_H % 32 == 0   (tile height alignment)
shard_W % 32 == 0   (tile width alignment)
```

This propagates upward: the full tensor dimensions must also be multiples of 32 for the shard dimensions to be tile-aligned after division.

### Bytes per tile by dtype

| Dtype | Bytes per element | Bytes per 32×32 tile |
|---|---|---|
| BF16 | 2.0 | 2,048 |
| BF8 | 1.0 | 1,024 |
| BF4 | 0.5 | 512 |

---

## Padding for Non-Tile-Aligned Shapes

If `d_model` or `d_ff` is not a multiple of 32, the tensor must be padded before conversion to TILE_LAYOUT. Padding rounds each dimension up to the nearest multiple of 32.

```python
import math

def pad_to_tile(dim: int, tile: int = 32) -> int:
    """Return dim rounded up to the nearest multiple of tile."""
    return math.ceil(dim / tile) * tile

# Example: DeepSeek-MoE-16B, d_ff = 1408
d_ff_raw    = 1408
d_ff_padded = pad_to_tile(d_ff_raw)   # ceil(1408 / 32) * 32 = 1408 (already aligned)
# 1408 / 32 = 44.0 -> already tile-aligned; no padding needed

# Hypothetical: d_ff = 1400
d_ff_raw    = 1400
d_ff_padded = pad_to_tile(d_ff_raw)   # ceil(1400 / 32) * 32 = 1408
# Padding adds 8 columns; 8 * d_model extra elements per projection
```

> **Warning:** Padding increases the stored tensor size beyond the logical weight count. For BF16 and a `[d_model, d_ff]` gate projection with 8 padding columns, the overhead is `d_model * 8 * 2` bytes. At d_model=4096 this is 65,536 bytes (~64 KB) per projection — small relative to the full matrix but non-negligible when multiplied across all experts and all projections. Always account for padded sizes when estimating device memory budgets.

### Checking alignment before loading

```python
def check_tile_alignment(shape: list[int], tile: int = 32) -> None:
    """Raise ValueError if any dimension is not tile-aligned."""
    for i, dim in enumerate(shape):
        if dim % tile != 0:
            raise ValueError(
                f"Dimension {i} (size {dim}) is not tile-aligned. "
                f"Pad to {pad_to_tile(dim)} before converting to TILE_LAYOUT."
            )
```

---

## Total Per-Expert Weight Memory

For a single expert in a SwiGLU MoE layer, the three weight matrices (gate w1, up w3, down w2) collectively determine the per-expert memory footprint. The formula is:

```
per_expert_bytes = (2 * d_model * d_ff + d_ff * d_model) * bytes_per_element
                 = 3 * d_model * d_ff * bytes_per_element
```

All three matrices have the same element count (`d_model * d_ff`), so the factor of 3 is exact (assuming tile-aligned dimensions with no padding waste).

### Mixtral 8x7B

Parameters: `d_model = 4096`, `d_ff = 14336`.

```python
d_model  = 4096
d_ff     = 14336
elements = d_model * d_ff           # 4096 * 14336 = 58,720,256 per projection
total_elements = 3 * elements       # 176,160,768 (gate + up + down)

# BF16
total_bytes_bf16  = total_elements * 2       # 352,321,536 bytes
total_mb_bf16     = total_bytes_bf16 / (1024 ** 2)   # ~335.9 MB

# BF8
total_bytes_bf8   = total_elements * 1       # 176,160,768 bytes
total_mb_bf8      = total_bytes_bf8 / (1024 ** 2)    # ~168.0 MB
```

| Dtype | Bytes per expert | Megabytes per expert |
|---|---|---|
| BF16 | 352,321,536 | ~335.9 MB |
| BF8 | 176,160,768 | ~168.0 MB |
| BF4 | 88,080,384 | ~84.0 MB |

> **Tip:** The commonly cited "352.3 MB" figure uses 1 MB = 1,000,000 bytes (SI units): 352,321,536 / 1,000,000 = 352.32 MB ≈ 352.3 MB. Using binary megabytes (1 MiB = 1,048,576 bytes) gives ~335.9 MiB. Verify which convention your memory budget uses to avoid off-by-a-few-percent errors.

### Qwen MoE (235B-A22B)

Parameters: `d_model = 7168`, `d_ff = 2048`.

```python
d_model  = 7168
d_ff     = 2048
elements = d_model * d_ff           # 7168 * 2048 = 14,680,064 per projection
total_elements = 3 * elements       # 44,040,192

# BF16
total_bytes_bf16  = total_elements * 2       # 88,080,384 bytes
total_mb_bf16     = total_bytes_bf16 / (1024 ** 2)   # ~84.0 MiB

# BF8
total_bytes_bf8   = total_elements * 1       # 44,040,192 bytes
total_mb_bf8      = total_bytes_bf8 / (1024 ** 2)    # ~42.0 MiB
```

| Dtype | Bytes per expert | Mebibytes per expert |
|---|---|---|
| BF16 | 88,080,384 | ~84.0 MiB |
| BF8 | 44,040,192 | ~42.0 MiB |
| BF4 | 22,020,096 | ~21.0 MiB |

### Total across all experts

Multiply per-expert bytes by `num_experts` to get the total weight-only DRAM budget:

| Model | num_experts | Per-expert (BF16) | Total (BF16) | Total (BF8) |
|---|---|---|---|---|
| Mixtral 8x7B | 8 | ~335.9 MiB | ~2.6 GiB | ~1.3 GiB |
| DeepSeek-MoE-16B | 64 | ~16.5 MiB* | ~1.03 GiB | ~0.52 GiB |
| Qwen MoE (235B-A22B) | 128 | ~84.0 MiB | ~10.5 GiB | ~5.25 GiB |

\* DeepSeek-MoE-16B: d_model=2048, d_ff=1408; `3 * 2048 * 1408 * 2 = 17,301,504 bytes = ~16.5 MiB` per expert.

> **Warning:** These figures cover only the MoE expert weights. Attention weights, layer norms, embedding tables, and activation buffers add substantially to total device memory. Always add at least a 20–30% overhead allowance for non-expert parameters and runtime activations when sizing your memory budget.

---

## Conversion Reference

Quick reference for per-element and per-tile costs:

```python
DTYPE_BYTES = {
    "bfloat16":  2.0,
    "bfloat8_b": 1.0,
    "bfloat4_b": 0.5,
}

TILE_SIZE = 32  # elements per tile side

def expert_weight_bytes(d_model: int, d_ff: int, dtype: str) -> int:
    """Total bytes for all three projections of one expert."""
    bytes_per_elem = DTYPE_BYTES[dtype]
    elements_per_projection = d_model * d_ff
    total_elements = 3 * elements_per_projection  # gate + up + down
    return int(total_elements * bytes_per_elem)

# Examples
print(expert_weight_bytes(4096, 14336, "bfloat16"))   # 352321536 bytes
print(expert_weight_bytes(4096, 14336, "bfloat8_b"))  # 176160768 bytes
print(expert_weight_bytes(7168, 2048,  "bfloat16"))   # 88080384 bytes
print(expert_weight_bytes(7168, 2048,  "bfloat8_b"))  # 44040192 bytes
```

---

## Next Steps

This file concludes Chapter 3. You now have the full picture of expert weight tensor shapes, valid shard grid selection, and byte-accurate memory estimation.

Proceed to **Chapter 4** for the end-to-end weight loading and sharding workflow: reading checkpoint tensors, applying dtype conversion, padding to tile alignment, constructing ShardSpec objects, and placing tensors into L1-sharded memory configs ready for `ttnn.matmul` dispatch.
