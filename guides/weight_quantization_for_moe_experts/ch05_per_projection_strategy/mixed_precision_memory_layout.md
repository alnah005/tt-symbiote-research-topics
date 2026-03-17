# Mixed-Precision Memory Layout

## Storing Three Projections with Different Dtypes

A mixed-precision MoE expert module holds three weight tensors per expert, each with a
distinct dtype. All three are stored in DRAM — expert weights are too large to reside
permanently in L1. The layout conventions below apply to a single device in a T3K mesh,
where each device holds a subset of the total experts.

### Tensor Shapes and Dtypes

For Qwen 235B-A22B: `d_model = 7168`, `d_ff = 2048`, `num_experts = 128` total,
`num_experts_per_device = 16` (on an 8-chip T3K mesh with uniform expert distribution).

| Tensor | Shape | Dtype | Memory config |
|---|---|---|---|
| `w1_experts` (gate) | `[num_experts_per_device, d_model, d_ff]` | `ttnn.bfloat4_b` | `ttnn.DRAM_MEMORY_CONFIG` |
| `w3_experts` (up) | `[num_experts_per_device, d_model, d_ff]` | `ttnn.bfloat4_b` | `ttnn.DRAM_MEMORY_CONFIG` |
| `w2_experts` (down) | `[num_experts_per_device, d_ff, d_model]` | `ttnn.bfloat8_b` | `ttnn.DRAM_MEMORY_CONFIG` |

The shapes reflect the transposition convention: weights are stored with the output
dimension first so that `ttnn.linear(x, w)` computes `x @ w.T` without an additional
transpose op. For gate and up projections the weight is `[d_ff, d_model]` stored as
the inner two dimensions of the per-device expert stack. For the down projection the
weight is `[d_model, d_ff]` — transposed relative to gate/up — because the down
projection maps `[batch, d_ff] -> [batch, d_model]`.

> **Warning:** The shape convention shown above (`w1_experts`: `[num_experts_per_device,
> d_model, d_ff]` and `w2_experts`: `[num_experts_per_device, d_ff, d_model]`) follows
> the transposed storage convention where the last two dimensions are the transposed
> weight matrix. Verify the dimension ordering used by your matmul program config before
> loading weights; inconsistent transposition is a common source of incorrect output
> that does not manifest as a shape error.

## Tile Memory Sizes

TTNN's tile unit is 32×32 elements. Tile sizes are 2048 B (bfloat16), 1024 B (bfloat8_b), and 512 B (bfloat4_b) per 32×32 tile (see Chapter 1, `bfloat16_format.md`). These tile sizes are the foundation for all footprint calculations below.

## DRAM Footprint per Expert

For Qwen (d_model=7168, d_ff=2048), the mixed-precision footprint is 2 × d_model × d_ff = 29,360,128 bytes ≈ 28.0 MB per expert, versus 84.0 MB at full bfloat16 — a 3× reduction. See Chapter 1, `bfloat4_b_format.md` for the byte-level derivation.

## Total Expert Weight Memory on a T3K System

For Qwen 235B-A22B: 128 total experts, 8 chips, 16 experts per chip.

**Per chip, mixed precision:**

```
16 experts × 28.00 MB/expert = 448 MB per chip
```

**All 8 chips combined (total expert parameter memory), mixed precision:**

```
128 experts × 28.00 MB/expert = 3,584 MB ≈ 3.50 GB
```

**All 8 chips combined, bfloat16 baseline:**

```
128 experts × 84.00 MB/expert = 10,752 MB ≈ 10.50 GB
```

**Per-chip DRAM headroom recovered by mixed precision: ~448 MB → frees ~672 MB per chip
compared to bfloat16.**

Each Wormhole B0 chip has 12 GB of DRAM. The 3.50 GB for expert weights under mixed
precision is well within budget, leaving ample room for attention weights, KV cache,
activations, and the non-expert FFN parameters.

## Tile Alignment Constraint

`bfloat4_b` requires that weight tensor dimensions be multiples of 32 (one tile side).
For Qwen 235B-A22B:

- `d_model = 7168 = 224 × 32` — tile-aligned.
- `d_ff = 2048 = 64 × 32` — tile-aligned.

Both dimensions are tile-aligned for Qwen 235B-A22B, so no padding is required. For
models where either dimension is not a multiple of 32, pad to the nearest tile boundary
before calling `ttnn.as_tensor` with `bfloat4_b`. Padding with zeros is safe for weight
matrices; remove the corresponding output slices if the padding affects the output shape.

> **Warning:** Misaligned dimensions cause incorrect tile packing for bfloat4_b. TTNN
> will not always raise an explicit error for misaligned shapes at conversion time; the
> failure may appear as subtly incorrect matmul outputs. Always verify tile alignment
> before converting to `bfloat4_b`.

## Code Pattern

```python
import ttnn
import torch

def load_mixed_precision_expert_weights(
    w1_list, w3_list, w2_list, device, num_experts_per_device
):
    """Load gate, up, and down expert weights with per-projection dtypes.

    Args:
        w1_list: List of gate weight tensors (one per expert), each [d_ff, d_model] bfloat16.
        w3_list: List of up weight tensors (one per expert), each [d_ff, d_model] bfloat16.
        w2_list: List of down weight tensors (one per expert), each [d_model, d_ff] bfloat16.
        device: TTNN device or mesh device.
        num_experts_per_device: Number of experts assigned to this device.

    Returns:
        (w1_tt, w3_tt, w2_tt): TTNN tensors for gate, up, and down projections.
    """
    # Stack per-expert weights into a single batched tensor for efficient dispatch
    w1_stacked = torch.stack(w1_list, dim=0)   # [num_experts_per_device, d_ff, d_model]
    w3_stacked = torch.stack(w3_list, dim=0)
    w2_stacked = torch.stack(w2_list, dim=0)   # [num_experts_per_device, d_model, d_ff]

    # Gate (w1): bfloat4_b — 4× memory reduction; SiLU absorbs quantization noise
    w1_tt = ttnn.as_tensor(
        w1_stacked,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Up (w3): bfloat4_b — 4× memory reduction; SwiGLU product dilutes uncorrelated errors
    w3_tt = ttnn.as_tensor(
        w3_stacked,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Down (w2): bfloat8_b — 2× memory reduction; higher precision required for residual stream
    w2_tt = ttnn.as_tensor(
        w2_stacked,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return w1_tt, w3_tt, w2_tt
```

## Summary Table

| Tensor | Shape (Qwen 235B-A22B, per device) | Dtype | Bytes/elem | Per-device size |
|---|---|---|---|---|
| `w1_experts` | `[16, 7168, 2048]` | `bfloat4_b` | 0.5 | 112 MB |
| `w3_experts` | `[16, 7168, 2048]` | `bfloat4_b` | 0.5 | 112 MB |
| `w2_experts` | `[16, 2048, 7168]` | `bfloat8_b` | 1.0 | 224 MB |
| **Total (mixed)** | — | — | — | **448 MB** |
| **Total (bfloat16)** | — | — | 2.0 | **1,344 MB** |
| **Reduction** | — | — | — | **3.0×** |

---

**Next:** [`qwen_adaptation_guide.md`](./qwen_adaptation_guide.md)
