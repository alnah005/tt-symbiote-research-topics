# Chapter 2 --- Projection Weights and Tensor Shapes

## Overview

This chapter derives the exact weight tensor shapes and activation tensor shapes
for every linear projection in the Gemma 4 31B text decoder. These shapes are
the foundation for TTNN `ttnn.linear` program configs, tensor-parallel sharding
decisions, and memory budget calculations.

Because Gemma 4 31B uses heterogeneous attention (two structurally different
layer types with different head counts and head dimensions), the attention
projection shapes vary by layer type. The FFN projections, by contrast, are
identical across all 60 layers.

After reading this chapter you will know:

- The exact weight shape of every Q, K, V, and O projection for both sliding
  and global layers.
- Why global layers have no V projection weight (K=V sharing).
- The weight shapes for the GeGLU gate, up, and down projections.
- The activation tensor shapes at each stage of the forward pass during
  single-token decode (batch=1).
- Why PLE contributes no projection weights in the 31B config.

## Reading Order

1. [`qkv_projections.md`](./qkv_projections.md) --- Q, K, V, and O projection
   weight shapes and decode activation shapes for both layer types.
2. [`ffn_projections.md`](./ffn_projections.md) --- GeGLU gate, up, and down
   projection weight shapes, uniform across all 60 layers.
3. [`ple_shapes.md`](./ple_shapes.md) --- Per-Layer Embedding tensor shapes
   (disabled in 31B but documented for completeness).

## Master Shape Table

The following table lists every linear projection in a single decoder layer,
with weight shapes given in PyTorch convention `[in_features, out_features]`.

### Attention Projections

| Projection | Weight Name | Sliding Shape | Global Shape | Notes |
|------------|-------------|---------------|--------------|-------|
| Q | `q_proj.weight` | [5376, 8192] | [5376, 16384] | 32 heads x head_dim |
| K | `k_proj.weight` | [5376, 4096] | [5376, 2048] | num_kv_heads x head_dim |
| V | `v_proj.weight` | [5376, 4096] | N/A | Global: K=V sharing, no V weight |
| O | `o_proj.weight` | [8192, 5376] | [16384, 5376] | Matches Q output dim |

### Attention Norm Parameters

| Norm | Weight Name | Sliding Shape | Global Shape | Notes |
|------|-------------|---------------|--------------|-------|
| Q-norm | `q_norm.weight` | [256] | [512] | Per-head dim, learned scale |
| K-norm | `k_norm.weight` | [256] | [512] | Per-head dim, learned scale |
| V-norm | (none) | --- | --- | No learned parameters (`with_scale=False`) |

### FFN Projections (Identical for All 60 Layers)

| Projection | Weight Name | Shape | Notes |
|------------|-------------|-------|-------|
| Gate | `mlp.gate_proj.weight` | [5376, 21504] | GeGLU gate path |
| Up | `mlp.up_proj.weight` | [5376, 21504] | GeGLU value path |
| Down | `mlp.down_proj.weight` | [21504, 5376] | Output projection |

### Layer-Level Norms (Identical for All 60 Layers)

| Norm | Weight Name | Shape |
|------|-------------|-------|
| Input layernorm | `input_layernorm.weight` | [5376] |
| Post-attention layernorm | `post_attention_layernorm.weight` | [5376] |
| Pre-FFN layernorm | `pre_feedforward_layernorm.weight` | [5376] |
| Post-FFN layernorm | `post_feedforward_layernorm.weight` | [5376] |

### PLE Projections (Disabled in 31B)

| Projection | Shape | Notes |
|------------|-------|-------|
| Per-layer embedding | N/A | `hidden_size_per_layer_input=0` |
| Per-layer gate | N/A | Not instantiated |
| Per-layer projection | N/A | Not instantiated |

## Per-Layer Parameter Counts

### Sliding Layer (BF16)

| Component | Parameters | Bytes (BF16) |
|-----------|-----------|--------------|
| Q projection | 5376 x 8192 = 44,040,192 | 88,080,384 |
| K projection | 5376 x 4096 = 22,020,096 | 44,040,192 |
| V projection | 5376 x 4096 = 22,020,096 | 44,040,192 |
| O projection | 8192 x 5376 = 44,040,192 | 88,080,384 |
| Gate projection | 5376 x 21504 = 115,605,504 | 231,211,008 |
| Up projection | 5376 x 21504 = 115,605,504 | 231,211,008 |
| Down projection | 21504 x 5376 = 115,605,504 | 231,211,008 |
| Norms (6 total) | 5376 x 4 + 256 x 2 = 22,016 | 44,032 |
| **Total** | **478,959,104** | **~958 MB** |

### Global Layer (BF16)

| Component | Parameters | Bytes (BF16) |
|-----------|-----------|--------------|
| Q projection | 5376 x 16384 = 88,080,384 | 176,160,768 |
| K projection | 5376 x 2048 = 11,010,048 | 22,020,096 |
| V projection | N/A (K=V sharing) | 0 |
| O projection | 16384 x 5376 = 88,080,384 | 176,160,768 |
| Gate projection | 5376 x 21504 = 115,605,504 | 231,211,008 |
| Up projection | 5376 x 21504 = 115,605,504 | 231,211,008 |
| Down projection | 21504 x 5376 = 115,605,504 | 231,211,008 |
| Norms (6 total) | 5376 x 4 + 512 x 2 = 22,528 | 45,056 |
| **Total** | **534,009,856** | **~1,068 MB** |

### Full Model Weight Budget

$$
50 \times 478{,}959{,}104 + 10 \times 534{,}009{,}856 + 262{,}144 \times 5{,}376 = 30{,}697{,}339{,}904 \approx 30.7\text{B params}
$$

At BF16, total weight memory is approximately **57.3 GB** (before any
quantization). See [Chapter 8](../ch8_performance/index.md) for the memory
budget analysis and quantization strategy.
