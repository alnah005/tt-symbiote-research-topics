# Chapter 3: Expert Weight Tensor Structure

## Overview

This chapter examines the concrete tensor shapes that appear in Mixture-of-Experts (MoE) feed-forward network (FFN) layers, how those shapes map to valid shard grid configurations, and how data type (dtype) choice determines memory footprint. The material builds directly on the TTNN memory architecture concepts from Chapter 1 and the ShardSpec arithmetic from Chapter 2.

By the end of this chapter you will be able to reason from a model configuration (d_model, d_ff, num_experts) all the way to a specific sharding plan and a byte-accurate memory estimate.

---

## Learning Objectives

1. **State the shapes** of gate, up, and down expert weight projections and explain what each dimension represents.
2. **Use the reference table** to identify valid shard grid options for a given weight shape.
3. **Explain why TILE_LAYOUT is required** for `ttnn.matmul` and describe the tile-alignment constraint it imposes on shard shapes.
4. **Compute total per-expert weight memory** for `bfloat16` and `bfloat8_b` dtypes given a model's d_model and d_ff values.

---

## Prerequisites

| Chapter | Topics Covered |
|---|---|
| Chapter 1 | TTNN memory architecture, TILE_LAYOUT, tensor memory hierarchy |
| Chapter 2 | ShardSpec, shard shape arithmetic, CoreRangeSet construction |

Read both chapters before proceeding. Familiarity with the SwiGLU activation function is helpful but not required.

---

## Reference Table: Canonical Expert Weight Shapes

The following models are used as concrete examples throughout this chapter. All dimension values are in elements (not bytes).

| Model | d_model | d_ff | num_experts | top_k |
|---|---|---|---|---|
| Mixtral 8x7B | 4096 | 14336 | 8 | 2 |
| DeepSeek-MoE-16B | 2048 | 1408 | 64 | 6 |
| Qwen MoE (235B-A22B) | 7168 | 2048 | 128 | 8 |

**Column definitions:**
- `d_model` — model hidden dimension; width of the residual stream
- `d_ff` — expert feed-forward (FF) hidden dimension; the "expansion" dimension inside each expert
- `num_experts` — total number of expert FFN blocks
- `top_k` — number of experts activated per token during inference

> **Tip:** top_k determines the compute load per forward pass, but it does not change the per-expert weight shapes. Memory planning is driven by `d_model`, `d_ff`, and `num_experts` alone.

---

## Chapter Structure

| File | Contents |
|---|---|
| `projection_shapes.md` | Gate, up, and down weight projection shapes; stacked tensor conventions |
| `tensor_to_shard_grid_mapping.md` | Rules for selecting valid shard grids; worked examples |
| `dtype_and_tile_layout.md` | Dtype sizes, tile alignment requirements, memory footprint calculations |

---

## Quick Reference: Per-Expert Weight Dimensions

For a complete per-expert weight shape reference including w1/w3/w2 notation, see [`projection_shapes.md`](./projection_shapes.md).

---

## Next Steps

Continue to [`projection_shapes.md`](./projection_shapes.md) for a detailed breakdown of each projection's shape, the rationale behind the SwiGLU structure, and how stacked expert tensors are organized in TTNN.
