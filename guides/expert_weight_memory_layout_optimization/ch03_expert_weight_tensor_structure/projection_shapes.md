# Expert Weight Projection Shapes

## Overview

Each expert in a SwiGLU-gated MoE (Mixture-of-Experts) feed-forward layer contains three weight matrices. This file describes their shapes, the role of each dimension, and the conventions used when tensors are stacked across experts in TTNN (Tenstorrent Neural Network library).

---

## The SwiGLU Expert Structure

A single MoE expert computes:

```python
# SwiGLU expert forward pass
# x: input token, shape [seq_len, d_model]
# w1, w3: gate and up projection weights
# w2: down projection weight

gate = x @ w1   # [seq_len, d_model] @ [d_model, d_ff] -> [seq_len, d_ff]
up   = x @ w3   # [seq_len, d_model] @ [d_model, d_ff] -> [seq_len, d_ff]

# SiLU activation applied element-wise to gate, then element-wise multiply with up
hidden = ttnn.silu(gate) * up  # [seq_len, d_ff]

# Project back to model dimension
output = hidden @ w2  # [seq_len, d_ff] @ [d_ff, d_model] -> [seq_len, d_model]
```

---

## Gate Projection (w1)

**Weight shape:** `[d_model, d_ff]`

- **Rows (`d_model`):** Input features from the residual stream. Each row corresponds to one input dimension.
- **Columns (`d_ff`):** Output features in the expert hidden dimension. Each column learns a feature detector in the expanded space.
- **Role:** Maps the input token representation into the expert's hidden space. The result is passed through the SiLU (Sigmoid Linear Unit) activation function before the element-wise multiply.

---

## Up Projection (w3)

**Weight shape:** `[d_model, d_ff]`

- **Shape is identical to the gate projection.**
- **Role:** The parallel SwiGLU path. The up projection output is not activated; instead it modulates the activated gate output via element-wise multiplication: `output = SiLU(gate) * up`.
- This gating mechanism is the defining feature of SwiGLU and is why two `[d_model, d_ff]` matrices are required instead of one.

> **Tip:** Because w1 and w3 share the same shape, they are often concatenated into a single `[d_model, 2 * d_ff]` tensor during weight loading to reduce kernel dispatch overhead. Be aware of this optimization when inspecting checkpoint files — what appears as a single `w_gate_up` tensor is logically two projections.

---

## Down Projection (w2)

**Weight shape:** `[d_ff, d_model]`

- **Rows (`d_ff`):** Input features from the expert hidden dimension.
- **Columns (`d_model`):** Output features back in the residual stream dimension.
- **Role:** Reduces the expanded hidden representation back to d_model and adds the result into the residual stream.
- **Shape relationship:** The down projection is the transpose of the gate/up shape (`[d_ff, d_model]` vs `[d_model, d_ff]`). This asymmetry matters for sharding: a shard configuration valid for the gate projection is not automatically valid for the down projection.

---

## Shape Summary

| Projection | Notation | Shape | Direction |
|---|---|---|---|
| Gate | w1 | `[d_model, d_ff]` | d_model → d_ff |
| Up | w3 | `[d_model, d_ff]` | d_model → d_ff |
| Down | w2 | `[d_ff, d_model]` | d_ff → d_model |

---

## Stacked Expert Tensor Conventions

When all experts are stored on the same device or transferred together, their weights are often stacked into a single tensor.

**Stacked gate/up projection:**

```python
# Shape: [num_experts, d_model, d_ff]
# Axis 0 indexes the expert; axes 1 and 2 are the weight matrix.
gate_all_experts = ttnn.Tensor(shape=[num_experts, d_model, d_ff], ...)
```

**Stacked down projection:**

```python
# Shape: [num_experts, d_ff, d_model]
down_all_experts = ttnn.Tensor(shape=[num_experts, d_ff, d_model], ...)
```

### When to use stacked tensors

| Use Case | Recommendation |
|---|---|
| Small num_experts (≤ 8, e.g. Mixtral 8x7B) | Stacked tensor feasible; enables batch_matmul dispatch |
| Large num_experts (64–128, e.g. DeepSeek, Qwen) | Per-device or per-group storage preferred; stacking entire tensor may exceed DRAM capacity |
| Dynamic routing (top_k varies) | Per-expert tensors simplify dispatch; stacked form requires index-gather overhead |

> **Warning:** The stacked `[num_experts, d_model, d_ff]` form requires a `batch_matmul` or equivalent dispatch. Ensure your routing logic correctly maps token indices to expert indices before the matmul, or tokens will be multiplied against the wrong expert weights silently — no shape error will be raised.

---

## TTNN Storage Convention

TTNN typically stores expert weights as separate tensors per expert, or as per-group tensors covering the subset of experts assigned to a given device. The stacked form is supported but requires explicit handling:

For the canonical `ttnn.from_torch(..., dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)` loading pattern, including dtype options and tile-alignment requirements, see [`dtype_and_tile_layout.md`](./dtype_and_tile_layout.md).

```python
# Stacked storage (batch_matmul path)
# Requires careful dimension ordering and a matching batch_matmul call
gate_stacked = ttnn.from_torch(
    torch.stack([w1_i for w1_i in all_w1], dim=0),  # [num_experts, d_model, d_ff]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)
```

---

## Concrete Shapes: Mixtral 8x7B

- `d_model` = 4096, `d_ff` = 14336, `num_experts` = 8, `top_k` = 2

| Projection | Shape | Elements per expert |
|---|---|---|
| Gate (w1) | `[4096, 14336]` | 58,720,256 (~58.7M) |
| Up (w3) | `[4096, 14336]` | 58,720,256 (~58.7M) |
| Down (w2) | `[14336, 4096]` | 58,720,256 (~58.7M) |

All three projections have the same element count (the matrix is square in terms of total elements). The stacked gate/up tensor is `[8, 4096, 14336]`.

---

## Concrete Shapes: Qwen MoE (235B-A22B)

- `d_model` = 7168, `d_ff` = 2048, `num_experts` = 128, `top_k` = 8

| Projection | Shape | Elements per expert |
|---|---|---|
| Gate (w1) | `[7168, 2048]` | 14,680,064 (~14.7M) |
| Up (w3) | `[7168, 2048]` | 14,680,064 (~14.7M) |
| Down (w2) | `[2048, 7168]` | 14,680,064 (~14.7M) |

Despite having 16× more experts than Mixtral 8x7B, each Qwen expert is ~4× smaller in element count. The stacked gate/up tensor would be `[128, 7168, 2048]` — 128 expert slices.

> **Tip:** Qwen MoE's large num_experts (128) combined with its relatively small per-expert d_ff (2048) makes intra-expert sharding within each `[7168, 2048]` matrix the preferred strategy, rather than sharding across the expert dimension.

---

**Next:** [`tensor_to_shard_grid_mapping.md`](./tensor_to_shard_grid_mapping.md)
