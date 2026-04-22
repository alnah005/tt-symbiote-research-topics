# Vision Components — TTNN Implementation

## Overview

This file walks through each TTNN module in the dots.ocr vision encoder, from the patch embedding layer that converts raw pixel patches into hidden-state vectors, through the 42 transformer blocks with their post-norm arrangement, to the `VisionTransformerTT` orchestrator that ties them together. The emphasis is on design choices that differ from a standard ViT port: the post-norm ordering imposed by `post_norm=True`, the SwiGLU activation formula in `VisionMLPTT`, and the 2D RoPE positional encoding in `VisionAttentionTT`.

All paths are relative to `models/demos/dots_ocr/tt/` unless noted otherwise.

---

## Configuration Plumbing: `DotsVisionModelArgs`

Before covering the individual components, it is useful to understand how configuration is threaded through the vision stack.

### `vision_config_dataclass.py`

Defines the configuration dataclass that holds the values parsed from `vision_config` in `config.json`. The fields map directly to the JSON keys: `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `patch_size`, `spatial_merge_size`, `temporal_patch_size`, `post_norm`, `rms_norm_eps`, and `use_bias`. There is no `num_key_value_heads` field — the vision encoder uses standard full multi-head attention with no GQA.

### `vision_model_config.py` — `DotsVisionModelArgs`

`DotsVisionModelArgs` is the TTNN-specific configuration object for the vision encoder. It wraps the dataclass values and adds the hardware-level fields required to build TTNN tensors:

- `mesh_device` — the `MeshDevice` (or `None` for CPU fallback) on which all vision tensors are allocated.
- `dtype` — the primary computation dtype; BFLOAT16 is used throughout the vision stack.
- `state_dict` — the full vision encoder state dict, provided by `load_dots_vision_state_dict()` in `tt/load.py`.
- Weight layout constants — `TILE_LAYOUT` and `BFLOAT16` are the standard TTNN layout and dtype for the vision encoder's weight matrices.

`DotsVisionModelArgs` is instantiated once and passed to every vision sub-module constructor. This avoids scattering `mesh_device` and `dtype` arguments across every call site.

---

## `PatchEmbedTT` (`vision_patch_embed.py`)

### Role

`PatchEmbedTT` maps a batch of image patches to the vision encoder's hidden space. It is the entry point of the TTNN vision stack, receiving patch tensors and producing the first hidden-state tensor at dimension `hidden_size=1536`.

### Patch Convolution as TTNN Matmul

The reference implementation of patch embedding uses a 2D convolution with kernel size `patch_size×patch_size=14×14` and stride 14 (non-overlapping patches). On TT hardware, a 14×14 convolution over a large spatial grid is more efficiently expressed as a matrix multiplication:

Each patch of shape $(14, 14, 3)$ is flattened to a vector of length $14 \times 14 \times 3 = 588$. The full set of patches for one image forms a matrix of shape `[S_patch, 588]`, where `S_patch = (H/14) × (W/14)` is the number of patches. The patch embedding projection is then:

$$\text{output} = \text{input} \cdot W_{embed}^T \quad W_{embed} \in \mathbb{R}^{1536 \times 588}$$

In TTNN, this is expressed as a single `ttnn.matmul` with the weight matrix $W_{embed}$ stored in `TILE_LAYOUT` / `BFLOAT16`. The output is `[B, 1, S_patch, 1536]`, with the sequence batch axis (`1`) inserted for compatibility with the downstream block tensor layout.

### Handling `grid_thw`

The processor provides `grid_thw` — a tensor of shape `[num_images, 3]` giving the temporal, height, and width patch counts `(T, H_p, W_p)` for each image in the batch. For static images, `T=1` always. `PatchEmbedTT` uses `grid_thw` to reshape the flat patch sequence back into a spatial grid for the 2D RoPE computation in `VisionAttentionTT`. The embed computation itself is agnostic to spatial arrangement; the reshape is performed after the matmul.

### Weight Layout

The patch embedding weight is stored in `TILE_LAYOUT` / `BFLOAT16` in the `DotsVisionModelArgs.state_dict`. `PatchEmbedTT` loads it once at construction time and retains the TTNN tensor for use in each forward call. There is no bias (`use_bias=false` in `vision_config`).

---

## `VisionBlockTT` (`vision_block.py`) — Post-Norm Architecture

### Role

`VisionBlockTT` is one of the 42 repeated transformer encoder blocks in the vision stack. Its structure follows the `post_norm=True` configuration: RMSNorm is applied **after** the residual addition, not before the sublayer computation.

### Post-Norm vs Pre-Norm

Standard ViT blocks (e.g., ViT-B/16) use pre-norm — LayerNorm is applied to the input before the attention or MLP sublayer, inside the residual branch:

$$x \leftarrow x + \text{Attention}(\text{LN}(x))$$
$$x \leftarrow x + \text{MLP}(\text{LN}(x))$$

dots.ocr's vision encoder uses post-norm. The residual addition is performed first, and RMSNorm is applied to the result:

$$x \leftarrow \text{RMSNorm}(x + \text{Attention}(x))$$
$$x \leftarrow \text{RMSNorm}(x + \text{MLP}(x))$$

The implication for the TTNN kernel graph is that the RMSNorm op follows the `ttnn.add` rather than preceding the attention or MLP projections. Each block contains exactly two RMSNorm operations in both arrangements, but the insertion point in the data-flow DAG is different. In the post-norm arrangement, the normalization output feeds directly into the **next** block's input (or into the post-trunk norm), not into the current block's attention QKV projections.

> **Note:** The ordering shown above is the contract in `VisionBlockTT`. Code that swaps the residual-add and RMSNorm order will pass shape checks but produce numerically wrong outputs. The `test_vision_pcc.py` test is the primary guard against this inversion.

### Forward Pass Skeleton

```python
# Attention sublayer (post-norm)
attn_out = self.attention(x)          # VisionAttentionTT forward
x = ttnn.add(x, attn_out)            # residual add
x = self.norm1(x)                    # RMSNorm AFTER residual

# MLP sublayer (post-norm)
mlp_out = self.mlp(x)                # VisionMLPTT forward
x = ttnn.add(x, mlp_out)             # residual add
x = self.norm2(x)                    # RMSNorm AFTER residual
```

`norm1` and `norm2` are instances of `VisionRMSNorm` (see below), both configured with `rms_norm_eps=1e-05`.

---

## `VisionAttentionTT` (`vision_attention.py`)

### Role

`VisionAttentionTT` implements the multi-head self-attention within each `VisionBlockTT`. The vision encoder uses standard full multi-head attention — there is no GQA in the vision encoder. All 12 heads serve as both query and key/value heads.

### Configuration

| Field | Value |
|-------|-------|
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 12 (full attention, no GQA) |
| `head_dim` | $1536 / 12 = 128$ |
| `use_bias` | `false` — no bias on Q, K, V, O projections |

This contrasts with the text decoder, which has GQA 12Q/2KV and `attention_bias=True`. The vision encoder is the simpler of the two attention configurations.

### TTNN Scaled Dot-Product Attention

The forward pass projects input `x` through Q, K, V weight matrices (each `[1536, 1536]`, no bias), reshapes to `[B, num_heads, S_patch, head_dim]`, and computes attention using `ttnn.transformer.scaled_dot_product_attention`. This is the standard TTNN fused attention kernel, which fuses the QK matmul, softmax, and V matmul into a single kernel dispatch. No explicit causal mask is applied — the vision encoder uses bidirectional (non-causal) self-attention over all patch tokens simultaneously.

The output projection (O matrix, `[1536, 1536]`, no bias) is applied after the attention heads are concatenated.

### 2D RoPE Positional Encoding

The vision encoder uses 2D Rotary Position Embedding (RoPE) to encode the spatial position of each patch within the image grid. This is distinct from the text decoder's 1D causal RoPE.

In 2D RoPE (following the Qwen2-VL convention), each patch at grid coordinate $(r, c)$ receives a composite positional encoding. The query and key head dimensions are split into two equal halves:

- The first half encodes the row coordinate $r$ (height).
- The second half encodes the column coordinate $c$ (width).

Each half applies standard 1D RoPE over its respective coordinate axis. The result is a full rotary encoding that respects both spatial dimensions independently.

The `grid_thw` tensor from `PatchEmbedTT` provides the `(H_p, W_p)` grid dimensions needed to generate the per-patch $(r, c)$ coordinates. `VisionAttentionTT` consumes these coordinates to construct the cos/sin rotation matrices before the QK attention computation.

> **Note:** `reference/rope.py` provides `Qwen2RopeHelper` for the text decoder's 1D RoPE. The vision encoder's 2D RoPE is a separate implementation; do not apply the text decoder's RoPE helper to the vision encoder.

---

## `VisionMLPTT` (`vision_mlp.py`) — SwiGLU

### Role

`VisionMLPTT` implements the feed-forward sublayer within each `VisionBlockTT`. It uses the SwiGLU activation function, the same gated MLP design used in the text decoder.

### Architecture: Three Projection Matrices

SwiGLU MLP uses three linear projections, conventionally named:

| Matrix | Role | Shape |
|--------|------|-------|
| `fc1` (gate) | Gating branch input | $1536 \times 4224$ |
| `fc3` (up) | Value branch input | $1536 \times 4224$ |
| `fc2` (down) | Output projection | $4224 \times 1536$ |

All three are square-free, bias-free (`use_bias=false`), and stored in `TILE_LAYOUT` / `BFLOAT16`.

### SwiGLU Formula

The forward pass is:

$$y = \text{fc2}\bigl(\text{SiLU}(\text{fc1}(x)) \cdot \text{fc3}(x)\bigr)$$

where $\cdot$ denotes element-wise multiplication. Expanded:

1. Compute gate: $g = \text{SiLU}(\text{fc1}(x)) = \text{fc1}(x) \cdot \sigma(\text{fc1}(x))$, where $\sigma$ is the sigmoid function.
2. Compute up: $u = \text{fc3}(x)$
3. Element-wise product: $h = g \cdot u$
4. Project down: $y = \text{fc2}(h)$

In TTNN, this is expressed as:

```python
gate = ttnn.silu(ttnn.linear(x, self.fc1_weight))   # [B, 1, S, 4224]
up   = ttnn.linear(x, self.fc3_weight)               # [B, 1, S, 4224]
h    = ttnn.mul(gate, up)                            # element-wise
y    = ttnn.linear(h, self.fc2_weight)               # [B, 1, S, 1536]
```

> **Note:** The matrix naming convention `fc1`/`fc2`/`fc3` follows the HuggingFace checkpoint key names for the dots.ocr vision MLP. `fc1` is the gate branch, `fc3` is the up branch, and `fc2` is the down projection — not the sequential order the names imply. Swapping `fc1` and `fc3` produces wrong gating behavior without a shape error because both matrices have the same shape `[1536, 4224]`. The `test_vision_components.py` PCC test is the guard against this swap.

### Intermediate Dimension

`intermediate_size=4224` is the expansion width. The ratio to `hidden_size` is $4224/1536 = 2.75$, which is smaller than the standard $4\times$ ViT feed-forward ratio. This reflects the compact MLP design choice described in Chapter 1.

---

## `VisionRMSNorm` (`vision_rmsnorm.py`)

### Role

`VisionRMSNorm` is a thin wrapper around `ttnn.rms_norm` (or the equivalent TTNN RMSNorm op), used in two contexts:

1. **Within `VisionBlockTT`** — as `norm1` (after attention sublayer) and `norm2` (after MLP sublayer) in each of the 42 blocks. In the post-norm arrangement these are the only normalization operations within a block.
2. **Post-trunk normalization** — applied once by `VisionTransformerTT` after the final (42nd) block, before the output is passed to `PatchMergerTT`.

All instances use `rms_norm_eps=1e-05`. This is distinct from the text decoder's `rms_norm_eps=1e-06`. The different epsilon values must not be mixed — loading text decoder RMSNorm weights with vision encoder epsilon settings or vice versa would produce small but measurable PCC degradation.

RMSNorm has no bias (`use_bias=false`). Each instance holds a scale vector of shape `[hidden_size]` = `[1536]`, loaded from the vision state dict.

---

## `VisionTransformerTT` (`vision.py`) — Orchestrator

### Role

`VisionTransformerTT` is the top-level TTNN vision encoder object. It owns the full vision stack from patch embedding to the post-trunk normalization, and exposes a `forward()` method that takes pixel patch data and returns vision feature tokens ready for `PatchMergerTT`.

### Initialization and Submodule Construction

At construction time, `VisionTransformerTT`:

1. Instantiates `PatchEmbedTT` from `DotsVisionModelArgs`.
2. Constructs a list of 42 `VisionBlockTT` instances, each initialized with the matching slice of the vision state dict (block-indexed weight keys from `load_dots_vision_state_dict()`).
3. Instantiates the post-trunk `VisionRMSNorm`.

All TTNN weight tensors are allocated on `mesh_device` at construction time and remain resident on device for the lifetime of the encoder object. The vision stack does not support hot-swapping weights.

### Forward Pass

See the component flow diagram in [`index.md`](./index.md) for the full stage-by-stage pipeline with tensor shapes.

### Mode Switching

For the hybrid/full-TTNN mode-switching logic and rationale, see [`index.md`](./index.md).

---

**Next:** [`patch_merger_and_fusion.md`](./patch_merger_and_fusion.md)
