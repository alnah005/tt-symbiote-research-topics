# Module Hierarchy and Data Flow

This file maps every module in the Gemma 4 vision pipeline, from raw pixel input through to the soft tokens consumed by the language model. Understanding this hierarchy is the foundation for planning TTNN operator coverage.

## Top-Level Structure

The vision pipeline spans two top-level modules:

```
Gemma4VisionModel
  ├── Gemma4VisionPatchEmbedder
  ├── Gemma4VisionEncoder
  │     ├── Gemma4VisionRotaryEmbedding
  │     └── Gemma4VisionEncoderLayer  ×27
  │           ├── Gemma4VisionAttention
  │           └── Gemma4VisionMLP
  └── Gemma4VisionPooler

Gemma4MultimodalEmbedder  (outside vision model, in the main Gemma4ForConditionalGeneration)
  ├── Gemma4RMSNorm  (without learnable scale)
  └── nn.Linear  (1152 → 5376)
```

## Gemma4VisionPatchEmbedder

The patch embedder converts raw image patches into hidden-state vectors and adds 2D positional information.

### Layers

| Layer | Type | Shape |
|-------|------|-------|
| `input_proj` | `nn.Linear` (no bias) | [3 * 16 * 16, 1152] = [768, 1152] |
| `position_embedding_table` | `nn.Parameter` | [2, 10240, 1152] |

### Forward Pass

**Inputs:**

- `pixel_values`: `[batch, num_patches, 768]` — each patch is a flattened 16x16x3 RGB tile
- `pixel_position_ids`: `[batch, num_patches, 2]` — (x, y) grid coordinates for each patch; padded patches use (-1, -1)
- `padding_positions`: `[batch, num_patches]` — boolean mask, `True` for padding patches

**Computation:**

1. **Value scaling**: `pixel_values = 2 * (pixel_values - 0.5)` — rescales from [0, 1] to [-1, 1]. There is no ImageNet mean/std normalization.
2. **Linear projection**: `hidden_states = input_proj(pixel_values)` producing `[batch, num_patches, 1152]`.
3. **Positional embeddings**: for each spatial dimension (x and y), the position IDs are one-hot encoded against the embedding table of size 10240, then matrix-multiplied with the corresponding row of `position_embedding_table`. The two dimensions are summed to produce a single `[batch, num_patches, 1152]` positional tensor. Padding positions are zeroed out.
4. **Addition**: `output = hidden_states + position_embeddings`.

> **Tip:** The positional embedding uses a one-hot lookup rather than `nn.Embedding`. This means position IDs are integer grid coordinates, not sequential token indices. The table size of 10240 sets the maximum number of distinct positions per spatial axis.

### TTNN Porting Notes

- `input_proj` is a standard bias-free linear: maps directly to `ttnn.linear`.
- The one-hot positional embedding computation involves `F.one_hot`, permute, batch matmul, and sum. This sequence may need to be fused or replaced with a gather-based approach in TTNN.

## Gemma4VisionEncoder

The encoder is a stack of 27 identical transformer layers, preceded by a shared 2D RoPE computation.

### Layers

| Layer | Type | Details |
|-------|------|---------|
| `rotary_emb` | `Gemma4VisionRotaryEmbedding` | Computes cos/sin tables from (x, y) position IDs |
| `layers[0..26]` | `Gemma4VisionEncoderLayer` | 27 identical encoder blocks |

### Forward Pass

1. A **bidirectional attention mask** is created from the padding mask (non-causal; all valid tokens attend to all other valid tokens).
2. **RoPE embeddings** are computed once from `pixel_position_ids` and reused across all 27 layers.
3. Each encoder layer is applied sequentially.

## Gemma4VisionRotaryEmbedding

This module computes 2D rotary position embeddings by processing the x and y coordinates independently and concatenating the results.

### RoPE Parameter Computation

The spatial dimension for each axis is:

$$
d_{\text{spatial}} = \frac{d_{\text{head}}}{2} = \frac{72}{2} = 36
$$

Inverse frequencies are computed as:

$$
f_i = \frac{1}{\theta^{2i / d_{\text{spatial}}}} \quad \text{for } i = 0, 1, \ldots, 17
$$

where $\theta = 100.0$ (the `rope_theta` parameter).

### Forward Pass

For each spatial dimension $k \in \{0, 1\}$ (x and y):

1. Extract positions: `dim_position_ids = position_ids[:, :, k]`
2. Compute frequencies: `freqs = inv_freq @ dim_position_ids` giving `[batch, 18, num_patches]`
3. Transpose: `freqs = freqs.transpose(1, 2)` giving `[batch, num_patches, 18]`
4. Double the frequencies: `emb = cat(freqs, freqs)` giving `[batch, num_patches, 36]`
5. Compute `cos_k = cos(emb)` and `sin_k = sin(emb)`

The final cos and sin tensors are concatenated across both dimensions:

- `cos = cat(cos_x, cos_y)` with shape `[batch, num_patches, 72]`
- `sin = cat(sin_x, sin_y)` with shape `[batch, num_patches, 72]`

### Multidimensional RoPE Application

The `apply_multidimensional_rope` function splits query/key tensors along the head dimension into per-axis chunks, applies standard RoPE to each chunk with the corresponding cos/sin slice, and concatenates the results:

```python
# x has shape [batch, seq_len, num_heads, head_dim=72]
# Split into 2 chunks of 36 along head_dim
x_parts = torch.split(x, [36, 36], dim=-1)
cos_parts = torch.split(cos, [36, 36], dim=-1)
sin_parts = torch.split(sin, [36, 36], dim=-1)

# Apply standard RoPE to each spatial dimension independently
y_x = apply_rotary_pos_emb(x_parts[0], cos_parts[0], sin_parts[0])
y_y = apply_rotary_pos_emb(x_parts[1], cos_parts[1], sin_parts[1])

output = torch.cat([y_x, y_y], dim=-1)  # [batch, seq_len, num_heads, 72]
```

> **Warning:** The 2D RoPE here differs from the standard 1D RoPE used in the language model. When porting, the RoPE kernel must handle 2D position IDs `[batch, seq_len, 2]` rather than scalar position indices.

## Gemma4VisionEncoderLayer

Each of the 27 encoder layers follows a pre-norm residual pattern with four RMSNorm layers (two bracketing attention, two bracketing the MLP).

### Layers

| Layer | Type | Parameters |
|-------|------|------------|
| `input_layernorm` | `Gemma4RMSNorm` | dim=1152, eps=1e-6, learnable scale |
| `self_attn` | `Gemma4VisionAttention` | See below |
| `post_attention_layernorm` | `Gemma4RMSNorm` | dim=1152, eps=1e-6, learnable scale |
| `pre_feedforward_layernorm` | `Gemma4RMSNorm` | dim=1152, eps=1e-6, learnable scale |
| `mlp` | `Gemma4VisionMLP` | See below |
| `post_feedforward_layernorm` | `Gemma4RMSNorm` | dim=1152, eps=1e-6, learnable scale |

### Forward Pass

```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states, position_embeddings, mask, position_ids)
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

> **Tip:** This is a "sandwich norm" pattern — normalization both before and after the sub-layer, with the residual added after the post-norm. This differs from the more common pre-norm-only pattern (where normalization is applied only before the sub-layer). The post-norms act as additional stabilizers on the residual stream.

## Gemma4VisionAttention

Multi-head self-attention with Q/K/V normalization and 2D RoPE.

### Layers

| Layer | Type | Shape |
|-------|------|-------|
| `q_proj` | `Gemma4ClippableLinear` | [1152, 16 * 72] = [1152, 1152] |
| `k_proj` | `Gemma4ClippableLinear` | [1152, 16 * 72] = [1152, 1152] |
| `v_proj` | `Gemma4ClippableLinear` | [1152, 16 * 72] = [1152, 1152] |
| `o_proj` | `Gemma4ClippableLinear` | [1152, 1152] |
| `q_norm` | `Gemma4RMSNorm` | dim=72, with learnable scale |
| `k_norm` | `Gemma4RMSNorm` | dim=72, with learnable scale |
| `v_norm` | `Gemma4RMSNorm` | dim=72, without learnable scale |

### Forward Pass

1. **Project** Q, K, V: each `[batch, seq_len, 1152]` reshaped to `[batch, seq_len, 16, 72]`
2. **Normalize** Q, K per-head with learnable RMSNorm; V with scale-free RMSNorm
3. **Apply 2D RoPE** to Q and K using multidimensional rotation (not applied to V)
4. **Transpose** to `[batch, 16, seq_len, 72]` for attention computation
5. **Attention**: standard scaled dot-product with bidirectional mask, `scaling=1.0` (no $1/\sqrt{d}$ — the QK norms serve as the scaling mechanism)
6. **Output projection**: reshape back to `[batch, seq_len, 1152]` and apply `o_proj`

> **Warning:** The attention scaling factor is `1.0`, not the usual `1/sqrt(head_dim)`. This is because Q and K are RMS-normalized per head, which implicitly controls the magnitude of the dot products. The TTNN attention kernel must account for this non-standard scaling.

### TTNN Porting Notes

- `Gemma4ClippableLinear` wraps `nn.Linear(bias=False)` with optional input/output clamping. For the 31B model, `use_clipped_linears=False`, so these are plain bias-free linears mapping to `ttnn.linear`.
- Q/K/V normalization before RoPE is unusual and requires per-head RMSNorm. This may need a custom or fused TTNN op.

## Gemma4VisionMLP

A gated MLP with GeLU activation (the `gelu_pytorch_tanh` variant, which is the tanh-approximated GeLU).

### Layers

| Layer | Type | Shape |
|-------|------|-------|
| `gate_proj` | `Gemma4ClippableLinear` | [1152, 4304] |
| `up_proj` | `Gemma4ClippableLinear` | [1152, 4304] |
| `down_proj` | `Gemma4ClippableLinear` | [4304, 1152] |

### Forward Pass

```python
output = down_proj(gelu(gate_proj(x)) * up_proj(x))
```

This is the standard SwiGLU-style gated linear unit, but using GeLU instead of SiLU as the activation function:

$$
\text{MLP}(x) = W_{\text{down}} \left( \text{GeLU}(W_{\text{gate}} x) \odot W_{\text{up}} x \right)
$$

### TTNN Porting Notes

- Three bias-free linears and one element-wise multiply: standard `ttnn.linear` and `ttnn.mul`.
- The `gelu_pytorch_tanh` activation maps to the tanh-approximated GeLU: `ttnn.gelu` with `approximate="tanh"`.

## Gemma4VisionPooler

The pooler reduces the spatial token count via 2D average pooling within kernel-sized grid cells and scales the result.

### Forward Pass

Given:
- `hidden_states`: `[batch, num_patches, 1152]`
- `output_length`: target number of output tokens (e.g., 280)

1. **Compute kernel size**: $k = \sqrt{\text{num\_patches} / \text{output\_length}}$. For the default token budget of 280 with `pooling_kernel_size=3`, the encoder output has $280 \times 9 = 2520$ patches and $k = 3$.
2. **Grid assignment**: divide each (x, y) position by $k$ (floor division) to assign patches to grid cells.
3. **Average pooling**: one-hot encode the grid cell assignments, normalize by $k^2$, and compute a weighted sum. This produces `[batch, output_length, 1152]`.
4. **Scaling**: multiply by $\sqrt{1152} \approx 33.94$.
5. **Mask computation**: produce a boolean validity mask for downstream padding removal.

> **Tip:** The pooler's $\sqrt{d}$ scaling compensates for the averaging operation that reduces hidden-state magnitudes. This is analogous to the $\sqrt{d}$ factor in attention but applied to the pooled representations.

### TTNN Porting Notes

- The grid-based pooling is not a standard 2D average pool (like `nn.AvgPool2d`) because the patches are arranged in a 1D sequence with explicit (x, y) coordinates. A custom TTNN kernel or a sequence of gather/scatter/reduce ops may be needed.

## Gemma4VisionModel — Complete Forward Pass

Putting it all together:

```
Inputs:
  pixel_values:       [batch, num_patches, 768]
  pixel_position_ids: [batch, num_patches, 2]

Step 1: Compute padding mask
  padding_positions = (pixel_position_ids == -1).all(dim=-1)

Step 2: Patch embedding
  hidden_states = PatchEmbedder(pixel_values, pixel_position_ids, padding_positions)
  → [batch, num_patches, 1152]

Step 3: Encoder (27 layers with 2D RoPE)
  hidden_states = Encoder(hidden_states, ~padding_positions, pixel_position_ids)
  → [batch, num_patches, 1152]

Step 4: Adaptive pooling
  output_length = num_patches // (pooling_kernel_size^2)
  hidden_states, mask = Pooler(hidden_states, pixel_position_ids, padding_positions, output_length)
  → [batch, output_length, 1152]

Step 5: Remove padding
  hidden_states = hidden_states[mask]
  → [total_valid_tokens, 1152]

Step 6: Standardization (optional)
  hidden_states = (hidden_states - std_bias) * std_scale
  → [total_valid_tokens, 1152]
```

## Gemma4MultimodalEmbedder — Projection to Language Model

After the vision model produces soft tokens, the `Gemma4MultimodalEmbedder` projects them into the language model's embedding space.

### Layers

| Layer | Type | Shape |
|-------|------|-------|
| `embedding_pre_projection_norm` | `Gemma4RMSNorm` | dim=1152, eps=1e-6, without learnable scale |
| `embedding_projection` | `nn.Linear` (no bias) | [1152, 5376] |

### Forward Pass

```python
embs_normed = rms_norm(inputs_embeds)    # [total_valid_tokens, 1152]
output = linear(embs_normed)             # [total_valid_tokens, 5376]
```

The projected soft tokens are then inserted into the language model's token embedding sequence at the positions marked by the image placeholder tokens (`<image>`, token ID 129090).

## Complete End-to-End Data Flow Summary

$$
\text{raw pixels} \xrightarrow{\text{patch + project}} [B, N, 1152] \xrightarrow{\text{27 layers + 2D RoPE}} [B, N, 1152] \xrightarrow{\text{pool}(k{=}3)} [B, N/9, 1152] \xrightarrow{\text{RMSNorm + linear}} [B, N/9, 5376]
$$

Where $N$ is the number of patches (variable per image) and $N/9$ is the number of soft tokens injected into the language model sequence.

---

**Next:** [`config_parameters.md`](./config_parameters.md)
