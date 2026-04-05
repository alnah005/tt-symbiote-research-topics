# Chapter 3: Vision Encoder

This chapter covers the complete vision pipeline in Gemma 4, from raw pixel patches through spatial pooling. The vision encoder is a bidirectional transformer that converts image patches into soft tokens for injection into the text decoder. All vision classes live in [`modular_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py) (inheritance/override form) and [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py) (fully resolved form). Refer to [Chapter 2](../ch2_configuration_hierarchy/index.md) for `Gemma4VisionConfig` parameter defaults.

## Module Tree

```
Gemma4VisionModel (Gemma4PreTrainedModel)
  |
  +-- patch_embedder: Gemma4VisionPatchEmbedder
  |     +-- input_proj: nn.Linear(3 * patch_size^2, hidden_size, bias=False)
  |     +-- position_embedding_table: Parameter [2, position_embedding_size, hidden_size]
  |
  +-- encoder: Gemma4VisionEncoder
  |     +-- rotary_emb: Gemma4VisionRotaryEmbedding
  |     +-- layers: ModuleList of Gemma4VisionEncoderLayer x num_hidden_layers
  |           +-- input_layernorm: Gemma4RMSNorm
  |           +-- self_attn: Gemma4VisionAttention
  |           |     +-- q_proj: Gemma4ClippableLinear(hidden_size, num_heads * head_dim)
  |           |     +-- k_proj: Gemma4ClippableLinear(hidden_size, num_kv_heads * head_dim)
  |           |     +-- v_proj: Gemma4ClippableLinear(hidden_size, num_kv_heads * head_dim)
  |           |     +-- o_proj: Gemma4ClippableLinear(num_heads * head_dim, hidden_size)
  |           |     +-- q_norm: Gemma4RMSNorm(head_dim, with_scale=True)
  |           |     +-- k_norm: Gemma4RMSNorm(head_dim, with_scale=True)
  |           |     +-- v_norm: Gemma4RMSNorm(head_dim, with_scale=False)
  |           +-- post_attention_layernorm: Gemma4RMSNorm
  |           +-- pre_feedforward_layernorm: Gemma4RMSNorm
  |           +-- mlp: Gemma4VisionMLP
  |           |     +-- gate_proj: Gemma4ClippableLinear(hidden_size, intermediate_size)
  |           |     +-- up_proj: Gemma4ClippableLinear(hidden_size, intermediate_size)
  |           |     +-- down_proj: Gemma4ClippableLinear(intermediate_size, hidden_size)
  |           +-- post_feedforward_layernorm: Gemma4RMSNorm
  |
  +-- pooler: Gemma4VisionPooler
  |
  +-- std_bias: Buffer [hidden_size]    (only if config.standardize)
  +-- std_scale: Buffer [hidden_size]   (only if config.standardize)
```

---

## 3.1 Gemma4ClippableLinear

`Gemma4ClippableLinear` wraps `nn.Linear(in_features, out_features, bias=False)` with optional input and output clamping for numerical stability.

```python
class Gemma4ClippableLinear(nn.Module):
    def __init__(self, config, in_features, out_features):
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))
```

When `Gemma4VisionConfig.use_clipped_linears` is `True`, the forward pass applies `torch.clamp` before and after the linear projection:

```python
def forward(self, hidden_states):
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
    hidden_states = self.linear(hidden_states)
    if self.use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
    return hidden_states
```

The four bound buffers (`input_min`, `input_max`, `output_min`, `output_max`) default to +/-infinity and are loaded from the checkpoint. Because they are registered buffers (not parameters), they are not trained but are serialized and restored with the model. This module is used in place of `nn.Linear` for all projections in both `Gemma4VisionAttention` and `Gemma4VisionMLP`.

---

## 3.2 Gemma4RMSNorm

The vision encoder uses `Gemma4RMSNorm` rather than standard LayerNorm. It has an important `with_scale` toggle:

```python
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
```

The normalization itself uses `torch.pow(mean_squared, -0.5)` instead of `torch.rsqrt()` to match JAX numerics:

```python
def _norm(self, hidden_states):
    mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
    return hidden_states * torch.pow(mean_squared, -0.5)
```

Computation is performed in float32 regardless of input dtype. When `with_scale=False`, no learnable weight is applied -- this variant is used exclusively for `v_norm` in `Gemma4VisionAttention`.

---

## 3.3 Gemma4VisionPatchEmbedder

This module converts flattened image patches into hidden-state vectors with 2D positional information.

### Inputs

| Argument | Shape | Description |
|---|---|---|
| `pixel_values` | `[B, num_patches, 3 * patch_size^2]` | Pre-flattened RGB patches |
| `pixel_position_ids` | `[B, num_patches, 2]` | (x, y) grid coordinates per patch; `-1` for padding |
| `padding_positions` | `[B, num_patches]` | Boolean mask, `True` where patch is padding |

### Architecture

There is no convolution. The projection is a plain linear layer:

```python
self.input_proj = nn.Linear(3 * self.patch_size**2, self.hidden_size, bias=False)
```

With `Gemma4VisionConfig.patch_size = 16`, the input dimension is `3 * 16^2 = 768` and the output is `Gemma4VisionConfig.hidden_size = 768`.

### Pixel Normalization

Before projection, pixel values are scaled from [0, 1] to [-1, 1]:

```python
pixel_values = 2 * (pixel_values - 0.5)
```

There is no per-channel ImageNet-style normalization.

### 2D Position Embedding

The position embedding table has shape `[2, position_embedding_size, hidden_size]` -- one row for x-positions, one for y-positions. With `Gemma4VisionConfig.position_embedding_size = 10240`, this supports up to 10240 distinct positions per axis.

Position encoding uses one-hot indexing rather than direct table lookup:

1. Clamp positions to `min=0` (maps padding positions from -1 to 0).
2. Create one-hot vectors: `F.one_hot(clamped_positions, num_classes=position_embedding_size)` producing `[B, num_patches, 2, position_embedding_size]`.
3. Permute to `[B, 2, num_patches, position_embedding_size]`.
4. Matrix multiply with the embedding table: `one_hot @ position_embedding_table` producing `[B, 2, num_patches, hidden_size]`.
5. Sum across the spatial dimension (dim=1) to combine x and y embeddings: `[B, num_patches, hidden_size]`.
6. Zero out embeddings at padding positions.

### Forward Output

```python
return self.input_proj(pixel_values) + position_embeddings
# Shape: [B, num_patches, hidden_size]
```

---

## 3.4 Gemma4VisionRotaryEmbedding

This class computes 2D factored Rotary Position Embeddings for spatial (x, y) patch positions. In the modular file it extends `LlamaRotaryEmbedding`; the modeling file contains a fully resolved version.

### Frequency Computation (compute_default_rope_parameters)

The key design choice: inverse frequencies are computed **independently for each spatial dimension** using `head_dim // 2` (called `spatial_dim`), rather than splitting a single global `inv_freq` vector in half.

```python
base = config.rope_parameters["rope_theta"]   # default: 100.0
dim = config.head_dim                          # default: 64
spatial_dim = dim // 2                         # 32

inv_freq = 1.0 / (base ** (arange(0, spatial_dim, 2) / spatial_dim))
# Shape: [spatial_dim // 2] = [16]
```

This means both x and y dimensions use the same set of 16 frequency values, each covering the full frequency range from 1.0 down to `1/100`.

### Forward Pass

The forward method takes `position_ids` of shape `[B, num_patches, 2]` and loops over the 2 spatial dimensions:

```python
for i in range(2):
    dim_position_ids = position_ids[:, :, i]          # [B, num_patches]
    freqs = inv_freq_expanded @ dim_position_ids_expanded  # [B, spatial_dim//2, num_patches]
    emb = cat((freqs, freqs), dim=-1)                 # [B, num_patches, spatial_dim]
    cos_i = emb.cos() * attention_scaling
    sin_i = emb.sin() * attention_scaling
    all_cos.append(cos_i)
    all_sin.append(sin_i)

cos = cat(all_cos, dim=-1)   # [B, num_patches, head_dim]  (32 + 32 = 64)
sin = cat(all_sin, dim=-1)   # [B, num_patches, head_dim]
```

Each spatial dimension produces cos/sin of length `spatial_dim = 32`. Concatenating across x and y gives cos/sin of length `head_dim = 64`.

---

## 3.5 apply_multidimensional_rope

This standalone function applies 2D factored RoPE to query or key tensors. It splits the input along the head dimension into `ndim` (=2) equal parts and applies standard rotary embedding to each part using that dimension's cos/sin slice.

```python
def apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):
    ndim = position_ids.shape[-1]               # 2
    num_input_channels = x.shape[-1]            # head_dim = 64
    num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))  # 2*(64//4) = 32

    split_sizes = [num_rotated_channels_per_dim] * ndim   # [32, 32]
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)

    y_parts = [apply_rotary_pos_emb(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim)
               for k in range(ndim)]
    return torch.cat(y_parts, dim=-1)
```

The standard `apply_rotary_pos_emb` uses the rotate-half formulation:

```python
def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)
```

Note: `unsqueeze_dim=2` is used in the vision path (tensors are `[B, seq, heads, head_dim]` before transpose), while the standard text path uses `unsqueeze_dim=1`.

---

## 3.6 Gemma4VisionAttention

This is the multi-head attention module for the vision encoder. In the modular file it extends `Gemma3Attention`, deleting several text-decoder-specific features.

### Key Differences from Gemma3Attention

| Feature | Gemma3Attention (text) | Gemma4VisionAttention |
|---|---|---|
| Linear projections | `nn.Linear` | `Gemma4ClippableLinear` |
| Value normalization | None | `v_norm = Gemma4RMSNorm(head_dim, with_scale=False)` |
| Attention scaling | `head_dim ** -0.5` | `1.0` (no scaling) |
| Causality | `is_causal = True` | `is_causal = False` (bidirectional) |
| Soft-capping | `attn_logit_softcapping` | Removed |
| Sliding window | `sliding_window` | Removed |
| RoPE type | 1D standard | 2D multidimensional |

### Forward Pass

```
Input: hidden_states [B, num_patches, hidden_size]

Q = q_proj(hidden_states).view([B, num_patches, num_heads, head_dim])
Q = q_norm(Q)
Q = apply_multidimensional_rope(Q, cos, sin, position_ids)
Q = Q.transpose(1, 2)  -->  [B, num_heads, num_patches, head_dim]

K = k_proj(hidden_states).view([B, num_patches, num_kv_heads, head_dim])
K = k_norm(K)
K = apply_multidimensional_rope(K, cos, sin, position_ids)
K = K.transpose(1, 2)  -->  [B, num_kv_heads, num_patches, head_dim]

V = v_proj(hidden_states).view([B, num_patches, num_kv_heads, head_dim])
V = v_norm(V)           <-- RMSNorm without scale
V = V.transpose(1, 2)  -->  [B, num_kv_heads, num_patches, head_dim]

attn_output = attention(Q, K, V, mask, scaling=1.0)
attn_output = attn_output.reshape([B, num_patches, num_heads * head_dim])
output = o_proj(attn_output)
```

The attention scaling of `1.0` is notable -- the Q/K norms (via `q_norm` and `k_norm`) are expected to regulate the logit magnitudes instead of the conventional `1/sqrt(head_dim)` factor. This is the QK-norm pattern from ViT research.

With `Gemma4VisionConfig` defaults: `num_attention_heads = 12`, `num_key_value_heads = 12` (no GQA in the vision encoder), `head_dim = 64`.

---

## 3.7 Gemma4VisionMLP

A gated MLP using `Gemma4ClippableLinear` for all projections, with `gelu_pytorch_tanh` activation:

```python
class Gemma4VisionMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = Gemma4ClippableLinear(config, hidden_size, intermediate_size)
        self.up_proj   = Gemma4ClippableLinear(config, hidden_size, intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, intermediate_size, hidden_size)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

With defaults: `hidden_size = 768`, `intermediate_size = 3072`. The expansion ratio is 4x. In the modular file this extends `Gemma3MLP`, replacing the `nn.Linear` projections with `Gemma4ClippableLinear`.

---

## 3.8 Gemma4VisionEncoderLayer

Each encoder layer follows a pre-norm + post-norm sandwich pattern with two residual connections. In the modular file this extends `Gemma3DecoderLayer`, but the forward pass removes all KV-cache logic.

### Forward Pass

```
Input: hidden_states [B, num_patches, hidden_size]

# --- Attention block ---
residual = hidden_states
hidden_states = input_layernorm(hidden_states)            # pre-attention norm
hidden_states = self_attn(hidden_states, ...)             # bidirectional attention
hidden_states = post_attention_layernorm(hidden_states)   # post-attention norm
hidden_states = residual + hidden_states                  # first residual

# --- MLP block ---
residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)  # pre-MLP norm
hidden_states = mlp(hidden_states)                        # gated MLP
hidden_states = post_feedforward_layernorm(hidden_states) # post-MLP norm
hidden_states = residual + hidden_states                  # second residual

Output: hidden_states [B, num_patches, hidden_size]
```

There are four `Gemma4RMSNorm` instances per layer: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, and `post_feedforward_layernorm`. This "sandwich norm" (pre-norm + post-norm around each sub-layer) differs from the standard transformer which uses only pre-norms.

No KV cache is used -- the vision encoder processes all patches in a single forward pass without autoregressive generation.

---

## 3.9 Gemma4VisionEncoder

The encoder holds the rotary embedding and layer stack, and creates the bidirectional attention mask.

```python
class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config):
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = ModuleList(
            [Gemma4VisionEncoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )
```

### Forward Pass

1. **Create bidirectional attention mask** via `create_bidirectional_mask()` from `transformers.masking_utils`. This converts the boolean padding mask (`True=valid`) into the format expected by the attention backend (e.g., additive float mask with `-inf` at padding positions for eager mode, or a boolean mask for SDPA).

2. **Compute position embeddings** once for all layers:
   ```python
   position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
   # Returns: (cos, sin), each [B, num_patches, head_dim]
   ```

3. **Run all layers** sequentially:
   ```python
   for layer in self.layers:
       hidden_states = layer(hidden_states, attention_mask, position_embeddings, position_ids)
   ```

4. **Return** `BaseModelOutputWithPast(last_hidden_state=hidden_states)`.

With `Gemma4VisionConfig.num_hidden_layers = 16`, the encoder has 16 transformer layers.

---

## 3.10 Gemma4VisionPooler

The pooler reduces the spatial resolution of the encoder output and scales the result.

### Initialization

```python
class Gemma4VisionPooler(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size ** 0.5  # sqrt(768) ~ 27.71
```

There are no learnable parameters.

### _avg_pool_by_positions

This method performs position-aware spatial pooling using one-hot kernel index grouping:

1. **Determine pooling kernel**: `k = int((input_seq_len // output_length) ** 0.5)`. With `Gemma4VisionConfig.pooling_kernel_size = 3`, a `3x3` grid of patches is pooled into one token, reducing the sequence length by 9x.

2. **Compute kernel indices**: Each patch is assigned to a pooling group based on its (x, y) position:
   ```python
   clamped_positions = pixel_position_ids.clamp(min=0)
   max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
   kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
   kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
   ```

3. **Build weight matrix**: `F.one_hot(kernel_idxs, output_length).float() / k_squared` creates a `[B, num_patches, output_length]` sparse weight matrix where each column averages the patches in one pooling group.

4. **Average pool**: `output = weights.transpose(1, 2) @ hidden_states.float()` produces `[B, output_length, hidden_size]`.

5. **Compute validity mask**: `mask = ~(weights == 0).all(dim=1)` yields `[B, output_length]` where `True` means the pooled position contains at least one real (non-padding) patch.

### Forward Pass

```python
def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
    # Zero out padding patches
    hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)

    # Pool if needed
    if hidden_states.shape[1] != output_length:
        hidden_states, padding_positions = self._avg_pool_by_positions(
            hidden_states, pixel_position_ids, output_length
        )

    # Scale by sqrt(hidden_size)
    hidden_states *= self.root_hidden_size

    return hidden_states, padding_positions
```

The scaling by `sqrt(hidden_size)` is applied after pooling, matching the convention used in PaLI-style vision-language models to balance the magnitude of vision soft tokens against text embeddings.

---

## 3.11 Gemma4VisionModel (Top-Level)

This is the public entry point for the vision encoder.

### Forward Pass

```python
def forward(self, pixel_values, pixel_position_ids, **kwargs):
    # 1. Compute output length after pooling
    pooling_kernel_size = self.config.pooling_kernel_size
    output_length = pixel_values.shape[-2] // (pooling_kernel_size ** 2)

    # 2. Identify padding patches
    padding_positions = (pixel_position_ids == -1).all(dim=-1)

    # 3. Patch embedding
    inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)

    # 4. Encoder (bidirectional transformer)
    output = self.encoder(
        inputs_embeds=inputs_embeds,
        attention_mask=~padding_positions,   # True = valid
        pixel_position_ids=pixel_position_ids,
    )

    # 5. Pooling
    hidden_states, pooler_mask = self.pooler(
        hidden_states=output.last_hidden_state,
        pixel_position_ids=pixel_position_ids,
        padding_positions=padding_positions,
        output_length=output_length,
    )

    # 6. Strip padding tokens
    hidden_states = hidden_states[pooler_mask]

    # 7. Optional standardization
    if self.config.standardize:
        hidden_states = (hidden_states - self.std_bias) * self.std_scale

    return BaseModelOutputWithPast(last_hidden_state=hidden_states)
```

Note: the output after step 6 is a **1D flattened** tensor of shape `[total_valid_tokens, hidden_size]` (batch dimension is collapsed). This is the format expected by `Gemma4MultimodalEmbedder`, which projects these soft tokens into the text decoder's hidden space.

---

## 3.12 End-to-End Data Flow

```
                         pixel_values                pixel_position_ids
                    [B, num_patches, 768]          [B, num_patches, 2]
                             |                            |
                             v                            v
               +-----------------------------+   +------------------+
               |  pixel normalization        |   |  clamp(-1 -> 0)  |
               |  2*(x - 0.5)               |   |  one_hot encode  |
               +-----------------------------+   +------------------+
                             |                            |
                             v                            v
                +------------------------+     +------------------------+
                | input_proj (Linear)    |     | position_embedding_table|
                | [B, P, 768] -> [B,P,768]|   | [2, 10240, 768]        |
                +------------------------+     | matmul + sum over dims |
                             |                 +------------------------+
                             |                            |
                             +----------- + -------------+
                                          |
                                          v
                               [B, num_patches, 768]
                                          |
                           +----------------------------------+
                           |      Gemma4VisionEncoder         |
                           |                                  |
                           |  rotary_emb(positions) -> cos,sin|
                           |  create_bidirectional_mask()     |
                           |                                  |
                           |  for each of 16 layers:          |
                           |    input_layernorm               |
                           |    VisionAttention (2D RoPE,     |
                           |      QK-norm, V-norm,            |
                           |      scaling=1.0, bidirectional) |
                           |    post_attention_layernorm      |
                           |    + residual                    |
                           |    pre_feedforward_layernorm     |
                           |    VisionMLP (gated, GeGLU-like) |
                           |    post_feedforward_layernorm    |
                           |    + residual                    |
                           +----------------------------------+
                                          |
                                          v
                               [B, num_patches, 768]
                                          |
                           +----------------------------------+
                           |      Gemma4VisionPooler          |
                           |                                  |
                           |  masked_fill(padding, 0)         |
                           |  _avg_pool_by_positions (3x3)    |
                           |    [B, P, 768] -> [B, P/9, 768]  |
                           |  *= sqrt(768) ~ 27.71            |
                           +----------------------------------+
                                          |
                                          v
                               [B, num_patches/9, 768]
                                          |
                           +----------------------------------+
                           |  Strip padding via pooler_mask   |
                           |  [total_valid_tokens, 768]       |
                           +----------------------------------+
                                          |
                           +----------------------------------+
                           | (optional) standardization       |
                           |  (x - std_bias) * std_scale      |
                           +----------------------------------+
                                          |
                                          v
                            [total_valid_tokens, 768]
                                    soft tokens
                                (to MultimodalEmbedder)
```

---

## TTNN Porting Considerations

### Gemma4ClippableLinear
The clamp-before/clamp-after pattern maps to `ttnn.clip` + `ttnn.linear`. The four bound buffers are scalars, so they can be stored as device constants. If `use_clipped_linears` is `False` (the default), these degenerate to plain `ttnn.linear` with no bias.

### 2D Position Embedding via One-Hot
The one-hot + matmul pattern (`F.one_hot` followed by `@`) is functionally equivalent to a gather/embedding lookup but structured as a dense matmul. On TT hardware, this could be implemented as either:
- A `ttnn.embedding` lookup (more efficient, avoids materializing a large one-hot matrix of shape `[B, num_patches, 10240]`).
- A direct `ttnn.matmul` if the one-hot sparsity cannot be exploited.

The sum across the x/y dimension is a simple `ttnn.sum(dim=1)`.

### Factored 2D RoPE
The RoPE computation involves a loop over 2 spatial dimensions with separate `inv_freq @ position_ids` matmuls per dimension. For TTNN:
- Pre-compute cos/sin tables on host and transfer to device, or compute `inv_freq @ positions` as two small matmuls on device.
- The `apply_multidimensional_rope` function splits the head dimension into 2 halves, applies standard rotary to each, and concatenates. This maps to `ttnn.split` + per-part `rotate_half` + `ttnn.concat`. The `rotate_half` itself is a split-negate-concat pattern.

### Sandwich Norm (4x RMSNorm per Layer)
Each encoder layer has four `Gemma4RMSNorm` instances. The `v_norm` uses `with_scale=False`, which means it has no learnable weight -- just normalization. All norms use `torch.pow(mean_squared, -0.5)` instead of `rsqrt`; for TTNN, `ttnn.rsqrt` should produce equivalent results with better performance.

### Attention with scaling=1.0
The QK-norm pattern (normalize Q and K, then use `scaling=1.0`) means the SDPA kernel should be called without the usual `1/sqrt(head_dim)` factor. Verify that the TTNN attention implementation supports a custom scaling factor or can be set to `1.0`.

### Vision Pooler
The `_avg_pool_by_positions` method constructs a sparse weight matrix via `F.one_hot` and performs pooling as a matrix multiply (`weights.T @ hidden_states`). For TTNN:
- The weight matrix is sparse (each row has exactly `k^2` non-zero entries all equal to `1/k^2`). A sparse matmul or a custom kernel could exploit this structure.
- Alternatively, if the spatial layout is regular (no padding in the middle), this reduces to a simple reshape + mean, which maps to `ttnn.reshape` + `ttnn.mean`.

### Standardization Buffers
The optional `std_bias` and `std_scale` buffers are loaded from the checkpoint. The operation `(x - bias) * scale` is elementwise and maps directly to `ttnn.sub` + `ttnn.mul`.

### Padding-Aware Operations
Several operations use `masked_fill` or `torch.where` with the padding mask. These map to `ttnn.where`. The final padding stripping (`hidden_states[pooler_mask]`) produces a ragged output -- for TTNN, this may need to be handled as a padded tensor with a separate length tensor, or the stripping can be deferred to the multimodal embedder.

---

**Next:** [Chapter 4 — Audio Encoder](../ch4_audio_encoder/index.md)
