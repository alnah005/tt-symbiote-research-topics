# Modification Required Modules

This file covers the six Gemma 3 TTNN modules that need targeted modifications to support the Gemma 4 vision encoder. These modules share structural similarity with their Gemma 3 counterparts but require changes to accommodate new features: sandwich normalization in the encoder block, 2D RoPE in attention, adaptive pooling in the projector, a different patch embedding strategy, new config parameters, and a rewritten checkpoint key mapping. Together they account for approximately 50% of the codebase and require an estimated 8-13 days of engineering effort.

All file paths are relative to `models/demos/multimodal/gemma3/tt/` for Gemma 3 and `models/demos/multimodal/gemma4/tt/` for the proposed Gemma 4 directory.

## `gemma_image_attention.py`

**Gemma 4 target:** `gemma4_vision_attention.py`
**Reuse class:** Modify (medium-high effort)
**Effort:** 2-3 days

The attention module requires the most modifications of any reusable module. The core Q/K/V and output projections have identical shapes, but normalization, positional encoding, and scaling all change.

### Side-by-Side Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Q projection | `[1152, 1152]` | `[1152, 1152]` |
| K projection | `[1152, 1152]` | `[1152, 1152]` |
| V projection | `[1152, 1152]` | `[1152, 1152]` |
| O projection | `[1152, 1152]` | `[1152, 1152]` |
| Attention type | MHA (16 heads) | MHA (16 heads, `num_key_value_heads=16`) |
| Attention bias | Yes | No |
| Scaling factor | $1/\sqrt{72} \approx 0.118$ | `1.0` |
| Q/K normalization | None | Per-head RMSNorm (learnable scale, dim=72) |
| V normalization | None | Per-head RMSNorm (no learnable scale, dim=72) |
| Positional encoding | None in attention (added at embedding) | 2D factored RoPE on Q and K |
| Mask type | Bidirectional (no causal mask) | Bidirectional (no causal mask) |

### Required Modifications

**Modification 1: Remove attention bias**

The Q/K/V/O projections in Gemma 4 have `bias=False`. If the existing Gemma 3 attention module uses `ttnn.linear` with bias, remove the bias parameter. This is a one-line change per projection.

**Modification 2: Add per-head Q/K/V normalization**

Gemma 4 applies per-head RMSNorm to Q, K, and V after projection but before any positional encoding:

- Q and K norms have learnable scale parameters (dim=72 per head, 16 heads).
- V norm has no learnable scale.

The normalization operates on the last dimension (head_dim=72) independently for each head. In TTNN, this maps to `ttnn.rms_norm` applied after reshaping to `[batch, seq, 16, 72]` or equivalently `[batch * seq * 16, 72]`.

```python
# After projection, reshape to expose heads
q = ttnn.reshape(q, [batch, seq, 16, 72])
k = ttnn.reshape(k, [batch, seq, 16, 72])
v = ttnn.reshape(v, [batch, seq, 16, 72])

# Per-head RMSNorm
q = self.q_norm(q)  # learnable scale
k = self.k_norm(k)  # learnable scale
v = self.v_norm(v)  # no learnable scale
```

> **Warning:** The per-head norms have weight shape `[16, 72]`, not `[1152]`. Each head has its own independent scale vector. Ensure the RMSNorm implementation broadcasts correctly over the batch and sequence dimensions while applying head-specific scales.

**Modification 3: Integrate 2D factored RoPE**

After QK-norm, apply 2D RoPE to Q and K (but not V). The cos/sin tables are precomputed based on the image's patch grid dimensions and passed into the attention module.

```python
# Apply 2D factored RoPE to Q and K only
q = apply_2d_rope(q, cos, sin)  # cos, sin: [batch, 1, num_patches, 72]
k = apply_2d_rope(k, cos, sin)
```

The `apply_2d_rope` function is implemented in the new `gemma4_vision_rope.py` module. For initial bringup, use the CPU-precomputed cos/sin tables (Strategy 1 from [Chapter 3](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md)).

> **Warning:** The order of operations is critical: project, reshape, normalize, then apply RoPE. If RoPE is applied before QK-norm, the rotation angles interact with the normalization scaling and produce incorrect results.

**Modification 4: Change scaling factor to 1.0**

Replace the standard $1/\sqrt{d}$ scaling with `scale=1.0` in the attention score computation. The QK-norm already normalizes the query and key magnitudes, making explicit scaling unnecessary and harmful.

```python
# Gemma 3 (WRONG for Gemma 4):
# attn_weights = ttnn.matmul(q, k_t) * (1.0 / math.sqrt(72))

# Gemma 4 (correct):
attn_weights = ttnn.matmul(q, k_t)  # scale=1.0, no division
```

> **Warning:** The combination of QK-norm + scale=1.0 is essential for numerical stability. If you accidentally leave the $1/\sqrt{d}$ scaling alongside QK-norm, attention scores will be too small and the model will produce degraded outputs. This is a common porting mistake.

**Modification 5: Remove attention mask bias (if present)**

Gemma 3 SigLIP may add a learned or fixed attention bias. Gemma 4 uses no attention bias — the bidirectional attention is unmasked. Remove any bias addition in the attention score computation.

### Modified Forward Pass (Pseudocode)

```python
def __call__(self, hidden_states, cos, sin):
    batch, seq, _ = hidden_states.shape

    # Q/K/V projections (no bias)
    q = self.q_proj(hidden_states)          # [batch, seq, 1152]
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # Reshape to [batch, seq, 16, 72]
    q = ttnn.reshape(q, [batch, seq, 16, 72])
    k = ttnn.reshape(k, [batch, seq, 16, 72])
    v = ttnn.reshape(v, [batch, seq, 16, 72])

    # Per-head RMSNorm
    q = self.q_norm(q)
    k = self.k_norm(k)
    v = self.v_norm(v)

    # 2D factored RoPE on Q and K only
    q = ttnn.transpose(q, 1, 2)            # [batch, 16, seq, 72]
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.transpose(v, 1, 2)
    q = apply_2d_rope(q, cos, sin)
    k = apply_2d_rope(k, cos, sin)

    # Attention with scale=1.0
    attn_weights = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
    attn_weights = ttnn.softmax(attn_weights, dim=-1)
    attn_output = ttnn.matmul(attn_weights, v)      # [batch, 16, seq, 72]

    # Reshape and output projection
    attn_output = ttnn.transpose(attn_output, 1, 2)  # [batch, seq, 16, 72]
    attn_output = ttnn.reshape(attn_output, [batch, seq, 1152])
    output = self.o_proj(attn_output)

    return output
```

### Validation Strategy

1. **Unit test the QK-norm** independently: feed random Q/K/V through the norm and verify PCC against PyTorch `Gemma4VisionRMSNormPerHead`.
2. **Unit test RoPE application**: verify rotated Q/K against HuggingFace `apply_multidimensional_rope`. PCC > 0.999.
3. **Full attention module test**: run a single layer's attention with real weights and compare output to HuggingFace. PCC > 0.999.
4. **Scaling sanity check**: verify that attention weights after softmax are not degenerate (not all near 0 or 1) — this catches the wrong-scale bug.

## `multi_modal_projector.py`

**Gemma 4 target:** `gemma4_multimodal_embedder.py`
**Reuse class:** Modify (medium effort)
**Effort:** 1-2 days

The multimodal projector maps vision encoder outputs to the language model's hidden dimension. Gemma 4 restructures this module with adaptive pooling and optional standardization.

### Side-by-Side Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Pre-projection norm | LayerNorm or none | RMSNorm (no learnable scale) |
| Projection | Linear (1152 to LM dim) | Linear (1152 to 5376, no bias) |
| Pooling | Fixed average pooling (to 256 tokens) | Adaptive 2D pooling with `pooling_kernel_size=3` |
| Output standardization | None | Optional: `(x - bias) * scale` with learned params |
| Token budget | Fixed 256 | Configurable: 70, 140, 280, 560, 1120 |

### Required Modifications

**Modification 1: Replace fixed pooling with adaptive 2D pooling**

This is the most significant change. Gemma 3 uses a simple average pooling that reduces a fixed 256-token sequence. Gemma 4's pooling operates on a 2D patch grid with configurable kernel size.

The adaptive pooling logic is complex enough to warrant its own module (`gemma4_vision_pooler.py`, covered in [new_implementation_modules.md](./new_implementation_modules.md)). The multimodal embedder calls the pooler rather than implementing pooling inline.

See [Chapter 4 — Adaptive Pooling Port](../ch04_patch_embedding_and_pooling/adaptive_pooling_port.md) for the detailed implementation strategy.

**Modification 2: Replace LayerNorm with RMSNorm (no learnable scale)**

The pre-projection normalization changes from LayerNorm to RMSNorm without a learnable scale parameter. This is a simple swap using the `has_weight=False` variant of the RMSNorm module from [direct_reuse_modules.md](./direct_reuse_modules.md#gemma_vision_rmsnormpy).

**Modification 3: Add output standardization**

After pooling and scaling by $\sqrt{1152}$, Gemma 4 optionally applies a learned standardization:

```python
# Optional standardization with learned parameters
if self.pool_bias is not None and self.pool_scale is not None:
    hidden_states = (hidden_states - self.pool_bias) * self.pool_scale
```

The `pool_bias` and `pool_scale` parameters have shape `[1152]` and are loaded from the checkpoint. In TTNN, this maps to element-wise subtract and multiply operations.

**Modification 4: Update projection dimensions**

The final linear projection maps from `hidden_size=1152` to the language model's `hidden_size=5376` (for the 31B model). Verify the output dimension matches the target language model configuration.

### Modified Forward Pass (Pseudocode)

```python
def __call__(self, hidden_states, position_ids):
    # Adaptive 2D pooling (replaces fixed pooling)
    hidden_states = self.pooler(hidden_states, position_ids)
    # hidden_states: [batch, output_tokens, 1152]

    # Scale by sqrt(hidden_size)
    hidden_states = hidden_states * math.sqrt(1152)

    # Optional standardization
    if self.pool_bias is not None:
        hidden_states = (hidden_states - self.pool_bias) * self.pool_scale

    # RMSNorm (no learnable scale) + linear projection
    hidden_states = self.norm(hidden_states)
    hidden_states = self.projection(hidden_states)  # [batch, output_tokens, 5376]

    return hidden_states
```

### Validation Strategy

1. **Validate pooling separately** using the standalone `gemma4_vision_pooler.py` module. This is the highest-risk component.
2. **Validate the projection path** (norm + linear) independently — this is straightforward and should pass PCC > 0.999 immediately.
3. **End-to-end validation**: feed real encoder outputs through the full embedder and compare against HuggingFace. PCC > 0.999.

## `gemma_conv2d_patch.py`

**Gemma 4 target:** `gemma4_vision_patch_embedder.py`
**Reuse class:** Modify (major rewrite)
**Effort:** 2-3 days

The patch embedding module changes substantially — from a Conv2d-based approach to a flatten-and-project approach with 2D learned position embeddings. This is listed under "modification" rather than "new" because the conceptual role (convert raw pixels to hidden states with positional information) is the same, and some infrastructure (weight loading, tensor layout) can be carried forward.

### Side-by-Side Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Patch extraction | `Conv2d(3, 1152, kernel_size=14, stride=14)` | Flatten 16x16x3 patch, then `nn.Linear(768, 1152)` |
| Position embedding | `nn.Embedding(4096, 1152)` — 1D learned | `nn.Parameter([2, 10240, 1152])` — 2D learned (x, y) |
| Position indexing | Sequential: 0, 1, ..., 4095 | 2D grid coordinates: (x, y) per patch |
| Input assumption | Fixed 896x896 square | Variable aspect ratio, divisible by 48 |
| Value scaling | ImageNet mean/std normalization | `2 * (pixel_values - 0.5)` internal scaling |

### Required Modifications

**Modification 1: Replace Conv2d with flatten + linear**

Gemma 4 does not use a convolution for patch extraction. Instead, it:

1. Reshapes the image into non-overlapping 16x16 patches.
2. Flattens each patch from `[16, 16, 3]` to `[768]`.
3. Projects through a linear layer: `nn.Linear(768, 1152)`.

In TTNN:

```python
# Input: pixel_values [batch, channels=3, height, width]
# Reshape to patches: [batch, num_patches, 16*16*3]
patches = extract_patches(pixel_values, patch_size=16)  # host-side reshape
patches = ttnn.reshape(patches, [batch, num_patches, 768])

# Linear projection to hidden_size
hidden_states = self.patch_projection(patches)  # [batch, num_patches, 1152]
```

The `extract_patches` operation is a reshape that can be done on the host before transfer, or on device using `ttnn.reshape` if the tensor is already in the right memory layout.

> **Tip:** The patch projection weight has shape `[1152, 768]`. Unlike the Conv2d weight `[1152, 3, 14, 14]` in Gemma 3, this is a standard 2D matrix that maps directly to `ttnn.linear` with no special handling for spatial dimensions.

**Modification 2: Replace 1D position embedding with 2D lookup**

The position embedding system changes entirely. Gemma 4 stores a table of shape `[2, 10240, 1152]` where:

- Index 0 contains x-axis position embeddings.
- Index 1 contains y-axis position embeddings.

For each patch at grid position `(x, y)`:

```python
pos_embed = position_embedding_table[0, x, :] + position_embedding_table[1, y, :]
```

This 2D lookup is complex enough to warrant its own module (`gemma4_vision_position_embedding.py`, covered in [new_implementation_modules.md](./new_implementation_modules.md)).

**Modification 3: Handle variable input shapes**

Gemma 3 assumes a fixed 896x896 input, producing a fixed sequence of 4096 patches (64x64). Gemma 4 accepts variable aspect ratios, producing a variable number of patches.

Implications for the patch embedder:

- The `num_patches` dimension varies across images.
- For batched inference, images must be padded to the maximum sequence length in the batch.
- A padding mask must be generated to indicate which patches are real vs. padding.

**Modification 4: Update value scaling**

Replace any ImageNet-style normalization with the Gemma 4 internal scaling:

```python
# Gemma 3: ImageNet normalization (done in preprocessing)
# pixel_values = (pixel_values - mean) / std

# Gemma 4: Simple rescaling (done inside the embedder)
pixel_values = 2.0 * (pixel_values - 0.5)
```

### What Can Be Reused from Gemma 3

Despite the major changes, several aspects of the existing module infrastructure are still useful:

- **Weight loading patterns**: the function signatures for loading and converting weights to TTNN tensors.
- **Tensor layout utilities**: converting between row-major and tile layout, choosing memory configs.
- **Module structure**: the `__init__` / `__call__` pattern, device management, and state dict handling.

### Validation Strategy

1. **Validate patch extraction**: compare the flattened patch tensor against HuggingFace's `Gemma4VisionPatchEmbedder` intermediate output. Exact match expected (this is just a reshape).
2. **Validate linear projection**: feed flattened patches through the TTNN linear layer and compare against PyTorch. PCC > 0.999.
3. **Validate position embedding addition**: covered by the `gemma4_vision_position_embedding.py` validation (see [new_implementation_modules.md](./new_implementation_modules.md)).
4. **End-to-end patch embedder**: input pixels, output embedded patches with positions. PCC > 0.999 against HuggingFace.

> **Risk (Medium):** Variable input shapes may prevent TTNN program caching. If the patch count changes between images, every new shape triggers recompilation. Mitigation: pad all images to the nearest supported token budget's patch count. The five standard budgets (70, 140, 280, 560, 1120) correspond to five fixed patch counts, so at most five compiled programs are needed.

## `gemma_image_block.py`

**Gemma 4 target:** `gemma4_vision_encoder_layer.py`
**Reuse class:** Modify
**Effort:** 1-2 days

The encoder block structure is the same — attention followed by MLP with residual connections — but four changes are needed to match the Gemma 4 architecture.

### Architecture Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Sub-layers | Attention + MLP | Attention + MLP |
| Norm type | LayerNorm | RMSNorm |
| Norms per layer | 2 (pre-attention, pre-MLP) | 4 (pre/post-attention, pre/post-MLP) |
| Residual pattern | `x + attn(norm(x))` | `x + post_norm(attn(pre_norm(x)))` |
| Layer count | 27 | 27 |

### Required Modifications

1. **Replace LayerNorm with RMSNorm.** Swap `ttnn.layer_norm` calls for `ttnn.rms_norm`.
2. **Add post-attention and post-MLP norms.** Each sub-layer now has two norms instead of one (sandwich norm pattern).
3. **Update residual connection pattern.** The forward pass becomes:

```python
def __call__(self, hidden_states, cos, sin):
    # Attention sub-layer with sandwich norm
    residual = hidden_states
    hidden_states = self.pre_attention_norm(hidden_states)
    hidden_states = self.attention(hidden_states, cos, sin)
    hidden_states = self.post_attention_norm(hidden_states)
    hidden_states = residual + hidden_states

    # MLP sub-layer with sandwich norm
    residual = hidden_states
    hidden_states = self.pre_mlp_norm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_mlp_norm(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states
```

4. **Pass RoPE cos/sin tensors** through to the attention sub-layer. Gemma 3's attention does not use RoPE, so the existing block does not forward these tensors. Add `cos` and `sin` as arguments.

### Validation Steps

1. Copy `gemma_image_block.py` and apply the four changes above.
2. Instantiate with random weights and validate the forward pass shape: input `[1, 840, 1152]` produces output `[1, 840, 1152]`.
3. Load real weights and validate PCC against the HuggingFace `Gemma4VisionEncoderLayer` for a single layer. PCC > 0.999.

## `model_config.py`

**Gemma 4 target:** `gemma4_model_config.py`
**Reuse class:** Modify
**Effort:** 1 day

### What Transfers

The Gemma 3 `model_config.py` typically contains:

- Memory configuration presets (DRAM interleaved, L1 interleaved, L1 sharded)
- Sharding strategies for matmul operations at specific shapes
- Data type configurations (weights in BF16 or BFP8, activations in BF16)
- Device grid mapping for the Wormhole 8x8 core grid

Because the weight matrix shapes are identical (`[1152, 1152]` for attention, `[1152, 4304]` and `[4304, 1152]` for MLP), the memory configs and sharding strategies carry over.

### What Must Be Added

| New Parameter | Value | Purpose |
|---------------|-------|---------|
| `patch_size` | 16 | Patch embedding dimension calculation |
| `pooling_kernel_size` | 3 | Adaptive pooling grid cell size |
| `position_embedding_size` | 10240 | 2D position embedding table size |
| `rope_theta` | 100.0 | Vision RoPE frequency base |
| `default_output_length` | 280 | Default token budget |
| `supported_token_budgets` | `[70, 140, 280, 560, 1120]` | All valid token budgets |
| `num_key_value_heads` | 16 | Explicit KV head count (MHA) |

Adding seven new parameters and potentially new memory configs for the 2D position embedding lookup and adaptive pooling operations makes this a substantive modification, not a simple copy.

### Validation Steps

1. Copy `model_config.py` and add the Gemma 4 parameters.
2. Verify that all existing sharding configs produce valid shard specs for the unchanged matrix shapes.
3. Add new memory configs for any new operations (2D position embedding lookup, adaptive pooling) if needed.

## `load_checkpoints.py`

**Gemma 4 target:** `gemma4_load_checkpoints.py`
**Reuse class:** Modify
**Effort:** 1-2 days

### What Transfers

The checkpoint loading infrastructure — downloading weights, converting formats, managing cache directories, splitting weights across devices for multi-chip configurations — is model-agnostic and transfers directly.

### What Must Be Updated

The weight key mapping between HuggingFace checkpoint keys and TTNN module parameter names needs a complete rewrite. At least ten key patterns change:

| Change | Gemma 3 Key Pattern | Gemma 4 Key Pattern |
|--------|---------------------|---------------------|
| Model prefix | `vision_tower.vision_model.*` | `vision_tower.*` |
| Encoder layers | `encoder.layers.{N}.*` | `encoder.layers.{N}.*` |
| Attention norms | *(none)* | `self_attn.q_norm.weight`, `self_attn.k_norm.weight` |
| V-norm | *(none)* | `self_attn.v_norm_weight` (no bias) |
| Post-attention norm | *(none)* | `post_attention_layernorm.weight` |
| Post-MLP norm | *(none)* | `post_mlp_layernorm.weight` |
| Patch embedding | `embeddings.patch_embedding.weight` | `patch_embedding.weight` (linear, not Conv2d) |
| Position embedding | `embeddings.position_embedding.weight` | `position_embedding_table` (shape `[2, 10240, 1152]`) |
| Multimodal embedder | `multi_modal_projector.*` | `multimodal_embedder.*` |
| Pooler weights | *(none)* | `multimodal_embedder.pool_bias`, `multimodal_embedder.pool_scale` |

> **Warning:** The Gemma 4 position embedding table has shape `[2, 10240, 1152]` — two axes, each with up to 10240 position entries. This is significantly larger than Gemma 3's `[4096, 1152]` embedding. Verify that the weight loading code handles 3D embedding tables correctly and does not assume a 2D shape.

### Validation Steps

1. Copy `load_checkpoints.py` and rewrite the key mapping dictionary.
2. Load a Gemma 4 checkpoint and verify that every expected key is found and mapped.
3. Print any unmapped keys — these indicate either new parameters that need handling or renamed parameters that the mapping missed.
4. Verify weight shapes match expectations (e.g., `patch_embedding.weight` should be `[1152, 768]` for Gemma 4, not `[1152, 3, 14, 14]` as in Gemma 3).

## Effort Summary

| Module | Key Changes | Effort | Risk Level |
|--------|-------------|--------|------------|
| `gemma_image_block.py` | Norm type swap, 2 added norms, changed residual, RoPE args | 1-2 days | Low-Medium (mechanical changes) |
| `model_config.py` | 7 new parameters, potential new memory configs | 1 day | Low (additive changes) |
| `load_checkpoints.py` | Key mapping rewrite (10+ changed patterns), 3D embedding table | 1-2 days | Medium (must map every key correctly) |
| `gemma_image_attention.py` | QK-norm, V-norm, 2D RoPE, scale=1.0, remove bias | 2-3 days | Medium-High (RoPE integration) |
| `multi_modal_projector.py` | Adaptive pooling, RMSNorm, standardization | 1-2 days | Medium (pooling is a new module) |
| `gemma_conv2d_patch.py` | Flatten+linear, 2D position embedding, variable shapes | 2-3 days | Medium (variable shapes) |
| **Total** | | **8-13 days** | |

The attention module is the highest-risk item because it integrates the most new components (QK-norm, V-norm, and 2D RoPE). Prioritize getting a single attention layer validated early — this unblocks the full encoder layer and reveals any issues with the RoPE implementation.

---

**Next:** [`new_implementation_modules.md`](./new_implementation_modules.md) — Modules that must be written from scratch for the Gemma 4 vision encoder.
