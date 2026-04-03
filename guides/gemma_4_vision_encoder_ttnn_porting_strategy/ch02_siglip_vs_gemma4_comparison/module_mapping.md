# Module-by-Module Mapping: Gemma 3 TTNN to Gemma 4

This file maps every existing Gemma 3 TTNN vision encoder file (under `models/demos/multimodal/gemma3/tt/`) to its Gemma 4 equivalent, classifying each as direct reuse, modify, or new implementation. Use this mapping to plan sprint tasks and estimate effort.

## Module Mapping Table

All file paths are relative to `models/demos/multimodal/gemma3/tt/` for Gemma 3 and `models/demos/multimodal/gemma4/tt/` for the proposed Gemma 4 directory.

| Gemma 3 TTNN File | Gemma 4 Equivalent | Reuse Class | Effort |
|--------------------|-------------------|-------------|--------|
| `gemma_image_mlp.py` | `gemma4_vision_mlp.py` | **Direct reuse** | < 1 day |
| `gemma_vision_rmsnorm.py` | `gemma4_vision_rmsnorm.py` | **Direct reuse** | < 1 day |
| `gemma_image_block.py` / `gemma_vision_block.py` | `gemma4_vision_encoder_layer.py` | **Modify** | 1-2 days |
| `gemma_image_attention.py` | `gemma4_vision_attention.py` | **Modify** | 2-3 days |
| `gemma_conv2d_patch.py` / `siglip_vision_embedding.py` | `gemma4_vision_patch_embedder.py` | **Modify (major)** | 2-3 days |
| `multi_modal_projector.py` | `gemma4_multimodal_embedder.py` | **Modify** | 1-2 days |
| `model_config.py` | `gemma4_model_config.py` | **Modify** | 1 day |
| `load_checkpoints.py` | `gemma4_load_checkpoints.py` | **Modify** | 1-2 days |
| *(none)* | `gemma4_vision_rope.py` | **New** | 2-3 days |
| *(none)* | `gemma4_vision_position_embedding.py` | **New** | 1-2 days |
| *(none)* | `gemma4_vision_pooler.py` | **New** | 2-3 days |
| *(none)* | `gemma4_variable_resolution.py` | **New** | 1-2 days |

## Detailed Module Analysis

### Direct Reuse Modules (~40-50%)

These modules can be copied from Gemma 3 and used with no or minimal changes. The dominant compute shapes (`hidden_size=1152`, `intermediate_size=4304`, `num_attention_heads=16`, `head_dim=72`) are identical.

#### `gemma_image_mlp.py` -> `gemma4_vision_mlp.py`

**Reuse class:** Direct reuse

The MLP is architecturally identical between Gemma 3 and Gemma 4:

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Gate projection | `[1152, 4304]` | `[1152, 4304]` |
| Up projection | `[1152, 4304]` | `[1152, 4304]` |
| Down projection | `[4304, 1152]` | `[4304, 1152]` |
| Activation | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` |
| Bias | None | None |
| Structure | `down(gelu(gate(x)) * up(x))` | `down(gelu(gate(x)) * up(x))` |

**Changes needed:**
- Update weight key names for checkpoint loading (e.g., `vision_tower.encoder.layers.N.mlp.gate_proj` to `vision_tower.encoder.layers.N.mlp.gate_proj` — verify exact key mapping)
- Verify `Gemma4ClippableLinear` wrapper is transparent for `use_clipped_linears=False`

> **Tip:** Since the MLP accounts for ~70% of per-layer parameters and ~60% of per-layer FLOPs, confirming its direct reusability early de-risks a large fraction of the port.

#### `gemma_vision_rmsnorm.py` -> `gemma4_vision_rmsnorm.py`

**Reuse class:** Direct reuse

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Dimension | 1152 | 1152 |
| Epsilon | 1e-6 | 1e-6 |
| Learnable scale | Yes | Yes (for most norms; V-norm has no learnable scale) |

**Changes needed:**
- Add support for the "no learnable scale" variant used by V-norm and the multimodal embedder's pre-projection norm
- Gemma 3 uses LayerNorm in SigLIP, while Gemma 4 uses RMSNorm. If the existing `gemma_vision_rmsnorm.py` already implements RMSNorm for the language model side, it can be reused directly for Gemma 4 vision

### Modification Required Modules (~30%)

These modules share structural similarity with their Gemma 3 counterparts but require targeted changes.

#### `gemma_image_attention.py` -> `gemma4_vision_attention.py`

**Reuse class:** Modify (medium effort)

The attention module requires the most modifications of any reusable module. The core Q/K/V projection and output projection shapes are identical, but the normalization, positional encoding, and scaling are all different.

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Q projection | `[1152, 1152]` | `[1152, 1152]` |
| K projection | `[1152, 1152]` | `[1152, 1152]` |
| V projection | `[1152, 1152]` | `[1152, 1152]` |
| O projection | `[1152, 1152]` | `[1152, 1152]` |
| Attention type | MHA (16 heads) | MHA (16 heads, `num_key_value_heads=16`) |
| Attention bias | Yes | No |
| Scaling factor | $1/\sqrt{72} \approx 0.118$ | `1.0` |
| Q/K normalization | None | Per-head RMSNorm (learnable) |
| V normalization | None | Per-head RMSNorm (no learnable scale) |
| Positional encoding | None in attention (added at embedding stage) | 2D factored RoPE applied to Q and K |
| Mask type | Bidirectional (no causal mask) | Bidirectional (no causal mask) |

**Modifications needed:**

1. **Remove attention bias** from Q/K/V/O projections
2. **Add per-head Q/K RMSNorm** (dim=72, learnable scale) after projection, before RoPE
3. **Add per-head V RMSNorm** (dim=72, no learnable scale) after projection
4. **Integrate 2D RoPE** — call the new `gemma4_vision_rope.py` module to apply multidimensional rotation to Q and K
5. **Change scaling factor** from $1/\sqrt{72}$ to `1.0`

```python
# Pseudocode for modified attention forward pass
q = q_proj(hidden_states)                   # [batch, seq, 1152]
k = k_proj(hidden_states)                   # [batch, seq, 1152]
v = v_proj(hidden_states)                   # [batch, seq, 1152]

# Reshape to [batch, seq, 16, 72]
q = q.view(batch, seq, 16, 72)
k = k.view(batch, seq, 16, 72)
v = v.view(batch, seq, 16, 72)

# NEW: Per-head normalization
q = q_norm(q)                               # Per-head RMSNorm, learnable
k = k_norm(k)                               # Per-head RMSNorm, learnable
v = v_norm(v)                               # Per-head RMSNorm, no learnable scale

# NEW: Apply 2D RoPE to Q and K only
q = apply_multidimensional_rope(q, cos, sin)
k = apply_multidimensional_rope(k, cos, sin)

# Transpose to [batch, 16, seq, 72] and compute attention with scale=1.0
attn_output = scaled_dot_product_attention(q, k, v, mask, scale=1.0)
output = o_proj(attn_output.reshape(batch, seq, 1152))
```

> **Warning:** The combination of QK-norm + scale=1.0 is essential for numerical stability. If you accidentally leave the standard $1/\sqrt{d}$ scaling in place alongside QK-norm, attention scores will be too small and the model will produce degraded outputs.

#### `gemma_image_block.py` / `gemma_vision_block.py` -> `gemma4_vision_encoder_layer.py`

**Reuse class:** Modify (low-medium effort)

The encoder layer structure is similar but the normalization pattern changes.

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Norm type | LayerNorm | RMSNorm |
| Norm count per layer | 2 (pre-attention, pre-MLP) | 4 (pre/post-attention, pre/post-MLP) |
| Residual pattern | `x + attn(norm(x))` | `x + post_norm(attn(pre_norm(x)))` |
| Sub-layers | Attention + MLP | Attention + MLP |

**Modifications needed:**

1. Replace LayerNorm with RMSNorm
2. Add post-attention and post-MLP normalization layers
3. Update residual connection pattern to sandwich norm
4. Pass RoPE cos/sin tensors through to the attention sub-layer

#### `gemma_conv2d_patch.py` / `siglip_vision_embedding.py` -> `gemma4_vision_patch_embedder.py`

**Reuse class:** Modify (major rewrite)

This module changes substantially enough that it is closer to a rewrite than a modification, but it is listed here rather than under "new" because the conceptual role (convert pixels to hidden states + add positions) is the same.

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Patch extraction | `Conv2d(3, 1152, kernel_size=14, stride=14)` | Flatten 16x16x3 patch, then `nn.Linear(768, 1152)` |
| Position embedding | `nn.Embedding(4096, 1152)` — 1D learned | `nn.Parameter([2, 10240, 1152])` — 2D learned (x, y axes) |
| Position indexing | Sequential: 0, 1, 2, ..., 4095 | 2D grid coordinates: (x, y) per patch |
| Input assumption | Fixed 896x896 square | Variable aspect ratio, divisible by 48 |

**Modifications needed:**

1. Replace Conv2d patch extraction with reshape + linear
2. Replace 1D sequential position embedding with 2D (x, y) lookup from the `position_embedding_table`
3. Handle variable-length patch sequences with padding mask
4. Implement the `2 * (pixel_values - 0.5)` value scaling (replacing any ImageNet normalization)

#### `multi_modal_projector.py` -> `gemma4_multimodal_embedder.py`

**Reuse class:** Modify (medium effort)

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Pre-projection norm | LayerNorm or none | RMSNorm (no learnable scale) |
| Projection | Linear (1152 to LM dim) | Linear (1152 to 5376, no bias) |
| Pooling | Fixed average pooling (to 256 tokens) | Adaptive 2D pooling with `pooling_kernel_size=3` |
| Output standardization | None | Optional: `(x - bias) * scale` with learned params |
| Token budget | Fixed 256 | Configurable: 70, 140, 280, 560, 1120 |

**Modifications needed:**

1. Replace fixed pooling with adaptive 2D grid-based pooling (this is complex enough that it warrants its own file — `gemma4_vision_pooler.py`)
2. Replace LayerNorm with RMSNorm (no learnable scale)
3. Add optional standardization pass
4. The final linear projection is structurally identical but with updated dimensions

### New Implementation Modules (~20%)

These modules have no Gemma 3 equivalent and must be written from scratch.

#### `gemma4_vision_rope.py` (New)

**Purpose:** Compute and apply 2D factored Rotary Position Embeddings for the vision encoder.

**Key responsibilities:**
1. Compute inverse frequency tables with `rope_theta=100.0`
2. Accept 2D position IDs `[batch, num_patches, 2]`
3. Compute cos/sin embeddings independently for x and y axes
4. Apply multidimensional RoPE: split Q/K along head_dim into two halves (36 each), apply standard rotation to each half with the corresponding axis's cos/sin, concatenate results

**Implementation strategies** (ranked by effort, detailed in [Chapter 3](../ch03_2d_factored_rope/index.md)):
1. **Precompute on CPU, apply on device** — compute cos/sin tables on host, transfer to device, use element-wise multiply
2. **Compose from existing TTNN ops** — split, apply 1D RoPE to each half, concat
3. **Custom TTNN kernel** — fused 2D RoPE for maximum performance

> **Tip:** For initial bringup, strategy 1 (CPU precompute) is recommended. The cos/sin tables depend only on the image grid dimensions, which are known before the forward pass begins. This approach unblocks attention layer validation without waiting for a custom kernel.

#### `gemma4_vision_position_embedding.py` (New)

**Purpose:** Implement the 2D learned position embedding lookup.

**Key responsibilities:**
1. Store the embedding table `[2, 10240, 1152]`
2. Accept 2D position IDs `[batch, num_patches, 2]`
3. For each patch, look up the x-position embedding and y-position embedding from their respective rows of the table
4. Sum the x and y embeddings to produce a single positional vector per patch
5. Zero out embeddings for padding positions (where position IDs are -1)

**TTNN implementation:**
- The one-hot based lookup in the HuggingFace reference (`F.one_hot` + batch matmul) can likely be replaced with `ttnn.embedding` for each axis, followed by `ttnn.add`
- Alternatively, gather the embedding vectors using index-based access and sum

#### `gemma4_vision_pooler.py` (New)

**Purpose:** Implement adaptive 2D average pooling with configurable token budgets and optional standardization.

**Key responsibilities:**
1. Compute grid cell assignments from (x, y) position IDs and pooling kernel size
2. Average hidden states within each grid cell
3. Scale output by $\sqrt{1152}$
4. Optionally apply standardization: `(x - bias) * scale` with learned parameters
5. Produce a validity mask for downstream use

**TTNN implementation challenges:**
- The pooling operates on a 1D sequence with explicit 2D coordinates, not a standard 2D spatial tensor
- A custom scatter-reduce or one-hot based weighted sum may be needed
- See [Chapter 4](../ch04_patch_embedding_and_pooling/index.md) for detailed implementation options

#### `gemma4_variable_resolution.py` (New)

**Purpose:** Host-side preprocessing logic for variable-resolution images.

**Key responsibilities:**
1. Compute target image dimensions given a token budget and aspect ratio
2. Enforce divisibility by 48 (patch_size * pooling_kernel_size)
3. Compute 2D position IDs (x, y grid coordinates) for each patch
4. Handle padding for batched images with different resolutions

This module runs entirely on the host CPU and is not a TTNN kernel, but it is required to produce the inputs that the TTNN vision encoder expects.

## Reuse Summary

| Category | Files | Estimated Effort | % of Total |
|----------|-------|-----------------|------------|
| **Direct reuse** | MLP, RMSNorm | 1-2 days | ~40-50% |
| **Modify** | Attention, encoder layer, patch embedder, projector, config, checkpoints | 7-12 days | ~30% |
| **New** | 2D RoPE, 2D position embedding, pooler, variable-resolution preprocessor | 6-10 days | ~20% |
| **Total** | 12 modules | **~14-24 days** (2-4 engineer-weeks) | 100% |

> **Warning:** The effort estimates above cover initial implementation and basic correctness validation (PCC against CPU reference). They do not include performance optimization, which may add 1-2 additional weeks. See [Chapter 7](../ch07_implementation_roadmap/index.md) for the complete phased timeline.

## Dependency Order for Implementation

The modules should be ported in dependency order to enable incremental validation:

```
1. gemma4_vision_rmsnorm.py          (no dependencies; validate against torch RMSNorm)
2. gemma4_vision_mlp.py              (depends on RMSNorm for weight loading verification)
3. gemma4_vision_rope.py             (no TTNN dependencies; validate against HF reference)
4. gemma4_vision_position_embedding.py (no TTNN dependencies; validate against HF reference)
5. gemma4_vision_attention.py        (depends on RMSNorm + RoPE)
6. gemma4_vision_encoder_layer.py    (depends on attention + MLP + RMSNorm)
7. gemma4_vision_patch_embedder.py   (depends on position embedding)
8. gemma4_vision_pooler.py           (standalone; can be validated independently)
9. gemma4_multimodal_embedder.py     (depends on pooler + RMSNorm + linear)
```

---

**Next:** [`positional_encoding_shift.md`](./positional_encoding_shift.md) — Deep dive into the positional encoding paradigm shift from absolute to 2D RoPE.
