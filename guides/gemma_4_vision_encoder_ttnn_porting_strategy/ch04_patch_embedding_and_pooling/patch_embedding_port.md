# Patch Embedding Port

This file covers the TTNN porting strategy for `Gemma4VisionPatchEmbedder`, the module that converts raw image patches into hidden-state vectors with 2D positional information. It is the entry point of the vision encoder and one of the two modules (along with the pooler) that differ most from Gemma 3.

## Gemma 4 Patch Embedding: Flatten + Linear

Unlike Gemma 3's SigLIP encoder, which uses a strided 2D convolution to simultaneously extract and project patches, Gemma 4 separates these into two stages:

1. **Preprocessing (host-side):** The image processor extracts non-overlapping 16x16 pixel patches and flattens each to a vector of length $3 \times 16 \times 16 = 768$. The resulting tensor has shape `[batch, num_patches, 768]`.

2. **Linear projection (model):** A bias-free `nn.Linear(768, 1152)` maps each flattened patch to the encoder's hidden dimension.

The value scaling `2 * (pixel_values - 0.5)` is applied before the linear projection, rescaling pixel values from `[0, 1]` to `[-1, 1]`. No ImageNet mean/std normalization is used.

### Forward Pass Summary

```python
# Inputs
#   pixel_values:       [batch, num_patches, 768]
#   pixel_position_ids: [batch, num_patches, 2]     (x, y grid coordinates)
#   padding_positions:  [batch, num_patches]         (True for padding patches)

# Step 1: Value scaling
pixel_values = 2 * (pixel_values - 0.5)

# Step 2: Linear projection
hidden_states = input_proj(pixel_values)  # [batch, num_patches, 1152]

# Step 3: 2D position embeddings (see next section)
position_embeddings = compute_2d_position_embeddings(pixel_position_ids, padding_positions)

# Step 4: Addition
output = hidden_states + position_embeddings  # [batch, num_patches, 1152]
```

## Comparison with Gemma 3 Conv2d Patch Embedding

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Patch extraction method | `Conv2d(3, 1152, kernel_size=14, stride=14)` | Host-side flatten to `[768]` per patch |
| Projection | Implicit in Conv2d weights | `nn.Linear(768, 1152, bias=False)` |
| Patch size | 14x14 pixels | 16x16 pixels |
| Input shape | Fixed `[batch, 3, 896, 896]` | Variable `[batch, num_patches, 768]` |
| Output shape | Fixed `[batch, 4096, 1152]` (64x64 grid) | Variable `[batch, num_patches, 1152]` |
| Position embedding | 1D learned `nn.Embedding(4096, 1152)` | 2D learned `nn.Parameter([2, 10240, 1152])` |
| Position indexing | Sequential: 0 to 4095 | 2D grid coordinates: (x, y) per patch |
| TTNN module | `gemma_conv2d_patch.py` | New: `gemma4_vision_patch_embedder.py` |

### Key Simplification for TTNN

The shift from Conv2d to flatten+linear is a net simplification for the TTNN port:

- **Conv2d with non-tile-aligned kernel is gone.** Gemma 3's `Conv2d(kernel_size=14, stride=14)` maps awkwardly to TTNN's 32x32 tile compute model. The kernel size 14 is not a multiple of 32, requiring padding or special handling. Gemma 4 eliminates this entirely.

- **Standard linear replaces specialized convolution.** The `nn.Linear(768, 1152)` maps directly to `ttnn.linear` with weight shape `[768, 1152]`. This is a well-optimized path in TTNN.

- **Patch extraction moves to the host.** The flattening of 16x16x3 patches happens during image preprocessing on the CPU, before the tensor is transferred to the device. The TTNN graph receives a pre-flattened `[batch, num_patches, 768]` tensor.

> **Tip:** The patch extraction being host-side means the TTNN graph starts with a standard 2D matmul-shaped input. This eliminates the need for `gemma_conv2d_patch.py` entirely and removes one of the most hardware-unfriendly operations from the Gemma 3 port.

## 2D Learned Position Embeddings

The most architecturally novel aspect of the patch embedder is its 2D position embedding scheme. Instead of a single learned embedding table indexed by a sequential position (as in Gemma 3 SigLIP), Gemma 4 uses a factored 2D embedding with independent tables for the x and y axes.

### Embedding Table Structure

The position embedding table has shape `[2, 10240, 1152]`:

- Axis 0 has two entries: index 0 for x-positions, index 1 for y-positions.
- Axis 1 spans 10240 possible positions per axis (the `position_embedding_size` config parameter).
- Axis 2 is the hidden dimension (1152).

### Reference Implementation

The HuggingFace reference computes position embeddings using a one-hot matrix multiplication approach:

```python
# position_ids shape: [batch, num_patches, 2]
# position_embedding_table shape: [2, 10240, 1152]
# padding_positions shape: [batch, num_patches]

position_embeddings = torch.zeros(batch, num_patches, 1152)

for dim in range(2):  # x and y
    # Extract positions for this dimension
    dim_ids = position_ids[:, :, dim]            # [batch, num_patches]

    # One-hot encode
    one_hot = F.one_hot(dim_ids, num_classes=10240).float()  # [batch, num_patches, 10240]

    # Zero out padding positions
    one_hot[padding_positions] = 0.0

    # Matrix multiply with embedding table
    # one_hot: [batch, num_patches, 10240]
    # table[dim]: [10240, 1152]
    dim_embedding = one_hot @ position_embedding_table[dim]  # [batch, num_patches, 1152]

    position_embeddings = position_embeddings + dim_embedding
```

> **Warning:** The reference uses `F.one_hot` and matrix multiplication rather than `nn.Embedding`. This is functionally equivalent to an embedding lookup but has different numerical behavior when position IDs are -1 (the padding sentinel). The one-hot approach naturally produces zero vectors for invalid positions (since they are zeroed out explicitly), while `nn.Embedding` would need index clamping or a separate masking step.

### TTNN Implementation Plan

The one-hot approach is computationally wasteful: it materializes a `[batch, num_patches, 10240]` tensor for each axis, most of which is zeros. For TTNN, an index-based lookup is more efficient.

**Recommended approach: `ttnn.embedding` lookup + element-wise add**

```python
# On device:
# position_embedding_table is stored as two separate weight tensors:
#   x_embed_table: [10240, 1152]  (on device, DRAM or L1)
#   y_embed_table: [10240, 1152]  (on device, DRAM or L1)

# x_pos_ids: [batch, num_patches]  (integer tensor, on device)
# y_pos_ids: [batch, num_patches]  (integer tensor, on device)

# Step 1: Lookup x-position embeddings
x_embeddings = ttnn.embedding(x_pos_ids, x_embed_table)  # [batch, num_patches, 1152]

# Step 2: Lookup y-position embeddings
y_embeddings = ttnn.embedding(y_pos_ids, y_embed_table)  # [batch, num_patches, 1152]

# Step 3: Sum the two positional components
position_embeddings = ttnn.add(x_embeddings, y_embeddings)  # [batch, num_patches, 1152]

# Step 4: Zero out padding positions
# Apply a mask tensor: [batch, num_patches, 1] broadcast over hidden_dim
position_embeddings = ttnn.mul(position_embeddings, valid_mask)
```

**Handling padding positions (-1 indices):**

The `ttnn.embedding` op does not support negative indices. Before the lookup, padding position IDs must be clamped to a valid index (e.g., 0). The masking in Step 4 then zeroes out the resulting embeddings for those positions. This is equivalent to the reference behavior.

```python
# On host (before transfer) or on device:
x_pos_ids_clamped = torch.clamp(x_pos_ids, min=0)
y_pos_ids_clamped = torch.clamp(y_pos_ids, min=0)
valid_mask = (padding_positions == False).unsqueeze(-1).float()  # [batch, num_patches, 1]
```

> **Tip:** The embedding tables are 10240 x 1152 = ~11.8M elements each (~23.6 MB in BF16 per table). These fit comfortably in DRAM on Wormhole. For L1 placement, the tables are too large for a single core's L1 (1 MB), but can be height-sharded across cores if embedding lookup latency becomes a bottleneck.

### Alternative: Gather-Based Approach

If `ttnn.embedding` does not support the required input shapes or data types, the lookup can be decomposed into:

1. Flatten position IDs to `[batch * num_patches]`.
2. Use `ttnn.index_select` or equivalent gather operation on the embedding table.
3. Reshape back to `[batch, num_patches, 1152]`.

This approach is more verbose but uses only primitive indexing operations.

## Full TTNN Patch Embedder Forward Pass

Combining the linear projection and positional embeddings:

```python
def gemma4_vision_patch_embedder_forward(
    pixel_values,       # [batch, num_patches, 768] - BF16, on device
    x_pos_ids,          # [batch, num_patches] - UINT32, on device (clamped, no -1)
    y_pos_ids,          # [batch, num_patches] - UINT32, on device (clamped, no -1)
    valid_mask,         # [batch, 1, 1, num_patches] - BF16, on device (0.0 for padding, 1.0 for valid)
    input_proj_weight,  # [768, 1152] - BF16, on device
    x_embed_table,      # [10240, 1152] - BF16, on device
    y_embed_table,      # [10240, 1152] - BF16, on device
):
    # Value scaling: [0,1] -> [-1,1]
    # Can be fused with the linear projection by pre-scaling the weight matrix:
    #   W_scaled = W * 2, bias_offset = W @ (-0.5 * 2) = -W @ 1
    # Or applied as element-wise ops:
    pixel_values = ttnn.sub(ttnn.mul(pixel_values, 2.0), 1.0)

    # Linear projection
    hidden_states = ttnn.linear(pixel_values, input_proj_weight)  # [batch, num_patches, 1152]

    # 2D position embedding lookup
    x_emb = ttnn.embedding(x_pos_ids, x_embed_table)   # [batch, num_patches, 1152]
    y_emb = ttnn.embedding(y_pos_ids, y_embed_table)   # [batch, num_patches, 1152]
    pos_emb = ttnn.add(x_emb, y_emb)                   # [batch, num_patches, 1152]

    # Zero out padding positions
    pos_emb = ttnn.mul(pos_emb, valid_mask)

    # Add positional information to hidden states
    output = ttnn.add(hidden_states, pos_emb)           # [batch, num_patches, 1152]

    return output
```

> **Tip:** The value scaling `2 * x - 1` can be fused into the linear projection weight to save two element-wise ops. Pre-multiply the weight matrix by 2 and add a bias vector equal to the negative column sums of the original weight, effectively computing `W * (2x - 1) = 2Wx - W*1`. This trades a one-time weight transformation for per-forward-pass savings.

## Variable Input Shapes: Program Caching and Tracing

The critical TTNN challenge for the patch embedder is that `num_patches` varies per image. This has direct consequences for program caching and tracing.

### How num_patches Varies

The number of patches depends on the resized image dimensions, which are determined by the token budget and aspect ratio:

| Token Budget | Approx. Total Patches | Example Grid Shapes |
|-------------|----------------------|---------------------|
| 70 | ~630 | 21x30, 30x21 |
| 140 | ~1260 | 30x42, 42x30 |
| 280 | ~2520 | 42x60, 60x42 |
| 560 | ~5040 | 60x84, 84x60 |
| 1120 | ~10080 | 84x120, 120x84 |

Within each token budget, different aspect ratios produce different grid shapes and therefore different `num_patches` values. The divisibility-by-48 constraint means that patch grid dimensions are always multiples of 3 along each axis (see [Chapter 1, `variable_resolution_processing.md`](../ch01_gemma4_vision_architecture/variable_resolution_processing.md)).

### Impact on TTNN Program Cache

TTNN's program cache is keyed on tensor shapes (among other attributes). When `num_patches` changes between images, the cache miss triggers recompilation of every kernel in the patch embedder:

- `ttnn.linear` with `[batch, N, 768]` input compiles a new program for each distinct `N`.
- `ttnn.embedding` with `[batch, N]` input similarly requires recompilation.
- `ttnn.add` and `ttnn.mul` with shape-dependent operands also miss cache.

For a deployment processing diverse images, this means the program cache grows proportionally to the number of distinct `num_patches` values encountered. Each unique shape incurs a compilation penalty on first occurrence (typically 10-100 ms per op).

### Mitigation Strategies

**Strategy 1: Pad to a Fixed Maximum Per Token Budget**

For each token budget, pad all inputs to the maximum possible `num_patches` for that budget. The attention mask already handles padding tokens, so the encoder produces correct results despite the wasted compute.

| Token Budget | Max Patches (budget ceiling) | Padding Overhead (worst case) |
|-------------|------------------------------|-------------------------------|
| 280 | ~2520 | ~25% for extreme aspect ratios |
| 560 | ~5040 | ~25% for extreme aspect ratios |

This reduces the number of distinct shapes to one per token budget (five total), making all five traceable.

> **Tip:** For initial bringup, fix the token budget to 280 (the default) and pad all inputs to the maximum patch count for that budget. This gives a single fixed shape through the entire pipeline, enabling tracing from day one.

**Strategy 2: Quantize to a Small Set of Supported Shapes**

Define a discrete set of supported `(height_patches, width_patches)` grid shapes (e.g., 10-20 shapes per token budget) and snap each image to the nearest supported shape. This limits program cache entries while preserving more aspect-ratio fidelity than padding to the maximum.

**Strategy 3: Accept Dynamic Shapes with Warm-Up**

Accept that the program cache will contain many entries and perform a warm-up pass at startup that compiles kernels for a representative set of shapes. This is the most flexible approach but has the highest memory cost for cached programs.

### Tracing Feasibility

TTNN tracing requires fixed tensor shapes throughout the traced graph. Given the variable `num_patches`:

- **Tracing is feasible if** all inputs are padded to a fixed shape per token budget (Strategy 1). The five token budgets yield five traces.
- **Tracing is not feasible if** arbitrary input shapes are allowed. The traced graph captures specific shapes and cannot adapt at replay time.

For production deployment, Strategy 1 (pad to fixed maximum per budget) combined with pre-tracing all five budgets is the recommended approach. The padding overhead is modest and the tracing benefit (eliminating per-step compilation and dispatch overhead) is substantial.

## Implementation Checklist

- [ ] Create `gemma4_vision_patch_embedder.py` in the Gemma 4 TTNN module directory
- [ ] Implement value scaling (`2x - 1`) as element-wise ops or fuse into the linear weight
- [ ] Port `input_proj` as `ttnn.linear` with weight shape `[768, 1152]`
- [ ] Split the position embedding table `[2, 10240, 1152]` into `x_embed_table` and `y_embed_table` during weight loading
- [ ] Implement position embedding lookup using `ttnn.embedding` for x and y axes
- [ ] Implement padding mask application (clamp position IDs, multiply by validity mask)
- [ ] Sum x and y position embeddings and add to projected hidden states
- [ ] Validate against HuggingFace reference with PCC > 0.999 for BF16
- [ ] Test with at least three different grid shapes (landscape, portrait, square) within the 280-token budget
- [ ] Decide on padding strategy (fixed max per budget recommended) and document the chosen approach
- [ ] If using fixed padding, verify that padding tokens do not affect final output after pooling and mask removal

---

**Next:** [`adaptive_pooling_port.md`](./adaptive_pooling_port.md) — Porting the adaptive 2D average pooler and the RMSNorm + linear projection.
