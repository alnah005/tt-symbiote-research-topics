# Adaptive Pooling Port

This file covers the TTNN porting strategy for `Gemma4VisionPooler` and the downstream `Gemma4MultimodalEmbedder`. The pooler reduces the encoded patch sequence to the target token budget via 2D average pooling, and the embedder projects the result into the language model's hidden dimension. Together, these form the exit path of the vision encoder.

## Gemma 4 Adaptive 2D Average Pooling

The pooler takes the encoder's output — a sequence of hidden states with explicit 2D spatial coordinates — and averages patches within grid cells to produce a smaller number of output tokens. This is the mechanism that maps ~2520 pre-pooling patches down to ~280 output tokens (for the default token budget).

### How the Pooler Works

The pooling operates in five steps:

**Step 1: Compute grid cell assignments**

Each patch has a 2D position `(x, y)` from its `pixel_position_ids`. The pooler assigns each patch to a grid cell by dividing its coordinates by the `pooling_kernel_size` (default 3):

```python
# pixel_position_ids: [batch, num_patches, 2]
# pooling_kernel_size: 3
cell_x = pixel_position_ids[:, :, 0] // pooling_kernel_size  # [batch, num_patches]
cell_y = pixel_position_ids[:, :, 1] // pooling_kernel_size  # [batch, num_patches]
```

For an image with a patch grid of 60x42 (landscape, 2520 patches):
- Cell x-coordinates range from 0 to 19 (60/3 = 20 cells along x).
- Cell y-coordinates range from 0 to 13 (42/3 = 14 cells along y).
- Total cells: 20 x 14 = 280, matching the default token budget.

**Step 2: Linearize cell assignments**

The 2D cell coordinates are converted to a 1D cell index:

```python
# cells_per_row = max(cell_x) + 1 = width_patches // pooling_kernel_size
cell_index = cell_y * cells_per_row + cell_x  # [batch, num_patches]
```

Each `cell_index` value identifies which output token a given patch contributes to. Padding patches (with position IDs of -1) produce invalid cell indices and are excluded.

**Step 3: One-hot encoding and normalized aggregation**

The reference implementation uses one-hot encoding of cell assignments to perform the averaging:

```python
# cell_index: [batch, num_patches] with values in [0, output_length)
# hidden_states: [batch, num_patches, 1152]

# One-hot encode cell assignments
one_hot = F.one_hot(cell_index, num_classes=output_length).float()  # [batch, num_patches, output_length]

# Normalize: each column sums to the number of patches in that cell (ideally kernel_size^2 = 9)
# Divide by kernel_size^2 to compute the average
one_hot = one_hot / (pooling_kernel_size ** 2)  # [batch, num_patches, output_length]

# Zero out padding positions
one_hot[padding_positions] = 0.0

# Weighted sum: transpose and matmul
# one_hot^T: [batch, output_length, num_patches]
# hidden_states: [batch, num_patches, 1152]
pooled = torch.bmm(one_hot.transpose(1, 2), hidden_states)  # [batch, output_length, 1152]
```

> **Warning:** The division by `pooling_kernel_size^2` assumes each grid cell contains exactly 9 patches. This holds when the patch grid dimensions are both divisible by 3 (guaranteed by the divisibility-by-48 constraint). Edge cells in images that do not perfectly fill the grid may have fewer patches, but the divisibility constraint prevents this from occurring.

**Step 4: Scaling**

The pooled output is scaled by $\sqrt{\text{hidden\_size}}$:

```python
pooled = pooled * math.sqrt(1152)  # scale factor ~33.94
```

This is standard hidden-dimension scaling (the same $\sqrt{d}$ convention used throughout the model, e.g., in attention layers). It keeps the magnitude of the pooled representations consistent with what downstream RMSNorm and linear projection layers expect.

**Step 5: Validity mask**

A boolean mask indicates which output tokens are valid (correspond to a non-empty grid cell):

```python
# Count patches per cell
cell_counts = one_hot.sum(dim=1)  # [batch, output_length]
valid_mask = cell_counts > 0      # [batch, output_length]
```

## The pooling_kernel_size=3 Interaction with the Grid

The relationship between `pooling_kernel_size`, `patch_size`, and the image dimensions is central to the pooler's operation:

$$
\text{output\_tokens\_per\_axis} = \frac{\text{image\_pixels\_per\_axis}}{\text{patch\_size} \times \text{pooling\_kernel\_size}} = \frac{\text{image\_pixels\_per\_axis}}{48}
$$

This means:
- Each output token aggregates information from a $48 \times 48$ pixel region of the original image.
- The token budget determines the total pixel count: $\text{budget} \times 48^2 = \text{budget} \times 2304$ pixels.
- For the default budget of 280: $280 \times 2304 = 645{,}120$ pixels.

### Grid Cell Sizes Are Uniform

Because the image dimensions are constrained to be divisible by 48 (= 16 x 3), and patches tile the image without remainder, every grid cell contains exactly $3 \times 3 = 9$ patches. There are no partial cells at the edges. This is a deliberate design choice that simplifies the pooler implementation.

| Token Budget | Total Patches | Output Tokens | Patches per Cell |
|-------------|---------------|---------------|-----------------|
| 70 | ~630 | ~70 | 9 |
| 140 | ~1260 | ~140 | 9 |
| 280 | ~2520 | ~280 | 9 |
| 560 | ~5040 | ~560 | 9 |
| 1120 | ~10080 | ~1120 | 9 |

> **Tip:** The uniform 9-patches-per-cell property means the pooler is mathematically equivalent to reshaping the patch sequence into a 2D grid, applying a non-overlapping 3x3 average pooling with stride 3, and flattening back to 1D. This observation opens the door to using standard 2D pooling ops in TTNN.

## Optional Standardization

After pooling and scaling, the vision model optionally applies a learned standardization transform:

```python
# std_bias: [1152] - learned parameter
# std_scale: [1152] - learned parameter
hidden_states = (hidden_states - std_bias) * std_scale
```

This is a simple element-wise operation that maps directly to TTNN:

```python
hidden_states = ttnn.mul(ttnn.sub(hidden_states, std_bias), std_scale)
```

The standardization parameters are present in the Gemma 4 31B checkpoint and should be loaded alongside the other vision encoder weights. If the model config indicates `use_standardization=False`, this step is skipped.

## Comparison with Gemma 3 Fixed Pooling

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Pooling type | Fixed average pooling | Adaptive 2D grid-based average pooling |
| Input patches | Fixed 4096 (64x64 grid) | Variable (depends on image size and aspect ratio) |
| Output tokens | Fixed 256 | Configurable: 70, 140, 280, 560, 1120 |
| Kernel size | Derived from fixed grid (4x4 blocks) | `pooling_kernel_size=3` (3x3 blocks) |
| Spatial awareness | Implicit (fixed grid ordering) | Explicit (2D position IDs determine cell assignment) |
| Padding handling | Not needed (fixed input) | Required (variable input, padding positions excluded) |
| Standardization | None | Optional learned bias + scale |
| Post-pooling scaling | None | $\sqrt{1152} \approx 33.94$ |
| TTNN module | Part of `multi_modal_projector.py` | New: `gemma4_vision_pooler.py` |

### What Changed and Why

The shift from fixed to adaptive pooling is driven by the variable-resolution input design:

1. **Variable grid shapes require explicit cell assignment.** In Gemma 3, the 64x64 grid is fixed, so pooling can be a simple reshape-and-mean. In Gemma 4, the grid shape changes per image, so the pooler must compute cell assignments from position IDs.

2. **Aspect-ratio preservation changes the grid shape, not the token count.** The divisibility-by-48 constraint on the resized dimensions ensures that (h/48)*(w/48) always equals the token budget exactly, so a 16:9 landscape and a 9:16 portrait targeting the same 280-token budget produce the same number of output tokens but with different grid shapes (e.g., a wider grid vs. a taller grid). The pooler must therefore handle variable grid geometries even though the total count is fixed (see the examples in [Chapter 1, `variable_resolution_processing.md`](../ch01_gemma4_vision_architecture/variable_resolution_processing.md)).

3. **The $\sqrt{d}$ scaling follows the standard hidden-dimension convention.** Gemma 3 did not need this because its pooling was integrated differently. Gemma 4 applies $\sqrt{\text{hidden\_size}}$ scaling — the same convention used throughout the model — to keep hidden-state magnitudes consistent for downstream layers.

## TTNN Implementation Options

Three approaches for implementing the adaptive pooler in TTNN, ranked by practicality.

### Option 1: Reshape to 2D Grid + ttnn.avg_pool2d

Since every grid cell contains exactly 9 patches (as established above), the pooler is mathematically equivalent to a standard non-overlapping 2D average pool:

```python
# Reshape from 1D sequence to 2D spatial grid
# hidden_states: [batch, num_patches, 1152]
# Rearrange to: [batch, 1152, height_patches, width_patches]  (NCHW format)
hidden_states_2d = ttnn.reshape(hidden_states, [batch, height_patches, width_patches, 1152])
hidden_states_2d = ttnn.permute(hidden_states_2d, [0, 3, 1, 2])  # -> [batch, 1152, H_p, W_p]

# Apply 2D average pooling with kernel_size=3, stride=3
pooled_2d = ttnn.avg_pool2d(hidden_states_2d, kernel_size=3, stride=3, padding=0)
# -> [batch, 1152, H_p/3, W_p/3]

# Flatten back to [batch, output_length, 1152]
pooled = ttnn.permute(pooled_2d, [0, 2, 3, 1])
pooled = ttnn.reshape(pooled, [batch, output_length, 1152])
```

**Advantages:**
- Uses a well-understood, potentially optimized TTNN op.
- The mathematical equivalence is exact (no approximation).
- Conceptually simple.

**Challenges:**
- `ttnn.avg_pool2d` may not support all combinations of kernel size, stride, and channel count. Verify that `kernel_size=3, stride=3, channels=1152` is supported.
- The reshape from 1D sequence to 2D grid requires knowing `height_patches` and `width_patches` per image, which vary. In a padded batch, all images share the same `num_patches` but may have different underlying grid shapes. If the batch uses a uniform padded grid, this approach works; if images have genuinely different grids, per-image processing may be needed.
- Padding patches must be handled correctly. If the input is padded to a uniform grid, padding patches should contain zeros so they do not corrupt the average. Since the grid dimensions are always multiples of 3, padding that preserves this property is required.

> **Tip:** This is the recommended approach for initial bringup if `ttnn.avg_pool2d` supports the required configuration. The reshape overhead is negligible compared to the matmul cost of the encoder layers.

### Option 2: Manual Reshape + ttnn.mean

If `ttnn.avg_pool2d` is not available or does not support the required parameters, the pooling can be decomposed into reshape and reduction operations:

```python
# hidden_states: [batch, num_patches, 1152]
# Reshape to group patches into 3x3 blocks:
# [batch, H_p/3, 3, W_p/3, 3, 1152]
hidden_states = ttnn.reshape(hidden_states, [batch, h_cells, 3, w_cells, 3, 1152])

# Transpose to bring the two kernel dimensions together:
# [batch, H_p/3, W_p/3, 3, 3, 1152]  (or equivalent)
hidden_states = ttnn.permute(hidden_states, [0, 1, 3, 2, 4, 5])

# Reshape to merge kernel dims: [batch, h_cells * w_cells, 9, 1152]
hidden_states = ttnn.reshape(hidden_states, [batch, h_cells * w_cells, 9, 1152])

# Average over the kernel dimension (dim=2)
pooled = ttnn.mean(hidden_states, dim=2)  # [batch, output_length, 1152]
```

**Advantages:**
- Uses only reshape, permute, and mean — all fundamental TTNN operations.
- No dependency on a specific pooling kernel.

**Challenges:**
- Multi-dimensional reshape and permute may not be fully optimized in TTNN for all rank combinations. The 6D reshape in particular may require careful memory layout management.
- The permute step may trigger a data copy if the memory layout does not support the requested axis reordering natively.
- The `ttnn.mean` op must support reduction along an arbitrary inner dimension.

**Simplification for TTNN's 4D Tensor Limit:**

TTNN tensors are limited to 4 dimensions in many ops. The 6D intermediate can be avoided by processing the two kernel dimensions sequentially:

```python
# Step 1: Average along the row kernel dimension
# hidden_states: [batch, num_patches, 1152] where num_patches = H_p * W_p
# Reshape to [batch, H_p, W_p, 1152]
hs = ttnn.reshape(hidden_states, [batch, h_patches, w_patches, 1152])

# Group along width: [batch, H_p, W_p/3, 3, 1152] -> not 4D
# Instead: reshape width into groups, keeping 4D:
# [batch * H_p, W_p/3, 3, 1152]
hs = ttnn.reshape(hs, [batch * h_patches, w_cells, 3, 1152])
hs = ttnn.mean(hs, dim=2)  # [batch * H_p, W_p/3, 1152]

# Step 2: Average along the height kernel dimension
hs = ttnn.reshape(hs, [batch, h_patches, w_cells, 1152])
# Group along height: [batch * W_p/3, H_p/3, 3, 1152]
hs = ttnn.permute(hs, [0, 2, 1, 3])  # [batch, w_cells, h_patches, 1152]
hs = ttnn.reshape(hs, [batch * w_cells, h_cells, 3, 1152])
hs = ttnn.mean(hs, dim=2)  # [batch * W_p/3, H_p/3, 1152]

# Reshape back to [batch, output_length, 1152]
hs = ttnn.reshape(hs, [batch, w_cells, h_cells, 1152])
hs = ttnn.permute(hs, [0, 2, 1, 3])  # [batch, h_cells, w_cells, 1152]
pooled = ttnn.reshape(hs, [batch, output_length, 1152])
```

This approach stays within 4D throughout and performs the 3x3 average as two sequential 1D reductions.

### Option 3: One-Hot Matmul (Matching Reference)

Directly replicate the reference implementation's one-hot matrix multiplication:

```python
# cell_assignment: [batch, num_patches] with values in [0, output_length)
# Construct the pooling matrix on host
pooling_matrix = torch.zeros(num_patches, output_length)
for i in range(num_patches):
    pooling_matrix[i, cell_assignment[i]] = 1.0 / 9.0

# Transfer to device
pooling_matrix_tt = ttnn.from_torch(pooling_matrix)

# Apply: hidden_states^T @ pooling_matrix
# hidden_states: [batch, num_patches, 1152]
# pooling_matrix: [num_patches, output_length]
pooled = ttnn.matmul(
    ttnn.permute(hidden_states, [0, 2, 1]),   # [batch, 1152, num_patches]
    pooling_matrix_tt                          # [num_patches, output_length]
)  # [batch, 1152, output_length]
pooled = ttnn.permute(pooled, [0, 2, 1])      # [batch, output_length, 1152]
```

**Advantages:**
- Mathematically identical to the reference.
- Uses `ttnn.matmul`, which is heavily optimized on Wormhole.
- The pooling matrix is sparse (each row has exactly one non-zero entry), but the dense matmul still works correctly.

**Challenges:**
- The pooling matrix has shape `[num_patches, output_length]`, e.g., `[2520, 280]`. This is a relatively small matmul and may not fully utilize Wormhole's compute grid.
- The matrix is mostly zeros (only ~0.36% non-zero for the 280-token budget). A dense matmul wastes significant compute on zero multiplications.
- The pooling matrix changes whenever `num_patches` or the grid shape changes, requiring reconstruction and retransfer per unique image shape.
- Two permute operations add overhead.

> **Risk (Medium):** If the image shapes are highly variable and the padding strategy from [`patch_embedding_port.md`](./patch_embedding_port.md) is not adopted, the pooling matrix must be recomputed and transferred for each unique shape. This adds host-device transfer latency that partially negates the benefit of running the pooler on device. Mitigation: fix the input shape per token budget (Strategy 1 from [`patch_embedding_port.md`](./patch_embedding_port.md)), which fixes the pooling matrix per budget.

### Recommendation

For initial bringup, **Option 1 (reshape + avg_pool2d)** is recommended if the TTNN op supports the required configuration. It is the simplest, most efficient, and most maintainable approach. If `ttnn.avg_pool2d` is unavailable, fall back to **Option 2 (manual reshape + mean)** with the 4D-safe decomposition.

Option 3 (one-hot matmul) is the closest to the reference and may be useful for initial correctness validation, but it is not recommended for production due to the sparse-matrix inefficiency.

## RMSNorm + Linear Projection (Multimodal Embedder)

After pooling (and optional standardization), the `Gemma4MultimodalEmbedder` projects the vision hidden states to the language model dimension. This module is straightforward and maps directly to existing TTNN ops.

### Reference Implementation

```python
class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(self, config):
        self.embedding_pre_projection_norm = Gemma4RMSNorm(
            dim=1152, eps=1e-6, with_scale=False
        )
        self.embedding_projection = nn.Linear(1152, 5376, bias=False)

    def forward(self, inputs_embeds):
        # inputs_embeds: [total_valid_tokens, 1152]
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        projected = self.embedding_projection(normed)
        return projected  # [total_valid_tokens, 5376]
```

### TTNN Implementation

```python
def gemma4_multimodal_embedder_forward(
    hidden_states,         # [batch, output_length, 1152] - BF16, on device
    rms_norm_weight,       # None (no learnable scale for this norm)
    projection_weight,     # [1152, 5376] - BF16, on device
):
    # RMSNorm without learnable scale
    hidden_states = ttnn.rms_norm(hidden_states, epsilon=1e-6)
    # Note: no weight multiplication since with_scale=False

    # Linear projection to language model dimension
    output = ttnn.linear(hidden_states, projection_weight)  # [batch, output_length, 5376]

    return output
```

**Key points:**

- The RMSNorm here has **no learnable scale parameter** (`with_scale=False`). It normalizes the hidden states to unit RMS but does not apply a learned per-channel rescaling. The existing `gemma_vision_rmsnorm.py` from Gemma 3 may need a flag to disable the scale, or this can be a simple `ttnn.rms_norm` call without a weight tensor.

- The linear projection `[1152, 5376]` is a standard bias-free matmul. The output dimension of 5376 is the language model's hidden size for the 31B variant. This is a medium-sized matmul that maps well to Wormhole's compute grid.

- The input to the embedder is `[batch, output_length, 1152]` (or `[total_valid_tokens, 1152]` after padding removal). For TTNN, keeping the padded representation through the embedder and removing padding afterward avoids ragged tensor shapes on device.

> **Tip:** The RMSNorm + linear projection is functionally identical to the same pattern used in many language model components. If the Gemma 3 TTNN codebase already has a fused RMSNorm + linear path, it can be reused directly with updated weight shapes.

## Complete Pooler + Embedder Pipeline

Putting the full exit path together:

```
Encoder output                                    [batch, num_patches, 1152]
    │
    ▼  Compute grid cell assignments
    │  (from pixel_position_ids and pooling_kernel_size=3)
    │
    ▼  2D average pooling (reshape + avg_pool2d)
Pooled output                                     [batch, output_length, 1152]
    │
    ▼  Scale by sqrt(1152) ≈ 33.94
    │
    ▼  Optional standardization: (x - bias) * scale
    │
    ▼  Validity mask computation
    │
    ▼  RMSNorm (no learnable scale)
    │
    ▼  Linear projection (1152 → 5376)
Projected output                                  [batch, output_length, 5376]
    │
    ▼  Remove padding (on host or at LM integration)
Soft tokens for language model                    [total_valid_tokens, 5376]
```

## Implementation Checklist

- [ ] Create `gemma4_vision_pooler.py` in the Gemma 4 TTNN module directory
- [ ] Implement grid cell assignment computation (can be done on host and passed as input)
- [ ] Implement 2D average pooling using Option 1 (avg_pool2d) or Option 2 (manual reshape+mean)
- [ ] Apply $\sqrt{1152}$ scaling after pooling
- [ ] Implement optional standardization (load `std_bias` and `std_scale` from checkpoint)
- [ ] Compute validity mask for downstream padding removal
- [ ] Validate pooler output against HuggingFace reference with PCC > 0.999 for BF16
- [ ] Create `gemma4_multimodal_embedder.py` (or extend existing projector module)
- [ ] Implement RMSNorm without learnable scale
- [ ] Implement linear projection `[1152, 5376]` using `ttnn.linear`
- [ ] Validate end-to-end pooler + embedder pipeline against HuggingFace reference
- [ ] Test with multiple token budgets (at minimum: 70, 280, 1120) to verify adaptive behavior
- [ ] Profile pooler latency and confirm it is a small fraction of total encoder time

---

**Next:** [Chapter 5 — CPU vs. TTNN Latency Analysis](../ch05_cpu_vs_ttnn_latency/index.md) — Estimating whether the porting effort for these modules is justified by the latency improvement.
