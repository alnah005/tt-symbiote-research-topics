# New Implementation Modules

This file covers the four modules that must be written from scratch for the Gemma 4 vision encoder TTNN port. These modules have no Gemma 3 equivalent and implement capabilities that are entirely new in Gemma 4: 2D factored RoPE, 2D learned position embeddings, variable-resolution image preprocessing, and adaptive 2D pooling. Together they account for approximately 35% of the codebase and require an estimated 6-10 days of engineering effort.

All new files are placed under `models/demos/multimodal/gemma4/tt/`.

## 2D RoPE Module

**File:** `gemma4_vision_rope.py`
**Effort:** 2-3 days
**Dependencies:** None (can be developed and validated independently)

### Purpose

Compute and apply 2D factored Rotary Position Embeddings for the Gemma 4 vision encoder attention layers. This module handles the core gap identified in [Chapter 3 — TTNN RoPE Gap Analysis](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md): the existing TTNN RoPE kernels support only 1D positions, while Gemma 4 vision requires 2D (x, y) spatial coordinates with a split head dimension.

### Design

The module has two components:

1. **Precomputation** (runs on CPU): given 2D position IDs and the inverse frequency table, produce cos/sin tables of shape `[batch, 1, num_patches, 72]`.
2. **Application** (runs on device): given Q or K tensors and the cos/sin tables, apply the rotation using element-wise TTNN operations.

### Recommended Implementation: Strategy 1 (CPU Precompute)

Per the recommendation in [Chapter 3](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md#recommendation), use CPU precomputation for initial bringup. The implementation is:

```python
import torch
import ttnn
import math

class TtGemma4VisionRoPE:
    """2D factored RoPE for Gemma 4 vision encoder."""

    def __init__(self, head_dim=72, rope_theta=100.0, device=None):
        self.head_dim = head_dim
        self.half_dim = head_dim // 2        # 36: dimensions per spatial axis
        self.freq_dim = head_dim // 4         # 18: frequency entries per axis
        self.device = device

        # Precompute inverse frequencies (same formula as standard RoPE)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.freq_dim, dtype=torch.float32) * 2.0 / self.half_dim)
        )
        self.inv_freq = inv_freq              # [18]

        # Cache: maps (height_patches, width_patches) -> (cos_tt, sin_tt)
        self._cache = {}

    def get_cos_sin(self, position_ids):
        """
        Precompute cos/sin on CPU and transfer to device.

        Args:
            position_ids: [batch, num_patches, 2] int tensor of (x, y) coords

        Returns:
            cos_tt: [batch, 1, num_patches, 72] on device
            sin_tt: [batch, 1, num_patches, 72] on device
        """
        inv_freq = self.inv_freq[None, :, None]  # [1, 18, 1]
        all_cos, all_sin = [], []

        for dim_idx in range(2):
            dim_pos = position_ids[:, :, dim_idx].float()     # [batch, num_patches]
            dim_pos = dim_pos[:, None, :]                      # [batch, 1, num_patches]
            freqs = torch.bmm(
                inv_freq.expand(position_ids.shape[0], -1, -1),
                dim_pos,
            ).transpose(1, 2)                                  # [batch, num_patches, 18]
            emb = torch.cat([freqs, freqs], dim=-1)            # [batch, num_patches, 36]
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())

        cos = torch.cat(all_cos, dim=-1).unsqueeze(1)         # [batch, 1, num_patches, 72]
        sin = torch.cat(all_sin, dim=-1).unsqueeze(1)

        cos_tt = ttnn.from_torch(cos.bfloat16(), device=self.device, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin.bfloat16(), device=self.device, layout=ttnn.TILE_LAYOUT)
        return cos_tt, sin_tt

    @staticmethod
    def apply(x_tt, cos_tt, sin_tt):
        """
        Apply 2D factored RoPE on device.

        Args:
            x_tt: [batch, num_heads, num_patches, 72] on device
            cos_tt: [batch, 1, num_patches, 72] on device
            sin_tt: [batch, 1, num_patches, 72] on device

        Returns:
            Rotated tensor, same shape as x_tt.
        """
        # Split along head_dim into x-axis (first 36) and y-axis (last 36)
        x_first = x_tt[:, :, :, :36]
        x_second = x_tt[:, :, :, 36:]
        cos_first = cos_tt[:, :, :, :36]
        cos_second = cos_tt[:, :, :, 36:]
        sin_first = sin_tt[:, :, :, :36]
        sin_second = sin_tt[:, :, :, 36:]

        def rotate_half(t):
            """Swap halves and negate the first, per standard RoPE."""
            half = t.shape[-1] // 2
            t1 = t[:, :, :, :half]
            t2 = t[:, :, :, half:]
            return ttnn.concat([ttnn.neg(t2), t1], dim=-1)

        # Apply rotation independently to each spatial half
        y_first = ttnn.add(
            ttnn.mul(x_first, cos_first),
            ttnn.mul(rotate_half(x_first), sin_first),
        )
        y_second = ttnn.add(
            ttnn.mul(x_second, cos_second),
            ttnn.mul(rotate_half(x_second), sin_second),
        )

        return ttnn.concat([y_first, y_second], dim=-1)
```

### Caching Strategy

The cos/sin tables depend only on the patch grid dimensions, not on the image content. For the five standard token budgets, cache the tables to avoid recomputation:

| Token Budget | Typical Patch Grid | Patches | cos/sin Size (BF16) |
|-------------|-------------------|---------|---------------------|
| 70 | 21x30 (or similar) | ~630 | ~181 KB |
| 140 | 30x42 | ~1260 | ~363 KB |
| 280 | 42x60 | ~2520 | ~726 KB |
| 560 | 60x84 | ~5040 | ~1.4 MB |
| 1120 | 84x120 | ~10080 | ~2.9 MB |

Even at the largest budget, the cos/sin tables are small enough to keep in device DRAM permanently. Transfer cost from host is negligible (< 250 microseconds at PCIe Gen4 bandwidth for the largest case).

> **Tip:** Precompute and cache cos/sin tables for the five standard budgets during model initialization. At inference time, select the cached tables based on the image's resolution. This eliminates all per-image RoPE precomputation cost.

### Validation Checklist

- [ ] Compute `inv_freq` and verify it matches HuggingFace `Gemma4VisionRotaryEmbedding.inv_freq` exactly in float32.
- [ ] For a test image with grid dimensions 42x60 (280 tokens after pooling, 2520 patches), compute cos/sin tables and verify exact match against HuggingFace output in float32.
- [ ] Apply rotation to random Q tensor `[1, 16, 2520, 72]` and verify PCC > 0.999 against HuggingFace `apply_multidimensional_rope` in BF16.
- [ ] Verify that `rotate_half` operates independently on each 36-element spatial half — not across the split boundary.
- [ ] Profile the element-wise application and confirm overhead is < 5% of total attention latency.

### Upgrade Path

If profiling reveals that the element-wise RoPE application is a bottleneck (unlikely for the vision encoder's moderate sequence lengths), the module can be upgraded to Strategy 2 (full on-device computation) or Strategy 3 (custom fused kernel) as described in [Chapter 3](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md#strategy-comparison). The `apply` interface remains the same regardless of the underlying implementation.

## 2D Learned Position Embedding

**File:** `gemma4_vision_position_embedding.py`
**Effort:** 1-2 days
**Dependencies:** None (can be developed and validated independently)

### Purpose

Implement the 2D learned position embedding lookup that adds spatial information to patch embeddings before the encoder layers. This replaces the 1D sequential position embedding used in Gemma 3 SigLIP.

### How It Works

Gemma 4 stores a position embedding table of shape `[2, 10240, 1152]`:

- `table[0, :, :]` contains position embeddings for the x-axis (10240 entries).
- `table[1, :, :]` contains position embeddings for the y-axis (10240 entries).

For each patch at grid position `(x, y)`:

```
position_embedding = table[0, x, :] + table[1, y, :]
```

The x and y embeddings are looked up independently and summed to form a single position vector per patch.

### Implementation

```python
class TtGemma4VisionPositionEmbedding:
    """2D learned position embedding for Gemma 4 vision encoder."""

    def __init__(self, device, state_dict, base_key="vision_tower.position_embedding_table"):
        # Load the [2, 10240, 1152] table
        table = state_dict[base_key]  # [2, 10240, 1152]

        # Split into x-axis and y-axis embedding tables
        self.x_table = ttnn.from_torch(
            table[0].unsqueeze(0),  # [1, 10240, 1152]
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        self.y_table = ttnn.from_torch(
            table[1].unsqueeze(0),  # [1, 10240, 1152]
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(self, position_ids):
        """
        Look up and sum position embeddings.

        Args:
            position_ids: [batch, num_patches, 2] int tensor of (x, y) coords.
                          Padding positions have value -1.

        Returns:
            pos_embed: [batch, num_patches, 1152] on device
        """
        x_ids = position_ids[:, :, 0]  # [batch, num_patches]
        y_ids = position_ids[:, :, 1]

        # Look up embeddings for each axis
        x_embed = ttnn.embedding(x_ids, self.x_table)  # [batch, num_patches, 1152]
        y_embed = ttnn.embedding(y_ids, self.y_table)

        # Sum x and y embeddings
        pos_embed = ttnn.add(x_embed, y_embed)

        # Zero out padding positions (where position_ids == -1)
        # The padding mask should be computed on the host and passed in
        return pos_embed
```

### Handling Padding Positions

Patches with `position_ids == -1` are padding and must produce zero-valued position embeddings. There are two approaches:

1. **Clamp-and-mask:** Replace -1 with 0 before the embedding lookup, then multiply the output by a binary mask that zeros out padding positions. This avoids out-of-bounds indexing.

2. **Extended table:** Add a zero row at index 0 of the embedding table, shift all valid indices by +1, and map -1 to 0. This eliminates the need for a separate masking step but wastes one table row.

Approach 1 is simpler and maps cleanly to TTNN ops:

```python
# Clamp invalid indices to 0
x_ids_safe = ttnn.maximum(x_ids, 0)
y_ids_safe = ttnn.maximum(y_ids, 0)

# Look up
x_embed = ttnn.embedding(x_ids_safe, self.x_table)
y_embed = ttnn.embedding(y_ids_safe, self.y_table)
pos_embed = ttnn.add(x_embed, y_embed)

# Mask: valid_mask is [batch, num_patches, 1], 1 for real patches, 0 for padding
pos_embed = ttnn.mul(pos_embed, valid_mask)
```

### Alternative: One-Hot Based Lookup

The HuggingFace reference implementation uses a one-hot + batch matmul approach:

```python
# HuggingFace reference (PyTorch):
one_hot_x = F.one_hot(x_ids, num_classes=10240).float()  # [batch, num_patches, 10240]
x_embed = torch.bmm(one_hot_x, table[0].unsqueeze(0).expand(batch, -1, -1))
```

This approach is functionally equivalent but creates a large intermediate tensor (`[batch, num_patches, 10240]`). The `ttnn.embedding` approach is preferred because it avoids this memory overhead.

> **Warning:** The position embedding table is large: `[2, 10240, 1152]` is approximately 47 MB in BF16. Ensure sufficient device DRAM is available. For multi-chip configurations, the table can be replicated on each device since it is read-only.

### Validation Checklist

- [ ] Load the `position_embedding_table` from a real checkpoint and verify shape is `[2, 10240, 1152]`.
- [ ] For a 42x60 patch grid, compute position IDs and look up embeddings. Verify PCC > 0.999 against HuggingFace.
- [ ] Verify that padding positions (position_ids == -1) produce zero-valued embeddings.
- [ ] Test with multiple grid sizes to confirm the lookup works for all five standard token budgets.
- [ ] Verify that the summed (x + y) embedding matches HuggingFace's one-hot matmul result.

## Variable-Resolution Image Preprocessor

**File:** `gemma4_variable_resolution.py`
**Effort:** 1-2 days
**Dependencies:** None (host-only, no TTNN dependencies)

### Purpose

This is a host-side (CPU) module that preprocesses images for the Gemma 4 vision encoder. It computes the target image dimensions, enforces the divisibility-by-48 constraint, extracts patches, and generates the 2D position IDs that the TTNN modules require.

This module does **not** run on the Tenstorrent device. It runs on the host CPU and produces tensors that are transferred to the device as inputs to the vision encoder.

### Responsibilities

1. **Compute target dimensions** given a token budget and the image's aspect ratio.
2. **Enforce divisibility by 48** (patch_size=16 times pooling_kernel_size=3).
3. **Resize the image** to the computed target dimensions.
4. **Apply value scaling:** `pixel_values = 2 * (pixel_values - 0.5)`.
5. **Extract patches** by reshaping the image into a sequence of flattened 16x16x3 vectors.
6. **Compute 2D position IDs** for each patch: `(x, y)` grid coordinates.
7. **Handle batching**: pad patch sequences to the maximum length in the batch and generate padding masks.

### Target Dimension Computation

The algorithm for computing target dimensions from a token budget:

```python
def compute_target_dimensions(
    original_height,
    original_width,
    token_budget=280,
    patch_size=16,
    pooling_kernel_size=3,
):
    """
    Compute target image dimensions that respect aspect ratio and constraints.

    The number of output tokens after pooling equals:
        (target_h / (patch_size * pooling_kernel_size)) *
        (target_w / (patch_size * pooling_kernel_size))

    This must equal the token_budget.

    Args:
        original_height, original_width: original image dimensions
        token_budget: target number of vision tokens (70, 140, 280, 560, 1120)
        patch_size: 16 for Gemma 4
        pooling_kernel_size: 3 for Gemma 4

    Returns:
        (target_height, target_width): both divisible by 48
    """
    divisor = patch_size * pooling_kernel_size  # 48

    # Total pixels budget: token_budget * divisor^2
    total_pixels = token_budget * (divisor ** 2)

    # Preserve aspect ratio
    aspect_ratio = original_width / original_height
    target_height = int(math.sqrt(total_pixels / aspect_ratio))
    target_width = int(target_height * aspect_ratio)

    # Round to nearest multiple of 48
    target_height = max(divisor, round(target_height / divisor) * divisor)
    target_width = max(divisor, round(target_width / divisor) * divisor)

    return target_height, target_width
```

### Position ID Generation

After resizing and patch extraction, generate 2D position IDs:

```python
def compute_position_ids(height_patches, width_patches):
    """
    Compute 2D position IDs for a patch grid.

    Args:
        height_patches: number of patches along height
        width_patches: number of patches along width

    Returns:
        position_ids: [1, num_patches, 2] tensor of (x, y) coordinates
    """
    num_patches = height_patches * width_patches
    position_ids = torch.zeros(1, num_patches, 2, dtype=torch.long)

    for idx in range(num_patches):
        y = idx // width_patches
        x = idx % width_patches
        position_ids[0, idx, 0] = x
        position_ids[0, idx, 1] = y

    return position_ids
```

For efficiency, this can be vectorized:

```python
def compute_position_ids(height_patches, width_patches):
    y_coords = torch.arange(height_patches).unsqueeze(1).expand(-1, width_patches).reshape(-1)
    x_coords = torch.arange(width_patches).unsqueeze(0).expand(height_patches, -1).reshape(-1)
    position_ids = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)
    return position_ids  # [1, num_patches, 2]
```

### Batching and Padding

When processing a batch of images with different aspect ratios:

1. Compute target dimensions independently for each image.
2. Find the maximum patch count across the batch.
3. Pad shorter sequences with zero-valued patches.
4. Set position IDs to -1 for padding positions.
5. Generate a boolean mask indicating real vs. padding patches.

```python
def batch_preprocess(images, token_budget=280):
    """
    Preprocess a batch of images with variable resolutions.

    Returns:
        pixel_patches: [batch, max_patches, 768] — flattened patches (padded)
        position_ids:  [batch, max_patches, 2]   — 2D coords (-1 for padding)
        valid_mask:    [batch, max_patches]       — True for real patches
    """
    all_patches = []
    all_pos_ids = []
    all_lengths = []

    for img in images:
        h, w = img.shape[-2:]
        target_h, target_w = compute_target_dimensions(h, w, token_budget)
        img_resized = resize(img, (target_h, target_w))
        img_scaled = 2.0 * (img_resized - 0.5)

        h_patches = target_h // 16
        w_patches = target_w // 16
        patches = extract_patches(img_scaled, patch_size=16)  # [1, num_patches, 768]
        pos_ids = compute_position_ids(h_patches, w_patches)  # [1, num_patches, 2]

        all_patches.append(patches)
        all_pos_ids.append(pos_ids)
        all_lengths.append(patches.shape[1])

    max_len = max(all_lengths)
    batch_size = len(images)

    pixel_patches = torch.zeros(batch_size, max_len, 768)
    position_ids = torch.full((batch_size, max_len, 2), -1, dtype=torch.long)
    valid_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (patches, pos_ids, length) in enumerate(
        zip(all_patches, all_pos_ids, all_lengths)
    ):
        pixel_patches[i, :length] = patches[0]
        position_ids[i, :length] = pos_ids[0]
        valid_mask[i, :length] = True

    return pixel_patches, position_ids, valid_mask
```

### Interaction with TTNN Modules

The preprocessor produces three tensors that are consumed by the TTNN vision encoder:

| Output | Shape | Consumer |
|--------|-------|----------|
| `pixel_patches` | `[batch, max_patches, 768]` | `gemma4_vision_patch_embedder.py` (linear projection) |
| `position_ids` | `[batch, max_patches, 2]` | `gemma4_vision_position_embedding.py` (2D lookup), `gemma4_vision_rope.py` (cos/sin precomputation), `gemma4_vision_pooler.py` (grid cell assignment) |
| `valid_mask` | `[batch, max_patches]` | Attention masking (exclude padding from attention), position embedding (zero out padding), pooler (exclude padding from averages) |

> **Tip:** The `position_ids` tensor is used by three different downstream modules. Compute it once on the host and reuse across all consumers to avoid redundant computation.

### Validation Checklist

- [ ] For a 1920x1080 image at 280-token budget: verify target dimensions are divisible by 48 and produce approximately 280 output tokens after pooling.
- [ ] For a 640x480 image at 280-token budget: verify aspect ratio is approximately preserved.
- [ ] Verify position IDs form a valid 2D grid: x-coordinates range from 0 to `width_patches-1`, y-coordinates from 0 to `height_patches-1`.
- [ ] Verify batch padding: shorter sequences are zero-padded with position IDs of -1.
- [ ] Compare the end-to-end preprocessed output against HuggingFace's `Gemma4ImageProcessor` for multiple test images.

> **Risk (Medium):** Variable input shapes across images in a batch may force different TTNN compiled programs. Mitigation: for batch inference, group images by the nearest standard token budget so all images in a batch have the same patch count. This enables a single compiled program per batch.

## Vision Pooler Module

**File:** `gemma4_vision_pooler.py`
**Effort:** 2-3 days
**Dependencies:** Position IDs (from `gemma4_variable_resolution.py`), RMSNorm

### Purpose

Implement the adaptive 2D pooling operation that reduces the patch sequence from the encoder output down to the configured token budget. The pooler groups patches into non-overlapping grid cells of size `pooling_kernel_size x pooling_kernel_size` (3x3 for Gemma 4) and averages the patch embeddings within each cell to produce a single output token per cell.

This module is called by `gemma4_multimodal_embedder.py` and is referenced in [modification_required_modules.md](./modification_required_modules.md#multi_modal_projectorpy) as the component that replaces Gemma 3's fixed average pooling.

### How It Works

1. **Grid cell assignment:** Using the 2D position IDs, assign each patch to a grid cell based on `floor(x / pooling_kernel_size)` and `floor(y / pooling_kernel_size)`.
2. **Per-cell averaging:** For each grid cell, compute the mean of all patch embeddings assigned to that cell.
3. **Padding awareness:** Exclude padding patches (position_ids == -1) from the averaging computation.
4. **Output ordering:** Flatten the 2D grid of pooled tokens into a 1D sequence in raster order (left-to-right, top-to-bottom).

### Implementation Sketch

```python
class TtGemma4VisionPooler:
    """Adaptive 2D pooling for Gemma 4 vision encoder."""

    def __init__(self, pooling_kernel_size=3):
        self.pooling_kernel_size = pooling_kernel_size

    def __call__(self, hidden_states, position_ids, valid_mask):
        """
        Pool encoder output patches into a reduced token sequence.

        Args:
            hidden_states: [batch, num_patches, 1152] encoder output
            position_ids:  [batch, num_patches, 2] (x, y) grid coordinates
            valid_mask:    [batch, num_patches] True for real patches

        Returns:
            pooled: [batch, num_output_tokens, 1152]
        """
        k = self.pooling_kernel_size

        # Compute grid cell indices for each patch
        cell_x = position_ids[:, :, 0] // k  # [batch, num_patches]
        cell_y = position_ids[:, :, 1] // k

        # Determine output grid dimensions
        max_cell_x = cell_x[valid_mask].max() + 1
        max_cell_y = cell_y[valid_mask].max() + 1
        num_output_tokens = max_cell_x * max_cell_y

        # Scatter-add patches into cells and divide by count
        # (Implementation uses TTNN gather/scatter or host-side index computation)
        # ...

        return pooled  # [batch, num_output_tokens, 1152]
```

### TTNN Implementation Considerations

The pooling operation involves irregular gather-scatter patterns (each grid cell collects a variable number of patches depending on the image dimensions and padding). Two approaches:

1. **Host-precomputed index maps:** Compute the cell assignment indices on the CPU and pass an index tensor to the device. Use `ttnn.embedding` or gather operations to collect patches per cell, then average. This is the recommended approach for initial bringup.

2. **On-device scatter-add:** Use `ttnn.scatter` (if available) to accumulate patches into cells directly on device. This avoids host-device synchronization but requires scatter support in TTNN.

### Validation Checklist

- [ ] For a 42x60 patch grid with `pooling_kernel_size=3`: verify output has 14x20 = 280 tokens.
- [ ] Verify that each output token is the mean of the 9 patches in its 3x3 cell.
- [ ] Verify correct handling of edge cells that may have fewer than 9 patches (when grid dimensions are not divisible by 3).
- [ ] Verify padding patches are excluded from cell averages.
- [ ] Compare output against HuggingFace `Gemma4VisionPooler` for real encoder outputs. PCC > 0.999.

> **Risk (Medium-High):** The irregular gather-scatter pattern may not map efficiently to TTNN's regular tiling model. If performance is poor, consider pre-sorting patches by cell assignment so that each cell's patches are contiguous in memory, enabling regular slice-and-reduce operations.

## New Module Summary

| Module | Purpose | Effort | Risk |
|--------|---------|--------|------|
| `gemma4_vision_rope.py` | 2D factored RoPE: precompute cos/sin, apply rotation | 2-3 days | Medium (numerical accuracy in BF16) |
| `gemma4_vision_position_embedding.py` | 2D learned position embedding lookup and sum | 1-2 days | Low (standard embedding + add) |
| `gemma4_variable_resolution.py` | Host-side preprocessing: resize, patch, position IDs | 1-2 days | Low (host-only, no TTNN ops) |
| `gemma4_vision_pooler.py` | Adaptive 2D pooling: grid cell averaging | 2-3 days | Medium-High (irregular gather-scatter) |
| **Total** | | **6-10 days** | |

The 2D RoPE module is the highest-risk new component because it must produce numerically accurate results in BF16 for the rotation to preserve attention quality across all 27 encoder layers. Validate it thoroughly against the HuggingFace float32 reference before integrating into the attention module.

The vision pooler is medium-high risk because the irregular gather-scatter pattern required for grid cell averaging may not map efficiently to TTNN's regular tiling model. Prioritize getting the host-precomputed index map approach working first.

The position embedding and variable-resolution preprocessor are lower risk because they use standard operations (embedding lookup, reshape, arithmetic) with no transcendental functions or precision-sensitive computations.

---

**Next:** [Chapter 7 — Implementation Roadmap](../ch07_implementation_roadmap/index.md) — Phased implementation plan, milestones, and risk register for the complete Gemma 4 vision encoder port.
