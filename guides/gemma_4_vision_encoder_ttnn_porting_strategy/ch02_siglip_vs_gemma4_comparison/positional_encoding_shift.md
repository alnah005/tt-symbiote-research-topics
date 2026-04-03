# Positional Encoding Shift: From Absolute to 2D RoPE

The most consequential architectural change between the Gemma 3 SigLIP encoder and the Gemma 4 vision encoder is the positional encoding system. This file explains both systems, why the change was made, and what it means for the TTNN port.

## Gemma 3: Learned Absolute Position Embeddings

### How It Works

The Gemma 3 SigLIP encoder uses a single 1D learned embedding table to encode position information.

| Property | Value |
|----------|-------|
| Embedding type | `nn.Embedding(4096, 1152)` |
| Position range | 0 to 4095 (sequential) |
| Application point | Added once to patch embeddings after the Conv2d projection |
| Spatial awareness | None — positions are assigned in raster-scan order (left-to-right, top-to-bottom) |
| Resolution support | Fixed 896x896 only ($896/14 = 64$ patches per side, $64 \times 64 = 4096$ total) |

### Forward Pass

```python
# Gemma 3 SigLIP patch embedding (simplified)
patches = conv2d_patch(pixel_values)          # [batch, 4096, 1152]
position_ids = torch.arange(4096)             # [4096]
position_embeddings = embedding_table(position_ids)  # [4096, 1152]
output = patches + position_embeddings        # [batch, 4096, 1152]
```

### Limitations

1. **Fixed resolution only.** The embedding table has exactly 4096 entries, matching the $64 \times 64$ patch grid of a 896x896 image. Any other resolution would require interpolation or retraining.

2. **No explicit spatial structure.** Position 0 is the top-left patch, position 63 is the end of the first row, and position 64 is the start of the second row. The model must learn that positions 0 and 64 are vertically adjacent purely from training data — this relationship is not structurally encoded.

3. **Pan-and-scan workaround.** To handle non-square images, Gemma 3 uses a "pan and scan" algorithm that crops the image into multiple 896x896 squares, encodes each independently, and concatenates the results. This is effective but wasteful: overlapping crops process the same pixels multiple times.

## Gemma 4: Dual Positional Encoding System

Gemma 4 replaces the single 1D system with two complementary mechanisms that work together.

### Mechanism 1: 2D Learned Position Embeddings

| Property | Value |
|----------|-------|
| Embedding type | `nn.Parameter([2, 10240, 1152])` |
| Position range | 0 to 10239 per axis (x and y independently) |
| Application point | Added once to patch embeddings after the linear projection |
| Spatial awareness | Explicit — separate embeddings for x-axis and y-axis |
| Resolution support | Variable (any grid up to 10240 patches per side) |

The embedding table has two rows: row 0 stores x-axis embeddings and row 1 stores y-axis embeddings. For each patch at grid position $(x, y)$:

$$
\text{pos\_emb}(x, y) = \text{table}[0, x, :] + \text{table}[1, y, :]
$$

This factored representation means:
- A $60 \times 42$ grid (2520 patches) uses only 102 unique embedding vectors (60 + 42), not 2520
- The model explicitly knows that two patches with the same x-coordinate share a horizontal position, regardless of their y-coordinates
- New aspect ratios are supported without any interpolation — the same embedding vectors are reused in new combinations

### Mechanism 2: 2D Factored RoPE

| Property | Value |
|----------|-------|
| Type | Rotary Position Embedding, factored across 2 spatial dimensions |
| Application point | Applied to Q and K tensors in every attention layer |
| Parameters | `rope_theta=100.0`, `head_dim=72` split into 2 halves of 36 |
| Spatial awareness | Explicit — independent rotation frequencies for x and y |
| Resolution support | Unbounded (RoPE generalizes to any position) |

The head dimension (72) is split in half:
- First 36 dimensions encode the **x-axis** position
- Last 36 dimensions encode the **y-axis** position

Each half is rotated using standard RoPE with independent frequency tables computed from the grid coordinates:

$$
f_i^{(x)} = \frac{1}{\theta^{2i/36}} \cdot x, \quad f_i^{(y)} = \frac{1}{\theta^{2i/36}} \cdot y, \quad \text{for } i = 0, \ldots, 17
$$

where $\theta = 100.0$.

### Why Two Mechanisms?

The dual system serves complementary purposes:

| Mechanism | What It Encodes | Where It Acts | Learnable? |
|-----------|----------------|---------------|------------|
| 2D learned embeddings | Absolute position identity ("I am at grid cell (3, 7)") | Added to hidden states once, before the encoder stack | Yes (fully learned) |
| 2D factored RoPE | Relative position relationships ("patch A is 2 cells to the right of patch B") | Modifies Q/K dot products in every attention layer | No (computed from position IDs) |

The learned embeddings give each patch a unique identity based on its absolute grid location. RoPE encodes relative distances, enabling the attention mechanism to compute position-dependent similarity: patches that are spatially close produce larger attention scores than those far apart.

> **Tip:** This dual approach is analogous to how some language models combine absolute position embeddings (added to token embeddings) with RoPE (applied in attention). The key difference is that the Gemma 4 vision encoder operates in 2D rather than 1D.

## Side-by-Side Comparison

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Embedding type | 1D learned absolute | 2D learned (factored x, y) + 2D factored RoPE |
| Table size | `[4096, 1152]` = 4.7M params | `[2, 10240, 1152]` = 23.6M params |
| Table memory (BF16) | 9.4 MB | 47.2 MB |
| Position indexing | Sequential integer (raster scan) | 2D grid coordinates `(x, y)` |
| Resolution flexibility | Fixed 896x896 only | Any resolution divisible by 48 |
| Spatial inductive bias | None (must be learned) | Explicit (factored axes + rotary relative encoding) |
| RoPE in attention | No | Yes, every layer (`rope_theta=100.0`) |
| Q/K modification | None | Per-head RMSNorm + 2D RoPE rotation |
| Parameter overhead | 4.7M | 23.6M (5x larger embedding table, but still <5% of encoder) |

## Why rope_theta = 100.0?

The base frequency $\theta$ controls the wavelength spectrum of the rotary embeddings:

$$
\text{wavelength}_i = 2\pi \cdot \theta^{2i/d_{\text{spatial}}}
$$

| Context | $\theta$ | Max position | Reasoning |
|---------|----------|-------------|-----------|
| Language model (text) | 10,000 - 1,000,000 | Up to 128K+ tokens | Long-range dependencies across many thousands of tokens |
| Vision encoder (Gemma 4) | 100.0 | Typically < 100 patches per axis | Short-range spatial positions; a 60x42 grid has max coordinate ~60 |

With $\theta = 100.0$ and $d_{\text{spatial}} = 36$:
- The shortest wavelength (highest frequency, $i=0$): $2\pi \approx 6.3$ positions
- The longest wavelength (lowest frequency, $i=17$): $2\pi \cdot 100^{34/36} \approx 487$ positions

This range comfortably covers the spatial extent of any supported image grid (max ~100 patches per axis at the 1120-token budget).

> **Warning:** Using the language model's `rope_theta` (10000.0 or higher) for the vision encoder would produce wavelengths far longer than the image grid, effectively collapsing the rotary embeddings to near-identity rotations. Always use the vision-specific `rope_theta=100.0`.

## Implications for TTNN

### What Transfers from Gemma 3

Very little of the Gemma 3 positional encoding implementation transfers to Gemma 4:

| Gemma 3 Component | Reusable? | Why |
|-------------------|-----------|-----|
| `nn.Embedding(4096, 1152)` lookup | No | Different table shape, different indexing scheme |
| Sequential position ID generation | No | Gemma 4 uses 2D grid coordinates, not sequential integers |
| Position embedding addition | Partially | The "add embeddings to hidden states" step is the same concept, but the embedding computation is different |

### New TTNN Requirements

#### Requirement 1: 2D Learned Position Embedding Lookup

The position embedding table `[2, 10240, 1152]` must be stored on device. For each image, the x-coordinates and y-coordinates of all patches are used to index into the table.

**TTNN approach:**
```python
# position_ids: [batch, num_patches, 2] — x and y coordinates
# table: [2, 10240, 1152] — on device

# Option A: Two ttnn.embedding lookups + add
x_emb = ttnn.embedding(position_ids[:, :, 0], table[0])  # [batch, num_patches, 1152]
y_emb = ttnn.embedding(position_ids[:, :, 1], table[1])  # [batch, num_patches, 1152]
pos_emb = ttnn.add(x_emb, y_emb)

# Option B: Match HF reference — one-hot encode, batch matmul, sum
# More complex but exactly matches reference numerics
x_onehot = one_hot(position_ids[:, :, 0], 10240)         # [batch, num_patches, 10240]
x_emb = ttnn.matmul(x_onehot, table[0])                  # [batch, num_patches, 1152]
# ... similarly for y
```

> **Tip:** Option A is simpler and likely sufficient for BF16 accuracy. Option B may be needed only if PCC validation reveals numerical divergence from the reference implementation.

**Memory consideration:** The full table is 47.2 MB in BF16. This fits comfortably in Wormhole L1 or DRAM and only needs to be loaded once per model initialization.

#### Requirement 2: 2D Factored RoPE in Every Attention Layer

This is the most significant new requirement. Every attention layer must:

1. Receive precomputed `cos` and `sin` tensors of shape `[batch, num_patches, 72]`
2. Split Q and K along the head dimension into two halves of 36
3. Apply standard RoPE rotation to each half with the corresponding cos/sin slice
4. Concatenate the rotated halves

**TTNN implementation strategies** (detailed analysis in [Chapter 3 — 2D Factored RoPE](../ch03_2d_factored_rope/index.md)):

| Strategy | Effort | Performance | Recommended For |
|----------|--------|-------------|-----------------|
| CPU precompute cos/sin, device apply | Low | Good (transfer overhead is small) | Initial bringup |
| Compose from TTNN ops (split + 1D RoPE + concat) | Medium | Good (if 1D RoPE kernel exists) | Mid-term |
| Custom fused 2D RoPE kernel | High | Best | Performance-critical deployment |

**Key constraint:** The cos/sin tables are the same for all 27 layers (computed once by `Gemma4VisionRotaryEmbedding` and passed through). They change only when the image grid dimensions change, so caching per-resolution tables is effective.

#### Requirement 3: Per-Head Q/K/V Normalization

Gemma 4 applies RMSNorm to Q, K, and V after projection but before RoPE. This normalization operates per-head (dimension 72, not 1152).

| Norm Target | Dimension | Learnable Scale | Position in Forward Pass |
|-------------|-----------|-----------------|-------------------------|
| Q | 72 | Yes | After q_proj, before RoPE |
| K | 72 | Yes | After k_proj, before RoPE |
| V | 72 | No | After v_proj |

**TTNN consideration:** The existing RMSNorm kernel operates on the last dimension. If Q/K/V are shaped as `[batch, seq, num_heads, head_dim]`, the norm applies along `head_dim=72`. This is a non-standard shape for RMSNorm (most uses are at `hidden_size=1152`). Verify that the TTNN RMSNorm kernel handles dim=72 efficiently, or consider fusing the norm with the head reshape.

#### Requirement 4: Variable-Length Position ID Sequences

Unlike Gemma 3's fixed 4096-position sequence, Gemma 4's position IDs vary per image. A batch of images may have different grid dimensions, requiring padding.

**TTNN consideration:** Variable sequence lengths affect:
- Program cache: different pad lengths may trigger recompilation
- Memory allocation: must accommodate the largest sequence in the batch
- Attention mask: must correctly mask padding positions

A practical mitigation is to pad all sequences in a batch to the same length and use the padding mask throughout the encoder. For tracing, pre-trace the five standard token budgets to avoid recompilation.

## Summary

The positional encoding shift from Gemma 3 to Gemma 4 is not an incremental change — it is a fundamental redesign:

1. **Gemma 3** uses a simple, fixed-size 1D embedding table. It is trivial to implement but inflexible.
2. **Gemma 4** uses a dual system (2D learned + 2D factored RoPE) that is more complex but enables variable-resolution processing with explicit spatial awareness.

For the TTNN port, this means:
- The position embedding module must be rewritten from scratch
- Every attention layer gains a new RoPE application step (27 layers x per-forward-pass)
- Per-head Q/K/V normalization adds ops before the RoPE step
- The RoPE cos/sin tables should be precomputed and cached per image resolution

The positional encoding changes account for the majority of the "new implementation" effort in the module mapping. Getting this right is the critical path for the Gemma 4 vision encoder port.

---

**Next:** [Chapter 3 — 2D Factored RoPE: Theory and TTNN Mapping](../ch03_2d_factored_rope/index.md) — A deep dive into the multidimensional RoPE mathematics and implementation strategies for TTNN.
