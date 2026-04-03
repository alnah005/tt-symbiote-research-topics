# Configuration Comparison: Gemma 3 SigLIP vs. Gemma 4 Vision Encoder

This file provides a side-by-side comparison of every configuration parameter that affects the TTNN port. Parameters are grouped into three categories: shared (identical values), changed (same concept, different value), and new (only present in Gemma 4).

## Side-by-Side Config Table

### Shared Parameters

These parameters are identical between the two encoders. TTNN modules that depend only on these values can be reused with minimal or no changes.

| Parameter | **Gemma 3 (SigLIP)** | **Gemma 4** | TTNN Impact |
|-----------|----------------------|-------------|-------------|
| `hidden_size` | 1152 | 1152 | All linear layer shapes, tensor sharding configs carry over |
| `num_hidden_layers` | 27 | 27 | Layer loop count, weight indexing unchanged |
| `num_attention_heads` | 16 | 16 | Head partitioning in attention kernel unchanged |
| `intermediate_size` | 4304 | 4304 | MLP gate/up/down projection shapes unchanged |
| `hidden_activation` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` | Same `ttnn.gelu` with `approximate="tanh"` |
| Head dimension (derived) | 72 (1152/16) | 72 (explicit `head_dim=72`) | Attention head slicing unchanged |
| `num_channels` | 3 | 3 | RGB input, patch flattening dimension unchanged |

> **Tip:** The fact that `hidden_size`, `intermediate_size`, and head count are all identical means the dominant compute kernels (attention matmuls at 1152x1152 and MLP matmuls at 1152x4304) have the same shapes. Any TTNN sharding or tiling strategies optimized for Gemma 3 will apply directly to Gemma 4.

### Changed Parameters

These parameters exist in both models but have different values. Each change requires a targeted modification in the corresponding TTNN module.

| Parameter | **Gemma 3 (SigLIP)** | **Gemma 4** | TTNN Impact |
|-----------|----------------------|-------------|-------------|
| `patch_size` | 14 | 16 | Patch embedding kernel/weights change; see [Impact of patch_size Change](#impact-of-patch_size-change) |
| `image_size` / input handling | Fixed 896x896 | Variable aspect ratio (divisible by 48) | No fixed sequence length; program cache implications |
| Patch embedding type | `Conv2d(3, 1152, kernel_size=14, stride=14)` | `nn.Linear(768, 1152)` with flatten | Replace `gemma_conv2d_patch.py` with `ttnn.linear` |
| Number of output tokens | Fixed 256 | Configurable: 70, 140, 280, 560, 1120 | Pooling logic rewrite; variable output shapes |
| Normalization type | `nn.LayerNorm(eps=1e-6)` | `Gemma4RMSNorm(eps=1e-6)` | Replace LayerNorm ops with RMSNorm ops |
| Normalization pattern | Pre-norm only (norm before attention/MLP) | Sandwich norm (pre-norm + post-norm) | Two additional RMSNorm calls per layer |
| Attention scaling | $1/\sqrt{72} \approx 0.118$ | `1.0` (QK-norms replace scaling) | Change scaling constant in attention kernel |
| Attention bias | Yes (in SigLIP) | `False` | Remove bias from Q/K/V/O linear ops |
| Position embedding count | 4096 (1D, fixed) | 10240 per axis (2D, variable) | Complete positional encoding rewrite |
| Multimodal projection | Linear with optional LayerNorm | RMSNorm (no learnable scale) + Linear | Minor change to projection module |

### New in Gemma 4

These parameters and features have no equivalent in Gemma 3 and require new TTNN implementations.

| Parameter / Feature | **Gemma 4 Value** | TTNN Impact |
|---------------------|-------------------|-------------|
| `num_key_value_heads` | 16 (explicit; MHA, not GQA) | No functional change vs. Gemma 3 (also MHA), but the config explicitly declares it |
| `head_dim` | 72 (explicit) | Same derived value as Gemma 3, now explicit in config |
| `pooling_kernel_size` | 3 | New adaptive pooling module needed |
| `position_embedding_size` | 10240 | New 2D learned embedding table: `[2, 10240, 1152]` |
| `rope_theta` | 100.0 | New 2D RoPE module with non-standard base frequency |
| `rope_type` | `"default"` | Standard inverse-frequency computation, but 2D factored |
| `default_output_length` | 280 | Configurable token budget system |
| `standardize` | `True` | Post-pooling standardization with learned bias/scale |
| `use_clipped_linears` | `False` (31B) | `Gemma4ClippableLinear` wraps `nn.Linear`; no-op for 31B but code must handle the wrapper |
| `attention_dropout` | 0.0 | Explicitly disabled; no impact |
| Q/K/V per-head RMSNorm | Q-norm, K-norm (learnable); V-norm (no learnable scale) | New per-head normalization before RoPE application |

## Impact of patch_size Change

The change from `patch_size=14` to `patch_size=16` has cascading effects through the pipeline.

### Patch Embedding Weights

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Embedding mechanism | `Conv2d(3, 1152, kernel_size=14, stride=14)` | `nn.Linear(768, 1152)` (flatten first) |
| Input dimension per patch | $3 \times 14 \times 14 = 588$ | $3 \times 16 \times 16 = 768$ |
| Weight tensor shape | `[1152, 3, 14, 14]` | `[1152, 768]` |
| Weight count | 677,376 | 884,736 |
| Bias | None | None |

The weight shapes are incompatible, so Gemma 3 patch embedding weights cannot be transferred to Gemma 4. The TTNN module must be rewritten.

### Sequence Length Effects

For a given image resolution, `patch_size=16` produces fewer patches than `patch_size=14`:

| Image Size | **Gemma 3** (patch 14) Patches | **Gemma 4** (patch 16) Patches | Reduction |
|-----------|-------------------------------|-------------------------------|-----------|
| 896 x 896 | $(896/14)^2 = 4096$ | $(896/16)^2 = 3136$ | -23.4% |
| 672 x 672 | $(672/14)^2 = 2304$ | $(672/16)^2 = 1764$ | -23.4% |
| 480 x 480 | $(480/14)^2 \approx 1176^*$ | $(480/16)^2 = 900$ | -23.5% |

$^*$ 480 is not divisible by 14; Gemma 3 would not use this resolution.

Fewer patches means:
- **Shorter attention sequence lengths**, reducing the quadratic attention cost
- **Smaller activation memory** per image in the encoder
- **Faster per-layer computation** for the same image size

However, Gemma 4 compensates by supporting larger total pixel counts at higher token budgets (up to ~2.58M pixels for the 1120-token budget), so the maximum sequence length can still be substantial.

### Divisibility Constraint

Gemma 4 requires both image dimensions to be divisible by $\text{patch\_size} \times \text{pooling\_kernel\_size} = 16 \times 3 = 48$. This is more restrictive than Gemma 3's requirement of divisibility by 14.

| Property | **Gemma 3 (SigLIP)** | **Gemma 4** |
|----------|----------------------|-------------|
| Minimum image divisor | 14 | 48 |
| Smallest valid square | 14 x 14 (1 patch) | 48 x 48 (9 patches, pools to 1 token) |

> **Warning:** The divisibility-by-48 constraint means the TTNN image preprocessor must pad or resize images to satisfy this condition. This is a preprocessing step that runs on the host before data transfer to device.

### TTNN Tile Alignment

Tenstorrent hardware operates on 32x32 tiles. The patch input dimension changes from 588 to 768:

| Dimension | Gemma 3 (588) | Gemma 4 (768) |
|-----------|---------------|---------------|
| Tile-aligned? | No (588 / 32 = 18.375) | Yes (768 / 32 = 24) |
| Padding needed | 20 elements to reach 608 | None |

> **Tip:** The Gemma 4 patch input dimension of 768 is perfectly tile-aligned (768 = 24 * 32), which eliminates the padding overhead that Gemma 3's 588-dimensional input incurs. This is a small but meaningful efficiency win for the patch embedding linear layer on TTNN.

## Summary

The configuration comparison reveals a model that preserves the core compute dimensions (hidden size, head count, MLP width, layer count) while substantially reworking the input processing, positional encoding, normalization, and output pooling. For TTNN porting:

1. **Matmul shapes are identical** — the dominant compute kernels transfer directly.
2. **Patch embedding is incompatible** — different mechanism (linear vs. Conv2d) and different input dimension (768 vs. 588).
3. **Positional encoding is completely new** — the most significant implementation effort; see [`positional_encoding_shift.md`](./positional_encoding_shift.md).
4. **Normalization changes are mechanical** — replacing LayerNorm with RMSNorm and adding sandwich norm is straightforward.
5. **Pooling requires a rewrite** — adaptive 2D pooling replaces fixed average pooling.

---

**Next:** [`module_mapping.md`](./module_mapping.md) — File-by-file mapping of existing Gemma 3 TTNN modules to Gemma 4 equivalents.
