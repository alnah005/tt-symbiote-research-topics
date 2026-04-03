# Chapter 2 — Gemma 3 SigLIP vs. Gemma 4 Vision Encoder Comparison

## Learning Objectives

After completing this chapter, you will be able to:

- Identify every configuration parameter that is shared, changed, or new between the Gemma 3 SigLIP encoder and the Gemma 4 vision encoder
- Quantify the impact of the `patch_size` change (14 to 16) on patch embedding weights, sequence lengths, and downstream compute
- Map every Gemma 3 TTNN vision module to its Gemma 4 equivalent and classify each as direct-reuse, modify, or new-implementation
- Explain why the shift from 1D absolute position embeddings to a dual 2D learned + 2D factored RoPE system is the most consequential architectural change for the TTNN port
- Estimate the overall code reuse percentage to inform sprint planning

## Prerequisites

- Completion of [Chapter 1 — Gemma 4 Vision Encoder Architecture Overview](../ch01_gemma4_vision_architecture/index.md)
- Familiarity with the existing Gemma 3 TTNN codebase at `models/demos/multimodal/gemma3/tt/`

## Chapter Contents

| File | Topic |
|------|-------|
| [`config_diff.md`](./config_diff.md) | Side-by-side config comparison: shared, changed, and new parameters |
| [`module_mapping.md`](./module_mapping.md) | Module-by-module mapping between Gemma 3 TTNN files and Gemma 4 equivalents |
| [`positional_encoding_shift.md`](./positional_encoding_shift.md) | Deep dive into the positional encoding paradigm shift and TTNN implications |

## Summary: What is Shared, What Differs, What is New

The following table provides a high-level classification of the architectural relationship between the two vision encoders. Each category is explored in detail in the linked files.

| Category | Items | Impact on TTNN Port |
|----------|-------|---------------------|
| **Shared** | `hidden_size=1152`, `num_hidden_layers=27`, `num_attention_heads=16`, `intermediate_size=4304`, `hidden_activation=gelu_pytorch_tanh`, `rms_norm_eps=1e-6`, gated MLP structure, pre-norm residual pattern | High reuse potential for MLP, RMSNorm, and encoder layer scaffolding |
| **Changed** | `patch_size` (14 to 16), image input handling (fixed 896x896 to variable aspect ratio), patch embedding mechanism (Conv2d to flatten+linear), normalization (LayerNorm to RMSNorm with sandwich pattern), attention scaling (1/sqrt(d) to QK-norm with scale=1.0) | Requires targeted modifications to existing modules |
| **New in Gemma 4** | `num_key_value_heads=16` (explicit), `head_dim=72` (explicit), `pooling_kernel_size=3`, `position_embedding_size=10240`, `rope_theta=100.0`, 2D learned position embeddings, 2D factored RoPE, adaptive pooling with standardization, Q/K/V RMSNorm | Requires new TTNN implementations; ~20% of codebase |

### Reuse Estimate at a Glance

| Reuse Category | Percentage | Module Count |
|---------------|------------|--------------|
| Direct reuse | ~40-50% | MLP, RMSNorm, encoder layer skeleton, model config infra, checkpoint loading infra |
| Modify | ~30% | Attention (add RoPE + QK-norm), projector (adaptive pooling), patch embedding (Conv2d to linear) |
| New implementation | ~20% | 2D RoPE module, 2D learned position embedding, variable-resolution preprocessor |

These estimates are refined in [`module_mapping.md`](./module_mapping.md) with per-file breakdowns.

## Overview

The Gemma 3 vision encoder is a SigLIP-based Vision Transformer: it takes fixed 896x896 square images, splits them into 14x14 patches, adds 1D learned absolute position embeddings, and runs 27 transformer layers with standard scaled dot-product attention. The output is condensed to 256 soft tokens via average pooling and projected to the language model dimension.

The Gemma 4 vision encoder preserves the core ViT blueprint (patch embedding, N transformer layers, projection) but rewrites nearly every detail:

1. **Input handling** changes from fixed-square to variable-aspect-ratio, with the constraint that both dimensions must be divisible by 48.
2. **Patch embedding** changes from Conv2d (kernel 14, stride 14) to flatten + linear (patch size 16).
3. **Positional encoding** changes from a single 1D learned embedding table (4096 positions) to a dual system: 2D learned embeddings (10240 positions per axis) plus 2D factored RoPE applied in every attention layer.
4. **Normalization** changes from LayerNorm to RMSNorm with a "sandwich" pattern (pre- and post- normalization around both attention and MLP).
5. **Attention** adds per-head Q/K/V RMSNorm and removes the standard $1/\sqrt{d}$ scaling factor (the QK-norms serve as the scaling mechanism).
6. **Pooling** changes from fixed average pooling (producing 256 tokens) to adaptive 2D pooling with configurable token budgets (70, 140, 280, 560, 1120) and optional output standardization.

Despite these changes, the fundamental compute profile is similar: 27 layers of attention + MLP at hidden dimension 1152 with 16 heads. This means existing TTNN sharding strategies, memory configurations, and matmul decompositions are likely to transfer with adjustments rather than full rewrites.
