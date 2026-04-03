# Chapter 4 --- Dual RoPE and Partial Rotary Embedding

## Overview

Gemma 4 31B uses two distinct Rotary Position Embedding (RoPE) configurations
--- one for each attention layer type --- making it one of the few production
models that requires two separate sets of cos/sin embedding tables at inference
time. This chapter covers the mathematical formulation, the per-layer-type
parameterization, and the TTNN mapping for both variants.

**Sliding layers** use standard RoPE with $\theta = 10{,}000$ and full rotation
across all 256 head dimensions. This is the familiar RoPE formulation used by
LLaMA, Mistral, and prior Gemma generations.

**Global layers** use proportional RoPE (p-RoPE) with $\theta = 1{,}000{,}000$
and `partial_rotary_factor=0.25`, meaning only the first 128 of 512 head
dimensions receive rotary encoding. The remaining 384 dimensions pass through
unchanged as pure semantic channels. This combination of high theta and partial
rotation is designed for robust long-context extrapolation to 256K tokens.

Both RoPE variants are applied to Q and K only. V vectors never receive
positional encoding in any layer type (see
[Chapter 3 --- K=V Sharing and V-Norm](../ch3_kv_sharing_and_vnorm/index.md)
for details on the V path).

## Quick Reference: Sliding RoPE vs Global p-RoPE

| Parameter | Sliding RoPE | Global p-RoPE |
|-----------|-------------|---------------|
| `rope_type` | `"default"` | `"proportional"` |
| $\theta$ (`rope_theta`) | 10,000 | 1,000,000 |
| `partial_rotary_factor` | 1.0 (implicit) | 0.25 |
| `head_dim` | 256 | 512 |
| Rotary dimensions | 256 (all) | 128 (first 25%) |
| Non-rotary dimensions | 0 | 384 (last 75%) |
| `inv_freq` length | 128 | 64 |
| Cos/sin table shape | `[max_seq_len, 256]` | `[max_seq_len, 128]` |
| Applied to | Q, K | Q, K |
| Not applied to | V | V |
| Layer count | 50 | 10 |
| TTNN module | `TTNNRotaryPositionEmbedding` or `TTNNDistributedRotaryPositionEmbedding` | `TTNNRotaryPositionEmbedding` (non-distributed; see text) |

### Note on cos/sin table shape for global p-RoPE

The reference HuggingFace implementation computes narrow `inv_freq` of length
64 and produces cos/sin tables of shape `[max_seq_len, 128]` (covering only
the rotary dimensions). The `apply_rotary_pos_emb` function handles partial
rotation via a split-apply-concat pattern: it splits the head tensor at the
rotary dimension boundary, applies RoPE to the first 128 dimensions, and
concatenates with the unchanged remainder. See
[`global_proportional_rope.md`](./global_proportional_rope.md) for the
detailed mechanics. A TTNN optimization (Strategy A) can alternatively use
full-width `[max_seq_len, 512]` tables with identity values in the
non-rotated columns, allowing a single RoPE kernel call on the entire head
tensor without splitting.

## Reading Order

1. [`sliding_rope.md`](./sliding_rope.md) --- Standard RoPE for sliding layers:
   theta=10000, full rotation, and direct TTNN mapping.
2. [`global_proportional_rope.md`](./global_proportional_rope.md) --- Proportional
   RoPE for global layers: theta=1M, partial rotation, the split-apply-concat
   pattern, and TTNN compatibility considerations.
3. [`rope_precomputation.md`](./rope_precomputation.md) --- Precomputation and
   storage of both cos/sin table sets, memory footprint analysis, and the
   per-step slicing strategy.

## Prerequisites

This chapter builds on:

- [Chapter 1 --- Architecture Overview](../ch1_architecture_overview/index.md):
  the RoPE parameters per layer type and the sliding/global layer distinction.
- [Chapter 2 --- Projection Weights and Tensor Shapes](../ch2_projection_shapes/index.md):
  Q and K activation shapes that determine the dimensionality of RoPE
  application.
- [Chapter 3 --- K=V Sharing and V-Norm](../ch3_kv_sharing_and_vnorm/index.md):
  the K path (which receives RoPE) vs the V path (which does not), and how
  K=V sharing interacts with partial RoPE in global layers.

---

**Next:** [Chapter 5 --- Heterogeneous Attention Module Design](../ch5_attention_module_design/index.md)
