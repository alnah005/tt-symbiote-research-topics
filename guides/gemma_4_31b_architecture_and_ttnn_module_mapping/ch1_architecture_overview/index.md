# Chapter 1 --- Gemma 4 31B Architecture Overview

## Overview

This chapter provides a complete architectural reference for the Gemma 4 31B
text decoder. It covers every submodule in the 60-layer decoder stack, the
heterogeneous attention design that mixes sliding-window and global layers, and
the novel components that distinguish Gemma 4 from prior Gemma generations.

After reading this chapter you will know:

- How the 60 decoder layers are organized into sliding and global types.
- The exact configuration parameters for each attention type, including head
  counts, head dimensions, RoPE settings, and projection shapes.
- What K=V sharing, V-norm, PLE, logit softcapping, and GeGLU mean in the
  context of this model.

## Reading Order

1. [`layer_organization.md`](./layer_organization.md) --- Layer layout, the
   5:1 sliding-to-global pattern, and the anatomy of a single decoder layer.
2. [`heterogeneous_attention_configs.md`](./heterogeneous_attention_configs.md)
   --- Side-by-side comparison of the two structurally different attention
   configurations.
3. [`novel_components.md`](./novel_components.md) --- K=V sharing, V-norm,
   Per-Layer Embeddings, logit softcapping, and GeGLU.

## Quick-Reference Table: Text Decoder Config Parameters

The following table lists every parameter from the `text_config` section of
`config.json` that is relevant to the text decoder. Values come from the
official `google/gemma-4-31B` checkpoint.

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_type` | `"gemma4_text"` | Architecture identifier |
| `vocab_size` | 262144 | Shared with vision; embedding table rows |
| `hidden_size` | 5376 | Model dimension $d_{model}$ |
| `num_hidden_layers` | 60 | Total decoder layers |
| `num_attention_heads` | 32 | Query heads (all layers) |
| `num_key_value_heads` | 16 | KV heads for sliding layers |
| `num_global_key_value_heads` | 4 | KV heads for global layers |
| `head_dim` | 256 | Head dimension for sliding layers |
| `global_head_dim` | 512 | Head dimension for global layers |
| `intermediate_size` | 21504 | FFN intermediate dim (all layers) |
| `hidden_activation` | `"gelu_pytorch_tanh"` | GeGLU gate activation |
| `rms_norm_eps` | 1e-6 | Epsilon for all RMSNorm layers |
| `attention_bias` | `false` | No bias in Q/K/V/O projections |
| `attention_dropout` | 0.0 | No dropout at inference |
| `attention_k_eq_v` | `true` | K=V sharing enabled in global layers |
| `sliding_window` | 1024 | Window size for sliding-attention layers |
| `max_position_embeddings` | 262144 | 256K context length |
| `final_logit_softcapping` | 30.0 | Logit capping magnitude |
| `tie_word_embeddings` | `true` | LM head shares embedding weights |
| `num_kv_shared_layers` | 0 | No cross-layer KV sharing in 31B |
| `hidden_size_per_layer_input` | 0 | PLE disabled in 31B (dim is 0) |
| `vocab_size_per_layer_input` | 262144 | PLE vocab (unused when dim=0) |
| `enable_moe_block` | `false` | No MoE in 31B (dense model) |
| `use_double_wide_mlp` | `false` | Standard FFN width |
| `use_bidirectional_attention` | `"vision"` | Bidirectional only for vision encoder |
| `use_cache` | `true` | KV caching enabled |
| `dtype` | `"bfloat16"` | Training/inference dtype |

### RoPE Parameters

| Layer Type | `rope_type` | `rope_theta` | `partial_rotary_factor` |
|------------|-------------|--------------|-------------------------|
| Sliding | `"default"` | 10000.0 | 1.0 (implicit; all dims rotated) |
| Global | `"proportional"` | 1000000.0 | 0.25 (128 of 512 dims rotated) |

### Layer Type Schedule

The `layer_types` array in `config.json` explicitly lists the type for each of
the 60 layers. The pattern repeats in groups of 6: five `sliding_attention`
followed by one `full_attention`. See
[`layer_organization.md`](./layer_organization.md) for the complete schedule.
