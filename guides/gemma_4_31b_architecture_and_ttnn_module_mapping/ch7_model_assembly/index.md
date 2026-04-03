# Chapter 7 --- Decoder Layer and Full Model Assembly

## Overview

This chapter assembles every component from the preceding chapters into the
complete `TTNNGemma4Model` module hierarchy. Where Chapters 2--6 designed
individual submodules (projections, norms, RoPE, attention, sharding), this
chapter shows how those submodules compose into the decoder layer, how the
decoder layer repeats 60 times in a heterogeneous loop, and how the top-level
model module orchestrates embedding, decoding, and logit generation.

After reading this chapter you will know:

- The exact structure of `TTNNGemma4DecoderLayer`, including PLE injection,
  pre-attention norm, attention dispatch, residual connections, pre-FFN norm,
  GeGLU FFN, and the second residual add.
- How `TTNNGemma4FFN` implements the GeGLU feed-forward network with fused or
  separate gate/up projections.
- How `TTNNGemma4PLE` handles per-layer embeddings (and why it is a no-op in
  the 31B config).
- The top-level `TTNNGemma4Model` structure: token embedding, the 60-layer
  decode loop, final norm, tied LM head, and logit softcapping.
- Weight loading, KV cache initialization, and decode loop orchestration.

## Module Hierarchy

The following diagram shows the complete module tree for Gemma 4 31B. Modules
marked with `x50` or `x10` indicate the count per layer type. Modules marked
`(disabled)` are present in the architecture but inactive in the 31B config.

```text
TTNNGemma4Model
 |
 +-- embed_tokens                      TTNNEmbedding [262144, 5376]
 |
 +-- per_layer_model_projection        (disabled -- PLE off in 31B)
 +-- embed_tokens_per_layer            (disabled -- PLE off in 31B)
 |
 +-- layers[0..59]                     TTNNGemma4DecoderLayer x60
 |    |
 |    +-- ple_injection                TTNNGemma4PLE (no-op in 31B)
 |    |
 |    +-- input_layernorm              TTNNDistributedRMSNorm [5376]
 |    |
 |    +-- self_attn                    TTNNGemma4SlidingAttention  x50
 |    |                           or   TTNNGemma4GlobalAttention   x10
 |    |    |
 |    |    +-- q_proj                  TTNNLinearIReplicatedWColSharded
 |    |    +-- k_proj                  TTNNLinearIReplicatedWColSharded (sliding)
 |    |    |                      or   TTNNLinear (replicated, global)
 |    |    +-- v_proj                  TTNNLinearIReplicatedWColSharded (sliding only)
 |    |    +-- o_proj                  TTNNLinearIColShardedWRowSharded
 |    |    +-- q_norm                  TTNNDistributedRMSNorm [head_dim]
 |    |    +-- k_norm                  TTNNDistributedRMSNorm [head_dim]
 |    |    +-- v_norm                  TTNNDistributedRMSNorm [head_dim] (with_scale=False)
 |    |    +-- rope                    TTNNDistributedRotaryPositionEmbedding (sliding)
 |    |                           or   TTNNRotaryPositionEmbedding (global, partial)
 |    |
 |    +-- post_attention_layernorm     TTNNDistributedRMSNorm [5376]
 |    |
 |    +-- mlp                          TTNNGemma4FFN
 |    |    |
 |    |    +-- gate_proj               TTNNLinearIReplicatedWColSharded [5376, 21504]
 |    |    +-- up_proj                 TTNNLinearIReplicatedWColSharded [5376, 21504]
 |    |    +-- down_proj               TTNNLinearIColShardedWRowSharded [21504, 5376]
 |    |
 |    +-- post_feedforward_layernorm   TTNNDistributedRMSNorm [5376]
 |    +-- layer_scalar                 Buffer (1.0 in 31B, no-op)
 |
 +-- norm                             TTNNDistributedRMSNorm [5376]
 |
 +-- lm_head                          Tied to embed_tokens (no separate weight)
```

## Reading Order

1. [`decoder_layer_module.md`](./decoder_layer_module.md) --- The
   `TTNNGemma4DecoderLayer` structure: PLE injection, norms, attention dispatch,
   FFN, and residual connections.
2. [`ffn_module.md`](./ffn_module.md) --- The `TTNNGemma4FFN` module: GeGLU
   implementation, fused vs separate projections, and program config
   recommendations.
3. [`ple_module.md`](./ple_module.md) --- The `TTNNGemma4PLE` module: per-layer
   embeddings, why PLE is disabled in 31B, and multimodal pad-token handling.
4. [`full_model_module.md`](./full_model_module.md) --- The top-level
   `TTNNGemma4Model`: embedding, decode loop, final norm, LM head, logit
   softcapping, weight loading, and KV cache initialization.

## Prerequisites

This chapter builds on all prior chapters:

- [Chapter 1 --- Architecture Overview](../ch1_architecture_overview/index.md):
  layer organization, the 5:1 pattern, config parameters.
- [Chapter 2 --- Projection Shapes](../ch2_projection_shapes/index.md): weight
  and activation tensor shapes for all projections.
- [Chapter 3 --- K=V Sharing and V-Norm](../ch3_kv_sharing_and_vnorm/index.md):
  the K=V mechanism and unscaled RMSNorm for V.
- [Chapter 4 --- Dual RoPE](../ch4_dual_rope/index.md): sliding vs global RoPE
  configurations and precomputed cos/sin tables.
- [Chapter 5 --- Attention Module Design](../ch5_attention_module_design/index.md):
  the base-class-with-subclasses design and complete forward passes for both
  attention types.
- [Chapter 6 --- TP Sharding](../ch6_tp_sharding/index.md): weight sharding,
  replicated global KV heads, KV cache memory budget.

## Key Constants

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 5376 |
| `intermediate_size` | 21504 |
| `num_hidden_layers` | 60 |
| `num_attention_heads` (Q) | 32 |
| Sliding layers | 50 (indices not in {5,11,17,23,29,35,41,47,53,59}) |
| Global layers | 10 (indices {5,11,17,23,29,35,41,47,53,59}) |
| `hidden_size_per_layer_input` | 0 (PLE disabled) |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | true |
| `rms_norm_eps` | 1e-6 |
| TP degree | 8 |
| `hidden_activation` | `gelu_pytorch_tanh` |

---

**Next:** [`decoder_layer_module.md`](./decoder_layer_module.md)
