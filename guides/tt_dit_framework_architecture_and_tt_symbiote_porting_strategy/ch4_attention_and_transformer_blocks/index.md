# Chapter 4: Joint Attention and Transformer Blocks

## Prerequisites

- [Chapter 1 -- Module and Parameter](../ch1_architecture_overview/module_and_parameter.md): understanding of `Module`, `Parameter`, `UnregisteredModule`, and `_prepare_torch_state`.
- [Chapter 2 -- Parallel Linear Layers](../ch2_parallelism_and_ccl/parallel_linear_layers.md): understanding of `ColParallelLinear`, `RowParallelLinear`, and tensor-parallel sharding.
- [Chapter 2 -- CCL Manager](../ch2_parallelism_and_ccl/ccl_manager.md): understanding of `CCLManager`, all-gather, and persistent buffers.
- [Chapter 3 -- Normalization Layers](../ch3_custom_layers_and_ops/normalization_layers.md): understanding of `RMSNorm`, `DistributedLayerNorm`, and the two-phase distributed norm pattern.

---

## Introduction

Attention is the computational core of every diffusion transformer. Unlike standard LLM attention -- where a single query attends to a single key/value sequence with causal masking -- Diffusion Transformers (DiTs) perform **joint attention** over two distinct sequences: a spatial sequence (image or video patch tokens) and a prompt sequence (text conditioning tokens). These sequences have separate QKV projections, separate RMSNorm per head, and separate RoPE embeddings, but they are concatenated along the sequence dimension for a single fused Scaled Dot-Product Attention (SDPA) call.

This chapter covers the two central building blocks in `tt_dit/blocks/`:

1. **`Attention`** (`blocks/attention.py`): the joint attention mechanism that fuses spatial and prompt sequences into a single SDPA call, with support for tensor parallelism and sequence parallelism.
2. **`TransformerBlock`** (`blocks/transformer_block.py`): the full transformer layer that wraps `Attention` with adaptive layer normalization (time-conditioned modulation), gated residual connections, and a feedforward network.

The chapter concludes with a comparison against TT-Symbiote's attention modules, identifying the architectural differences that make porting non-trivial.

## How DiT Attention Differs from LLM Attention

| Property | LLM Attention (TT-Symbiote) | DiT Joint Attention (TT-DiT) |
|---|---|---|
| **Sequences** | Single input sequence | Two sequences: spatial + prompt |
| **QKV projections** | One set of Q/K/V linear layers | Two sets: `to_qkv` (spatial) + `add_qkv_proj` (prompt) |
| **Head normalization** | Optional per-head Q/K RMSNorm (`TTNNGR00TSelfAttention`) | Per-head RMSNorm on Q and K separately |
| **Positional encoding** | RoPE applied to Q/K jointly | RoPE applied to spatial and prompt Q/K separately |
| **SDPA** | Standard causal or non-causal SDPA | Joint SDPA: concatenates spatial+prompt K/V, returns separate spatial+prompt outputs |
| **KV cache** | Paged KV cache for autoregressive decoding | No KV cache (full recomputation each denoising step) |
| **Output projections** | Single `o_proj` | Two: `to_out` (spatial) + `to_add_out` (prompt) |
| **Modulation** | None | Adaptive LayerNorm with time-conditioned shift/scale/gate (handled at block level) |
| **Parallelism** | GQA with optional TP | TP on head dimension + optional SP via ring attention |

## Chapter Files

1. [`joint_attention.md`](./joint_attention.md) -- Walkthrough of the `Attention` class in `blocks/attention.py`: fused QKV projection, per-head RMSNorm, head padding, weight preparation via `_prepare_torch_state` and `_reshape_and_merge_qkv`, RoPE application, the two SDPA execution paths (ring-joint vs. standard joint), post-attention projections, `UnregisteredModule` for weight sharing, and `context_head_factors`.

2. [`transformer_block.md`](./transformer_block.md) -- Walkthrough of the `TransformerBlock` class in `blocks/transformer_block.py`: adaptive layer normalization with time-conditioned modulation (shift, scale, gate), the attention sub-block with gating, the feedforward sub-block with gating, the dual-stream architecture (spatial + prompt pathways), and the `context_pre_only` variant.

3. [`comparison_with_symbiote_attention.md`](./comparison_with_symbiote_attention.md) -- TT-Symbiote's attention hierarchy (`TTNNSDPAAttention`, `TTNNSelfAttention`, `LlamaAttention`, `TTNNGR00TSelfAttention`) versus TT-DiT's `Attention`. Key differences: no joint attention, no per-head QKV norm, different SDPA ops, no adaptive modulation, paged KV cache vs. no KV cache, and implications for porting.

## Architectural Summary

The following diagram shows how a single `TransformerBlock` processes spatial and prompt streams:

```
                    spatial                      prompt                    time_embed
                      |                            |                          |
                      v                            v                         silu
              DistributedLayerNorm         DistributedLayerNorm               |
              (shift, scale from time)     (shift, scale from time)    norm1_linear
                      |                            |                  norm1_context_linear
                      v                            v                          |
                 all_gather(TP)              all_gather(TP)            6 chunks each
                      |                            |
                      +----------+   +-------------+
                                 |   |
                                 v   v
                        Attention (joint SDPA)
                           |            |
                      spatial_attn  prompt_attn
                           |            |
                      * gate_attn   * gate_attn
                           |            |
                     + residual    + residual
                           |            |
              DistributedLayerNorm  DistributedLayerNorm
              (shift, scale from time) (shift, scale from time)
                           |            |
                      all_gather    all_gather
                           |            |
                   ParallelFeedForward  ParallelFeedForward
                           |            |
                      * gate_ff    * gate_ff
                           |            |
                     + residual    + residual
                           |            |
                           v            v
                        spatial       prompt
```

The `Attention` block itself is structured as:

```
  spatial ---> to_qkv ---> split_heads ---> norm_q, norm_k ---> apply_rope(spatial_rope)
                                                                       |
  prompt  ---> add_qkv_proj ---> split_heads ---> norm_added_q, norm_added_k ---> apply_rope(prompt_rope)
                                                                                       |
                               +------ optional context_head_factors on add_q ------+  |
                               |                                                       |
                               v                                                       v
                          joint_scaled_dot_product_attention(q, k, v, add_q, add_k, add_v)
                                    |                              |
                              spatial_out                     prompt_out
                                    |                              |
                             concatenate_heads              concatenate_heads
                                    |                              |
                              all_gather(TP)                all_gather(TP)
                                    |                              |
                               to_out                        to_add_out
```

## Key Takeaways

1. **Joint attention is the defining mechanism** of DiT models. The spatial and prompt sequences share a single SDPA computation, which means the prompt tokens attend to all spatial tokens and vice versa. This is fundamentally different from LLM self-attention or cross-attention.

2. **Per-head RMSNorm on Q and K** is applied after the QKV split but before RoPE. TT-Symbiote's `TTNNGR00TSelfAttention` provides an optional per-head Q/K RMSNorm capability, but it is not enabled in all model families. This normalization is essential for training stability in DiT models.

3. **Two SDPA execution paths** exist: `ttnn.transformer.joint_scaled_dot_product_attention` for single-device or TP-only configurations, and `ttnn.transformer.ring_joint_scaled_dot_product_attention` for sequence parallelism across devices.

4. **Adaptive layer normalization** in `TransformerBlock` modulates the hidden states using time-step-conditioned shift, scale, and gate parameters, giving the model a way to condition its behavior on the diffusion time step.

5. **No KV cache** is needed because DiT models process the full spatial+prompt sequence at every denoising step (there is no autoregressive generation). This is a fundamental simplification compared to LLM attention.

---

**Next:** [`joint_attention.md`](./joint_attention.md)
