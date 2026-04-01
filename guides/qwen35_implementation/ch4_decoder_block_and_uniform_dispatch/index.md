# Chapter 4 — Decoder Block and Uniform Dispatch

## Overview

This chapter explains how `DeltaNetDecoderBlock` provides a single, uniform forward
signature that lets the model's inference loop dispatch identically to every layer —
regardless of whether that layer holds a `GatedDeltaNet` or a `GatedAttention` module,
and regardless of whether the MLP is a standard dense `MLP` or a `Qwen35MoE`.

The two key design decisions are:

1. **Factory construction** — `attention_class` and `mlp_class` constructor parameters
   select the concrete sub-module at build time; the forward loop never branches on layer
   type.
2. **Silent ignore** — DeltaNet layers silently drop the RoPE, KV-cache, and position
   arguments that full-attention layers need; the forward signature is identical for both.

## Prerequisites

- Chapter 2 (`ch2_gated_deltanet_linear_attention_on_blackhole/`) for `GatedDeltaNet`
  internals, state initialization, and the recurrence equations.
- Chapter 3 (`ch3_gated_attention_full_attention_layers/`) for `GatedAttention`, partial
  RoPE, the output gate, and the `pre_wo_hook` mechanism.

Chapter 5 (`ch5_mixture_of_experts/`) covers `Qwen35MoE` in depth. You can read this
chapter without reading Chapter 5 first — `mlp_class` just accepts any callable with the
shared constructor signature.

## Reading Order

| File | Content |
|------|---------|
| [`block_structure.md`](./block_structure.md) | `attention_class`/`mlp_class` factory pattern, `DistributedNorm` wrappers, 27B vs A3B build loops |
| [`forward_signature.md`](./forward_signature.md) | Uniform forward signature, silent ignore of DeltaNet-irrelevant args, residual memory layout |
| [`mlp_dispatch.md`](./mlp_dispatch.md) | MLP vs `Qwen35MoE` substitution, state dict prefix isolation |

Read the files in the order listed above. [`block_structure.md`](./block_structure.md)
establishes what is built; [`forward_signature.md`](./forward_signature.md) explains how
it runs; [`mlp_dispatch.md`](./mlp_dispatch.md) closes the loop on the MLP side.

## Cross-References

- The L1 CB clash workaround for `hidden_dim=17408` on Blackhole is documented in
  [`forward_signature.md`](./forward_signature.md).
- The `mlp_weight_cache_path` parameter for isolating expert weight caches is covered in
  [`mlp_dispatch.md`](./mlp_dispatch.md).
- Per-layer activation dtype tuning via `decoders_optimizations` is introduced in
  [`forward_signature.md`](./forward_signature.md) and is relevant to Chapter 6 (Weight
  Precision).
