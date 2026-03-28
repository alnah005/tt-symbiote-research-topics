# Chapter 7 — Existing Implementations, Kernel Gaps, and Development Roadmap

## Overview

This chapter answers **Q8: Which TTNN primitives exist for Gated Delta Net and Gated Attention, and what new kernels are needed?** It also synthesizes the gap analysis from Chapter 4 into a prioritized development roadmap.

The key distinction this chapter maintains is between two kinds of gaps:

- **Gap — requires wiring:** The operation is expressible with existing TTNN primitives but is not yet connected in the model code. No new kernel development is needed; the gap closes with Python-level changes.
- **Gap — requires custom kernel:** The operation has no adequate TTNN equivalent. Closing the gap requires developing a new TT-Metalium kernel or porting an existing CUDA/Triton kernel.

The Gated Attention forward pass is substantially complete in TTNN. The Gated Delta Net decode and prefill paths require four new custom kernels: `recurrent_gated_delta_rule` (decode), `causal_conv1d_fn` (prefill conv1d), `causal_conv1d_update` (decode conv1d), and `chunk_gated_delta_rule` (prefill recurrence). A fifth gap, `FusedRMSNormSwishGate`, is optional — it is composable from existing `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` primitives but lacks a single fused kernel form.

## Sections

1. [`existing_ttnn_primitives_survey.md`](./existing_ttnn_primitives_survey.md) — Comprehensive audit of TTNN and tt-transformers primitives applicable to the hybrid forward pass. Organized into: available and already used, available but not yet connected for DeltaNet, and not yet implemented.

2. [`tt_transformers_review.md`](./tt_transformers_review.md) — Review of existing tt-transformers model code relevant to the hybrid architecture: `TTNNQwen3LinearAttention`, `TTNNQwen3FullAttention`, `TTNNQwenPagedAttentionKVCache`, and the flash-linear-attention library dependency.

3. [`development_roadmap.md`](./development_roadmap.md) — Prioritized roadmap for full on-device Gated Delta Net execution on T3K. Five priorities, from highest decode-latency impact to prefill throughput, with estimated kernel complexity and per-priority memory and shape constraints.

## Key Take-Away

The Gated Attention layer is already on-device in TTNN (SDPA, Q/K RMSNorm, Q gating, paged KV cache, all CCL). The input and output projections for both layer types are on-device. The four required custom-kernel gaps — `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, and `chunk_gated_delta_rule` — all relate to the DeltaNet core and currently fall back to `flash-linear-attention` (Triton/CUDA) and `causal-conv1d` (CUDA C extension). Porting the fused recurrent decode kernel is the highest-priority action: it eliminates the host round-trip on the critical decode path and enables L1-resident state operation.

---

**Previous:** [Chapter 6 — T3K Sharding Strategy](../ch6_t3k_sharding/index.md)
