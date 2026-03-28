# Chapter 4 — TTNN Primitive Mapping

## Overview

This chapter answers two key research questions:

- **Q4**: Which operations in Gated Delta Net and Gated Attention map directly to existing TTNN primitives, and which require custom kernel development?
- **Q8**: What is the complete TTNN op-by-op breakdown for a single decode step and a full prefill pass through both model components?

## Current State Summary

**Gated Attention** is substantially TTNN-accelerated. All operations in the forward pass — input projections, Q gating, Q/K normalization, RoPE, KV cache management, GQA repeat, scaled dot-product attention (both prefill and decode), output projection, and all-gather — are covered by existing TTNN primitives. No PyTorch fallback is required for the core attention path.

**Gated Delta Net** is partially accelerated. TTNN handles the large linear projections (`in_proj_qkv`, `in_proj_z`, `out_proj`) and the all-gather communication primitive. The four components unique to the DeltaNet recurrence — causal conv1d, the recurrent gated delta rule, the chunkwise gated delta rule, and gated RMSNorm — currently fall back to PyTorch. Two small projections (`in_proj_a`, `in_proj_b`) are run via PyTorch `nn.Linear` but are straightforward to wire into TTNN. The four kernel gaps are the primary development target for full T3K acceleration of this layer.

## Sections

1. [**`gated_delta_net_decode_step.md`**](./gated_delta_net_decode_step.md) — Step-by-step TTNN op mapping for one Gated Delta Net decode step (B=1, T=1), with tensor shapes and status tags for each operation.

2. [**`gated_delta_net_prefill_pass.md`**](./gated_delta_net_prefill_pass.md) — TTNN op mapping for Gated Delta Net prefill (B=1, T=full sequence), covering the chunkwise delta rule and conv1d paths.

3. [**`gated_attention_ttnn_ops.md`**](./gated_attention_ttnn_ops.md) — TTNN ops for the Gated Attention forward pass across both prefill and decode, demonstrating the fully-accelerated attention path.

4. [**`kernel_gap_summary.md`**](./kernel_gap_summary.md) — Consolidated kernel gap audit table and prioritized recommendations for custom kernel development.
