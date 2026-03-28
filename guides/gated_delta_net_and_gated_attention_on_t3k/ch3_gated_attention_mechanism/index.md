# Chapter 3: Gated Attention Mechanism

## Overview

This chapter dissects the **Gated Attention** layer used in Qwen3.5-35B-A3B. Before going further, a critical disambiguation:

> **"Gated Attention" in this guide always means `Qwen3_5MoeGatedAttention` / `TTNNQwen3FullAttention`.**
> This is standard scaled dot-product attention (SDPA) extended with a learned sigmoid gate applied to the query vectors, plus independent RMSNorm on Q and K. It is **not** Gated Linear Attention (GLA). GLA is a separate recurrent architecture (related to linear attention); it does not appear in Qwen3.5-35B-A3B.

Gated Attention appears at every 4th layer in the 40-layer model (indices 3, 7, 11, …, 39 — 10 layers total). These are the only layers that maintain a paged KV cache. All remaining layers are Gated Delta Net layers (covered in Chapters 1–2).

## This Chapter Answers

**Research Question 2:** What is the Gated Attention mechanism in Qwen3.5-35B-A3B, and how do its tensor shapes compare to vanilla MHA and GQA?

## Learning Objectives

By the end of this chapter you will be able to:

1. Trace the full forward pass of a Gated Attention layer from `hidden_states [B, T, 2048]` to output `[B, T, 2048]`, including every projection, gate, norm, RoPE, and SDPA step.
2. Explain why the Q+gate projection is sized `[B, T, 8192]` (doubling the standard Q projection) and what the sigmoid gate does geometrically.
3. State the exact shapes of Q, K, V, and the gate tensor at prefill vs. decode, and reason about their memory footprint.
4. Compare Gated Attention tensor shapes against vanilla MHA, standard GQA, and Gated Delta Net.

## Sections

- [`gated_attention_formulation.md`](./gated_attention_formulation.md) — Step-by-step derivation of the Gated Attention forward pass with full shape arithmetic.
- [`gated_vs_vanilla_attention_shapes.md`](./gated_vs_vanilla_attention_shapes.md) — Tensor shape comparison table and key architectural differences from vanilla MHA, GQA, and Gated Delta Net.

## Symbols Used in This Chapter

| Symbol | Value | Meaning |
|--------|-------|---------|
| H | 2048 | Model hidden size |
| d_h | 256 | Head dimension (Gated Attention) |
| n_q_h | 16 | Number of query heads |
| n_kv_h | 2 | Number of KV heads |
| rotary_dim | 64 | Number of head dims subject to RoPE |
| B | — | Batch size |
| T | — | Sequence length |
