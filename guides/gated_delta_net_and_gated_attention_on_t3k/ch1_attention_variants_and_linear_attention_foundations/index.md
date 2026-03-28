# Chapter 1 — Attention Variants and Linear Attention Foundations

## Overview

Before Gated Delta Net can be understood in depth, you need a precise working vocabulary for the space of recurrent attention mechanisms it inhabits. This chapter builds that vocabulary from first principles: starting from standard causal attention, deriving linear attention as a fixed-size RNN, and then mapping the landscape of variants — RetNet, GLA, Mamba2, and DeltaNet — against a common state-update framework. Chapter 2 then builds Gated Delta Net directly on top of these foundations.

## Learning Objectives

After completing this chapter you should be able to:

1. State the SDPA formula and identify where sequence-length cost accumulates in both the prefill and decode regimes.
2. Identify the structural difference between GQA (used in Gated Attention layers) and standard MHA.
3. Derive the linear attention RNN recurrence from the kernel-function substitution for softmax.
4. Explain why the vanilla linear attention state S degrades as a retrieval structure over long sequences.
5. Place RetNet, GLA, Mamba2, and DeltaNet on a single taxonomy axis defined by their forgetting gate G_t and write mechanism.
6. State precisely what DeltaNet's delta rule minimizes and why that differs from an additive write.

## Sections

| # | File | What it covers |
|---|------|----------------|
| 1 | [`standard_softmax_attention.md`](./standard_softmax_attention.md) | SDPA formula, KV cache complexity, GQA configuration |
| 2 | [`linear_attention_rnn_equivalence.md`](./linear_attention_rnn_equivalence.md) | Kernel substitution, state-matrix recurrence, forgetting limitation |
| 3 | [`linear_attention_variants_comparison.md`](./linear_attention_variants_comparison.md) | RetNet, GLA, Mamba2, DeltaNet side-by-side; forward reference to Chapter 2 |

## Key Takeaway

Linear attention replaces the O(n) KV cache with a fixed-size state matrix **S ∈ R^{d_k × d_v}**, enabling O(1) decode at the cost of limited expressiveness. The variants in this chapter differ in exactly one thing: how they update S — specifically, whether the forgetting gate G_t is data-independent (RetNet), row-wise data-dependent (GLA), scalar data-dependent (Mamba2), or replaced entirely by a targeted error-correcting write (DeltaNet). Gated Delta Net, covered in Chapter 2, is the first variant that applies both a full forgetting gate and the delta rule in a single recurrence.
