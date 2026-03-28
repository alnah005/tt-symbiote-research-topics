# Chapter 5 — Roofline Analysis: Gated Delta Net on a Single Wormhole Chip

## Overview

This chapter answers **Q6: Is Gated Delta Net compute-bound or memory-bandwidth-bound on Wormhole hardware, and what does that imply for kernel design?**

Using the roofline model, we characterize the arithmetic intensity of both the decode step and the prefill pass for Gated Delta Net. The analysis is performed for a **single Wormhole ASIC** (one chip = half of one n300 card), which is the natural unit of analysis before introducing multi-device sharding.

> **Note on T3K sharding:** Chapter 6 covers how state matrices and attention tensors are sharded across all 8 chips of a T3K system. Sharding changes the per-device arithmetic intensity — each chip holds a fraction of the total state and processes a subset of heads, which shifts the balance between compute and memory traffic. The single-device analysis here is the necessary foundation for that discussion.

## Sections

1. [`wormhole_hardware_specs.md`](./wormhole_hardware_specs.md) — Per-chip hardware parameters, memory hierarchy, bandwidth figures, and the roofline ridge point for a single Wormhole ASIC in T3K.

2. [`roofline_decode_and_prefill.md`](./roofline_decode_and_prefill.md) — Step-by-step FLOP and byte counts for the Gated Delta Net decode step (B=1, T=1) and prefill pass (B=1, T=8192), arithmetic intensity calculations, comparison to Gated Attention at long context, and L1 feasibility analysis.
