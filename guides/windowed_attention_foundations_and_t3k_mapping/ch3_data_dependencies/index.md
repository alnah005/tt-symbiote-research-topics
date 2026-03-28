# Chapter 3 — Data Dependencies and Memory Access Patterns

Chapter 2 established the physical data structure that the windowed KV cache
occupies in device DRAM: a fixed-shape tensor `[B, H, w, d]` that functions as
a circular buffer, with slot assignment `slot(t) = t mod w` and a companion
scalar `pos_offset = t - w + 1` that anchors the circular indices to absolute
sequence positions. The buffer is allocated once, never resized, and mutated by
one slot write per decode step.

This chapter asks the complementary question: when attention computation
actually reads from that buffer, what is the exact memory access pattern, and
what does it cost? The answer differs substantially between the two operational
phases of autoregressive inference.

## Reading Order

1. [`prefill_access_patterns.md`](./prefill_access_patterns.md) — the prefill
   phase: band-diagonal mask structure on the `[T, T]` score matrix, the
   sliding-stripe KV read pattern, whether the kernel can be expressed as
   masked full-attention or requires a tiled streaming approach, and a
   detailed arithmetic intensity analysis as a function of `w` and `T`.

2. [`decode_access_patterns.md`](./decode_access_patterns.md) — the decode
   phase: how a single query vector reads exactly `w` K vectors and `w` V
   vectors from the circular buffer, the bandwidth reduction factor `w / T`
   relative to full-attention decode, the data dependency graph showing the
   absence of inter-token dependencies within the window read, and the
   implication for pipeline scheduling on Wormhole.

## Connection to Chapter 2

The fixed-shape `[B, H, w, d]` circular buffer and its wrap-boundary two-segment read structure are fully described with diagrams in [`decode_access_patterns.md`](./decode_access_patterns.md).

## What Comes Next

Chapter 4 — TTNN Primitive Operations and Tensor Shapes — maps the access
patterns characterised here onto concrete TTNN ops (`ttnn.matmul`,
`ttnn.update_cache`, `ttnn.scaled_dot_product_attention`), documents the
expected tensor shapes at each op boundary, and identifies where current TTNN
primitives natively support the circular-gather semantics and where workarounds
are required.
