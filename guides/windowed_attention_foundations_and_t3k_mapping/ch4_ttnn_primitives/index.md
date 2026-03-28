# Chapter 4 — TTNN Primitive Operations and Tensor Shapes

Chapters 1–3 established the mathematical definition of windowed attention, the
circular-buffer KV cache layout, and the memory access patterns for both prefill
and decode. This chapter translates those abstract descriptions into concrete
TTNN operations, specifying the exact tensor shapes at every op boundary, the
program configuration knobs that govern performance, and the places where current
TTNN primitives support the windowed case natively versus where workarounds or
new kernels are required.

This chapter addresses two of the guide's core research questions directly:

- **Q4** — Which TTNN ops and tensor shapes are required for windowed attention
  decode and prefill on T3K?
- **Q8** — Where do gaps exist between the windowed attention algorithm's
  requirements and the available TTNN/tt-transformers kernel surface?

## Reading Order

1. [`decode_primitives.md`](./decode_primitives.md)
2. [`prefill_primitives.md`](./prefill_primitives.md)
3. [`kernel_or_op_gap_analysis.md`](./kernel_or_op_gap_analysis.md)

## Connection to Prior Chapters

The tensor shapes used throughout this chapter originate in Chapter 2's circular
buffer derivation. The fixed-shape KV tensor `[B, H_kv, w, d]` introduced in
[`../ch2_kv_cache_management/circular_buffer_layout.md`](../ch2_kv_cache_management/circular_buffer_layout.md)
is the primary input to every op described here. The two-segment read structure
and the write-ordering constraint established in
[`../ch3_data_dependencies/decode_access_patterns.md`](../ch3_data_dependencies/decode_access_patterns.md)
directly motivate the window-enforcement strategies discussed in
[`decode_primitives.md`](./decode_primitives.md). The band-diagonal mask analysed
in
[`../ch3_data_dependencies/prefill_access_patterns.md`](../ch3_data_dependencies/prefill_access_patterns.md)
is constructed concretely in [`prefill_primitives.md`](./prefill_primitives.md).

## What Comes Next

Chapter 5 — Paged KV Cache Interaction — takes the circular-buffer tensor shapes
established here and examines how they interact with the paged KV cache in
tt-transformers, specifically whether `paged_sdpa_decode` can enforce a window
constraint and what interface changes would be needed.

Chapter 6 — T3K Mesh Sharding — uses the per-layer tensor shapes from this
chapter to compute per-device shard sizes and collective communication volumes
for the 1×8 Wormhole mesh.

Chapter 7 — Roofline Analysis and Existing Kernel Survey — extends the gap
analysis started in [`kernel_or_op_gap_analysis.md`](./kernel_or_op_gap_analysis.md)
with hardware-grounded roofline numbers and a comprehensive survey of all
Flash-Attention style tiled kernels available in TTNN.
