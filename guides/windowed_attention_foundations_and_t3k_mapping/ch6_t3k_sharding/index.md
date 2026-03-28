# Chapter 6 — T3K Mesh Sharding and CCL Implications

Chapters 1–4 built the single-device picture of windowed attention: the
mathematics of the window constraint, the circular-buffer KV cache layout, the
memory access patterns during decode and prefill, and the concrete TTNN
operations and tensor shapes. Chapter 5 extended that picture to a paged
storage backend on a single device. Chapter 6 moves to the multi-device
setting and asks how the windowed KV cache is distributed across the eight
Wormhole chips of a T3K board and what collective communication operations
that distribution requires.

The answer depends critically on which dimension of the KV tensor
`[B, H_kv, w, d]` is sharded across devices. Two strategies are developed and
compared in detail. The chapter concludes with an analysis of the correctness
requirements for applying the window boundary after sharding, including the
edge cases that arise when `w` is not divisible by the number of devices.

## Prerequisites

This chapter requires tensor shape definitions from Chapter 4. Specifically:

- The GQA decode shapes `[B, H_q, 1, d]` / `[B, H_kv, w, d]` introduced in
  [`../ch4_ttnn_primitives/decode_primitives.md`](../ch4_ttnn_primitives/decode_primitives.md).
- The distinction between the fill phase (`T < w`) and steady-state decode
  (`T >= w`) from
  [`../ch4_ttnn_primitives/decode_primitives.md`](../ch4_ttnn_primitives/decode_primitives.md).
- The arithmetic intensity result from
  [`../ch3_data_dependencies/decode_access_patterns.md`](../ch3_data_dependencies/decode_access_patterns.md)
  establishing that windowed decode is bandwidth-bound (AI ≈ 1 FLOP/byte),
  which informs the CCL cost model in this chapter.
- The window size parameter `w` and the head count notation `H_q`, `H_kv`
  defined in
  [`../ch1_math_foundations/window_size_parameter.md`](../ch1_math_foundations/window_size_parameter.md)
  and
  [`../ch1_math_foundations/full_vs_windowed_attention.md`](../ch1_math_foundations/full_vs_windowed_attention.md).

## Reading Order

1. [`sharding_strategies.md`](./sharding_strategies.md) — Describes the T3K
   topology, derives the two candidate sharding strategies, and provides a
   decision matrix with a recommendation.

2. [`ccl_operations.md`](./ccl_operations.md) — Identifies the CCL primitives
   (`ttnn.all_gather`, `ttnn.reduce_scatter`) required under each sharding
   strategy, quantifies the bandwidth demand against T3K Ethernet link speed,
   and analyses overlap opportunities.

3. [`per_device_window_application.md`](./per_device_window_application.md) —
   Analyses whether the window boundary is applied globally before sharding or
   per-device after sharding, proves correctness for the head-parallel case, and
   addresses the divisibility edge cases that arise in sequence-parallel sharding.

## Chapter Scope

### T3K Topology

The full topology description — 1×8 linear mesh, per-device resources, and
CCL primitive overview — is in
[`sharding_strategies.md`](./sharding_strategies.md).

### Sharding Strategies

[`sharding_strategies.md`](./sharding_strategies.md) derives two candidate
strategies (head-parallel and sequence-parallel) and concludes that
head-parallel sharding is correct for windowed attention decode on T3K because
its CCL is off the critical path, independent of `w`, and achieves genuine
`N×` parallelism. See that file for the full decision matrix and rationale.

### CCL Operations

[`ccl_operations.md`](./ccl_operations.md) maps each strategy to its required
CCL primitives (`ttnn.all_gather`, `ttnn.reduce_scatter`), quantifies bandwidth
demand (≈ 224 KiB for head-parallel vs. ≈ 448 MiB for sequence-parallel at
w=4096), and analyses overlap opportunities and window-size scaling.

### Per-Device Window Application

[`per_device_window_application.md`](./per_device_window_application.md)
addresses the question of when and where the window boundary is enforced. For
head-parallel sharding the answer is straightforward: each device applies the
full `w`-length window independently to its own heads, so the global window
constraint is automatically satisfied. For sequence-parallel sharding the
picture is more subtle: each device holds a slice of the global window, and
the slice boundaries must be computed carefully when `w` is not divisible by
eight to avoid exposing stale tokens at the inter-device boundaries.

## What Comes Next

Chapter 7 — Roofline Analysis and Existing Kernel Survey — takes the per-layer
FLOPs and memory-access volumes derived from the tensor shapes in Chapter 4
and plots them against Wormhole hardware limits to determine whether windowed
attention decode is compute-bound or bandwidth-bound. It also surveys the
existing TTNN kernel surface (`ttnn.scaled_dot_product_attention`,
`ttnn.scaled_dot_product_attention_decode`, `paged_sdpa_decode`) to identify
which gaps remain between current support and a complete windowed-attention
implementation. The CCL cost model from this chapter feeds into Chapter 7's
end-to-end latency projections for T3K multi-chip decode.
