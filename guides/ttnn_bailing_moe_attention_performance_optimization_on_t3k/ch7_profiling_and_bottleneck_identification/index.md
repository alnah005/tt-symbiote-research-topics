# Chapter 7 — Profiling and Bottleneck Identification

## Scope

This chapter provides a practical, step-by-step guide to profiling the full `TTNNBailingMoEAttention` forward pass at op-level granularity on T3K. It is the measurement foundation that all previous chapters assume: when a chapter marks a latency figure as `[ESTIMATE]`, this chapter shows how to replace that estimate with a `[MEASURED]` value on your hardware.

**Chapter 7 answers Question 8 of the guide:**

- **Question 8:** How do you profile the `TTNNBailingMoEAttention` forward at op-level granularity on T3K, and given a profiling result, which optimization from earlier chapters should you apply first?

The chapter is organized around two complementary tools — TTNN op timers and Tracy — and closes with a decision tree that maps any observed profiling result to the appropriate optimization chapter.

## Prerequisites

Chapter 7 synthesizes all previous material. Readers must understand:

- **Fused QKV projection and all-reduce** — (Chapter 2, [`fusion_mechanics.md`](../ch2_fused_qkv_projection/fusion_mechanics.md) and [`num_links_tuning.md`](../ch2_fused_qkv_projection/num_links_tuning.md)). The `TTNNLinearIColShardedWAllReduced` op and the CCL all-reduce that follows it are the two sub-components most commonly identified as dominant by profiling on large hidden sizes.
- **Host round-trip cost** — (Chapter 3, [`host_transfer_overhead.md`](../ch3_host_roundtrip_replication/host_transfer_overhead.md)). The `_to_replicated` host round-trip (see Chapter 3) appears as a host-side dead zone in Tracy.
- **Memory-config transition catalog** — (Chapter 4, [`transition_cost_model.md`](../ch4_memory_config_transitions/transition_cost_model.md)). Each `ttnn.to_memory_config` call emits a distinct op in the timer report. Knowing which transitions are expected allows rapid identification of an unexpected outlier.
- **SDPA and compute config** — (Chapter 5, [`paged_sdpa_chunk_sizes.md`](../ch5_sdpa_and_compute_config/paged_sdpa_chunk_sizes.md) and [`math_fidelity_tradeoff.md`](../ch5_sdpa_and_compute_config/math_fidelity_tradeoff.md)). The `paged_sdpa_decode` kernel is a single large entry in the op timer report; Chapter 5 explains what configuration knobs affect it.
- **QK norm and non-distributed RoPE** — (Chapter 6, [`qk_norm_latency.md`](../ch6_rope_and_qk_norm/qk_norm_latency.md) and [`partial_rotary_rope.md`](../ch6_rope_and_qk_norm/partial_rotary_rope.md)). Non-distributed RoPE adds further decode overhead (see Chapter 6) and appears as an identifiable op cluster in the timer report. The two norm-related cost figures reported across chapters measure different things and cover different scopes — report each independently when profiling:
  - The Ch4 figure (83–86 µs) and the Ch6 figure (74–92 µs) share the ~32 µs T_norm_in component (T_norm_in_Q + T_norm_in_K), so they must NOT be added together. When profiling, the timer report will surface the `to_memory_config` ops and the `RMSNorm` kernel ops as separate entries; map them to the appropriate chapter's cost model individually.
  - **Full QK norm overhead (Ch6): 74–92 µs [ESTIMATE]** (T_norm_in + TTNNRMSNorm dispatch + T_norm_out + reshape dispatch×4); T2a/T2b are excluded from this figure as they are costed in Ch4

## Profiling Tools at a Glance

Two complementary tools are used throughout this chapter. The table below summarizes when to reach for each one.

Profiling tool comparison — T3K decode step

| Tool | Granularity | Setup complexity | Best for |
|---|---|---|---|
| TTNN op timers | Per-TTNN-op wall time from host dispatch perspective | Low — environment variable or one-liner wrapper | First pass: ranking ops by dispatch time, identifying dispatch-bound vs. execution-bound ops |
| Tracy | Per-kernel device-side cycles + host-side spans | Medium — custom build + server + annotations | Deep dive: measuring actual device execution time, PCIe transfer gaps, CCL overlap |

## Reading Order

Work through the files in this order:

1. [`ttnn_op_timers.md`](./ttnn_op_timers.md) — How to enable op-level timing, how to read the output report, and how to map op names to `TTNNBailingMoEAttention` source call sites.
2. [`tracy_profiling.md`](./tracy_profiling.md) — Tracy build setup on T3K, annotating the attention forward pass, capturing a single decode step trace, and reading the timeline.
3. [`bottleneck_decision_tree.md`](./bottleneck_decision_tree.md) — Decision tree mapping any profiling result to the correct optimization chapter, with example scenarios and the recommended iteration loop.

**Start reading:** [TTNN Op Timers](ttnn_op_timers.md)
