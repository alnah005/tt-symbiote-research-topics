# Chapter 7 — Roofline Analysis and Existing Kernel Survey

Chapters 1–6 built the complete single- and multi-device picture of windowed
attention: its mathematical formulation, KV cache lifecycle, memory access
patterns, TTNN primitive mappings, paged storage interaction, and T3K sharding.
Chapter 7 closes the guide by addressing two remaining questions:

- **Q7 (Roofline):** Is windowed attention decode compute-bound or
  bandwidth-bound on Wormhole at typical window sizes? What is the expected
  throughput and how does it compare to full attention?
- **Q8 (Kernel Survey):** Which existing TTNN and tt-transformers ops support
  windowed attention today, and where do the gaps lie between current support
  and a complete implementation?

Together these questions determine the implementation path: whether a new kernel
is required, or whether caller-side conventions combined with existing ops are
sufficient.

## Prerequisites

This chapter draws on results from earlier chapters:

- The arithmetic intensity derivation (AI ≈ 1 FLOP/byte for decode) and the
  prefill AI formula from
  [`../ch3_data_dependencies/decode_access_patterns.md`](../ch3_data_dependencies/decode_access_patterns.md)
  and
  [`../ch3_data_dependencies/prefill_access_patterns.md`](../ch3_data_dependencies/prefill_access_patterns.md).
- The tensor shapes `[B, H_q, 1, d]` / `[B, H_kv, w, d]` and op interfaces
  from
  [`../ch4_ttnn_primitives/decode_primitives.md`](../ch4_ttnn_primitives/decode_primitives.md).
- The detailed gap analysis from
  [`../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md`](../ch4_ttnn_primitives/kernel_or_op_gap_analysis.md),
  which this chapter summarises at a higher level of abstraction.
- The head-parallel sharding recommendation from
  [`../ch6_t3k_sharding/sharding_strategies.md`](../ch6_t3k_sharding/sharding_strategies.md).

## Reading Order

1. [`roofline_analysis.md`](./roofline_analysis.md) — Plots windowed attention
   decode against the Wormhole roofline model. Establishes that decode is deeply
   bandwidth-bound at all practical window sizes and batch sizes less than ~100.
   Quantifies the throughput advantage of windowed attention over full attention
   using a comparison table across `w` and `T` values.

2. [`existing_kernel_survey.md`](./existing_kernel_survey.md) — Surveys
   `ttnn.scaled_dot_product_attention`, `ttnn.scaled_dot_product_attention_decode`,
   `paged_sdpa`, and `paged_sdpa_decode` against five capability dimensions.
   Consolidates the gaps into a single summary table and provides a recommended
   implementation path for each gap, concluding that no new kernel is required
   for a correct initial implementation.

