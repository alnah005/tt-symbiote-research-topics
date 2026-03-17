# Tracy Profiling and MoE Forward Pass Analysis

This guide teaches you to use the Tracy profiler and the TTNN device profiler together to
analyze the performance of a Mixture-of-Experts (MoE) forward pass on Wormhole B0 and T3K
hardware, locate latency gaps between named zones, and map each gap to a concrete
optimization action. The motivating problem is a 16ms unaccounted gap observed in the MoE
forward pass of Qwen 235B-A22B and DeepSeek-V3 on T3K; the method generalizes to any TTNN
workload where host-dispatch overhead, device synchronization barriers, or CCL collective
latency is suspected to be limiting throughput.

---

## Prerequisites

- Python 3.8+ and a working tt-metal installation with TTNN
- Basic TTNN familiarity: `ttnn.matmul`, `ttnn.to_device`, `ttnn.MemoryConfig`
- Access to Wormhole B0 hardware (single-chip for Chapters 1–4; T3K 8-chip mesh for Chapters 5–7 CCL patterns)
- Conceptual knowledge of MoE routing: gating network, top-K selection, dispatch/combine pattern

---

## Chapters

| Chapter | What you learn |
|---|---|
| [Ch 1 — Tracy Profiler Overview](ch1_tracy_overview/index.md) | Tracy data model, two-process architecture, `TRACY_ENABLE` flag, and how Tracy differs from the TTNN device profiler |
| [Ch 2 — Setting Up Tracy Profiling](ch2_tracy_setup/index.md) | CMake build flags (`ENABLE_TRACY=ON`), launch order, `TRACY_NO_EXIT=1`, output artifacts (`.tracy` file and `profile_log_device.csv`), common failure modes |
| [Ch 3 — TTNN Op-Level Profiling API](ch3_ttnn_profiling_api/index.md) | `TT_METAL_DEVICE_PROFILER=1`, custom Tracy zone markers in Python, `process_ops_logs.py`, reconstructing total MoE time from CSV output |
| [Ch 4 — MoE Forward Pass Op Breakdown](ch4_moe_op_breakdown/index.md) | Complete 13-step op sequence (dispatch / expert matmul / combine phases), per-phase latency budgets on Wormhole B0, T3K CCL ops, Qwen 235B reference configuration |
| [Ch 5 — Identifying the 16ms Gap](ch5_identifying_gap/index.md) | Tracy GUI navigation, gap measurement via `tracy-csvexport`, three-method attribution (host Python / sync barrier / CCL), four common gap patterns A–D |
| [Ch 6 — Sequence Length Scaling Analysis](ch6_sequence_length_scaling/index.md) | Why `seq_len` is the primary variable, four scaling behaviors (O(1) / O(seq_len) / sublinear / non-monotonic), controlled 7-point sweep, log-log fitting, mixed-gap decomposition |
| [Ch 7 — Interpretation and Next Steps](ch7_interpretation_and_next_steps/index.md) | Mapping gap patterns to TTNN optimization levers, writing a gap analysis document, building a prioritized optimization backlog |

---

## Key Facts Reference

**Dispatch patterns (single-chip):**

| Pattern | Description | Dispatch count |
|---|---|---|
| A | Fused SwiGLU (gate and up projections merged) | 3 dispatches per expert |
| B | Unfused SwiGLU (separate gate, up, silu, multiply) | 4 dispatches per expert |
| C | T3K with CCL | Adds `allgather` before expert matmuls and `reduce_scatter` after; wraps Pattern A or B |

**CCL scaling law:** latency scales as O(num_active_tokens × d_model) — does not divide by
num_chips because every chip must participate in the collective.

**Memory-bound condition:** expert matmuls are memory-bound when
`expert_capacity / 32 < num_cores`, where `expert_capacity = seq_len × top_k / num_experts`.

For Qwen 235B (top_k=8, num_experts=128): `expert_capacity = seq_len / 16`.
Memory-bound regime: seq_len < ~40,960.

**Benchmark protocol:** 20 timed iterations; report median and p95; warm-up at least 3
iterations before timing begins.

**T3K ethernet bandwidth:** ~7 GB/s effective per inter-chip link.

---

## How to Read This Guide

**First-timers:** read sequentially from Chapter 1 through Chapter 7. Each chapter declares
its prerequisites explicitly; skipping ahead without the stated background will produce
undefined references to zone names, CSV columns, and hardware constants defined in earlier
chapters.

**Practitioners already capturing Tracy traces:** start at Chapter 5 (gap identification)
and work through Chapter 7 (optimization backlog). Chapter 4's op sequence table and
configuration block are the most likely references you will need from the earlier material.

---

## Start Here

Begin with [Chapter 1 — Tracy Profiler Overview](./ch1_tracy_overview/index.md).
