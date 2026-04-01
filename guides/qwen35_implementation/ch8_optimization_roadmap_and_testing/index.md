# Chapter 8 — Optimization Roadmap and Testing

This chapter covers the planned performance optimizations and the complete testing
infrastructure for validating Qwen3.5 on Blackhole. Together they answer two questions:
*What needs to change to close the gap to theoretical peak?* and *How do we verify
correctness after each change?*

## Reading order

1. [`optimization_roadmap.md`](./optimization_roadmap.md) — Metal Trace, Multi-CQ overlap, per-row MoE routing, and the enabling path for the fused DeltaNet kernel
2. [`testing_infrastructure.md`](./testing_infrastructure.md) — all test files, their scope, classes, and PCC thresholds
3. [`running_tests.md`](./running_tests.md) — exact commands to run each test and what to expect

## Prerequisites

- Chapter 7 (bottleneck analysis) — motivates each optimization in the roadmap
- Chapters 2–5 (module implementations) — needed to understand what each test validates

## What ships today vs what is planned

| Optimization | Status |
|---|---|
| Fused `gated_delta_net` kernel | **Deployed** (PCC 0.999997) |
| Device-side partial RoPE | **Deployed** (eliminates 5 syncs/attention layer) |
| Metal Trace | **Planned** (requires stable tensor addresses — already ensured by in-place `ttnn.copy`) |
| Multi-CQ overlap | **Planned** (complementary to Metal Trace) |
| Per-row MoE routing | **Planned** (future work; current code assumes same-prompt batch) |

---

**Next:** [`optimization_roadmap.md`](./optimization_roadmap.md)
