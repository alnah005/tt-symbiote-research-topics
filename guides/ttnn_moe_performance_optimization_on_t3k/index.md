# TTNNMoE Performance Optimization on T3K

This guide provides a structured, research-backed path for profiling and optimizing `TTNNMoE` on a T3K system (1×8 Wormhole mesh). It covers the full forward-pass anatomy, collective communication costs, expert dispatch bottlenecks, matmul configuration tuning, profiling toolchains, and CPU fallback elimination — culminating in a prioritized action plan.

**Target audience:** ML systems engineers and hardware-aware model developers who are already familiar with TTNN fundamentals and the basics of Mixture-of-Experts (MoE) model structure.

---

## Chapters

| # | Chapter | Description |
|---|---------|-------------|
| 1 | [MoE Forward Pass Anatomy](ch1_moe_forward_pass_anatomy/index.md) | Full anatomy of `TTNNMoE.forward` and `TTNNExperts.forward`; prerequisite for all other chapters. |
| 2 | [CCL Latency and Topology](ch2_ccl_latency_and_topology/index.md) | Collective communication (all-gather + reduce-scatter) latency costs and topology settings. |
| 3 | [Expert Dispatch Pipeline Profiling](ch3_expert_dispatch_pipeline_profiling/index.md) | Stage-by-stage profiling of `TTNNExperts.forward` to identify the bottleneck at batch=1 decode. |
| 4 | [Matmul Config and Math Fidelity](ch4_matmul_config_and_math_fidelity/index.md) | `_make_sparse_matmul_program_config` tuning and HiFi2 vs LoFi evaluation for expert matmuls. |
| 5 | [Profiling Methodology](ch5_profiling_methodology/index.md) | Tracy and TTNN op-timer profiling workflows; router 3-pass centering cost analysis. |
| 6 | [CPU Fallback Elimination](ch6_cpu_fallback_elimination/index.md) | Audit and elimination of the `Glm4MoeNaiveMoeHybrid` CPU fallback path. |
| 7 | [End-to-End Optimization Summary](ch7_end_to_end_optimization_summary/index.md) | Prioritized optimization matrix and sequenced action plan synthesizing Chapters 1–6. |

---

## Research Questions Quick Reference

| Question | Topic | Answered In |
|----------|-------|-------------|
| Q1 | What CCL topology and link settings minimize all-gather / reduce-scatter latency on T3K? | [Ch2](ch2_ccl_latency_and_topology/index.md) |
| Q2 | Which stage of `TTNNExperts.forward` is the dominant bottleneck at batch=1 decode? | [Ch3](ch3_expert_dispatch_pipeline_profiling/index.md) |
| Q3 | Is the current `_make_sparse_matmul_program_config` output optimal for the active expert tile shape? | [Ch4](ch4_matmul_config_and_math_fidelity/index.md) |
| Q4 | What is the accuracy/latency trade-off between HiFi2 and LoFi for expert matmuls? | [Ch4](ch4_matmul_config_and_math_fidelity/index.md) |
| Q5 | How much time is spent applying per-expert weights relative to the matmul itself? | [Ch3](ch3_expert_dispatch_pipeline_profiling/index.md) |
| Q6 | Can the `Glm4MoeNaiveMoeHybrid` CPU fallback path be fully eliminated? | [Ch6](ch6_cpu_fallback_elimination/index.md) |
| Q7 | What is the latency cost of the router's 3-pass centering operation? | [Ch5](ch5_profiling_methodology/index.md) |
| Q8 | What is the recommended profiling toolchain for isolating per-op costs in `TTNNMoE`? | [Ch5](ch5_profiling_methodology/index.md) |

---

## Recommended Reading Order

Read **Chapter 1 first** (prerequisite), then Chapters 2–6 in order or via the Quick Reference table above, then **Chapter 7 last** (synthesis and action plan).
