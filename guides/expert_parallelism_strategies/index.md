# Expert Parallelism Strategies

## Overview

This guide addresses how to efficiently run Qwen3.5-35B — a Mixture-of-Experts (MoE) model with 256 total experts and top-8 routing per token — across a T3K 8-device mesh. The three central questions are:

1. **Communication:** How do tokens travel from originating devices to the devices that hold the selected experts, and back?
2. **Assignment:** Which of the 256 experts should live on which of the 8 devices?
3. **Routing overhead:** How can the router's forward pass be made fast enough to avoid becoming a bottleneck on the critical path?

The guide builds from fundamentals (MoE architecture and the routing problem) through communication primitives, device assignment strategies, routing optimization, and pipeline design, concluding with a concrete recommended configuration for Qwen3.5-35B on T3K.

**Target audience:** ML systems engineers and compiler/runtime developers with transformer architecture familiarity, basic Python/TTNN knowledge, and awareness of multi-device parallelism. No prior experience with all-to-all collectives or expert sharding is assumed.

---

## Chapter Index

| Chapter | Description |
|---|---|
| [Ch 1 — MoE Fundamentals and the Routing Problem](ch01_moe_fundamentals/index.md) | MoE layer structure, routing math, dispatch/combine pattern, and Qwen3.5-35B architectural constants |
| [Ch 2 — All-to-All Communication Primitives](ch02_all_to_all_primitives/index.md) | `all_to_all_dispatch` and `all_to_all_combine` semantics, buffer layouts, and latency/throughput model |
| [Ch 3 — Alternative Expert Routing Schemes](ch03_alternative_routing_schemes/index.md) | Expert sharding, pipeline parallelism, hierarchical routing, and a quantitative scheme comparison matrix |
| [Ch 4 — Expert-to-Device Assignment for 256 Experts on 8 Devices](ch04_expert_device_assignment/index.md) | Uniform partitioning, load-aware bin-packing, expert replication, and topology-aware placement |
| [Ch 5 — Routing Weight Computation and Overhead Minimization](ch05_routing_weight_optimization/index.md) | Router forward pass, top-k selection efficiency, weight normalization timing, and kernel fusion |
| [Ch 6 — Fused Dispatch-Compute-Combine Pipeline](ch06_fused_dispatch_compute_combine/index.md) | Six-stage MoE pipeline, micro-batch double-buffering, expert FFN tiling, and end-to-end latency model |
| [Ch 7 — Load Balancing at Inference Time](ch07_load_balancing/index.md) | Load imbalance detection, capacity overflow handling, and dynamic routing strategies |
| [Ch 8 — Optimal Strategy for Qwen3.5-35B on T3K](ch08_qwen35b_t3k_strategy/index.md) | Synthesis: concrete recommended configuration with justification from all prior chapters |

---

## Chapter Summaries

### [Chapter 1 — MoE Fundamentals and the Routing Problem](ch01_moe_fundamentals/index.md)

Establishes the vocabulary and mathematical foundation. Covers the MoE layer structure (gating network, expert bank, top-k selection), the routing equations ($g = xW_r$, softmax, top-k), and the core systems challenge: distributing 256 experts across 8 devices forces token embeddings to cross device boundaries. Introduces the dispatch/combine communication pattern at a conceptual level and provides Qwen3.5-35B's specific architectural constants ($E=256$, $k=8$, $H=7168$, $N=8$).

### [Chapter 2 — All-to-All Communication Primitives](ch02_all_to_all_primitives/index.md)

Deep-dives into the all-to-all collective as the canonical mechanism for MoE expert dispatch and combine. Defines the operation precisely, covers TTNN implementation details (pre-dispatch packing, tile alignment, shard layout), derives the bandwidth/latency model for T3K Ethernet, and identifies the batch-size threshold below which communication overhead exceeds expert FFN compute.

### [Chapter 3 — Alternative Expert Routing Schemes](ch03_alternative_routing_schemes/index.md)

Benchmarks three alternatives to all-to-all: expert sharding via all-gather, pipeline expert parallelism, and hierarchical two-level routing. For each scheme, derives communication volume, load-balance sensitivity, and implementation complexity. Concludes with a decision flowchart and recommendation for Qwen3.5-35B on T3K.

### [Chapter 4 — Expert-to-Device Assignment for 256 Experts on 8 Devices](ch04_expert_device_assignment/index.md)

Addresses static and dynamic strategies for assigning 256 experts to 8 devices. Covers uniform round-robin assignment (32 experts/device), load-aware bin-packing using per-expert routing frequencies, expert replication for hot experts, and topology-aware placement to minimize hop counts on the T3K 1×8 linear mesh.

### [Chapter 5 — Routing Weight Computation and Overhead Minimization](ch05_routing_weight_optimization/index.md)

Dissects the router sub-graph: linear projection ($W_r \in \mathbb{R}^{7168 \times 256}$), softmax vs. sigmoid normalization, and top-k selection. Analyzes why partial sort is ~7× cheaper than full sort for $E=256$, $k=8$. Presents fusion opportunities (projection + top-k + index extraction as one kernel) and the strategy of deferring weight normalization to the combine step.

### [Chapter 6 — Fused Dispatch-Compute-Combine Pipeline](ch06_fused_dispatch_compute_combine/index.md)

Assembles the six-stage MoE pipeline (route → pack/dispatch → expert FFN → combine → unpack/accumulate → residual add) and introduces micro-batch double-buffering to overlap router compute with dispatch/combine communication. Derives buffer requirements, covers expert FFN tiling across 80 Tensix cores, and provides a parameterized end-to-end latency model for Qwen3.5-35B at decode batch sizes.

### [Chapter 7 — Load Balancing at Inference Time](ch07_load_balancing/index.md)

Addresses runtime load imbalance: how to quantify it (coefficient of variation $CV$, hot-expert threshold), what happens when expert capacity overflows (token dropping vs. overflow routing), and how to apply dynamic mitigation (load-aware routing adjustment, temperature scaling, expert replication, expert score biases calibrated on a representative dataset).

### [Chapter 8 — Optimal Strategy for Qwen3.5-35B on T3K](ch08_qwen35b_t3k_strategy/index.md)

Synthesis chapter. Consolidates recommendations from all prior chapters into a single actionable configuration: uniform 32-experts-per-device assignment, all-to-all dispatch/combine as the primary communication mechanism, BF16 routing weights with deferred normalization, capacity factor CF=1.25, and per-expert score biases calibrated on 512 samples. Closes with open questions for future investigation.

---

## Cross-Chapter Dependency Map

```
Chapter 1 (MoE Fundamentals)
    └── Chapter 2 (All-to-All Primitives)
            ├── Chapter 3 (Alternative Routing Schemes)
            │       └── Chapter 4 (Expert Device Assignment) [also depends on Ch 1]
            ├── Chapter 4 (Expert Device Assignment) [also depends on Ch 1, Ch 3]
            └── Chapter 5 (Routing Weight Optimization) [also depends on Ch 1]
                    └── Chapter 6 (Fused Pipeline) [also depends on Ch 2, Ch 4]
                            └── Chapter 7 (Load Balancing) [also depends on Ch 1, Ch 4]
                                    └── Chapter 8 (Synthesis) [depends on all chapters]
```

Read chapters in order 1 → 8 for a complete understanding. Chapter 8 is a synthesis and requires all prior chapters.
