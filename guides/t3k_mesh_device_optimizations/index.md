# T3K Mesh Device Optimizations

## Overview

This guide covers how to configure, tune, and profile Mixture-of-Experts (MoE) inference — specifically Qwen3MoE — on a Tenstorrent T3K 8-device Wormhole mesh. It answers the practical questions that arise when moving from a single-device TTNN implementation to a production-ready multi-chip deployment:

1. **Topology:** How are the 8 T3K chips connected, and what does that mean for collective communication latency?
2. **API:** How do you initialize a `MeshDevice`, distribute tensors, and invoke collective operations?
3. **Communication tuning:** How do you select the optimal `num_links` value for `ttnn.all_to_all`?
4. **Memory placement:** When should tensors go in L1 SRAM vs. DRAM, and why does it matter for decode vs. prefill?
5. **Expert parallelism:** How do you map 256 experts across 8 devices and route tokens efficiently?
6. **Profiling:** How do you identify whether a workload is compute-bound, memory-bandwidth-bound, or communication-bound?

**Target audience:** ML systems engineers and kernel developers integrating or optimizing LLM inference on T3K. Familiarity with general LLM inference concepts, basic TTNN API usage, and distributed inference at a conceptual level is assumed. T3K-specific topology, mesh API internals, and MoE expert parallelism are covered from first principles.

---

## Chapter Index

| Chapter | Description |
|---|---|
| [Ch 1 — T3K Hardware Topology and Interconnect Fundamentals](ch01_t3k_topology/index.md) | Physical layout, device IDs, Ethernet link bandwidth, and topology implications for collective operations |
| [Ch 2 — TTNN MeshDevice API for Multi-Chip Operations](ch02_ttnn_mesh_api/index.md) | `MeshDevice` setup, tensor distribution (sharding vs. replication), and collective primitives (`all_to_all`, `all_reduce`, `reduce_scatter`) |
| [Ch 3 — All-to-All Operations and `num_links` Tuning on T3K](ch03_all_to_all_num_links/index.md) | Role of all-to-all in MoE dispatch/combine, `num_links` definition and tuning, and benchmarking methodology |
| [Ch 4 — Memory Configuration: L1 vs. DRAM for Decode and Prefill](ch04_memory_config/index.md) | Wormhole B0 memory hierarchy, `ttnn.MemoryConfig` API, and concrete placement recommendations by tensor type and inference phase |
| [Ch 5 — Expert Parallelism on T3K: Mapping Experts to Devices](ch05_expert_parallelism/index.md) | Expert placement strategies, token dispatch construction, and combine/accumulation for Qwen3MoE on T3K |
| [Ch 6 — Profiling and Performance Analysis on T3K](ch06_profiling/index.md) | TTNN profiler, device performance counters, and a structured bottleneck diagnosis guide |
| [Ch 7 — End-to-End Integration: TTNNQwen3MoE on T3K](ch07_end_to_end_integration/index.md) | Complete reference implementation: MoE layer pseudocode, decode loop, prefill considerations, and validation |

---

## Chapter Summaries

### [Chapter 1 — T3K Hardware Topology and Interconnect Fundamentals](ch01_t3k_topology/index.md)

Establishes the physical and logical layout of T3K: 8 Wormhole chips on a single board in a 1×8 linear mesh, device IDs 0–7, the `(1,8)` logical mesh shape used by TTNN. Characterizes the chip-to-chip Ethernet links: per-link unidirectional bandwidth (~12.5 GB/s), aggregate bidirectional bandwidth, link latency in µs, and multi-hop routing cost. Derives why ring-based collectives fit the linear 1×8 topology and introduces `num_links` as the primary bandwidth tuning knob.

### [Chapter 2 — TTNN MeshDevice API for Multi-Chip Operations](ch02_ttnn_mesh_api/index.md)

Covers the TTNN abstractions for multi-device operation. `MeshDevice` constructor parameters, device ordering conventions, and initialization sequence. `TensorSpec` and `ShardSpec` for describing tensor distribution (row-wise vs. column-wise sharding, replicated vs. sharded). The collective primitives: `ttnn.all_to_all` (signature, key parameters including `num_links`, `memory_config`, `cluster_axis`), `ttnn.all_reduce`, `ttnn.reduce_scatter`, and `ttnn.all_gather`.

### [Chapter 3 — All-to-All Operations and `num_links` Tuning on T3K](ch03_all_to_all_num_links/index.md)

Deep-dives into the all-to-all collective as the critical primitive for MoE expert dispatch and combine. Derives the data volume at each inference phase (prefill vs. decode). Explains `num_links`: the parameter controls how many Ethernet links are used per collective, with near-linear bandwidth scaling up to saturation. Key finding: **`num_links=1` for decode** (small payloads, latency-bound), **`num_links=2` for prefill** (large payloads, throughput-bound). Provides benchmark methodology for empirical tuning.

### [Chapter 4 — Memory Configuration: L1 vs. DRAM for Decode and Prefill](ch04_memory_config/index.md)

Explains the Wormhole B0 memory hierarchy (80 Tensix cores × 1.5 MB L1 = 120 MB aggregate L1; DRAM per device). Covers the `ttnn.MemoryConfig` API and `TensorMemoryLayout` variants. Key recommendation table: KV cache → DRAM; decode activations and routing scores → L1; all-to-all buffers → L1 for decode (≤448 KB at B=32), DRAM for prefill (can exceed 900 MB); expert FFN intermediates → L1 if fits.

### [Chapter 5 — Expert Parallelism on T3K: Mapping Experts to Devices](ch05_expert_parallelism/index.md)

Synthesizes topology, collective API, bandwidth model, and memory hierarchy into a practical guide for MoE expert parallelism. Compares four expert placement strategies (naive uniform, load-balanced, locality-aware, expert replication). Derives the token dispatch flow from router output through send-buffer packing and `ttnn.all_to_all`. Covers the combine phase: reverse all-to-all, weighted accumulation of $k=8$ expert outputs per token, and overlap opportunities with the next layer.

### [Chapter 6 — Profiling and Performance Analysis on T3K](ch06_profiling/index.md)

Teaches how to use TTNN's profiling infrastructure to identify bottlenecks before tuning. The 5-step workflow: establish correctness baseline → enable profiler → parse op-level timing → identify bottleneck category via hardware counters → apply targeted remediation. Two regimes: decode (B≤32) is typically communication-bound; prefill is typically compute-bound. Applying the wrong fix (e.g., increasing `num_links` when compute-bound) wastes engineering time.

### [Chapter 7 — End-to-End Integration: TTNNQwen3MoE on T3K](ch07_end_to_end_integration/index.md)

Integrates all prior material into a production-ready reference implementation. Provides a full `moe_layer_t3k` pseudocode function applying topology, mesh API, `num_links`, memory config, and expert parallelism in the correct order. Shows the complete autoregressive decode loop with memory lifecycle management and batch-padding conventions. Maps prefill vs. decode parameter differences (memory placement, `num_links`, capacity $C$) to the chapter that introduced each.

---

## Key Recommendations at a Glance

| Decision | Decode (B≤32) | Prefill |
|---|---|---|
| `num_links` | 1 | 2 |
| Activation memory | `L1_MEMORY_CONFIG` | `DRAM_MEMORY_CONFIG` |
| All-to-all buffers | L1 | DRAM |
| KV cache | DRAM | DRAM |
| Expert weights | DRAM | DRAM |

---

## Cross-Chapter Dependency Map

| Chapter | Depends On |
|---|---|
| Ch 1: T3K Topology | None (foundational) |
| Ch 2: TTNN MeshDevice API | Ch 1 |
| Ch 3: All-to-All / `num_links` | Ch 1, Ch 2 |
| Ch 4: Memory Config | Ch 2 |
| Ch 5: Expert Parallelism | Ch 1, Ch 2, Ch 3, Ch 4 |
| Ch 6: Profiling | Ch 2, Ch 3, Ch 4, Ch 5 |
| Ch 7: End-to-End Integration | Ch 1 through Ch 6 |

Read chapters in order 1 → 7 for a complete understanding.
