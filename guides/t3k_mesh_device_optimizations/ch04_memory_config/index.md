# Chapter 4: Memory Configuration — L1 vs. DRAM for Decode and Prefill

## Prerequisites

This chapter assumes familiarity with:

- **Chapter 1** — T3K topology, Ethernet fabric, and the 8-chip linear mesh
- **Chapter 2** — `ttnn` mesh API: `MeshDevice`, `MeshTensor`, and distributed tensor sharding primitives
- **Chapter 3** — All-to-all operation mechanics, `num_links`, and bandwidth utilization on T3K

Readers who have not completed Chapters 1–3 will encounter undefined terms. Return here after completing them.

---

## Goal

This chapter explains the **Wormhole B0 memory hierarchy** and provides actionable decision criteria for placing tensors in L1 SRAM versus DRAM during inference. Specific goals:

1. Build a precise mental model of L1 and DRAM on Wormhole B0: capacities, bandwidths, allocation semantics, and failure modes.
2. Provide a **decision framework** for tensor placement — not a single rule, but a set of criteria based on tensor size, access pattern, and phase (decode vs. prefill).
3. Give **concrete recommendations** for the major tensor categories in a MoE LLM on T3K: KV cache, activations, routing scores, and all-to-all buffers.

---

## When to Read This Chapter vs. Accepting TTNN Defaults

TTNN's default memory placement is DRAM interleaved for all output tensors. For many workloads this is correct and sufficient. You do not need to read this chapter in detail if:

- You are not hitting `ttnn.exceptions.MemoryAllocationError` during kernel execution.
- Profiling shows no significant latency attributable to DRAM bandwidth saturation.
- You are in early prototyping and correctness is the priority over throughput.

Read this chapter when:

- L1 allocation errors appear during kernel execution, indicating you need to understand why and how to restructure tensor placement.
- Profiling shows decode step latency is dominated by memory access time rather than compute time.
- You are ready to tune a working implementation for production latency targets.
- You need to reason about the all-to-all buffer volumes at scale (long sequences, large batch) and whether DRAM bandwidth will be a bottleneck.

> **Tip:** The recommended workflow is: get a correct implementation with TTNN defaults first, then use the profiler to identify memory-bound hotspots, then apply the placement guidance in this chapter incrementally.

---

## Summary Table: Memory Placement Recommendations by Tensor Type

The following table gives top-level recommendations. Detailed justification for each row is in the sub-files listed in the Reading Order section.

| Tensor Type | Phase | Recommended Placement | Key Reason |
|---|---|---|---|
| KV cache | Decode | DRAM (`ttnn.DRAM_MEMORY_CONFIG`) | Large, persistent across steps; grows with sequence length |
| KV cache | Prefill | DRAM (`ttnn.DRAM_MEMORY_CONFIG`) | Generated incrementally; shared with decode without migration |
| Current-token hidden states / query | Decode | L1 (`ttnn.L1_MEMORY_CONFIG`) | Shape `[B, H]`; 1 row/core when HEIGHT_SHARDED — 14.0 KB/core at B=32, H=7168 |
| Expert routing scores / top-k indices | Decode | L1 (`ttnn.L1_MEMORY_CONFIG`) | Shape `[B, E]` = `[32, 256]` = 16 KB total; trivially small |
| All-to-all dispatch/combine buffers | Decode | L1 (when B is small) | ~448 KB total at B=32; 1 row/core HEIGHT_SHARDED — 14.0 KB/core |
| All-to-all dispatch/combine buffers | Prefill | DRAM (`ttnn.DRAM_MEMORY_CONFIG`) | Can exceed 900 MB at B=32, S=2048 — far exceeds L1 |
| Attention Q/K/V projections | Prefill (B·S ≤ 2,880) | L1 HEIGHT_SHARDED | Q+K+V fit simultaneously (~1.1 MB/core at B=1,S=2048; ~1.48 MB/core at threshold B·S=2,880) |
| Attention Q/K/V projections | Prefill (B·S > 2,880) | DRAM interleaved | Q+K+V combined exceeds per-core L1 budget |
| Full activation tensors (hidden states) | Prefill | DRAM interleaved | 29+ MB for S=2048; far exceeds aggregate L1 budget |
| Expert FFN intermediates | Decode | L1 if fits, else DRAM | Depends on E_d and sharding; verify per-core budget |

All values above assume Qwen3.5-35B parameters: `E=256`, `k=8`, `H=7168`, `E_d=32` experts per device, BF16 precision (2 bytes/element) on a T3K device with 80 Tensix cores per chip.

---

## Chapter Notation

| Symbol | Meaning | Typical Value (Qwen3.5-35B / T3K) |
|---|---|---|
| $B$ | Batch size | 1–32 (decode), 1–4 (prefill) |
| $S$ | Sequence length | 1 (decode), 512–32768 (prefill) |
| $H$ | Hidden dimension | 7168 |
| $E$ | Total number of experts | 256 |
| $k$ | Top-k experts per token | 8 |
| $N$ | Number of T3K devices in the mesh | 8 |
| $E_d$ | Experts per device ($E / N$) | 32 |
| $C$ | Expert capacity per device = $\lceil k B S / E \rceil$ | 1 (decode, B=32), 2048 (prefill, B=32, S=2048) |
| $H_{KV}$ | KV head dimension | Model-dependent |
| $n_{\text{cores}}$ | Tensix cores per Wormhole B0 chip | 80 (8×10 grid) |
| $L1_{\text{core}}$ | L1 SRAM per core | 1.5 MB |
| $L1_{\text{chip}}$ | Aggregate L1 per chip | 120 MB |
| $t_{\text{tile}}$ | Tile size in elements | 1024 (32×32) |

---

## Reading Order

Read the sub-files in this order:

1. **`wormhole_memory_hierarchy.md`** — Hardware foundations: L1 and DRAM capacities, bandwidths, circular buffer allocation, and shard layouts. Read this first to build the mental model that the later files assume.

2. **`memory_config_api.md`** — The `ttnn` API for specifying memory placement: `MemoryConfig`, `TensorMemoryLayout` variants, predefined configs, and how to migrate tensors between L1 and DRAM. Read this second for the vocabulary used in the strategy files.

3. **`decode_memory_strategy.md`** — Placement recommendations specific to the decode phase. Includes worked L1 budget estimates at B=32, H=7168 and a trade-off table.

4. **`prefill_memory_strategy.md`** — Placement recommendations for the prefill phase. Covers chunked prefill, all-to-all buffer sizing at scale, and the prefill-to-decode KV cache handoff.

---

## References

- TT-NN API documentation: `ttnn.MemoryConfig`, `ttnn.DRAM_MEMORY_CONFIG`, `ttnn.L1_MEMORY_CONFIG`
- Tenstorrent Wormhole B0 Architecture Overview (internal specification)
- Chapters 1–3 of this guide
- `tt-metal` GitHub repository: `ttnn/cpp/ttnn/tensor/tensor_spec.hpp`, `ttnn/cpp/ttnn/tensor/types.hpp`
