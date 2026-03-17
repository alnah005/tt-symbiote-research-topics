# Chapter 6: Fused Dispatch-Compute-Combine Pipeline

## Overview

This chapter assembles the primitives developed in Chapters 2, 4, and 5 into a single
optimized execution pipeline for the Mixture-of-Experts (MoE) layers of Qwen3.5-35B on
T3K hardware. The goal is not merely to run the stages sequentially, but to overlap them
using micro-batch pipelining so that communication latency is hidden behind compute.

Earlier chapters treated each stage in isolation:

- Chapter 2 established the all-to-all dispatch and combine collectives over the T3K
  1×8 linear mesh (`ch02_all_to_all_primitives/all_to_all_dispatch.md`,
  `ch02_all_to_all_primitives/all_to_all_combine.md`).
- Chapter 4 showed how Qwen3.5-35B's 256 experts are partitioned uniformly across 8 devices,
  giving $E_d = 32$ local experts per device, and derived the per-expert weight footprint
  $W_{\text{expert}} = 6HD$ bytes in BF16
  (`ch04_expert_device_assignment/uniform_partitioning.md`).
- Chapter 5 introduced router kernel fusion and described double-buffering as a technique
  to pipeline dispatch with routing
  (`ch05_routing_weight_optimization/router_kernel_fusion.md`).

Chapter 6 wires these components together and answers three questions:

1. What is the precise dependency graph between the six pipeline stages?
2. How does micro-batch double-buffering reduce observed end-to-end latency?
3. What are the memory and numerical-precision constraints that bound the design?

## Prerequisites

| Prerequisite | Why it is needed |
|---|---|
| Chapter 2: All-to-All Primitives | Dispatch and combine volumes, link bandwidth model |
| Chapter 4: Expert Device Assignment | $E_d = 32$, $W_{\text{expert}}$, capacity factor $C$ |
| Chapter 5: Routing Weight Optimization | Router kernel fusion, double-buffer mechanics |

## Key Insight

A single MoE layer contains six sequential stages. The router (Stage 1) is a small,
fast projection that produces token-to-expert assignments. Its latency is typically much
less than the all-to-all dispatch plus expert FFN compute plus combine round-trip. By
splitting the incoming token batch $B$ into two micro-batches and staggering their
execution, the router's work on micro-batch $i+1$ overlaps with the dispatch, expert
compute, and combine phases of micro-batch $i$. This overlap converts $T_{\text{route}}$
from a serialized cost into a hidden cost — provided that double-buffered dispatch and
combine buffers are allocated so that the two micro-batches never share a buffer.

At decode batch sizes ($B \leq 32$), communication dominates. At large prefill batch
sizes, expert FFN compute dominates. The pipeline design presented in this chapter is
beneficial across both regimes.

## Why Fusing Matters

The word "fused" in the chapter title refers to two distinct but related ideas:

1. **Kernel-level fusion:** The combine all-to-all output and the routing weight
   accumulation (Stage 5) can be merged into a single kernel pass that reads expert
   outputs from the combine buffer, multiplies by normalized routing scores, and
   writes directly to the output (or residual) buffer. This avoids materializing a
   separate $[B, k, H]$ intermediate tensor in DRAM or L1.

2. **Stage-level pipelining:** Micro-batch double-buffering fuses the execution
   timelines of successive micro-batches so that the router (Stage 1) for batch $i+1$
   executes concurrently with Stages 2–5 for batch $i$. This is a software pipeline
   in the classical sense: stages are fused in time rather than in kernel code.

Together, these two fusions address the two primary inefficiencies in a naive serial
implementation: (a) intermediate tensor materialization that wastes L1 and DRAM
bandwidth, and (b) router idle time that adds pure latency with no overlap.

## Six-Stage Summary

The pipeline stages for one MoE layer are listed here for reference. Full details
are in `pipeline_design.md`.

| Stage | Name | Operation | Key Tensor Shapes |
|---|---|---|---|
| 1 | Route | Router projection + sigmoid + top-$k$ | In: $[B, H]$; Out: indices $[B, k]$, scores $[B, k]$ |
| 2 | Pack/Dispatch | Scatter into send buffer; all-to-all | Send: $[C \cdot E_d, H]$ per device |
| 3 | Expert FFN | Local matmul for $E_d = 32$ experts | Per expert: $[\text{rc}_e, H] \to [\text{rc}_e, H]$ |
| 4 | Combine | All-to-all (return outputs) | Recv: $[B, k, H]$ on originating device |
| 5 | Unpack+Accum | Weighted scatter-add | In: $[B, k, H]$; Out: $[B, H]$ |
| 6 | Residual Add | Output += residual stream | In/Out: $[B, H]$ |

At $B = 32$ and $C = 2$, the dispatch send buffer per destination device is
$[64, 7168] \times 2$ bytes $= 896$ KB. The double-buffer scheme requires two such
buffers per direction, totaling 7.0 MB across all buffers — well within the aggregate
120 MB L1 of 80 Wormhole B0 Tensix cores.

## Chapter Navigation

| File | Contents |
|---|---|
| [`pipeline_design.md`](./pipeline_design.md) | Six-stage dependency graph, micro-batch pipelining scheme, buffer sizing |
| [`expert_ffn_tiling.md`](./expert_ffn_tiling.md) | Parallel expert execution on 80 Tensix cores, L1 weight streaming, sparsity exploitation |
| [`combine_accumulation.md`](./combine_accumulation.md) | Weighted scatter-add algorithm, race condition avoidance, L1 vs. DRAM strategy |
| [`end_to_end_latency_model.md`](./end_to_end_latency_model.md) | Parameterized latency formula, regime identification, calibration procedure |

## Scope and Known Limitations

This chapter focuses on the single-device perspective: it describes what one device in
the T3K cluster does during each stage of a single MoE layer. All-to-all collectives
(Stages 2 and 4) are treated as black-box primitives whose latency is parameterized by
link bandwidth; for their internals see Chapter 2.

**Known gaps and unverified values:**

- The FFN intermediate dimension $D$ for Qwen3.5-35B is marked [UNVERIFIED] throughout
  this chapter. All latency estimates that depend on $D$ should be recalculated once
  the confirmed value is available.
- The Wormhole B0 sustained FLOP rate for BF16 matmul is not confirmed in this guide.
  The end-to-end latency model in `end_to_end_latency_model.md` provides a calibration
  procedure to measure it directly.
- The T3K 1×8 linear topology introduces hop penalties for non-adjacent device pairs.
  The single-link bandwidth model ($\text{BW} = 12.5$ GB/s) used here is a lower bound;
  actual dispatch/combine latency must be measured with the TTNN profiler to capture
  multi-hop effects.
- GDF (greedy deterministic fair) routing from Chapter 3 is mentioned in the latency
  model as an alternative that can reduce dispatch volume by approximately 23% relative
  to uniform routing at $N = 8$.

## Hardware and Model Constants (Quick Reference)

| Symbol | Value | Source |
|---|---|---|
| $E$ | 256 experts | Qwen3.5-35B architecture |
| $k$ | 8 experts per token | Qwen3.5-35B architecture |
| $N$ | 8 devices | T3K 1×8 linear mesh |
| $E_d$ | 32 experts per device | $E / N$ |
| $H$ | 7168 | Qwen3.5-35B hidden dimension |
| $f_{\text{avg}}$ | $1/32$ | $k/E$ |
| CF | 1.25 (default) | Capacity factor |
| $C(B{=}32)$ | 2 | $\lceil 32 \times 1.25 / 32 \rceil$ |
| Link bandwidth | 12.5 GB/s per Ethernet link | Wormhole B0 / T3K spec |
| Tensix cores per device | 80 | Wormhole B0 |
| L1 per core | 1.5 MB | Wormhole B0 |

## References

- Qwen3.5-35B model card and architecture specification.
- TT-Metalium TTNN documentation: `ttnn.Topology.Linear`, `cluster_axis=1`,
  `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`.
- Wormhole B0 Architecture Guide: Tensix core count (80), L1 per core (1.5 MB),
  aggregate L1 (120 MB), Ethernet link bandwidth (12.5 GB/s).
- Chapter 2 of this guide: All-to-All Primitives
  (`ch02_all_to_all_primitives/all_to_all_dispatch.md`,
  `ch02_all_to_all_primitives/all_to_all_combine.md`).
- Chapter 3 of this guide: Alternative Routing Schemes (GDF communication cost).
- Chapter 4 of this guide: Expert Device Assignment and Uniform Partitioning
  (`ch04_expert_device_assignment/uniform_partitioning.md`).
- Chapter 5 of this guide: Routing Weight Optimization and Router Kernel Fusion
  (`ch05_routing_weight_optimization/router_kernel_fusion.md`).
