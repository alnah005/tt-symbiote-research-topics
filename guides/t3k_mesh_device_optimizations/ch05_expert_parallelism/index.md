# Chapter 5: Expert Parallelism on T3K — Mapping Experts to Devices

> **Quick Reference — No New TTNN API Symbols Introduced in This Index**
>
> This index file introduces no new `ttnn` API symbols. All API symbols used in this chapter (`ttnn.all_to_all`, `ttnn.Topology.Linear`, `MeshDevice`) are defined in earlier chapters. See `ch02_ttnn_mesh_api/collective_primitives.md` and `ch03_all_to_all_num_links/num_links_parameter.md`.

## Prerequisites

This chapter assumes familiarity with all four preceding chapters of this guide:

- **Chapter 1** (`ch01_t3k_topology/`) — T3K physical layout: the 1×8 linear mesh, device IDs 0–7, `ttnn.Topology.Linear`, Ethernet link bandwidth (~12.5 GB/s per link), and the average hop count of 3.0 across all device pairs.
- **Chapter 2** (`ch02_ttnn_mesh_api/`) — The `ttnn` mesh API: `MeshDevice`, `MeshTensor`, distributed sharding primitives, and the `ttnn.all_to_all` collective as described in `tensor_distribution.md` and `collective_primitives.md`.
- **Chapter 3** (`ch03_all_to_all_num_links/`) — All-to-all operation mechanics for Mixture-of-Experts (MoE) dispatch and combine, the `num_links` parameter, and how payload size determines optimal link count; see `all_to_all_in_moe.md` and `num_links_parameter.md`.
- **Chapter 4** (`ch04_memory_config/`) — Wormhole B0 memory hierarchy, L1 vs. DRAM tensor placement decisions, and concrete buffer size recommendations for decode and prefill; see `decode_memory_strategy.md`, `prefill_memory_strategy.md`, and `wormhole_memory_hierarchy.md`.

Readers who have not completed Chapters 1–4 will encounter undefined terms and forward references. Return here after completing them.

---

## Goal

This chapter synthesizes the topology knowledge from Chapter 1, the collective API from Chapter 2, the bandwidth model from Chapter 3, and the memory hierarchy from Chapter 4 into a practical guide for **expert parallelism**: how to assign 256 experts across 8 T3K devices, route token embeddings to the correct devices, and accumulate expert outputs back at the originating device.

Specific goals:

1. Explain and compare four expert placement strategies — naive uniform, load-balanced, locality-aware, and expert replication — with quantitative criteria for choosing among them.
2. Derive the token dispatch flow from router output through all-to-all to expert compute, including send buffer construction and capacity padding.
3. Describe the reverse all-to-all (combine phase) and the weighted accumulation of $k = 8$ expert outputs per token.
4. Identify overlap opportunities between communication and compute in both decode and prefill phases.

---

## Chapter Overview

### How the Prior Chapters Connect Here

Expert parallelism on T3K is the convergence of four concerns:

| Concern | Source Chapter | Relevance Here |
|---|---|---|
| Topology: hop counts and link bandwidth | Ch01 `t3k_physical_layout.md` | Determines all-to-all latency; locality-aware placement minimizes average hop count |
| Collective API: `ttnn.all_to_all` | Ch02 `collective_primitives.md` | The dispatch and combine operations are both `ttnn.all_to_all` calls |
| Bandwidth and `num_links` | Ch03 `num_links_parameter.md` | Payload size at each batch size determines whether `num_links=1` or `num_links=2` is optimal |
| Memory: L1 vs. DRAM for buffers | Ch04 `decode_memory_strategy.md` | Send and receive buffers must fit in L1 for decode; DRAM is required for prefill |

No single chapter covers all four; this chapter is the integration point.

### Model and Hardware Constants

All quantitative analysis in this chapter uses Qwen3.5-35B parameters on an 8-device T3K mesh.

| Symbol | Meaning | Value |
|---|---|---|
| $E$ | Total experts | 256 |
| $k$ | Top-$k$ experts per token | 8 |
| $N$ | T3K devices | 8 |
| $E_d$ | Experts per device ($E/N$) | 32 |
| $H$ | Hidden dimension | 7168 |
| $f_{\text{avg}}$ | Average per-expert routing frequency | $k/E = 1/32$ |
| $C$ | Expert capacity per dispatch step | $\lceil k \cdot B \cdot \text{CF} / E \rceil = \lceil B \cdot 1.25 / 32 \rceil$ |
| CF | Capacity factor | 1.25 |

---

## Navigation

| File | Description |
|---|---|
| [`expert_placement_strategies.md`](./expert_placement_strategies.md) | Four placement strategies — naive uniform, load-balanced, locality-aware, and expert replication — with memory and load-balance analysis for each |
| [`token_routing_and_dispatch.md`](./token_routing_and_dispatch.md) | On-device router output, send buffer construction, capacity padding, all-to-all dispatch mechanics, and latency estimates by batch size |
| [`combine_and_accumulation.md`](./combine_and_accumulation.md) | Reverse all-to-all structure, weighted combination of $k=8$ expert outputs, in-place accumulation strategies, and overlap opportunities |

### Recommended Reading Order

1. Start with [`expert_placement_strategies.md`](./expert_placement_strategies.md) to decide on an assignment strategy before implementing dispatch logic. The placement decision affects the dispatch-to-device mapping and the need for replication metadata.
2. Read [`token_routing_and_dispatch.md`](./token_routing_and_dispatch.md) to implement the dispatch path: router integration, send buffer construction, and the `ttnn.all_to_all` dispatch call with the correct `memory_config` for your phase (decode vs. prefill).
3. Read [`combine_and_accumulation.md`](./combine_and_accumulation.md) last, as it depends on the dispatch metadata structures introduced in step 2.

---

## When to Read Which File

| Scenario | Recommended File |
|---|---|
| Choosing expert assignment for a new deployment | [`expert_placement_strategies.md`](./expert_placement_strategies.md) |
| Debugging token drop events at runtime | [`token_routing_and_dispatch.md`](./token_routing_and_dispatch.md) Section 3 |
| Implementing or tuning the router dispatch kernel | [`token_routing_and_dispatch.md`](./token_routing_and_dispatch.md) Section 5 |
| Verifying numerical accuracy of combine output | [`combine_and_accumulation.md`](./combine_and_accumulation.md) Section 4 |
| Reducing prefill MoE layer latency through overlap | [`combine_and_accumulation.md`](./combine_and_accumulation.md) Section 5 |
| Understanding `num_links` interaction with batch size | [`token_routing_and_dispatch.md`](./token_routing_and_dispatch.md) Section 4, then [`ch03_all_to_all_num_links/num_links_parameter.md`](../ch03_all_to_all_num_links/num_links_parameter.md) |

---

## MoE Forward Pass: Full Data Flow

The complete MoE layer on T3K proceeds as follows. Each step references the file that covers it in detail.

```
[All devices: hidden states [B, H]]
        │
        ▼
  Router matmul + top-k                ← token_routing_and_dispatch.md §1
        │
        ▼
  Send buffer packing [N, C*E_d, H]   ← token_routing_and_dispatch.md §2
        │
        ▼
  ttnn.all_to_all (dispatch)           ← token_routing_and_dispatch.md §4–5
        │
        ▼
  Local expert FFN compute             ← (expert weight placement: expert_placement_strategies.md §1–4)
        │
        ▼
  ttnn.all_to_all (combine)            ← combine_and_accumulation.md §1
        │
        ▼
  Weighted accumulation [B, H]         ← combine_and_accumulation.md §2–3
        │
        ▼
  Residual add + layer norm            ← (next layer)
```

The two `ttnn.all_to_all` calls (dispatch and combine) each transfer approximately 3.2–6.4 MB per device at $B=1$–$32$ decode, and up to 939.5 MB per device at $B=32, S=2048$ prefill. Memory placement (L1 vs. DRAM) follows the rules in Chapter 4 (`ch04_memory_config/`).

---

## Expert Placement and Communication Summary

For the four placement strategies (naive uniform, load-balanced, locality-aware, expert replication) with full derivations, formulas, and worked examples, see `expert_placement_strategies.md`.

For dispatch volume calculations, capacity formula, and per-$B$ byte totals, see `token_routing_and_dispatch.md` Section 4.

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — T3K 1×8 mesh topology, device IDs 0–7, hop-count derivation
- `ch01_t3k_topology/topology_implications_for_collectives.md` — Impact of linear mesh topology on collective latency
- `ch02_ttnn_mesh_api/tensor_distribution.md` — Distributed tensor sharding on `MeshDevice`
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API reference
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Dispatch/combine data volume derivations
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` bandwidth model and tuning
- `ch04_memory_config/decode_memory_strategy.md` — L1 placement for decode-phase buffers
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement for prefill-phase all-to-all buffers
- `ch04_memory_config/wormhole_memory_hierarchy.md` — Wormhole B0 memory capacities and bandwidths
- `expert_parallelism_strategies/ch04_expert_device_assignment/uniform_partitioning.md` — $W_{\text{expert}} = 6HD$ bytes definition (Section 2)
- `expert_parallelism_strategies/ch04_expert_device_assignment/expert_replication.md` — Replication factor formula and dispatch metadata
