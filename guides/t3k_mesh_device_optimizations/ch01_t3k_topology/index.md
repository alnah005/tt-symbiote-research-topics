# Chapter 1 — T3K Hardware Topology and Interconnect Fundamentals

This chapter establishes the physical and logical layout of the T3K system, characterizes the chip-to-chip Ethernet interconnect that binds its eight Wormhole devices into a programmable mesh, and derives the topology constraints that govern every collective communication decision made in later chapters. Read this chapter before proceeding to Chapter 2 or any later material; the coordinate system, bandwidth figures, and ring-collective analysis on the linear 1x8 mesh introduced here are assumed knowledge throughout the rest of the guide.

---

## What This Chapter Covers

**T3K physical layout** ([`t3k_physical_layout.md`](t3k_physical_layout.md))
Establishes device numbering, the `(1,8)` logical mesh, and neighbor adjacency wiring.

**Ethernet link bandwidth** ([`ethernet_link_bandwidth.md`](ethernet_link_bandwidth.md))
Quantifies per-link and aggregate bandwidth, link latency, multi-hop routing cost, and link saturation behavior.

**Topology implications for collectives** ([`topology_implications_for_collectives.md`](topology_implications_for_collectives.md))
Derives why ring-based collectives fit the linear 1x8 mesh, with hop count tables for ring, tree, and direct-connect variants, and introduces `num_links` as the primary bandwidth tuning knob.

---

## Prerequisites

This chapter has no dependencies on other chapters in this guide. However, readers are expected to arrive with the following background knowledge:

- **General LLM inference concepts**: familiarity with the prefill (prompt-processing) and decode (token-generation) phases, KV cache structure, and the distinction between compute-bound and memory-bandwidth-bound operation.
- **Distributed inference concepts**: conceptual understanding of tensor parallelism, pipeline parallelism, and collective operations (all-reduce, all-gather, scatter). You do not need to know their TTNN API forms; those are covered in Chapter 2.
- **Basic Tenstorrent TTNN API**: tensor creation, device placement, and operation dispatch at a beginner level. You do not need experience with multi-device or `MeshDevice` APIs.
- **Mixture-of-Experts architecture**: understanding of the router, expert selection, and top-K gating mechanism at a conceptual level. Qwen3MoE-specific details appear later; general MoE intuition is sufficient here.

**External documentation pointers:**
- Tenstorrent Wormhole architecture documentation (internal): covers Tensix core layout, NOC topology, L1 and DRAM memory system. Recommended background for Chapter 4.
- TTNN Python API reference: the `ttnn` module docstrings and the `tt-metal` repository README.
- T3K hardware bring-up guide: physical connector layout, power requirements, and host PCIe attachment — not required for software optimization but useful context.

---

## Reading Order

For readers new to T3K, read the three files in the order listed:

1. [`t3k_physical_layout.md`](t3k_physical_layout.md) — establishes the coordinate system and device adjacency structure that all subsequent files reference.
2. [`ethernet_link_bandwidth.md`](ethernet_link_bandwidth.md) — quantifies the bandwidth and latency characteristics of the links described in the physical layout file.
3. [`topology_implications_for_collectives.md`](topology_implications_for_collectives.md) — applies the hardware facts from the first two files to the communication patterns used by Mixture-of-Experts (MoE) inference workloads.

Readers already familiar with the T3K board layout may skip [`t3k_physical_layout.md`](t3k_physical_layout.md) and begin at [`ethernet_link_bandwidth.md`](ethernet_link_bandwidth.md), but should verify that their mental model of device IDs 0–7 and the `(1, 8)` logical mesh shape matches the conventions used here, since Chapter 2 assumes them precisely.

---

## Key Concepts Introduced in This Chapter

| Concept | Introduced In | First Used Outside Ch 1 |
|---|---|---|
| T3K logical mesh shape `(1, 8)` | `t3k_physical_layout.md` | Ch 2: `MeshDevice` constructor |
| Device IDs 0–7 and `(row, col)` coordinates | `t3k_physical_layout.md` | Ch 2: tensor distribution |
| Intra-board vs. inter-board links | `t3k_physical_layout.md` | Ch 7: reference configuration |
| Per-link unidirectional bandwidth (GB/s) | `ethernet_link_bandwidth.md` | Ch 3: `num_links` tuning |
| Aggregate bidirectional bandwidth | `ethernet_link_bandwidth.md` | Ch 3: `num_links` tuning |
| Link latency (µs) | `ethernet_link_bandwidth.md` | Ch 3: decode all-to-all analysis |
| Multi-hop routing cost | `ethernet_link_bandwidth.md` | Ch 5: expert placement strategies |
| Link saturation threshold | `ethernet_link_bandwidth.md` | Ch 6: device performance counters |
| Ring collective on 1x8 mesh | `topology_implications_for_collectives.md` | Ch 3: all-to-all in MoE |
| Hop count analysis | `topology_implications_for_collectives.md` | Ch 5: expert placement |
| `num_links` as a tunable parameter | `ethernet_link_bandwidth.md` | Ch 3: `num_links_parameter.md` |
| `cluster_axis=1` (T3K coordinate-axis parameter for single-board collective operations) | Previewed here (Ch 1, `t3k_physical_layout.md`); formally defined in Ch 2 (`collective_primitives.md`) | Ch 2: `collective_primitives.md` — first applied use in collective call signatures |
