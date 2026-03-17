# Chapter 2 — TTNN MeshDevice API for Multi-Chip Operations

This chapter introduces the TTNN abstractions for managing a multi-device T3K system in software: how to create and tear down a `MeshDevice`, how to distribute tensors across eight Wormhole chips, and how to invoke the collective communication primitives that move data between them. Every API call and configuration decision described here is grounded in the topology established by Chapter 1. Read Chapter 1 before reading this chapter.

---

## What This Chapter Covers

**MeshDevice setup** ([`mesh_device_setup.md`](mesh_device_setup.md))
Explains the `MeshDevice` constructor parameters, device ordering conventions, initialization and teardown sequences, and common pitfalls.

**Tensor distribution** ([`tensor_distribution.md`](tensor_distribution.md))
Describes `TensorSpec` and `ShardSpec`, the row-wise and column-wise sharding strategies, the distinction between replicated and sharded tensors, and the API patterns for placing tensors on a mesh target.

**Collective primitives** ([`collective_primitives.md`](collective_primitives.md))
Documents `ttnn.all_to_all`, `ttnn.all_reduce`, `ttnn.reduce_scatter`, and `ttnn.all_gather`, including their signatures, key parameters, synchronization semantics, and error handling.

---

## Prerequisites

- **Chapter 1 — T3K Hardware Topology and Interconnect Fundamentals** must be read in full before this chapter. Specifically, this chapter assumes you are already familiar with:
  - The T3K `(1, 8)` logical mesh shape and device IDs 0–7 (`t3k_physical_layout.md`)
  - The `(row, col)` coordinate system and how logical coordinates map to physical chip positions (`t3k_physical_layout.md`)
  - The `cluster_axis` parameter convention: `cluster_axis=1` for single-board T3K (see `collective_primitives.md` for details)
  - Per-link Ethernet bandwidth of ~12.5 GB/s and the `num_links` concept (`ethernet_link_bandwidth.md`)
  - Use `ttnn.Topology.Linear` (not `Topology.Ring`) on T3K (see `collective_primitives.md` for the full rationale)

  If any of those concepts are unfamiliar, start with Chapter 1.

---

## TTNN Multi-Device Concepts Introduced in This Chapter

The table below summarizes the new TTNN concepts this chapter introduces and where each first appears in later chapters.

| Concept | Introduced In | First Used Outside Ch 2 |
|---|---|---|
| `MeshDevice` constructor and parameters | `mesh_device_setup.md` | Ch 5: expert weight placement |
| Device ID ordering in `MeshDevice` | `mesh_device_setup.md` | Ch 5: expert placement strategies |
| `MeshDevice` teardown pattern | `mesh_device_setup.md` | Ch 7: reference configuration |
| `TensorSpec` and `ShardSpec` | `tensor_distribution.md` | Ch 4: memory configuration API |
| Row-wise vs. column-wise sharding | `tensor_distribution.md` | Ch 5: expert weight placement |
| Replicated vs. sharded tensors | `tensor_distribution.md` | Ch 4: decode memory strategy |
| `ttnn.from_torch` with mesh placement | `tensor_distribution.md` | Ch 5: token routing and dispatch |
| `ttnn.to_device` for mesh targets | `tensor_distribution.md` | Ch 5: token routing and dispatch |
| `ttnn.all_to_all` signature and parameters | `collective_primitives.md` | Ch 3: all-to-all in MoE |
| `ttnn.all_reduce` | `collective_primitives.md` | Ch 7: reference configuration |
| `ttnn.reduce_scatter` and `ttnn.all_gather` | `collective_primitives.md` | Ch 5: combine and accumulation |
| `cluster_axis` as a collective parameter | `collective_primitives.md` | Ch 3: all-to-all in MoE |
| Blocking vs. async dispatch semantics | `collective_primitives.md` | Ch 6: bottleneck diagnosis |

---

## Reading Order

For readers new to TTNN multi-device programming, read the three files in the order listed:

1. [`mesh_device_setup.md`](mesh_device_setup.md) — establishes the `MeshDevice` object that all subsequent operations target. You cannot distribute tensors or invoke collectives until a `MeshDevice` is initialized, so this is the natural starting point.
2. [`tensor_distribution.md`](tensor_distribution.md) — describes how to place data on the mesh. Understanding sharding and replication strategies is necessary before the collective operations in the next file make sense.
3. [`collective_primitives.md`](collective_primitives.md) — covers the operations that move data between devices. These are the operations that drive the Ethernet links described in Chapter 1 and form the communication backbone of MoE expert parallelism.

Readers who have previously used `MeshDevice` and are already comfortable with tensor placement may skip to [`collective_primitives.md`](collective_primitives.md) directly, but should check that their understanding of `TensorSpec` and `ShardSpec` terminology matches the conventions in [`tensor_distribution.md`](tensor_distribution.md), as those terms appear in the collective API signatures.

---

## Key Terminology

The following terms are used throughout this chapter with specific meanings. They are defined in detail in the files listed, but a brief preview reduces ambiguity when reading the file descriptions above.

**`MeshDevice`** — The TTNN Python object representing the full set of T3K Wormhole chips as a single programmable unit. It maps logical `(row, col)` coordinates to physical device IDs and manages the dispatch queues, firmware state, and memory contexts for all devices in the mesh.

**Shard** — A contiguous slice of a tensor assigned to one device in the mesh. A tensor with eight shards, one per device, is said to be "fully sharded" across the mesh.

**Replicated tensor** — A tensor where the same data exists on every device. No sharding occurs; every device holds a complete copy. Common for small tensors (routing scores, layer norms, biases) that are used identically by every device's compute kernel.

**`cluster_axis`** — An integer (0 or 1) that selects which axis of the mesh a collective traverses. `cluster_axis=1` for T3K (see `collective_primitives.md` for details).

**`num_links`** — An integer (1–4) controlling how many Ethernet links between adjacent device pairs are allocated to a collective operation. Introduced conceptually in Chapter 1; formally used in collective call signatures defined in `collective_primitives.md`.
