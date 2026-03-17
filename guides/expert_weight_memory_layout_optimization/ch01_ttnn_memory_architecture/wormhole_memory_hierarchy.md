# Wormhole B0 Memory Hierarchy

## Physical DRAM Topology

Wormhole B0 has six DRAM controllers, each managing two GDDR6 banks. Every bank holds 1 GB, giving a total of 12 GB of device DRAM across all 12 banks. This is not a flat address space from the perspective of access cost: each controller has its own NoC-visible tile address, and different controllers are reached by different numbers of NoC hops from any given Tensix core.

```
Wormhole B0 DRAM layout (logical):

  Controller 0 ─── Bank 0 (1 GB)
               └── Bank 1 (1 GB)

  Controller 1 ─── Bank 2 (1 GB)
               └── Bank 3 (1 GB)

  Controller 2 ─── Bank 4 (1 GB)
               └── Bank 5 (1 GB)

  Controller 3 ─── Bank 6 (1 GB)
               └── Bank 7 (1 GB)

  Controller 4 ─── Bank 8 (1 GB)
               └── Bank 9 (1 GB)

  Controller 5 ─── Bank 10 (1 GB)
               └── Bank 11 (1 GB)

Total: 12 banks × 1 GB = 12 GB
```

When TTNN allocates a tensor with `ttnn.DRAM_MEMORY_CONFIG`, it distributes the tensor's pages in round-robin order across all 12 banks. This is the interleaved allocation strategy. The consequence is that any single Tensix core reading a large tensor will issue reads to all 12 DRAM banks, traversing whatever NoC distance is required to reach each controller.

---

## L1 SRAM Per Tensix Core

Each Tensix core contains 1.5 MB of L1 SRAM. This memory is shared across the five RISC-V sub-cores that make up a Tensix tile (BRISC, NCRISC, TRISC0, TRISC1, TRISC2). In practice, the memory is logically divided between:

- **Circular buffers (CBs):** input and output staging areas used by the data movement sub-cores (BRISC and NCRISC) to transfer data between DRAM/NoC and the compute pipeline
- **Compute scratch space:** working memory for the arithmetic sub-cores (TRISC0, TRISC1, TRISC2)

The 1.5 MB budget must cover all of these simultaneously. For a matmul kernel, this typically means the weight tile and the activation tile must both fit within the CB allocation for that core. This constraint is the primary reason large expert weight tensors cannot simply be moved wholesale into L1: the full weight matrix of a single expert in a large MoE model (e.g., `[4096, 14336]` in bfloat16) requires 4096 × 14336 × 2 bytes ≈ 117 MB, far exceeding per-core L1.

### Total On-Chip L1

The full Wormhole B0 Tensix grid is 8 columns × 10 rows = 80 Tensix cores. At 1.5 MB each, the aggregate L1 capacity is:

```
80 cores × 1.5 MB = 120 MB nominal
```

In practice, harvested rows (cores disabled at the factory due to yield) reduce the available core count. A production Wormhole card typically exposes a 7×10, 8×9, or similar usable grid after harvesting, giving approximately 105–120 MB of usable on-chip L1. This is the total budget shared across all concurrent tensors and kernel buffers.

> **Warning:** Do not assume all 80 cores are available. Always query the device's compute grid at runtime with `device.compute_with_storage_grid_size()` rather than hardcoding grid dimensions.

---

## DRAM Bandwidth vs L1 Bandwidth

The bandwidth figures below are approximate and represent sustained throughput under favorable access patterns (sequential, aligned, full-width bursts). Real throughput under weight-tensor access patterns will be lower due to fragmentation, NoC contention, and kernel scheduling overhead.

| Memory tier | Approximate peak bandwidth | Notes |
|---|---|---|
| GDDR6 (single controller) | ~50 GB/s | 6 controllers total, 2 banks each; aggregate theoretical ~300–320 GB/s |
| GDDR6 (all controllers, ideal) | ~300–320 GB/s | Achieved only with perfectly parallel, balanced access across all 12 banks |
| L1 SRAM (single core) | ~1 TB/s | Local read; no NoC traversal required |
| NoC link (single hop) | ~100–150 GB/s | Bidirectional; shared across all cores transmitting on that link |

The gap between L1 bandwidth (~1 TB/s) and DRAM bandwidth (~300 GB/s aggregate) is roughly 3x at best and grows wider under contention. For weight-heavy inference — where the same expert weight matrix is read once per forward pass but must supply many Tensix cores simultaneously — DRAM bandwidth is nearly always the bottleneck, not arithmetic throughput.

This is why shard strategy matters: an interleaved layout forces every core to read from every DRAM controller over a shared NoC, saturating the links. A sharded layout gives each bank a dedicated subset of the tensor, enabling parallel prefetch without cross-controller contention.

---

## The Network-on-Chip (NoC)

The NoC is the on-chip interconnect that allows Tensix cores to issue read and write requests to DRAM tiles, to L1 in remote cores, and to other on-chip resources (Ethernet, PCIe). Wormhole uses a 2D mesh NoC topology where each tile (Tensix core, DRAM tile, or Ethernet tile) is a node.

### NoC Hop Model

A "hop" is a single link traversal in the mesh. Each hop adds latency. The latency per hop is on the order of a few nanoseconds; the number of hops between a Tensix core and a DRAM controller tile depends on physical placement in the mesh.

Key properties of the Wormhole NoC relevant to weight access:

1. **Shared links:** The NoC links between mesh nodes are shared. If multiple Tensix cores simultaneously issue reads to DRAM tiles that route through the same link segment, they compete for bandwidth on that link.

2. **Distance matters:** A Tensix core in the center of the grid has shorter average path lengths to all DRAM controller tiles than a core at the edge. Under interleaved allocation, all 12 DRAM banks are accessed for every tensor read, and the cores with longer average paths pay a higher latency per request.

3. **Pipelining:** The NoC is pipelined — multiple in-flight requests can be outstanding simultaneously. The kernel's data movement sub-cores (BRISC and NCRISC) are responsible for issuing these pre-fetch requests to overlap data transfer with compute.

4. **Multicast:** The NoC supports multicast writes, where a single write is fanned out to multiple destination cores. This is used in weight broadcast patterns (one DRAM→many L1 cores) and is more efficient than issuing one unicast write per destination.

### Practical Impact on Weight Reads

For a weight tensor in interleaved DRAM, a single matmul kernel reading the full tensor will issue reads distributed across all 12 DRAM banks. If 8 Tensix cores are running the same matmul simultaneously (as in a batched expert dispatch), each of those 8 cores issues its own set of reads to all 12 banks. The NoC links near the DRAM tiles receive up to 8× the single-core traffic, creating a contention bottleneck.

The resolution — discussed in `interleaved_vs_sharded.md` and the following chapters — is to shard the weight tensor so that each DRAM bank owns a disjoint subset of the weight tiles, and each Tensix core reads only from its assigned bank. This converts a many-to-many NoC traffic pattern into a one-to-one pattern, eliminating contention.

---

## Next Steps

Proceed to `memory_config_api.md` to learn how to express DRAM placement, L1 placement, and sharding directives using the `ttnn.MemoryConfig` API. The bandwidth and hop-count concepts introduced here will be referenced when evaluating which `TensorMemoryLayout` value is appropriate for a given weight tensor shape.
