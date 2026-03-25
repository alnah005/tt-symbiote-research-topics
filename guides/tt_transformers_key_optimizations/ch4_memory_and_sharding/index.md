# Chapter 4 — Memory and Sharding

This chapter covers the memory hierarchy and sharding strategies that make LLM inference viable on Tensix hardware. Efficient inference on Tenstorrent devices depends on keeping the right tensors in the right memory tier at every point in the computation — and on choosing sharding patterns that let consecutive ops pass data through L1 without unnecessary DRAM round-trips. The two content files in this chapter establish the full hardware memory picture and then map the four L1 sharding modes onto the transformer layers where each is applied.

---

## Key Concepts at a Glance

| Concept | What it is | Where it matters |
|---|---|---|
| L1 SRAM (120 KB/core) | Per-core fast local memory; holds active tiles during compute | All sharding strategies; sets the tile-size ceiling |
| `packer_l1_acc` | Packer accumulates into L1 output buffer instead of read-modify-write to DRAM | Multi-tile matmul K-loop; eliminates one DRAM read per K-block |
| DRAM (per-chip) | Shared off-chip storage; weights, KV cache, large activations | Weight streaming; paged KV layout; activation spill |
| NoC (Network-on-Chip) | On-chip interconnect between Tensix cores and DRAM banks | Multicast of weight tiles; DRAM-sharded parallel bank reads |
| Ethernet (multi-chip) | On-board chip-to-chip links (N300: 2-chip, T3K: 8-chip ring) | `all_gather`, `reduce_scatter`, `all_reduce` for tensor parallelism |
| Height sharding | Split batch × sequence rows across cores | Attention prefill; 1D decode matmul activation layout |
| Block sharding | 2D tile distribution across core grid rows and columns | Large intermediate activations that exceed per-core L1 |
| DRAM-sharded | Weight columns distributed across DRAM banks; each core owns its bank | Decode linear layers (QKV, MLP FF1/FF2, output projection) |
| 1D ring-sharded | Input sharded along K dimension; partial dot-products reduced | Large-K reductions parallelized across cores |

---

## Reading Order

1. **`tensix_memory_hierarchy.md`** — Start here to understand L1, DRAM, NoC, and Ethernet as distinct tiers. The sizing arithmetic for L1 attention buffers and the `packer_l1_acc` mechanism are introduced here and assumed throughout the rest of the guide.

2. **`sharding_patterns.md`** — Applies the memory hierarchy to the four sharding modes: height, block, DRAM-sharded width, and 1D ring. Each mode is mapped to its corresponding `MatmulMultiCoreReuse*ProgramConfig` class and the transformer layers it serves.

---

## Navigation

Previous: [Chapter 3 — MatMul Optimizations](../ch3_matmul_optimizations/index.md)

Next: [Chapter 5 — Multi-Device Scaling](../ch5_multi_device_scaling/index.md) *(forthcoming)*
