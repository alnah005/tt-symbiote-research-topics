# Wormhole Hardware Specs — Single Chip in T3K

> **Source note:** Wormhole chip in T3K — half of n300, 64 Tensix cores, 288 GB/s DRAM bandwidth.

Each chip in a T3K system is one Wormhole ASIC, corresponding to half of one n300 card. A T3K system contains 4×n300 cards = **8 chips** arranged in a 1×8 linear mesh.

## Per-Chip Compute and Memory

| Parameter | Value |
|---|---|
| Tensix cores | 64 per chip |
| L1 SRAM per core | 1.5 MB |
| L1 SRAM aggregate (per chip) | 64 × 1.5 MB = **96 MB** |
| DRAM | 12 GB GDDR6 per chip |
| DRAM bandwidth | **288 GB/s** per chip |
| BF16 peak compute | **131 TFLOPS** per chip |

## Roofline Ridge Point

The ridge point is the arithmetic intensity at which a kernel transitions from memory-bandwidth-bound to compute-bound:

```
Ridge point = Peak compute / Peak memory bandwidth
            = 131 × 10^12 FLOP/s / 288 × 10^9 bytes/s
            = 454.9 FLOP/byte
            ≈ 455 FLOP/byte
```

Any kernel with arithmetic intensity **below 455 FLOP/byte** is memory-bandwidth-bound on a single Wormhole chip. Any kernel **above 455 FLOP/byte** is compute-bound.

## Inter-Chip Interconnect

| Link | Bandwidth |
|---|---|
| Die-to-die (same n300 card) | 200 Gbps = 25 GB/s |
| QSFP-DD (inter-card) | 200 Gbps = 25 GB/s per port |
| Effective all-gather bandwidth (empirical, 8 devices) | ~25 GB/s |

The T3K mesh is a **1×8 linear topology**. All-gather operations traverse this linear chain; the empirical effective bandwidth for collective communication across all 8 devices is approximately **~25 GB/s** — roughly 11.5× slower than per-chip DRAM bandwidth. This disparity makes inter-chip communication a critical bottleneck for any sharding strategy that requires synchronizing state tensors across devices.

## Memory Hierarchy Summary

| Level | Capacity | Bandwidth |
|---|---|---|
| Register (FPU) | — | ~10+ TB/s (per-core theoretical) |
| L1 SRAM | 1.5 MB/core, 96 MB aggregate per chip | ~10 TB/s aggregate per chip |
| DRAM (GDDR6) | 12 GB per chip | **288 GB/s** per chip |
| Ethernet CCL (inter-chip) | — | ~25 GB/s effective |

The L1 SRAM is the critical fast tier: data that fits in L1 and remains resident across a fused kernel avoids DRAM round-trips entirely. Chapter 5's roofline analysis identifies the conditions under which the Gated Delta Net state matrix can be kept L1-resident.

---

**Next:** [`roofline_decode_and_prefill.md`](./roofline_decode_and_prefill.md)
