# Chapter 4: Prefetch Patterns and Bandwidth

## Overview

This chapter examines how the physical layout of expert weight tensors in DRAM determines the efficiency of memory access during matmul execution. Two access patterns — interleaved and DRAM-sharded — produce measurably different effective bandwidths on Wormhole B0. Understanding the NoC hop model and the DRAM controller topology is prerequisite to selecting the correct `TensorMemoryLayout` for bandwidth-sensitive workloads.

The chapter builds directly on the shard layout API from Chapter 2 and the tensor shape analysis from Chapter 3. After completing this chapter you will be able to predict bandwidth behavior from layout choice alone and apply that prediction to the decode regime where expert matmuls are bandwidth-bound.

---

## Learning Objectives

1. **Describe the NoC packet model** and explain how destination coordinates and hop count affect DRAM read latency.
2. **Identify the 6 DRAM controller columns** on Wormhole B0 and explain why interleaved access distributes requests unevenly across NoC paths.
3. **Explain why DRAM-sharded layout eliminates NoC hotspots** by fixing the core-to-bank mapping.
4. **Apply double-buffering** to overlap DMA prefetch of the next shard with compute on the current shard.
5. **Use the roofline model** to determine whether a given expert matmul configuration is bandwidth-bound or compute-bound.
6. **Estimate effective bandwidth** from kernel timing and byte counts, and compare against the Wormhole B0 peak of ~300 GB/s.

---

## Prerequisites

| Chapter | Topics Covered |
|---|---|
| Chapter 1 | TTNN memory architecture, TILE_LAYOUT, tensor memory hierarchy |
| Chapter 2 | ShardSpec, shard shape arithmetic, CoreRangeSet construction |
| Chapter 3 | Expert weight projection shapes, per-expert memory footprint |

All three chapters must be completed before proceeding. Familiarity with GDDR6 burst access semantics is helpful but not required.

---

## Conceptual Diagram: Interleaved vs. Sharded DRAM Access

The diagram below shows how 4 Tensix cores fetch weight tiles under each layout during a matmul kernel. DRAM controllers are labeled D0–D5 (columns in the NoC grid).

```
INTERLEAVED ACCESS (round-robin across all 6 DRAM columns)
--------------------------------------------------------------
Core(0,0) ──► D0 ──── 1 hop
              D1 ──── 2 hops   <- same core issues 6 different requests
              D2 ──── 3 hops      each to a different DRAM column
              D3 ──── 4 hops
              D4 ──── 5 hops
              D5 ──── 6 hops

Core(1,0) ──► D0 ──── 1 hop   <- overlapping traffic on D0 path
              D1 ──── 2 hops      creates NoC hotspot
              ...

    Result: variable latency, NoC link contention, bandwidth loss 20-40%

DRAM-SHARDED ACCESS (each core owns one DRAM bank range)
--------------------------------------------------------------
Core(0,0) ──► D0 ──── 1 hop (always)
Core(1,0) ──► D1 ──── 1 hop (always)
Core(2,0) ──► D2 ──── 1 hop (always)
Core(3,0) ──► D3 ──── 1 hop (always)
Core(4,0) ──► D4 ──── 1 hop (always)
Core(5,0) ──► D5 ──── 1 hop (always)

    Result: deterministic short path, no hotspots, near-peak bandwidth
```

---

## Chapter Structure

| File | Contents |
|---|---|
| [`noc_and_dram_access.md`](./noc_and_dram_access.md) | NoC packet model; DRAM controller topology; interleaved access hotspots |
| [`sharded_access_pattern.md`](./sharded_access_pattern.md) | DRAM-sharded layout; shard orientation; double-buffering interaction |
| [`bandwidth_estimation.md`](./bandwidth_estimation.md) | Effective bandwidth model; roofline analysis; decode regime arithmetic intensity |

---

## Key Constants (Wormhole B0)

| Parameter | Value |
|---|---|
| DRAM controllers | 6 |
| GDDR6 banks | 12 (2 per controller) |
| Peak DRAM bandwidth | ~300 GB/s |
| Tensix cores | 80 (8×10 grid) |
| L1 per core | 1.5 MB |
| BF16 tile size | 2,048 bytes (32×32×2) |
| Peak compute | ~131 TFLOP/s (BF16) |
| Ridge point | ~437 FLOP/byte |
