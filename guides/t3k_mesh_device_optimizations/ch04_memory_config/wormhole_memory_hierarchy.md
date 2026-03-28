# Wormhole Memory Hierarchy: L1 SRAM and DRAM

## Overview

Every Wormhole B0 chip exposes two physically distinct memory subsystems to TTNN kernels:

- **L1 SRAM** — fast, private, per-core scratchpad memory
- **DRAM** — large, shared, high-bandwidth off-core storage

Understanding the capacity and performance characteristics of each is a prerequisite for making correct memory placement decisions. This file covers both levels in detail.

---

## L1 SRAM

### Capacity and Core Topology

Each Wormhole B0 chip contains an **8×10 grid of Tensix cores** for a total of **80 compute cores per chip**. Each Tensix core has its own dedicated **L1 SRAM of 1.5 MB**.

$$L1_{\text{chip}} = n_{\text{cores}} \times L1_{\text{core}} = 80 \times 1.5\,\text{MB} = 120\,\text{MB}$$

The aggregate figure of 120 MB is the theoretical maximum if every core's L1 is fully utilized. In practice, kernel CB allocations use only a fraction of this at any given moment.

> **Warning:** L1 is **private to each core**. It is not a shared cache or a unified buffer pool. Core A cannot directly read from Core B's L1 without an explicit Network-on-Chip (NoC) transfer. The 120 MB aggregate is only meaningful when a tensor is explicitly sharded so that each core holds its own slice. The per-core limit of 1.5 MB is the binding constraint for any individual kernel.

### Access Latency

A Tensix core reads its own L1 in approximately **1 cycle**. Reads via the NoC from a neighboring core's L1 are substantially higher. DRAM accesses involve ~100+ cycles of latency due to DRAM bus arbitration and row activation. The exact latency values depend on congestion state; treat these as order-of-magnitude figures.

| Memory | Approx. Read Latency | Notes |
|---|---|---|
| Local L1 (own core) | ~1 cycle | Direct SRAM access |
| Remote L1 (NoC hop) | ~tens of cycles | Varies with NoC congestion |
| DRAM | ~100+ cycles | Row activation + bus latency |

### Persistence

L1 contents are **not automatically preserved across kernel launches**. When a kernel completes, its circular buffers are deallocated, and the underlying L1 pages can be reused by the next kernel. To persist data across ops, either:

1. Write the output to DRAM explicitly (the default TTNN behavior for output tensors).
2. Use an L1 sharded tensor that is explicitly kept alive as an input to the next op.

If you pass an L1 tensor as input to a subsequent op without specifying how to re-use the L1 placement, TTNN may allocate a new L1 region or copy to DRAM depending on the op's requirements.

### L1 Banking and Allocation Granularity

L1 is allocated in units called **circular buffers (CBs)**. The minimum allocation quantum is one tile:

| Data Format | Tile Shape | Bytes per Tile |
|---|---|---|
| BF16 | 32×32 | 2048 bytes |
| BFP8 | 32×32 | 1088 bytes |
| uint8 | 32×32 | 1024 bytes |

When a TTNN op is dispatched to the device, its kernel code allocates input CBs and output CBs in the L1 of each core it runs on. Multiple CBs coexist in L1 simultaneously during the op's execution window. The total CB footprint across all CBs for all concurrent kernels on a core must not exceed 1.5 MB.

Because CB allocation is tile-aligned, a tensor whose logical shape does not divide evenly into 32×32 tiles will be padded up to the nearest tile boundary. This padding consumes real L1 bytes and must be included in budget estimates.

---

## DRAM

### Capacity

Wormhole B0 devices include multiple DRAM channels providing several gigabytes of total capacity per chip (exact per-channel count and per-channel bandwidth should be verified against the Wormhole B0 technical documentation — these figures are [UNVERIFIED]).

> **Warning:** The aggregate DRAM bandwidth figure of ~300 GB/s cited in some contexts refers to the peak theoretical bandwidth across all DRAM channels on a single Wormhole chip. Per-channel bandwidth and effective bandwidth under realistic access patterns will differ. Verify against Wormhole B0 hardware specifications before using these numbers in performance models.

### Access Pattern

DRAM is **shared across all 80 Tensix cores**. Any core can issue read/write requests to any DRAM address. Bandwidth is maximized when accesses are spread evenly across all DRAM channels, which is why TTNN's default interleaved layout distributes tensor data across banks.

DRAM is slower per-access than L1 but has far larger capacity and a single coherent view accessible from all cores.

### Persistence Across Ops

DRAM tensors persist as long as the `ttnn.Tensor` object is alive on the Python side. TTNN allocates a buffer in DRAM when a tensor is created (or when an op writes its output there) and frees it when the tensor is garbage collected or explicitly deallocated. This makes DRAM the natural home for state that must survive multiple kernel launches — most importantly, the KV cache.

---

## Comparison: L1 vs. DRAM

| Property | L1 SRAM | DRAM |
|---|---|---|
| Capacity | 1.5 MB per core (120 MB aggregate) | Several GB per chip (multi-channel) |
| Read latency | ~1 cycle (local) | ~100+ cycles |
| Bandwidth | Very high for local access; limited by NoC for cross-core | ~300 GB/s aggregate [UNVERIFIED] |
| Sharing | Private to each core | Shared across all cores |
| Persistence | Lost between ops unless explicitly maintained | Persists across ops |
| Allocation unit | Tile-aligned circular buffer | Buffer-aligned block |
| Overflow behavior | Hard allocation failure (no eviction) | Limited by device memory |
| Typical use | Activations, routing scores, small working buffers | KV cache, large activation tensors, all-to-all buffers for prefill |

---

## DRAM Interleaving

TTNN distributes interleaved DRAM tensors across all available DRAM banks automatically when `TensorMemoryLayout.INTERLEAVED` is used. The layout algorithm assigns successive tile stripes to successive DRAM bank addresses in a round-robin fashion. For a tensor with $T$ total tiles and $B_{\text{dram}}$ DRAM banks:

$$\text{tile } i \rightarrow \text{bank } (i \bmod B_{\text{dram}})$$

This maximizes aggregate bandwidth by ensuring that a sequence of tile reads touches all banks rather than saturating one. It is the correct default for large tensors accessed sequentially (e.g., KV cache reads during attention).

TTNN exposes this as `ttnn.DRAM_MEMORY_CONFIG`; the constructor equivalent and predefined constant are shown in `memory_config_api.md`.

---

## Circular Buffers (CBs) in L1

Every TTNN kernel op manages its L1 usage through circular buffers. A CB is a named, tile-aligned region of L1 that a kernel reads from or writes to. The kernel compiler assigns CB IDs and sizes during program compilation.

For each core executing an op, the CB footprint is:

$$\text{CB total} = \sum_{i} n_{\text{tiles},i} \times \text{bytes per tile}$$

where $i$ ranges over all active CBs for that op (input CBs, output CBs, intermediate CBs).

Example: a matmul kernel on one core with:
- Input A CB: 4 tiles of BF16 = $4 \times 2048 = 8{,}192$ bytes
- Input B CB: 4 tiles of BF16 = $4 \times 2048 = 8{,}192$ bytes
- Output CB: 4 tiles of BF16 = $4 \times 2048 = 8{,}192$ bytes
- Intermediate accumulator CB: 2 tiles = $2 \times 2048 = 4{,}096$ bytes

Total CB footprint = 28,672 bytes ≈ 28 KB — well within the 1.5 MB per-core budget.

The CB footprint scales with:
- **Tensor shard size**: if the op is run on fewer cores, each core holds a larger shard, increasing per-core CB size.
- **Pipelining depth**: TTNN may allocate double-buffered CBs (2× tiles) to overlap compute and data movement. This is beneficial for throughput but doubles the CB footprint.

> **Warning:** If the sum of all active CB allocations on a core exceeds 1.5 MB, TTNN raises a `MemoryAllocationError` at program compilation time (not at runtime). This error has no automatic recovery path — there is no L1-to-DRAM spill mechanism. You must restructure the op (fewer tiles per CB, more cores, or move some tensors to DRAM).

---

## L1 Shard Layouts

When a tensor is placed in L1 with sharding, each core holds a contiguous slice of the tensor in its local L1. TTNN supports three sharding orientations:

### `HEIGHT_SHARDED`

The tensor's row dimension is split across cores. For a tensor of shape $[M, N]$ (in tiles: $M_t \times N_t$):

$$\text{per-core shard shape} = \left[\left\lceil\frac{M_t}{n_{\text{cores}}}\right\rceil, N_t\right] \times \text{bytes per tile}$$

All $N_t$ column tiles are present on every core, but only a subset of row tiles. This is the natural layout for ops that process different rows independently (e.g., layernorm row-wise).

### `WIDTH_SHARDED`

The tensor's column dimension is split across cores. Each core holds all rows but only a fraction of columns:

$$\text{per-core shard shape} = \left[M_t, \left\lceil\frac{N_t}{n_{\text{cores}}}\right\rceil\right] \times \text{bytes per tile}$$

Used when column-parallel operations are needed without row communication.

### `BLOCK_SHARDED`

Both dimensions are split. Cores are arranged in a 2D grid of shape $(r \times c)$ where $r \times c = n_{\text{cores}}$:

$$\text{per-core shard shape} = \left[\left\lceil\frac{M_t}{r}\right\rceil, \left\lceil\frac{N_t}{c}\right\rceil\right] \times \text{bytes per tile}$$

Block sharding minimizes per-core shard size for square-ish tensors and is typically used for 2D matmuls.

### Checking Whether a Shard Fits

Before applying an L1 shard layout, compute the per-core shard size and verify it is within budget:

```python
import math

def per_core_bytes_height_sharded(
    M: int,          # total rows (elements, not tiles)
    N: int,          # total columns (elements, not tiles)
    n_cores: int,    # number of cores to shard across
    bytes_per_element: float = 2.0,  # BF16
    tile_size: int = 32,
) -> int:
    """
    Compute the per-core L1 footprint for HEIGHT_SHARDED layout.
    Includes tile-alignment padding.
    """
    M_t = math.ceil(M / tile_size)  # rows in tiles
    N_t = math.ceil(N / tile_size)  # cols in tiles
    per_core_M_t = math.ceil(M_t / n_cores)
    bytes_per_tile = tile_size * tile_size * bytes_per_element
    return per_core_M_t * N_t * int(bytes_per_tile)

L1_PER_CORE = 1.5 * 1024 * 1024  # 1.5 MB in bytes

shard_bytes = per_core_bytes_height_sharded(M=512, N=7168, n_cores=80)
print(f"Per-core shard: {shard_bytes / 1024:.1f} KB")
print(f"Fits in L1: {shard_bytes <= L1_PER_CORE}")
# Per-core shard: 448.0 KB  (1 tile-row × 224 tile-cols × 2048 bytes/tile)
# Fits in L1: True
```

---

## Summary

- L1 is fast, private, and scarce (1.5 MB/core). Place tensors here when they are small enough to fit after sharding across cores, and when latency reduction from avoiding DRAM is measurable.
- DRAM is large, shared, and persistent but slower. Use it as the default for any tensor whose size or persistence requirements exceed what L1 can hold.
- CB allocation failures are hard errors. Budget L1 carefully before enabling L1 placement.
- DRAM interleaving across banks is automatic with `ttnn.DRAM_MEMORY_CONFIG` and is the correct default for large tensors.
- The choice between L1 and DRAM is a per-tensor, per-phase decision — covered in `decode_memory_strategy.md` and `prefill_memory_strategy.md`.

---

## References

- Tenstorrent Wormhole B0 Architecture Overview (internal specification — DRAM channel count and per-channel bandwidth should be verified against this document)
- `tt-metal` source: `ttnn/cpp/ttnn/tensor/types.hpp` — `BufferType`, `TensorMemoryLayout`
- `tt-metal` source: `ttnn/cpp/ttnn/operations/core/core.hpp` — `to_memory_config`
- Chapter 4 index: `ch04_memory_config/index.md`

---

**Next:** [memory_config_api.md](./memory_config_api.md)
