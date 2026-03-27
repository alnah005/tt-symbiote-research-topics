# T3K Topology Primer

This file describes the T3K hardware platform at the level of detail needed to reason about tensor sharding and collective communication latency in `TTNNBailingMoEAttention`. It covers the physical chip layout, Ethernet interconnect bandwidth, and the TTNN sharding primitives that map logical tensor partitions onto that hardware.

## T3K Physical Layout

T3K is a Tenstorrent inference card containing **8 Wormhole n300 chips** arranged on a single board. From TTNN's perspective these 8 chips form a **1×8 logical mesh**: one row of 8 devices indexed `[0, 1, 2, 3, 4, 5, 6, 7]`. "Chip" and "device" are interchangeable throughout this guide.

```
┌─────────────────────────────────────────────────────────────────┐
│  T3K Board                                                      │
│                                                                 │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                       │
│  │ WH 0 │──│ WH 1 │──│ WH 2 │──│ WH 3 │                       │
│  └──────┘  └──────┘  └──────┘  └──────┘                       │
│      │                                │                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                       │
│  │ WH 7 │──│ WH 6 │──│ WH 5 │──│ WH 4 │                       │
│  └──────┘  └──────┘  └──────┘  └──────┘                       │
│                                                                 │
│  (Logical mesh index: 0–7, left-to-right in TTNN)              │
└─────────────────────────────────────────────────────────────────┘

Host (x86 CPU + DRAM)
  │
  └─ PCIe Gen4 x16 → one or more chips (gateway chips)
```

The host connects via PCIe to the T3K; the gateway chip(s) then relay data to the remaining chips over Ethernet. The host is not directly connected to all 8 chips simultaneously over independent PCIe lanes in the standard T3K configuration.

### Per-Chip Resources

Each Wormhole n300 chip provides:

- **Tensix cores:** 80 programmable compute cores, each containing matrix-multiply and vector FPUs.
- **L1 SRAM:** 1.5 MB per Tensix core (local scratchpad; not coherent across cores).
- **DRAM:** 12 GB of LPDDR4 DRAM attached to the chip (shared across all Tensix cores on that chip).
- **Ethernet cores:** 16 dedicated Ethernet MAC cores for chip-to-chip communication, running at 100 Gb/s per link.

The L1 SRAM is the fastest memory tier (sub-microsecond access latency). DRAM is orders of magnitude larger but slower. The boundary between L1-resident and DRAM-resident tensors is the most important memory hierarchy decision in TTNN kernel design.

## Ethernet Interconnect and CCL Bandwidth

### Physical Links

Each n300 chip has **16 Ethernet ports**, each capable of 100 Gb/s bidirectional. In the T3K 1×8 mesh, the chips are connected in a ring topology: each chip connects to its two neighbors, and the ring wraps around. Some chips also have cross-ring links (diagonal connections) that shorten the diameter of the mesh.

The total aggregate Ethernet bandwidth available from a single chip is up to `16 × 100 Gb/s = 1600 Gb/s = 200 GB/s` [ESTIMATE], but the usable bandwidth for a single collective operation depends on how many links the CCL library is configured to use.

### CCL `num_links` Parameter

TTNN's Collective Communication Library (CCL) exposes a `num_links` parameter on operations such as `ttnn.all_gather` and `ttnn.all_reduce`. This parameter controls **how many Ethernet links are recruited for the collective**.

- `num_links=1`: Traffic is routed through a single Ethernet link per hop. This is the most conservative setting and uses the least fabric bandwidth.
- `num_links=2`: Two links are used per hop, doubling the available bandwidth for that collective.
- `num_links=4`: Four links per hop; maximum practically used in current configurations.

The effect on latency depends on whether the collective is bandwidth-limited or latency-limited:

- For **small tensors** (collective payload ≪ 1 MB), the operation is latency-limited. Each Ethernet link introduces fixed startup overhead (message framing, routing table lookup, protocol handshake). Adding more links does not help and may add overhead from coordination.
- For **large tensors** (payload in the multi-MB range), the operation is bandwidth-limited. Doubling `num_links` roughly halves transfer time, subject to fabric contention.

Ethernet CCL latency and bandwidth characteristics for T3K.

| Metric | Value | Notes |
|--------|-------|-------|
| Per-link raw bandwidth | 100 Gb/s = 12.5 GB/s | Hardware specification |
| Usable CCL bandwidth per link | ~10–11 GB/s [ESTIMATE] | Protocol overhead deducted |
| All-reduce latency, small tensor (<64 KB) | ~5–20 µs [ESTIMATE] | Latency-dominated; `num_links` has little effect |
| All-reduce latency, large tensor (>1 MB) | bandwidth-limited | Scales as `payload / (num_links × 10 GB/s)` [ESTIMATE] |
| All-gather latency, 8 chips, `num_links=1` | ~15–40 µs [ESTIMATE] | For typical decode-step tensor sizes |
| Ethernet link startup latency | ~2–5 µs [ESTIMATE] | Per-hop fixed cost |

The relevance of `num_links` to the fused QKV all-reduce is analyzed in detail in (see Chapter 2, `num_links_tuning.md`).

### PCIe Host Transfer

The host connects to T3K via PCIe Gen4. PCIe bandwidth characteristics affect any operation that moves data between device DRAM and host DRAM.

PCIe bandwidth characteristics for T3K host transfers.

| Metric | Value | Notes |
|--------|-------|-------|
| PCIe Gen4 x16 theoretical BW | 32 GB/s bidirectional | Per gateway chip |
| Practical host-to-device BW | ~20–25 GB/s [ESTIMATE] | Protocol and DMA overhead |
| Practical device-to-host BW | ~20–25 GB/s [ESTIMATE] | Typically symmetric |
| Effective BW for small transfers | Much lower | Fixed DMA setup cost dominates |

For decode-step tensors (a single token's activations: `1 × 1 × 4096 × 2 bytes = 8 KB`), PCIe transfers are heavily latency-limited. The `_to_replicated` host round-trip analyzed in (see Chapter 3, `host_transfer_overhead.md`) operates in this regime.

## TTNN Sharding Primitives

TTNN uses a set of structured sharding strategies to distribute tensors across the Tensix cores within a chip and, with multi-device tensors, across chips in the mesh. The following primitives are referenced repeatedly in Chapters 2–6.

### TensorMemoryLayout

`TensorMemoryLayout` is an enum that specifies how a tensor's elements are partitioned and where shards are stored. The four variants used by `TTNNBailingMoEAttention` are:

**INTERLEAVED**

The tensor is stored in DRAM, with tiles distributed round-robin across DRAM banks. No spatial locality is guaranteed between tiles in the same row or column. This is the default layout for tensors that do not fit in L1 or do not have a performance-critical sharding requirement.

```python
# Example: creating an interleaved DRAM memory config
import ttnn
mem_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)
```

**HEIGHT_SHARDED**

The tensor is divided along its height (row) dimension. Each shard is a contiguous block of rows, and each shard is stored in the L1 of a different Tensix core (or a different chip, in a multi-device setting). HEIGHT_SHARDED is well-suited for operations that process rows independently, such as per-token or per-head operations.

In the context of `TTNNBailingMoEAttention`, the RoPE application uses a HEIGHT_SHARDED layout where each shard corresponds to one attention head's worth of data. The shard shape for RoPE is `(TILE_SIZE, head_dim)` — one tile-row of Q or K per shard — matched to the RoPE kernel's expected input format.

```python
# Example: HEIGHT_SHARDED MemoryConfig for a Q tensor across 16 heads
shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
    shape=[32, 128],  # (TILE_SIZE, head_dim)
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
mem_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=shard_spec,
)
```

**WIDTH_SHARDED**

The tensor is divided along its width (column) dimension. Each shard is a contiguous block of columns. WIDTH_SHARDED is used when an operation is parallelized over the output feature dimension — for example, after a column-parallel linear layer where each chip holds a different subset of output columns.

**BLOCK_SHARDED**

The tensor is divided into 2D blocks: each shard is a rectangular sub-matrix defined by both a row range and a column range. BLOCK_SHARDED is the most flexible layout and is used when neither pure row nor pure column partitioning matches the operation's data access pattern.

Summary of TensorMemoryLayout variants and their primary use cases in TTNNBailingMoEAttention.

| Layout | Partition Dimension | Typical Use Case in Attention |
|--------|---------------------|-------------------------------|
| `INTERLEAVED` | None (round-robin tiles) | Default/fallback; tensors in DRAM between ops |
| `HEIGHT_SHARDED` | Rows | Per-head ops (RoPE, QK norm input/output) |
| `WIDTH_SHARDED` | Columns | Post-column-parallel linear (QKV projection output) |
| `BLOCK_SHARDED` | Rows and columns | General 2D parallelism; large matmul inputs |

### ShardSpec

`ShardSpec` defines the shape and core placement of a single shard within a `HEIGHT_SHARDED`, `WIDTH_SHARDED`, or `BLOCK_SHARDED` tensor. Its primary fields are:

- `grid`: A `CoreRangeSet` specifying which Tensix cores hold shards. On a single chip, this is a set of `CoreRange` objects identifying core coordinates in the chip's `(x, y)` grid. On a multi-device tensor, grids are specified per-chip and the multi-device sharding is handled at a higher level by the mesh device configuration.
- `shape`: A `[height, width]` pair giving the dimensions of one shard in elements (not tiles). The shard shape must be a multiple of the tile size (32×32 for most dtypes) to avoid padding overhead.
- `orientation`: `ROW_MAJOR` or `COL_MAJOR`, specifying whether shards are assigned to cores in row-major or column-major order when the grid has multiple rows and columns.

### MemoryConfig

`MemoryConfig` is the top-level descriptor that TTNN operations use to specify where and how to store a tensor. It combines:

- `memory_layout`: One of the `TensorMemoryLayout` variants above.
- `buffer_type`: `L1` (on-chip SRAM) or `DRAM` (off-chip LPDDR4).
- `shard_spec`: Optional; required for HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED layouts.

Every `ttnn.to_memory_config` call (a memory-config transition) specifies a target `MemoryConfig`. The cost of that transition depends on the amount of data moved and the source/destination buffer types (L1-to-L1 is fast; DRAM-to-L1 is slower and bandwidth-limited). The framework for reasoning about these costs is developed in (see Chapter 4, `transition_cost_model.md`).

### Multi-Device Tensor Distribution

For multi-device tensors across the T3K 1×8 mesh, TTNN adds a second level of distribution on top of the per-chip `MemoryConfig`:

- **Sharded across devices:** The tensor is partitioned in some dimension across the 8 chips, with each chip holding a disjoint slice. For example, a column-sharded QKV weight matrix distributes different columns to different chips.
- **Replicated across devices:** All 8 chips hold an identical copy of the tensor. This is required for operations like `paged_sdpa_decode` whose kernel expects the full Q tensor to be available locally on each chip.

The conversion between these two distribution modes is where most CCL operations arise. An all-gather converts sharded → replicated (each chip broadcasts its shard to all others). An all-reduce performs a reduce (sum) across shards, keeping the result replicated. These operations are the subject of (see Chapter 2, `fusion_mechanics.md`) and (see Chapter 3, `roundtrip_mechanics.md`).

## Summary: How T3K Topology Constrains Attention Design

The attention layer's design on T3K is constrained by three topology facts:

1. **8 chips, fixed.** The 1×8 mesh forces all parallelism into exactly 8 parts. Critically, `4 / 8 = 0.5` KV heads per chip is not an integer, which forces GQA grouping and broadcast to be handled at the kernel level — see the GQA section in `ling_model_overview.md`.

2. **Ethernet is the inter-chip link.** There is no shared DRAM or cache-coherent fabric between chips. Every inter-chip communication is an explicit CCL operation over Ethernet, with measurable and non-negligible latency. This makes the number and size of CCL calls in the decode path a primary optimization target.

3. **PCIe is the host-device link.** The `_to_replicated` round-trip traverses PCIe twice; for small decode-step tensors this is latency-dominated — see Chapter 3, `host_transfer_overhead.md`.

These three facts recur throughout the optimization analysis in Chapters 2–7. The table below provides a quick reference for the quantitative values used in later chapters.

Quick reference: T3K topology constants used throughout this guide.

| Quantity | Value |
|----------|-------|
| Number of chips | 8 |
| Mesh shape (logical) | 1×8 |
| Ethernet links per chip | 16 |
| Per-link bandwidth | 100 Gb/s = 12.5 GB/s |
| Usable CCL bandwidth per link | ~10–11 GB/s [ESTIMATE] |
| L1 SRAM per Tensix core | 1.5 MB |
| Chip DRAM capacity | 12 GB LPDDR4 |
| PCIe gen / lanes to host | Gen4 x16 |
| Practical host↔device PCIe BW | ~20–25 GB/s [ESTIMATE] |

---

**Next:** [Chapter 2 — Fused QKV Projection](../ch2_fused_qkv_projection/index.md)
