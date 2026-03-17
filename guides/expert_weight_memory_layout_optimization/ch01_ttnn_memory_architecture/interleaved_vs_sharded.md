# Interleaved vs Sharded Allocation

## How Interleaved Allocation Works

When TTNN allocates a tensor with `TensorMemoryLayout.INTERLEAVED`, it divides the tensor into fixed-size pages and distributes those pages in round-robin order across all available banks of the target buffer type.

For `buffer_type=DRAM` on Wormhole B0, "all available banks" means all 12 GDDR6 banks. The page size is determined by the tensor's data type and tile layout; for a `bfloat16` tensor in `TILE_LAYOUT`, one tile (32×32 elements) occupies 2048 bytes, and the allocator uses tile-granularity pages.

```
Interleaved DRAM allocation for a weight tensor [128 tiles H × 448 tiles W]:

  Tile (0,0)   → Bank 0
  Tile (0,1)   → Bank 1
  Tile (0,2)   → Bank 2
  ...
  Tile (0,11)  → Bank 11
  Tile (0,12)  → Bank 0   ← wraps back to Bank 0
  Tile (0,13)  → Bank 1
  ...
  Tile (1,0)   → Bank ?   ← continues incrementing from where row 0 left off
```

The key property: **no single bank owns a contiguous region of the tensor**. Any sequential access — reading an entire row of the weight matrix, for example — must touch all 12 banks in interleaved sequence.

---

## What Interleaved Means for Expert Weight Tensors

Consider a weight tensor for a single expert with shape `[num_experts, d_model, d_ff]`, e.g., `[8, 4096, 14336]` in bfloat16. When stored in `ttnn.DRAM_MEMORY_CONFIG` (interleaved), the tiles of every expert's weight sub-matrix are spread across all 12 DRAM banks.

When the MoE router dispatches tokens to an expert and the matmul kernel begins reading that expert's weight slice `[4096, 14336]`, it must issue reads to all 12 DRAM controllers. At 128 × 448 = 57,344 tiles per expert, and 12 banks, each bank holds approximately 4,779 tiles. But those tiles are not contiguous within the bank — they are interleaved with tiles from all other experts stored in the same allocation.

This access pattern has two costs:

1. **Latency cost:** Each DRAM bank access has a round-trip latency including the NoC traversal. Non-contiguous access within a bank reduces the effectiveness of GDDR6's internal row-buffer cache, causing more row activations and higher per-access latency.

2. **Bandwidth cost under parallelism:** If multiple Tensix cores are executing matmuls for different experts simultaneously (the common case in MoE), each core issues reads to all 12 banks independently. The NoC links near each DRAM controller receive traffic from multiple cores at once, leading to queuing and reduced effective bandwidth.

---

## How Sharded Allocation Works

With a sharded layout (e.g., `TensorMemoryLayout.WIDTH_SHARDED`), the tensor is divided into a fixed number of contiguous chunks called shards. Each shard is assigned to a specific bank or core. The tensor is not interleaved — shard N lives entirely in bank (or core) N.

```
WIDTH_SHARDED allocation for [4096, 14336] across 8 shards:

  Shard 0: columns [0,    1791]  → Bank group 0
  Shard 1: columns [1792, 3583]  → Bank group 1
  Shard 2: columns [3584, 5375]  → Bank group 2
  Shard 3: columns [5376, 7167]  → Bank group 3
  Shard 4: columns [7168, 8959]  → Bank group 4
  Shard 5: columns [8960, 10751] → Bank group 5
  Shard 6: columns [10752, 12543]→ Bank group 6
  Shard 7: columns [12544, 14335]→ Bank group 7
```

Each shard is a contiguous block of `[4096, 1792]` elements. Reading shard 0 means issuing sequential read requests to bank group 0 only — no cross-bank traversal, no round-robin pattern, and full exploitation of GDDR6's row-buffer locality within that bank.

### Sharding and Core Grids

For DRAM-sharded tensors, the `ShardSpec.grid` describes the set of DRAM bank groups that hold the shards. The correspondence is:

```
CoreCoord(col, row) in the ShardSpec grid → a specific DRAM bank group
```

For L1-sharded tensors, the `ShardSpec.grid` describes the set of Tensix cores whose L1 SRAM holds the shards. Each core's local L1 holds its shard, and the core's compute kernel reads directly from its own L1 without any NoC traversal.

---

## NoC Contention Under Interleaved Access

The following diagram illustrates why interleaved access causes contention when multiple cores run concurrently.

```
Interleaved: 4 Tensix cores reading the same weight tensor simultaneously

  Core A ──────────────────────────────────┐
  Core B ──────────────────────────────────┤──→ NoC link X ──→ DRAM Bank 0
  Core C ──────────────────────────────────┤
  Core D ──────────────────────────────────┘

  Core A ──────────────────────────────────┐
  Core B ──────────────────────────────────┤──→ NoC link Y ──→ DRAM Bank 1
  Core C ──────────────────────────────────┤
  Core D ──────────────────────────────────┘

  ... (repeated for all 12 banks)
```

Every core competes on every NoC link to every DRAM controller. The effective bandwidth available to each core is approximately `total_DRAM_BW / (num_cores × num_banks_accessed_per_core)` in the worst case, though NoC pipelining partially mitigates this.

```
Sharded: 8 Tensix cores, each reading from its dedicated DRAM bank shard

  Core 0 ──→ NoC link to Bank 0 (exclusive) ──→ Shard 0
  Core 1 ──→ NoC link to Bank 1 (exclusive) ──→ Shard 1
  Core 2 ──→ NoC link to Bank 2 (exclusive) ──→ Shard 2
  ...
  Core 7 ──→ NoC link to Bank 7 (exclusive) ──→ Shard 7
```

Each core traverses a single dedicated NoC path. Bank contention is eliminated. Each core receives the full bandwidth of its assigned DRAM bank group.

---

## The Reshard Pattern: DRAM-Sharded to L1-Sharded

The canonical flow for high-performance weight tensor reads in TTNN is a two-step placement strategy:

1. **Store the weight tensor in DRAM-sharded layout.** This distributes the tensor across DRAM banks and enables parallel prefetch with dedicated NoC paths per bank.

2. **Reshard into L1-sharded layout immediately before the matmul.** Once in L1, the matmul kernel reads from the core's own L1 SRAM at ~1 TB/s bandwidth with zero NoC latency.

```python
import ttnn

# Step 1: Load expert weights into DRAM with WIDTH_SHARDED layout.
# (ShardSpec construction covered in Chapter 2.)
dram_sharded_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=dram_shard_spec,   # defined in ch02
)

weight_dram = ttnn.from_torch(
    expert_weight_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_sharded_cfg,
)

# Step 2: Move to L1-sharded immediately before the matmul.
# The L1 shard grid matches the compute grid that will run the matmul.
l1_sharded_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=l1_shard_spec,     # defined in ch02
)

weight_l1 = ttnn.to_memory_config(weight_dram, l1_sharded_cfg)

# Step 3: Run matmul. The kernel reads weight tiles from local L1.
output = ttnn.matmul(activation, weight_l1, memory_config=output_cfg)

# Step 4: Free L1 immediately — do not hold it across unrelated ops.
ttnn.deallocate(weight_l1)
```

> **Tip:** The `ttnn.to_memory_config` call in step 2 is not free — it initiates a DMA transfer from DRAM into L1 across the NoC. The benefit is that this transfer is parallelized across all 8 (or N) shards simultaneously, each using its own NoC path. The latency of this transfer is what DRAM-sharded layout minimizes compared to interleaved.

---

## Choosing Between Interleaved and Sharded

Use this decision table as a starting point. The chapters that follow provide quantitative guidance for specific expert weight tensor shapes.

| Scenario | Recommended layout | Reason |
|---|---|---|
| Single expert, single-core matmul, small tensor | `DRAM_MEMORY_CONFIG` (interleaved) | No parallelism to exploit; simplest config |
| Large expert weights read by a single core repeatedly | DRAM WIDTH_SHARDED → L1 WIDTH_SHARDED | Parallel prefetch from DRAM, then zero-latency compute from L1 |
| Many experts, each processed by a dedicated core | DRAM HEIGHT_SHARDED across expert count | Each core prefetches its expert's weights from a dedicated DRAM bank |
| Very large weight matrix, 2D core grid available | DRAM BLOCK_SHARDED → L1 BLOCK_SHARDED | Both row and column dimensions benefit from shard parallelism |
| Activation tensors between ops in same graph | `L1_MEMORY_CONFIG` (interleaved) | Short-lived; interleaved L1 is simpler and sufficient for activations |
| Bias vectors used across many matmul cores | `DRAM_MEMORY_CONFIG` or L1 SINGLE_BANK | Small size; no sharding benefit; simplicity preferred |

---

## Next Steps

This completes Chapter 1. You now have the physical and API-level vocabulary needed to understand weight tensor placement on Wormhole B0:

- The 12-bank DRAM topology and its bandwidth/latency characteristics
- The `MemoryConfig` API and its `BufferType` and `TensorMemoryLayout` arguments
- The difference between interleaved round-robin distribution and contiguous sharded placement
- The two-step DRAM-sharded → L1-sharded reshard pattern for matmul

---

**Next:** [Chapter 2 — DRAM-Sharded Memory Layout](../ch02_dram_sharded_memory_layout/index.md)
