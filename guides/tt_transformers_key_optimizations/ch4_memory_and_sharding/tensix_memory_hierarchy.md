# The Tensix Memory Hierarchy

Tenstorrent hardware exposes four distinct memory tiers to a running kernel: L1 SRAM local to each core, per-chip DRAM, the on-chip NoC that connects cores and DRAM banks, and on-board Ethernet links that connect chips in a multi-device configuration. Every performance decision in tt-transformers is ultimately about which tier holds a tensor at the moment a core needs to read or write it.

---

## L1 SRAM — 120 KB per Tensix Core

Each Tensix core has 120 KB of L1 SRAM. It is local to the core: reading from L1 requires no NoC transaction and no DRAM arbitration. The entire working set for a tile computation — input tiles, weight tiles, accumulator — must fit in these 120 KB simultaneously.

### Buffer Partitioning and Double-Buffering

L1 is partitioned into three logical regions: input A buffer, input B buffer, and output buffer. This maps directly to the three circular buffer (CB) slots that the TT-Metalium programming model exposes: `cb_in0`, `cb_in1`, and `cb_out0`. Each CB slot can hold one or more tiles.

Ping-pong double-buffering is the standard pattern: each CB slot holds two tile positions so that the Reader RISC-V kernel can prefetch tile N+1 while the Compute kernel processes tile N. The two RISC-V cores within a single Tensix core run asynchronously, so the prefetch and compute phases genuinely overlap when the workload is DRAM-bandwidth-bound. See [flash_attention_prefill.md](../ch2_attention_optimizations/flash_attention_prefill.md) for the concrete double-buffering implementation in the attention kernel.

### `packer_l1_acc`

The `packer_l1_acc` flag in `WormholeComputeKernelConfig` controls where the Packer unit accumulates partial sums. The standard path is read-modify-write: after each inner-loop iteration, the Packer reads the current partial sum from the output buffer (which may reside in DRAM), adds the Dst register contents, and writes the updated sum back. For a large matmul with many K-blocks, this means one DRAM read per K-block per output tile.

When `packer_l1_acc=True`, the Packer accumulates directly into the L1 output buffer instead. The DRAM read at each K-block boundary is eliminated; only the final write from L1 to DRAM happens once per output tile. This is a throughput gain, not a correctness requirement — the numerical result is identical.

`packer_l1_acc` is only valid when the output buffer is in L1 (not DRAM). If the output CB is mapped to DRAM, the flag has no effect. In practice, `packer_l1_acc` is always paired with an L1-sharded or L1-output matmul configuration.

### L1 Sizing Constraints for Attention

FlashAttention-2 keeps Q, K, V, and O tiles live in L1 during the attention score loop. Q and O have shape `Br × D`; K and V have shape `Bc × D`. At BF16 (2 bytes per element) and head dimension D=128, the four buffers consume:

```
Q + K + V + O = (2 × Br + 2 × Bc) × D × 2 bytes
```

| Br | Bc | Buffer footprint | Valid? |
|---|---|---|---|
| 64 | 64 | (2×64 + 2×64) × 128 × 2 = 64 KB | Yes |
| 128 | 64 | (2×128 + 2×64) × 128 × 2 = 96 KB (+ score tiles) | Yes, with careful sizing |
| 128 | 128 | (2×128 + 2×128) × 128 × 2 = 128 KB | No — exceeds 120 KB without tighter buffering |

At Br=128, Bc=128, the four buffers alone exceed 120 KB. The maximum valid configuration without redesigning the buffer layout is Br=128, Bc=64, which leaves 24 KB headroom for softmax statistics and score tiles. The tradeoff is general: larger tiles reduce per-element overhead (fewer NoC round-trips per element, better FPU utilization) but require fitting more tiles simultaneously in the 120 KB window.

---

## DRAM — Per-Chip Off-Chip Storage

DRAM holds tensors that are too large for L1: model weights, the KV cache, and activation tensors for large-batch decode. DRAM has higher capacity than L1 but higher latency and must be accessed via the NoC.

### Weight Storage and Streaming

Model weights are stored in DRAM in their target compute dtype (BFP8 or BFP4 for MLP; BFP8 for attention; BF16 for special layers), converted on the host before device execution. During compute, weight tiles are streamed through the Tensix core's FPU one block at a time. The weight tiles travel from DRAM to L1 via NoC DMA and then to the FPU's Unpacker unit. They are not cached across ops — each new matmul re-reads its required weight tiles from DRAM unless they happen to be residual in L1 from a prior op.

### KV Cache Layout

The paged KV cache is stored in DRAM in a flat block pool with shape:

```
[1, n_kv_heads, n_blocks × block_size, head_dim]
```

The leading dimension is `1` — the pool is shared across the entire batch. A separate page table tensor (`[batch, max_blocks_per_seq]`, int32, DRAM) maps each sequence's logical positions to physical block indices. The attention kernel issues per-block NoC DMA reads keyed by the page table rather than reading a contiguous KV range. For the full paged KV mechanism see [paged_attention_kv_cache.md](../ch2_attention_optimizations/paged_attention_kv_cache.md).

### Activation Spill

For small to moderate batch sizes in decode (M=1–32), activation tensors fit comfortably in L1 across the core grid — this is the condition that makes L1-sharded decode viable. For large-batch decode or very wide intermediate tensors, activation shards may exceed 120 KB per core and spill to DRAM. A spilled activation reintroduces the DRAM read that L1 sharding was designed to eliminate, restoring the bandwidth bottleneck. Avoiding activation spill is why shard sizes are checked explicitly against 120 KB when configuring the core grid.

---

## NoC — Network-on-Chip

The NoC connects all Tensix cores and DRAM banks on the die. Every DRAM read, multicast, and inter-core data transfer uses NoC bandwidth.

### DRAM-Sharded Matmul: Bank Parallelism

`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exploits dedicated per-core DRAM bank ownership to achieve maximum aggregate DRAM bandwidth with no inter-bank contention. The full layout mechanism and shape constraint are covered in [sharding_patterns.md](sharding_patterns.md#width-sharded--dram-sharded----weight-tensors-for-decode); for throughput numbers see [l1_sharding_for_matmul.md](../ch3_matmul_optimizations/l1_sharding_for_matmul.md).

### Multicast: One DRAM Read, Many Cores

For prefill workloads using `MatmulMultiCoreReuseMultiCastProgramConfig`, one source core reads a weight tile block from DRAM and multicasts it simultaneously to all cores in the grid via the NoC. The weight is read once from DRAM regardless of how many cores receive it. This eliminates the O(n_cores) redundant DRAM reads that the standard 2D config would incur when many cores need the same weight tiles.

---

## On-Chip Ethernet — Multi-Chip Communication

Multi-chip configurations use on-board Ethernet links between devices. The collective operations that run over these links are the primary mechanism for tensor parallelism in tt-transformers.

| Configuration | Topology | Maximum TP degree |
|---|---|---|
| N300 | 2 chips, on-board Ethernet links | TP=2 |
| T3K | 8 chips in Ethernet ring | TP=8 |

The three collective operations provided by TTNN's CCL (Collective Communication Library) are:

- `ttnn.all_gather` — each device collects a full copy of the distributed tensor; used after column-parallel projections to reconstruct the full activation before the next op
- `ttnn.reduce_scatter` — each device contributes partial sums; aggregated result is distributed across devices; used after row-parallel projections
- `ttnn.all_reduce` — combines all_gather and reduce into a single collective; used in All-Reduce variants of tensor parallelism

Ethernet bandwidth is finite and per-step latency accumulates across decode steps. For small batch sizes, CCL latency can dominate decode inter-token latency. Chapter 5 covers the TP degree selection and CCL pipelining strategies that minimize this overhead.

---

## Key Takeaways

- L1 SRAM is 120 KB per Tensix core and is partitioned into input A, input B, and output circular buffers. Double-buffering with two CB slots per region overlaps DRAM prefetch with FPU compute.
- `packer_l1_acc=True` eliminates one DRAM read per K-block in multi-tile matmul accumulation by directing the Packer to accumulate into the L1 output buffer. It is a throughput improvement, not a correctness change, and requires the output buffer to be in L1.
- The 120 KB L1 ceiling is the hard constraint on attention tile sizes: at D=128, Br=128, Bc=64 (96 KB) fits; Br=128, Bc=128 (128 KB) does not without buffer redesign.
- DRAM-sharded matmul and NoC multicast are described in [sharding_patterns.md](sharding_patterns.md#width-sharded--dram-sharded----weight-tensors-for-decode) and [the Multicast subsection above](#multicast-one-dram-read-many-cores) respectively; together they maximise aggregate DRAM bandwidth for decode and minimise redundant DRAM reads for prefill.
- `ttnn.all_gather`, `ttnn.reduce_scatter`, and `ttnn.all_reduce` run over on-board Ethernet links (N300: TP=2, T3K: TP=8) and constitute the CCL overhead budget that limits how aggressively tensor parallelism can be applied.

---

## Further Reading

- `WormholeComputeKernelConfig` fields including `packer_l1_acc`: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` and `math_fidelity_and_data_formats.md` in Chapter 1
- Circular buffer programming model: TT-Metalium documentation, `tt_metal/programming_examples/`
- DRAM-sharded config shape constraint and throughput numbers: [l1_sharding_for_matmul.md](../ch3_matmul_optimizations/l1_sharding_for_matmul.md)
- Paged KV cache DRAM layout and page table DMA access pattern: [paged_attention_kv_cache.md](../ch2_attention_optimizations/paged_attention_kv_cache.md)
- CCL operations: `ttnn/cpp/ttnn/operations/ccl/` in the tt-metal repository

---

[Back to Chapter 4 Index](index.md) | [Next: Sharding Patterns](sharding_patterns.md)
