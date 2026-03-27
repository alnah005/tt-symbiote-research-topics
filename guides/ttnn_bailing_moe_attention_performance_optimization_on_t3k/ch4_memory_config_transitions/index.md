# Chapter 4 — Memory-Config Transitions

## Scope

This chapter catalogs every memory-configuration transition that occurs in the Ling decode path — from the moment the replicated QKV tensor arrives on device through to the tensor layouts presented to `paged_sdpa_decode`. It counts the total per-step transition cost, identifies which transition dominates, and locates the concrete code changes that could eliminate or reduce the overhead.

Every forward pass of `TTNNBailingMoEAttention` in decode mode performs a sequence of `ttnn.to_memory_config` calls (explicit or implicit) that move tensor data between L1 banks and DRAM banks, or between different L1 sharding configurations. These transitions are not free: each one involves either a data copy kernel or a format-conversion kernel dispatched to the Tensix cores. At decode batch=1 the tensor sizes are small, so the cost is dominated by kernel-launch overhead and synchronisation rather than bandwidth. But the transitions are numerous, they occur on every single generated token, and together they constitute a measurable fraction of attention step latency.

This chapter answers **Question 3** of the guide: *What memory-configuration transitions occur in the decode path, what does each one cost, and which can be eliminated?*

## Prerequisites

Readers should have completed Chapters 1, 2, and 3 before proceeding. The specific concepts required are:

- **TTNN sharding primitives** — `TensorMemoryLayout` variants (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, INTERLEAVED), `ShardSpec` (shard shape + shard grid), and `MemoryConfig` (layout + shard spec + buffer type) (see Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md))
- **Post-replication tensor layout** — the QKV tensor entering this chapter's analysis is a `(1, 1, 32, 3072)` BF16 TILE_LAYOUT `ReplicateTensorToMesh` tensor in `DRAM_MEMORY_CONFIG` on each chip, produced by `_to_replicated` (see Chapter 3, [`roundtrip_mechanics.md`](../ch3_host_roundtrip_replication/roundtrip_mechanics.md)). Note: the logical shape is `(1, 1, 1, 3072)` in ROW_MAJOR terms; the sequence dimension is padded from 1 to 32 (one full tile row) when the tensor is converted to TILE_LAYOUT, yielding the `(1, 1, 32, 3072)` shape seen throughout this chapter.
- **T3K core grid** — each chip has 80 Tensix cores arranged in a grid; sharding distributes tensor tiles across these cores, and shard shape determines how many cores are used (see Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md))

## Why Memory-Config Transitions Matter in Decode

During prefill, individual kernel latencies are large (long sequences, large matmuls), and the relative overhead of a few memory-config transitions is small. At decode batch=1, the opposite is true: the primary kernels operate on single-token tensors with shapes like `(1, 1, 1, 128)`, and kernel launch overhead can be comparable to execution time. A memory-config transition that takes 5–10 µs represents a substantial fraction of a total attention decode latency that may be 50–200 µs [ESTIMATE].

The Ling decode path is particularly exposed because:

1. The RoPE kernel (`TTNNRotaryPositionEmbedding`) requires its input in HEIGHT_SHARDED layout, but the QKV tensor arrives in DRAM INTERLEAVED layout after the host round-trip.
2. The paged KV-cache update kernel (`paged_update_on_device`) requires K and V in a specific re-sharded layout that differs from the post-RoPE layout.
3. QK normalisation (`TTNNRMSNorm`) requires a 2D INTERLEAVED input, forcing additional reshape and to-memory-config calls around it (covered in detail in Chapter 6).

The result is a chain of **seven to nine distinct memory-config transitions** per decode step (7 when `use_qk_norm=False`; 9 when `use_qk_norm=True`), each adding overhead that compounds across thousands of generated tokens. The transitions are:

| Symbol | Direction | Trigger |
|---|---|---|
| T1a | DRAM→L1 HEIGHT_SHARDED | RoPE Q input (`TTNNRotaryPositionEmbedding`) |
| T1b | DRAM→L1 HEIGHT_SHARDED | RoPE K input (`TTNNRotaryPositionEmbedding`) |
| T2a | L1→DRAM INTERLEAVED | Post-RoPE Q eviction |
| T2b | L1→DRAM INTERLEAVED | Post-RoPE K eviction |
| T3a | DRAM→L1 re-sharded | `paged_update_on_device` K input |
| T3b | DRAM→L1 re-sharded | `paged_update_on_device` V input |
| T4 | L1→DRAM (or reshape) | Post-SDPA output transition |
| T_norm_in_Q | DRAM→L1 INTERLEAVED | QK norm Q input (`TTNNRMSNorm`, only when `use_qk_norm=True`) |
| T_norm_in_K | DRAM→L1 INTERLEAVED | QK norm K input (`TTNNRMSNorm`, only when `use_qk_norm=True`) |

**Combined overhead with `use_qk_norm=True`: approximately 83–86 µs [ESTIMATE] per decode step** (all 9 transitions: T1a, T1b, T2a, T2b, T_norm_in_Q, T_norm_in_K, T3a, T3b, T4). This is the full Ch4 total — it is not a base cost to which any further cost is added. The 64 µs figure cited elsewhere is the eliminable **subset** of this total (transitions that could be removed with layout-preserving changes); do not add 64 µs to 83–86 µs. T_norm_out_Q and T_norm_out_K (~32 µs combined) are post-norm output transitions that complete the QK norm round-trip; they are cataloged in Chapter 6 (`qk_norm_latency.md`) and are NOT included in Ch4's 83–86 µs figure. See `transition_cost_model.md` for the per-transition breakdown.

## Reading Order

Work through the files in this order:

1. [`decode_tensor_lifecycle.md`](./decode_tensor_lifecycle.md) — Annotated diagram of the complete per-step tensor journey: input shape, memory config, and the `ttnn.to_memory_config` call sequence at each stage.
2. [`transition_cost_model.md`](./transition_cost_model.md) — How TTNN executes a memory-config transition at the hardware level, expected cycle counts, and per-transition cost estimates for the Ling decode step.
3. [`optimization_opportunities.md`](./optimization_opportunities.md) — Which transitions are eliminable, which are kernel-mandated, and the exact source locations in `TTNNBailingMoEAttention` where changes would be made.

## Key Symbols Used in This Chapter

| Symbol | Meaning |
|---|---|
| `N_q` | Number of Q heads = 16 |
| `N_kv` | Number of KV heads = 4 |
| `H` | head\_dim = 128 elements |
| `T` | TILE\_SIZE = 32 elements |
| `rope_shard_mem_q` | `MemoryConfig` for HEIGHT\_SHARDED RoPE Q input: shard `(T, H)` = `(32, 128)`, grid `CoreCoord(0,0)→CoreCoord(7,1)` (8 cols × 2 rows = 16 cores) |
| `rope_shard_mem_k` | `MemoryConfig` for HEIGHT\_SHARDED RoPE K input: shard `(T, H)` = `(32, 128)`, grid `CoreCoord(0,0)→CoreCoord(3,0)` (4 cols × 1 row = 4 cores) |
| `t_trans` | Latency of a single memory-config transition |
| `t_total` | Sum of all transition latencies per decode step |

**Start reading:** [Decode Tensor Lifecycle](decode_tensor_lifecycle.md)
