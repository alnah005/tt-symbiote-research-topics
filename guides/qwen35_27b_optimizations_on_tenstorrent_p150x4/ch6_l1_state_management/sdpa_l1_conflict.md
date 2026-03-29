# SDPA Circular Buffer Conflict

The rolling window strategy (Chapter 6, `l1_state_design.md`) keeps 3 GDN layers' states in L1 at a time and swaps groups at the boundary between each set of 3 GDN layers and the following attention layer. But the attention layer itself presents a problem: its Scaled Dot-Product Attention (SDPA) kernel temporarily allocates large circular buffers in L1 that can overlap with the address range occupied by HEIGHT_SHARDED GDN state tensors.

This conflict is the primary remaining blocker for full HEIGHT_SHARDED L1 state deployment.

## L1 Address Space Layout

On a Blackhole compute core, L1 memory is divided into regions:

1. **Circular buffer region** (low addresses): Allocated by the ttnn runtime for kernel input/output buffers. Each kernel dispatch can request different CB sizes; the runtime packs them starting from a base address.
2. **Tensor storage region** (high addresses): Where sharded tensors (like HEIGHT_SHARDED GDN state) are placed. These persist across kernel dispatches as long as the tensor is alive.

The critical measurement from profiling: the SDPA kernel's circular buffer region extends to approximately **1,264 KB per core**. This means any HEIGHT_SHARDED tensor must be placed above this watermark to avoid corruption during attention layer execution.

## The Conflict

The GDN state for a single pair is 32 KB (16 tiles of 2048 bytes). With HEIGHT_SHARDED layout, the state tiles are distributed across compute cores. The number of pairs per core depends on the grid configuration: for `num_pairs = B * Nv_TP = 32 * 12 = 384` distributed across the available cores, each core holds multiple pairs' worth of state.

The problem arises from the 1-in-4 layer pattern:

1. **GDN layers 0-2** execute with their state in L1. The fused GDN kernel uses relatively modest CB allocations -- the 26 circular buffers described in Chapter 4 fit well within the lower L1 region. No conflict.
2. **Attention layer 3** executes. The SDPA flash attention kernel requests significantly larger CBs for its Q, K, V, and output chunks. These CBs expand to fill approximately 1,264 KB of each core's L1.
3. **If GDN state is sharded into the same L1 address range**, the SDPA CB allocation silently overwrites the state data. The subsequent GDN layers (4-6) would then read corrupted state, producing incorrect outputs.

The swap mechanism from `_swap_l1_state()` saves GDN states back to DRAM before the attention layer runs. But with HEIGHT_SHARDED layout, the L1 allocation itself persists -- the shard addresses remain mapped even while the data is conceptually "saved." The SDPA kernel does not know about these sharded regions and can allocate CBs that overlap with them.

## Why L1 INTERLEAVED Avoids This

L1 INTERLEAVED state uses `ttnn.L1_MEMORY_CONFIG` with interleaved layout. When `_swap_l1_state()` deallocates the L1 tensor with `ttnn.deallocate()`, the L1 pages are fully released back to the allocator. The SDPA kernel then has the full L1 available for its CBs. When the next GDN group loads, `ttnn.to_memory_config()` allocates fresh L1 pages that do not conflict with any active CB region.

This is why L1 INTERLEAVED works with up to 4 layers while HEIGHT_SHARDED is limited to 1-2 layers in practice: the INTERLEAVED path has a clean allocate-deallocate cycle around each attention layer, while the HEIGHT_SHARDED path tries to maintain persistent shard mappings.

## Potential Solutions

Several approaches could resolve the conflict:

### 1. Explicit L1 Address Partitioning

Reserve a fixed region of L1 above the 1,264 KB SDPA watermark for GDN state shards. This requires:

- Configuring the ttnn runtime to limit CB allocation below a specified address
- Placing HEIGHT_SHARDED tensors above that address
- Ensuring the remaining L1 above the watermark is sufficient for the required state tiles per core

On Blackhole, each Tensix core has 1,504 KB of L1. With the SDPA CB region consuming 1,264 KB, only ~240 KB remains. A single pair's state is 32 KB, so approximately 7 pairs' state could fit per core above the watermark. Whether this is sufficient depends on the core grid assignment.

### 2. Reduce SDPA CB Footprint

Tune the SDPA `SDPAProgramConfig` to use smaller chunk sizes during decode attention. The current configuration uses `q_chunk = k_chunk = 256` for longer sequences. Reducing chunk sizes would shrink the CB allocation, lowering the watermark and leaving more room for GDN state. The trade-off is potentially reduced SDPA throughput.

### 3. Zero-Copy Pre-Allocated L1 Buffers

Pre-allocate the GDN state L1 buffers at fixed addresses during model initialization, before any kernel dispatch. Then coordinate the SDPA kernel's CB allocator to avoid those addresses. This would require modifications to the ttnn CB allocation logic to support "reserved regions" in L1.

### 4. Hybrid INTERLEAVED + SHARDED Approach

Use HEIGHT_SHARDED for GDN layers and fall back to INTERLEAVED (or even DRAM) for the state swap around attention layers. The swap would:

1. Copy HEIGHT_SHARDED state to a temporary DRAM buffer before the attention layer
2. Deallocate the L1 shards
3. Run the attention layer with full L1 available
4. Re-allocate L1 shards and copy state back from DRAM

This preserves the benefit of direct L1 memcpy during GDN execution while avoiding the SDPA conflict, at the cost of additional swap overhead around each attention layer.

## Current Status

| Configuration | Validated | Notes |
|---|---|---|
| DRAM state (baseline) | All 48 layers | Full correctness, 14.6 tok/s/user |
| L1 INTERLEAVED state | 4 layers | Correct output, clean allocate/deallocate cycle |
| HEIGHT_SHARDED state | 1-2 layers | Correct "Paris" output; SDPA conflict blocks scaling |

The immediate path forward is to resolve the SDPA CB conflict using one of the approaches above. The expected performance impact of eliminating NOC state I/O is significant: for each GDN layer, the fused kernel currently issues 16 NOC tile reads and 16 NOC tile writes per pair (384 pairs total = 12,288 NOC transactions per layer). HEIGHT_SHARDED reduces this to zero NOC transactions for state, leaving only the Q/K/V/scalar reads from DRAM and the output writes -- which are inherently needed and cannot be further optimized.

---

**Previous:** [`height_sharded_kernel.md`](./height_sharded_kernel.md) | **Next:** [Chapter 7 — Performance Analysis and Remaining Bottlenecks](../ch7_performance_analysis/index.md)
