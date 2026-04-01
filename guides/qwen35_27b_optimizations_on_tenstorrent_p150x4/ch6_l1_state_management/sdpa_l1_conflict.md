# SDPA Circular Buffer Conflict

The rolling window strategy keeps 3 GDN layers' states in L1 at a time and swaps groups at the boundary between each set of 3 GDN layers and the following attention layer. But the attention layer itself presents a problem: its Scaled Dot-Product Attention (SDPA) kernel temporarily allocates large circular buffers in L1 that can overlap with the address range occupied by HEIGHT_SHARDED GDN state tensors.

This conflict is the primary remaining blocker for full HEIGHT_SHARDED L1 state deployment.

## L1 Address Space Layout

On a Blackhole compute core, L1 memory is divided into two regions:

1. **Circular buffer region (low addresses).** Allocated by the ttnn runtime for kernel input/output buffers. Each kernel dispatch can request different CB sizes; the runtime packs them starting from a base address.
2. **Tensor storage region (high addresses).** Where sharded tensors such as HEIGHT_SHARDED GDN state are placed. These persist across kernel dispatches as long as the tensor is alive.

The SDPA kernel's circular buffer region extends to approximately **1,264 KB per core**. Any HEIGHT_SHARDED tensor must be placed above this watermark to avoid corruption during attention layer execution.

## The Conflict

The GDN state for a single pair is 32 KB (16 tiles of 2048 bytes). With HEIGHT_SHARDED layout across the 96-core grid, each core holds 4 pairs worth of state: $4 \times 32\,\text{KB} = 128\,\text{KB}$ per core.

The problem arises from the 1-in-4 layer pattern:

1. **GDN layers 0-2 execute.** The fused GDN kernel uses modest CB allocations — the 26 circular buffers described in Chapter 4 fit well within the lower L1 region. No conflict.
2. **Attention layer 3 executes.** The SDPA flash attention kernel requests significantly larger CBs for its Q, K, V, and output chunks. These CBs expand to fill approximately 1,264 KB of each core's L1.
3. **If GDN state is sharded into the same address range**, the SDPA CB allocation silently overwrites the state data. The subsequent GDN layers 4-6 would then read corrupted state, producing incorrect outputs.

The swap mechanism in `_swap_l1_state()` saves GDN states back to DRAM before the attention layer runs. But with HEIGHT_SHARDED layout the L1 shard allocation persists — the shard addresses remain mapped even while the data is conceptually saved to DRAM. The SDPA kernel has no knowledge of these sharded regions and can allocate CBs that overlap with them.

## Why L1 INTERLEAVED Avoids This

L1 INTERLEAVED state uses `ttnn.L1_MEMORY_CONFIG` with interleaved layout. When `_swap_l1_state()` deallocates the L1 tensor with `ttnn.deallocate()` (line 268 of `model.py`), the L1 pages are fully released back to the allocator. The SDPA kernel then has the full L1 available for its CBs. When the next GDN group loads, `ttnn.to_memory_config()` allocates fresh L1 pages that do not conflict with any active CB region.

This is why L1 INTERLEAVED works correctly with up to 4 layers while HEIGHT_SHARDED is limited to 1-2 layers: the INTERLEAVED path has a clean allocate-deallocate cycle around each attention layer, while the HEIGHT_SHARDED path tries to maintain persistent shard mappings.

## Potential Solutions

### 1. Explicit L1 Address Partitioning

Reserve a fixed region of L1 above the 1,264 KB SDPA watermark for GDN state shards. This requires:

- Configuring the ttnn runtime to cap CB allocation below a specified address
- Placing HEIGHT_SHARDED tensors above that address via pre-allocated buffers

On Blackhole, each Tensix core has 1,504 KB of L1. With the SDPA CB region consuming 1,264 KB, approximately 240 KB remains above the watermark. A single pair's state is 32 KB, so roughly 7 pairs' state could fit per core above the watermark. Whether this is sufficient depends on whether fewer cores with more pairs per core can cover the 384-pair total.

### 2. Reduce SDPA CB Footprint

Tune the SDPA `SDPAProgramConfig` to use smaller chunk sizes during decode attention. The current configuration uses `q_chunk = k_chunk = 256` for longer sequences; reducing chunk sizes would lower the CB watermark and leave more room for GDN state. The trade-off is potentially reduced SDPA throughput.

### 3. Zero-Copy Pre-Allocated L1 Buffers

Pre-allocate GDN state L1 buffers at fixed addresses during model initialization, before any kernel dispatch, then coordinate the SDPA kernel's CB allocator to avoid those addresses. This requires modifications to the ttnn CB allocation logic to support reserved regions in L1.

### 4. Hybrid INTERLEAVED + SHARDED Approach

Use HEIGHT_SHARDED for GDN layers and fall back to INTERLEAVED (or DRAM) for the swap around attention layers:

1. Copy HEIGHT_SHARDED state to a temporary DRAM buffer before the attention layer
2. Deallocate the L1 shards (freeing the address range for SDPA CBs)
3. Run the attention layer with full L1 available
4. Re-allocate L1 shards and copy state back from DRAM

This preserves zero-NOC state access during GDN execution while avoiding the conflict, at the cost of additional swap overhead around each attention layer.

## Current Status

| Configuration | Validated | Notes |
|---|---|---|
| DRAM state (baseline) | All 48 layers | Full correctness, 14.6 tok/s/user |
| L1 INTERLEAVED state | 4 layers | Correct output; clean allocate/deallocate cycle avoids SDPA conflict |
| HEIGHT_SHARDED state | 1-2 layers | Correct "Paris" output (`test_e2e_l1_hs.py`); SDPA conflict blocks scaling beyond 2 layers |

---

**Next:** [Chapter 7 — Performance Analysis and Remaining Bottlenecks](../ch7_performance_analysis/index.md)
