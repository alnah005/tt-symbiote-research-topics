# Decode Tensor Lifecycle: Memory Configs at Each Stage

## Overview

This file provides an annotated walkthrough of the complete tensor memory-configuration journey during a single Ling decode step — from the replicated QKV tensor produced by `_to_replicated`, through the RoPE and optional QK normalisation stages, into the paged KV-cache update, and finally to the SDPA inputs. At each stage, the exact `MemoryConfig` (buffer type, `TensorMemoryLayout`, and `ShardSpec`) is stated, together with the `ttnn.to_memory_config` call (or equivalent) responsible for the transition.

The Ling model configuration throughout: `num_heads=16`, `num_kv_heads=4`, `head_dim=128`, `hidden_size=4096`, `partial_rotary_factor < 1.0`, `use_qk_norm=True`.

## Notation

Tensor shapes use the convention `(batch, seq_len, num_heads, head_dim)` where relevant. At decode batch=1, seq_len=1, so all shapes begin `(1, 1, ...)`. TILE_LAYOUT pads the innermost two dimensions to the nearest multiple of 32 (the tile size `T=32`); actual allocated shapes may therefore be padded. Memory config shorthand used in diagrams:

- `DRAM/ITVL` = `DRAM_MEMORY_CONFIG`, `INTERLEAVED`
- `L1/ITVL` = L1 buffer, `INTERLEAVED`
- `L1/HS(grid, shard)` = L1 buffer, `HEIGHT_SHARDED`, shard grid `grid`, shard shape `shard`

## Stage 0: QKV Tensor After `_to_replicated`

After the host round-trip described in Chapter 3, each chip holds the following tensor:

```
Tensor: qkv_replicated
  Logical shape: (1, 1, 1, 3072)
  Tile-padded shape: (1, 1, 32, 3072)   [padded to TILE_LAYOUT; 32 rows = 1 tile row]
  dtype: BF16
  layout: TILE_LAYOUT
  memory_config: DRAM_MEMORY_CONFIG (INTERLEAVED, DRAM buffer)
  distribution: ReplicateTensorToMesh — full tensor on every chip
```

The `3072` column dimension breaks down as:
```
3072 = (N_q + 2·N_kv) · H = (16 + 2·4) · 128 = 24 · 128
```

This tensor is the common starting point for all subsequent operations.

## Stage 1: Q/K/V Split by Head Group

The first operation on `qkv_replicated` is a slice (or `ttnn.split`) to separate the Q, K, and V head groups:

```python
# Conceptual split — actual implementation may use ttnn.slice or index notation
q_raw = qkv_replicated[:, :, :, :N_q * H]         # columns 0..2047
k_raw = qkv_replicated[:, :, :, N_q*H : (N_q+N_kv)*H]  # columns 2048..2559
v_raw = qkv_replicated[:, :, :, (N_q+N_kv)*H :]   # columns 2560..3071
```

Column extents:
```
Q columns: 0    to 2047  (16 heads × 128 = 2048 elements)
K columns: 2048 to 2559  ( 4 heads × 128 =  512 elements)
V columns: 2560 to 3071  ( 4 heads × 128 =  512 elements)
```

After slicing, each sub-tensor retains the same DRAM INTERLEAVED memory config since `ttnn.slice` preserves the source memory config by default. The tile-padded shapes are:

```
q_raw: (1, 1, 32, 2048)  DRAM/ITVL  BF16 TILE_LAYOUT
k_raw: (1, 1, 32,  512)  DRAM/ITVL  BF16 TILE_LAYOUT
v_raw: (1, 1, 32,  512)  DRAM/ITVL  BF16 TILE_LAYOUT
```

No `ttnn.to_memory_config` is required here; `ttnn.slice` is not a memory-config transition. However, the slice does involve a data movement kernel that reads from DRAM and writes to a new DRAM or L1 buffer, so it has a non-zero cost (discussed in `transition_cost_model.md`).

## Stage 2: Reshape for Per-Head View

Before RoPE and QK normalisation, the Q and K tensors are reshaped from a flat head-channel layout into a per-head layout:

```python
# Q: from (1, 1, 32, 2048) → (1, N_q, 32, H) = (1, 16, 32, 128)
q_heads = ttnn.reshape(q_raw, (1, N_q, 1, H))   # logical: (1, 16, 1, 128)
# tile-padded: (1, 16, 32, 128)

# K: from (1, 1, 32, 512) → (1, N_kv, 32, H) = (1, 4, 32, 128)
k_heads = ttnn.reshape(k_raw, (1, N_kv, 1, H))  # logical: (1, 4, 1, 128)
# tile-padded: (1, 4, 32, 128)
```

Reshape in TTNN is a zero-copy metadata operation when the source tensor is contiguous and the new shape is compatible with the existing tile boundaries. When this condition holds, no data movement occurs and the memory config is unchanged. Both reshaped tensors remain in `DRAM/ITVL`.

## Stage 3: Transition T1 — DRAM INTERLEAVED to HEIGHT_SHARDED for RoPE

This is the first explicit memory-config transition. `TTNNRotaryPositionEmbedding` (the non-distributed variant selected when `partial_rotary_factor < 1.0`) requires its input in HEIGHT_SHARDED L1 layout. Q and K use separate shard specs (`rope_shard_mem_q` and `rope_shard_mem_k`) because they have different head counts and therefore different core grids, defined as:

### Definition of `rope_shard_mem_q` and `rope_shard_mem_k`

```python
# Q variant — 16 Q heads require 16 cores.
# The Wormhole chip has 8 columns (0–7), so 16 cores must span 2 rows.
# CoreCoord(0,0) → CoreCoord(7,1) covers 8 cols × 2 rows = 16 cores.
rope_shard_mem_q = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
        # grid covers 16 cores in a 2-row × 8-col rectangle (one core per Q head)
        shape=[T, H],       # = [32, 128]
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

# K variant — 4 KV heads require 4 cores.
# 4 cores fit within a single row of 8 columns, so a 1-row × 4-col grid suffices.
rope_shard_mem_k = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        # grid covers 4 cores in a 1-row × 4-col rectangle (one core per KV head)
        shape=[T, H],       # = [32, 128]
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

**Why this specific shard shape `(T, H)` = `(32, 128)`?**

The choice of `(TILE_SIZE, head_dim)` = `(32, 128)` as the shard shape is driven by three constraints that converge on this value:

1. **Tile alignment:** TTNN kernels operating on L1 data require the shard shape to be a multiple of the tile size in both dimensions. `TILE_SIZE=32` is therefore the minimum row count; one tile row is used because the decode token contributes exactly one row of Q/K data per head (seq_len=1, padded to 32).

2. **Head-local computation:** RoPE applies independently per head. Each shard contains exactly one head's worth of data: `head_dim=128` elements per row, one tile-row deep. A Tensix core with one such shard can execute RoPE for that head without reading data from any other core. This is the structural reason HEIGHT_SHARDED is used rather than WIDTH_SHARDED: sharding by height (i.e., by the head-row dimension) partitions the tensor along head boundaries, which aligns naturally with the per-head independence of RoPE.

3. **Core grid efficiency:** With shard shape `(32, 128)` and Q shape `(1, 16, 32, 128)` (tile-padded), the total number of shards for Q is `16` (one per head). On T3K chips with 80 Tensix cores, 16 cores are occupied — a small fraction of the available grid. Because the Wormhole chip has only 8 columns (0–7), the 16 Q-head cores are arranged as a 2-row × 8-column rectangle (`CoreCoord(0,0)` to `CoreCoord(7,1)`). For K with 4 KV heads, only 4 cores are needed and they fit within a single row (`CoreCoord(0,0)` to `CoreCoord(3,0)`). This occupancy is sparse but unavoidable at batch=1; increasing the shard height would not reduce core count since there is only one tile row of data.

The transition itself is:

```python
# Transition T1a: Q from DRAM/ITVL to L1/HS for RoPE
q_rope_in = ttnn.to_memory_config(q_heads, memory_config=rope_shard_mem_q)
# rope_shard_mem_q grid: 16 cores for 16 Q heads — 8 cols × 2 rows (CoreCoord(0,0)→CoreCoord(7,1))

# Transition T1b: K from DRAM/ITVL to L1/HS for RoPE
k_rope_in = ttnn.to_memory_config(k_heads, memory_config=rope_shard_mem_k)
# rope_shard_mem_k grid: 4 cores for 4 KV heads
```

Diagram of T1a (Q only):

```
DRAM (INTERLEAVED)                      L1 (HEIGHT_SHARDED)
─────────────────────                   ─────────────────────────────────────────
q_heads                                 q_rope_in
shape: (1,16,32,128) tile-padded  ──►  shard per core: (32,128) BF16
DRAM buffer, tile-interleaved           16 cores (8 cols × 2 rows), each holds 1 head's tile
                           ttnn.to_memory_config(q_heads, rope_shard_mem_q)
```

The data movement path for T1 is **DRAM → L1**: tiles are read from DRAM and written into L1 banks on the target cores. This is the most expensive class of transition because it traverses the chip's NOC (network-on-chip) from DRAM controllers to Tensix L1.

## Stage 4: RoPE Execution (HEIGHT_SHARDED In-Place or To New Buffer)

`TTNNRotaryPositionEmbedding` executes with `q_rope_in` and `k_rope_in` in L1. The cos/sin tables are pre-computed and are also in L1 (or DRAM, depending on sequence position caching). The kernel reads from and writes back to L1-sharded buffers. With `partial_rotary_factor < 1.0`, only the first `rotary_dim = int(partial_rotary_factor * head_dim)` elements of each head are rotated; the remaining elements are passed through unchanged.

Post-RoPE output shapes (tile-padded):
```
q_rope_out: (1, 16, 32, 128)  L1/HS, shard=(32,128), 16 cores
k_rope_out: (1,  4, 32, 128)  L1/HS, shard=(32,128),  4 cores
```

V is not passed through RoPE. It remains in `DRAM/ITVL` at `(1, 1, 32, 512)` from Stage 2 (or a pre-split form).

## Stage 5: Transition T2 — L1 HEIGHT_SHARDED to DRAM INTERLEAVED (Post-RoPE)

After RoPE, Q and K are evicted from L1 to DRAM INTERLEAVED as an intermediate step. For Q, this eviction (T2a) is driven by the requirement of `TTNNRMSNorm`: when `use_qk_norm=True`, `TTNNRMSNorm` requires Q in DRAM INTERLEAVED (or a 2D INTERLEAVED L1 format — see Chapter 6), not the HEIGHT_SHARDED L1 layout produced by RoPE. When `use_qk_norm=False`, T2a may not be required, or may be driven by a different consumer. For K, T2b is an intermediate step before re-sharding for `paged_update_on_device`.

```python
# Transition T2a: Q post-RoPE, L1/HS → DRAM/ITVL
q_post_rope = ttnn.to_memory_config(q_rope_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Transition T2b: K post-RoPE, L1/HS → DRAM/ITVL
k_post_rope = ttnn.to_memory_config(k_rope_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

The data movement path for T2 is **L1 → DRAM**: tiles are evicted from L1 banks and written to DRAM. This is symmetric with T1 but in the opposite direction; cost is similar.

**Note on QK normalisation:** When `use_qk_norm=True` (which is the case for Ling), additional transitions occur around `TTNNRMSNorm` for both Q and K. These are analysed in detail in Chapter 6 (`qk_norm_latency.md`). For the purposes of this chapter's lifecycle diagram, QK norm is treated as occurring between T2 and T3, and is represented as a pair of transitions (T_norm_in, T_norm_out) that temporarily move Q and K to a 2D INTERLEAVED L1 format for the norm kernel. The shapes and cost of those transitions are included in the total count in `transition_cost_model.md`.

## Stage 6: Transition T3 — Re-Shard K and V for `paged_update_on_device`

Before the KV cache can be updated, K and V must be in the memory config expected by `paged_update_on_device`. The paged update kernel writes the current token's K and V vectors into the appropriate page slots in the KV cache. It requires K and V in a specific L1-sharded layout aligned to the cache's page granularity.

The required memory config for `paged_update_on_device` inputs depends on the cache page size and the number of KV heads. For Ling's `N_kv=4`, `head_dim=128`, page_size=P (typically 32 or 64 tokens), the required layout is:

```python
kv_update_mem = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({ttnn.CoreRange(
            ttnn.CoreCoord(0, 0), ttnn.CoreCoord(N_kv - 1, 0)
        )}),  # 4 cores, one per KV head
        shape=[T, H],       # = [32, 128]
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

```python
# Transition T3a: K, DRAM/ITVL → L1/HS for paged_update_on_device
k_update_in = ttnn.to_memory_config(k_post_rope, memory_config=kv_update_mem)

# Transition T3b: V, DRAM/ITVL → L1/HS for paged_update_on_device
#
# IMPORTANT: v_raw has shape (1, 1, 32, 512) (tile-padded) but kv_update_mem
# uses a shard shape of (32, 128), which targets a per-head layout of
# (1, 4, 32, 128). These shapes are incompatible: ttnn.to_memory_config cannot
# perform a shape change — it only moves data between memory configs. The reshape
# from (1, 1, 32, 512) to (1, 4, 32, 128) is a REQUIRED prerequisite and must
# be applied before to_memory_config is called. Omitting this reshape will cause
# a shape mismatch error at runtime.
v_reshaped = ttnn.reshape(v_raw, (1, 4, 32, 128))
# v_reshaped: (1, 4, 32, 128)  DRAM/ITVL  BF16 TILE_LAYOUT
# (reshape is a zero-copy metadata operation when tile boundaries are compatible)

v_update_in = ttnn.to_memory_config(v_reshaped, memory_config=kv_update_mem)
```

The data movement path for T3 is **DRAM → L1** (same class as T1).

## Stage 7: `paged_update_on_device` Execution

The paged KV-cache update kernel writes `k_update_in` and `v_update_in` into the pre-allocated paged KV-cache buffers using the provided block tables. The KV cache is in DRAM, so this involves L1 → DRAM writes from within the kernel. After the update, `k_update_in` and `v_update_in` are no longer needed as standalone tensors.

## Stage 8: Transition T4 — SDPA Input Layout

`paged_sdpa_decode` (source: `ttnn/cpp/ttnn/operations/transformer/sdpa/`) reads its Q input and the paged KV cache directly. Q must be presented in a specific memory config; for the GQA decode case with `N_q=16`, `N_kv=4`, the expected layout is typically INTERLEAVED DRAM or a specific HEIGHT_SHARDED L1 layout depending on the kernel variant.

In the current implementation, Q is presented in DRAM INTERLEAVED (`q_post_rope` from T2a), which is the accepted format for the paged SDPA decode kernel operating in GQA mode:

```python
# Q for SDPA: q_post_rope already in DRAM/ITVL — no further transition needed
# if the kernel accepts DRAM INTERLEAVED input.

# If the kernel requires L1 sharded input, there would be an additional
# Transition T4: DRAM/ITVL → L1/HS for SDPA
sdpa_q = ttnn.to_memory_config(q_post_rope, memory_config=sdpa_q_mem)
# This transition is conditional on the SDPA kernel's input requirements.
```

The paged KV-cache buffers (the actual page pool) are maintained in DRAM by the paging runtime and are accessed directly by the SDPA kernel through block table lookups. No explicit `to_memory_config` is required on the cache pool itself.

## Complete Transition Map

The diagram below summarises all transitions in reading order:

```
                           DECODE STEP — TENSOR MEMORY CONFIG LIFECYCLE
                           ═══════════════════════════════════════════════

[Stage 0]  qkv_replicated          (1,1,32,3072)   DRAM/ITVL    ← from _to_replicated
              │
              ├─ ttnn.slice ──────► q_raw           (1,1,32,2048)  DRAM/ITVL
              │                    k_raw            (1,1,32, 512)  DRAM/ITVL
              │                    v_raw            (1,1,32, 512)  DRAM/ITVL
              │
              ├─ ttnn.reshape ────► q_heads         (1,16,32,128)  DRAM/ITVL
              │                    k_heads          (1, 4,32,128)  DRAM/ITVL
              │
[T1a]       ttnn.to_memory_config ► q_rope_in      (1,16,32,128)  L1/HS(16c,(32,128))
[T1b]       ttnn.to_memory_config ► k_rope_in      (1, 4,32,128)  L1/HS( 4c,(32,128))
              │
              │   [RoPE kernel executes in L1]
              │
[T2a]       ttnn.to_memory_config ► q_post_rope    (1,16,32,128)  DRAM/ITVL
[T2b]       ttnn.to_memory_config ► k_post_rope    (1, 4,32,128)  DRAM/ITVL
              │
              │   [QK Norm: T_norm_in / T_norm_out — see Chapter 6]
              │   (additional transitions for RMSNorm around Q and K)
              │
[T3a]       ttnn.to_memory_config ► k_update_in   (1, 4,32,128)  L1/HS( 4c,(32,128))
[T3b]       ttnn.to_memory_config ► v_update_in   (1, 4,32,128)  L1/HS( 4c,(32,128))
              │
              │   [paged_update_on_device writes K,V to paged cache in DRAM]
              │
[T4]        (conditional) ────────► sdpa_q        (1,16,32,128)  L1/HS or DRAM/ITVL
              │
              └──► paged_sdpa_decode (reads Q from above + paged KV cache from DRAM)
```

Table: Summary of explicit memory-config transitions per decode step

| ID | Tensor | Source MemoryConfig | Destination MemoryConfig | Data path | Notes |
|---|---|---|---|---|---|
| T1a | Q | DRAM/ITVL `(1,16,32,128)` | L1/HS shard=(32,128), 16 cores (8 cols × 2 rows) | DRAM→L1 | Required by RoPE kernel |
| T1b | K | DRAM/ITVL `(1, 4,32,128)` | L1/HS shard=(32,128),  4 cores (4 cols × 1 row)  | DRAM→L1 | Required by RoPE kernel |
| T2a | Q | L1/HS shard=(32,128), 16 cores (8 cols × 2 rows) | DRAM/ITVL `(1,16,32,128)` | L1→DRAM | Post-RoPE eviction |
| T2b | K | L1/HS shard=(32,128),  4 cores (4 cols × 1 row)  | DRAM/ITVL `(1, 4,32,128)` | L1→DRAM | Post-RoPE eviction |
| T3a | K | DRAM/ITVL `(1, 4,32,128)` | L1/HS shard=(32,128),  4 cores | DRAM→L1 | Required by `paged_update_on_device` |
| T3b | V | DRAM/ITVL `(1, 1,32,512)` → reshaped | L1/HS shard=(32,128),  4 cores | DRAM→L1 | Required by `paged_update_on_device` |
| T4 | Q | DRAM/ITVL `(1,16,32,128)` | L1/HS or DRAM/ITVL | DRAM→L1 or no-op | Conditional on SDPA input requirements |

Note: QK norm transitions (two additional DRAM→L1 and L1→DRAM pairs, one each for Q and K) are enumerated in Chapter 6, `qk_norm_latency.md`, and included in the total cost in `transition_cost_model.md` as T_norm.

## Source Code Locations

The `ttnn.to_memory_config` calls described above are located in:

```
models/tt_transformers/tt/attention.py          # TTNNBailingMoEAttention.forward()
models/tt_transformers/tt/rope.py               # TTNNRotaryPositionEmbedding
models/tt_transformers/tt/norm.py               # TTNNRMSNorm input prep
```

The `rope_shard_mem_q` and `rope_shard_mem_k` configurations are typically constructed in the model initialisation (`__init__`) and cached as instance attributes so they are not re-allocated on every forward pass. The Q variant is shown below:

```python
# Typical pattern in TTNNBailingMoEAttention.__init__
self.rope_shard_mem_q = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        grid=ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))
            # 8 cols × 2 rows = 16 cores for 16 Q heads
            # NOTE: CoreCoord(self.num_heads - 1, 0) would resolve to CoreCoord(15, 0)
            # for num_heads=16, which exceeds the Wormhole chip's 8-column limit (0–7)
            # and causes a runtime failure. The corrected layout uses 2 rows of 8 cores.
        }),
        shape=[32, self.head_dim],
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    ),
)
```

---

**Next:** [Transition Cost Model](./transition_cost_model.md)
