# State Tensor Memory Layout for T3K

This file specifies the memory layout, sizing, and initialization of the recurrent state tensor `S` for the DeltaNet decode step on a T3K (8-device) system. It also covers the secondary conv state tensor and tile alignment analysis.

## State Tensor Shape and Size

The recurrent state `S` stores the association matrix for each attention head. Its full logical shape (across all devices) is:

```
S: [B, num_v_heads, d_k, d_v] = [1, 32, 128, 128]   dtype: BF16
```

Size per full logical tensor:

```
32 heads × 128 × 128 elements × 2 bytes (BF16) = 1,048,576 bytes = 1 MB per layer
```

## Head-Parallel Sharding on T3K

T3K has 8 devices. Under head-parallel sharding, each device owns `32 / 8 = 4` heads. Each device holds:

```
S_local: [B, nH_local, d_k, d_v] = [1, 4, 128, 128]   dtype: BF16
```

Size per device per layer:

```
4 heads × 128 × 128 × 2 bytes = 131,072 bytes = 128 KB per device per layer
```

> **Note:** The 128 KB figure is exact (131,072 bytes = 128 × 1,024 bytes). References to "3.84 MB" are a decimal approximation error — the correct total is computed below.

## Total DRAM for All DeltaNet Layers

Qwen3.6-35B-A3B has 30 DeltaNet (linear attention) layers.

```
30 layers × 128 KB = 3,840 KB = 3,840 / 1,024 MB = 3.75 MB per device
```

This is the total DRAM footprint of all recurrent state tensors on one device, at B=1 decode.

## Memory Configuration

```python
import ttnn

# State tensor memory config — must persist across dispatch calls
STATE_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG

# State tensor layout — d_k=d_v=128 are multiples of 32, no padding needed
STATE_LAYOUT = ttnn.TILE_LAYOUT

# Initialization at model load time (per device, per layer)
S_init = ttnn.zeros(
    shape=[1, 4, 128, 128],          # [B, nH_local, d_k, d_v]
    dtype=ttnn.bfloat16,
    layout=STATE_LAYOUT,
    device=device,
    memory_config=STATE_MEMORY_CONFIG,
)
```

## Why DRAM, Not L1

L1 is per-core scratchpad memory on each Tensix core. It is not guaranteed to retain values between separate kernel dispatches — the runtime may reclaim and reallocate L1 buffers between calls. Because the recurrent state `S` must survive from one decode step to the next (potentially thousands of steps), it must reside in DRAM, which persists for the lifetime of the tensor object.

Relevant properties:

| Property | DRAM | L1 |
|----------|------|----|
| Persists between dispatch calls | Yes | No |
| Capacity (per device) | ~12 GB (T3K) | ~1.5 MB per Tensix |
| Bandwidth to compute | Lower | Higher |
| Required for state | Yes | No |

The state is read from DRAM into L1 at the start of each decode step (ops 6 and 7) and written back to DRAM at the end (op 10). This round-trip is unavoidable but is entirely on-device — it does not involve the host CPU and is not trace-breaking.

> **Warning:** Do not store `S` in L1 between decode steps. An L1 allocation that is not explicitly retained will be reclaimed by the runtime dispatcher between kernel launches. The state will silently contain garbage values on the next step.

## L1 Feasibility During Kernel Execution

Although `S` must live in DRAM between steps, the computation itself reads one head's worth of state into L1 at a time. Per-head state size:

```
1 head × 128 × 128 × 2 bytes = 32,768 bytes = 32 KB per head
```

A single Tensix core has 1.5 MB of L1. Holding the per-head state (32 KB) plus the intermediate tensors for ops 6–11 is well within budget. The per-head working set peaks at approximately 65.5 KB at two non-overlapping moments: (1) during ops 6–7, S_prev and S_decayed are both live (2 × 32 KB = 64 KB); (2) during ops 9–10, S_decayed and write are both live (2 × 32 KB = 64 KB). S_prev is released after op 7, before write is produced at op 9 — they are never simultaneously in L1. Adding six intermediate vectors at 256 bytes each (~1.5 KB), the total peak is approximately 65.5 KB — well within a single Tensix core's 1.5 MB L1.

## Tile Alignment Analysis

TTNN tile operations require the innermost two dimensions to be multiples of 32.

| Dimension | Value | Tiles | Aligned? |
|-----------|-------|-------|----------|
| d_k | 128 | 4 | Yes |
| d_v | 128 | 4 | Yes |
| State matrix [d_k, d_v] | [128, 128] | 4 × 4 = 16 | Yes |

No padding is required for any of the 12 ops. The `[128, 128]` state matrix divides cleanly into 16 tiles of `[32, 32]` each, which is the favorable case for TTNN tile layout. Models with head dimensions that are not multiples of 32 would require explicit padding before allocating the state tensor.

## Conv State

In addition to the recurrence state `S`, the DeltaNet implementation includes a short convolution state for the input projections. Its shape and sizing:

```
conv_state: [B, mixed_dim/8, 4] = [1, 1024, 4]   dtype: BF16
```

Where `mixed_dim = 8,192` for Qwen3.6-35B-A3B, so `mixed_dim/8 = 1,024` per device under sharding.

```
Size per device per layer: 1 × 1,024 × 4 × 2 bytes = 8,192 bytes = 8 KB
Total for 30 layers: 30 × 8 KB = 240 KB per device
```

The conv state uses `ttnn.DRAM_MEMORY_CONFIG` for persistence, but unlike the recurrence state it must use `ttnn.ROW_MAJOR_LAYOUT` because its last dimension is 4, which is not a multiple of 32 and is therefore incompatible with TTNN tile layout. Initialization:

```python
conv_state_init = ttnn.zeros(
    shape=[1, 1024, 4],
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

## Summary of DRAM Footprint per Device

| Tensor | Shape (per device, per layer) | Size per layer | 30 layers |
|--------|-------------------------------|----------------|-----------|
| Recurrence state S | [1, 4, 128, 128] BF16 | 128 KB | 3,840 KB = 3.75 MB |
| Conv state | [1, 1024, 4] BF16 | 8 KB | 240 KB |
| **Total** | | **136 KB** | **4,080 KB ≈ 3.98 MB** |

The combined DRAM overhead for all persistent state across all DeltaNet layers is under 4 MB per device on T3K, representing a negligible fraction of the 12 GB per-device DRAM capacity.

## Forward Reference

Chapter 3 covers how these DRAM tensors are managed under `ttnn.graph_trace` — specifically the buffer aliasing pattern required to update `S` in-place across decode steps without creating new allocations that would break the static trace.
