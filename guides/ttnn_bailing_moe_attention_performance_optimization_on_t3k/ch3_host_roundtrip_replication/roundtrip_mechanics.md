# Round-Trip Mechanics of `_to_replicated`

## Overview

`_to_replicated` is the internal helper in `TTNNBailingMoEAttention` that converts the all-reduced QKV output tensor from its post-all-reduce distribution type into the `ReplicateTensorToMesh` distribution that `paged_sdpa_decode` requires. The operation involves no arithmetic; it is a pure data movement operation across the PCIe bus. This file traces every sub-step in execution order, identifies the TTNN and PyTorch APIs involved, and quantifies the data volumes transferred per direction.

## Input Tensor: State After the Fused QKV All-Reduce

To understand why `_to_replicated` is needed, the precise state of the tensor at its input must be established.

After `TTNNLinearIColShardedWAllReduced` completes its all-reduce, each chip holds the full result of the fused QKV projection. At Ling's decode configuration (batch=1, seq_len=1), the logical tensor shape is:

```
QKV_all_reduced: shape (1, 1, 1, 3072)
  dim 0: batch = 1
  dim 1: sequence length = 1
  dim 2: (unused outer head dim, kept for layout compatibility) = 1
  dim 3: total head output channels = (Nq + 2·Nkv) · head_dim = (16 + 4 + 4) · 128 = 3072
dtype: BF16
```

The logical values on all 8 chips are numerically identical — the all-reduce has already summed the per-chip partial projections and distributed the result everywhere. However, **the TTNN runtime does not automatically promote a tensor to `ReplicateTensorToMesh` distribution just because all chips hold the same values**. The distribution type is a metadata property of the `ttnn.Tensor` object, not something inferred from content.

At this point, the tensor's TTNN multi-device type is tagged with the **all-reduce output memory config**, not `ReplicateTensorToMesh`. To be precise about what "column-sharded" means here: it describes the WEIGHT MATRIX used during the matmul (`TTNNLinearIColShardedWAllReduced` distributes its weight matrix column-wise, so each chip's weight shard is `(4096, 384)`). During the matmul each chip computes a PARTIAL output of shape `(1, 1, 1, 384)` — its local weight shard's contribution. The all-reduce (reduce-scatter + all-gather) then combines these partial outputs so that, by the time the all-reduce is complete, **every chip holds an identical full-sized copy of the result: `(1, 1, 1, 3072)` in BF16**. The column-sharded description therefore applies to the weight distribution during computation, not to the final all-reduce output tensor. The distinction is critical: `paged_sdpa_decode` inspects the distribution type at dispatch time and will fail or produce incorrect results if it receives a tensor tagged as anything other than `ReplicateTensorToMesh`.

## Why `paged_sdpa_decode` Requires Replicated Distribution

The `paged_sdpa_decode` kernel (source path: `ttnn/cpp/ttnn/operations/transformer/sdpa/`) is designed to operate fully independently on each chip. Each chip runs the attention computation against its local slice of the paged KV-cache. For this to be correct, every chip must see the **full Q, K, and V vectors** for the current decode token — not a shard.

The kernel's input validation expects its QKV argument to be a `ReplicateTensorToMesh`-distributed tensor. This distribution type guarantees, by construction, that each chip's local sub-tensor is a complete copy of the logical tensor, not a shard. A sharded tensor — even one where all shards are numerically identical — does not satisfy this contract because:

1. The per-chip sub-tensor of a sharded tensor is only a portion of the logical tensor shape. Note that `(1, 1, 1, 384)` is the intermediate PARTIAL result each chip holds during the matmul (before the all-reduce); it is NOT the all-reduce output. After the all-reduce completes, each chip holds the full `(1, 1, 1, 3072)` result. If this post-all-reduce tensor were nonetheless tagged as a shard (rather than replicated), the kernel would still misinterpret the per-chip shape and produce incorrect results.
2. The TTNN runtime uses the distribution type to determine memory layout and stride metadata passed to the kernel binary. Mismatched metadata causes incorrect address calculations in the Tensix kernels.

There is no TTNN primitive that reinterprets the distribution tag of an existing device-resident tensor in-place without a data movement operation. Therefore the conversion must materialise the full tensor and redistribute it.

## Step-by-Step Trace of `_to_replicated`

The implementation in `TTNNBailingMoEAttention` follows this sequence:

### Step 1: `ConcatMeshToTensor` — Device to Host

```python
# TTNN multi-device tensor → single torch.Tensor on CPU
qkv_host = ttnn.to_torch(
    qkv_all_reduced,
    mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
)
```

`ConcatMeshToTensor` with `dim=3` reads the local sub-tensor from each of the 8 chips and concatenates them along dimension 3 (the channel dimension) on the host. However, because this tensor was produced by an all-reduce — meaning every chip already holds the full `(1, 1, 1, 3072)` logical tensor — the concatenation along dim=3 would produce a `(1, 1, 1, 3072×8) = (1, 1, 1, 24576)` tensor on the host if the per-chip sub-tensors are the full shape.

In practice, the TTNN output memory config after the all-reduce leaves each chip's sub-tensor at `(1, 1, 1, 3072)` (the full logical size), so `ConcatMeshToTensor` with `dim=3` concatenates 8 identical `(1, 1, 1, 3072)` tensors into `(1, 1, 1, 24576)` on the host. A slice `[:, :, :, :3072]` (or equivalent) is then taken to recover the intended `(1, 1, 1, 3072)` tensor.

An alternative implementation uses `dim=-1` with a mesh composer configured to select only the first chip's data (`ttnn.ConcatMeshToTensor` with `concat_dim` matching the sharding axis for a truly sharded post-all-reduce output). The exact behaviour depends on the `output_memory_config` of the all-reduce, which determines whether each chip's local sub-tensor is `(1, 1, 1, 3072)` or `(1, 1, 1, 384)`. Regardless of the exact slice taken, the net result on the host is a single `torch.Tensor` of shape `(1, 1, 1, 3072)` in BF16.

**Data pulled from device to host (per direction, total across all chips):**

```
Per chip DRAM read:
  shape: (1, 1, 1, 3072), dtype: BF16 → 3072 × 2 bytes = 6,144 bytes = 6.0 KB

PCIe transfers from device to host:
  Number of chips transferring: 8
  Bytes per chip: 6,144 bytes
  Total bytes device→host: 8 × 6,144 = 49,152 bytes = 48.0 KB
  *** 48 KB of data is transferred across PCIe (8 identical 6 KB copies). ***
  *** After concatenation, only 6 KB of unique data is retained (the        ***
  *** remaining 7/8 of the 48 KB, i.e. 42 KB, is redundant and discarded.  ***
```

In a well-optimised `ConcatMeshToTensor` implementation, data from each chip arrives over its dedicated PCIe link. Each chip's PCIe connection to the host is independent, so the 8 transfers can proceed in parallel if the host PCIe root complex supports it. In practice on T3K, the transfers are largely serialised at the host DRAM write stage.

### Step 2: Torch Tensor Construction on Host

After `ttnn.to_torch` returns, the result is a standard CPU `torch.Tensor`:

```python
qkv_host: torch.Tensor
  shape: (1, 1, 1, 3072)
  dtype: torch.bfloat16
  device: cpu
  memory: host DRAM, contiguous
```

This step has no PCIe cost; it is a CPU-side operation that finalises the tensor descriptor in PyTorch. The cost is negligible (sub-microsecond for a 6 KB tensor) and is dominated by Python object allocation overhead, estimated at **< 5 µs** [ESTIMATE].

### Step 3: `from_torch` with `ReplicateTensorToMesh` — Host to Device

```python
qkv_replicated = ttnn.from_torch(
    qkv_host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`ReplicateTensorToMesh` instructs TTNN to copy the full `(1, 1, 1, 3072)` tensor to every chip's DRAM identically. The host PCIe controller initiates 8 independent DMA writes — one to each chip.

**Data pushed from host to device (per direction, total across all chips):**

```
Per chip write:
  shape: (1, 1, 1, 3072), dtype: BF16 → 6,144 bytes = 6.0 KB

PCIe transfers from host to device:
  Number of chips written to: 8
  Bytes per chip: 6,144 bytes
  Total bytes host→device: 8 × 6,144 = 49,152 bytes = 48.0 KB
```

**LAYOUT ASSUMPTION — Host→Device Transfer Size:** The 6 KB per-chip figure above assumes the tensor is transferred in **ROW_MAJOR_LAYOUT**, so no tile-padding is applied during the PCIe write. If the tensor is instead transferred in **TILE_LAYOUT** (as requested by the `layout=ttnn.TILE_LAYOUT` argument in `from_torch`), TTNN pads the tensor to 32×32 element tile boundaries before transmission:

```
TILE_LAYOUT padding for (1, 1, 1, 3072) BF16:
  Row dimension:    1 row  → padded to 32 rows  (nearest multiple of 32)
  Col dimension: 3072 cols → padded to 3072 cols (already a multiple of 32)
  Tile-padded elements: 32 × 3072 = 98,304
  Tile-padded byte count: 98,304 × 2 bytes = 196,608 bytes ≈ 192 KB per chip
```

This means that if the tensor is in TILE_LAYOUT at transfer time, the host→device PCIe transfer is **~192 KB per chip** (32× larger than the 6 KB unpadded figure), and the total across 8 chips is **~1.5 MB** [ESTIMATE].

The analysis in this chapter assumes the tensor is read and written in **ROW_MAJOR_LAYOUT** (6 KB per chip). If the tensor is in TILE_LAYOUT, substitute ~192 KB per chip into all latency calculations; at 20 GB/s practical PCIe throughput, 192 KB transfers in **~9.6 µs per chip** [ESTIMATE] rather than 0.3 µs.

### Step 4: Output Tensor State

After `from_torch` completes, `qkv_replicated` is a `ttnn.Tensor` with:

```
shape:        (1, 1, 1, 3072)
dtype:        BF16
layout:       TILE_LAYOUT
distribution: ReplicateTensorToMesh (all 8 chips hold full (1,1,1,3072))
memory:       DRAM_MEMORY_CONFIG on each chip
```

This tensor is ready for the Q/K/V slice operations that precede `paged_sdpa_decode`.

## Data Volume Summary

Table: PCIe data volumes transferred per `_to_replicated` call (Ling decode step, batch=1)

**Note: all figures below assume ROW_MAJOR_LAYOUT at transfer time (no tile-padding applied). If the tensor is in TILE_LAYOUT, substitute ~192 KB per chip for the host→device direction; see the layout assumption note in Step 3 above.**

| Direction | Per-chip bytes (ROW_MAJOR) | Per-chip bytes (TILE_LAYOUT) | Total bytes — 8 chips (ROW_MAJOR) | Notes |
|---|---|---|---|---|
| Device → Host | 6,144 B (6.0 KB) | ~196,608 B (~192 KB) [ESTIMATE] | 49,152 B (48.0 KB) | `ConcatMeshToTensor`; 8 identical copies transferred, only 6 KB unique data retained after concat+slice. See Step 3 for layout dependency. |
| Host → Device | 6,144 B (6.0 KB) | 196,608 B (~192 KB) [ESTIMATE] | 49,152 B (48.0 KB) | `ReplicateTensorToMesh`; same value written to each of 8 chips; if TILE_LAYOUT, ~192 KB per chip, ~1.5 MB total [ESTIMATE] |
| **Round-trip total (ROW_MAJOR)** | **12,288 B (12.0 KB)** | — | **98,304 B (96.0 KB)** | Sum of both directions assuming ROW_MAJOR_LAYOUT |

The payload is tiny in absolute terms under the ROW_MAJOR_LAYOUT assumption. The 6 KB per chip per direction figure is far below the threshold where PCIe bandwidth becomes the bottleneck — at 20 GB/s practical throughput, 6 KB transfers in **0.3 µs** [ESTIMATE]. If the tensor is in TILE_LAYOUT at transfer time, the host→device cost rises to **~9.6 µs per chip** [ESTIMATE] at the same bandwidth. Regardless of layout, the actual cost of `_to_replicated` is dominated by **PCIe transaction overhead and host synchronisation latency**, not by bandwidth. This is analysed in detail in `host_transfer_overhead.md`.

## Source Code Locations

The relevant TTNN APIs involved in this operation are:

```
ttnn/python/ttnn/multi_device.py          # ConcatMeshToTensor, ReplicateTensorToMesh
ttnn/python/ttnn/operations/core.py       # ttnn.to_torch, ttnn.from_torch
ttnn/cpp/ttnn/tensor/tensor.cpp           # Device-host DMA path
ttnn/cpp/ttnn/distributed/api.cpp         # Multi-device tensor distribution logic
```

The `TTNNBailingMoEAttention` attention layer implementation that calls `_to_replicated` is located in:

```
models/tt_transformers/tt/attention.py    # (path relative to tt-metal repo root)
```

---

**Next:** [Host Transfer Overhead](./host_transfer_overhead.md)
