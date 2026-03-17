# TTNN Programming Model

This file explains how TTNN represents tensors, how you control where and how they are stored in memory, and how a Python `ttnn` call becomes compute on a Tensix grid. Understanding this machinery is essential for reasoning about memory pressure and kernel dispatch overhead in MoE workloads.

---

## 1. Tensors in TTNN

### 1.1 Shapes and Named Dimensions

TTNN tensors have shapes written as `[dim0, dim1, ...]`. Throughout this guide, dimension names are used consistently:

| Symbol | Meaning |
|--------|---------|
| `batch` | Batch size |
| `seq` | Sequence length (number of tokens in the input) |
| `d_model` | Model hidden dimension |
| `d_ff` | Expert feed-forward hidden dimension |
| `n_experts` | Total number of experts in the MoE layer |
| `top_k` | Number of experts each token is routed to |

A token activation tensor entering an MoE layer has shape `[batch, seq, d_model]`. After the router selects `top_k` experts per token, a dispatched batch for a single expert has shape `[tokens_for_expert, d_model]` — where `tokens_for_expert` varies per expert and per forward pass.

TTNN tensors are created with explicit shape, dtype, and layout:

```python
import ttnn
import torch

# Host tensor → TTNN tensor on device
activation = ttnn.as_tensor(
    torch_activation,                 # PyTorch tensor on CPU
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,          # must be tile layout for matmul
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

### 1.2 Dtypes: bfloat16 and bfloat8_b

TTNN supports several numeric formats. For MoE workloads the two most important are:

**`ttnn.bfloat16`** (BF16): 16-bit brain float. Sign bit, 8 exponent bits, 7 mantissa bits. Same dynamic range as float32 but half the storage. Each tile is `32 × 32 × 2 = 2048 bytes = 2 KB`.

**`ttnn.bfloat8_b`** (BFP8): 8-bit block floating point. Each value stores 1 sign bit + 7 mantissa bits = 8 bits total. The mantissa field is 7 bits wide — the same width as BF16's mantissa — but unlike BF16, BFP8 does not store a per-value exponent; instead it uses a shared block exponent stored once per **16-value block** (one half-row of the 32×32 tile, i.e., 16 consecutive values in row-major order). A 32×32 tile holds 1024 values, so there are 64 separate shared exponents per tile — one for each group of 16 contiguous values. This per-16-value exponent granularity means values only need to fit within the dynamic range defined by their local group of 16, not the full tile's dynamic range. Each tile stores 1024 value bytes plus 64 exponent bytes (one per 16-value block) = **1088 bytes** of total storage. This reduces L1 pressure by ~1.88× relative to BF16 and significantly increases effective DRAM bandwidth for weight loading.

The trade-off is accuracy: BFP8 introduces quantization error from the shared-exponent representation. For MoE expert FFN weights, BFP8 is commonly used with acceptable accuracy loss (typically < 0.5% degradation on benchmark tasks). For router logits and attention computations, BF16 is generally preferred to avoid routing instability.

> **Tip:** Switching weight tensors from BF16 to BFP8 is one of the highest-leverage single changes for improving memory-bound MoE throughput. Before doing so, verify accuracy on a held-out validation set — don't assume the accuracy loss is acceptable for your specific model.

```python
# BFP8 weight tensor — reduces L1 and DRAM pressure by ~1.88× vs. BF16
weight = ttnn.as_tensor(
    torch_weight,
    dtype=ttnn.bfloat8_b,             # 1088 bytes per tile instead of 2048 bytes (BF16)
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

### 1.3 Tile Layout vs. Row-Major Layout

TTNN tensors can be in two layouts:

**`ttnn.TILE_LAYOUT`**: Data is stored in **32×32 tile order**. The last two dimensions of the tensor are tiled in row-major order within each tile, and tiles themselves are arranged in row-major order. This is the only layout that can be fed directly to the FPU for matmul or convolution.

**`ttnn.ROW_MAJOR_LAYOUT`**: Data is stored in the standard C-contiguous (row-major) order. Some TTNN ops accept row-major tensors directly; others require tile layout and will raise an error or silently convert (with a performance cost) if given row-major input.

> **Warning:** Implicit layout conversion between ROW_MAJOR_LAYOUT and TILE_LAYOUT is a common source of unexpected overhead. Always pass `layout=ttnn.TILE_LAYOUT` explicitly when constructing tensors that will be inputs to `ttnn.matmul`. Check for layout mismatch warnings in TTNN's verbose logging (`TTNN_ENABLE_LOGGING=1`).

The tile layout requirement means **tensor dimensions must be padded to multiples of 32**. If your `d_model` is 4096, that is already a multiple of 32 (`4096 / 32 = 128 tiles`). If your `d_model` were 3000, TTNN would need to pad it to 3008 (the next multiple of 32), wasting some memory and compute. For MoE models, standard `d_model` values (2048, 4096, 8192) are always multiples of 32 by design.

---

## 2. Memory Configs

Memory configs in TTNN control two things: **where** a tensor lives (DRAM or L1), and **how** it is distributed across the storage medium (interleaved or sharded).

### 2.1 DRAM_MEMORY_CONFIG

```python
ttnn.DRAM_MEMORY_CONFIG
```

The default. Tensor pages are distributed (interleaved) across all available DRAM channels. Reads and writes go through the DRAM controllers; the tensor does not reside in any core's L1 except transiently during computation.

Use DRAM_MEMORY_CONFIG for:
- Tensors too large to fit in L1 (most weight matrices in MoE models)
- Tensors that are only read or written once per forward pass
- Intermediate tensors whose consumers are not the immediately next kernel

### 2.2 L1_MEMORY_CONFIG

```python
ttnn.L1_MEMORY_CONFIG
```

The tensor's pages are placed in L1 SRAMs, interleaved across cores. This avoids DRAM round-trips for short-lived intermediates that are immediately consumed by the next kernel on the same or neighboring cores.

Use L1_MEMORY_CONFIG for:
- Small intermediate tensors (e.g., router logits, softmax outputs) that are produced and consumed within the same kernel sequence
- Tensors that will be fed into a subsequent kernel where L1 input is faster than DRAM input

> **Warning:** L1_MEMORY_CONFIG (interleaved) distributes tensor pages across many cores' L1s. If the tensor is only consumed by a single core, the pages living on remote cores must be fetched via the NoC. Whether this is faster or slower than DRAM depends on NoC congestion and tensor size. Profile before assuming L1_MEMORY_CONFIG is always better than DRAM_MEMORY_CONFIG.

### 2.3 Interleaved vs. Sharded Placement

The distinction between interleaved and sharded is about **which core owns which part of the tensor**:

**Interleaved**: Tensor pages are striped across all banks (DRAM banks or L1s of many cores) in a round-robin pattern. No single core owns any particular logical slice of the tensor. This maximizes bandwidth for streaming access but means any single core must issue NoC reads to collect its slice before computing.

**Sharded**: Each core owns a specific, contiguous slice of the tensor, pinned to that core's L1. The assignment is explicit — you specify the shard shape and the core grid over which the tensor is sharded.

```python
# Example: shard a [num_tokens, d_model] activation tensor across an 8-core grid
# Each core owns [num_tokens / 8, d_model] of the activations
shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    [num_tokens // 8, d_model],       # shard shape per core
    ttnn.ShardOrientation.ROW_MAJOR,  # shards are assigned in row-major core order
    False,                            # do not halo (no overlap between shards)
)
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # sharding along the height (row) dimension
    ttnn.BufferType.L1,
    shard_spec,
)
activation_sharded = ttnn.as_tensor(
    torch_activation,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=memory_config,
)
```

Sharding is the mechanism that eliminates DRAM round-trips for activations in fully fused kernel pipelines. When the activation tensor entering an expert matmul is already sharded in L1 across the expert's assigned core grid, the kernel can start computing immediately without issuing any DRAM reads. Chapter 5 covers sharding strategies for MoE expert weights in depth.

> **Tip:** A tensor must be sharded on the same core grid that the consuming kernel uses. Passing a tensor sharded on a `CoreGrid(4, 8)` to a kernel configured for `CoreGrid(8, 4)` will result in either a runtime error or an implicit and expensive reshard operation. Always verify grid alignment between tensor sharding and kernel config.

---

## 3. The Op Dispatch Model

### 3.1 From Python Call to Metalium Kernel

When you call `ttnn.matmul(A, B, ...)` in Python, the following sequence occurs:

1. **Python dispatch**: TTNN's Python layer validates tensor shapes, dtypes, and memory configs. It selects the appropriate program config (either from what you passed explicitly or from an auto-selection heuristic).

2. **Program selection**: TTNN looks up a precompiled kernel program that matches the op type and config. If the program is not in the program cache (see section 3.2), it compiles the Metalium kernel from source.

3. **Kernel compilation (if needed)**: Metalium JIT-compiles the RISC-V kernel programs for BRISC, NCRISC, and TRISC0/1/2. This compilation can take hundreds of milliseconds to several seconds for a complex kernel. The compiled binary is cached to disk to avoid recompilation on subsequent runs.

4. **Command queue dispatch**: The compiled kernel is enqueued on the device's command queue. Tensor addresses (L1 or DRAM physical addresses), grid coordinates, and program config parameters are embedded as runtime arguments.

5. **Device execution**: The Tensix cores in the assigned grid execute the BRISC/NCRISC/TRISC programs. NCRISC streams tiles from DRAM into L1 circular buffers. TRISC0/1 unpack tiles for the FPU. The FPU executes tile-level multiply-accumulate operations. TRISC2 packs results back to L1. NCRISC writes completed output tiles to DRAM (or leaves them in L1 if the output memory config is L1).

6. **Python return**: TTNN returns a Python tensor handle. The actual compute may not be complete yet — TTNN uses asynchronous dispatch. The result tensor is only guaranteed to be valid after `ttnn.synchronize_device(device)` or when the tensor is read back to host.

```python
import ttnn
import torch

# This call enqueues work on the device asynchronously
output = ttnn.matmul(
    activation,   # [M, K] tensor in DRAM or L1
    weight,       # [K, N] tensor in DRAM or L1
    dtype=ttnn.bfloat16,
    core_grid=ttnn.CoreGrid(y=4, x=8),
)

# output is a tensor handle; device is still computing
# Only needed if you want to read the result back to CPU right now
ttnn.synchronize_device(device)
result_torch = ttnn.to_torch(output)
```

### 3.2 Tracing and Program Caching

**Program caching** is TTNN's mechanism for reusing compiled kernel programs across calls with identical configurations. A program cache key is formed from:
- The op type (matmul, add, etc.)
- The program config (including `per_core_M`, `per_core_N`, `out_subblock_h`, `out_subblock_w`, etc.)
- The tensor shapes
- The tensor dtypes
- The memory configs (DRAM vs. L1, interleaved vs. sharded)

If any of these change between calls, the program cache misses and the kernel must be re-selected (and possibly recompiled). This is why **static shapes matter for performance**: a matmul where `M` varies between calls (because different numbers of tokens are dispatched to different experts in each forward pass) will either cause repeated program cache misses or require padding to a fixed maximum M.

> **Warning:** Dynamic token counts in MoE routing are the primary source of program cache misses in practice. If `tokens_for_expert` varies call to call, consider padding all expert batches to the same size (the maximum expected token count or a fixed power of two), accepting the wasted compute in exchange for program cache hits. Chapter 3 discusses this trade-off in detail.

**Tracing** is a stronger optimization than program caching. In TTNN, you can trace a forward pass:

```python
# Capture a trace of a full forward pass with fixed shapes
tid = ttnn.begin_trace_capture(device, cq_id=0)

# Run the forward pass once to capture all kernel dispatches
output = model_forward(inputs_trace)  # all shapes must be static during this call

ttnn.end_trace_capture(device, tid, cq_id=0)

# Subsequent calls replay the captured trace with near-zero Python overhead
# Input tensors must use the same memory addresses captured during trace
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
```

A traced forward pass bypasses Python entirely on replay: the captured sequence of Metalium commands is replayed from a pre-built command buffer on the device. This eliminates all Python overhead (typically 10–50 µs per op) and is the highest-performance execution mode.

The strict requirement of tracing is that **all tensor shapes and memory addresses must be identical on every replay**. For MoE models this typically means:
- Token counts are padded to a fixed size
- Expert assignments (which expert each token goes to) are not changed after capture
- The routing decision itself must happen outside the trace (since it is dynamic by nature)

A common pattern is to trace only the expert FFN computation (which has static weight shapes) and run the router and token dispatch outside the trace.

---

## 4. Async Dispatch and the Command Queue

TTNN uses an asynchronous dispatch model. Operations are enqueued on a command queue and executed on the device while the Python thread continues. Two command queues are available (cq_id 0 and cq_id 1), which can be used to pipeline data transfers and compute:

```python
# Enqueue a data prefetch on queue 1 while compute runs on queue 0
ttnn.copy_host_to_device_tensor(next_batch, device, cq_id=1)  # prefetch next batch
output = ttnn.matmul(activation, weight, ...)                  # compute on queue 0 (default)

# Synchronize only when the result is needed
ttnn.synchronize_device(device, queue_id=0)
```

For MoE workloads, dual-queue dispatch can be used to overlap the next batch's token dispatch (memory transfer) with the current batch's expert FFN computation (compute). This is an advanced optimization covered in Chapter 7.

---

## Summary

| Concept | Key Fact |
|---------|----------|
| Tile layout requirement | Must be `TILE_LAYOUT` for matmul; dimensions padded to multiples of 32 |
| BF16 tile size | 2 KB (32×32 × 2 bytes) |
| BFP8 tile size | 1088 bytes (1024 value bytes + 64 exponent bytes) |
| DRAM_MEMORY_CONFIG | Interleaved across DRAM channels; default for large tensors |
| L1_MEMORY_CONFIG | Interleaved across core L1s; best for small intermediates |
| Sharded placement | Tensor slices pinned to specific cores' L1; eliminates DRAM reads for activations |
| Program cache key | Op type + config + shapes + dtypes + memory configs |
| Tracing | Captures full forward pass for near-zero Python overhead on replay; requires static shapes |

---

---

**Next:** [matmul_fundamentals_in_ttnn.md](./matmul_fundamentals_in_ttnn.md)
