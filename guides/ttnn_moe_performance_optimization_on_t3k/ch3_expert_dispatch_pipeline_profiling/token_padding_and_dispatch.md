# Token Padding and All-to-All Dispatch

## Context

This file addresses:
- **Q2** — Which stage of the expert dispatch pipeline dominates latency at batch=1 decode?

Specifically, it covers Stages 1 and 2 of `TTNNExperts.forward`:

- **Stage 1** (`moe.py:L1191–L1212`): Token padding to `SPARSITY_BLOCK_SIZE=32`.
- **Stage 2** (`moe.py:L1214–L1230`): Layout conversion to `ROW_MAJOR` and `ttnn.all_to_all_dispatch`.

---

## Stage 1: Token Padding (moe.py:L1191–L1212)

### What happens

At decode time, the input `x` has shape `(1, 1, batch_size * seq_len, hidden_size)`. For batch=1 decode with a single token, that is `(1, 1, 1, hidden_size)`. The padding step expands the token dimension to the next multiple of `SPARSITY_BLOCK_SIZE=32`:

```python
# moe.py:L52
SPARSITY_BLOCK_SIZE = 32

# moe.py:L1191–1212 (schematic)
pad_to = SPARSITY_BLOCK_SIZE
n_tokens = batch_size * seq_len            # = 1 at decode
n_padded = ceil(n_tokens / pad_to) * pad_to  # = 32

x = ttnn.pad(x, [..., n_padded, ...], value=0)
topk_experts_indices = ttnn.pad(topk_experts_indices, [..., n_padded, ...], value=0)
topk_experts_weights = ttnn.pad(topk_experts_weights, [..., n_padded, ...], value=0)
```

After padding, `x` has shape `(1, 1, 32, hidden_size)` for the batch=1 decode case.

Three separate pad operations run: one on the activation tensor, one on the expert index tensor, and one on the expert weight tensor.

### SPARSITY_BLOCK_SIZE=32: rationale and constraints

The value 32 equals the TTNN tile size (`ttnn.TILE_SIZE = 32`). The sparse matmul kernels operate on blocks of tokens; each block must align to a tile boundary because `MatmulMultiCoreReuseMultiCast1DProgramConfig` requires `per_core_M` in whole tiles — `per_core_M=1` means each core processes exactly one 32-row tile-row of the activation matrix. A sub-tile block would make `per_core_M` fractional, which the config does not support.

Two additional couplings make reducing the constant unsafe:

1. The `moe_expert_token_remap` op (`moe.py:L1238–L1245`) generates `sparsity_t`, a metadata structure whose layout depends on `SPARSITY_BLOCK_SIZE`. Changing the constant without updating the kernel produces malformed metadata.
2. `TOPK_MIN_WIDTH=64` (`moe.py:L51`) is twice `SPARSITY_BLOCK_SIZE` by convention and controls the token-count threshold that activates the sparse path. Halving one without halving the other breaks the dispatch-vs-sparse branching logic.

`SPARSITY_BLOCK_SIZE=32` is therefore the minimum safe value. The appropriate response to padding overhead is to amortize it across more tokens (increase batch size) or eliminate redundant pad/unpad round-trips — not to reduce the constant. At batch=1 decode, the single real token is padded to 32 tokens (`SPARSITY_BLOCK_SIZE`), so every expert matmul processes 32 rows regardless of actual occupancy.

---

## Stage 2: Layout Conversion and all_to_all_dispatch (moe.py:L1214–L1230)

### Layout conversion (moe.py:L1214–L1223)

Before calling `ttnn.all_to_all_dispatch`, the padded tensors must be converted to `ROW_MAJOR` layout. `all_to_all_dispatch` operates on row-major data; the activation tensor arrives in `TILE_LAYOUT` from the upstream gate projection. Three layout conversions happen:

```python
# moe.py:L1214–1223 (schematic)
x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
# topk_experts_weights does not require layout conversion here
```

The layout conversion for `x` operates on a `(1, 1, 32, hidden_size)` tile-layout tensor and produces a `(1, 1, 32, hidden_size)` row-major tensor. For hidden=4096, this is `32 × 4096 × 2 bytes = 262 144 B = 256 KB`. The conversion itself is a data-movement op in SRAM/DRAM and is expected to be cheap (< 5 µs), but it is worth confirming in measurement.

### all_to_all_dispatch (moe.py:L1225–L1230)

```python
# moe.py:L1225–1230
x_dispatched, expert_mapping_tensors = ttnn.all_to_all_dispatch(
    x_rm,
    topk_experts_indices_rm,
    expert_mapping_tensors,
    cluster_axis=1,
)
```

`all_to_all_dispatch` routes each token to its assigned expert device. With `cluster_axis=1`, the routing occurs along the 8-device axis of the T3K mesh.

**What it does:**

- Each device holds the full padded activation tensor (32 tokens, post-all-gather from Chapter 2).
- Each token has a `topk_experts_indices` entry specifying which expert(s) it should be processed by.
- The dispatch op routes tokens to the device(s) hosting their assigned experts. A token assigned to expert 37 (device 2 in an 8-device setup with 128 experts uniformly distributed: 128/8=16 experts per device) is sent from all 8 devices to device 2.
- `expert_mapping_tensors` carries metadata used by the corresponding `all_to_all_combine` to route outputs back.

**Message size at batch=1 decode:**

Each device sends at most `ceil(32 × topk / n_devices)` tokens to each destination. For topk=2, 32 padded tokens, 8 devices:

```
tokens_per_device = 32 × 2 / 8 = 8 tokens per destination device (average)
message_per_hop = 8 × hidden_size × 2 bytes = 8 × 4096 × 2 = 65 536 B = 64 KB
```

This is larger than the all-gather message (8 KB) by roughly 8×. The dispatch is therefore more bandwidth-intensive than the all-gather, and topology choice matters more here.

**Topology and cluster_axis:**

Unlike `all_gather_async` (which operates without a `cluster_axis` parameter), `all_to_all_dispatch` uses `cluster_axis=1` explicitly, meaning the routing traverses the 8-device row of the mesh. The exact topology (Linear vs. Ring) used internally by `all_to_all_dispatch` depends on the TTNN CCL implementation and is not exposed as a call-site parameter; it should be confirmed from the op's documentation or source.

---

## Measuring Dispatch Latency in Isolation

### Method 1: TTNN op timer with synchronize barrier

```python
import ttnn
import time

# Build inputs matching TTNNExperts.forward Stage 1–2 shapes
# GLM-4-MoE: hidden=4096, padded_tokens=32, topk=2
padded_tokens = 32
hidden = 4096
topk = 2

x_rm = ttnn.from_torch(
    torch.zeros(1, 1, padded_tokens, hidden, dtype=torch.bfloat16),
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
indices_rm = ttnn.from_torch(
    torch.zeros(1, 1, padded_tokens, topk, dtype=torch.int32),
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Warmup
for _ in range(20):
    out, meta = ttnn.all_to_all_dispatch(
        x_rm, indices_rm, expert_mapping_tensors, cluster_axis=1
    )
    ttnn.synchronize_device(device)

# Timed measurement
N = 100
t0 = time.perf_counter()
for _ in range(N):
    out, meta = ttnn.all_to_all_dispatch(
        x_rm, indices_rm, expert_mapping_tensors, cluster_axis=1
    )
    ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"all_to_all_dispatch mean: {(t1 - t0) / N * 1e6:.1f} µs")
```

**Key discipline:** `ttnn.synchronize_device` after each call forces the async dispatch to complete before the timer captures `t1`. Without it, the measured time is the host-side enqueue latency, not the actual device execution.

### Method 2: Measure Stage 1 and Stage 2 separately

To decompose the padding cost from the dispatch cost, insert synchronization barriers between stages:

```python
# Stage 1 only: measure padding
t0 = time.perf_counter()
for _ in range(N):
    x_padded = ttnn.pad(x, padded_shape, value=0)
    ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"token padding mean: {(t1 - t0) / N * 1e6:.1f} µs")

# Stage 2 only: measure layout conversion + dispatch
t0 = time.perf_counter()
for _ in range(N):
    x_rm = ttnn.to_layout(x_padded, ttnn.ROW_MAJOR_LAYOUT)
    out, meta = ttnn.all_to_all_dispatch(x_rm, indices_rm, expert_mapping_tensors, cluster_axis=1)
    ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"layout_cast + dispatch mean: {(t1 - t0) / N * 1e6:.1f} µs")
```

### Method 3: Tracy zone annotations

To visualize the dispatch timeline relative to other stages, wrap the Stage 2 call site with a Tracy zone marker. See Chapter 5 for annotation syntax. The Tracy trace will show whether `all_to_all_dispatch` overlaps with any other device work or runs sequentially.

---

## Expected Observations and Interpretation

| Observation | Interpretation |
|---|---|
| Dispatch latency > 100 µs | Dispatch dominates; investigate expert-to-device mapping for load imbalance |
| Dispatch latency < 20 µs | Dispatch is not the bottleneck; focus on Stages 4–5 (sparse matmul) |
| Padding latency > 10 µs | Unexpected; three `ttnn.pad` ops on small tensors should be ~1–3 µs each |
| Layout conversion latency > 5 µs | Investigate SRAM vs DRAM placement; conversion may be stalling on DRAM bandwidth |

**Expert load imbalance at batch=1:** With only 1 real token and topk=2, exactly 2 expert devices receive dispatch traffic. The remaining 6 devices receive zero tokens (only padding tokens whose expert index is the filler value). This extreme imbalance means the dispatch cost is structurally a latency problem, not a throughput problem: the two active expert devices must process their dispatched tokens, and all 8 devices must wait for `all_to_all_dispatch` to complete before proceeding.

If the dispatch op returns a ragged result (different token counts per device), the `all_to_all_combine` in Stage 6 must handle the same imbalance in reverse. Both dispatch and combine latency should be measured together in the same isolation experiment for a complete Stage 2 + Stage 6 accounting.

---

**Next:** [`sparse_matmul_profiling.md`](./sparse_matmul_profiling.md)
