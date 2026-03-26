# Weight Application Overhead

## Context

This file addresses:
- **Q5** — What is the cost of the post-combine weight application step, and is there a lower-overhead alternative?
- **Q2** — Which stage of the expert dispatch pipeline dominates latency at batch=1 decode?

Source range: `moe.py:L1315–L1341` (Stage 7 of `TTNNExperts.forward`).

---

## What Stage 7 Does

After `ttnn.all_to_all_combine` assembles expert outputs back on their originating devices (`moe.py:L1307–L1312`), Stage 7 applies the router's per-expert weights to each token's expert output and sums across the top-k dimension.

The router produces two tensors per token:
- `topk_experts_indices`: which expert(s) processed this token.
- `topk_experts_weights`: scalar weights (e.g., softmax scores) that determine how much each expert's output contributes to the final token representation.

For topk=2, each token has two expert outputs and two corresponding weights. The final token representation is the weighted sum:

```
token_output = w1 × expert1_output + w2 × expert2_output
```

### The Current Implementation (moe.py:L1321–L1335)

The combined output after `all_to_all_combine` and reshape has shape:

```
(topk, batch_size * seq_len, 1, hidden_size)
```

For batch=1 decode with topk=2 and hidden=4096, this is `(2, 1, 1, 4096)`.

The weight application sequence is (`moe.py:L1321–L1335`, schematic):

```python
# Shape entering Stage 7: (topk, n_tokens, 1, hidden_size) = (2, 1, 1, 4096)
# topk_experts_weights shape after unsqueeze: (n_tokens, topk, 1, 1) → needs broadcasting

weights = topk_experts_weights          # (n_tokens, topk)  = (1, 2)
weights = ttnn.unsqueeze(weights, 0)    # (1, n_tokens, topk, 1)     -- unsqueeze at dim 0
weights = ttnn.unsqueeze(weights, 0)    # (1, 1, n_tokens, topk, 1) -- unsqueeze again
# ... reshape to align with combined_output dims

# Broadcast weights to match combined_output hidden dimension:
weights_broadcast = ttnn.repeat(weights, repeat_dims=(hidden_size, 1, 1, 1))
# shape: (hidden_size, n_tokens, topk, 1) or equivalent

# Permute to align axes:
weights_perm = ttnn.permute(weights_broadcast, (3, 1, 2, 0))
# Reorder to: (1, n_tokens, topk, hidden_size)

weights_tiled = ttnn.to_layout(weights_perm, ttnn.TILE_LAYOUT)

# Elementwise multiply: each expert output scaled by its weight
weighted = ttnn.mul(combined_output, weights_tiled)
# weighted shape: (topk, n_tokens, 1, hidden_size)

# Sum across the topk dimension (dim=0):
output = ttnn.sum(weighted, dim=0)
# output shape: (1, n_tokens, 1, hidden_size) = (1, 1, 1, 4096) at batch=1 decode
```

The key op is `ttnn.repeat(weights, repeat_dims=(hidden_size, 1, 1, 1))`: it takes a single scalar weight value and replicates it `hidden_size=4096` times along the first dimension to create a broadcastable weight tensor matching the hidden dimension of the expert output.

### Tensor sizes involved

At batch=1 decode, topk=2, hidden=4096:

| Tensor | Shape | Elements | Memory (bf16) |
|---|---|---|---|
| `combined_output` | (2, 1, 1, 4096) | 8192 | 16 384 B = 16 KB |
| `topk_experts_weights` | (1, 2) | 2 | 4 B |
| `weights_broadcast` (after repeat) | (4096, 1, 2, 1) or equiv. | 8192 | 16 384 B = 16 KB |
| `weights_perm` (after permute) | (1, 1, 2, 4096) or equiv. | 8192 | 16 384 B = 16 KB |
| `weighted` | (2, 1, 1, 4096) | 8192 | 16 KB |
| `output` | (1, 1, 1, 4096) | 4096 | 8 KB |

All tensors are small. The concern is not memory bandwidth but rather op-dispatch latency: each `ttnn.repeat`, `ttnn.permute`, `ttnn.to_layout`, `ttnn.mul`, and `ttnn.sum` call incurs a device kernel launch overhead independent of tensor size. At batch=1 decode, these small tensors may be entirely kernel-launch-overhead-dominated.

### Estimated overhead breakdown (batch=1 decode)

| Op | Expected latency | Rationale |
|---|---|---|
| `ttnn.unsqueeze` × 2 | 1–3 µs each | Metadata-only reshape, minimal data movement |
| `ttnn.repeat` (4096×) | 5–15 µs | Must allocate and fill a 16 KB output tensor |
| `ttnn.permute` | 3–8 µs | Data layout reorganization of 16 KB |
| `ttnn.to_layout` (→ TILE) | 2–5 µs | ROW_MAJOR → TILE_LAYOUT conversion |
| `ttnn.mul` (elementwise) | 2–5 µs | 8192-element multiply |
| `ttnn.sum` (dim=0) | 3–8 µs | Reduce 2 slices of 4096 elements each |
| **Stage 7 total** | **~20–50 µs** | Summing across all ops |

This is a rough estimate. The actual cost depends on SRAM vs DRAM placement, kernel launch batching, and whether the TTNN scheduler can pipeline adjacent elementwise ops. Measurement is required.

---

## Measuring Stage 7 in Isolation

```python
import ttnn
import torch
import time

hidden_size = 4096
topk = 2
n_tokens = 1  # batch=1 decode

# Build synthetic combined_output: (topk, n_tokens, 1, hidden_size)
combined_out = ttnn.from_torch(
    torch.randn(topk, n_tokens, 1, hidden_size, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Router weights: (n_tokens, topk)
weights_torch = torch.softmax(torch.randn(n_tokens, topk), dim=-1).bfloat16()
topk_weights = ttnn.from_torch(
    weights_torch,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

def apply_weights_current(combined_out, topk_weights):
    """Replicate the moe.py:L1321–1335 weight application."""
    w = ttnn.unsqueeze(topk_weights, 0)
    w = ttnn.unsqueeze(w, 0)
    w_broadcast = ttnn.repeat(w, repeat_dims=(hidden_size, 1, 1, 1))
    w_perm = ttnn.permute(w_broadcast, (3, 1, 2, 0))
    w_tiled = ttnn.to_layout(w_perm, ttnn.TILE_LAYOUT)
    weighted = ttnn.mul(combined_out, w_tiled)
    return ttnn.sum(weighted, dim=0)

# Warmup
for _ in range(20):
    out = apply_weights_current(combined_out, topk_weights)
    ttnn.synchronize_device(device)

# Timed
N = 100
t0 = time.perf_counter()
for _ in range(N):
    out = apply_weights_current(combined_out, topk_weights)
    ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"Stage 7 current impl mean: {(t1 - t0) / N * 1e6:.1f} µs")
```

To decompose the individual ops, insert `ttnn.synchronize_device` between each call and time them separately. This adds synchronization overhead to the individual measurements but isolates the dominant op.

---

## Alternative: Elementwise Multiply After Reshape

The `ttnn.repeat` + `ttnn.permute` sequence creates a full 16 KB weight broadcast tensor solely to enable elementwise multiplication with `combined_output`. A simpler approach avoids the broadcast allocation by reshaping the weight tensor to a form that TTNN can broadcast without materializing the full expanded tensor.

### Code sketch of the alternative

```python
def apply_weights_alternative(combined_out, topk_weights):
    """
    Alternative weight application without ttnn.repeat broadcasting.

    combined_out: (topk, n_tokens, 1, hidden_size) in TILE_LAYOUT
    topk_weights: (n_tokens, topk) scalar weights

    Strategy: reshape weights to (topk, n_tokens, 1, 1) and rely on
    TTNN implicit broadcasting along the hidden_size dimension.
    """
    # Reshape topk_weights from (n_tokens, topk) to (topk, n_tokens, 1, 1)
    # so that it aligns with combined_out's leading dimensions.
    w = ttnn.reshape(topk_weights, (topk, n_tokens, 1, 1))
    # Convert to TILE_LAYOUT for the multiply
    w_tiled = ttnn.to_layout(w, ttnn.TILE_LAYOUT)

    # TTNN broadcast multiply: (topk, n_tokens, 1, 1) broadcasts over
    # (topk, n_tokens, 1, hidden_size) → result shape (topk, n_tokens, 1, hidden_size)
    # This avoids allocating the (4096, 1, 2, 1) intermediate tensor.
    weighted = ttnn.mul(combined_out, w_tiled)  # implicit broadcast on last dim

    # Sum across topk dimension
    return ttnn.sum(weighted, dim=0)
```

**Why this may be faster:**

1. `ttnn.repeat` with `repeat_dims=(4096, 1, 1, 1)` allocates and writes a 16 KB tensor. With the reshape approach, `w` is a `(2, 1, 1, 1)` tensor (4 bytes), and `ttnn.to_layout` operates on a trivially small object.
2. Eliminating the `ttnn.permute` call removes one data-reorganization kernel.
3. TTNN's `ttnn.mul` supports implicit broadcasting when one operand's dimension is 1 and the other's is > 1. If the TTNN `mul` kernel implements this without materializing the broadcast, the total data moved is cut by ~50%.

**Risk:** TTNN broadcasting behavior for `ttnn.mul` must be confirmed. If the TTNN mul kernel does not support implicit broadcast on `hidden_size` dimension and instead requires equal shapes, it will trigger a shape mismatch error. Verify against the TTNN API docs before deploying.

**Alternative without broadcast reliance:**

```python
def apply_weights_loop_free(combined_out, topk_weights):
    """
    Variant that avoids both ttnn.repeat and implicit broadcast,
    instead using einsum-style scaling if available, or explicit
    per-topk slice multiply-accumulate.
    """
    # Slice the topk dimension, scale each slice, accumulate
    topk_val = combined_out.shape[0]  # = 2

    # Extract weight slices: (n_tokens, 1) for each k
    output = None
    for k in range(topk_val):
        # expert_out_k: (1, n_tokens, 1, hidden_size)
        expert_out_k = combined_out[k:k+1, :, :, :]
        # weight_k: (n_tokens,) scalar → reshape to (1, n_tokens, 1, 1)
        w_k = ttnn.reshape(topk_weights[:, k:k+1], (1, n_tokens, 1, 1))
        w_k_tiled = ttnn.to_layout(w_k, ttnn.TILE_LAYOUT)
        scaled_k = ttnn.mul(expert_out_k, w_k_tiled)
        output = scaled_k if output is None else ttnn.add(output, scaled_k)
    return output
```

This loop-based approach avoids both `ttnn.repeat` and `ttnn.sum`, replacing them with 2 multiplies and 1 add (for topk=2). However, Python-level loops with TTNN ops incur host-side dispatch overhead per iteration; for topk=2 this is acceptable, but it would not scale to topk > 4.

### Benchmarking the alternatives

Use the same warmup-loop harness from [`token_padding_and_dispatch.md`](./token_padding_and_dispatch.md) and [`sparse_matmul_profiling.md`](./sparse_matmul_profiling.md), substituting the three candidate weight-application implementations as the timed block.

Record results in this table:

| Implementation | Stage 7 latency (µs) | Notes |
|---|---|---|
| Current (`repeat` + `permute` + `mul` + `sum`) | ___ | moe.py:L1321–1335 |
| Alternative (reshape + broadcast `mul` + `sum`) | ___ | Avoids 16 KB allocation |
| Loop-based (`mul` × topk + `add`) | ___ | topk=2 only |

---

## Interaction with Padding Removal

Stage 7 ends with a padding removal step at `moe.py:L1338–L1341`:

```python
# moe.py:L1338–1341
output = ttnn.slice(output, begins=[0, 0, 0, 0], ends=[1, 1, n_tokens_real, hidden_size])
```

At batch=1 decode, this slices from shape `(1, 1, 32, hidden_size)` to `(1, 1, 1, hidden_size)`. The `ttnn.slice` op on a small DRAM tensor is expected to cost 1–3 µs. It should be timed together with Stage 7 in the isolation benchmark above.

---

## Summary: When Does Stage 7 Matter?

Stage 7 is a post-processing step that operates on the final expert output. Its cost is independent of the number of experts and scales primarily with `hidden_size` and `topk`. For the current model configs (hidden=4096, topk=2), it is a fixed overhead that applies to every forward pass regardless of expert routing.

At batch=1 decode, if the sparse matmuls (Stages 4–5) take 20–50 µs and Stage 7 takes 20–50 µs, Stage 7 represents a non-trivial fraction of total `TTNNExperts.forward` time. The alternative implementation is worth testing: if the reshape + broadcast approach saves 10–20 µs, it is a straightforward code change with no model-level impact.

---

**Next:** [`bottleneck_summary.md`](./bottleneck_summary.md)
