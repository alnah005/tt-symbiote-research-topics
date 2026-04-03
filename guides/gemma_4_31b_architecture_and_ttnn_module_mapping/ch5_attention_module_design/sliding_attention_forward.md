# Sliding Attention Decode Forward Pass

This file provides the complete step-by-step decode forward pass for
sliding-window attention layers in Gemma 4 31B. There are 50 such layers
(all layer indices except 5, 11, 17, 23, 29, 35, 41, 47, 53, 59). Each
sliding layer uses 32 query heads and 16 KV heads with `head_dim=256`,
standard RoPE (theta=10000, full rotation), and a 1024-token window
constraint.

All tensor shapes are shown for batch=1 single-token decode (`B=1, S=1`).

## Forward Pass Overview

```text
hidden_states [1, 1, 5376]
      |
      +---> q_proj [5376, 8192] --> Q [1, 1, 8192]
      |                                   |
      |                              reshape [1, 32, 1, 256]
      |                                   |
      |                              q_norm (scaled RMSNorm, gamma in R^256)
      |                                   |
      |                              RoPE (full, theta=10K, 256 dims)
      |                                   |
      +---> k_proj [5376, 4096] --> K [1, 1, 4096]
      |                                   |
      |                              reshape [1, 16, 1, 256]
      |                                   |
      |                              k_norm (scaled RMSNorm, gamma in R^256)
      |                                   |
      |                              RoPE (full, theta=10K, 256 dims)
      |                                   |
      |                              --> KV cache K slot
      |
      +---> v_proj [5376, 4096] --> V [1, 1, 4096]
                                          |
                                     reshape [1, 16, 1, 256]
                                          |
                                     v_norm (unscaled RMSNorm, no gamma)
                                          |
                                     --> KV cache V slot
                                          |
                                  paged_sdpa_decode
                                  (sliding_window_size=1024)
                                          |
                                     attn_output [1, 32, 1, 256]
                                          |
                                     reshape [1, 1, 8192]
                                          |
                                     o_proj [8192, 5376]
                                          |
                                     output [1, 1, 5376]
```

## Step 1 --- Q, K, V Projections

### Separate Projections

The sliding layer has three independent projection weight matrices:

```python
# Q projection
query_states = ttnn.linear(hidden_states, q_proj_weight)   # [1, 1, 5376] x [5376, 8192] -> [1, 1, 8192]

# K projection
key_states = ttnn.linear(hidden_states, k_proj_weight)     # [1, 1, 5376] x [5376, 4096] -> [1, 1, 4096]

# V projection
value_states = ttnn.linear(hidden_states, v_proj_weight)   # [1, 1, 5376] x [5376, 4096] -> [1, 1, 4096]
```

### Fused QKV Alternative

For better hardware utilization, the three projections can be fused into a
single matmul by concatenating the weight matrices:

```python
# Fused weight: [W_Q | W_K | W_V] = [5376, 16384]
# where 16384 = 8192 + 4096 + 4096
qkv = ttnn.linear(hidden_states, fused_qkv_weight)        # [1, 1, 5376] x [5376, 16384] -> [1, 1, 16384]

# Slice to recover individual projections
query_states = qkv[:, :, :8192]                            # [1, 1, 8192]
key_states = qkv[:, :, 8192:12288]                         # [1, 1, 4096]
value_states = qkv[:, :, 12288:]                           # [1, 1, 4096]
```

The fused approach replaces three small matmuls (each partially utilizing the
compute grid) with one large matmul that achieves better Wormhole core
utilization. This is the recommended approach for production.

### Reshape to Per-Head Format

```python
query_states = ttnn.reshape(query_states, [1, 32, 1, 256])    # [B, H_q, S, D]
key_states = ttnn.reshape(key_states, [1, 16, 1, 256])        # [B, H_kv, S, D]
value_states = ttnn.reshape(value_states, [1, 16, 1, 256])    # [B, H_kv, S, D]
```

Alternatively, if using `ttnn.experimental.nlp_create_qkv_heads` on the fused
QKV output, the split and reshape happen in a single operation.

## Step 2 --- Q-Norm, K-Norm, and V-Norm

All three norms are RMSNorm applied per-head over the `head_dim=256` axis.
Q-norm and K-norm use learned scale parameters ($\gamma \in \mathbb{R}^{256}$).
V-norm uses no learned scale (see
[Chapter 3](../ch3_kv_sharing_and_vnorm/vnorm_implementation.md)).

```python
# Q-norm: scaled RMSNorm
query_states = ttnn_distributed_rms_norm(query_states, q_norm_weight, eps=1e-6)
# shape unchanged: [1, 32, 1, 256]

# K-norm: scaled RMSNorm
key_states = ttnn_distributed_rms_norm(key_states, k_norm_weight, eps=1e-6)
# shape unchanged: [1, 16, 1, 256]

# V-norm: unscaled RMSNorm (all-ones dummy weight)
value_states = ttnn_distributed_rms_norm(value_states, v_norm_ones_weight, eps=1e-6)
# shape unchanged: [1, 16, 1, 256]
```

### Ordering Note

In the HuggingFace reference, the norm-then-RoPE order is:
1. Project Q, K, V
2. Reshape to per-head format
3. Apply Q-norm and K-norm (V-norm also applied here)
4. Transpose to `[B, H, S, D]` (already in this layout after reshape)
5. Apply RoPE to Q and K

The TTNN implementation follows the same order. V-norm is applied before
the KV cache write, ensuring normalized values are stored in the cache.

## Step 3 --- Standard RoPE on Q and K

Sliding layers use full-rotation RoPE with $\theta = 10{,}000$. All 256
dimensions of each head participate in the rotation.

```python
# cos, sin tables: precomputed [max_seq_len, 256], sliced to current position
# cos_slice: [1, 256], sin_slice: [1, 256]

query_states, key_states = ttnn_distributed_rope(
    query_states,   # [1, 32, 1, 256]
    key_states,     # [1, 16, 1, 256]
    cos_slice,      # [1, 256]
    sin_slice       # [1, 256]
)
# shapes unchanged: Q [1, 32, 1, 256], K [1, 16, 1, 256]
```

Because all 256 dimensions are rotated (`partial_rotary_factor=1.0`), the
standard `TTNNDistributedRotaryPositionEmbedding` applies without modification.
No split-apply-concat is needed.

See [Chapter 4 --- Sliding RoPE](../ch4_dual_rope/sliding_rope.md) for the
full mathematical formulation and cos/sin table precomputation.

## Step 4 --- KV Cache Update (Paged, Window-Bounded)

The normalized and RoPE-encoded K and V tensors are written to the paged KV
cache. The sliding layer's KV cache is bounded to 1024 tokens by the
`sliding_window_size` parameter of the SDPA call (see Step 5), but the cache
update itself writes to the page table unconditionally.

```python
# current_pos: ttnn.Tensor [B], int32, on device --- the absolute token position
kv_cache.paged_update_on_device(
    key_states,      # [1, 16, 1, 256]
    value_states,    # [1, 16, 1, 256]
    layer_idx=self.layer_idx,
    current_pos=current_pos
)
```

### Paged Cache Geometry

The paged KV cache block pool for a sliding layer has shape:

```text
K pool: [max_num_blocks, 16, block_size, 256]
V pool: [max_num_blocks, 16, block_size, 256]
```

With `block_size=64` and `window=1024`, the minimum number of blocks per
sequence is $\lceil 1024 / 64 \rceil = 16$ blocks. However, because the
paged cache write uses absolute position addressing (the page allocator
assigns pages linearly as the sequence grows), the page table may contain
more than 16 entries for sequences longer than 1024 tokens. The window
constraint is enforced at SDPA time, not at cache-write time.

### Window Enforcement at Cache Level

Two strategies exist for managing the relationship between the paged cache
and the 1024-token window:

**Strategy 1 --- Let SDPA handle windowing.** Write all K/V tokens to the
paged cache without restriction. At SDPA time, pass
`sliding_window_size=1024` to `paged_sdpa_decode`, which instructs the kernel
to attend only to the most recent 1024 positions. Older pages remain
allocated but are not read by the kernel.

**Strategy 2 --- Circular-buffer-as-pages.** Allocate exactly
$\lceil 1024 / \text{block\_size} \rceil$ pages per sequence and write in
circular fashion, as described in the
[windowed attention guide](../../windowed_attention_foundations_and_t3k_mapping/ch5_paged_kv_cache/paged_sdpa_and_windowing.md).
This bounds memory usage to exactly 1024 tokens per layer.

**Recommendation for Gemma 4 31B:** Strategy 1 is preferred for initial
bringup because `paged_sdpa_decode` natively supports `sliding_window_size`
(see [`paged_sdpa_sliding_window.md`](./paged_sdpa_sliding_window.md)). This
avoids the complexity of circular page table management. Strategy 2 should
be considered for production serving where memory efficiency is critical
(50 sliding layers x unused pages adds up).

## Step 5 --- Paged SDPA Decode With Sliding Window

```python
attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
    input_tensor_q=query_states,       # [1, 1, 32, 256] (after layout transform)
    input_tensor_k=kv_cache.k_cache,   # [max_num_blocks, 16, block_size, 256]
    input_tensor_v=kv_cache.v_cache,   # [max_num_blocks, 16, block_size, 256]
    cur_pos_tensor=current_pos,        # [B], int32
    scale=1.0 / (256 ** 0.5),         # = 1/16
    sliding_window_size=1024,          # restrict attention to last 1024 tokens
    page_table_tensor=page_table,      # [B, max_pages_per_seq], int32
    program_config=sliding_sdpa_config,
    compute_kernel_config=compute_config,
)
# attn_output: [1, 1, 32, 256] (padded to [1, 1, pnh, 256] if nh not multiple of 32)
```

### Q Tensor Layout for `paged_sdpa_decode`

The `paged_sdpa_decode` kernel expects Q in `[1, B, nh, dh]` layout (with
the leading 1 encoding the single decode step). The per-head Q tensor
`[B, H_q, S, D] = [1, 32, 1, 256]` must be transposed to
`[1, 1, 32, 256]` before the call.

### GQA Handling

With 32 query heads and 16 KV heads, the GQA group size is 2 (each KV head
serves 2 query heads). The `paged_sdpa_decode` kernel handles this natively
by broadcasting each KV head across its query head group during the
dot-product and value-aggregation phases. No explicit KV head repetition is
needed.

### `sliding_window_size` Behavior

When `sliding_window_size=1024` is provided, the kernel restricts each query
to attend only to KV positions in the range
$[\max(0, T - 1024 + 1), T]$ where $T$ is the current token position
(from `cur_pos_tensor`). Positions outside this range are masked with
$-\infty$ before softmax, effectively zeroing their attention weights.

This interacts with the paged KV cache as follows: the kernel still reads the
page table to locate the relevant pages, but only loads pages that overlap
with the window range. Pages containing exclusively out-of-window tokens are
skipped entirely, saving DRAM bandwidth.

See [`paged_sdpa_sliding_window.md`](./paged_sdpa_sliding_window.md) for a
detailed investigation of this parameter's behavior.

### Program Config for Sliding SDPA

The `SDPADecodeProgramConfig` for sliding layers should be tuned for:

- **`k_chunk_size`**: With `window=1024` and `block_size=64`, the kernel
  iterates over $\lceil 1024 / \text{k\_chunk\_size} \rceil$ tiles.
  Setting `k_chunk_size=256` gives 4 iterations; `k_chunk_size=512` gives 2.
- **`compute_with_storage_grid_size`**: The 8x8 Wormhole core grid.
  With `B=1` and `H_q=32`, each core can handle one query head.
- **L1 working set**: At `k_chunk_size=256`, `head_dim=256`, BF16:
  K tile = 256 x 256 x 2 = 128 KB, V tile = 128 KB, Q = 512 bytes.
  Total ~256 KB per core --- well within the 1.5 MB L1 budget.

## Step 6 --- Output Projection

The SDPA output is reshaped from per-head format back to a flat vector and
projected through `W_O`:

```python
# Reshape from [1, 1, 32, 256] to [1, 1, 8192]
attn_output = ttnn.reshape(attn_output, [1, 1, 8192])

# O projection (row-parallel under TP=8)
output = ttnn.linear(attn_output, o_proj_weight)   # [1, 1, 8192] x [8192, 5376] -> [1, 1, 5376]

# All-reduce after row-parallel matmul (under TP)
output = ttnn.all_reduce(output)
```

The output shape `[1, 1, 5376]` matches `hidden_size` and is ready for the
residual connection in the decoder layer.

## Complete Tensor Shape Trace

| Step | Operation | Input Shape(s) | Output Shape | Notes |
|------|-----------|----------------|--------------|-------|
| 1a | Q projection | [1, 1, 5376] | [1, 1, 8192] | `ttnn.linear` |
| 1b | K projection | [1, 1, 5376] | [1, 1, 4096] | `ttnn.linear` |
| 1c | V projection | [1, 1, 5376] | [1, 1, 4096] | `ttnn.linear` |
| 1d | Q reshape | [1, 1, 8192] | [1, 32, 1, 256] | `ttnn.reshape` |
| 1e | K reshape | [1, 1, 4096] | [1, 16, 1, 256] | `ttnn.reshape` |
| 1f | V reshape | [1, 1, 4096] | [1, 16, 1, 256] | `ttnn.reshape` |
| 2a | Q-norm | [1, 32, 1, 256] | [1, 32, 1, 256] | Scaled RMSNorm |
| 2b | K-norm | [1, 16, 1, 256] | [1, 16, 1, 256] | Scaled RMSNorm |
| 2c | V-norm | [1, 16, 1, 256] | [1, 16, 1, 256] | Unscaled RMSNorm |
| 3 | RoPE (Q, K) | Q + K + cos/sin | Q + K (same shapes) | Full rotation, 256 dims |
| 4 | KV cache update | K [1,16,1,256], V [1,16,1,256] | (in-place) | Paged write |
| 5 | paged_sdpa_decode | Q [1,1,32,256], K/V cache, page_table | [1,1,32,256] | window=1024 |
| 6a | Reshape | [1, 1, 32, 256] | [1, 1, 8192] | Flatten heads |
| 6b | O projection | [1, 1, 8192] | [1, 1, 5376] | `ttnn.linear` |

## TTNN Pseudocode (Sliding Subclass)

```python
class TTNNGemma4SlidingAttention(TTNNGemma4AttentionBase):

    def _project_kv_and_rope(self, hidden_states, query_states, cos, sin):
        # KV projections (separate K and V)
        key_states = ttnn.linear(hidden_states, self.k_proj_weight)
        value_states = ttnn.linear(hidden_states, self.v_proj_weight)

        # Reshape to per-head format
        key_states = ttnn.reshape(key_states, [1, 16, 1, 256])
        value_states = ttnn.reshape(value_states, [1, 16, 1, 256])

        # K-norm (scaled) and V-norm (unscaled)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        # Standard RoPE on Q and K (full rotation, theta=10K)
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def _sdpa(self, query_states, kv_cache, current_pos, page_table):
        # Transpose Q to paged_sdpa_decode expected layout: [1, B, nh, dh]
        q_for_sdpa = ttnn.reshape(query_states, [1, 1, 32, 256])

        return ttnn.transformer.scaled_dot_product_attention_decode(
            input_tensor_q=q_for_sdpa,
            input_tensor_k=kv_cache.get_k_cache(self.layer_idx),
            input_tensor_v=kv_cache.get_v_cache(self.layer_idx),
            cur_pos_tensor=current_pos,
            scale=self.scale,
            sliding_window_size=1024,
            page_table_tensor=page_table,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
```

---

**Next:** [`global_attention_forward.md`](./global_attention_forward.md)
