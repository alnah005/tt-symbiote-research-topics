# Global Attention Decode Forward Pass

This file provides the complete step-by-step decode forward pass for global
(full-causal) attention layers in Gemma 4 31B. There are 10 such layers at
indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59. Each global layer uses 32
query heads and 4 KV heads with `head_dim=512`, K=V weight sharing,
proportional RoPE (theta=1M, partial_rotary_factor=0.25), and full causal
attention (no window constraint).

All tensor shapes are shown for batch=1 single-token decode (`B=1, S=1`).

## Forward Pass Overview

```text
hidden_states [1, 1, 5376]
      |
      +---> q_proj [5376, 16384] --> Q [1, 1, 16384]
      |                                    |
      |                               reshape [1, 32, 1, 512]
      |                                    |
      |                               q_norm (scaled RMSNorm, gamma in R^512)
      |                                    |
      |                               partial RoPE (128/512 dims, theta=1M)
      |                                    |
      +---> k_proj [5376, 2048] ---> shared_kv [1, 1, 2048]
                                           |
                                      reshape [1, 4, 1, 512]
                                           |
                        +------------------+------------------+
                        |                                     |
                   K path                                V path
                        |                                     |
                   k_norm (scaled RMSNorm,              v_norm (unscaled RMSNorm,
                    gamma in R^512)                      no gamma)
                        |                                     |
                   partial RoPE                          (no RoPE)
                   (128/512 dims, theta=1M)                   |
                        |                                     |
                   --> KV cache K slot                   --> KV cache V slot
                        |                                     |
                        +------------------+------------------+
                                           |
                                    paged_sdpa_decode
                                    (full causal, no window)
                                           |
                                      attn_output [1, 32, 1, 512]
                                           |
                                      reshape [1, 1, 16384]
                                           |
                                      o_proj [16384, 5376]
                                           |
                                      output [1, 1, 5376]
```

## Step 1 --- Q Projection and Shared K/V Projection

### Q Projection

```python
query_states = ttnn.linear(hidden_states, q_proj_weight)
# [1, 1, 5376] x [5376, 16384] -> [1, 1, 16384]
```

This is the largest attention projection in the model. At BF16, the global Q
weight is 176 MB --- roughly double the sliding Q weight.

### Shared K/V Projection (K=V Sharing)

Global layers have `attention_k_eq_v=True`, meaning a single K projection
weight serves both keys and values. There is no V projection weight.

```python
shared_kv = ttnn.linear(hidden_states, k_proj_weight)
# [1, 1, 5376] x [5376, 2048] -> [1, 1, 2048]
```

This is the smallest attention projection (2048 output dim), reflecting the
high GQA ratio (32Q : 4KV = 8:1) combined with K=V sharing.

### Fused QK Alternative

The Q and K projections can be fused into a single matmul. Note that this is
Q+K fusion, **not** Q+K+V --- the V projection does not exist.

```python
# Fused weight: [W_Q | W_K] = [5376, 18432]
# where 18432 = 16384 + 2048
qk = ttnn.linear(hidden_states, fused_qk_weight)
# [1, 1, 5376] x [5376, 18432] -> [1, 1, 18432]

# Slice to recover Q and shared K/V
query_states = qk[:, :, :16384]       # [1, 1, 16384]
shared_kv = qk[:, :, 16384:]          # [1, 1, 2048]
```

The fused QK matmul is larger (18432 output dim) than the sliding fused QKV
(16384 output dim), achieving good compute utilization on Wormhole.

### Reshape to Per-Head Format

```python
query_states = ttnn.reshape(query_states, [1, 32, 1, 512])   # [B, H_q, S, D]
shared_kv = ttnn.reshape(shared_kv, [1, 4, 1, 512])          # [B, H_kv, S, D]
```

## Step 2 --- Clone/Split Into K and V Tensors

After reshape, the `shared_kv` tensor must be consumed by two divergent paths:
the K path (K-norm + partial RoPE) and the V path (V-norm only). The critical
requirement is that the K path operations must not corrupt the input to the V
path.

### Functional Approach (Recommended)

If the TTNN K-norm and V-norm operations both produce **new output tensors**
(i.e., they are not in-place), no explicit clone is needed:

```python
# Both norms consume shared_kv as a read-only input
key_states = self.k_norm(shared_kv)      # new tensor [1, 4, 1, 512]
value_states = self.v_norm(shared_kv)    # new tensor [1, 4, 1, 512]
```

This is the recommended approach because it avoids the memory and latency
cost of an explicit `ttnn.clone`. See
[Chapter 3 --- K=V Sharing](../ch3_kv_sharing_and_vnorm/k_eq_v_mechanism.md)
for the full analysis.

### Explicit Clone (Fallback)

If any downstream operation on the K path writes in-place to its input buffer:

```python
key_states = shared_kv                         # alias
value_states = ttnn.clone(shared_kv)           # physical copy
key_states = self.k_norm(key_states)           # may modify in-place
value_states = self.v_norm(value_states)       # safe: separate buffer
```

## Step 3 --- K Path: Scaled RMSNorm + Partial p-RoPE

### K-Norm (Scaled)

```python
key_states = ttnn_distributed_rms_norm(key_states, k_norm_weight, eps=1e-6)
# key_states: [1, 4, 1, 512]
# k_norm_weight: gamma in R^512, learned per-element scale
```

This is a standard scaled RMSNorm with a learned $\gamma$ vector. Unlike
V-norm, K-norm has a weight tensor loaded from the checkpoint
(`layers.{i}.self_attn.k_norm.weight`).

### Partial RoPE (128/512 Dimensions)

Global layers use proportional RoPE with $\theta = 1{,}000{,}000$ and
`partial_rotary_factor=0.25`. Only the first 128 of 512 dimensions are
rotated; the remaining 384 pass through unchanged.

```python
# cos_slice: [1, 128], sin_slice: [1, 128]  (narrow tables, theta=1M)
# Using Strategy B (split-apply-concat) from Chapter 4:

rotary_dim = 128

# Split Q into rotary and pass-through
q_rot = query_states[:, :, :, :rotary_dim]      # [1, 32, 1, 128]
q_pass = query_states[:, :, :, rotary_dim:]      # [1, 32, 1, 384]

# Split K into rotary and pass-through
k_rot = key_states[:, :, :, :rotary_dim]         # [1, 4, 1, 128]
k_pass = key_states[:, :, :, rotary_dim:]         # [1, 4, 1, 384]

# Apply RoPE only to rotary dimensions
q_rot, k_rot = ttnn_rope(q_rot, k_rot, cos_slice, sin_slice)

# Concatenate back
query_states = ttnn.concat([q_rot, q_pass], dim=-1)    # [1, 32, 1, 512]
key_states = ttnn.concat([k_rot, k_pass], dim=-1)      # [1, 4, 1, 512]
```

Alternatively, `TTNNRotaryPositionEmbedding` handles the split-apply-concat
internally when `cos.shape[-1] < q.shape[-1]`, so the caller can simply pass
the narrow cos/sin tables:

```python
query_states, key_states = self.rope(query_states, key_states, cos_slice, sin_slice)
# TTNNRotaryPositionEmbedding detects partial rotation and handles the split internally
```

See [Chapter 4 --- Global p-RoPE](../ch4_dual_rope/global_proportional_rope.md)
for the mathematical formulation and the full-width tables alternative
(Strategy A).

### Why Non-Distributed RoPE

Global layers must use `TTNNRotaryPositionEmbedding` (the non-distributed
variant) because `TTNNDistributedRotaryPositionEmbedding` does not currently
support `partial_rotary_factor < 1.0`. This is a software limitation, not a
fundamental constraint. Each device applies RoPE independently to its local
Q/K head slices using the same cos/sin tables --- the practical performance
impact is minimal. See
[Chapter 4](../ch4_dual_rope/global_proportional_rope.md) for the
compatibility analysis.

## Step 4 --- V Path: Unscaled RMSNorm (No RoPE)

```python
value_states = ttnn_distributed_rms_norm(value_states, v_norm_ones_weight, eps=1e-6)
# value_states: [1, 4, 1, 512]
# v_norm_ones_weight: all-ones dummy weight (no learned scale)
```

The V path receives **no RoPE**. Value vectors carry semantic content that
should remain position-invariant. The attention mechanism's position
sensitivity comes entirely from the Q and K vectors.

Note that V-norm was applied to `shared_kv` (the raw K projection output
before K-norm or RoPE), not to the K-norm output. This is correct because
V-norm and K-norm are independent operations that both consume the same shared
input.

## Step 5 --- KV Cache Update (Paged, Full Causal)

```python
kv_cache.paged_update_on_device(
    key_states,      # [1, 4, 1, 512]
    value_states,    # [1, 4, 1, 512]
    layer_idx=self.layer_idx,
    current_pos=current_pos
)
```

### Paged Cache Geometry

The paged KV cache block pool for a global layer has shape:

```text
K pool: [max_num_blocks, 4, block_size, 512]
V pool: [max_num_blocks, 4, block_size, 512]
```

Global layers use full causal attention with no window constraint. The page
table grows linearly with sequence length up to `max_seq_len`. At 256K
context with `block_size=64`, a global layer's KV cache requires up to
$\lceil 262144 / 64 \rceil = 4096$ pages per sequence.

### Memory Per Global Layer

At `block_size=64`, `num_kv_heads=4`, `head_dim=512`, BF16:

```math
\text{per block} = 2 \times 4 \times 64 \times 512 \times 2 = 524{,}288 \text{ bytes} = 512 \text{ KB}
```

```math
\text{per layer at 256K} = 4096 \times 512 \text{ KB} = 2 \text{ GB}
```

This is a significant per-device cost. Under TP=8, the sharding strategy for
global KV heads determines how this memory is distributed. See
[Chapter 6](../ch6_tp_sharding/index.md) for the analysis.

## Step 6 --- Paged SDPA Decode With Full Causal

```python
attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
    input_tensor_q=query_states,        # [1, 1, 32, 512] (after layout transform)
    input_tensor_k=kv_cache.k_cache,    # [max_num_blocks, 4, block_size, 512]
    input_tensor_v=kv_cache.v_cache,    # [max_num_blocks, 4, block_size, 512]
    cur_pos_tensor=current_pos,         # [B], int32
    scale=1.0 / (512 ** 0.5),          # = 1/sqrt(512) ~= 0.0442
    sliding_window_size=None,           # full causal --- no window restriction
    page_table_tensor=page_table,       # [B, max_pages_per_seq], int32
    program_config=global_sdpa_config,
    compute_kernel_config=compute_config,
)
# attn_output: [1, 1, 32, 512] (padded to [1, 1, pnh, 512])
```

### GQA Handling

With 32 query heads and 4 KV heads, the GQA group size is 8 (each KV head
serves 8 query heads). This is a more aggressive grouping than sliding layers
(group size 2). The `paged_sdpa_decode` kernel handles this natively by
broadcasting each KV head across its 8 query heads.

### Attention Scale

The attention scale for global layers is $1/\sqrt{512} \approx 0.0442$, which
is smaller than the sliding layer scale of $1/\sqrt{256} = 0.0625$. This
lower scale softens the attention distribution, which is appropriate for
the larger head dimension.

### Full Causal vs Windowed

With `sliding_window_size=None`, the kernel computes attention over all KV
positions from 0 to T (the current token position). This means the kernel
must load all allocated pages --- up to 4096 pages at 256K context. The DRAM
bandwidth cost scales linearly with sequence length.

### Program Config for Global SDPA

The `SDPADecodeProgramConfig` for global layers differs from sliding layers:

- **`k_chunk_size`**: Must be tuned for potentially long KV sequences. At
  256K context and `k_chunk_size=512`, the kernel iterates $262144/512 = 512$
  times per head. A larger `k_chunk_size` (e.g., 1024 or 2048) reduces
  iterations but increases L1 pressure.
- **L1 working set**: At `k_chunk_size=512`, `head_dim=512`, BF16:
  K tile = 512 x 512 x 2 = 512 KB, V tile = 512 KB.
  Total ~1 MB per core --- this is approaching the 1.5 MB L1 limit. A
  `k_chunk_size` of 256 (K tile = 256 KB, V tile = 256 KB, total ~512 KB)
  may be necessary.
- **Core grid**: With 4 KV heads and GQA group size 8, fewer cores may be
  needed for the KV head dimension, but the 8 query heads per group still
  benefit from parallelism.

## Step 7 --- Output Projection

```python
# Reshape from [1, 1, 32, 512] to [1, 1, 16384]
attn_output = ttnn.reshape(attn_output, [1, 1, 16384])

# O projection (row-parallel under TP=8)
output = ttnn.linear(attn_output, o_proj_weight)
# [1, 1, 16384] x [16384, 5376] -> [1, 1, 5376]

# All-reduce after row-parallel matmul (under TP)
output = ttnn.all_reduce(output)
```

The output shape `[1, 1, 5376]` is identical to the sliding layer output.
This uniform output dimension is what allows the decoder layer's residual
connection to work identically for both layer types.

## Complete Tensor Shape Trace

| Step | Operation | Input Shape(s) | Output Shape | Notes |
|------|-----------|----------------|--------------|-------|
| 1a | Q projection | [1, 1, 5376] | [1, 1, 16384] | `ttnn.linear` |
| 1b | K/V projection (shared) | [1, 1, 5376] | [1, 1, 2048] | Single `ttnn.linear` |
| 1c | Q reshape | [1, 1, 16384] | [1, 32, 1, 512] | `ttnn.reshape` |
| 1d | shared_kv reshape | [1, 1, 2048] | [1, 4, 1, 512] | `ttnn.reshape` |
| 2a | Q-norm | [1, 32, 1, 512] | [1, 32, 1, 512] | Scaled RMSNorm |
| 2b | K-norm (from shared_kv) | [1, 4, 1, 512] | [1, 4, 1, 512] | Scaled RMSNorm, new tensor |
| 2c | V-norm (from shared_kv) | [1, 4, 1, 512] | [1, 4, 1, 512] | Unscaled RMSNorm, new tensor |
| 3a | Q partial RoPE | [1, 32, 1, 512] | [1, 32, 1, 512] | 128/512 dims rotated |
| 3b | K partial RoPE | [1, 4, 1, 512] | [1, 4, 1, 512] | 128/512 dims rotated |
| 4 | KV cache update | K [1,4,1,512], V [1,4,1,512] | (in-place) | Paged write |
| 5 | paged_sdpa_decode | Q [1,1,32,512], K/V cache, page_table | [1,1,32,512] | Full causal |
| 6a | Reshape | [1, 1, 32, 512] | [1, 1, 16384] | Flatten heads |
| 6b | O projection | [1, 1, 16384] | [1, 1, 5376] | `ttnn.linear` |

## Fused QKV Optimization When V Shares K

The standard fused QKV optimization concatenates Q, K, and V weights into a
single matrix. With K=V sharing, V has no weight, so the fused weight packs
only Q and K:

```text
Sliding fused: [W_Q | W_K | W_V] = [5376, 16384]   (3-way split after matmul)
Global fused:  [W_Q | W_K]       = [5376, 18432]   (2-way split after matmul)
```

After the fused QK matmul, the output is sliced into Q and shared_KV:

```python
fused_output = ttnn.linear(hidden_states, fused_qk_weight)  # [1, 1, 18432]
query_states = fused_output[:, :, :16384]                     # Q: [1, 1, 16384]
shared_kv = fused_output[:, :, 16384:]                        # K/V: [1, 1, 2048]
```

The shared_kv tensor then enters the K=V divergent path (K-norm + RoPE for K,
V-norm for V) as described in Steps 2--4.

### Weight Construction for Fused QK

When loading from a HuggingFace checkpoint, the fused weight is constructed by
concatenating the Q and K projection weights:

```python
# For a global layer at index i:
q_weight = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]   # [16384, 5376]
k_weight = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]   # [2048, 5376]

# Concatenate along output dim (dim=0 in [out, in] convention)
fused_qk_weight = torch.cat([q_weight, k_weight], dim=0)            # [18432, 5376]

# Transpose to [in, out] for TTNN: [5376, 18432]
fused_qk_weight = fused_qk_weight.T
```

There is **no** `v_proj.weight` in the state dict for global layers. The
weight loader must handle this absence gracefully.

## TTNN Pseudocode (Global Subclass)

```python
class TTNNGemma4GlobalAttention(TTNNGemma4AttentionBase):

    def _project_kv_and_rope(self, hidden_states, query_states, cos, sin):
        # Shared K/V projection (K=V sharing: single weight, no V proj)
        shared_kv = ttnn.linear(hidden_states, self.k_proj_weight)
        shared_kv = ttnn.reshape(shared_kv, [1, 4, 1, 512])

        # Divergent paths from shared input (functional, no clone needed)
        # K path: scaled RMSNorm + partial RoPE
        key_states = self.k_norm(shared_kv)      # new tensor
        # V path: unscaled RMSNorm, no RoPE
        value_states = self.v_norm(shared_kv)    # new tensor

        # Partial RoPE on Q and K (128/512 dims, theta=1M)
        # TTNNRotaryPositionEmbedding handles split-apply-concat internally
        query_states, key_states = self.rope(query_states, key_states, cos, sin)

        return query_states, key_states, value_states

    def _sdpa(self, query_states, kv_cache, current_pos, page_table):
        # Transpose Q to paged_sdpa_decode expected layout: [1, B, nh, dh]
        q_for_sdpa = ttnn.reshape(query_states, [1, 1, 32, 512])

        return ttnn.transformer.scaled_dot_product_attention_decode(
            input_tensor_q=q_for_sdpa,
            input_tensor_k=kv_cache.get_k_cache(self.layer_idx),
            input_tensor_v=kv_cache.get_v_cache(self.layer_idx),
            cur_pos_tensor=current_pos,
            scale=self.scale,
            sliding_window_size=None,      # full causal
            page_table_tensor=page_table,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
```

## Key Implementation Considerations

1. **Tensor aliasing in K=V sharing.** The K-norm and V-norm operations must
   both consume `shared_kv` as a read-only input. Verify that the TTNN
   RMSNorm kernel allocates a new output buffer rather than writing in-place
   to the input. If in-place behavior is detected, switch to explicit
   `ttnn.clone` before the K path.

2. **Two different KV cache instances.** Global layers and sliding layers
   have different KV cache geometries (4 heads x 512 dim vs 16 heads x
   256 dim, full causal vs windowed). The model must maintain separate
   `TTNNPagedAttentionKVCache` configurations or use a unified cache with
   per-layer configuration.

3. **Program config L1 pressure.** The `head_dim=512` in global layers
   doubles the per-tile memory requirement compared to sliding layers
   (`head_dim=256`). The `k_chunk_size` in the SDPA program config may need
   to be halved to stay within L1 budget.

4. **Attention scale difference.** Global layers use scale $1/\sqrt{512}$
   while sliding layers use $1/\sqrt{256}$. This is derived from `head_dim`
   and must be set per-layer-type, not globally.

---

**Next:** [`paged_sdpa_sliding_window.md`](./paged_sdpa_sliding_window.md)
