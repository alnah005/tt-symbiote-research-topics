# Fused Attention

## Current Implementation

Attention in TT-Symbiote is composed from several modules across `modules/attention.py` and `modules/rope.py`. The two primary attention classes are:

- **`LlamaAttention`** (attention.py, line 885): used for LLaMA, Gemma, and similar decoder-only LLMs
- **`TTNNSelfAttention`** (attention.py, line 553): used for ViT-family encoder models

### LlamaAttention Forward Pass

The `LlamaAttention.forward()` method (line 949) executes the following pipeline:

```
Step 1: QKV Projection
    query, key, value = self.qkv_proj(hidden_states)
    # Uses TTNNFusedQKVSelfAttention: single matmul with fused Q/K/V weights
    # then ttnn.experimental.nlp_create_qkv_heads to split into heads

Step 2: RoPE Application
    query, key = self.rope(query, key, cos, sin)
    # TTNNRotaryPositionEmbedding.forward() -- see rope.py line 67

Step 3: KV Cache Update
    key, value = past_key_values.update(key, value, layer_idx, cache_kwargs)

Step 4: SDPA
    attn_out = self.sdpa(self, query, key, value, None, scaling=..., is_causal=True)
    # TTNNSDPAAttention.forward() -> ttnn.transformer.scaled_dot_product_attention

Step 5: Head Concatenation
    attn_out = ttnn.experimental.nlp_concat_heads(attn_out)

Step 6: Output Projection
    return self.o_proj(attn_out)
```

### TTNNRotaryPositionEmbedding Details

The RoPE module (rope.py, line 64) is a complex operation with significant overhead:

```python
# For partial rotary (rotary_dim < head_dim):
q_rot = q[:, :, :, :rotary_dim]          # Slice
q_pass = q[:, :, :, rotary_dim:]          # Slice
# Pad to tile boundaries
q_rot = ttnn.pad(q_rot, ...)              # Pad
cos = ttnn.pad(cos, ...)                  # Pad
sin = ttnn.pad(sin, ...)                  # Pad
# Apply rotation
q_rot_embedded = ttnn.experimental.rotary_embedding(q_rot, cos, sin)  # Compute
# Slice back and concatenate
q_rot_embedded = q_rot_embedded[:, :, :, :rotary_dim]  # Slice
q_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)  # Concat
```

For full rotary (the common case in LLaMA), this simplifies to a single `ttnn.experimental.rotary_embedding` call, but still requires separate handling for Q and K.

### TTNNSDPAAttention Details

The SDPA module (attention.py, line 313) wraps `ttnn.transformer.scaled_dot_product_attention` but includes fallback logic:

```python
# Primary path:
attn_output = ttnn.transformer.scaled_dot_product_attention(
    query, key, value, is_causal=is_causal, scale=scaling, ...)

# Fallback matmul path (when SDPA fails):
key_t = ttnn.permute(key, (0, 1, 3, 2))
scores = ttnn.matmul(query, key_t)
scores = ttnn.multiply(scores, scale)
scores = ttnn.softmax(scores, dim=-1)
attn_output = ttnn.matmul(scores, value)
```

## Performance Bottleneck Analysis

### DRAM Round-Trips in the Attention Pipeline

For a LLaMA-style model with `hidden_size=4096`, `num_heads=32`, `head_dim=128`, `seq_len=1024`:

| Step | Op | Tensor Size (BF16) | DRAM Write | DRAM Read (next) |
|------|----|--------------------|-----------|-----------------|
| 1a | QKV matmul | 3 x [1, 32, 1024, 128] = 24 MB | Yes | Yes (step 1b) |
| 1b | `nlp_create_qkv_heads` | Q + K + V = 24 MB | Yes | Yes (step 2) |
| 2 | RoPE (Q) | [1, 32, 1024, 128] = 8 MB | Yes | Yes (step 4) |
| 2 | RoPE (K) | [1, 32, 1024, 128] = 8 MB | Yes | Yes (step 3) |
| 4 | SDPA output | [1, 32, 1024, 128] = 8 MB | Yes | Yes (step 5) |
| 5 | `nlp_concat_heads` | [1, 1024, 4096] = 8 MB | Yes | Yes (step 6) |
| 6 | Output projection | [1, 1024, 4096] = 8 MB | Yes | Yes (next layer) |

**Total intermediate DRAM traffic within attention: ~88 MB written + ~88 MB read = ~176 MB per layer.**

### Specific Fusion Candidates

Three sub-sequences within attention are independently fusible:

1. **QKV Projection + Head Split**: Already partially fused in `TTNNFusedQKVSelfAttention` (weights concatenated, single matmul + `nlp_create_qkv_heads`). TT-Lang can eliminate the intermediate `query_key_value` tensor.

2. **RoPE Application**: Currently two separate `rotary_embedding` calls (one for Q, one for K). Can be fused into one kernel that processes both Q and K, and for partial rotary, avoids the slice-pad-compute-slice-concat dance.

3. **Fused Softmax + Value Multiply**: In the fallback SDPA path, `softmax(scores)` writes to DRAM then `matmul(scores, value)` reads it back. Fusing these keeps the softmax output in L1.

## TT-Lang Kernel Designs

### Design 1: Fused RoPE Kernel

This eliminates the slice/pad/concat overhead for partial rotary and processes Q and K in a single kernel launch:

```python
@ttl.operation(grid="auto")
def fused_rope(
    q_in: ttnn.Tensor,     # [batch, n_q_heads, seq_len, head_dim]
    k_in: ttnn.Tensor,     # [batch, n_k_heads, seq_len, head_dim]
    cos: ttnn.Tensor,      # [1, 1, seq_len, rotary_dim]
    sin: ttnn.Tensor,      # [1, 1, seq_len, rotary_dim]
    q_out: ttnn.Tensor,
    k_out: ttnn.Tensor,
) -> None:
    seq_tiles = q_in.shape[2] // ttl.TILE_SHAPE[0]
    head_tiles = q_in.shape[3] // ttl.TILE_SHAPE[1]
    rotary_tiles = cos.shape[3] // ttl.TILE_SHAPE[1]

    q_dfb = ttl.make_dataflow_buffer_like(q_in, shape=(1, 1), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k_in, shape=(1, 1), block_count=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(1, 1), block_count=2)
    sin_dfb = ttl.make_dataflow_buffer_like(sin, shape=(1, 1), block_count=2)
    q_out_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(1, 1), block_count=2)
    k_out_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for st in range(seq_tiles):
            # Load cos/sin tile (shared between Q and K)
            with cos_dfb.wait() as cos_blk, sin_dfb.wait() as sin_blk:
                # Process Q heads
                for ht in range(head_tiles):
                    with q_dfb.wait() as q_blk:
                        if ht < rotary_tiles:
                            # Rotary portion: q_rot = q * cos + rotate_half(q) * sin
                            q_rotated = q_blk * cos_blk + ttl.math.rotate_half(q_blk) * sin_blk
                        else:
                            # Pass-through portion (partial rotary)
                            q_rotated = q_blk
                        with q_out_dfb.reserve() as q_out_blk:
                            q_out_blk.store(q_rotated)

                # Process K heads (same cos/sin, no re-read from DRAM)
                for ht in range(head_tiles):
                    with k_dfb.wait() as k_blk:
                        if ht < rotary_tiles:
                            k_rotated = k_blk * cos_blk + ttl.math.rotate_half(k_blk) * sin_blk
                        else:
                            k_rotated = k_blk
                        with k_out_dfb.reserve() as k_out_blk:
                            k_out_blk.store(k_rotated)
```

**Key advantage:** cos/sin tiles are loaded once and reused for both Q and K. No slice/pad/concat overhead.

### Design 2: Fused QKV + RoPE Kernel

The most aggressive fusion: a single kernel that performs QKV projection, splits heads, and applies RoPE -- all without writing intermediates to DRAM:

```python
@ttl.operation(grid="auto")
def fused_qkv_rope(
    hidden_states: ttnn.Tensor,  # [batch, seq_len, hidden_size]
    qkv_weight: ttnn.Tensor,     # [hidden_size, 3*hidden_size]
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    q_out: ttnn.Tensor,
    k_out: ttnn.Tensor,
    v_out: ttnn.Tensor,
) -> None:
    # Compute QKV matmul tiles
    # For each output tile, determine if it belongs to Q, K, or V
    # If Q or K: apply RoPE in-register before writing
    # If V: write directly

    @ttl.compute()
    def compute():
        for seq_tile in range(seq_tiles):
            for head_tile in range(total_head_tiles):
                # Accumulate matmul
                acc = ttl.math.fill(0)
                for k_tile in range(hidden_tiles):
                    with h_dfb.wait() as h_blk, w_dfb.wait() as w_blk:
                        acc += h_blk @ w_blk

                # Determine Q/K/V region and apply RoPE if needed
                if head_tile < q_head_tiles:
                    # Q region: apply RoPE
                    with cos_dfb.wait() as c, sin_dfb.wait() as s:
                        acc = acc * c + ttl.math.rotate_half(acc) * s
                    with q_out_dfb.reserve() as out_blk:
                        out_blk.store(acc)
                elif head_tile < q_head_tiles + k_head_tiles:
                    # K region: apply RoPE
                    with cos_dfb.wait() as c, sin_dfb.wait() as s:
                        acc = acc * c + ttl.math.rotate_half(acc) * s
                    with k_out_dfb.reserve() as out_blk:
                        out_blk.store(acc)
                else:
                    # V region: no RoPE
                    with v_out_dfb.reserve() as out_blk:
                        out_blk.store(acc)
```

**Savings:** Eliminates the `query_key_value` intermediate (24 MB for LLaMA-7B scale) and the separate RoPE kernel launch.

### Design 3: Fused Softmax-Value in SDPA Fallback

When `ttnn.transformer.scaled_dot_product_attention` is not available (shape constraints, unsupported config), the fallback path in `_matmul_attention` (line 322) can be fused:

```python
@ttl.operation(grid="auto")
def fused_softmax_value_matmul(
    scores: ttnn.Tensor,   # [batch, heads, seq_q, seq_kv] -- pre-scaled, pre-masked
    value: ttnn.Tensor,    # [batch, heads, seq_kv, head_dim]
    out: ttnn.Tensor,
) -> None:
    @ttl.compute()
    def compute():
        for batch_head in range(batch * heads):
            for q_tile in range(seq_q_tiles):
                # Softmax over KV dimension (in L1)
                row_max = ttl.math.fill(-inf)
                for kv_tile in range(seq_kv_tiles):
                    with scores_dfb.wait() as s_blk:
                        row_max = ttl.math.max(row_max, s_blk)

                # Second pass: compute exp and sum
                exp_sum = ttl.math.fill(0)
                for kv_tile in range(seq_kv_tiles):
                    with scores_dfb.wait() as s_blk:
                        exp_blk = ttl.math.exp(s_blk - row_max)
                        exp_sum += ttl.math.reduce_sum(exp_blk)

                # Third pass: normalize and multiply by V
                acc = ttl.math.fill(0)
                for kv_tile in range(seq_kv_tiles):
                    with scores_dfb.wait() as s_blk, v_dfb.wait() as v_blk:
                        normed = ttl.math.exp(s_blk - row_max) / exp_sum
                        acc += normed @ v_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)
```

**Note:** This is a tile-streaming three-pass softmax, not a Flash Attention style online softmax. Flash Attention uses a single-pass online softmax with rescale-and-accumulate to avoid re-reading scores. Here, score tiles are re-read from DFBs across three passes (max, exp-sum, normalize-and-multiply), but all three passes operate on L1-resident tiles — no intermediate results are written to or re-read from DRAM.

## Integration with PagedAttentionKVCache

The `TTNNPagedAttentionKVCache` class (attention.py, line 77) manages block-based KV storage for decode. The fused attention kernels interact with it at two points:

1. **Prefill:** `paged_fill_cache()` writes K/V to the page table after RoPE. With fused QKV+RoPE, the K/V outputs write directly to the paged cache instead of an intermediate DRAM tensor.

2. **Decode:** `paged_sdpa_decode()` calls `ttnn.transformer.paged_scaled_dot_product_attention_decode` which already reads from the paged cache. The fused softmax-value kernel would need to accept page-table-based K/V access, which requires extending the DFB read pattern to use page-table indirection.

## Expected Benefit

| Fusion | DRAM Saved Per Layer | Launch Reduction | Complexity |
|--------|---------------------|-----------------|------------|
| Fused RoPE (Q+K) | ~16 MB (Q_rot + K_rot intermediates) | 2 launches -> 1 | Low |
| Fused QKV + RoPE | ~40 MB (QKV intermediate + RoPE intermediates) | 3 launches -> 1 | Medium |
| Fused Softmax-Value (fallback) | ~8 MB (softmax output) | 2 launches -> 1 | Medium |
| **All three combined** | **~64 MB per layer** | **7 launches -> 3** | High |

For a 32-layer LLaMA model, fusing all three saves ~2 GB of DRAM traffic per forward pass during prefill.

---

**Next:** [`fused_activations.md`](./fused_activations.md)
