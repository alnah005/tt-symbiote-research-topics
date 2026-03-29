# Attention Architecture: Five Differences from Standard Transformers

The `Qwen35Attention` class in `attention.py` implements a modified multi-head attention mechanism. Its module docstring enumerates five key differences from the standard `tt_transformers` attention implementation:

1. **Partial RoPE**: Only 64 of 256 head dimensions receive rotary position embeddings
2. **QK L2 norms**: Query and key vectors are L2-normalized with a learned scale (not RMSNorm in the traditional sense)
3. **Sigmoid output gating**: The attention output is element-wise multiplied by `sigmoid(gate)`
4. **Fused Q+gate projection**: A single weight matrix `wqkv` projects to `[Q, gate]` interleaved, producing `NH * HD * 2` outputs
5. **Separate K/V projections**: Unlike fused QKV in standard attention, K and V use independent weight matrices `wk` and `wv`

These modifications work together to give the model more expressive control over what information flows through the attention layer, while the partial RoPE and L2 norms improve training stability at the 27B parameter scale.

## Projection Structure

In `forward_decode()`, the three projections are dispatched as separate DRAM-sharded matmuls:

```python
# Fused Q+gate: [1, B, dim] -> [1, B, NH*HD*2] = [1, 32, 6*256*2] = [1, 32, 3072]
qg_tt = _unshard(_shard_linear(x, tw["wqkv"], act_shard, self.args.attn_qg_progcfg, self.compute_cfg))

# Key: [1, B, dim] -> [1, B, NKV*HD] = [1, 32, 1*256] = [1, 32, 256]
kp_tt = _unshard(_shard_linear(x, tw["wk"], act_shard, self.args.attn_k_progcfg, self.compute_cfg))

# Value: [1, B, dim] -> [1, B, NKV*HD] = [1, 32, 1*256] = [1, 32, 256]
vp_tt = _unshard(_shard_linear(x, tw["wv"], act_shard, self.args.attn_v_progcfg, self.compute_cfg))
```

The Q+gate output is then reshaped to `[1, B, NH, HD*2]` and split:

```python
qg_r = ttnn.reshape(qg_tt, (1, B, NH, HD * 2))  # [1, 32, 6, 512]
q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD))           # [1, 32, 6, 256]
gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2))   # [1, 32, 6, 256]
```

Here `NH = n_local_heads = 6` (24 Q heads / TP=4) and `NKV = n_local_kv_heads = 1` (4 KV heads / TP=4). Each device handles 6 query heads but only 1 KV head, so K and V projections are much smaller than the Q+gate projection.

## Partial RoPE

Standard transformers apply RoPE to all head dimensions. Qwen3.5-27B applies RoPE to only the **first 64 of 256 dimensions**, defined by the constant `ROPE_DIM = 64` in `model_config.py`. The remaining 192 dimensions pass through unchanged. This is handled by the `Qwen35PartialRopeSetup` class in `rope.py` and the functions `apply_partial_rope_decode` and `apply_partial_rope_prefill`.

### Qwen35PartialRopeSetup

This class extends `RotarySetup` (from `models.tt_transformers.tt.rope`) but initializes with `head_dim=ROPE_DIM` instead of the full 256:

```python
class Qwen35PartialRopeSetup(RotarySetup):
    def __init__(self, device, batch_size, head_dim, max_seq_len, rope_theta=10_000_000.0, ...):
        super().__init__(
            device=device, batch_size=batch_size,
            head_dim=ROPE_DIM,  # 64, not the full 256
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            ...
        )
        self.full_head_dim = head_dim
```

The constructor precomputes cos/sin tables in **HuggingFace split-halves format**:

```python
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
t = torch.arange(max_seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)                # [max_seq_len, 32]
emb = torch.cat([freqs, freqs], dim=-1)          # [max_seq_len, 64] -- split-halves format
```

The resulting `_cos_table` and `_sin_table` tensors have shape `[1, max_seq_len, 64]`, stored in TILE_LAYOUT on all devices via `ReplicateTensorToMesh`.

The `get_rot_mats()` method uses `ttnn.embedding` to look up per-position cos/sin values, returning tensors of shape `[1, B, 1, ROPE_DIM]` for decode.

### apply_partial_rope_decode

The decode RoPE function in `rope.py` implements the slice-rotate-concat pattern:

```python
def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim=ROPE_DIM):
    # x: [1, B, n_heads, HD=256]
    x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))           # first 64 dims
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd))          # remaining 192 dims

    # HF split-halves rotation: [-x2, x1] where x1=first_half, x2=second_half
    r1 = ttnn.slice(x_rope, (0, 0, 0, 0), (1, B, n_heads, rope_dim // 2))     # dims 0-31
    r2 = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, B, n_heads, rope_dim))  # dims 32-63
    x_rot = ttnn.concat([ttnn.neg(r2), r1], dim=-1)

    roped = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    result = ttnn.concat([roped, x_pass], dim=-1)   # [1, B, n_heads, 256] restored
```

This is applied identically to both Q and K in `forward_decode()`:

```python
q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B)   # NH=6
k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B)  # NKV=1
```

### apply_partial_rope_prefill

The prefill variant operates on the head-major layout `[1, n_heads, seq_len, HD]` instead of the batch-major `[1, B, n_heads, HD]` used in decode. The slice and rotation logic is identical, but all slice coordinates use `n_heads` in dim 1 and `seq_len` in dim 2:

```python
x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, n_heads, seq_len, rope_dim))
x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd))
```

For prefill, the cos/sin tables are obtained via `get_prefill_rot_mats()`, which slices the precomputed tables to `[1, 1, seq_len, ROPE_DIM]` -- these broadcast across all heads.

## QK L2 Normalization

After projection and before RoPE, both Q and K undergo L2 normalization followed by multiplication with a learned scale:

```python
q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])
```

The `_rms_norm_dev` function calls `ttnn.rms_norm(x, epsilon=1e-6)`, which computes `x / sqrt(mean(x^2) + eps)` along the last dimension. This is mathematically equivalent to L2 normalization (dividing by the L2 norm scaled by `sqrt(dim)`) and is numerically more stable at scale. The learned scale weights `q_norm` and `k_norm` are per-head-dimension vectors that allow the model to control the effective magnitude of the normalized Q and K vectors.

This normalization stabilizes the dot product attention scores, preventing them from growing with model dimension and improving training stability -- a technique particularly important at the 27B parameter scale.

## Sigmoid Output Gating

After the attention computation produces `attn_out`, it is element-wise gated by the sigmoid of the gate vector (which was projected alongside Q from the same input):

```python
gate_val = ttnn.sigmoid(gate)
gated = ttnn.multiply(attn_out, gate_val)
```

This adds a learned, input-dependent scaling factor to every attention output dimension. Unlike the residual connection (which adds the attention output to the input), the sigmoid gate can selectively suppress or pass through attention contributions on a per-dimension basis. The gate shares the same projection input as Q, so it has full access to the input context when deciding how much attention output to allow.

## Output Projection and All-Reduce

After gating, the output is reshaped from `[1, B, NH, HD]` to `[1, B, NH*HD]` = `[1, 32, 1536]` (per device), passed through the row-parallel output projection `wo`, and then all-reduced across the 4 TP devices:

```python
gated_flat = ttnn.reshape(gated, (1, B, NH * HD))           # [1, 32, 1536]
wo_partial = _unshard(_shard_linear(gated_flat, tw["wo"], act_shard_out, self.args.attn_wo_progcfg, self.compute_cfg))
wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
return self._all_reduce(wo_partial)
```

The `_all_reduce` method uses `tt_all_reduce` with the CCL ring topology, summing partial results across 4 devices to produce the final `[1, 1, B, dim]` output where `dim=5120`.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`dram_sharded_decode.md`](./dram_sharded_decode.md)
