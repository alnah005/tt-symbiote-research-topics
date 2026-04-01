# Attention Architecture: Five Differences from Standard Transformers

The `Qwen35Attention` class in `attention.py` implements a modified multi-head attention mechanism. Its module docstring (lines 4–13) enumerates five key differences from the standard `tt_transformers` attention implementation:

1. **Partial RoPE**: Only 64 of 256 head dimensions receive rotary position embeddings (`ROPE_DIM = 64` in `model_config.py:40`)
2. **QK L2 norms**: Query and key vectors are L2-normalized with a learned scale, not the standard pre-norm RMSNorm applied to hidden states
3. **Sigmoid output gating**: The attention output is element-wise multiplied by `sigmoid(gate)` before the output projection
4. **Fused Q+gate projection**: A single weight matrix `wqkv` projects to `[Q, gate]` interleaved, producing `NH * HD * 2` outputs per device
5. **Separate K/V projections**: Unlike fused QKV in standard attention, K and V use independent weight matrices `wk` and `wv`

## Projection Structure

In `forward_decode()` (attention.py:136), the three projections are dispatched as separate DRAM-sharded matmuls:

```python
# Fused Q+gate: projects to NH*HD*2 outputs per device
qg_tt = _unshard(_shard_linear(x, tw["wqkv"], act_shard, self.args.attn_qg_progcfg, self.compute_cfg))

# Key: projects to NKV*HD outputs per device
kp_tt = _unshard(_shard_linear(x, tw["wk"],   act_shard, self.args.attn_k_progcfg,  self.compute_cfg))

# Value: projects to NKV*HD outputs per device
vp_tt = _unshard(_shard_linear(x, tw["wv"],   act_shard, self.args.attn_v_progcfg,  self.compute_cfg))
```

The Q+gate output is then reshaped to `[1, B, NH, HD*2]` and split along the last dimension (attention.py:156–159):

```python
qg_r = ttnn.reshape(qg_tt, (1, B, NH, HD * 2))
q    = ttnn.slice(qg_r, (0, 0, 0,  0),  (1, B, NH, HD))
gate = ttnn.slice(qg_r, (0, 0, 0, HD),  (1, B, NH, HD * 2))
```

With TP=4 on P150x4: `NH = n_local_heads = 6` (24 Q heads / 4 devices) and `NKV = n_local_kv_heads = 1` (4 KV heads / 4 devices). Each device handles 6 query heads but only 1 KV head, so K and V projections are much smaller than the Q+gate projection.

## Partial RoPE

Standard transformers apply RoPE to all head dimensions. Qwen3.5-27B applies RoPE to only the **first 64 of 256 dimensions**, defined by `ROPE_DIM = 64` in `model_config.py:40`. The remaining 192 dimensions pass through unchanged. This is handled by the `Qwen35PartialRopeSetup` class in `rope.py` and the functions `apply_partial_rope_decode` and `apply_partial_rope_prefill`.

### Qwen35PartialRopeSetup

This class extends `RotarySetup` (from `models.tt_transformers.tt.rope`) but passes `head_dim=ROPE_DIM` (64) to the parent instead of the full 256 (rope.py:15–43):

```python
class Qwen35PartialRopeSetup(RotarySetup):
    def __init__(self, device, batch_size, head_dim, max_seq_len, rope_theta=10_000_000.0, ...):
        super().__init__(
            device=device, batch_size=batch_size,
            head_dim=ROPE_DIM,   # 64, not the full head_dim=256
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            ...
        )
        self.full_head_dim = head_dim
```

The constructor precomputes cos/sin tables in **HuggingFace split-halves format** (rope.py:46–60):

```python
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
t = torch.arange(max_seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)              # [max_seq_len, 32]
emb = torch.cat([freqs, freqs], dim=-1)       # [max_seq_len, 64] -- split-halves
```

The resulting `_cos_table` and `_sin_table` tensors have shape `[1, max_seq_len, 64]`, stored in TILE_LAYOUT on all devices via `ReplicateTensorToMesh`.

The `get_rot_mats()` method (rope.py:63–81) uses `ttnn.embedding` to look up per-position cos/sin values, then transposes to return tensors of shape `[1, B, 1, ROPE_DIM]` for use in decode.

### apply_partial_rope_decode

The decode RoPE function (rope.py:134) operates on the batch-major layout `[1, B, n_heads, HD]` and implements the slice-rotate-concat pattern:

```python
def apply_partial_rope_decode(x, cos_tt, sin_tt, n_heads, batch_size, rope_dim=ROPE_DIM):
    # x: [1, B, n_heads, HD=256]; cos_tt/sin_tt: [1, B, 1, 64]
    x_rope = ttnn.slice(x, (0, 0, 0,        0), (1, B, n_heads, rope_dim))  # first 64 dims
    x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd))        # remaining 192 dims

    # HF split-halves rotation: form [-x2, x1] where x1 = dims 0-31, x2 = dims 32-63
    r1     = ttnn.slice(x_rope, (0, 0, 0,           0), (1, B, n_heads, rope_dim // 2))
    r2     = ttnn.slice(x_rope, (0, 0, 0, rope_dim // 2), (1, B, n_heads, rope_dim))
    x_rot  = ttnn.concat([ttnn.neg(r2), r1], dim=-1)

    roped  = ttnn.add(ttnn.multiply(x_rope, cos_tt), ttnn.multiply(x_rot, sin_tt))
    result = ttnn.concat([roped, x_pass], dim=-1)   # restored to [1, B, n_heads, 256]
```

This is applied to both Q and K in `forward_decode()` (attention.py:172–174):

```python
q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH,  B)   # NH=6
k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B)   # NKV=1
```

### apply_partial_rope_prefill

The prefill variant (rope.py:96) operates on the head-major layout `[1, n_heads, seq_len, HD]` instead of the batch-major `[1, B, n_heads, HD]` used in decode. The rotation logic is identical; only the slice coordinate positions change:

```python
x_rope = ttnn.slice(x, (0, 0, 0,        0), (1, n_heads, seq_len, rope_dim))
x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, n_heads, seq_len, hd))
```

For prefill, the cos/sin tables are obtained via `get_prefill_rot_mats()` (rope.py:84–93), which slices the precomputed tables to `[1, 1, seq_len, ROPE_DIM]` — these broadcast across all heads during the multiply.

## QK L2 Normalization

After projection and before RoPE, both Q and K undergo L2 normalization followed by multiplication with a learned per-dimension scale (attention.py:167–169, 391–392):

```python
q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])
```

The `_rms_norm_dev` function (attention.py:44–46) calls `ttnn.rms_norm(x, epsilon=1e-6)`, which computes:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}}$$

along the last dimension. This is mathematically L2 normalization (up to a factor of $\sqrt{d}$) and is numerically stable at scale. The learned scale weights `q_norm` and `k_norm` are per-head-dimension vectors, loaded from the state dict, that allow the model to control the effective magnitude of the normalized Q and K vectors after normalization.

## Sigmoid Output Gating

After the attention computation produces `attn_out`, it is element-wise gated by the sigmoid of the gate vector (attention.py:233–237):

```python
gate_val = ttnn.sigmoid(gate)
gated    = ttnn.multiply(attn_out, gate_val)
```

The gate was projected alongside Q from the same input `x` using `wqkv`, so it has access to the full input context when deciding how much of each attention output dimension to pass through. Values near 0 suppress that dimension; values near 1 pass it through unchanged.

## Output Projection and All-Reduce

After gating, the output is reshaped from `[1, B, NH, HD]` to `[1, B, NH*HD]` (1536 per device with TP=4), passed through the row-parallel output projection `wo`, and then all-reduced across the 4 TP devices (attention.py:240–248):

```python
gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
wo_partial  = _unshard(_shard_linear(gated_flat, tw["wo"],
                                     act_shard_out, self.args.attn_wo_progcfg, self.compute_cfg))
wo_partial  = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
return self._all_reduce(wo_partial)
```

The `_all_reduce` method (attention.py:294–300) calls `tt_all_reduce` with `cluster_axis=0` and `dim=3`, summing partial results across 4 devices to produce the final `[1, 1, B, dim]` output where `dim=5120`.

---

**Next:** [`dram_sharded_decode.md`](./dram_sharded_decode.md)
