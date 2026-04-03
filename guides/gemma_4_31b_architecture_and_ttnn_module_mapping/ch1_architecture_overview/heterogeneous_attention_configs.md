# Heterogeneous Attention Configurations

## Two Attention Types in One Model

Gemma 4 31B is architecturally heterogeneous: two structurally different
attention configurations coexist within the same decoder stack. Unlike models
that simply vary the attention mask (e.g., alternating causal and sliding), Gemma 4
varies the **head count**, **head dimension**, **RoPE configuration**, and
**KV projection strategy** between layer types. This means the weight tensors
have different shapes depending on the layer type.

Both attention types share the same `Gemma4TextAttention` class. The layer type
is determined at construction time via `config.layer_types[layer_idx]`, which
sets the `is_sliding` flag and configures all dependent parameters.

## Side-by-Side Comparison

| Parameter | Sliding Attention | Global Attention |
|-----------|-------------------|------------------|
| Layer type string | `"sliding_attention"` | `"full_attention"` |
| Count of layers | 50 | 10 |
| Query heads (`num_attention_heads`) | 32 | 32 |
| KV heads | 16 (`num_key_value_heads`) | 4 (`num_global_key_value_heads`) |
| GQA group size | 32 / 16 = 2 | 32 / 4 = 8 |
| Head dimension | 256 (`head_dim`) | 512 (`global_head_dim`) |
| Attention window | 1024 tokens (sliding) | Full causal (no window) |
| RoPE type | `"default"` | `"proportional"` |
| RoPE theta | 10000.0 | 1000000.0 |
| Partial rotary factor | 1.0 (all dims rotated) | 0.25 (128 of 512 dims) |
| Rotary dims | 256 | 128 |
| Non-rotary dims | 0 | 384 |
| K=V sharing (`attention_k_eq_v`) | No (separate K and V projections) | Yes (V reuses K projection) |
| V projection weight | Present | Absent (`None`) |
| Q-norm | RMSNorm(256, with scale) | RMSNorm(512, with scale) |
| K-norm | RMSNorm(256, with scale) | RMSNorm(512, with scale) |
| V-norm | RMSNorm(256, no scale) | RMSNorm(512, no scale) |
| Attention bias | `false` | `false` |

## Projection Weight Shapes

Because head dimension and KV head count differ between the two types, the
linear projection weights have different shapes:

| Projection | Sliding Shape | Global Shape | Notes |
|------------|---------------|--------------|-------|
| Q (`q_proj`) | [5376, 8192] | [5376, 16384] | $32 \times 256 = 8192$; $32 \times 512 = 16384$ |
| K (`k_proj`) | [5376, 4096] | [5376, 2048] | $16 \times 256 = 4096$; $4 \times 512 = 2048$ |
| V (`v_proj`) | [5376, 4096] | N/A (absent) | Global layers reuse K projection for V |
| O (`o_proj`) | [8192, 5376] | [16384, 5376] | Matches Q output dim |

### Per-Layer Attention Parameter Count (BF16)

For a single sliding layer:

$$
\text{Sliding attn params} = 5376 \times 8192 + 5376 \times 4096 + 5376 \times 4096 + 8192 \times 5376 = 132,120,576
$$

For a single global layer (K=V sharing eliminates the V projection):

$$
\text{Global attn params} = 5376 \times 16384 + 5376 \times 2048 + 16384 \times 5376 = 187,170,816
$$

## K=V Sharing in Global Layers

When `attention_k_eq_v=True` and the layer is not sliding, the attention module
sets `use_alternative_attention=True`. This has two effects:

1. **V projection is not instantiated**: `self.v_proj = None`.
2. **K projection output is reused as V input**: In the forward pass, when
   `self.v_proj is None`, the code executes `value_states = key_states` (before
   either has been normalized or had RoPE applied).

After this shared projection, the K and V tensors diverge:

- **K path**: `k_norm` (scaled RMSNorm with learned weight) followed by RoPE
  (partial, 128/512 dims).
- **V path**: `v_norm` (unscaled RMSNorm, `with_scale=False`) with **no RoPE**.

This means K and V start from the same linear projection but receive different
normalization and positional encoding. The net effect is that the single
`k_proj` weight matrix serves as a shared basis, while the divergent
post-processing creates the distinct K and V representations the attention
mechanism requires.

```text
  hidden_states
       |
       v
  k_proj (linear)  [5376, 2048]
       |
       +------ key_states ------+------ value_states ------+
       |                        |                           |
       v                        v                           |
   k_norm                   v_norm                          |
   (RMSNorm, scaled)       (RMSNorm, no scale)             |
       |                        |                           |
       v                        v                           |
   RoPE (partial)          (no RoPE)                        |
   128/512 dims                 |                           |
       |                        |                           |
       v                        v                           |
   key_states              value_states                     |
   (for SDPA)              (for SDPA)                       |
```

## Implications for TTNN Implementation

The heterogeneous attention design creates several implementation challenges:

1. **Two sets of weight shapes**: The `TTNNModule` for attention must handle
   different weight tensor dimensions depending on the layer type. Program
   configs for `ttnn.linear` will differ between sliding and global layers.

2. **Two KV cache configurations**: Sliding layers store a 1024-token window;
   global layers store the full sequence. The paged KV cache must accommodate
   both. The higher GQA ratio in global layers (8 vs. 2) significantly reduces
   KV cache memory for long-context full-attention, which is critical because
   global layers must cache the entire sequence (up to 256K tokens) rather than
   just a 1024-token window.

3. **Two RoPE configurations**: Cos/sin embedding tables must be precomputed
   for both theta=10000 (full rotation, dim=256) and theta=1000000 (partial
   rotation, dim=128).

4. **Conditional V projection**: Global layers skip the V matmul entirely,
   replacing it with a clone of the K projection output. The TTNN forward pass
   must branch on layer type.

5. **Different TP sharding**: With 16 KV heads for sliding and 4 for global,
   the tensor-parallel sharding strategy across the T3K 8-device mesh differs.
   See Chapter 6 for analysis.

---

**Next:** [`novel_components.md`](./novel_components.md)
