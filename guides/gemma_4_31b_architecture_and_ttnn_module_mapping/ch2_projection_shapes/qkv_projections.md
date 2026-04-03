# QKV Projections

This file derives the exact weight and activation tensor shapes for the Q, K, V,
and O linear projections in both sliding and global attention layers. These
shapes are the starting point for configuring `ttnn.linear` program configs and
tensor-parallel sharding.

All projections use no bias (`attention_bias=false`), so each projection is a
single weight matrix with no additive bias term.

## Shape Derivation

The general formulas for attention projection shapes follow from the config
parameters defined in
[Chapter 1](../ch1_architecture_overview/index.md):

```math
W_Q: [\text{hidden size},\; \text{num heads} \times \text{head dim}]
```

```math
W_K: [\text{hidden size},\; \text{num kv heads} \times \text{head dim}]
```

```math
W_V: [\text{hidden size},\; \text{num kv heads} \times \text{head dim}]
```

```math
W_O: [\text{num heads} \times \text{head dim},\; \text{hidden size}]
```

Substituting the config values for each layer type yields the concrete shapes
below.

## Q Projection

| Parameter | Sliding | Global |
|-----------|---------|--------|
| `num_attention_heads` | 32 | 32 |
| `head_dim` | 256 | 512 |
| Q output dim | 32 x 256 = 8192 | 32 x 512 = 16384 |
| **Weight shape** | **[5376, 8192]** | **[5376, 16384]** |

The Q projection is the largest attention weight in global layers. At BF16, the
global Q weight occupies 176 MB per layer --- roughly double the sliding Q
weight (88 MB).

### Q Reshape After Projection

After the matmul, the Q activation is reshaped from a flat vector into
per-head format for the attention computation:

```math
[B, S, \text{num heads} \times \text{head dim}] \rightarrow [B, \text{num heads}, S, \text{head dim}]
```

For batch=1 single-token decode:

- Sliding: `[1, 1, 8192]` -> `[1, 32, 1, 256]`
- Global: `[1, 1, 16384]` -> `[1, 32, 1, 512]`

## K Projection

| Parameter | Sliding | Global |
|-----------|---------|--------|
| `num_kv_heads` | 16 | 4 |
| `head_dim` | 256 | 512 |
| K output dim | 16 x 256 = 4096 | 4 x 512 = 2048 |
| **Weight shape** | **[5376, 4096]** | **[5376, 2048]** |

The global K projection is the smallest attention weight because it combines
few KV heads (4) with K=V sharing. This is an intentional design: the high GQA
ratio (32Q : 4KV = 8:1) in global layers drastically reduces the KV cache
memory required for full-context attention over up to 256K tokens.

### K Reshape After Projection

```math
[B, S, \text{num kv heads} \times \text{head dim}] \rightarrow [B, \text{num kv heads}, S, \text{head dim}]
```

For batch=1 single-token decode:

- Sliding: `[1, 1, 4096]` -> `[1, 16, 1, 256]`
- Global: `[1, 1, 2048]` -> `[1, 4, 1, 512]`

### K Post-Processing

After reshape, the K tensor receives two operations before entering the KV
cache:

1. **K-norm**: `RMSNorm(head_dim, eps=1e-6, with_scale=True)` --- a learned
   per-element scale applied per-head.
2. **RoPE**: Rotary position embedding applied to Q and K.
   - Sliding: full rotation, theta=10000, all 256 dims.
   - Global: partial rotation, theta=1000000, first 128 of 512 dims.

See [Chapter 4](../ch4_dual_rope/index.md) for full RoPE details.

## V Projection

### Sliding Layers (Separate V Projection)

| Parameter | Value |
|-----------|-------|
| `num_kv_heads` | 16 |
| `head_dim` | 256 |
| V output dim | 16 x 256 = 4096 |
| **Weight shape** | **[5376, 4096]** |

Sliding layers have a dedicated `v_proj` weight matrix with the same shape as
`k_proj`. The V activation is reshaped identically to K:

- Decode: `[1, 1, 4096]` -> `[1, 16, 1, 256]`

### Global Layers (K=V Sharing --- No V Weight)

In global layers, `attention_k_eq_v=true` causes the V projection to be
eliminated entirely. The `v_proj` attribute is set to `None` and no weight
tensor is instantiated.

Instead, the K projection output is assigned to both `key_states` and
`value_states` **before** any normalization or RoPE is applied. The two tensors
then diverge through separate post-processing paths:

```text
hidden_states  [1, 1, 5376]
      |
      v
  k_proj linear  [5376, 2048]
      |
      +--- key_states [1, 1, 2048] ---+--- value_states [1, 1, 2048]
      |                                |
      v                                v
  reshape [1, 4, 1, 512]          reshape [1, 4, 1, 512]
      |                                |
      v                                v
  k_norm (scaled RMSNorm)         v_norm (unscaled RMSNorm)
      |                                |
      v                                |
  RoPE (partial, 128/512 dims)        (no RoPE)
      |                                |
      v                                v
  key_states for SDPA             value_states for SDPA
```

This saves one `[5376, 2048]` weight matrix per global layer (~22 MB at BF16,
~220 MB total across 10 global layers). See
[Chapter 3](../ch3_kv_sharing_and_vnorm/index.md) for the full K=V sharing
mechanism and TTNN implementation.

### V Post-Processing (Both Layer Types)

After reshape, the V tensor receives V-norm:

- **V-norm**: `RMSNorm(head_dim, eps=1e-6, with_scale=False)` --- normalizes
  by RMS magnitude without a learned scale parameter.
- **No RoPE** is applied to V in either layer type.

## O Projection

| Parameter | Sliding | Global |
|-----------|---------|--------|
| `num_attention_heads` | 32 | 32 |
| `head_dim` | 256 | 512 |
| O input dim | 32 x 256 = 8192 | 32 x 512 = 16384 |
| **Weight shape** | **[8192, 5376]** | **[16384, 5376]** |

The O projection maps the concatenated multi-head attention output back to
`hidden_size`. Its input dimension matches the Q output dimension.

### O Activation Shapes

The SDPA output is first reshaped from per-head format back to a flat vector,
then projected:

```math
[B, \text{num heads}, S, \text{head dim}] \rightarrow [B, S, \text{num heads} \times \text{head dim}]
```

For batch=1 single-token decode:

- Sliding: `[1, 32, 1, 256]` -> `[1, 1, 8192]` -> O proj -> `[1, 1, 5376]`
- Global: `[1, 32, 1, 512]` -> `[1, 1, 16384]` -> O proj -> `[1, 1, 5376]`

After the O projection, the output is always `[B, S, 5376]` regardless of layer
type. This uniform output shape is what allows the residual connection to work
identically for both layer types.

## Decode Activation Shape Summary

The following table shows activation shapes at each stage of the attention
forward pass for batch=1 single-token decode (`B=1, S=1`).

### Sliding Layer Activations

| Stage | Shape | Notation |
|-------|-------|----------|
| Input hidden states | [1, 1, 5376] | [B, S, hidden_size] |
| After Q proj | [1, 1, 8192] | [B, S, num_heads x head_dim] |
| Q reshaped | [1, 32, 1, 256] | [B, H, S, D] |
| After K proj | [1, 1, 4096] | [B, S, num_kv_heads x head_dim] |
| K reshaped | [1, 16, 1, 256] | [B, Hkv, S, D] |
| After V proj | [1, 1, 4096] | [B, S, num_kv_heads x head_dim] |
| V reshaped | [1, 16, 1, 256] | [B, Hkv, S, D] |
| SDPA output | [1, 32, 1, 256] | [B, H, S, D] |
| O proj input | [1, 1, 8192] | [B, S, num_heads x head_dim] |
| O proj output | [1, 1, 5376] | [B, S, hidden_size] |

### Global Layer Activations

| Stage | Shape | Notation |
|-------|-------|----------|
| Input hidden states | [1, 1, 5376] | [B, S, hidden_size] |
| After Q proj | [1, 1, 16384] | [B, S, num_heads x head_dim] |
| Q reshaped | [1, 32, 1, 512] | [B, H, S, D] |
| After K proj (shared) | [1, 1, 2048] | [B, S, num_kv_heads x head_dim] |
| K reshaped | [1, 4, 1, 512] | [B, Hkv, S, D] |
| V (clone of K pre-norm) | [1, 4, 1, 512] | [B, Hkv, S, D] |
| SDPA output | [1, 32, 1, 512] | [B, H, S, D] |
| O proj input | [1, 1, 16384] | [B, S, num_heads x head_dim] |
| O proj output | [1, 1, 5376] | [B, S, hidden_size] |

## TTNN Implementation Notes

### Program Config Considerations

The attention projections span a wide range of matmul sizes:

- **Smallest**: Global K projection `[5376, 2048]` --- a relatively narrow
  matmul that may be memory-bound on Wormhole.
- **Largest**: Global Q and O projections at `[5376, 16384]` and `[16384, 5376]`
  --- large enough to be compute-bound with good utilization.

Each unique shape requires its own `ttnn.linear` program config. Since sliding
and global layers have different projection shapes, the model needs at minimum
two sets of attention program configs (one per layer type).

### Fused QKV Optimization

For sliding layers, the Q, K, and V projections can be fused into a single
matmul by concatenating the weight matrices:

```math
W_{QKV}^{\text{sliding}} = [W_Q \mid W_K \mid W_V] \quad \text{shape: } [5376, 16384]
```

where `16384 = 8192 + 4096 + 4096`.

For global layers, K=V sharing means only Q and K projections exist. These can
be fused as:

```math
W_{QK}^{\text{global}} = [W_Q \mid W_K] \quad \text{shape: } [5376, 18432]
```

where `18432 = 16384 + 2048`.

The fused output is then sliced to recover individual Q, K, and (for sliding) V
tensors. The advantage is a single large matmul instead of 2--3 smaller ones,
which typically achieves better hardware utilization.

### Column-Parallel and Row-Parallel Sharding

Under tensor parallelism across the T3K 8-device mesh:

- **Q, K, V projections** use column-parallel sharding: the output dimension is
  split across devices.
- **O projection** uses row-parallel sharding: the input dimension is split
  across devices, with an `ttnn.all_reduce` to combine partial results.

See [Chapter 6](../ch6_tp_sharding/index.md) for the full sharding analysis.

---

**Next:** [`ffn_projections.md`](./ffn_projections.md)
