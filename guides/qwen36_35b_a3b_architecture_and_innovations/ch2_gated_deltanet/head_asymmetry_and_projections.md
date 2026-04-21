# Head Asymmetry and Projections

This file documents the QK/V head asymmetry, the GQA-style expansion that resolves it, the
complete projection inventory with tensor shapes, and the causal conv1d local-mixing mechanism.
All shapes are for Qwen3.6-35B-A3B (H = 2048, $H_k$ = 16, $H_v$ = 32, $d_k = d_v$ = 128).

---

## 1. QK/V Head Asymmetry

Qwen3.6-35B-A3B uses a **grouped query** configuration in its Gated DeltaNet layers:

```
Key/Query heads  (H_k):  16
Value heads      (H_v):  32
Key/Query dim    (d_k):  128
Value dim        (d_v):  128
GQA ratio:                2   (H_v / H_k = 32 / 16)
```

There are twice as many value heads as key/query heads. This asymmetry is deliberate:

- **Parameter savings**: projecting Q and K to only 16 heads halves their projection cost
  compared to projecting to 32 heads.
- **State dimensionality preserved**: the recurrent state $S$ is still shaped
  $[H_v, d_k, d_v] = [32, 128, 128]$ per batch element, because one key/query head is
  shared across 2 value heads. The state capacity is unchanged.
- **Retrieval quality**: each K/Q head serves two V heads, so retrieval uses a consistent
  addressing vector across the pair. This matches the GQA pattern used in the Gated Attention
  (softmax) layers of the same model.

### GQA Expansion via `repeat_interleave`

After the input projection and conv1d (see below), Q and K are reshaped into
$[B, T, H_k, d_k] = [B, T, 16, 128]$ and then expanded:

```python
# Q and K start at shape [B, T, 16, 128]
Q = Q.repeat_interleave(gqa_ratio, dim=2)   # → [B, T, 32, 128]
K = K.repeat_interleave(gqa_ratio, dim=2)   # → [B, T, 32, 128]
```

`repeat_interleave(2)` duplicates each head in place: head 0 → heads 0 and 1, head 1 → heads
2 and 3, etc. After this expansion, Q and K have the same head count as V (32 heads), and the
recurrence can be applied identically across all 32 (Q, K, V) head triples.

The duplication is logically equivalent to saying: Q-head 0 and K-head 0 are each responsible
for addressing two state matrices (the matrices for V-head 0 and V-head 1). This is the same
GQA sharing convention used in the Gated Attention layers, applied here to the linear attention
recurrence.

---

## 2. Full Projection Inventory

A single Gated DeltaNet layer applies six projections or transformations to the input hidden
state $x \in [B, T, H]$. In the TTNN implementation these are fused where possible into a
single matmul (`in_proj_all`), but conceptually they are distinct.

### 2.1 Combined QKV Projection: `in_proj_qkv`

```
in_proj_qkv:  [B, T, H] → [B, T, key_dim×2 + value_dim]
                        = [B, T, 2×2048 + 4096]
                        = [B, T, 8192]
```

where `key_dim = H_k × d_k = 16 × 128 = 2048` and `value_dim = H_v × d_v = 32 × 128 = 4096`.

This output is then split along the last dimension:

```
Q:  [B, T, 2048]   (key_dim)
K:  [B, T, 2048]   (key_dim)
V:  [B, T, 4096]   (value_dim)
```

and reshaped into heads:

```
Q:  [B, T, 16, 128]   = [B, T, H_k, d_k]
K:  [B, T, 16, 128]   = [B, T, H_k, d_k]
V:  [B, T, 32, 128]   = [B, T, H_v, d_v]
```

Note: the conv1d is applied to the concatenated QKV flat tensor **before** this reshape and
split. See Section 3.

### 2.2 Output Gate Projection: `in_proj_z`

```
in_proj_z:  [B, T, H] → [B, T, value_dim]
                      = [B, T, 4096]
```

The output $Z$ is used in the gated RMSNorm post-recurrence (see `delta_rule_formulation.md` §6
for the full formula and derivation). $z_t$ is reshaped to $[B, T, H_v, d_v] = [B, T, 32, 128]$
before the gate is applied per-head.

### 2.3 Decay Input Projection: `in_proj_a`

```
in_proj_a:  [B, T, H] → [B, T, H_v]
                      = [B, T, 32]
```

Each of the 32 output scalars $a_t[h]$ feeds into the decay gate computation for head $h$:

$$\alpha_t[h] = -\exp(A_{\log}[h]) \cdot \text{softplus}(a_t[h] + \text{dt\_bias}[h])$$

$$g_t[h] = \exp(\alpha_t[h])$$

This is a one-scalar-per-value-head parameterization. In the Qwen3.5-9B configuration the same
projection also has 32 outputs (one per head). In both cases each value head has its own
independent decay rate at every token step, controlled by a low-dimensional linear readout of
the hidden state.

### 2.4 Beta Logit Projection: `in_proj_b`

```
in_proj_b:  [B, T, H] → [B, T, H_v]
                      = [B, T, 32]
```

Each scalar $b_t[h]$ passes through sigmoid to produce the update rate for head $h$:

$$\beta_t[h] = \sigma(b_t[h])$$

The same one-scalar-per-value-head structure as `in_proj_a`. Large $\beta_t[h]$ (near 1) means
"correct aggressively"; small $\beta_t[h]$ (near 0) means "mostly read, minimal update."

### 2.5 Causal Conv1d

```
conv1d input:   [B, conv_dim, T] = [B, 8192, T]
kernel:         depthwise, kernel_size=4, stride=1, causal
output:         [B, 8192, T]
```

where `conv_dim = 2 × key_dim + value_dim = 2 × 2048 + 4096 = 8192`. The conv1d is applied to
the flat QKV slice (i.e., to the combined output of `in_proj_qkv`) before splitting into Q, K,
V. This is a depthwise convolution: each channel is convolved independently.

After the conv1d, a SiLU activation is applied elementwise. The output is then split and
reshaped into Q, K, V per the shapes in Section 2.1.

### 2.6 Output Projection: `out_proj`

```
out_proj:  [B, T, value_dim] → [B, T, H]
                             = [B, T, 4096] → [B, T, 2048]
```

The recurrence output (after gated RMSNorm) is reshaped from $[B, T, H_v, d_v]$ to
$[B, T, H_v \cdot d_v] = [B, T, 4096]$ and then projected back to the model's hidden dimension.

---

## 3. Conv1d Local Mixing

### Purpose

The causal conv1d (kernel size 4) gives each token a **local receptive field** of 4 adjacent
tokens. This provides short-range context that the recurrent state $S$ does not naturally
capture: while $S$ can in principle encode any past token's information, its fixed capacity
means local context from the immediately preceding 1–3 tokens may not be reliably retained
across many intervening layers. The conv1d directly mixes the projected Q, K, V activations of
the current and 3 prior tokens.

This is structurally similar to the local convolution in Mamba: a short-range mixer applied to
the input projections before the recurrence, complementing the recurrence's long-range
(but bounded-capacity) global mixing.

### Prefill (Batch Mode)

During prefill, the conv1d operates in standard `F.conv1d` mode over the full sequence:

```python
# qkv_flat: [B, conv_dim, T]
conv_out = F.conv1d(qkv_flat, weight, bias=None, groups=conv_dim, padding=kernel_size - 1)
conv_out = conv_out[:, :, :T]   # trim causal padding
conv_out = F.silu(conv_out)
```

The depthwise (groups = conv_dim) formulation means each of the 8192 channels has its own
set of 4 scalar weights — 8192 × 4 = 32,768 parameters total for the conv1d layer.

### Decode (Shift Register)

During autoregressive decode, only one token is processed at a time (T = 1). The standard
`F.conv1d` cannot be applied directly because it would need to access the 3 preceding token
projections. These are maintained in a 4-slot circular shift register on device:

```
conv_state: [B, conv_dim, kernel_size] = [B, 8192, 4]
```

At each decode step:

1. The new QKV projection for the current token is written into the oldest slot of the shift
   register (overwriting it in-place to preserve tensor addresses for Metal Trace).
2. The shift register pointer advances.
3. The weighted sum of all 4 slots is computed using the conv1d weights.
4. SiLU is applied to the sum.

```
conv_out[t] = SiLU( Σ_{i=0}^{3}  weight[i] * conv_state[(oldest + i) mod 4] )
```

The in-place write pattern (using `ttnn.copy` into a pre-allocated tensor) is essential for
Metal Trace compatibility: Trace captures a fixed graph with fixed tensor addresses, so
allocating a new tensor on every step would invalidate the captured graph.

### Conv1d and the QK/V Split

The conv1d is applied to the full 8192-dimensional flat QKV vector — Q (2048), K (2048), and V
(4096) channels are all convolved together before the split. This means:

- Q channels at position $t$ are a weighted mixture of Q projections at positions $t-3, t-2,
  t-1, t$.
- K channels at position $t$ are a weighted mixture of K projections at those same positions.
- V channels similarly.

The mixing is applied before the L2 normalization of Q and K. This ordering means the conv1d
can rotate the projection vector (due to the weighted sum) before normalization forces the
result onto the unit sphere. The combined effect is that each token's key, query, and value
see a local window of context before entering the global recurrence.

---

## 4. Summary: Projection Shapes at a Glance

| Projection | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| `in_proj_qkv` | [B, T, 2048] | [B, T, 8192] | Q+K+V concatenated |
| `in_proj_z` | [B, T, 2048] | [B, T, 4096] | Output gate (SiLU in gated RMSNorm) |
| `in_proj_a` | [B, T, 2048] | [B, T, 32] | Decay logit, one per V head |
| `in_proj_b` | [B, T, 2048] | [B, T, 32] | Beta logit, one per V head |
| `conv1d` | [B, 8192, T] | [B, 8192, T] | Depthwise causal, kernel_size=4, then SiLU |
| `out_proj` | [B, T, 4096] | [B, T, 2048] | Post-recurrence projection to hidden dim |

After conv1d and split, Q and K are each expanded from 16 to 32 heads via
`repeat_interleave(2)` before the recurrence loop.

---

**Next:** [`comparison_to_linear_attention_variants.md`](./comparison_to_linear_attention_variants.md)
