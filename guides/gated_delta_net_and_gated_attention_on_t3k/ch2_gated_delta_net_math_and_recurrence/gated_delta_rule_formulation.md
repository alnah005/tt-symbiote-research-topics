# Gated Delta Rule: Mathematical Formulation

## 1. Motivation

Chapter 1 established two complementary shortcomings in the existing linear attention family:

- **GLA** introduces a data-dependent, per-column forgetting gate that allows the state to selectively discard old information. However the write is always a direct outer product `v_t k̃_t^T` — the model writes whatever value it sees into the state, regardless of what is already stored there. There is no mechanism to correct prediction errors.
- **DeltaNet** applies the delta rule: it measures the prediction error between what the state currently associates with key $\tilde{k}_t$ and the new target value $v_t$, then writes only the correction. This gives precise, targeted writes. But standard DeltaNet sets no coarse decay gate, so the state accumulates all past writes without global forgetting.

**Gated Delta Net** combines both mechanisms: scalar coarse forgetting from GLA and error-correcting writes from DeltaNet. The result is a state that (a) decays uniformly over time so old associations fade, and (b) corrects itself toward new associations rather than blindly overwriting them.

---

## 2. Core Recurrence

For a single head at sequence step `t`, let:

| Symbol | Shape | Description |
|--------|-------|-------------|
| `S_t` | `[d_k, d_v]` | Recurrent state matrix at step t |
| $\tilde{k}_t$ | `[d_k]` | L2-normalized key vector |
| $\tilde{q}_t$ | `[d_k]` | L2-normalized query vector |
| `v_t` | `[d_v]` | Value vector |
| $g_t$ | scalar $\in (0, 1)$ | Scalar decay gate |
| $\beta_t$ | scalar $\in (0, 1)$ | Update rate (delta step size) |
| `o_t` | `[d_v]` | Output vector for this head |

The recurrence is:

$$g_t = \exp(\alpha_t) \qquad \alpha_t < 0 \;\Rightarrow\; g_t \in (0, 1)$$

$$\beta_t = \sigma(b_t) \qquad \beta_t \in (0, 1)$$

$$S_t = g_t \cdot S_{t-1} + \tilde{k}_t \bigl(\beta_t \cdot (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)\bigr)^\top$$

$$o_t = S_t^\top \tilde{q}_t$$

### Dimensional consistency check

All terms are verified $\in \mathbb{R}^{d_k \times d_v}$: $g_t \cdot S_{t-1} \in \mathbb{R}^{d_k \times d_v}$, the outer product $\tilde{k}_t \cdot (\text{correction})^\top \in \mathbb{R}^{d_k \times d_v}$, and the readout $S_t^\top \tilde{q}_t \in \mathbb{R}^{d_v}$.

---

## 3. Interpretation of Each Term

### 3.1 $g_t \cdot S_{t-1}$ — Coarse Forgetting

The entire state is multiplied by the scalar $g_t \in (0, 1)$. Every entry of the `d_k × d_v` state matrix decays by the same factor. When $g_t$ is close to 0 the state is nearly wiped; when $g_t$ is close to 1 forgetting is minimal. This is the GLA-style mechanism: a single input-dependent scalar controls global memory retention at this step.

### 3.2 $g_t \cdot S_{t-1}^\top \tilde{k}_t$ — Predicted Value Under Decayed State

After decaying, the model retrieves what the decayed state associates with the current key: it forms the matrix-vector product $(g_t \cdot S_{t-1})^\top \tilde{k}_t \in \mathbb{R}^{d_v}$. This is the model's current best prediction of $v_t$ given key $\tilde{k}_t$.

### 3.3 $\beta_t \cdot (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)$ — Delta Correction

The prediction error is $v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t \in \mathbb{R}^{d_v}$. The update rate $\beta_t = \sigma(b_t) \in (0, 1)$ scales how aggressively the state corrects toward the true value. When $\beta_t \approx 1$ the correction is large; when $\beta_t \approx 0$ the existing state is trusted.

This term is the DeltaNet mechanism: instead of writing $v_t$ unconditionally, only the error residual is written back.

### 3.4 Outer Product Write $\tilde{k}_t (\text{correction})^\top$

The correction vector is written into the state by taking the outer product with key $\tilde{k}_t$. The key acts as the addressing vector: the update is concentrated at directions in state-space aligned with $\tilde{k}_t$. This is an associative memory update — future queries with keys similar to $\tilde{k}_t$ will retrieve the corrected value.

### 3.5 $o_t = S_t^\top \tilde{q}_t$ — Output Retrieval

The output is a linear read from the updated state using the L2-normalized query $\tilde{q}_t$. This is identical to the standard linear attention output step.

---

## 4. Comparison to Standard DeltaNet

Standard DeltaNet (without gating) is the special case $g_t = 1$ at every step:

$$S_t = S_{t-1} + \tilde{k}_t \bigl(\beta_t \cdot (v_t - S_{t-1}^\top \tilde{k}_t)\bigr)^\top \qquad [\text{DeltaNet},\; g_t = 1]$$

There is no coarse decay — the state accumulates all past updates without forgetting. Gated Delta Net adds the $g_t$ multiplier to both the state carry and the prediction, enabling selective forgetting on a per-step basis.

---

## 5. Comparison to GLA

GLA (Gated Linear Attention) uses a row-wise outer-product gate $G_t = \alpha_t \mathbf{1}^\top$ where $\alpha_t \in \mathbb{R}^{d_k}$ is a data-dependent column vector from the input projection and $\mathbf{1} \in \mathbb{R}^{d_v}$ is the all-ones vector, giving $G_t \in \mathbb{R}^{d_k \times d_v}$. The operation $G_t \odot S_{t-1}$ scales row i of S by $\alpha_t[i]$ — row-wise (per key-dimension) decay. GLA writes the value directly:

$$S_t = (\alpha_t \mathbf{1}^\top) \odot S_{t-1} + \tilde{k}_t v_t^\top \qquad [\text{GLA}]$$

Key differences from Gated Delta Net:

1. **Gating scope**: GLA applies a row-wise (per key-dimension) gate — row i of S scaled by $\alpha_t[i]$; Gated Delta Net uses a single scalar $g_t$ applied uniformly to the entire state.
2. **Write mechanism**: GLA writes $\tilde{k}_t v_t^\top$ unconditionally; Gated Delta Net writes only the error correction $\beta_t (v_t - g_t S_{t-1}^\top \tilde{k}_t) \tilde{k}_t^\top$.
3. **Error correction**: GLA has none; Gated Delta Net corrects toward the target value.

---

## 6. Decay Gate Derivation (Qwen3.5-35B-A3B)

The scalar decay $g_t = \exp(\alpha_t)$ requires $\alpha_t < 0$ to keep $g_t \in (0, 1)$. In Qwen3.5-35B-A3B this is achieved as follows:

$$\alpha_t = -\exp(A_{\log}) \cdot \text{softplus}(a_t + \text{dt bias})$$

$$g_t = \exp(\alpha_t)$$

Where:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `A_log` | `[H_v]` per-head scalar | Learned log-space parameter; `exp(A_log) > 0` always |
| `a_t` | `[B, T, 4]` | Input-dependent signal from `in_proj_a` (4 groups for 35B-A3B; see §7) |
| `dt_bias` | `[H_v]` per-head scalar | Learned per-head bias |

Step-by-step sign analysis:

1. `exp(A_log) > 0` always (exponent of a real).
2. `softplus(a_t + dt_bias) > 0` always (softplus is strictly positive).
3. Their product is strictly positive, negated by the leading `−`: so `α_t < 0` always.
4. Therefore `g_t = exp(α_t) ∈ (0, 1)` for any finite input. At `α_t = 0` (which cannot be reached), `g_t = 1`.

The `softplus` nonlinearity ensures smooth, positive sensitivity to the input `a_t`; `A_log` sets the per-head baseline decay rate; `dt_bias` allows per-head offset tuning without requiring large input activations.

---

## 7. Full Projection Inventory — One Gated Delta Net Layer (Qwen3.5-35B-A3B)

Model constants for this section:

| Symbol | Value | Note |
|--------|-------|------|
| H | 2048 | Hidden dimension |
| key_dim | 2048 | H_k × d_k = 16 × 128 |
| value_dim | 4096 | H_v × d_v = 32 × 128 |
| H_k | 16 | Number of key/query heads |
| H_v | 32 | Number of value heads |
| d_k | 128 | Key/query head dimension |
| d_v | 128 | Value head dimension |
| num_v_groups | 4 | Decay/beta grouping for 35B-A3B (one scalar per group of 8 v-heads) |

### 7.1 Combined QKV projection

```
in_proj_qkv:  [B, T, H]  →  [B, T, key_dim×2 + value_dim]
                          =  [B, T, 2×2048 + 4096]
                          =  [B, T, 8192]
```

Split along last dimension:

```
Q  [B, T, key_dim]   = [B, T, 2048]
K  [B, T, key_dim]   = [B, T, 2048]
V  [B, T, value_dim] = [B, T, 4096]
```

Reshape into heads:

```
Q  [B, T, H_k, d_k] = [B, T, 16, 128]
K  [B, T, H_k, d_k] = [B, T, 16, 128]
V  [B, T, H_v, d_v] = [B, T, 32, 128]
```

Repeat Q and K by factor 2 along the head dimension to match V heads (each K/Q head is shared across 2 V heads):

```
Q̃  [B, T, H_v, d_k] = [B, T, 32, 128]    (repeat_interleave × 2)
K̃  [B, T, H_v, d_k] = [B, T, 32, 128]    (repeat_interleave × 2)
```

After this repeat, all 32 heads share a common state shape `[d_k, d_v] = [128, 128]`.

### 7.2 Output gate projection

```
in_proj_z:  [B, T, H]  →  [B, T, value_dim]
                        =  [B, T, 4096]
```

`Z` is used in the post-attention gated RMSNorm: `output = RMSNorm(o) ⊙ sigmoid(Z)`.

### 7.3 Log-decay input projection

```
in_proj_a:  [B, T, H]  →  [B, T, num_v_groups]
                        =  [B, T, 4]
```

The 35B-A3B configuration uses `num_v_groups = 4` (i.e., one decay scalar per group of 8 V-heads). Each of the 4 scalars is broadcast to its 8 V-heads before computing $\alpha_t$.

Note: the 9B configuration uses `num_v_groups = 32` (one decay scalar per V-head, no grouping). Throughout this guide we use the 35B-A3B values.

### 7.4 Beta logit projection

```
in_proj_b:  [B, T, H]  →  [B, T, num_v_groups]
                        =  [B, T, 4]
```

Same grouping as `in_proj_a`. Each scalar $b_t$ passes through $\sigma(\cdot)$ to produce $\beta_t$.

### 7.5 Causal conv1d

Applied to the concatenated QKV activations before the split, to inject local context:

```
conv1d input:   [B, key_dim×2 + value_dim, T] = [B, 8192, T]
kernel size:    4
output:         [B, 8192, T]   (same shape, causal, depthwise)
```

During decode (T=1) this operates as a state update over the stored `conv_state [B, 8192, 4]`.

### 7.6 Output projection

```
out_proj:  [B, T, value_dim]  →  [B, T, H]
                              =  [B, T, 4096]  →  [B, T, 2048]
```

---

## 8. State Matrix Dimensions and Memory

### Per-head state

```
S ∈ R^{d_k × d_v} = R^{128 × 128}
Elements: 128 × 128 = 16,384
BF16 bytes: 16,384 × 2 = 32,768 bytes = 32 KB
```

### Per-layer state (all heads, batch B)

```
Full state shape: [B, H_v, d_k, d_v] = [B, 32, 128, 128]

Elements per batch element: 32 × 128 × 128 = 524,288
BF16 bytes per batch element:
  524,288 × 2 = 1,048,576 bytes = 1,024 KB ≈ 1 MB

Total BF16 bytes: B × 1,048,576 ≈ B × 1 MB per layer
```

This memory is **independent of sequence length T** — a fixed-size recurrent state that does not grow as the context extends.

---

**Next:** [`parallelism_and_scan.md`](./parallelism_and_scan.md)
