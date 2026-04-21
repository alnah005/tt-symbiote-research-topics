# Delta Rule Formulation

This file derives the complete Gated DeltaNet recurrence, explains every term, and documents
all scalar parameters and state dimensions for Qwen3.6-35B-A3B.

---

## 1. Motivation

Standard linear attention replaces the $O(T^2)$ softmax attention matrix with a fixed-size
recurrent state $S \in \mathbb{R}^{d_k \times d_v}$, giving $O(1)$ decode cost. The limitation
is that the state accumulates all past writes without any forgetting: every token leaves a
residual in $S$ that never decays, so the state gradually becomes saturated and retrieval
quality degrades over long sequences.

Two independent mechanisms address different aspects of this limitation:

- **GLA-style scalar gating**: multiply the entire state by a data-dependent scalar $g_t \in
  (0, 1)$ at each step, causing all past associations to decay uniformly. This provides coarse
  global forgetting but does not allow targeted correction of individual associations.
- **The delta rule**: rather than writing a new value unconditionally, measure the prediction
  error between what the state currently associates with key $\tilde{k}_t$ and the target value
  $v_t$, then write only the correction. This precisely overwrites stale memories at a specific
  key direction without disturbing orthogonal associations. However, without any global decay,
  irrelevant old context is never flushed.

**Gated DeltaNet** combines both. The result is a state that (a) decays globally over time so
old context fades, and (b) corrects individual associations toward current targets rather than
blindly overwriting them.

---

## 2. Core Recurrence

For a single head $h$ at sequence step $t$:

$$g_t = \exp(\alpha_t), \qquad \alpha_t < 0 \;\Rightarrow\; g_t \in (0, 1)$$

$$\beta_t = \sigma(b_t), \qquad \beta_t \in (0, 1)$$

$$S_t = g_t \cdot S_{t-1} + \tilde{k}_t \Bigl(\beta_t \cdot \bigl(v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t\bigr)\Bigr)^\top$$

$$o_t = S_t^\top \!\left(\frac{\tilde{q}_t}{\sqrt{d_k}}\right)$$

**Symbols:**

| Symbol | Shape | Description |
|--------|-------|-------------|
| $S_t$ | $[d_k, d_v] = [128, 128]$ | Recurrent state matrix at step $t$ |
| $\tilde{k}_t$ | $[d_k] = [128]$ | L2-normalized key vector |
| $\tilde{q}_t$ | $[d_k] = [128]$ | L2-normalized query vector; divided by $\sqrt{d_k}$ in the output formula above |
| $v_t$ | $[d_v] = [128]$ | Value vector |
| $g_t$ | scalar $\in (0, 1)$ | Decay gate |
| $\beta_t$ | scalar $\in (0, 1)$ | Delta update rate |
| $o_t$ | $[d_v] = [128]$ | Per-head output vector |

**Dimensional consistency.** Every term of the state update is in $\mathbb{R}^{d_k \times d_v}$:

- $g_t \cdot S_{t-1} \in \mathbb{R}^{d_k \times d_v}$ (scalar times matrix).
- $g_t \cdot S_{t-1}^\top \tilde{k}_t \in \mathbb{R}^{d_v}$ (matvec retrieval).
- $\beta_t \cdot (v_t - \cdots) \in \mathbb{R}^{d_v}$ (scalar times vector).
- $\tilde{k}_t \cdot (\cdots)^\top \in \mathbb{R}^{d_k \times d_v}$ (rank-1 outer product).
- Output: $S_t^\top (\tilde{q}_t / \sqrt{d_k}) \in \mathbb{R}^{d_v}$ (matvec retrieval with scaled query).

---

## 3. Term-by-Term Interpretation

### 3.1 Decay: $g_t \cdot S_{t-1}$

The scalar $g_t \in (0, 1)$ multiplies every entry of the $d_k \times d_v$ state matrix uniformly.
When $g_t$ is close to 0, the state is nearly erased — all past associations fade. When $g_t$
is close to 1, the state is preserved with minimal forgetting. This is the coarse forgetting
mechanism: a single data-dependent scalar controls how much of the model's entire memory is
retained at each step.

### 3.2 Retrieval Under Decayed State: $g_t \cdot S_{t-1}^\top \tilde{k}_t$

After decaying, the model reads back what the decayed state associates with the current key
$\tilde{k}_t$. The result is a $d_v$-dimensional predicted value: the model's current best
estimate of what should be stored at key $\tilde{k}_t$.

### 3.3 Delta Correction: $\beta_t \cdot (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)$

The prediction error is $(v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t) \in \mathbb{R}^{d_v}$. The
scalar $\beta_t = \sigma(b_t) \in (0, 1)$ controls how aggressively the state corrects toward
the true value. When $\beta_t \approx 1$, the full correction is applied. When $\beta_t \approx
0$, the state is nearly read-only at this step.

This is the DeltaNet mechanism: the write is not $v_t$ itself but only the residual error
between $v_t$ and the current prediction. Repeated writes to the same key direction converge
rather than accumulate, so the state does not grow unboundedly dense at any one direction.

### 3.4 Rank-1 Write: $\tilde{k}_t (\text{correction})^\top$

The delta correction vector (shape $[d_v]$) is written into the state by forming an outer
product with the key vector $\tilde{k}_t$ (shape $[d_k]$). The resulting rank-1 matrix (shape
$[d_k, d_v]$) is added to the decayed state. The key acts as the address: the update is
concentrated at directions in state-space aligned with $\tilde{k}_t$. Future queries $\tilde{q}_t$
whose direction is similar to $\tilde{k}_t$ will retrieve the corrected association; orthogonal
queries are unaffected.

### 3.5 Output Query: $o_t = S_t^\top (\tilde{q}_t / \sqrt{d_k})$

After the state update, the output is a linear read from the updated state using the
L2-normalized and $1/\sqrt{d_k}$-scaled query $\tilde{q}_t / \sqrt{d_k}$. This produces a
$d_v$-dimensional output vector that represents the state's association with the current query.
The $1/\sqrt{d_k}$ factor stabilizes the magnitude of $o_t$ as the state $S_t$ accumulates
rank-1 outer products over a long sequence (see Section 5).

---

## 4. Decay Gate Derivation

The decay gate $g_t = \exp(\alpha_t)$ requires $\alpha_t < 0$ to keep $g_t \in (0, 1)$. In
Qwen3.6-35B-A3B this is computed as:

$$\alpha_t = -\exp(A_{\log}) \cdot \text{softplus}(a_t + \text{dt\_bias})$$

$$g_t = \exp(\alpha_t)$$

**Parameters involved:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `A_log` | $[H_v] = [32]$ per-layer | Learned log-space decay rate; $\exp(A_{\log}) > 0$ always |
| $a_t$ | $[B, T, H_v] = [B, T, 32]$ | Per-token per-head decay input from `in_proj_a` |
| `dt_bias` | $[H_v] = [32]$ per-layer | Learned per-head bias on the decay input |

**Sign analysis (why $\alpha_t < 0$ is guaranteed):**

1. $\exp(A_{\log}) > 0$ always — the exponential of any real number is strictly positive.
2. $\text{softplus}(a_t + \text{dt\_bias}) > 0$ always — softplus maps all reals to $(0, \infty)$.
3. Their product is strictly positive.
4. The leading $-$ sign negates the product, giving $\alpha_t < 0$ for any finite input.
5. Therefore $g_t = \exp(\alpha_t) \in (0, 1)$ unconditionally.

The $\text{softplus}$ nonlinearity $\log(1 + e^x)$ ensures smooth positive sensitivity to the
input $a_t$. The parameter `A_log` sets a per-head baseline decay rate. The `dt_bias` allows
per-head offset tuning without requiring large activations from the projection. In the TTNN
implementation, $-\exp(A_{\log})$ is precomputed and stored on device at construction time
(shape [1, $H_v$, 1, 1]) to avoid recomputing it at every token step.

---

## 5. L2 Normalization of Q and K

Before entering the recurrence, both the query and key vectors are L2-normalized per head:

$$\tilde{q}_t = \frac{q_t}{\sqrt{\|q_t\|_2^2 + \epsilon}}, \qquad
  \tilde{k}_t = \frac{k_t}{\sqrt{\|k_t\|_2^2 + \epsilon}}$$

with $\epsilon = 10^{-6}$.

This normalization is essential for numerical stability. Without it, the outer product
$\tilde{k}_t (\text{correction})^\top$ can have unbounded magnitude, causing the state matrix
$S$ to grow without bound as the sequence lengthens. After L2 normalization, $\|\tilde{k}_t\|_2
= 1$, so the magnitude of each rank-1 write is bounded by $\beta_t \cdot \|v_t - \text{prediction}\|_2$,
which depends only on the current value error rather than the accumulated scale of the keys.

After L2 normalization, the query is also scaled by $1 / \sqrt{d_k}$:

$$\tilde{q}_{\text{scaled}} = \tilde{q}_t \cdot \frac{1}{\sqrt{d_k}} = \tilde{q}_t \cdot \frac{1}{\sqrt{128}}$$

This scaling controls the magnitude of state reads. The state $S_t$ accumulates $O(T)$
rank-1 outer products over a sequence of length $T$, so without scaling the output
$o_t = S_t^\top \tilde{q}_t$ would grow in magnitude with sequence length. The $1/\sqrt{d_k}$
factor stabilizes the scale of $o_t$ irrespective of sequence length. Note that the saturation
argument used in standard dot-product attention does not apply here because both $\tilde{q}_t$
and $\tilde{k}_t$ are already L2-normalized, so their dot product is bounded in $[-1, 1]$ by
Cauchy-Schwarz regardless of $d_k$; the scaling is instead needed to counteract the growing
magnitude of $S_t$ itself.

---

## 6. Gated RMSNorm Output

After the recurrence produces output $o_t$ (shape $[H_v, d_v] = [32, 128]$ across all heads),
it passes through a gated RMSNorm before the output projection:

$$\text{normed}_t = o_t \cdot \left(\text{mean}_{d_v}(o_t^2) + \epsilon\right)^{-1/2} \cdot w_{\text{norm}}$$

Here $\text{mean}_{d_v}$ denotes the mean taken over the last dimension — i.e., over the $d_v = 128$
per-head elements for each head independently. The result is a per-head scalar used to
normalize that head's output vector.

$$\text{output}_t = \text{normed}_t \cdot \text{SiLU}(z_t)$$

where:

- $w_{\text{norm}}$ is the per-dimension learned scale from `norm.weight` (shape $[d_v] = [128]$).
- $z_t$ is the output of `in_proj_z` (shape $[B, T, H_v \cdot d_v] = [B, T, 4096]$, reshaped
  to $[B, T, H_v, d_v]$ before the gate).
- $\text{SiLU}(x) = x \cdot \sigma(x)$ is the sigmoid-weighted linear unit.

The SiLU gate allows each head and dimension to selectively suppress or amplify the normalized
output. This is analogous to the gating in SwiGLU FFN blocks. The gate $z_t$ is a learned linear
function of the input hidden state, so the model can vary the output gain dynamically based on
context.

In the fused `ttnn.experimental.gated_delta_net` kernel, this entire post-recurrence block
(RMSNorm + SiLU gate) is computed inside the kernel using the `norm_w` and `z_flat` inputs,
avoiding a separate host dispatch.

---

## 7. State Matrix Dimensions and Memory

### Per-head state

```
S ∈ R^{d_k × d_v} = R^{128 × 128}
Elements:           128 × 128 = 16,384
Float32 bytes:      16,384 × 4 = 65,536 bytes = 64 KB
BF16 bytes:         16,384 × 2 = 32,768 bytes = 32 KB
```

### Per-layer state (all heads, batch B=1)

```
Full state shape:   [B, H_v, d_k, d_v] = [1, 32, 128, 128]
Elements (B=1):     32 × 128 × 128 = 524,288
Float32 bytes:      524,288 × 4 = 2,097,152 bytes ≈ 2 MB
BF16 bytes:         524,288 × 2 = 1,048,576 bytes ≈ 1 MB
```

### Why float32?

The `mamba_ssm_dtype: "float32"` configuration means the state $S$ is kept in fp32. This is
required for numerical stability: the state accumulates outer products from every token in the
sequence. BF16 has only 7 mantissa bits; accumulated rounding error becomes visible after
10–20 tokens per layer and produces garbage output past 30+ layers at sequence lengths of
interest (>1K tokens). Float32 provides 23 mantissa bits, sufficient for thousands of
accumulated rank-1 updates.

The activations flowing into and out of the state (Q, K, V projections) can be BF16; only the
state itself requires float32 precision.

### Comparison to KV cache

For a Gated Attention layer with $n_q = 16$ query heads, $n_{kv} = 2$ KV heads, and head
dimension $d_h = 256$:

```
KV cache per layer at sequence length T (BF16):
  2 × n_kv × d_h × T × 2 bytes = 2 × 2 × 256 × T × 2 = 2048 × T bytes
  At T = 1,024:   2 MB
  At T = 16,384:  32 MB
  At T = 131,072: 256 MB

Gated DeltaNet recurrent state per layer (FP32, B=1):
  Always ≈ 2 MB, regardless of T.
```

The recurrent state has constant memory cost. This is the fundamental advantage of linear
attention for long-context inference: 30 Gated DeltaNet layers × 2 MB ≈ 60 MB total state,
compared to a KV cache that would reach 60 MB per Gated Attention layer at T ≈ 30K tokens.

---

**Next:** [`head_asymmetry_and_projections.md`](./head_asymmetry_and_projections.md)
