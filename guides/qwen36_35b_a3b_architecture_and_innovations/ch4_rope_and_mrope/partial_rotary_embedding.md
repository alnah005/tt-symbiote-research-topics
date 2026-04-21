# Partial Rotary Embedding in Gated Attention Layers

## Scope: Gated Attention Layers Only

Partial RoPE is applied **exclusively inside Gated Attention layers** (every 4th layer, indices 0, 4, 8, …). Gated DeltaNet layers — which constitute the other three of every four layers — apply L2 normalization to their Q and K projections and carry **no positional encoding whatsoever**. This document covers only the Gated Attention path.

---

## Dimensions and Parameters

The Gated Attention layer operates with:

- Hidden dimension H = 2048
- Query heads n_q = 16, KV heads n_kv = 2 (GQA ratio 8:1)
- Head dimension d_h = 256 (explicit config field; Q projection outputs 16 × 256 = 4096 dims, not H=2048)
- `partial_rotary_factor` = 0.25
- **rotary_dim** = d_h × partial_rotary_factor = 256 × 0.25 = **64**
- `rope_theta` = 10,000,000
- Maximum context length = 262,144 tokens

The cos/sin cache is precomputed with shape [max_seq_len, rotary_dim] = [262144, 64], **not** [262144, 256]. Only these 64 values are ever read per token per head.

---

## Head Vector Decomposition

Every Q and K head vector of dimension 256 is split into two contiguous subvectors before RoPE is applied:

```
h = [h_rot | h_pass]
     ←64→   ←192→
```

- `h_rot` (dims 0–63): receives rotary encoding
- `h_pass` (dims 64–255): passed through unchanged

The output after partial RoPE is:

```math
\text{RoPE}_{\text{partial}}(h, m) = \bigl[\,\text{RoPE}(h_{\text{rot}},\, m),\;\; h_{\text{pass}}\,\bigr]
```

where m is the absolute position of the token and `RoPE(·, m)` is the standard rotary operation on a 64-dimensional vector.

For the standard 2D rotary operation on a vector pair $(x_{2i}, x_{2i+1})$ at position m:

```math
\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix}
=
\begin{bmatrix} \cos(m\,\theta_i) & -\sin(m\,\theta_i) \\ \sin(m\,\theta_i) & \cos(m\,\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
```

which is equivalently written using complex multiplication:

```math
(x_{2i} + i\,x_{2i+1})\,e^{i\,m\,\theta_i}
```

---

## Frequency Spectrum

With rotary_dim = 64 there are 32 frequency pairs, indexed $i = 0, 1, \ldots, 31$. The per-pair base frequency is:

```math
\theta_i = \text{rope\_theta}^{-2i/\text{rotary\_dim}} = 10^{7\,\times\,(-2i/64)}
```

The extremes of the spectrum:

| Pair index i | Formula | Approximate value |
|---|---|---|
| 0 (highest frequency pair) | $10^{7 \times 0}$ | 1.0 (no decay) |
| 1 | $10^{7 \times (-2/64)}$ | ≈ 0.604 |
| 15 | $10^{7 \times (-30/64)}$ | ≈ 5.24 × 10^{-4} |
| 31 (lowest frequency pair) | $10^{7 \times (-62/64)}$ | ≈ 1.65 × 10^{-7} |

Compared to a standard RoPE with rope_theta = 10,000 and rotary_dim = 128 (a common baseline), Qwen3.6's combination of a 10M base with only 64 rotary dims produces a **much more gradual frequency decay**. The fastest pair (i=0) has period 2π positions — identical in all configurations — but the slowest pair (i=31) has a period of roughly 40 million positions, far exceeding the 262K context window. This means even the lowest-frequency rotary pair completes fewer than one full cycle across the maximum context, preventing the aliasing that occurs when low-frequency components wrap around.

### Why Not Full RoPE on 256 Dims?

With full RoPE on d_h = 256, there would be 128 frequency pairs. Pairs at low indices (near i=0) have the highest base frequencies (shortest periods), introducing rapid oscillations in the attention logits. At long distances (large |m - n|), these high-frequency components cause the inner product $Q_m \cdot K_n$ to oscillate rapidly and effectively average to zero — which degrades retrieval of distant tokens. By restricting RoPE to the first 64 dimensions, Qwen3.6:

1. Keeps 192 dimensions position-agnostic, preserving content similarity across all distances.
2. Concentrates all positional signal in a dedicated low-dimensional subspace.
3. Uses a high theta (10M) to further slow the fastest frequencies even within the 64-dim subspace.

---

## Q/K RMSNorm Before RoPE

In Qwen3.6 Gated Attention layers, both Q and K tensors are independently RMS-normalized **before** rotary encoding is applied. This is separate from the layer-level input normalization and from the L2 normalization used in Gated DeltaNet.

The operation on a single head vector h is:

```math
\text{RMSNorm}(h) = \frac{h}{\sqrt{\frac{1}{d_h}\sum_{j=0}^{d_h-1} h_j^2 + \epsilon}} \;\odot\; \gamma
```

where $\gamma \in \mathbb{R}^{d_h}$ is a learned per-head scale and $\epsilon$ is a small constant for numerical stability.

The full per-head sequence in a Gated Attention layer is therefore:

```
Q_proj  →  RMSNorm(Q)  →  partial RoPE  →  attention scores
K_proj  →  RMSNorm(K)  →  partial RoPE  →  attention scores
V_proj                                  →  weighted sum
```

The RMSNorm stabilizes attention logit scale, which is important given the very large head_dim (256). Without it, the $\sqrt{d_h}$ denominator in scaled dot-product attention would need to cancel a wide dynamic range in the raw projections.

**Distinction from Gated DeltaNet:** In Gated DeltaNet layers, Q and K are L2-normalized (unit norm enforced, no learned scale) and **no positional encoding follows**. The two normalizations serve different purposes and must not be confused.

---

## cos/sin Cache Shape and Indexing

The implementation precomputes:

```
cos_cache: [max_seq_len, rotary_dim]  =  [262144, 64]
sin_cache: [max_seq_len, rotary_dim]  =  [262144, 64]
```

At inference for a batch of shape [B, T, n_q, d_h]:

1. Gather rows from cos_cache and sin_cache at the T active position indices → shape [T, 64].
2. Broadcast to [B, T, n_q or n_kv, 64].
3. Apply rotation to the first 64 dims of each head.
4. Concatenate with the remaining 192 dims (no-op pass-through).

The key implementation detail: the cos/sin buffers are indexed and stored at rotary_dim = 64, not at head_dim = 256. Any code that mistakenly allocates [max_seq_len, 256] cosine buffers would waste memory and produce incorrect results.

---

## TTNN Deployment

The existing Qwen3.5 implementation in the TTNN stack already handles partial rotary embedding via the `partial_rotary_factor` parameter. Qwen3.6 uses the same factor (0.25) with the same head_dim (256), which means:

- No changes are required to the rotary embedding kernel.
- No changes are required to the cos/sin precomputation pipeline.
- The only delta versus Qwen3.5 is the increased `rope_theta` (10M vs the Qwen3.5 value), which is a scalar parameter passed at initialization — no structural code change.

Verification checklist for a Qwen3.6 TTNN port:
- [ ] `rope_theta` configuration reads `10_000_000` (not a Qwen3.5 default).
- [ ] cos/sin cache is allocated with last dim = 64 (not 256).
- [ ] RMSNorm is applied to Q and K before the rotary kernel is invoked.
- [ ] Gated DeltaNet layers do **not** invoke any rotary kernel.

---

**Next:** [`mrope_multimodal_positions.md`](./mrope_multimodal_positions.md)
