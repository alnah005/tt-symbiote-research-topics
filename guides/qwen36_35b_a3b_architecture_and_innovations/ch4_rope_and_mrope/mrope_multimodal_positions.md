# M-RoPE: Multimodal Rotary Position Embedding

## What M-RoPE Is

M-RoPE (Multimodal Rotary Position Embedding) is an extension of partial RoPE that assigns **independent position IDs** to the three semantic axes of a vision token — temporal (frame index), spatial height (vertical patch position), and spatial width (horizontal patch position) — while still reducing to standard sequential RoPE for pure text tokens.

M-RoPE is active **only in Gated Attention layers**, the same layers that run partial RoPE. The rotary_dim = 64 and all other parameters from the previous section remain unchanged. The only difference between partial RoPE and M-RoPE is how position IDs are assigned per token; the rotation mathematics are identical.

---

## The mrope_section Split

The 64 rotary dimensions are divided into 32 (real, imaginary) pairs. These 32 pairs are partitioned into three contiguous sections by the `mrope_section` parameter:

```
mrope_section = [11, 11, 10]
```

| Section | Name | Pairs | Dimension range (of 64 rotary dims) |
|---|---|---|---|
| 0 | Temporal | 11 pairs | dims 0–21 |
| 1 | Height (spatial-y) | 11 pairs | dims 22–43 |
| 2 | Width (spatial-x) | 10 pairs | dims 44–63 |

Each section gets its own independent position ID per token. For a token at position m, the three IDs are written:

```
(m_t,  m_h,  m_w)
```

The rotary angle for frequency pair i at the position given by the section that owns pair i is:

```math
\phi_i =
\begin{cases}
m_t \cdot \theta_i & \text{if pair } i \in \text{section 0 (temporal, pairs 0–10)} \\
m_h \cdot \theta_i & \text{if pair } i \in \text{section 1 (height, pairs 11–21)} \\
m_w \cdot \theta_i & \text{if pair } i \in \text{section 2 (width, pairs 22–31)}
\end{cases}
```

where $\theta_i = \text{rope\_theta}^{-2i/\text{rotary\_dim}}$ as defined in the previous section, and the pair-to-section mapping is determined by the cumulative sums of mrope_section: [0, 11), [11, 22), [22, 32).

For **text tokens**, the position IDs are set equal across all three sections:

```
m_t = m_h = m_w = m   (sequential token index)
```

This makes M-RoPE degenerate exactly to standard partial RoPE with a single scalar position m.

---

## Per-Token Position ID Assignment

### Text Tokens

Every text token, whether in a pure-text prompt or interspersed between image tokens, receives three identical position IDs equal to its sequential index in the flattened token sequence:

```
token at sequence position m  →  (m_t=m, m_h=m, m_w=m)
```

The resulting rotary encoding is identical to what would be computed by standard RoPE with position m. No special handling is required in the attention kernel.

### Vision Tokens

A video or image is first encoded by the **vision encoder** (a separate ViT-style model with its own internal position encoding). The output vision features are projected into the text decoder's hidden dimension H = 2048 via a linear MLP. Each projected patch token is then assigned M-RoPE position IDs according to its location in the original media:

```
vision token at (frame f, row r, column c)  →  (m_t=f, m_h=r, m_w=c)
```

For a static image there is a single frame, so m_t = 0 for all vision tokens.

The attention inner product between a query at position $(m_t, m_h, m_w)$ and a key at position $(n_t, n_h, n_w)$ depends only on the **differences** within each section:

```math
Q_m \cdot K_n \;\propto\; \sum_{i \in S_0} f(m_t - n_t, i) \;+\; \sum_{i \in S_1} f(m_h - n_h, i) \;+\; \sum_{i \in S_2} f(m_w - n_w, i)
```

where $f(\Delta, i) = \cos(\Delta \cdot \theta_i)$ is the rotary inner-product kernel. This means attention naturally decays with spatial and temporal distance within each dimension independently, giving the model inductive bias to prefer attending to spatially or temporally nearby patches.

---

## mrope_interleaved = true

The `mrope_interleaved = true` flag controls the **layout** of cos/sin values within the rotary_dim = 64 buffer. It does not change which section a dimension belongs to.

With `mrope_interleaved = true`, the cos and sin buffers for a given token store values in the interleaved (real/imaginary pair) order:

```
[cos(phi_0), cos(phi_0), cos(phi_1), cos(phi_1), ..., cos(phi_31), cos(phi_31)]
  pair 0 real   pair 0 imag   pair 1 real   pair 1 imag
```

rather than the contiguous order:

```
[cos(phi_0), cos(phi_1), ..., cos(phi_31), cos(phi_0), cos(phi_1), ..., cos(phi_31)]
   real halves ...                           imag halves ...
```

The interleaved layout matches the dimension ordering expected by the rotation kernel: element 2i is paired with element 2i+1, so the cos/sin factors must appear at positions 2i and 2i+1 respectively. Most implementations use the interleaved layout internally; `mrope_interleaved = true` in the config confirms this is the canonical form and prevents any transposition during buffer construction.

---

## M-RoPE and the Vision Encoder

The vision encoder (the ViT component) has its own internal 2D position encoding applied to patch embeddings. This encoding is **independent** of M-RoPE and operates entirely within the vision encoder's attention layers. After the vision encoder processes the patches and the MLP projects the outputs into the text decoder's embedding space, the patch token positions are re-encoded via M-RoPE within each Gated Attention layer of the text decoder.

There is therefore **no coupling** between the vision encoder's position encoding and M-RoPE: the vision encoder's positional signals are absorbed into the projected features, and M-RoPE assigns fresh geometric position IDs to those features within the decoder's attention mechanism.

---

## Text-Only Inference: No M-RoPE Changes Needed

For text-only inference (no image or video tokens in the prompt), the position ID tensor has shape [B, 3, T] where all three rows are identical and equal to the sequential token positions:

```
position_ids[b, 0, :] = [0, 1, 2, ..., T-1]   # temporal
position_ids[b, 1, :] = [0, 1, 2, ..., T-1]   # height
position_ids[b, 2, :] = [0, 1, 2, ..., T-1]   # width
```

Because all three rows are identical, gathering from the cos/sin cache for each section and concatenating produces exactly the same result as computing cos/sin for a single sequential position vector. The rotary kernel sees no difference from standard partial RoPE.

In the TTNN deployment for text-only inference, the simplest correct implementation is to:

1. Ignore the three-section structure entirely.
2. Pass a single position ID vector [B, T] and compute cos/sin with shape [T, 64] as in standard partial RoPE.
3. Apply the rotation to the first 64 dims of each Q and K head as described in the previous section.

This is valid because `m_t = m_h = m_w = m` for all text tokens, making the three-section structure a mathematical identity that produces the same rotation as using the scalar position m throughout.

---

## TTNN Deployment Summary

| Inference mode | M-RoPE handling | Code delta vs Qwen3.5 |
|---|---|---|
| Text-only | Treat as standard partial RoPE | None |
| Text + static image | Build [B, 3, T] position_ids with frame=0 for patches | New position_ids builder |
| Text + video | Build [B, 3, T] position_ids with frame indices | New position_ids builder |

The cos/sin computation kernel itself does not change across modes; only the position ID input changes. The mrope_section = [11, 11, 10] split is only needed when building the per-token position ID tensor for vision inputs.

---

**Next:** [Chapter 5 — Multi-Token Prediction](../ch5_multi_token_prediction/index.md)
