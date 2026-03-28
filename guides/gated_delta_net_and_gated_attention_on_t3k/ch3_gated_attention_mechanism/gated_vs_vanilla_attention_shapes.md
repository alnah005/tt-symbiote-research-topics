# Gated Attention vs. Vanilla Attention: Shape Comparison

This section provides a systematic comparison of tensor shapes across attention variants, followed by a detailed analysis of what makes Gated Attention architecturally distinctive.

---

## 1. Shape Comparison Table

The table below covers three variants at two operating points: **prefill** (T > 1, shown as T) and **decode** (T = 1).

Abbreviations used in the table:
- **n_h** = total number of heads (MHA, where n_q = n_kv = n_h)
- **n_q_h** = query heads
- **n_kv_h** = KV heads
- **d_h** = head dimension
- **d_k** = key/query head dim (used in DeltaNet comparison at end of section)

### Vanilla MHA (example: n_h=16, d_h=128, hidden=2048)

| Tensor | Prefill shape | Decode shape |
|--------|--------------|--------------|
| Q (post-proj, pre-norm) | [B, T, 16, 128] | [B, 1, 16, 128] |
| K | [B, T, 16, 128] | [B, 1, 16, 128] |
| V | [B, T, 16, 128] | [B, 1, 16, 128] |
| Gate | — (none) | — (none) |
| Q post-norm | — (no norm) | — (no norm) |
| SDPA output | [B, 16, T, 128] | [B, 16, 1, 128] |
| Reshaped output | [B, T, 2048] | [B, 1, 2048] |

Shape check: $n_h \times d_h = 16 \times 128 = 2048$. Output matches hidden size directly.

### Standard GQA (example: n_q_h=16, n_kv_h=4, d_h=128, hidden=2048)

| Tensor | Prefill shape | Decode shape |
|--------|--------------|--------------|
| Q (post-proj) | [B, T, 16, 128] | [B, 1, 16, 128] |
| K (post-proj) | [B, T, 4, 128] | [B, 1, 4, 128] |
| V (post-proj) | [B, T, 4, 128] | [B, 1, 4, 128] |
| Gate | — (none) | — (none) |
| Q post-norm | — (no norm) | — (no norm) |
| K/V after GQA expand (4×) | [B, T, 16, 128] | [B, 1, 16, 128] |
| SDPA output | [B, 16, T, 128] | [B, 16, 1, 128] |
| Reshaped output | [B, T, 2048] | [B, 1, 2048] |

Shape check: $n_{q\_h} \times d_h = 16 \times 128 = 2048$. Expand factor: $n_{q\_h} / n_{kv\_h} = 16 / 4 = 4\times$.

### Qwen3.5 Gated Attention (n_q_h=16, n_kv_h=2, d_h=256, hidden=2048)

| Tensor | Prefill shape | Decode shape |
|--------|--------------|--------------|
| Q (post-proj, pre-gate) | [B, T, 16, 256] | [B, 1, 16, 256] |
| K (post-proj) | [B, T, 2, 256] | [B, 1, 2, 256] |
| V (post-proj) | [B, T, 2, 256] | [B, 1, 2, 256] |
| Gate (sigmoid) | [B, T, 16, 256] | [B, 1, 16, 256] |
| Q_gated = Q ⊙ gate | [B, T, 16, 256] | [B, 1, 16, 256] |
| Q post-RMSNorm | [B, T, 16, 256] | [B, 1, 16, 256] |
| K post-RMSNorm | [B, T, 2, 256] | [B, 1, 2, 256] |
| K/V after GQA expand (8×) | [B, T, 16, 256] | [B, 1, 16, 256] |
| SDPA output | [B, 16, T, 256] | [B, 16, 1, 256] |
| Reshaped output | [B, T, 4096] | [B, 1, 4096] |
| After o_proj | [B, T, 2048] | [B, 1, 2048] |

Shape check: $n_{q\_h} \times d_h = 16 \times 256 = 4096$. The reshape produces 4096, and o_proj $[4096 \to 2048]$ brings it back to hidden size. Note this differs from MHA/GQA examples above where the reshape directly produces hidden size — Gated Attention needs an explicit down-projection step.

Expand factor: $n_{q\_h} / n_{kv\_h} = 16 / 2 = 8\times$.

---

## 2. Key Differences from Vanilla MHA

### Difference 1: The Gate Tensor

Vanilla MHA has no gate. The Q+gate projection in Gated Attention outputs 8192 features instead of the 4096 a plain Q projection would:

```
Vanilla Q proj weight:       [n_q_h × d_h, H]     = [4096, 2048]
Gated Attention Q+gate proj: [n_q_h × d_h × 2, H] = [8192, 2048]
```

> **Convention note:** Weight shapes throughout this chapter follow the PyTorch `nn.Linear` convention, where the weight matrix is stored as `[out_features, in_features]`. In a forward pass the computation is `x @ weight^T`, so `weight^T` has shape `[in_features, out_features]` and the matrix multiplication is dimensionally consistent.

This doubles the Q-side projection parameter count and produces the extra gate activations `[B, T, 16, 256]` that must be materialized in memory and computed at every forward pass. See [`gated_attention_formulation.md`](./gated_attention_formulation.md) Section 3 for the geometric interpretation.

### Difference 2: Gate-Before-Norm Ordering

```
Standard gate-norm order:  normalize(x) → gate ⊙ normalize(x)
Gated Attention order:     gate ⊙ x     → normalize(gate ⊙ x)
```

See [`gated_attention_formulation.md`](./gated_attention_formulation.md) Section 4 for the gate-before-norm explanation.

### Difference 3: Independent RMSNorm on Both Q and K

Vanilla MHA applies no post-projection normalization to Q or K. Standard GQA likewise typically skips it. Gated Attention normalizes both independently:

```
Vanilla MHA:         Q_proj → Q_rope → SDPA
Gated Attention:     Q_proj → gate ⊙ Q → RMSNorm(Q_gated) → Q_rope → SDPA
                     K_proj → RMSNorm(K_proj) → K_rope → SDPA
```

The per-head weight vectors `q_norm_weight [256]` and `k_norm_weight [256]` are learned and allow the model to independently rescale each dimension after normalization. Normalizing K before SDPA bounds the key vector norms, which stabilizes attention score magnitudes regardless of input scale.

### Difference 4: Extreme GQA (n_kv_h=2)

Common GQA configurations use 4 or 8 KV heads. With n_kv_h=2 and n_q_h=16, Gated Attention uses the most aggressive grouping ratio among widely deployed models:

```
Standard GQA example:  n_q_h=16, n_kv_h=4  →  4× repeat
Gated Attention:       n_q_h=16, n_kv_h=2  →  8× repeat
```

The 8× repeat means only 2 K and V head vectors capture the full context; 16 Q heads then query those 2 representations. This reduces KV cache size by 8× relative to MHA (at the same head dim), but places a strong structural constraint on what the KV representations must encode. See Section 3 for a per-layer memory breakdown.

### Difference 5: rotary_dim=64 << d_h=256

Standard models often rotate all or most of the head dimension. Gated Attention rotates only the first 64 of 256 dimensions:

```
Fraction of d_h rotated: 64 / 256 = 25%
```

The remaining 192 dimensions per head are positionally invariant — they encode only content. This means 75% of the attention computation in the dot product is pure content matching, with position encoded in only one quarter of the similarity score.

---

## 3. Memory Footprint of the Gate Tensor

The gate tensor `[B, T, n_q_h, d_h] = [B, T, 16, 256]` is an intermediate activation that must be allocated for the duration of the forward pass (and the backward pass during training). Its size scales linearly with T:

```
gate_sigmoid elements = B × T × 16 × 256 = B × T × 4,096
gate_sigmoid bytes (BF16) = B × T × 4,096 × 2 = B × T × 8,192
```

**At decode (T=1):**

```
gate_sigmoid = B × 1 × 8,192 bytes = B × 8 KB
```

For B=32: `32 × 8 KB = 256 KB`. Completely negligible.

**At prefill (T=8192, e.g. chunked prefill chunk size):**

```
gate_sigmoid = B × 8,192 × 8,192 bytes = B × 67,108,864 bytes = B × 64 MB
```

For B=1: 64 MB per Gated Attention layer. With 10 Gated Attention layers if all are active simultaneously: 640 MB of gate activations. In practice, chunked prefill limits T to a manageable chunk size, and activations are typically computed and discarded layer by layer — but this is the instantaneous peak for a single layer.

**Summary:**

| T | gate_sigmoid per layer (B=1) |
|---|------------------------------|
| 1 (decode) | 8 KB |
| 512 | 4 MB |
| 2048 | 16 MB |
| 8192 | 64 MB |

---

## 4. Comparison to Gated Delta Net

For completeness, here is how Gated Attention tensor shapes compare to the Gated Delta Net layers from Chapters 1–2.

| Property | Gated Delta Net | Gated Attention |
|----------|----------------|-----------------|
| Layer type | Recurrent (linear attention) | SDPA (quadratic in T) |
| Query/key head dim | d_k = 128 | d_h = 256 |
| Value head dim | d_v = 128 | d_h = 256 |
| Number of query heads | H_v = 32 | n_q_h = 16 |
| Number of KV heads | H_v = 32 (same) | n_kv_h = 2 |
| Total output features | H_v × d_v = 32 × 128 = 4,096 | n_q_h × d_h = 16 × 256 = 4,096 |
| Output projection | [4096 → 2048] | [4096 → 2048] |
| KV cache | None (state is d_k × d_v = 128 × 128 = 16,384 elements per head) | Paged KV cache |
| Memory scaling with T | O(1) per head (fixed state) | O(T) per KV head |

Both architectures produce a 4096-dimensional pre-projection representation that is mapped back to the 2048-dimensional hidden state by a shared output projection structure. Despite their very different operational mechanisms, the interface to the rest of the model is identical in shape.

The complementary design is intentional: Gated Delta Net handles the 30 non-attention layers with O(1) memory per additional token, while the 10 Gated Attention layers provide full SDPA expressivity (with O(T) KV cache) for tasks that require arbitrary token-pair interactions.

---

**Next:** [Chapter 4 — TTNN Primitive Mapping: Decode and Prefill Forward Passes](../ch4_ttnn_primitive_mapping/index.md)
