# Gated Attention: Forward Pass Derivation

This section walks through every step of a `Qwen3_5MoeGatedAttention` / `TTNNQwen3FullAttention` forward pass. All shape arithmetic is shown explicitly. Symbols follow the conventions in [`index.md`](./index.md).

---

## 1. Input

```
hidden_states: [B, T, H] = [B, T, 2048]
```

`hidden_states` arrives from the preceding layer norm. `B` is batch size, `T` is the current sequence length (T > 1 for prefill, T = 1 for decode), and H = 2048 is the model hidden size.

---

## 2. Input Projections

### 2.1 Q + Gate Projection

A single linear layer projects hidden states into both the query vectors and the gating vectors simultaneously:

```
q_gate_proj weight: [n_q_h × d_h × 2, H] = [16 × 256 × 2, 2048] = [8192, 2048]

hidden_states [B, T, 2048] × weight^T [2048, 8192]
  → q_gate_raw: [B, T, 8192]
```

The projection is then split along the last axis at position 4096:

```
q_gate_raw [B, T, 8192]
  split at dim=-1, size 4096 each
  → q_raw:    [B, T, 4096]   (first half)
  → gate_raw: [B, T, 4096]   (second half)
```

Reshape to head layout:

```
q_raw    [B, T, 4096] → reshape → Q_raw:    [B, T, n_q_h, d_h] = [B, T, 16, 256]
gate_raw [B, T, 4096] → reshape → gate_raw: [B, T, n_q_h, d_h] = [B, T, 16, 256]
```

### 2.2 KV Projection

A separate linear layer produces K and V together:

```
kv_proj weight: [n_kv_h × d_h × 2, H] = [2 × 256 × 2, 2048] = [1024, 2048]

hidden_states [B, T, 2048] × weight^T [2048, 1024]
  → kv_raw: [B, T, 1024]
```

Split at position 512:

```
kv_raw [B, T, 1024]
  split at dim=-1, size 512 each
  → k_raw: [B, T, 512]
  → v_raw: [B, T, 512]
```

Reshape to head layout:

```
k_raw [B, T, 512] → reshape → K_raw: [B, T, n_kv_h, d_h] = [B, T, 2, 256]
v_raw [B, T, 512] → reshape → V_raw: [B, T, n_kv_h, d_h] = [B, T, 2, 256]
```

---

## 3. Q-Gating

The gate is applied to the raw query vectors before any normalization. This is an element-wise operation after a sigmoid activation:

$$\text{gate sigmoid} = \sigma(\text{gate raw})$$

$$Q_{\text{gated}} = Q_{\text{raw}} \odot \text{gate sigmoid}$$

Both tensors have shape `[B, T, 16, 256]`.

**Geometric interpretation.** Each element of `gate_sigmoid` lies in (0, 1). Multiplying by Q entry-wise performs input-dependent suppression: query dimensions that the model deems irrelevant for the current token are scaled toward zero. This gives the model the ability to selectively "blank out" parts of its query representation on a per-token, per-head, per-dimension basis — a form of soft feature selection that standard attention lacks.

The gate is learned jointly with Q through the shared projection weight, so the model can co-adapt Q values and their gating masks.

---

## 4. Normalization (Per-Head RMSNorm)

Both Q_gated and K_raw are independently normalized. Each head has its own RMSNorm weight vector of shape `[d_h] = [256]`.

```
q_norm_weight: [256]   (one per query head — or shared; tied per layer)
k_norm_weight: [256]   (one per KV head — or shared; tied per layer)

Q_normed = RMSNorm(Q_gated, weight=q_norm_weight):  [B, T, 16, 256]
K_normed = RMSNorm(K_raw,   weight=k_norm_weight):  [B, T,  2, 256]
```

The normalization is applied over the last dimension (d_h = 256) for each (B, T, head) slice independently. Shapes are unchanged.

**Order note.** The sequence is: project → $\sigma(\text{gate raw}) \odot Q$ → $\text{RMSNorm}(Q_{\text{gated}})$. Most models that use both gating and normalization apply normalization first. Here the gating precedes the norm, which means the norm operates on already-masked query vectors, stabilizing the magnitude of the surviving signal rather than the raw projection.

V is not normalized.

---

## 5. Rotary Position Embedding (RoPE)

RoPE is applied to the normalized Q and K tensors. The rotary dimension is 64, which is only 25% of the head dimension d_h = 256:

```
rotary_dim = 64

Q_normed [B, T, 16, 256]:
  first 64 dims of d_h → rotated with RoPE frequencies
  remaining 192 dims   → passed through unchanged
  → Q_rope: [B, T, 16, 256]   (shape unchanged)

K_normed [B, T, 2, 256]:
  first 64 dims of d_h → rotated with RoPE frequencies
  remaining 192 dims   → passed through unchanged
  → K_rope: [B, T, 2, 256]    (shape unchanged)
```

The rotation is in the (cos, sin) style standard to RoPE. Because only 64 out of 256 dimensions carry positional signal, the remaining 192 dimensions of each head are position-agnostic — they encode content only. This is a deliberate design choice that reduces the fraction of head capacity consumed by positional encoding.

---

## 6. KV Cache Update

Gated Attention layers maintain a paged KV cache (there are 10 such layers in the 40-layer model). After RoPE:

```
K_rope [B, T, 2, 256] → write to KV cache at positions [cur_pos : cur_pos + T]
V_raw  [B, T, 2, 256] → write to KV cache at positions [cur_pos : cur_pos + T]
```

Cache layout after update (full context up to T_max):

```
K cache: [B, n_kv_h, T_max, d_h] = [B, 2, T_max, 256]
V cache: [B, n_kv_h, T_max, d_h] = [B, 2, T_max, 256]
```

For prefill the entire input range is written. For decode a single new position is appended. The paged cache allocates blocks of slots; the logical shape above is the conceptual dense layout.

---

## 7. GQA Expansion (K and V Repeat)

Gated Attention uses Grouped Query Attention: n_q_h = 16 query heads share n_kv_h = 2 KV heads. Each KV head is shared by:

```
n_q_h / n_kv_h = 16 / 2 = 8 query heads
```

K and V are repeated (or interleaved) 8× along the head dimension before SDPA:

```
K_rope [B, T, 2, 256]  → repeat 8×  → K_exp: [B, T, 16, 256]
V_raw  [B, T, 2, 256]  → repeat 8×  → V_exp: [B, T, 16, 256]
```

After this step all three tensors Q_rope, K_exp, V_exp have the same head count (16) and head dimension (256).

---

## 8. Scaled Dot-Product Attention (SDPA)

The attention computation differs between prefill and decode. In both cases the heads dimension is moved before T for standard SDPA conventions:

```
Q_rope: [B, T, 16, 256] → transpose → [B, 16, T, 256]
K_exp:  [B, T, 16, 256] → transpose → [B, 16, T, 256]   (prefill)
V_exp:  [B, T, 16, 256] → transpose → [B, 16, T, 256]   (prefill)
```

**Prefill (T > 1):**

$$\text{scores} = \frac{Q_{\text{rope}} \cdot K_{\text{exp}}^\top}{\sqrt{d_h}}$$

Shape: $[B, 16, T, 256] \cdot [B, 16, 256, T] \to [B, 16, T, T]$

$$\text{weights} = \text{softmax}(\text{scores})$$

Shape: $[B, 16, T, T]$ (causal mask applied before softmax)

$$\text{attn output} = \text{weights} \cdot V_{\text{exp}}$$

Shape: $[B, 16, T, T] \cdot [B, 16, T, 256] \to [B, 16, T, 256]$

Shape check: `sqrt(d_h) = sqrt(256) = 16.0`.

**Decode (T = 1):**

```
attn_output = scaled_dot_product_attention_decode(Q_rope, K_cache, V_cache, cur_pos)

  Q_rope:   [B, 16, 1, 256]

  -- read from KV cache (stored with n_kv_h = 2 heads) --
  K_cache:  [B, 2, S, 256]    where S = total cached sequence length
  V_cache:  [B, 2, S, 256]

  -- GQA 8× expansion (after cache read, before SDPA) --
  K_cache:  [B, 2, S, 256]  → repeat 8×  →  K_exp: [B, 16, S, 256]
  V_cache:  [B, 2, S, 256]  → repeat 8×  →  V_exp: [B, 16, S, 256]

  scores = (Q_rope @ K_exp^T) / sqrt(d_h)
         = (Q_rope @ K_exp^T) / 16          →  [B, 16, 1, S]
  weights = softmax(scores):                    [B, 16, 1, S]
  attn_output = weights @ V_exp             →  [B, 16, 1, 256]
```

During decode the K and V read from the cache already span the full context; no masking is needed beyond ignoring unwritten slots. The GQA expansion to 16 heads is performed in-place after the cache read and before the dot product — the cache itself is always allocated and written with `n_kv_h = 2` heads.

---

## 9. Output Projection

The attention output is reshaped and projected back to the model hidden size:

```
attn_output [B, 16, T, 256]
  → transpose heads/time: [B, T, 16, 256]
  → reshape:              [B, T, n_q_h × d_h] = [B, T, 16 × 256] = [B, T, 4096]
```

Shape check: `n_q_h × d_h = 16 × 256 = 4096`.

```
o_proj weight: [H, n_q_h × d_h] = [2048, 4096]

attn_output [B, T, 4096] × weight^T [4096, 2048]
  → output: [B, T, 2048]
```

The output `[B, T, 2048]` is added to the residual stream.

---

## 10. Summary: Complete Shape Flow

```
Input hidden_states            [B, T, 2048]
  ↓ q_gate_proj
q_gate_raw                     [B, T, 8192]
  ↓ split (4096 / 4096)
Q_raw                          [B, T, 4096]  →  reshape  →  [B, T, 16, 256]
gate_raw                       [B, T, 4096]  →  reshape  →  [B, T, 16, 256]
  ↓ kv_proj
kv_raw                         [B, T, 1024]
  ↓ split (512 / 512)
K_raw                          [B, T, 512]   →  reshape  →  [B, T,  2, 256]
V_raw                          [B, T, 512]   →  reshape  →  [B, T,  2, 256]
  ↓ sigmoid(gate_raw)
gate_sigmoid                   [B, T, 16, 256]
  ↓ Q_raw ⊙ gate_sigmoid
Q_gated                        [B, T, 16, 256]
  ↓ RMSNorm per head
Q_normed                       [B, T, 16, 256]
K_normed                       [B, T,  2, 256]
  ↓ RoPE (first 64 dims)
Q_rope                         [B, T, 16, 256]
K_rope                         [B, T,  2, 256]
  ↓ KV cache write; GQA expand (8×)
K_exp                          [B, T, 16, 256]
V_exp                          [B, T, 16, 256]
  ↓ SDPA
attn_output                    [B, 16, T, 256]
  ↓ transpose + reshape
                               [B, T, 4096]
  ↓ o_proj
output                         [B, T, 2048]
```

---

**Next:** [`gated_vs_vanilla_attention_shapes.md`](./gated_vs_vanilla_attention_shapes.md)
