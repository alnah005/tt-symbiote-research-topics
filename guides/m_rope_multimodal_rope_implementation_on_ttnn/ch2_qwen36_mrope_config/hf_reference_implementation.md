# HuggingFace Reference Implementation

## 1. Overview

The HuggingFace reference implementation for M-RoPE is split across two functions:

1. `Qwen2_5_VLRotaryEmbedding.forward()` — assembles the multimodal cos/sin tensor from 3D position IDs and the shared frequency table.
2. `apply_multimodal_rotary_pos_emb()` — applies the assembled cos/sin to query and key tensors.

This file traces both functions exactly so that the TTNN implementation in Chapter 4 can replicate their numerical behavior. These two functions together constitute the numerical ground truth for TTNN validation.

---

## 2. `apply_multimodal_rotary_pos_emb` Walkthrough

### Function signature

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    # q: [B, num_q_heads, S, head_dim]
    # k: [B, num_kv_heads, S, head_dim]
    # cos, sin: [B, S, rotary_dim]  (assembled from three sections)
    # mrope_section: [s_t, s_h, s_w] (e.g., [11, 11, 10])
```

At the point this function is called, `cos` and `sin` are already fully assembled multimodal tensors of shape `[B, S, rotary_dim]`. The three-section gather has already happened upstream in `Qwen2_5_VLRotaryEmbedding.forward()` (covered in section 3 below).

### Step-by-step trace

**Step a — Unsqueeze for the head dimension.**

```python
cos = cos.unsqueeze(unsqueeze_dim)   # [B, 1, S, rotary_dim]
sin = sin.unsqueeze(unsqueeze_dim)   # [B, 1, S, rotary_dim]
```

This broadcasts cos/sin across all attention heads when multiplied against `q` and `k`.

**Step b — Split q and k into rotated and pass-through portions.**

Only the first `rotary_dim` dimensions of each head receive RoPE. The remaining `head_dim - rotary_dim` dimensions pass through unchanged:

```python
# Apply rotate-half to only the first rotary_dim dimensions of q and k
q_rot  = q[..., :rotary_dim]    # [B, num_q_heads, S, rotary_dim]
q_pass = q[..., rotary_dim:]    # [B, num_q_heads, S, head_dim - rotary_dim]

k_rot  = k[..., :rotary_dim]    # [B, num_kv_heads, S, rotary_dim]
k_pass = k[..., rotary_dim:]    # [B, num_kv_heads, S, head_dim - rotary_dim]
```

For Qwen3.6: `rotary_dim = 64`, `head_dim = 128`, so `q_rot` has 64 dimensions and `q_pass` has 64 dimensions.

**Step c — Apply the rotate-half transformation.**

The `rotate_half` helper rearranges the vector so that dimension $i$ is paired with dimension $i + \text{rotary\_dim}/2$:

```python
def rotate_half(x):
    # x: [..., rotary_dim]
    x1 = x[..., : x.shape[-1] // 2]   # first half:  dimensions [0, rotary_dim/2)
    x2 = x[..., x.shape[-1] // 2 :]   # second half: dimensions [rotary_dim/2, rotary_dim)
    return torch.cat([-x2, x1], dim=-1)
```

This is the rotate-half convention: pairs $(x_i,\, x_{i + \text{rotary\_dim}/2})$, not adjacent pairs $(x_{2i},\, x_{2i+1})$.

**Step d — Compute the embedded output.**

```python
q_embedded = q_rot * cos + rotate_half(q_rot) * sin
q_out = torch.cat([q_embedded, q_pass], dim=-1)

k_embedded = k_rot * cos + rotate_half(k_rot) * sin
k_out = torch.cat([k_embedded, k_pass], dim=-1)
```

---

## 3. Cos/Sin Assembly in `Qwen2_5_VLRotaryEmbedding.forward()`

This is where the multimodal cos/sin tensor is assembled from 3D position IDs and the shared half-length frequency table.

```python
# position_ids: [3, B, S]
# cos_table, sin_table: [max_seq_len, rotary_dim // 2]

s_t, s_h, s_w = mrope_section  # [11, 11, 10]

# Three separate row-gather operations, one per coordinate:
cos_t = cos_table[position_ids[0]][:, :, :s_t]           # [B, S, 11]
cos_h = cos_table[position_ids[1]][:, :, s_t:s_t+s_h]    # [B, S, 11]
cos_w = cos_table[position_ids[2]][:, :, s_t+s_h:]        # [B, S, 10]

# Assemble half-length vector then duplicate for rotate-half:
cos_half = torch.cat([cos_t, cos_h, cos_w], dim=-1)      # [B, S, 32]
cos_full = torch.cat([cos_half, cos_half], dim=-1)        # [B, S, 64]
# Similarly for sin_full
```

**Why the duplication step is necessary.** The rotate-half transformation pairs dimension $i$ with dimension $i + 32$ for Qwen3.6 (where $\text{rotary\_dim}/2 = 32$). The rotation formula for pair $(x_i, x_{i+32})$ is:

```math
\begin{pmatrix} x_i' \\ x_{i+32}' \end{pmatrix}
=
\begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}
\begin{pmatrix} x_i \\ x_{i+32} \end{pmatrix}
```

Both $x_i$ and $x_{i+32}$ need the same $\cos\theta_i$ and $\sin\theta_i$ value. Concatenating `cos_half` with itself makes `cos_full[..., i] == cos_full[..., i+32]` for all $i$, satisfying this requirement.

---

## Position ID Construction

Position ID shapes, dtypes, and construction logic for text-only and vision inputs are covered in [`position_id_construction.md`](./position_id_construction.md).

---

## 5. Key Finding

> **Key Finding:** The HuggingFace reference splits the cos/sin assembly from the rotation application. The `Qwen2_5_VLRotaryEmbedding.forward()` method assembles the multimodal cos/sin tensor; `apply_multimodal_rotary_pos_emb()` applies it. The TTNN implementation must replicate both steps. Chapter 4 maps `forward()` to `ttnn.embedding` calls and `apply_multimodal_rotary_pos_emb()` to the existing rotate-half kernel.

---

## 6. Forward References

- Chapter 4 (`../ch4_ttnn_implementation/extension_approach.md`) shows how the assembly and rotation operations above map to TTNN ops.
- Chapter 3 (`../ch3_text_only_reduction/mathematical_equivalence_proof.md`) proves that when all three position ID rows are identical, the assembled cos/sin is numerically equal to the standard 1D RoPE output.

---

**Next:** [`position_id_construction.md`](./position_id_construction.md)
