# Existing TTNNRotaryPositionEmbedding Gap Analysis

## Section 1: What `TTNNRotaryPositionEmbedding` Currently Does

**Constructor:** Computes frequency pairs for each dimension pair `i` in `[0, rotary_dim/2)`:

```math
θ_i = 1 / rope_theta^(2i / rotary_dim)
```

For Qwen3.6: `rotary_dim = 64`, `rope_theta = 1000000.0`, producing 32 frequency values. These are used to build `[max_seq_len, rotary_dim/2]` cos and sin tables (or a combined `[max_seq_len, rotary_dim]` table) covering all token positions up to `max_seq_len`.

**Forward:** Given a 1D position index — a scalar `cur_pos` at decode or a `[seq_len]` tensor at prefill — the class slices a contiguous row range:

```python
cos_slice = cos_table[cur_pos : cur_pos + seq_len]  # [seq_len, rotary_dim/2]
sin_slice = sin_table[cur_pos : cur_pos + seq_len]  # [seq_len, rotary_dim/2]
```

**Rotate-half:** Q and K are split into two halves along the head dimension, each `[..., rotary_dim/2]`:

```python
q_rot1 = q1 * cos - q2 * sin
q_rot2 = q2 * cos + q1 * sin
q_out = concat([q_rot1, q_rot2], dim=-1)
```

(Both halves are rotated. The unrotated "pass-through" dimensions — the last `head_dim - rotary_dim = 64` dims — are handled separately in the partial RoPE application.)

**Partial RoPE:** Only the first `rotary_dim = 64` dimensions of the `head_dim = 128` vector are rotated. The last 64 dimensions pass through unchanged. This is why `TTNNRotaryPositionEmbedding` (not `TTNNDistributedRotaryPositionEmbedding`) is used for Qwen3.6 — the distributed variant is forced off when `partial_rotary_factor < 1.0` due to shape misalignment.

---

## Section 2: Gap 1 — 1D Position Index vs. 3D Position ID Tensor

**Current interface:** The class accepts a scalar or 1D `[seq_len]` position index — one integer per token.

**M-RoPE requirement:** A `[3, batch_size, seq_len]` tensor providing three integers per token: temporal axis (frame index), height axis (patch row), and width axis (patch column).

Passing a 3D position ID tensor to the current class would either fail silently (if the class interprets the tensor shape incorrectly) or require destructuring outside the class — spreading M-RoPE-specific logic into the caller rather than encapsulating it in the RoPE layer. Both outcomes are unacceptable for a clean integration.

---

## Section 3: Gap 2 — Single Contiguous Slice vs. Three Independent Gathers

**Current access pattern:** `cos_table[cur_pos : cur_pos + seq_len]` — always a contiguous row range, with the same position coordinate used for all dimension pairs simultaneously.

**M-RoPE access pattern:** Three separate row gathers using different position coordinates per axis, each returning full rows, then column-sliced and concatenated:

```
# Step 1: gather full rows for each axis (random-access, via ttnn.embedding)
cos_all_t = ttnn.embedding(position_ids[0], cos_table)  # [batch, seq_len, 32]
cos_all_h = ttnn.embedding(position_ids[1], cos_table)  # [batch, seq_len, 32]
cos_all_w = ttnn.embedding(position_ids[2], cos_table)  # [batch, seq_len, 32]

# Step 2: slice to section columns and concatenate
cos_t = cos_all_t[:, :, :s_t]          # temporal section: columns [:11]
cos_h = cos_all_h[:, :, s_t:s_t+s_h]  # height section:   columns [11:22]
cos_w = cos_all_w[:, :, s_t+s_h:]     # width section:    columns [22:32]
cos   = concat([cos_t, cos_h, cos_w], dim=-1)  # [batch, seq_len, 32]
```

This is a fundamentally different access pattern: random-access per-token row lookup (using `ttnn.embedding`) rather than a range slice. The position coordinates for the temporal, height, and width sections are independently defined per token and can differ arbitrarily (e.g., for a 3×4 image patch grid, all tokens in the same row share the same height index but have distinct width indices).

---

## Section 4: What Does NOT Need to Change

> **Key Finding:** The two gaps above are the only changes required. The rotate-half kernel, the DRAM storage layout, and the partial RoPE application are all modality-agnostic and work without modification.

- **Rotate-half kernel:** Operates on whatever assembled cos/sin tensor is provided. It has no knowledge of where cos/sin came from — whether a contiguous slice or a three-gather concatenation. No changes needed.
- **TTNN elementwise multiply:** The `q * cos` and `k * sin` operations are unchanged; they depend only on the assembled cos/sin shape matching Q/K, which it does.
- **DRAM placement of cos/sin table:** The same `[max_seq_len, 32]` half-table works for both standard RoPE and M-RoPE. No additional storage, no shape change.
- **Partial RoPE application:** The first 64 of 128 head dimensions are rotated; the last 64 pass through. This boundary is unchanged for M-RoPE.
- **Text-only fast path:** When all three position axes are identical (sequential IDs), the three-gather path produces identical output to the current contiguous slice. The existing text-only path can continue using the fast slice — no overhead added for text-only inference.

### Summary

| Component | Change needed? | Why |
|---|---|---|
| Frequency table construction | No | Same frequencies, same shape |
| Table storage / DRAM placement | No | Same shape, same dtype |
| Position input signature | Yes | 1D → 3D position ID |
| Table lookup mechanism | Yes | Contiguous slice → three gathers + concat |
| Rotate-half kernel | No | Operates on assembled cos/sin, modality-agnostic |
| Partial RoPE application | No | First 64 dims unchanged |

---
**Next:** [`extension_approach.md`](./extension_approach.md)
