# Extension Approach: Option A

## Section 1: Option A — Extend `TTNNRotaryPositionEmbedding`

Add a `use_mrope: bool` flag and `mrope_section: list[int]` constructor parameter to the existing class. When `use_mrope=False` (default), the class behaves exactly as before — every existing call site, test, and inference path is unaffected. When `use_mrope=True`, the forward accepts a `[3, batch, seq_len]` position ID tensor and performs the three-gather construction described in the gap analysis.

This approach requires no new file, no module registration change, and no caller-side restructuring beyond passing the flag and the 3D position ID tensor.

---

## Section 2: Constructor Changes

```python
class TTNNRotaryPositionEmbedding:
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_seq_len: int,
        rope_theta: float,
        use_mrope: bool = False,
        mrope_section: list[int] | None = None,  # [s_t, s_h, s_w], required if use_mrope=True
    ):
        # Frequency table construction unchanged
        # Precomputes cos_table [max_seq_len, rotary_dim/2] and sin_table [max_seq_len, rotary_dim/2]
        # Store mrope_section for use in forward
        ...
```

The cos/sin table shape and values are identical to the current implementation. For Qwen3.6: `cos_table` and `sin_table` are each `[max_seq_len, 32]` in BF16, placed on device DRAM. The `mrope_section = [11, 11, 10]` parameter is stored as an instance attribute and consulted only in `_forward_mrope`; it has no effect on the table construction or the standard forward path.

If `use_mrope=True` and `mrope_section is None`, the constructor should raise `ValueError` at construction time rather than failing silently at forward time.

---

## Section 3: Forward Signature Change

```python
def forward(
    self,
    q: ttnn.Tensor,  # [..., head_dim]
    k: ttnn.Tensor,  # [..., head_dim]
    position_ids: ttnn.Tensor,  # [seq_len] for standard; [3, batch, seq_len] for M-RoPE
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    if self.use_mrope:
        return self._forward_mrope(q, k, position_ids)
    else:
        return self._forward_standard(q, k, position_ids)
```

The `position_ids` parameter is intentionally polymorphic: when `use_mrope=False`, it is the existing 1D index; when `use_mrope=True`, it is the `[3, batch, seq_len]` tensor. The dispatch happens once per forward call and is negligible overhead compared to the gather operations themselves.

---

## Section 4: M-RoPE Forward Logic

The three-gather + concat construction mirrors HuggingFace's `apply_multimodal_rotary_pos_emb()` exactly, translated to TTNN ops:

```python
def _forward_mrope(self, q, k, position_ids_3d):
    # position_ids_3d: [3, batch, seq_len]
    s_t, s_h, s_w = self.mrope_section  # [11, 11, 10] for Qwen3.6

    # Three independent gathers — each produces [batch, seq_len, rotary_dim/2]
    # then column-sliced to the section width
    cos_t = ttnn.embedding(position_ids_3d[0], self.cos_table)[:, :, :s_t]
    cos_h = ttnn.embedding(position_ids_3d[1], self.cos_table)[:, :, s_t:s_t+s_h]
    cos_w = ttnn.embedding(position_ids_3d[2], self.cos_table)[:, :, s_t+s_h:]

    sin_t = ttnn.embedding(position_ids_3d[0], self.sin_table)[:, :, :s_t]
    sin_h = ttnn.embedding(position_ids_3d[1], self.sin_table)[:, :, s_t:s_t+s_h]
    sin_w = ttnn.embedding(position_ids_3d[2], self.sin_table)[:, :, s_t+s_h:]

    # Concatenate to form full assembled cos/sin [batch, seq_len, rotary_dim/2]
    cos = ttnn.concat([cos_t, cos_h, cos_w], dim=-1)
    sin = ttnn.concat([sin_t, sin_h, sin_w], dim=-1)

    # Duplication: [batch, seq_len, rotary_dim/2] -> [batch, seq_len, rotary_dim]
    cos = ttnn.concat([cos, cos], dim=-1)
    sin = ttnn.concat([sin, sin], dim=-1)

    # Apply rotate-half (unchanged)
    return apply_rotary_pos_emb(q, k, cos, sin)
```

**Gather-then-slice approach:** `ttnn.embedding` retrieves full rows of shape `[rotary_dim/2]` from the `[max_seq_len, rotary_dim/2]` table, indexed by the per-token position integers. The subsequent slice `[:, :, :s_t]` extracts only the relevant columns for that section. This uses a single shared table with 6 embedding lookups (3 cos + 3 sin) and 4 concat operations — 2 section concatenations plus 2 duplication concatenations.

An alternative is to precompute three separate sub-tables (one per section) from the full table at construction time. This reduces gather output size but adds construction-time overhead and DRAM footprint. The gather-then-slice approach is simpler and preferred for initial bring-up.

---

## Section 5: Backward Compatibility Guarantee

When `use_mrope=False`, the class is bit-for-bit identical to the current implementation. The `_forward_standard` method is the existing forward body extracted without modification. No existing test will break, and the text-only inference path for Qwen3.6 continues to use `use_mrope=False` — it never enters `_forward_mrope` and incurs no overhead from the M-RoPE code path.

---
**Next:** [`new_class_approach.md`](./new_class_approach.md)
