# Trace-Safe Alternatives to `ttnn.pad`

This file explains why `ttnn.pad` is unsafe inside a TTNN trace bracket and describes the available alternatives. The problem is not unique to the partial RoPE bug: any operation that allocates a new device buffer inside a trace bracket is trace-unsafe. The partial RoPE implementation is a specific instance of this general problem, and the solutions here apply broadly to any forward-pass operation that currently relies on runtime padding.

---

## 1. Why `ttnn.pad` Is Trace-Unsafe

TTNN traces (compiled execution graphs) require that all device tensor allocations happen before the trace is captured, not during replay. When a trace is replayed:

- The sequence of kernel launches is fixed at capture time.
- Buffer addresses used by each kernel are resolved at capture time and baked into the trace.
- Any operation that allocates a new buffer during replay breaks this assumption: the allocator may return a different address, or may not be called at all if the trace is replaying a previously-cached sequence.

`ttnn.pad` creates a new device buffer to hold the padded result. When called inside the forward pass — which is inside the trace bracket — this allocation occurs during replay, violating the trace contract. The symptom is typically a crash or silent data corruption during traced inference.

> **Key Finding:** `ttnn.pad` allocates a new device buffer every time it is called. Calling it inside a trace bracket is trace-unsafe regardless of the sizes involved. This is not a limitation of the partial RoPE implementation in particular; it applies to all uses of `ttnn.pad` inside `model.forward`.

---

## 2. Primary Solution: Strategy C

The cleanest resolution for partial RoPE is Strategy C: construct the cos/sin table with the correct layout in `__init__` so that no runtime padding is ever needed. The forward pass receives a pre-existing device tensor of the exact shape the op requires — no pad, no concat, no allocation.

See [`strategy_c_precomputed_full_head_cos_sin.md`](./strategy_c_precomputed_full_head_cos_sin.md) for the full construction. The rest of this file addresses situations where Strategy C is not applicable — for example, when Strategy A (slice-apply-concat) is being used and the input slice or output concat requires a buffer of the right shape.

---

## 3. Alternative 1 — Pre-Allocated Zeros Buffer with `ttnn.concat`

The idea is to replace the `ttnn.pad` call with a concatenation between the input slice and a pre-allocated zeros buffer of the padding size.

```python
class TTNNRotaryPositionEmbedding:
    def __init__(self, rotary_dim, head_dim, max_seq_len, batch, heads, device, ...):
        pad_width = 64 - (rotary_dim % 64)  # e.g. 64 - 48 = 16
        if pad_width == 64:
            pad_width = 0  # rotary_dim is already tile-aligned

        if pad_width > 0:
            # Pre-allocate the zeros padding buffer once; never reallocated
            self.pad_zeros = ttnn.zeros(
                [batch, heads, max_seq_len, pad_width],
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.pad_zeros = None

    def forward(self, x, ...):
        x_rot = x[..., :rotary_dim]   # slice to rotated region

        if self.pad_zeros is not None:
            # Replace ttnn.pad with concat against pre-allocated buffer
            x_rot_padded = ttnn.concat([x_rot, self.pad_zeros], dim=-1)
        else:
            x_rot_padded = x_rot

        out_rot = ttnn.experimental.rotary_embedding(x_rot_padded, cos, sin)
        ...
```

### Trace-safety caveat

The `ttnn.concat` call itself allocates an output buffer. To make even the concat trace-safe, the concat output buffer must also be pre-allocated and the concat must be performed into that buffer using an in-place or output-buffer-providing variant. Whether TTNN exposes such a variant depends on the op implementation.

> **Note:** Alternative 1 eliminates the `ttnn.pad` allocation but may not eliminate the `ttnn.concat` output allocation. It is safer than `ttnn.pad` in practice because the zeros buffer is stable (same address across replays), but it does not guarantee trace safety unless the concat output is also pre-allocated.

---

## 4. Alternative 2 — `ttnn.copy` into a Pre-Allocated Identity Buffer

This alternative pre-allocates a full `[batch, heads, max_seq_len, head_dim]` buffer initialized to identity values (cos=1.0, sin=0.0 for the cos/sin tensors, or zeros for input padding). At forward time, the current-position values are copied into the appropriate slice of the pre-allocated buffer using `ttnn.copy` or an in-place write. The `__init__` and `forward` structure is identical to Strategy C — see [`strategy_c_precomputed_full_head_cos_sin.md` Section 7](./strategy_c_precomputed_full_head_cos_sin.md) for the full code.

This approach is structurally identical to Strategy C. The only difference is in how the cos/sin values at positions `[0, rotary_dim)` are chosen: Alternative 2 places real cos/sin values at `[0, rotary_dim)` and identity at `[rotary_dim, head_dim)`, while Strategy C (the fully correct variant) also duplicates the frequency values at Region 3 (`[head_dim/2, head_dim/2 + rotary_dim/2)`) to account for the kernel's pairing structure.

> **Note:** Alternative 2 without the Region 3 frequency duplication still has the incorrect pairing problem for positions `[0, rotary_dim/2)`: the kernel pairs `input[j]` with `input[j + head_dim/2]`, but the cos/sin at `j + head_dim/2` is 1.0/0.0 (identity), so `output[j] = input[j] * c_j + input[j + head_dim/2] * (-s_j)` — wrong (uses `input[j + head_dim/2]` instead of the rotate-half partner). Full correctness requires Strategy C's Region 3 duplication.

---

## 5. Comparison of Approaches

| Approach | Eliminates `ttnn.pad` | Trace-safe | Correct output | Complexity |
|---|---|---|---|---|
| Current implementation (`ttnn.pad` inside forward) | No | No | No (PCC ~0.71) | Low |
| Alternative 1 (pre-allocated zeros + `ttnn.concat`) | Yes | Partial (concat output may still allocate) | Only if rotary_dim % 64 == 0 | Medium |
| Alternative 2 (`ttnn.copy` into pre-allocated buffer, no Region 3 duplication) | Yes | Yes | Partial (positions `[0, rotary_dim/2)` still wrong) | Medium |
| Strategy C (full precomputed table with Region 3 duplication) | Yes | Yes | Yes (given correct input layout) | Medium |

---

## 6. Recommendation

Strategy C is the recommended approach for all cases where `head_dim % 64 == 0` and `rotary_dim < head_dim`. It:

- Eliminates all runtime allocation in `forward`.
- Is fully trace-compatible.
- Requires only a one-time construction of the cos/sin table in `__init__`.
- Produces correct output given the input head uses the `head_dim/2`-split pairing convention.

Alternatives 1 and 2 are retained here for reference because they represent intermediate steps that engineers may reach when debugging the trace-safety problem before arriving at Strategy C. Neither is recommended as a final implementation.

> **Key Finding:** The correct and trace-compatible solution is to move all buffer construction into `__init__` and ensure that `forward` performs zero device memory allocations. Strategy C achieves this by precomputing the full `[max_seq_len, head_dim]` cos/sin table with identity and duplicated-frequency values. No other approach achieves all three properties (correctness, trace safety, and no runtime allocation) simultaneously.

---

**Return to:** [Chapter 4 Index](./index.md)
