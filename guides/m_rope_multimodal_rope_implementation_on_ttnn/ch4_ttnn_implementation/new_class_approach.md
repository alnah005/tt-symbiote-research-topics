# New Class Approach: Option B

## Section 1: Option B — New `TTNNMRoPERotaryPositionEmbedding` Class

A standalone class that handles only M-RoPE, with no backward-compatibility branching. The forward interface is unambiguous: it always accepts a `[3, batch, seq_len]` position ID tensor. There is no `use_mrope` flag, no standard-path fallback, and no conditional dispatch in the decode hot path.

This is the preferable architecture for production because the class contract is explicit — any caller using this class is using M-RoPE, and the class implementation has no dead code for the standard case.

---

## Section 2: Interface

```python
class TTNNMRoPERotaryPositionEmbedding:
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_seq_len: int,
        rope_theta: float,
        mrope_section: list[int],  # required; [s_t, s_h, s_w]
    ):
        # Same frequency table construction as TTNNRotaryPositionEmbedding
        # cos_table [max_seq_len, rotary_dim/2], sin_table [max_seq_len, rotary_dim/2]
        self.s_t, self.s_h, self.s_w = mrope_section

    def forward(self, q, k, position_ids_3d):
        # Always performs 3-gather construction
        # No branching, no use_mrope flag
        ...
```

`mrope_section` is a required parameter with no default — there is no valid M-RoPE configuration without it. The class raises `ValueError` at construction time if `sum(mrope_section) != rotary_dim // 2`, catching misconfiguration before any inference runs.

The cos/sin table construction is identical to `TTNNRotaryPositionEmbedding`: the same frequency formula, same `[max_seq_len, rotary_dim/2]` shape, same BF16 dtype, same DRAM placement. The two classes differ only in their forward method and constructor signature.

---

## Section 3: Trade-offs

| Criterion | Option A (extension) | Option B (new class) |
|---|---|---|
| Code impact | Modifies existing class | New file, no existing changes |
| Backward compatibility | Guaranteed (flag=False path unchanged) | N/A — separate class |
| Hot path cleanliness | Conditional branch in decode loop | No branching |
| Test isolation | Tests must cover both flag states | M-RoPE tests fully isolated |
| Module registration changes | None | Register new class in symbiote module map |
| Preferred for | Initial bring-up | Production implementation |

The conditional branch in Option A (`if self.use_mrope`) is evaluated once per decode step. For long-running inference (thousands of decode steps), this is negligible in absolute time but represents unnecessary code that runs in a tight loop. Option B eliminates this entirely.

Module registration is the primary additional cost of Option B. TT-Symbiote maintains a module map that routes model configuration to TTNN layer classes; adding a new class requires an entry in that map. This is low-effort but touches infrastructure that Option A avoids entirely.

---

## Section 4: Recommendation

For TT-Symbiote initial M-RoPE bring-up, use **Option A**. It avoids touching module registration logic, keeps the text-only fast path unchanged, and limits the diff surface to the single existing class — reducing review burden and regression risk during the bring-up phase.

When M-RoPE is fully validated and becomes a first-class inference mode (not just a research bring-up), refactor to **Option B** to eliminate the conditional branch overhead in the decode hot path and achieve clean separation between standard and multimodal RoPE implementations. The refactor is mechanical: extract `_forward_mrope` into the new class's forward, drop the `use_mrope` flag, and update the module map.

---
**Next:** [`pre_computed_cos_sin_strategy.md`](./pre_computed_cos_sin_strategy.md)
