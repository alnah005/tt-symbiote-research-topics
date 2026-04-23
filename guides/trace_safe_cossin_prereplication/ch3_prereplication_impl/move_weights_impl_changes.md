# Concrete Code Changes: `move_weights_to_device_impl` and `forward`

This file provides the concrete change plan — what to add to `move_weights_to_device_impl` and what to change in `forward`, with annotated code and rationale for each choice. By the end you will have a complete, copy-ready implementation template that applies the pre-allocation pattern from Chapter 2 to the cos/sin case and eliminates `_ensure_replicated` from the traced call path.

---

## Section 1: Change 1 — Add Pre-Allocation to `move_weights_to_device_impl`

```python
# In TTNNQwen3FullAttention.move_weights_to_device_impl
# (or parent class, if cos/sin allocation is shared across attention types)
# TODO: confirm exact class — expected in
#   models/tt_symbiote/nn/attention/qwen3_full_attention.py

def move_weights_to_device_impl(self):
    # --- existing weight transfers ---
    # ... (QKV projection weights, output projection weights, etc.)
    # --- existing _decode_cur_pos allocation ---
    # ... (unchanged)

    # Pre-allocate replicated cos/sin buffers for trace-safe decode steps.
    # why: these buffers must exist at stable device addresses before
    #      ttnn.begin_trace_capture is called; the per-step values are written
    #      into them via ttnn.copy inside the traced region (see forward below).
    rotary_dim = self.rotary_dim  # TODO: confirm attribute name
    self._cos_replicated = ttnn.from_torch(
        torch.zeros(1, 1, 1, rotary_dim, dtype=torch.bfloat16),
        # dtype: bfloat16 — see downstream_op_constraints.md §4
        dtype=ttnn.bfloat16,
        # layout: TILE_LAYOUT — see downstream_op_constraints.md §2
        layout=ttnn.TILE_LAYOUT,
        device=self.mesh_device,
        # mesh_mapper: ReplicateTensorToMesh — see replicated_mesh_mapping.md §2
        mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        # why: persistent decode-session buffer — see downstream_op_constraints.md §3
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    self._sin_replicated = ttnn.from_torch(
        torch.zeros(1, 1, 1, rotary_dim, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=self.mesh_device,
        mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

`torch.zeros` is used for initialization — the actual values do not matter because the first `ttnn.copy` inside the trace bracket (at the first decode step) will overwrite them before they are read by `ttnn.experimental.rotary_embedding`.

---

## Section 2: Change 2 — Replace `_ensure_replicated` in `forward`

```python
# In TTNNQwen3FullAttention.forward

def forward(
    self,
    hidden_states: ttnn.Tensor,
    cos: ttnn.Tensor,      # decode-step cos, shape [1, 1, 1, rotary_dim] (device tensor)
    sin: ttnn.Tensor,      # decode-step sin, shape [1, 1, 1, rotary_dim] (device tensor)
    ...
) -> ttnn.Tensor:

    # BEFORE:
    # cos = self._ensure_replicated(cos)  # trace-unsafe: allocates new device buffer each call
    # sin = self._ensure_replicated(sin)  # trace-unsafe: allocates new device buffer each call

    # AFTER:
    # Update the pre-allocated replicated buffers with this step's cos/sin values.
    # why: ttnn.copy writes into the existing stable buffer at self._cos_replicated's address;
    #      no new device buffer is allocated; the DMA command recorded in the trace
    #      references the same stable address on every replay.
    ttnn.copy(cos, self._cos_replicated)
    ttnn.copy(sin, self._sin_replicated)

    # All downstream ops use self._cos_replicated and self._sin_replicated,
    # whose addresses are stable and whose content has just been updated.
    # (Replace all uses of `cos` and `sin` below this point with
    #  self._cos_replicated and self._sin_replicated.)
    ...
```

The `cos` and `sin` arguments to `forward` must themselves be on-device tensors with stable addresses. See [`../ch4_copy_trace_safety/source_tensor_stability.md`](../ch4_copy_trace_safety/source_tensor_stability.md) for analysis of where these come from at decode time.

---

## Section 3: What NOT to Do

> **Warning:** Do NOT place the `ttnn.copy` calls before `begin_trace_capture`. If the copy happens outside the trace bracket, the trace command buffer will not include the per-step update — every replay will use the values from the capture run (position 0), silently producing wrong rotary embeddings for positions 1, 2, 3, ...

> **Warning:** Do NOT use `ttnn.clone(cos)` as a replacement for `_ensure_replicated`. `ttnn.clone` allocates a new destination buffer on every call — it is as trace-unsafe as `ttnn.from_torch`.

---

## Section 4: Placement Summary

| Operation | Location | Trace-safety |
|---|---|---|
| `ttnn.from_torch(zeros, ...)` to create `_cos_replicated` | `move_weights_to_device_impl` (before trace) | Safe: runs once, before capture |
| `ttnn.copy(cos, self._cos_replicated)` | `forward` (inside trace bracket) | Safe: writes into stable address |
| `ttnn.experimental.rotary_embedding(..., cos=self._cos_replicated)` | `forward` (inside trace bracket) | Safe: reads from stable address |
| `_ensure_replicated(cos)` (removed) | `forward` (inside trace bracket) | Unsafe: allocates new buffer each step |
