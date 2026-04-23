# Non-Tile-Aligned `rotary_dim` Interaction

This document addresses the interaction between the pre-replication change and `rotary_dim` values that are not multiples of 32 (the tile boundary for `TILE_LAYOUT`). For Qwen3 with `rotary_dim = 64`, this interaction is clean — 64 is tile-aligned and no special handling is required. For other model configurations with non-tile-aligned `rotary_dim`, the pre-allocation plan from Chapter 3 requires modification. This document defines the boundary between the two cases and explains why this guide focuses exclusively on the tile-aligned case.

---

## Tile Alignment and `TILE_LAYOUT`

`TILE_LAYOUT` requires that the last two dimensions of a tensor be multiples of 32. For a cos/sin tensor with shape `[1, 1, 1, rotary_dim]`, the constraint is `rotary_dim % 32 == 0`.

For Qwen3: `rotary_dim = 64`, and `64 % 32 == 0`. The pre-allocated buffer can use `TILE_LAYOUT` directly, with no padding, no layout conversion, and no changes to the downstream `ttnn.experimental.rotary_embedding` call.

For a hypothetical model with `rotary_dim = 48`: `48 % 32 != 0`. The pre-allocated buffer cannot use `TILE_LAYOUT` with shape `[1, 1, 1, 48]` without either padding to `[1, 1, 1, 64]` or switching to `ROW_MAJOR_LAYOUT`.

---

## Tile-Aligned Case (Qwen3, `rotary_dim = 64`)

The pre-replication plan from Chapters 3 and 4 applies without modification:

```python
# In move_weights_to_device_impl:
# why: 64 is a multiple of 32, so TILE_LAYOUT is valid for [1, 1, 1, 64].
#      No padding or layout conversion is needed.
self._cos_replicated = ttnn.zeros(
    shape=[1, 1, 1, self.rotary_dim],   # [1, 1, 1, 64]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
```

This is the canonical pattern. `ttnn.copy` into this buffer is straightforward because source and destination share the same layout and shape.

---

## Non-Tile-Aligned Case

For `rotary_dim` values that are not multiples of 32, two options exist:

**Option A — Use `ROW_MAJOR_LAYOUT`:**

Pre-allocate `_cos_replicated` with `ROW_MAJOR_LAYOUT`. This avoids padding but may require a layout conversion before `ttnn.experimental.rotary_embedding` if that kernel requires `TILE_LAYOUT`. Layout conversions inside the trace bracket allocate new buffers and are trace-unsafe; the conversion would need to happen outside the bracket or via a separate pre-allocated intermediate buffer.

**Option B — Pad to the next tile boundary:**

Pre-allocate `_cos_replicated` with shape `[1, 1, 1, rotary_dim_padded]` where `rotary_dim_padded = ((rotary_dim + 31) // 32) * 32`. Use `TILE_LAYOUT`. The padding columns are initialized to zero and do not affect the rotary embedding computation if the kernel respects the unpadded width. This option preserves `TILE_LAYOUT` throughout but introduces a shape mismatch between the source cos slice (`[1, 1, 1, rotary_dim]`) and the destination buffer (`[1, 1, 1, rotary_dim_padded]`), which `ttnn.copy` may not handle without explicit padding of the source.

Neither option is analyzed further in this guide. The numerical correctness implications of non-tile-aligned `rotary_dim` — including the interaction with partial rotary embeddings — are covered in the separate research topic at `partial_rotary_non_tile_aligned_numerics/`.

> **Note:** This guide focuses exclusively on the tile-aligned Qwen3 case (`rotary_dim = 64`). All code patterns, assertions, and test plans in Chapters 3 through 6 assume `rotary_dim % 32 == 0`.

---

## The Unsqueeze Issue

An additional shape concern arises if the caller passes cos/sin with shape `[1, 1, rotary_dim]` (three dimensions) rather than the expected `[1, 1, 1, rotary_dim]` (four dimensions). If `TTNNQwen3FullAttention.forward` previously relied on `ttnn.unsqueeze` inside the traced path to add the batch dimension, that call is trace-unsafe: `ttnn.unsqueeze` allocates a new view tensor, and if that allocation happens inside the trace bracket, the address is not stable at replay time.

The fix is to ensure the cos/sin tensor already has the correct four-dimensional shape before entering any code inside the trace bracket:

- Pre-allocate `_cos_replicated` with shape `[1, 1, 1, rotary_dim]`.
- Perform the `unsqueeze` outside the trace bracket, before `begin_trace_capture` is called, or as part of the pre-trace cos/sin preparation that feeds into the stable kwarg buffer.
- Inside the trace bracket, `ttnn.copy(cos, self._cos_replicated)` assumes `cos` already has shape `[1, 1, 1, rotary_dim]`.

If the incoming `cos` slice from the DRAM table already has the correct four-dimensional shape (which is typical when the table is built with that shape), no unsqueeze is needed anywhere in the traced path.

> **Warning:** Never call `ttnn.unsqueeze`, `ttnn.reshape`, or any other view-creating operation inside the trace bracket unless that operation is documented to be trace-safe. Check whether the operation allocates a new device buffer or merely creates a Python-side view object over the same device memory. If it allocates a new buffer, it is trace-unsafe.
