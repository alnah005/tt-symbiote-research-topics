# The Generalizable Pre-Allocation Pattern

This file extracts the abstract four-step pre-allocation pattern from the `_decode_cur_pos` example, then identifies how cos/sin position embeddings differ from a scalar position index and what those differences mean for the pre-allocation design. By the end you will have the abstract template and the cos/sin-specific constraints that Chapter 3 implements concretely.

---

## The Four-Step Pre-Allocation Pattern

The `_decode_cur_pos` example in [`decode_cur_pos_walkthrough.md`](./decode_cur_pos_walkthrough.md) generalizes to the following four-step template for any decode-step tensor that changes value at each step but must be trace-safe:

### Step 1 — Allocate in `move_weights_to_device_impl`

In `move_weights_to_device_impl` (or an equivalent pre-capture setup hook that runs before `ttnn.begin_trace_capture`), allocate the device tensor using `ttnn.from_torch` or `ttnn.zeros` with the full target shape, dtype, layout, and memory config. Store a reference to the tensor on the module instance (`self`).

```python
# BEFORE trace capture — inside move_weights_to_device_impl
self._my_tensor = ttnn.from_torch(
    torch.zeros(TARGET_SHAPE, dtype=TARGET_TORCH_DTYPE),
    dtype=TARGET_TTNN_DTYPE,
    layout=TARGET_LAYOUT,          # e.g., ttnn.TILE_LAYOUT or ttnn.ROW_MAJOR_LAYOUT
    device=self.mesh_device,
    mesh_mapper=TARGET_MESH_MAPPER, # e.g., ReplicateTensorToMesh or ShardTensorToMesh
    memory_config=TARGET_MEM_CONFIG, # e.g., ttnn.DRAM_MEMORY_CONFIG
)
# why: storing on self keeps the device buffer alive and at its address
#      for the entire lifetime of the module instance; Python garbage
#      collection cannot reclaim it while self holds a reference.
```

The choice of `torch.zeros` as the initial value is a placeholder. The content is meaningless at this point; only the shape, dtype, layout, and memory config matter for establishing the stable address.

### Step 2 — Verify replication before trace capture

Before `ttnn.begin_trace_capture` is called, verify that the pre-allocated tensor is on device with the expected replication or sharding factor. For a tensor that must be replicated across all TP devices, verify that `ttnn.get_device_tensors(self._my_tensor)` returns one tensor per device and that each per-device tensor has the expected shape.

```python
# Debug assertion — run during warm-up, guarded by a debug flag in production
if DEBUG_ASSERTIONS:
    device_tensors = ttnn.get_device_tensors(self._my_tensor)
    assert len(device_tensors) == self.mesh_device.get_num_devices(), (
        f"Expected one tensor per device, got {len(device_tensors)}"
    )
    for t in device_tensors:
        assert list(t.shape) == EXPECTED_PER_DEVICE_SHAPE, (
            f"Wrong shape: {list(t.shape)} vs {EXPECTED_PER_DEVICE_SHAPE}"
        )
```

### Step 3 — Update via `ttnn.copy` inside the traced region

Inside `forward` (and therefore inside the capture bracket), update the tensor content by copying from the step's source data into the pre-allocated buffer. Use `ttnn.copy(source, destination)` where `destination` is `self._my_tensor`. No new device buffer is allocated.

```python
# INSIDE the trace capture bracket — at the top of forward
# why: the copy must be inside the bracket so that it is recorded in
#      the command buffer and re-executed on every replay, ensuring
#      each replay sees the current step's data before any kernel reads.
ttnn.copy(current_step_source, self._my_tensor)

# AFTER: use self._my_tensor downstream; its content is now current
```

> **Warning:** If `ttnn.copy` is placed outside the capture bracket (before `ttnn.begin_trace_capture`), it will not be recorded in the command buffer. On replay, the update will not be re-executed — the kernel will read whatever value was in the buffer from the previous replay. For per-step tensors that change value at every step, the copy must be inside the bracket.

### Step 4 — Pass the updated device tensor to downstream ops

Use `self._my_tensor` in place of the original tensor throughout the rest of `forward`. Because `self._my_tensor` is a persistent device tensor at a stable address, all downstream ops record its address during capture and reference the same stable address on every replay.

```python
output = some_downstream_op(
    ...,
    my_tensor=self._my_tensor,  # stable address, current content
    ...
)
```

---

## What Makes cos/sin Different from `_decode_cur_pos`

The `_decode_cur_pos` pattern applies directly to cos/sin, but two structural differences affect the specific choices in Steps 1 and 2:

### Difference (a) — cos/sin have non-scalar shape

`_decode_cur_pos` is a scalar (shape `[1]`). cos/sin position embeddings have a multi-dimensional shape. For a single-token decode step (seq_len=1) in Qwen3, the expected shape passed to `ttnn.experimental.rotary_embedding` is typically `[1, 1, 1, rotary_dim]` (batch=1, 1, seq_len=1, rotary_dim). cos/sin tensors are position-dependent and shared across all heads; the second axis is a broadcast 1, not num_heads. The exact shape depends on whether `ttnn.unsqueeze` is applied before passing cos/sin to the rotary embedding op.

This shape must be determined precisely before the pre-allocated buffer is created in Step 1. The buffer shape must match the shape that the downstream rotary embedding op expects to receive, accounting for any intermediate shape transforms (such as `ttnn.unsqueeze`) that occur in the current `forward` method. See [`../ch3_prereplication_impl/downstream_op_constraints.md`](../ch3_prereplication_impl/downstream_op_constraints.md) for the derivation of the required shape.

### Difference (b) — cos/sin must be replicated across TP devices

`_decode_cur_pos` is a scalar index that is the same on every device — `ReplicateTensorToMesh` is used, but the replication requirement is trivially satisfied because the value is a single integer.

For cos/sin, replication is a functional requirement: `ttnn.experimental.rotary_embedding` requires the full cos/sin frequency table on each device so that each device can apply the rotation to its own head shard. If cos/sin were sharded along the `rotary_dim` axis, each device would hold only a partial frequency table — for example, rotary dimension components 0–31 but not 32–63. Every device's heads require the full frequency table to compute correct rotary embeddings, so sharding along `rotary_dim` would produce incorrect results.

This is the original bug that `_ensure_replicated` was introduced to fix (see Chapter 1, [`../ch1_trace_incompatibility/ensure_replicated_call_site.md`](../ch1_trace_incompatibility/ensure_replicated_call_site.md)). The pre-replication fix moves the `ReplicateTensorToMesh` mapping into `move_weights_to_device_impl` as part of the initial buffer allocation, eliminating the need for `_ensure_replicated` to detect and correct sharding at runtime inside the forward pass.

---

## The Design Decision

Given the two structural differences above, the design decision is:

**Follow the `_decode_cur_pos` pattern exactly, with the following cos/sin-specific choices:**

1. **Shape**: allocate a 4D buffer of shape `[1, 1, 1, rotary_dim]` (single-token decode shape), matching the shape that `ttnn.experimental.rotary_embedding` expects after any `ttnn.unsqueeze` transforms.
2. **Layout**: `ttnn.TILE_LAYOUT`, because `ttnn.experimental.rotary_embedding` requires tile-aligned input for compute. Allocating in `ROW_MAJOR_LAYOUT` would force a layout conversion inside the trace, which may allocate a transient buffer — a potential trace-safety concern.
3. **Memory config**: `ttnn.DRAM_MEMORY_CONFIG`, matching the pattern for persistent pre-allocated buffers in this model.
4. **Mesh mapper**: `ReplicateTensorToMesh(self.mesh_device)`, ensuring every device in the TP mesh holds a full copy of the cos/sin table for the current decode position.
5. **Instance attributes**: `self.cos_replicated` and `self.sin_replicated`, allocated in `move_weights_to_device_impl` and updated via `ttnn.copy` at the top of `forward`.

> **Note:** The `rotary_dim` value for Qwen3 is 64, which is a multiple of 32, so the last dimension requires no padding. However, TILE_LAYOUT pads both innermost dimensions: the second-to-last dimension (seq_len = 1) is not a multiple of 32 and will be padded to 32. The effective on-device shape is `[1, 1, 32, 64]`. Downstream ops that receive this buffer must accept the padded shape. For non-tile-aligned `rotary_dim` values, the layout choice must be revisited — this is covered by the separate "partial rotary embedding numerical correctness" research topic.

---

## Forward Reference

The concrete implementation of this design decision — the exact code changes to `move_weights_to_device_impl` and `TTNNQwen3FullAttention.forward` — is in [`../ch3_prereplication_impl/move_weights_impl_changes.md`](../ch3_prereplication_impl/move_weights_impl_changes.md).

For the analysis of whether `TracedRun._alloc_kwarg_tensor` provides an alternative allocation point, see [`traced_run_alloc_kwarg_tensor.md`](./traced_run_alloc_kwarg_tensor.md).
