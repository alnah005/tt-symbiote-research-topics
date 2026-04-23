# Source Tensor Stability: Where cos/sin Come From at Decode Time

The correctness of `ttnn.copy(cos, self._cos_replicated)` inside the trace depends on BOTH operands being stable pre-existing device tensors. Chapter 3 establishes that the destination (`_cos_replicated`) is stable because it is pre-allocated in `move_weights_to_device_impl`. This file addresses the other operand: the `cos` argument passed to `forward` at each decode step. The question is whether that argument is a device tensor with a stable address, or whether it is freshly created (or freshly moved to device) on each step in a way that would invalidate the trace.

---

## Where cos/sin Come From in `TTNNRotaryPositionEmbedding`

`TTNNRotaryPositionEmbedding` precomputes a full cos/sin table during initialization. The table covers all sequence positions up to `max_seq_len` and has shape `[1, 1, max_seq_len, rotary_dim]`. It is allocated as a device DRAM tensor before any trace capture begins, so its device address is fixed for the lifetime of the decode session.

```python
# During TTNNRotaryPositionEmbedding.__init__ (before trace capture):
# why: the full table is computed once and stored on device DRAM;
#      its address is stable for the entire decode session.
self._cos_table = ttnn.from_torch(
    precomputed_cos,                     # shape [1, 1, max_seq_len, rotary_dim]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
self._sin_table = ttnn.from_torch(
    precomputed_sin,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

At decode time, the caller slices the table at the current position `cur_pos` **outside the trace bracket** (in eager Python, before `ttnn.execute_trace` is called), then passes the resulting view as the `cos`/`sin` argument to the traced `forward`:

```python
# OUTSIDE the trace bracket — runs in eager Python before each execute_trace call:
# why: Metal Trace replay does NOT re-execute Python; the slice must happen here,
#      before the trace runs, so the view's device address reflects the current step.
cos = self._cos_table[:, :, cur_pos:cur_pos + 1, :]   # shape [1, 1, 1, rotary_dim]
sin = self._sin_table[:, :, cur_pos:cur_pos + 1, :]   # shape [1, 1, 1, rotary_dim]
# Then pass cos/sin as kwargs to execute_trace (TracedRun's kwarg pre-allocation
# mechanism copies the view's contents into a stable pre-allocated device buffer
# before replay begins; the trace's DMA command reads from that stable buffer address).
```

The slice is a view into the pre-allocated DRAM tensor — it does not allocate a new device buffer. The view's device address points into the DRAM table at the offset for the current `cur_pos`. Because the slice runs in eager Python before each `execute_trace` call, the `cos` argument that arrives in the traced `forward` already points to the correct position's data for that step.

Inside the trace, `cos` is a stable device tensor at a fixed pre-allocated address (the address baked into the trace's DMA command at capture time). `ttnn.copy(cos, self._cos_replicated)` reads from that fixed address — whose CONTENTS were updated in eager mode before the replay — and writes into `_cos_replicated`, also at a fixed pre-allocated address.

> **Trace Invariant:** The source tensor's device address is baked into the trace at capture time. Updating a Python variable (like `cur_pos`) after capture does NOT change which device memory the DMA reads from. The source must be a stable pre-allocated buffer that is updated in eager mode OUTSIDE the trace bracket before each replay. Per-step cos/sin values flow into the trace because the caller updates the kwarg buffer's contents in eager mode before `execute_trace` — not because Python re-executes the slice during replay.

---

## What Would Make the Source Unstable

Two design choices would make the `cos`/`sin` source unstable and break the trace:

**CPU computation moved to device inside the trace bracket:**

```python
# INCORRECT — trace-unsafe:
# why: ttnn.from_torch allocates a new device buffer on every call;
#      the new address is not known at capture time;
#      the trace records an address that becomes stale on replay step 1.
cos_cpu = compute_cos_for_position(cur_pos)       # CPU tensor
cos = ttnn.from_torch(cos_cpu, device=mesh_device)  # new device buffer each step
ttnn.copy(cos, self._cos_replicated)              # source address is NOT stable ✗
```

**Slice that produces a copy rather than a view:**

```python
# INCORRECT — trace-unsafe:
# why: if the slice operation internally calls an op like ttnn.pad that produces
#      a new device buffer (rather than a view), the source gets a new address
#      each step; the DMA command in the trace records a stale address.
cos = ttnn.pad(self._cos_table[:, :, cur_pos:cur_pos + 1, :], ...)  # new buffer ✗
```

Both of these cases produce a new device buffer address at each step. The DMA command recorded in the trace embeds the address from the capture run. On replay step 1, that captured address still points to the capture-time data (position 0), not the position-1 data that was intended. The output is wrong, and no runtime error fires.

See the `Trace Invariant` in [`what_copy_records.md`](./what_copy_records.md) for the complete statement.

---

## Forward Reference: Pre-Replication of the Source Table Itself

The cos/sin DRAM table inside `TTNNRotaryPositionEmbedding` must itself be allocated on the correct devices (with the correct mesh mapper) before trace capture begins. The analysis of how that table is moved to device — and the mesh mapping required so that each device holds the right shard — is covered in Ch3's discussion of `move_weights_to_device_impl`. See [`../ch3_prereplication_impl/move_weights_impl_changes.md`](../ch3_prereplication_impl/move_weights_impl_changes.md).
