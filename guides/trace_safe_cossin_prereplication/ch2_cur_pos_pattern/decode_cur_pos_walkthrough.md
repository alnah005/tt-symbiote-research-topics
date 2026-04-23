# The _decode_cur_pos Walkthrough

This file walks through the `_decode_cur_pos` pre-allocation code in `move_weights_to_device_impl`, annotates each choice, and traces the update path through `ttnn.copy` at each decode step. By the end you will be able to identify the three properties that make this pattern trace-safe and understand why they are each necessary.

> **Note:** The tt-symbiote source repository was not accessible at a local path during the writing of this guide. The code shown below is reconstructed from the plan specification (`guides/trace_safe_cossin_prereplication/plan.md`), the research topic description in `research_topics.md`, and cross-references in related guides. Where exact line numbers are cited they are approximate. A TODO marker is placed at any point where direct source confirmation was not possible.

---

## Allocation in `move_weights_to_device_impl`

`move_weights_to_device_impl` is called once, during the model initialization phase, before any trace capture begins. Its purpose is to move weight tensors (and any auxiliary persistent tensors) from host memory to device DRAM, establishing their stable addresses. `_decode_cur_pos` is allocated here because it must exist at a stable address before the capture bracket opens.

```python
# From TTNNQwen3FullAttention.move_weights_to_device_impl
# (or a parent class in the same file)
# TODO: confirm exact file path — expected at
#   models/experimental/tt_symbiote/nn/attention/qwen3_full_attention.py

def move_weights_to_device_impl(self):
    # --- weight tensors moved to device first (standard pattern) ---
    # ... (QKV projection weights, output projection weights, etc.)

    # Allocate the decode current-position scalar.
    # why: this tensor must exist at a stable device address before
    #      ttnn.begin_trace_capture is called; any allocation inside
    #      the capture bracket would bake a transient address into the
    #      command buffer, which would be invalid on replay.
    self._decode_cur_pos = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        # why: int32 matches the expected dtype of cur_pos_tensor in
        #      ttnn.transformer.scaled_dot_product_attention_decode;
        #      bfloat16 or float32 would cause a dtype mismatch at the
        #      SDPA decode op boundary.
        dtype=ttnn.int32,
        # why: ROW_MAJOR_LAYOUT avoids tile-padding overhead for a
        #      scalar (shape [1]); a single int32 value padded to a
        #      32x32 tile would waste 1023 elements and add conversion
        #      overhead when passing to ops that expect a scalar index.
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=self.mesh_device,
        # why: ReplicateTensorToMesh places one full copy on every
        #      device in the TP mesh so that each device's SDPA kernel
        #      sees the same current position index; a sharded
        #      distribution would give each device only a partial view,
        #      which is meaningless for a scalar.
        mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # why: DRAM is appropriate for a persistent scalar that lives
        #      for the entire decode session; L1 is reserved for
        #      intermediate activations with short lifetimes.
    )
    # why: storing on self ensures the buffer is kept alive for the
    #      lifetime of the module instance; if it were a local variable,
    #      the Python garbage collector could reclaim the device buffer
    #      before the trace is captured, invalidating the stable address.
```

> **Key Finding:** The allocation call above uses `ttnn.from_torch` — the same function that is forbidden inside the trace capture bracket. What makes this safe is its placement: it is called inside `move_weights_to_device_impl`, which runs before `ttnn.begin_trace_capture`. The buffer address produced here is stable from this moment onward; it is the address that will later be baked into the command buffer during capture.

---

## Per-Step Update via `ttnn.copy`

At each decode step, the current position integer is wrapped into a host tensor, then written into the pre-allocated device buffer using `ttnn.copy`. The critical property is that `ttnn.copy` does not allocate a new device buffer — it writes into the existing one at its stable address.

```python
# Inside the decode loop (or at the top of forward, inside the trace bracket)
# TODO: confirm exact call site — may be in TTNNQwen3FullAttention.forward
#       or in a parent class decode loop

def forward(
    self,
    hidden_states: ttnn.Tensor,
    current_pos: int,           # host-side Python integer for the current step
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    ...
) -> ttnn.Tensor:

    # Wrap the current position as a host tensor, then copy into the
    # pre-allocated device scalar.
    # why: constructing a torch tensor on host is a host operation and
    #      is trace-safe (it does not touch the device command queue);
    #      the ttnn.copy call below is the device operation that IS
    #      recorded in the trace command buffer.
    cur_pos_torch = torch.tensor([current_pos], dtype=torch.int32)
    cur_pos_host = ttnn.from_torch(cur_pos_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn.copy(cur_pos_host, self._decode_cur_pos)  # trace-safe: writes into stable address, no reallocation
    # why: ttnn.copy enqueues a DMA transfer from cur_pos_host to the
    #      device buffer at self._decode_cur_pos's stable address;
    #      no new device buffer is allocated; the DMA command recorded
    #      in the trace references the same stable address on every replay.

    # All downstream ops receive self._decode_cur_pos, whose address is
    # stable and whose content has just been updated for this step.
    attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
        ...,
        cur_pos_tensor=self._decode_cur_pos,
        ...
    )
    ...
```

> **Warning:** The host-side `ttnn.from_torch` call for `cur_pos_host` (without `device=`) is safe because it produces a host-resident tensor (no device allocation). The device-resident `self._decode_cur_pos` was allocated once in `move_weights_to_device_impl` and is never re-allocated. Only the content changes, via `ttnn.copy`.

---

## The Three Properties That Make This Pattern Work

### Property 1 — Allocation before trace capture begins

`_decode_cur_pos` is allocated in `move_weights_to_device_impl`, which is called during model initialization — before `TracedRun._capture_trace` opens the `ttnn.begin_trace_capture` bracket. The device buffer address is therefore established and stable before any command is recorded.

If the allocation were inside `forward` (inside the capture bracket), the address would be transient: it would be baked into the command buffer during the capture run and then potentially freed or reused before the first replay. The result would be silent data corruption on replay.

### Property 2 — Fixed shape and dtype throughout the decode loop

`_decode_cur_pos` has shape `[1]` and dtype `int32` for every decode step, regardless of batch size, sequence length, or position index. This is necessary because the command buffer records the concrete shape and dtype constraints expected by each kernel. A change in shape or dtype between decode steps would produce a tensor that the recorded kernels cannot process, causing a device error or silent miscomputation.

The fixed shape also means the pre-allocated buffer can serve every step without reallocation: the same one int32 value is overwritten at each step via `ttnn.copy`.

### Property 3 — Update via `ttnn.copy` (not `ttnn.from_torch`) inside the trace

`ttnn.copy(source, destination)` enqueues a DMA command that writes `source`'s data into `destination`'s existing device buffer. It is recorded in the trace command buffer during capture and re-issued verbatim on every replay, writing the step's position value into the stable buffer before any subsequent kernel reads from it.

If `ttnn.from_torch(device=self.mesh_device)` were used instead of `ttnn.copy`, a new device buffer would be allocated on each forward call. During capture, the new buffer's address would be baked into the command buffer. During replay, Python does not re-execute — the `ttnn.from_torch` call is not re-run, the new buffer is not re-allocated, and the command buffer references a stale address.

---

## Forward Reference

For the generalizable pattern extracted from these three properties — and for the analysis of how cos/sin differ from `_decode_cur_pos` — see [`pattern_generalization.md`](./pattern_generalization.md).
