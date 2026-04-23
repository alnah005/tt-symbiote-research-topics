# Test Plan

This document defines five concrete tests that together give high confidence in the correctness and trace safety of the pre-replication change. The tests are ordered from simplest (non-traced correctness) to most integrated (full hybrid decoder trace). Each test includes a setup description, the inputs required, the validation criterion, and the purpose it serves in the overall verification strategy.

---

## Test 1 — Non-Traced Forward Correctness

**Purpose:** Establish that the pre-replication change does not alter the numerical output of `TTNNQwen3FullAttention.forward` on a standard non-traced decode step. This is the baseline correctness gate; all subsequent tests depend on this passing.

**Setup:**
1. Load `TTNNQwen3FullAttention` with weights on a T3K mesh.
2. Call `move_weights_to_device_impl` to pre-allocate `_cos_replicated` and `_sin_replicated`.
3. Prepare inputs: query `q` of shape `[batch, n_heads, 1, head_dim]`, key `k` of shape `[batch, n_kv_heads, 1, head_dim]`, cos of shape `[1, 1, 1, rotary_dim]` replicated across all 8 devices, sin of shape `[1, 1, 1, rotary_dim]` replicated across all 8 devices.
4. Prepare an FP32 reference output using the same forward pass logic in PyTorch on CPU (no TTNN).

**Inputs:**
- Random `q`, `k` tensors; decode position index 0.
- Cos/sin sliced from the pre-computed DRAM table at position 0, already replicated.

**Execution:**
- Call `TTNNQwen3FullAttention.forward` in eager mode (no trace bracket).
- Collect the output tensor, copy to host.

**Validation:**
- Compute Pearson Correlation Coefficient (PCC) between the TTNN output and the FP32 reference.
- Assert PCC > 0.999.

> **Note:** PCC > 0.999 is the standard threshold for bfloat16 vs FP32 comparison in this codebase. A lower PCC at this stage indicates a logic error in the `ttnn.copy` call or the downstream variable update in `forward`.

---

## Test 2 — Trace Capture and Single-Replay Correctness

**Purpose:** Confirm that the trace can be captured with the pre-replication change in place and that a single replay produces numerically correct output.

**Setup:**
1. Load `TTNNQwen3FullAttention` and call `move_weights_to_device_impl`.
2. Perform the compile (warm-up) run: call `forward` once in eager mode to trigger kernel compilation.
3. Begin trace capture: call `ttnn.begin_trace_capture(mesh_device, trace_buffer_size=...)`.
4. Call `TTNNQwen3FullAttention.forward` once inside the trace bracket (this is the capture pass). Pass cos/sin for decode position 0 via the pre-trace kwarg buffer update.
5. End trace capture: call `ttnn.end_trace_capture(mesh_device, ...)` and store the returned trace ID.
6. Prepare cos/sin for decode position 1 (outside the trace bracket): slice from the DRAM table and call `ttnn.copy` into the stable kwarg buffer. Then call `ttnn.execute_trace(mesh_device, trace_id)`.

**Inputs:**
- Decode position 1 cos/sin values; same `q`, `k` inputs as Test 1 but at position 1.

**Validation:**
- Compare the trace replay output to the non-traced eager output from Test 1 (run at position 1 for comparison).
- Assert PCC > 0.999.

> **Warning:** If the trace replay output matches the capture-pass output (position 0) rather than the replay-pass input (position 1), the cos/sin buffer is stale — the `ttnn.copy` inside the trace is not executing, or the copy source was not updated before `ttnn.execute_trace` was called. This is the primary failure mode to watch for.

---

## Test 3 — Multi-Step Replay Consistency

**Purpose:** Confirm that eight consecutive `ttnn.execute_trace` calls each produce the correct output for their respective decode position, and that there is no state leak between steps.

**Setup:**
1. Complete Test 2 setup through trace capture.
2. For each decode position `p` in `{1, 2, 3, 4, 5, 6, 7, 8}`:
   a. Outside the trace bracket, slice cos/sin for position `p` from the DRAM table.
   b. Call `ttnn.copy(cos_slice, stable_cos_buf)` and `ttnn.copy(sin_slice, stable_sin_buf)` to update the input buffers that the trace reads from.
   c. Call `ttnn.execute_trace(mesh_device, trace_id)`.
   d. Copy the output to host and store it.
3. For each position `p`, also run the non-traced eager `forward` with the same `q`, `k`, and the correct cos/sin for position `p`.

**Inputs:**
- Eight distinct decode positions; cos/sin values differ at each position.

**Validation:**
- For each position `p`, compute PCC between the trace replay output and the corresponding non-traced eager output.
- Assert PCC > 0.999 for all eight positions.

**Purpose of eight steps specifically:** Eight steps verify that the cos/sin buffer is correctly refreshed between replay calls and that no replay iteration reads stale values from a previous call. Each `ttnn.execute_trace` call replays on all 8 T3K devices simultaneously — a single replay already exercises all device-local DMA commands; running more replays does not add per-device coverage. The count of eight is chosen to be small enough to run quickly while large enough to distinguish a first-replay fluke from consistent correct behavior across multiple decode positions. Repeated replays expose stale-buffer bugs that a single-step test cannot detect.

> **Note:** This test is the most important of the five for confirming that `ttnn.copy` inside the trace correctly propagates new values on each replay. A single-step test (Test 2) can pass even if the copy is partially broken; multi-step testing exposes stale-buffer bugs.

---

## Test 4 — Warm-Up Guard Preservation

**Purpose:** Confirm that the `rotary_dim % 64 != 0` guard in `TTNNRotaryPositionEmbedding.forward` still fires when a sharded cos/sin tensor is passed, even after the pre-replication change.

**Setup:**
1. Instantiate `TTNNQwen3FullAttention` but do NOT call `move_weights_to_device_impl`. This leaves `self._cos_replicated = None` (or the attribute absent if the `__init__` default was not added).
2. Alternatively: call `move_weights_to_device_impl` normally, but then directly call `TTNNRotaryPositionEmbedding.forward` with a sharded cos tensor (bypassing the `ttnn.copy` call in `TTNNQwen3FullAttention.forward`).

**Inputs:**
- A cos tensor of shape `[1, 1, 1, rotary_dim]` sharded across 8 devices: each device holds shape `[1, 1, 1, rotary_dim / 8]` = `[1, 1, 1, 8]`.
- Any `q`, `k` inputs.

**Execution:**
- Call `TTNNRotaryPositionEmbedding.forward` directly with the sharded cos tensor.
- Wrap the call in `pytest.raises(...)` or equivalent.

**Validation:**
- Assert that the expected error is raised (the `rotary_dim % 64 != 0` guard).
- Assert that the error message identifies the sharding issue (check the error string if the guard has a descriptive message).

**Purpose:** This test guards against the scenario where a future refactor accidentally removes or bypasses the guard, or where the guard is moved to a code path that no longer executes during warm-up.

> **Note:** If the guard has been changed to check `self._cos_replicated` instead of the incoming argument, this test structure needs adjustment — the guard must be tested against the tensor it now inspects. The test validates the guard's continued effectiveness regardless of which tensor it checks, as long as it correctly catches the sharded case.

---

## Test 5 — Full Hybrid Decoder Trace (Integration Smoke Test)

**Purpose:** Confirm that the pre-replication change unblocks full trace capture of the complete decoder block, including both the `TTNNQwen3FullAttention` layer and any co-located `DeltaNet` or other attention-variant layers that share the trace bracket.

**Setup:**
1. Instantiate the full decoder block (all layers for one transformer depth level), including `TTNNQwen3FullAttention` and any delta-net layers.
2. Call `move_weights_to_device_impl` for all sub-modules that have one.
3. Run the compile (warm-up) pass on the full decoder block.
4. Begin trace capture for the full decoder block forward pass.
5. Execute one full forward pass inside the trace bracket.
6. End trace capture.
7. Execute one trace replay with updated cos/sin for a different decode position.

**Inputs:**
- A full decoder-block input tensor (hidden states) at decode time.
- Attention mask, key-value cache state, and position index.
- Cos/sin for the replay position.

**Validation:**
- Compute PCC between the full decoder block trace replay output and the non-traced eager output for the same inputs.
- Assert PCC > 0.99. (The threshold is slightly relaxed from 0.999 to account for numerical accumulation across multiple operations in the full block.)
- Additionally, confirm that no `ttnn.from_torch` call is made inside the trace bracket during the capture pass. This can be verified by temporarily patching `ttnn.from_torch` to raise an error inside the capture context, or by inspecting the trace command buffer for unexpected host-to-device transfers.

**Purpose:** This is the end-to-end smoke test. Tests 1 through 4 validate the `TTNNQwen3FullAttention` component in isolation. Test 5 confirms that the full-stack trace capture works in the context it will actually be used, and that no other layer in the decoder block is inadvertently performing a trace-unsafe allocation that was previously masked by the `TTNNQwen3FullAttention` failure mode.
