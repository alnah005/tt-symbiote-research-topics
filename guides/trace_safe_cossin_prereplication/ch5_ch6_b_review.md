# B Review — Chapters 5 and 6: Warm-Up Guard Preservation and Integration

## Pass 1

### Issues found: 3

---

**Issue 1 — `guard_mechanism_analysis.md`, Section "When the Guard Fires", imprecise claim about Python execution inside the trace bracket**

**Error:** The text states: "inside the trace bracket, Python code does not execute; the guard check is Python code and therefore runs only during the capture pass, which behaves like a warm-up."

The first clause is wrong as written. Python code *does* execute during the capture pass — the capture pass is the one Python-executed run inside the trace bracket during which device commands are recorded. Python does not execute during *replay* iterations. Saying "inside the trace bracket, Python code does not execute" contradicts the key operational fact: the capture pass runs Python eagerly, which is precisely why the guard fires during that pass. Only subsequent `ttnn.execute_trace` calls skip Python.

**Correction:** Change to: "inside the trace bracket, Python code executes once during the capture pass (recording device commands) but does not execute on subsequent replay calls. The guard check is Python code and therefore runs during the capture pass but not on any replay iteration."

---

**Issue 2 — `integration_checklist.md`, Post-Implementation Checks, warm-up guard test instruction**

**Error:** The post-implementation check states: "Run the warm-up guard test: call `forward` with a sharded (not replicated) cos/sin tensor passed as the argument, bypassing `move_weights_to_device_impl`. Confirm that `TTNNRotaryPositionEmbedding.forward` raises the expected error."

This test cannot produce the described outcome under the new code path. After the change, `TTNNQwen3FullAttention.forward` calls `ttnn.copy(cos, self._cos_replicated)` before passing `self._cos_replicated` to `TTNNRotaryPositionEmbedding.forward`. `ttnn.copy` writes data into `self._cos_replicated` without changing its memory layout or mapper. `self._cos_replicated` was pre-allocated with `ReplicateTensorToMesh` and retains full `rotary_dim` columns regardless of what sharded data was copied into it. Therefore `TTNNRotaryPositionEmbedding.forward` will receive a tensor with `rotary_dim` columns and the guard will not fire. This is internally contradicted by `guard_mechanism_analysis.md`, which correctly states: "The only way the guard would fire after the change is if `_cos_replicated` were accidentally pre-allocated with a sharded mapper instead of a replicated mapper."

The test as written would silently pass (no error raised) and give the implementer a false assurance that the guard is working against sharded inputs when it is not.

**Correction:** Replace the guard test with the correct postcondition: verify that passing a sharded cos tensor to `TTNNQwen3FullAttention.forward` does NOT raise (because the copy writes data without changing layout, and `self._cos_replicated` retains full columns). The actual guard-preservation test — i.e., confirming the guard fires on a sharded mapper at pre-allocation time — must be performed by calling `TTNNRotaryPositionEmbedding.forward` directly with a sharded tensor (as Test 4 in `test_plan.md` correctly does), not by passing a sharded tensor through `TTNNQwen3FullAttention.forward`.

---

**Issue 3 — `test_plan.md`, Test 3 "Purpose of eight steps specifically", incorrect rationale**

**Error:** The text states: "Eight steps cover all devices in the T3K mesh. This ensures that per-device DMA commands in the trace are all updating the correct device-local buffers, and that no device is reading stale cos/sin from a previous step."

This rationale is wrong. Each call to `ttnn.execute_trace` replays the recorded command queue on all devices in the mesh simultaneously — it does not proceed one device at a time across successive replay calls. A single replay already exercises all 8 devices' DMA commands. Running 8 steps does not provide "one step per device" coverage in any meaningful sense. The actual value of running 8 steps is to verify that the cos/sin buffer is correctly refreshed (via `ttnn.copy` outside the bracket) before each replay, and that no step reads values from a prior step — i.e., it tests temporal state isolation across replay iterations, not spatial coverage across devices.

**Correction:** Change the rationale to: "Eight steps verify that the cos/sin buffer is correctly refreshed between replay calls and that no replay iteration reads stale values from a previous call. The count eight is chosen to be small enough to run quickly while large enough to distinguish a first-replay fluke from consistent correct behavior across multiple decode positions. A single replay (Test 2) can pass even if the refresh mechanism is intermittently broken; repeated replays expose stale-buffer bugs reliably."

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:

1. **`guard_mechanism_analysis.md`, "When the Guard Fires", third bullet:** Replace "inside the trace bracket, Python code does not execute" with a formulation that distinguishes the capture pass (Python executes, records commands) from replay iterations (Python does not execute).

2. **`integration_checklist.md`, Post-Implementation Checks, warm-up guard test:** Rewrite the guard test to reflect the correct post-change behavior: a sharded input to `TTNNQwen3FullAttention.forward` does not trigger the guard because `ttnn.copy` does not change the layout of `self._cos_replicated`. Direct the guard test to call `TTNNRotaryPositionEmbedding.forward` directly with a sharded tensor (consistent with Test 4 in `test_plan.md`).

3. **`test_plan.md`, Test 3, "Purpose of eight steps specifically":** Replace the incorrect "covers all devices" rationale with the correct rationale: repeated replays verify temporal state isolation (correct buffer refresh between calls), not per-device spatial coverage.

## Pass 2

### Verification of Pass 1 fixes

**Fix 1 (guard_mechanism_analysis.md — capture pass vs replay distinction):** APPLIED — The third bullet in "When the Guard Fires" now reads: "Decode steps executed inside the trace bracket — but note that the trace bracket contains one Python-executed capture pass (during which device commands are recorded) and then zero or more replay calls (which re-issue those commands without executing Python). The guard check is Python code and therefore runs during the capture pass but not on any replay iteration."

**Fix 2 (integration_checklist.md — guard test rewritten):** APPLIED — The guard test bullet now reads: "Run the warm-up guard test by directly calling `TTNNRotaryPositionEmbedding.forward` with a sharded cos tensor (shape `[1, 1, 1, rotary_dim / 8]` per device) as described in Test 4 of `test_plan.md`. Do not attempt to trigger the guard via `TTNNQwen3FullAttention.forward`: after the change, that path calls `ttnn.copy(cos, self._cos_replicated)` before passing `self._cos_replicated` downstream. `ttnn.copy` writes data into `self._cos_replicated` without changing its memory layout or mapper; `self._cos_replicated` retains full `rotary_dim` columns regardless of what sharded data was written. The guard will therefore never fire via the outer `forward` path — confirm instead that a sharded cos passed to `TTNNQwen3FullAttention.forward` does NOT raise, which is the expected and correct post-change behavior."

**Fix 3 (test_plan.md — 8-step rationale corrected):** APPLIED — The "Purpose of eight steps specifically" sentence now reads: "Eight steps verify that the cos/sin buffer is correctly refreshed between replay calls and that no replay iteration reads stale values from a previous call. Each `ttnn.execute_trace` call replays on all 8 T3K devices simultaneously — a single replay already exercises all device-local DMA commands; running more replays does not add per-device coverage. The count of eight is chosen to be small enough to run quickly while large enough to distinguish a first-replay fluke from consistent correct behavior across multiple decode positions. Repeated replays expose stale-buffer bugs that a single-step test cannot detect."

### Issues found: 1

---

**Issue 1 — `integration_checklist.md`, Post-Implementation Checks, guard test bullet: incorrect claim that passing a sharded cos to `TTNNQwen3FullAttention.forward` does NOT raise**

**Location:** Post-Implementation Checks, second bullet, last sentence: "confirm instead that a sharded cos passed to `TTNNQwen3FullAttention.forward` does NOT raise, which is the expected and correct post-change behavior."

**Error:** This claim is wrong. After the change, `TTNNQwen3FullAttention.forward` calls `ttnn.copy(cos, self._cos_replicated)` before the guard is reached. The source tensor `cos` is sharded: per-device shape `[1, 1, 1, rotary_dim / 8]` = `[1, 1, 1, 8]`. The destination `self._cos_replicated` is replicated: per-device shape `[1, 1, 1, rotary_dim]` = `[1, 1, 1, 64]`. These per-device shapes are incompatible. `ttnn.copy` requires source and destination to have matching shapes; it does not reinterpret or broadcast across a shape mismatch. The call will raise a shape mismatch error at the `ttnn.copy` site — before control ever reaches `TTNNRotaryPositionEmbedding.forward` and the `rotary_dim % 64 != 0` guard.

The reasoning in the bullet ("ttnn.copy writes data into `self._cos_replicated` without changing its memory layout or mapper; `self._cos_replicated` retains full `rotary_dim` columns regardless of what sharded data was written") is correct about the mapper being preserved, but it silently assumes that `ttnn.copy` can copy between tensors with different per-device shapes. That assumption is false. `ttnn.copy` is an in-place data write into an existing buffer — it requires the source and destination to be shape-compatible. A sharded source (8 columns per device) and a replicated destination (64 columns per device) fail this requirement.

The practical consequence: the checklist directs the implementer to verify that the outer `forward` path "does NOT raise" with a sharded cos input, implying that any raised exception is a test failure. In reality, an exception will always be raised (at `ttnn.copy`, not at the guard), and the implementer following the checklist will misinterpret a correct `ttnn.copy` shape error as a test failure and may incorrectly modify the implementation to suppress it.

**Correction:** Remove the sentence "confirm instead that a sharded cos passed to `TTNNQwen3FullAttention.forward` does NOT raise, which is the expected and correct post-change behavior." Replace it with: "Do not attempt to pass a sharded cos tensor to `TTNNQwen3FullAttention.forward` as a test probe: the `ttnn.copy(cos, self._cos_replicated)` call will raise a shape mismatch error before the guard is reached, because the sharded source (8 columns per device) is shape-incompatible with the replicated destination (64 columns per device). The guard-preservation test must be conducted by calling `TTNNRotaryPositionEmbedding.forward` directly with a sharded tensor, as described in Test 4 of `test_plan.md`."

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. `integration_checklist.md` Post-Implementation Checks, guard test bullet: replaced "confirm instead that a sharded cos passed to `TTNNQwen3FullAttention.forward` does NOT raise" with accurate statement — a sharded cos (8 columns per device) is shape-incompatible with `self._cos_replicated` (64 columns per device) so `ttnn.copy` will raise a shape mismatch error at the copy site; directed all guard testing exclusively to calling `TTNNRotaryPositionEmbedding.forward` directly

## Pass 3

### Verification of Pass 2 fix

**Fix (integration_checklist.md — "does NOT raise" removed, shape mismatch warning added):** APPLIED — The guard test bullet in Post-Implementation Checks now reads: "Run the warm-up guard test by directly calling `TTNNRotaryPositionEmbedding.forward` with a sharded cos tensor (shape `[1, 1, 1, rotary_dim / 8]` per device) as described in Test 4 of `test_plan.md`. Do not attempt to trigger the guard via `TTNNQwen3FullAttention.forward`: after the change, that path calls `ttnn.copy(cos, self._cos_replicated)` before passing `self._cos_replicated` downstream. `ttnn.copy` writes data into `self._cos_replicated` without changing its memory layout or mapper; `self._cos_replicated` retains full `rotary_dim` columns regardless of what sharded data was written. The guard will therefore never fire via the outer `forward` path. Do not use a sharded cos passed to `TTNNQwen3FullAttention.forward` as a guard test vector: the per-device shard has `rotary_dim / 8` columns while `self._cos_replicated` has `rotary_dim` columns — shape-incompatible for `ttnn.copy` — so a shape mismatch error will be raised at the `ttnn.copy` call site, not at the guard. All guard testing must be done by calling `TTNNRotaryPositionEmbedding.forward` directly with a sharded tensor, as described in Test 4 of `test_plan.md`."

### Issues found: 0

---

No issues found. Chapters 5 and 6 approved.

---
