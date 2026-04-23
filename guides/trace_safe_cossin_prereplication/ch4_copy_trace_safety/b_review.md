# B Review — Chapter 4: Copy Trace Safety

## Pass 1

### Issues found: 2

---

**Issue 1:** `source_tensor_stability.md`, section "Where cos/sin Come From in `TTNNRotaryPositionEmbedding`", lines 40-42

**Error:** The file states that the slice runs inside `TTNNRotaryPositionEmbedding.forward` (which is called during the traced forward), then claims:

> "When the trace replays, `cur_pos` has been updated externally (outside the trace bracket), so the slice points to a different offset — but the source tensor's base device address, which is what the DMA command uses, remains stable."

This is self-contradictory and incorrect. Metal Trace does NOT re-execute Python on replay (domain fact 8). If the slice operation `self._cos_table[:, :, cur_pos:cur_pos+1, :]` runs inside the trace bracket, it runs exactly once at capture time (with whatever `cur_pos` is at capture). The resulting view's device address offset — pointing into the DRAM table at the capture-time position — is baked into the DMA command. On replay, no Python runs, so `cur_pos` updates in Python have no effect on the baked source address. The DMA command will always copy from the capture-time offset (position 0), not from the updated position. The statement "the slice points to a different offset" on replay is false — it always points to the same captured offset.

**Correct description:** For the source address to reflect the updated `cur_pos` on each replay, the slice must happen OUTSIDE the trace bracket (in the Python call site, before `ttnn.execute_trace`). The caller updates `cur_pos`, computes `cos = self._cos_table[:, :, cur_pos:cur_pos+1, :]` in eager Python, and passes the resulting view (with its updated offset address) as the `cos` argument. The trace bracket then sees `cos` as a pre-existing device tensor at the correct address for that step. The baked DMA command copies from that address, which is stable within the trace execution for that replay call. The "different offset per step" comes from Python re-slicing outside the trace before each `execute_trace` call, not from anything happening inside the replay.

If instead the file intends to say that the slice happens before the trace bracket and the resulting view (a stable device tensor at the correct offset) is passed in as an argument, then the code snippet showing the slice inside `forward` (called "during traced forward") contradicts that intent and must be corrected to clearly place the slice outside the bracket.

---

**Issue 2:** `replay_correctness_verification.md`, section "Verification Protocol", Step 3 and Step 4, lines 23-43

**Error:** Step 3 says "the next call to `TTNNRotaryPositionEmbedding.forward` slices the correct position from the DRAM table before the trace runs." Step 4 calls `execute_trace(cur_pos=1)` and asserts that inside the trace, "`ttnn.copy` must now update `_cos_replicated` with position-1 cos/sin values."

This is consistent with Issue 1. If the slice is inside the traced `forward`, the replay will not re-execute the Python slice with the new `cur_pos`. The DMA source address baked in the trace is fixed at the capture-time offset. Claiming that `_cos_replicated` will be "updated with position-1 cos/sin values" on replay step 1 is wrong under that design — it would still copy position-0 values (the captured source address).

The protocol is only correct if the slice (and thus `cur_pos`-dependent source address selection) happens outside the trace bracket before each `execute_trace` call. The protocol should explicitly state this: before calling `execute_trace(cur_pos=1)`, the caller must re-slice the DRAM table at position 1 in eager Python and pass the resulting view as the `cos` argument to `forward` — but since the trace does not call `forward` again, this means the `cos` input tensor passed to the trace must be updated externally before replay. The protocol as written omits this mechanism and implies the trace itself handles the position update, which is incorrect.

**Correction:** Add an explicit step between Step 3 and Step 4: re-compute the `cos`/`sin` view outside the trace bracket using the new `cur_pos`, ensuring the view's device address points to the position-1 data in the DRAM table. Only then call `execute_trace`. Without this step, the verification protocol describes behavior that cannot occur given how Metal Trace replay works.

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:
1. `source_tensor_stability.md`: Removed false claim that slice re-runs on replay with different offset; added Trace Invariant clarifying that source address is baked at capture time and kwarg buffers are updated in eager mode outside the trace bracket
2. `replay_correctness_verification.md`: Added explicit protocol step for eager kwarg buffer update outside the trace bracket before each execute_trace call

---

## Pass 2

### Issues found: 1

---

**Issue 1:** `what_copy_records.md`, section "Source Address Stability", line 64

**Error:** The "Stable source (correct design)" bullet states:

> "The slice is a view into the stable DRAM tensor and **inherits its device address**. No new buffer is created. The source address is stable and trace-safe."

This is incorrect. A slice view at offset `cur_pos` does NOT inherit a single stable address — each different `cur_pos` produces a different device-address offset into the DRAM table. The view's address changes every step. Saying "the source address is stable" is therefore false for the slice view itself.

The correct mechanism (as described in `source_tensor_stability.md`'s Trace Invariant box and in step 3a of `replay_correctness_verification.md`) is: the slice view's CONTENTS are copied in eager mode into a stable pre-allocated kwarg buffer BEFORE `execute_trace` is called. It is the kwarg buffer's address — not the slice view's address — that is baked into the trace's DMA command at capture time. That kwarg buffer address does not change between steps; only its contents are updated in eager mode.

The "Stable source" bullet in `what_copy_records.md` omits the kwarg buffer entirely and incorrectly attributes stability to the slice view's address. This contradicts domain facts 2 and 3, and contradicts the correct explanation given in `source_tensor_stability.md` and `replay_correctness_verification.md`.

**Correct description:** The slice view does not allocate a new buffer, but its device address offset changes with `cur_pos`. Stability is achieved by the TracedRun kwarg pre-allocation mechanism: the view's contents are copied into a stable pre-allocated kwarg buffer in eager mode before each `execute_trace` call. The trace's DMA command reads from the kwarg buffer's address (fixed at capture time), not from the slice view's address. The "Stable source" bullet should describe this mechanism rather than attributing stability to the view's address.

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. `what_copy_records.md`: Removed false claim that the slice view "inherits its device address" and is therefore stable. Replaced with correct description: slice offset changes with `cur_pos`; stability comes from TracedRun's pre-allocated kwarg buffer whose address is baked into the trace at capture time; the slice happens outside the trace bracket in eager Python and its contents are copied into the stable kwarg buffer before each `execute_trace` call; inside the trace, `cos` refers to the stable-address kwarg buffer.

---

## Pass 3

### Issues found: 0

None. Chapter is correct.

Verification notes per file:

- `what_copy_records.md` — "Stable source (correct design)" bullet correctly states that the slice offset changes with `cur_pos` (not stable), that stability is achieved through the TracedRun kwarg pre-allocation mechanism, that the slice runs in eager Python outside the trace bracket, and that the trace DMA command reads from the stable kwarg buffer address. Consistent with domain facts 2, 3, and 4.

- `source_tensor_stability.md` — Correctly places the slice outside the trace bracket (lines 31–44). The Trace Invariant box correctly states that the source address is baked at capture time, that Python variable updates do not affect the baked DMA source address, and that per-step values flow into the trace by updating the kwarg buffer's contents in eager mode before each `execute_trace` call. The distinction between the slice view (variable offset, produced outside the bracket) and the stable kwarg buffer (fixed address, baked into the trace) is correctly maintained throughout. Consistent with domain facts 1, 3, and 4.

- `replay_correctness_verification.md` — Step 3a explicitly includes the eager kwarg buffer update before `execute_trace`: slices the DRAM table at the current `cur_pos` in eager Python, then copies into the stable pre-allocated kwarg buffer via `ttnn.copy` outside the trace bracket. Steps 4 and 5 both reference and repeat this pattern. The stale-value failure mode section and PCC threshold (> 0.999) are correctly stated per domain facts 6 and 7. Consistent with domain facts 1, 2, 3, and 6.

- `index.md` — Navigational intro only; makes no incorrect claims about trace semantics.
