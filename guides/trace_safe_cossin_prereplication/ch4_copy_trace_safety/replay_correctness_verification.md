# Replay Correctness Verification

After implementing `ttnn.copy` inside the trace bracket, the most important verification is that cos/sin values are updated correctly on every consecutive replay step. The failure mode to guard against is stale values: if the copy does not execute during replay (because it was accidentally placed outside the trace bracket), replay step N+1 uses the cos/sin baked at capture time rather than the position-(N+1) values. The output matches the reference for step 0 but silently diverges for all subsequent steps. This failure produces no runtime error and no `TT_FATAL` — the only detection is a numerical check against a reference implementation at each step.

---

## Verification Protocol

Run the following sequence against a reference (non-traced) rotary embedding implementation. The reference should compute rotary embeddings using float32 precision on CPU or in an eager-mode device run.

1. **Capture the trace** using cos/sin values for position 0. The compile run and capture run both use `cur_pos = 0`. The trace command buffer now contains a DMA command that copies position-0 cos/sin into `_cos_replicated`, followed by `ttnn.experimental.rotary_embedding`.

2. **Execute the trace once (step 0).** Inside the trace, `ttnn.copy` updates `_cos_replicated` with position-0 cos/sin values before `ttnn.experimental.rotary_embedding` runs. Compute PCC between the trace output and the reference output for position 0. Expect PCC > 0.999 in BF16 vs. float32.

   ```python
   # Step 0 verification:
   # why: confirms that the trace executes correctly on the first replay.
   output_step0 = execute_trace(cur_pos=0)
   reference_step0 = reference_rotary_embedding(hidden_states, cur_pos=0)
   pcc_step0 = compute_pcc(output_step0, reference_step0)
   assert pcc_step0 > 0.999, f"Step 0 PCC {pcc_step0:.4f} below threshold"
   ```

3. **Update the external position counter** so that the next trace execution will use cos/sin for position 1. This update happens outside the trace bracket — it modifies the Python variable that controls which slice of the DRAM table is prepared before the trace runs.

   ```python
   # Update position counter outside the trace bracket:
   # why: the trace does not re-execute Python; cur_pos must be updated
   #      externally so that the eager update step below prepares the correct
   #      position's cos/sin before execute_trace is called.
   cur_pos = 1
   ```

3a. **Update the pre-allocated `cos`/`sin` kwarg buffers in EAGER mode** (outside the trace bracket) before calling `execute_trace`. For a full explanation of why this mechanism works, see [`source_tensor_stability.md`](./source_tensor_stability.md).

   ```python
   # Eager kwarg buffer update — OUTSIDE the trace bracket, before execute_trace:
   # why: Metal Trace replay does NOT re-execute Python; the only way to change
   #      what the trace reads is to update the contents of the stable pre-allocated
   #      buffer whose address was baked into the trace's DMA command at capture time.
   #      The buffer's device address is unchanged; only its contents are updated here.
   cos_cur = cos_table[:, :, cur_pos:cur_pos + 1, :]   # eager slice — runs in Python
   sin_cur = sin_table[:, :, cur_pos:cur_pos + 1, :]
   # TracedRun (or equivalent) copies cos_cur/sin_cur contents into the
   # stable pre-allocated kwarg buffers before issuing execute_trace:
   ttnn.copy(cos_cur, preallocated_cos_kwarg)  # eager copy — NOT inside trace
   ttnn.copy(sin_cur, preallocated_sin_kwarg)  # eager copy — NOT inside trace
   ```

4. **Execute the trace a second time (step 1).** Inside the trace, `ttnn.copy(cos, self._cos_replicated)` reads from the stable pre-allocated `cos` kwarg buffer — whose contents were updated in step 3a to hold position-1 values — and writes into `_cos_replicated`. This is correct because the kwarg buffer's device address (baked into the trace) is unchanged, but its contents now reflect position 1. Verify output against reference for position 1.

   ```python
   # Step 1 verification:
   # why: this is the critical check — step 3a (eager kwarg buffer update) is what
   #      makes position-1 cos/sin flow into the trace; if that step is omitted,
   #      _cos_replicated still holds position-0 values and the output will be wrong.
   output_step1 = execute_trace(cur_pos=1)
   reference_step1 = reference_rotary_embedding(hidden_states, cur_pos=1)
   pcc_step1 = compute_pcc(output_step1, reference_step1)
   assert pcc_step1 > 0.999, f"Step 1 PCC {pcc_step1:.4f} below threshold"
   ```

5. **Repeat for positions 2 through 7** (at minimum). For each step, perform the eager kwarg buffer update (step 3a pattern) before calling `execute_trace`. Running at least 8 consecutive steps catches off-by-one errors in position indexing and ensures that the eager update correctly refreshes the kwarg buffer's contents on every replay. Each step should produce PCC > 0.999 against the corresponding float32 reference.

   ```python
   # Steps 2-7 verification loop:
   for cur_pos in range(2, 8):
       # Eager kwarg buffer update outside the trace bracket:
       cos_cur = cos_table[:, :, cur_pos:cur_pos + 1, :]
       sin_cur = sin_table[:, :, cur_pos:cur_pos + 1, :]
       ttnn.copy(cos_cur, preallocated_cos_kwarg)
       ttnn.copy(sin_cur, preallocated_sin_kwarg)
       # Now execute the trace — DMA reads from the updated kwarg buffer:
       output = execute_trace(cur_pos=cur_pos)
       reference = reference_rotary_embedding(hidden_states, cur_pos=cur_pos)
       pcc = compute_pcc(output, reference)
       assert pcc > 0.999, f"Step {cur_pos} PCC {pcc:.4f} below threshold"
   ```

---

## Failure Mode: Copy Outside the Trace Bracket

> **Warning:** If `ttnn.copy` is placed OUTSIDE the trace bracket (before `begin_trace_capture`), it will NOT execute during replay. Replay step N+1 will use the cos/sin values baked at capture time (position 0), producing numerically wrong outputs for all steps after 0. This failure is silent — no `TT_FATAL` fires. The only detection is a PCC check against the reference.

The following placement is INCORRECT:

```python
# INCORRECT — ttnn.copy outside the trace bracket:
# why: the copy runs once during setup but is NOT recorded in the trace;
#      replay re-issues only the commands inside begin/end_trace_capture;
#      _cos_replicated keeps the position-0 values for every replay step.
ttnn.copy(cos, self._cos_replicated)        # outside bracket — NOT in trace ✗
ttnn.begin_trace_capture(mesh_device, trace_id)
output = model.forward(hidden_states, cos, sin, ...)
ttnn.end_trace_capture(mesh_device, trace_id)
```

The diagnostic signature of this failure: step 0 PCC is > 0.999 (because position-0 cos/sin were written before capture and are correct for step 0); step 1 PCC drops significantly (position-1 hidden states are rotated by position-0 cos/sin); the PCC drop repeats for every step beyond 0.

---

## PCC Threshold

The acceptance threshold for all steps is **PCC > 0.999** in BF16 against a float32 reference. BF16 introduces small rounding errors relative to float32; a PCC above 0.999 confirms that the errors are within normal numerical tolerance and that the cos/sin update is working correctly. A PCC below 0.99 on any step after step 0 is a strong indicator of the stale-value failure mode described above.

---

## Connection to Chapter 6

This verification protocol is implemented as Test 3 in [`../ch6_integration_and_testing/test_plan.md`](../ch6_integration_and_testing/test_plan.md).
