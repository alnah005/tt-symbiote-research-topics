# Metal Trace Integration Checklist

This file provides a step-by-step checklist for integrating the on-device DeltaNet decode into a Metal Trace capture/replay loop. Metal Trace (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`) records the sequence of device commands issued during capture and replays them without re-traversing Python. For this replay to be correct across multiple decode steps, every tensor that changes between steps (the state `S`, the KV cache) must be updated in-place via fixed DRAM buffer addresses — no host crossings, no new tensor allocations, no shape changes inside the trace bracket.

**Prerequisites:** Tasks 1–5 from `task_list_and_priority.md` must be complete and passing correctness tests (PCC > 0.999 per step, as defined in Ch6 `pcc_accuracy_thresholds.md`). Do not attempt trace integration until the non-traced on-device implementation is validated.

---

## Step 1 — Pre-allocate state tensors as persistent on-device buffers

Before entering any trace capture bracket, allocate all state tensors (`recurrent_states` for all 30 DeltaNet layers, `conv_states` for all 30 layers) as on-device `ttnn.Tensor` objects using `ttnn.zeros` or `ttnn.allocate_tensor_on_device`. This must happen during the model setup (warm-up) phase.

```python
# During model setup — outside any trace bracket
for layer_idx in range(num_deltanet_layers):
    recurrent_states[layer_idx] = ttnn.zeros(
        shape=[1, H, d_k, d_v],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

The buffer handles (DRAM addresses) returned by this allocation persist across trace replays. The state is initialized to zeros; the first decode step will produce a meaningful `S_1` from an all-zero initial state, which is correct (initial state is zero by convention for a new sequence).

**Checkpoint:** Confirm that `type(recurrent_states[0])` is `ttnn.Tensor` (not `torch.Tensor`) before proceeding to Step 2.

---

## Step 2 — Confirm no host-tensor creation inside the trace bracket

The trace capture bracket must not contain any call to `ttnn.from_torch`, `ttnn.to_torch`, `torch.tensor`, or any Python operation that creates a new `torch.Tensor` from device data or vice versa. Calls of this type will either raise an error during trace capture or silently produce incorrect results during trace replay (because the captured command references a stale address).

Instrument the forward pass with a guard during initial trace testing:

```python
_IN_TRACE_CAPTURE = False

original_from_torch = ttnn.from_torch
def guarded_from_torch(*args, **kwargs):
    if _IN_TRACE_CAPTURE:
        raise RuntimeError("ttnn.from_torch called inside trace bracket — host crossing detected")
    return original_from_torch(*args, **kwargs)
ttnn.from_torch = guarded_from_torch

# Identical guard for ttnn.to_torch
```

Set `_IN_TRACE_CAPTURE = True` before `ttnn.begin_trace_capture` and `False` after `ttnn.end_trace_capture`. Remove the guards after trace debugging is complete.

**Checkpoint:** One full forward pass of the DeltaNet layers inside the guarded bracket completes without raising a `RuntimeError`.

---

## Step 3 — Verify in-place state update is trace-compatible

The DeltaNet decode writes the updated state `S_new` back into the persistent pre-allocated DRAM buffer after each step. This in-place write uses `ttnn.copy` or `ttnn.assign`:

```python
# Inside the DeltaNet forward (inside the trace bracket)
S_new = _compute_deltanet_step(S_prev, q_tilde, k_tilde, v, g_t, beta_t)
ttnn.copy(src=S_new, dst=recurrent_states[layer_idx])
```

In-place writes to pre-allocated DRAM buffers are Metal Trace compatible because:
1. The destination buffer address is fixed at allocation time and does not change across trace replays.
2. `ttnn.copy` is implemented as a tile-by-tile DMA from the computed result buffer to the destination buffer — a deterministic sequence of NOC writes that the trace engine can replay without modification.
3. After each `ttnn.execute_trace` call, the persistent buffer at `recurrent_states[layer_idx]` contains the updated `S_new` from that decode step. The next `ttnn.execute_trace` call reads this updated value as `S_prev` — providing the correct recurrent state carry-over.

**Checkpoint:** Run two sequential trace executions and verify that the output of the second trace differs from the first (confirming that the state update from the first replay is visible to the second).

---

## Step 4 — Run `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`

With Steps 1–3 verified, run a trace capture of the full decoder stack (all 30 DeltaNet layers and all 10 full-attention layers):

```python
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

# Run one full decode step through the model
output = model.decode_step(input_ids, recurrent_states, conv_states, kv_cache)

ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
```

The trace capture executes the full forward pass once (the "compilation" step) while recording all device commands. This will take longer than a normal forward pass; this is expected.

**Checkpoint:** `ttnn.end_trace_capture` completes without raising any exception. No TTNN assertion errors, no device hang, no Python exception during the trace compilation step.

---

## Step 5 — Execute trace for 10 decode steps and verify correctness

Run `ttnn.execute_trace` for 10 consecutive decode steps. After each step, compare the trace output against the non-traced TTNN reference (run with the same inputs, starting from the same initial state):

```python
for step in range(10):
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    output_traced = ttnn.to_torch(output_buffer)    # read output from pre-allocated output buffer

    # Compare against non-traced reference
    output_ref = model.decode_step_nontrace(input_ids[step], ...)
    pcc = pearson_correlation(output_traced.flatten(), output_ref.flatten())
    assert pcc > 0.99, f"Step {step}: output PCC {pcc:.6f} < 0.99"
```

**Checkpoint:** All 10 steps pass PCC > 0.99 for output logits and PCC > 0.999 for per-step state `S`. (The 0.99 logit threshold reflects BF16 rounding through the full attention and MLP layers; the tighter 0.999 threshold applies to DeltaNet-specific `S_new` and `o_t` outputs, as defined in Ch6 `pcc_accuracy_thresholds.md`.)

---

## Step 6 — Run 1000-step decode loop and verify state does not diverge

Run a 1000-step decode loop with trace. After each step, compute the L2 norm of the difference between the traced state and the non-traced reference state:

```python
for step in range(1000):
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    S_traced = ttnn.to_torch(recurrent_states[0])   # layer 0 as representative
    l2_diff = torch.norm(S_traced - S_ref[step]).item()
    l2_history.append(l2_diff)

# Assert L2 norm does not grow monotonically (no unbounded divergence)
assert l2_history[-1] < 10 * l2_history[0], "State divergence detected over 1000 steps"
```

The DeltaNet decay gate `g_t < 1` bounds state divergence (see Ch6 `pcc_accuracy_thresholds.md`, Section 3). Any monotonically growing L2 norm indicates a bug in the in-place state update (e.g., the trace is writing to the wrong buffer address, or the state is not being updated between replays).

**Checkpoint:** L2 norm of state difference remains bounded (does not grow monotonically with step count) over 1000 steps.

---

## Step 7 — Profile the traced decode loop with Tracy

Enable device profiling and capture a Tracy timeline:

```bash
TT_METAL_DEVICE_PROFILER=1 python run_decode.py --use-trace --steps 100
```

Inspect the Tracy timeline to verify:
1. DeltaNet layers no longer appear as CPU-side execution gaps (gaps in the device timeline where the host CPU is running the `recurrent_gated_delta_rule` fallback).
2. DeltaNet layer ops appear as contiguous device-side kernel launches on the timeline, fully overlapping with neighboring full-attention layer ops.
3. The total DeltaNet contribution to per-step decode latency is consistent with the estimates in Ch6 `on_device_latency_estimate.md` (~1 ms for composed form, ~177 µs for fused kernel form).

**Checkpoint:** No CPU-side gaps corresponding to DeltaNet layers in the Tracy device timeline.

---

## Step 8 — Measure per-step decode latency with and without trace

Run a latency comparison:

```python
# Without trace (on-device composed TTNN form, Task 5)
latency_no_trace = measure_decode_step_latency(model, use_trace=False, steps=1000)

# With trace (ttnn.execute_trace)
latency_trace = measure_decode_step_latency(model, use_trace=True, steps=1000)

print(f"Without trace: {latency_no_trace['p50_us']:.1f} µs p50, {latency_no_trace['p99_us']:.1f} µs p99")
print(f"With trace:    {latency_trace['p50_us']:.1f} µs p50, {latency_trace['p99_us']:.1f} µs p99")
```

Expected: the DeltaNet contribution to per-step latency drops from 9–21 ms (host fallback) to approximately 0.36–1.8 ms (composed TTNN, no trace) to approximately ~0.36 ms (composed TTNN, with trace — Python dispatch overhead eliminated by trace replay; optimistic bound of the analytic range). With the fused kernel (Task 6) and trace, the DeltaNet contribution should be approximately 177 µs.

**Checkpoint:** Measured DeltaNet latency with trace is at least 10× lower than the host fallback baseline recorded in Ch6 `host_roundtrip_latency.md`.

---

> **Warning:** The `synchronize_device` removal (required by the companion topics "trace-safe cos/sin pre-replication in TTNNQwen3FullAttention" and "removing `synchronize_device` from `maybe_all_gather`") must also be complete before an end-to-end trace capture of the entire Qwen3.6-35B-A3B decoder stack (including both DeltaNet and full-attention layers) works correctly. This guide's implementation — covering only the DeltaNet layers — is independent of those companion fixes and can be validated in isolation by running a decoder stack containing only DeltaNet layers during trace testing. Integration with the full decoder stack (DeltaNet + full-attention under a single trace) should occur after all companion prerequisites are resolved.
