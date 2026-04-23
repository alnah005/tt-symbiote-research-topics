# Multi-Replay Stability

This file describes the trace replay consistency test that confirms the modified `_maybe_all_gather` with `ttnn.experimental.all_gather_async` and cycling semaphores produces identical outputs across N consecutive trace replays. By the end of this file you will know how to run the stability test, detect deadlocks, and verify that traced and non-traced calls do not corrupt each other's semaphore state.

> **Note:** This test applies only to the Type B2 implementation path (async CCL with cycling semaphores). Type A (synchronous `ttnn.all_gather`, no `synchronize_device`) does not use cycling semaphores and does not require this test — trace replay stability for Type A follows from the standard Metal Trace contract.

---

## Why Stability Matters Beyond First-Replay Correctness

Passing [`functional_correctness.md`](./functional_correctness.md) in non-traced mode and producing a correct first trace replay are necessary but not sufficient conditions. Several failure modes manifest only on replay 2 or later:

- **Stale semaphore skip-through:** If the semaphore reset in steps 10–11 of the [Chapter 6 wrapper checklist](../ch6_implementation/trace_capture_wrapper_changes.md) is missing or applies to the wrong handle, the baked-in `all_gather_async` completion signal reads a non-zero value left by the previous replay. The device-side kernel interprets this as "already complete" and returns immediately, before the all_gather has run on replay N. Output is stale or corrupted.
- **Index aliasing across replays:** If the `TT_CCL` index fields are not restored before each `execute_trace`, the cycling counter selects a different semaphore handle than what is baked in the trace, causing mismatched L1 addresses and a silent no-op or deadlock.
- **Intermittent races under OS scheduling variation:** A race condition that is consistently masked by OS scheduling luck on replay 1 may manifest on replay 3 or 7. Running N≥10 replays with variable inter-replay timing increases the probability of exposing such races.

---

## Trace Replay Consistency Test

### Setup

1. Implement the changes from [Chapter 6](../ch6_implementation/index.md), including the wrapper checklist from [`trace_capture_wrapper_changes.md`](../ch6_implementation/trace_capture_wrapper_changes.md).
2. Run the functional correctness test ([`functional_correctness.md`](./functional_correctness.md)) and confirm PCC > 0.999 in non-traced mode before proceeding.
3. Instantiate `TTNNQwen3FullAttention` (or the full hybrid layer) for decode.

### Capture

4. Run the compile (warm-up) forward pass in non-traced mode to populate the program cache.
5. Execute the pre-capture checklist (steps 1–4 in the Chapter 6 wrapper checklist).
6. Call `ttnn.begin_trace_capture(...)`.
7. Run one decode forward pass of `_maybe_all_gather` (as part of the full attention forward).
8. Call `ttnn.end_trace_capture(...)`.
9. Execute the post-capture checklist (steps 6–7 in the Chapter 6 wrapper checklist).

### N-Replay Loop

10. For each replay `i` in `range(N)` where `N >= 10`:

    a. Execute the pre-replay checklist (steps 8–11 in the Chapter 6 wrapper checklist).
    b. Call `ttnn.execute_trace(...)`.
    c. Transfer the output tensor to CPU: `output_cpu_i = ttnn.to_torch(output_tensor)`.
    d. If `i == 0`: store `output_cpu_0` as the reference replay output.
    e. If `i > 0`: compute `pcc_value, _ = comp_pcc(output_cpu_0, output_cpu_i, pcc=0.9999)`.
    f. Assert `pcc_value` is True. If not, fail with: `"Trace replay {i} output differs from replay 0: possible semaphore stale-skip or aliasing."`.

**Acceptance criterion:** All N replays produce outputs with PCC > 0.9999 relative to replay 0.

> **Note:** The threshold of 0.9999 (four nines) is tighter than the functional correctness threshold (0.999) because trace replay should be deterministic. Any value below 0.9999 across replays indicates a non-deterministic element in the replay path, which in a fully deterministic Metal Trace should not exist. If values are consistently between 0.999 and 0.9999, investigate whether any Python-side tensor modification or a non-traced post-processing step introduces the deviation.

---

## Deadlock Detection

If a semaphore is not correctly reset before `execute_trace`, the `all_gather_async` kernel's device-side wait for the completion semaphore will never fire, and `execute_trace` will hang indefinitely.

Wrap each `execute_trace` call with a timeout:

```python
import signal

class TraceReplayTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TraceReplayTimeoutError("execute_trace timed out — possible semaphore deadlock")

TRACE_REPLAY_TIMEOUT_S = 10  # 10 seconds; adjust for large models

for i in range(N):
    # Pre-replay semaphore reset checklist (Chapter 6, steps 8–11)
    restore_semaphore_state(tt_ccl, cluster_axis, ag_idx_snapshot, barrier_idx_snapshot)
    reset_semaphore_values(tt_ccl, cluster_axis, ag_idx_snapshot, barrier_idx_snapshot)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRACE_REPLAY_TIMEOUT_S)
    try:
        ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        signal.alarm(0)  # Cancel alarm on success
    except TraceReplayTimeoutError:
        raise AssertionError(
            f"execute_trace deadlocked on replay {i}. "
            f"Check that semaphore handles were reset before execute_trace. "
            f"ag_idx_snapshot={ag_idx_snapshot}, barrier_idx_snapshot={barrier_idx_snapshot}, "
            f"cluster_axis={cluster_axis}."
        ) from None
```

> **Note:** `signal.SIGALRM` is available on Linux and macOS but not on Windows. On Windows, use a threading-based timeout instead.

A deadlock on replay `i=0` indicates a problem with the pre-capture semaphore reset (step 3 in the Chapter 6 pre-capture checklist). A deadlock on replay `i=1` or later indicates a problem with the pre-replay reset (steps 10–11 in the Chapter 6 pre-replay checklist) — the semaphore was not reset between replay 0 and replay 1.

---

## Stress Test: Traced and Non-Traced Interleaving

In production, the attention module is called in two modes within a single session:
- **Prefill:** non-traced, often with larger input tensors (seq_len > 1)
- **Decode:** traced, batch=1

If the cycling semaphore indices advance during non-traced prefill calls, and then the trace wrapper assumes the indices are at the pre-capture snapshot values, the semaphore handle selected for the trace replay will not match the handle baked into the trace.

### Stress Test Procedure

1. Perform a non-traced prefill forward pass (seq_len > 1, no trace bracket). The `_maybe_all_gather` call during prefill advances `ag_semaphores_idx` by 1.
2. Perform a traced decode capture using the wrapper checklist from [Chapter 6](../ch6_implementation/trace_capture_wrapper_changes.md). The snapshot captures the post-prefill index value.
3. Run 5 traced decode replays (using the pre-replay checklist); confirm all PCC > 0.9999.
4. Run another non-traced prefill forward pass. The index advances again.
5. Run 5 more traced decode replays. Confirm all PCC > 0.9999.
6. Repeat steps 4–5 for a total of 5 prefill–decode cycles.

**Acceptance criterion:** All traced decode replays across all cycles produce PCC > 0.9999 relative to the first traced decode replay.

> **Warning:** A common implementation error is to snapshot the index at model initialization time rather than immediately before `begin_trace_capture`. If the snapshot is taken too early (before any non-traced prefill call has run), the snapshot reflects a different index than the one that will be live when the capture begins. Always take the snapshot immediately before `begin_trace_capture`, not at module construction time.

---

## Summary of Acceptance Criteria

| Test | Condition | Threshold |
|---|---|---|
| N-replay consistency (N=10) | PCC(replay_i, replay_0) for all i | > 0.9999 |
| Deadlock detection | execute_trace completes within timeout | 10 s per replay |
| Traced/non-traced interleaving (5 cycles) | PCC(decode_replay, reference) | > 0.9999 |

Passing all three constitutes a complete multi-replay stability validation. Proceed to [`latency_measurement.md`](./latency_measurement.md) only after all three pass.
