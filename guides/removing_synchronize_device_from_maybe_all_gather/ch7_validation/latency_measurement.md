# Latency Measurement

This file describes the before/after latency measurement procedure for confirming that removing `ttnn.synchronize_device()` from `_maybe_all_gather` delivers the throughput improvement predicted in [Chapter 5](../ch5_latency_analysis/index.md), and for isolating the two independent contributions — `synchronize_device` removal and trace dispatch overhead elimination — once the full trace is operational. By the end of this file you will have a complete measurement plan with attribution methodology and a checklist for recording results.

---

## Prerequisites

Run [`functional_correctness.md`](./functional_correctness.md) and [`multi_replay_stability.md`](./multi_replay_stability.md) first. Latency measurement is only meaningful for a validated implementation. A latency improvement obtained from a numerically incorrect implementation is not a valid result.

---

## Measurement 1: synchronize_device Removal Benefit (Non-Traced Mode)

This measurement isolates the cost of `ttnn.synchronize_device()` itself from all other effects. Run both configurations in non-traced mode, so there is no trace capture or replay overhead to confound the comparison.

**Configuration A (baseline):** Original implementation with synchronous `ttnn.all_gather` + `ttnn.synchronize_device()`.

**Configuration B (modified):** Modified implementation — either Type A (no `synchronize_device`, synchronous `ttnn.all_gather`) or Type B2 (no `synchronize_device`, `all_gather_async` with cycling semaphores). No trace bracket in either case.

### Procedure

1. Run 30 decode forward steps in a loop, discarding the first 5 as warm-up. Measure the wall-clock duration of each full decode forward pass (or the per-layer forward pass if full-stack isolation is needed).

2. Record the median and 95th-percentile per-step latency for each configuration.

3. Compute the improvement:

   ```
   delta_latency = median_A - median_B
   improvement_percent = (delta_latency / median_A) × 100
   ```

4. Compare `delta_latency` against the per-step estimate from [Chapter 5, `throughput_improvement_estimate.md`](../ch5_latency_analysis/throughput_improvement_estimate.md):

   ```
   expected_delta = K × T_sync_measured
   ```

   where K is the total number of `_maybe_all_gather` calls per step and `T_sync_measured` is the median value from the bracket measurement in [Chapter 5, `measuring_the_cost.md`](../ch5_latency_analysis/measuring_the_cost.md).

**Acceptance criterion:** `delta_latency` matches `expected_delta` within ±50% (accounting for OS scheduling noise and measurement methodology differences). A value far outside this range indicates either a measurement error or a confounding change (such as a different code path being activated by the modification).

---

## Attribution Confirmation: Isolating synchronize_device as the Cause

If Configuration B differs from Configuration A in more than just the `synchronize_device` deletion (for example, Type B2 also changes the all_gather variant), it is important to confirm that the latency improvement is attributable to the `synchronize_device` removal and not to a change in all_gather behavior.

### Procedure

3a. Create **Configuration C**: apply the Type B2 structural changes (wire in `TT_CCL`, replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, add cycling semaphores) but **temporarily re-insert** `ttnn.synchronize_device(self.mesh_device)` after the `all_gather_async` call:

```python
x = ttnn.experimental.all_gather_async(
    x, ..., multi_device_global_semaphore=..., barrier_semaphore=...,
)
ttnn.synchronize_device(self.mesh_device)  # temporarily re-inserted for attribution test
```

3b. Measure per-step latency for Configuration C.

3c. Confirm that `median_C ≈ median_A` (within ±20%). This verifies that the all_gather variant change itself does not explain the latency improvement.

3d. Remove the re-inserted `synchronize_device` (restoring Configuration B) and confirm `median_B < median_A`.

If `median_C` differs significantly from `median_A`, investigate whether `all_gather_async` has different execution characteristics (e.g., different kernel dispatch overhead or different PCIe scheduling behavior) that contribute independently to the measured improvement.

---

## Measurement 2: Full-Trace Latency (After Trace Enablement)

Once the modified implementation passes the multi-replay stability test and the full attention stack is captured under Metal Trace, measure the combined improvement from both contributions:

1. `synchronize_device` removal (measured in Measurement 1 above).
2. Trace dispatch overhead elimination (all per-op Python dispatch overhead folded into a single `execute_trace` call).

**Configuration D:** Original implementation in non-traced mode (Configuration A from Measurement 1). This is the baseline.

**Configuration E:** Modified implementation with `enable_trace=True` (trace capture active, `execute_trace` for decode steps).

### Procedure

Run 30 traced decode steps (N≥2 for replay — skip the capture step). Record per-step latency. Compute:

```
total_improvement = median_D - median_E
sync_removal_contribution = delta_latency  (from Measurement 1)
trace_dispatch_contribution = total_improvement - sync_removal_contribution
```

Present both contributions separately in the measurement record. The trace dispatch contribution is typically 2–5× larger than the `synchronize_device` removal contribution for deep stacks.

> **Note:** Use `METAL TRACE REPLAY SESSION ID >= 2` rows in the Tracy ops CSV to identify per-decode-step trace replay durations. These rows correspond to actual trace replays (not the capture or compile runs) and give the most accurate per-step latency.

---

## Tracy Measurement Procedure

For precise per-step latency with device-side correlation, use Tracy:

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_TRACE_TRACKING=1

python3 -m tracy -r -- pytest tests/test_qwen3_attention_decode.py::test_decode_step_latency \
    -k "batch1 and T3K and traced" \
    --no-header -rN
```

In the Tracy ops CSV, look for:
- `METAL TRACE REPLAY SESSION ID >= 2` rows: the per-replay duration is the traced decode step latency.
- Rows corresponding to the `all_gather_async` kernel within each replay: confirms that the all_gather runs at the correct position in the replay timeline.
- The gap between consecutive trace replay rows: should be close to zero if the host is issuing `execute_trace` calls back-to-back without intervening Python work.

For the Python wall-clock measurement (simpler, less precise):

```python
import time

latencies = []
for step in range(35):
    t0 = time.perf_counter()
    output = model.decode_step(input_ids, ...)
    ttnn.synchronize_device(mesh_device)   # only for timing measurement; not in the traced path
    t1 = time.perf_counter()
    if step >= 5:  # skip warm-up
        latencies.append((t1 - t0) * 1000.0)

import statistics
print(f"Median per-step latency: {statistics.median(latencies):.2f} ms")
print(f"P95 per-step latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms")
```

> **Warning:** When measuring latency in traced mode, do not insert `ttnn.synchronize_device()` between `execute_trace` calls for timing purposes — this re-introduces a blocking wait that inflates the measurement. Instead, add a single `ttnn.synchronize_device()` call outside the timed loop (or after all decode steps complete) to drain the device before the program exits, and use Tracy's device timeline for per-step precision.

---

## Recording Results

Fill in the following table after completing all measurements:

```
## Latency Measurement Results (TODO: fill in)

Date: TODO
T3K firmware: TODO
Model: TTNNQwen3 hybrid decoder, N=? layers
Batch size: 1
Dtype: TODO (e.g., bfloat16)

| Configuration | Mode | Median latency | P95 latency | Notes |
|---|---|---|---|---|
| A: original (all_gather + synchronize_device) | non-traced | TODO ms | TODO ms | baseline |
| B: modified (no synchronize_device)           | non-traced | TODO ms | TODO ms | Measurement 1 |
| C: modified + re-inserted synchronize_device  | non-traced | TODO ms | TODO ms | attribution check |
| E: modified (no synchronize_device)           | traced     | TODO ms | TODO ms | Measurement 2 |

Derived values:
  delta_latency (A → B): TODO ms  (expected: K × T_sync_measured = TODO ms)
  total_improvement (A → E): TODO ms
  sync_removal_contribution: TODO ms  (= A - B)
  trace_dispatch_contribution: TODO ms  (= total - sync_removal)
  throughput improvement (A → E): TODO %
```

These values complete the validation loop: they confirm the latency model from [Chapter 5](../ch5_latency_analysis/throughput_improvement_estimate.md) and document the final performance state of the implementation for future reference.
