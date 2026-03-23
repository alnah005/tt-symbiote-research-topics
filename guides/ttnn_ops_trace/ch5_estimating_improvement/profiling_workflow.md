# Profiling Workflow

The preceding files in this chapter covered the concepts and formulas; this file is the execution guide. It walks you through the complete end-to-end workflow: instrument your model, capture a baseline profile, implement the trace, validate that replay produces numerically identical outputs, benchmark the traced version, compare results to the baseline, and set up regression tests that detect when a code change invalidates the captured trace. Every command, environment variable, and code pattern you need is included in-line.

---

## Workflow Overview

The six stages are sequential. Do not skip the baseline or numerical validation stages — they are the only evidence that the trace implementation is both correct and faster.

```
Stage 1: Instrument model
         │
         ▼
Stage 2: Run baseline (untraced)
         │  Outputs: avg step latency (ms), per-op dispatch CSV
         ▼
Stage 3: Implement trace capture and replay
         │
         ▼
Stage 4: Validate trace replay (numerical correctness)
         │  Outputs: max absolute difference, max relative difference
         ▼
Stage 5: Benchmark traced version and compare
         │  Outputs: speedup ratio, dispatch overhead eliminated
         ▼
Stage 6: Regression testing
         │  Outputs: CI test that detects trace invalidation
```

---

## Stage 1: Instrument the Model

### Add profiling hooks to your decode step

Before running any profiling, ensure your decode script has the following structure. The warm-up block is mandatory — cold-path kernel selection costs (described in Chapter 1) inflate dispatch times on the first few calls and must not appear in your baseline measurements.

```python
import ttnn
import torch
import time
import os

def run_decode_loop(device, model, input_tensor, num_steps: int, warmup_steps: int = 5):
    """
    Run the decode loop with warm-up and return per-step timing.

    Returns a list of step latencies in microseconds.
    """
    latencies_us = []

    # Warm-up: pay the cold-path kernel selection cost before measuring.
    for _ in range(warmup_steps):
        _ = model.decode_step(input_tensor)
    ttnn.synchronize_device(device)

    # Measurement loop.
    for _ in range(num_steps):
        t0 = time.perf_counter_ns()
        _ = model.decode_step(input_tensor)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1_000.0)

    return latencies_us
```

> **Note:** Calling `ttnn.synchronize_device(device)` inside the measurement loop ensures each latency sample captures a complete host-to-device round trip: the time from starting the first dispatch to the time the last kernel on the device finished. This is the correct denominator for the speedup model in `estimating_trace_speedup.md`. For production throughput benchmarking you would remove the per-step sync, but for latency measurement you must keep it.

---

## Stage 2: Run the Baseline

### Environment variables for a profiled baseline run

Set the same environment variables described in [`measuring_dispatch_overhead.md`](./measuring_dispatch_overhead.md) (Enabling Profiling section).

```bash
# Optional: direct the CSV output to a specific path.
export TT_METAL_PROFILER_OUTPUT="./baseline_profile.csv"
```

### Baseline benchmark script

```python
import ttnn
import torch
import time
import statistics

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

# --- Load model and allocate tensors here ---
# model = load_your_model(device)
# input_tensor = allocate_input_tensor(device)

NUM_STEPS = 50
WARMUP = 5

latencies_us = run_decode_loop(device, model, input_tensor,
                               num_steps=NUM_STEPS, warmup_steps=WARMUP)

mean_us = statistics.mean(latencies_us)
p50_us  = statistics.median(latencies_us)
p99_us  = sorted(latencies_us)[int(0.99 * len(latencies_us))]

print(f"Baseline (untraced) step latency")
print(f"  Mean:  {mean_us:>8.1f} us  ({mean_us / 1000:.3f} ms)")
print(f"  P50:   {p50_us:>8.1f} us  ({p50_us  / 1000:.3f} ms)")
print(f"  P99:   {p99_us:>8.1f} us  ({p99_us  / 1000:.3f} ms)")

ttnn.close_device(device)
```

Record the mean and P50 latency values. These are the `total_step_time` inputs to the speedup formula.

Parse `baseline_profile.csv` using the snippet from `measuring_dispatch_overhead.md` to obtain `dispatch_overhead`. At this point you have both inputs to the speedup formula and can compute the predicted speedup before writing any trace code.

---

## Stage 3: Implement Trace Capture and Replay

Once the predicted speedup passes the checklist from `estimating_trace_speedup.md`, implement the trace. The canonical code pattern from Chapter 3 (`trace_api.md`) is reproduced here with profiling-specific additions.

```python
import ttnn
import torch
import time
import statistics

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

# --- Load model and allocate tensors here ---
# The input_tensor must be allocated at a fixed device address for the
# lifetime of the trace. Allocate it before capture and never reallocate.

# Warm-up before capture: ensures the kernel selection warm-path is active
# so that the capture run does not pay cold-path costs.
for _ in range(5):
    _ = model.decode_step(input_tensor)
ttnn.synchronize_device(device)

# ── Capture phase ─────────────────────────────────────────────────────────────

ttnn.synchronize_device(device)
ttnn.begin_trace_capture(device, cq_id=0)
output_tensor = model.decode_step(input_tensor)
trace_id = ttnn.end_trace_capture(device, cq_id=0)

# Save a reference output for numerical validation (Stage 4).
ttnn.synchronize_device(device)
capture_output_ref = ttnn.to_torch(output_tensor).clone()

# ── Replay benchmark ──────────────────────────────────────────────────────────

NUM_STEPS = 50
latencies_traced_us = []

for _ in range(NUM_STEPS):
    t0 = time.perf_counter_ns()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter_ns()
    latencies_traced_us.append((t1 - t0) / 1_000.0)

mean_traced_us = statistics.mean(latencies_traced_us)
p50_traced_us  = statistics.median(latencies_traced_us)

print(f"Traced step latency")
print(f"  Mean:  {mean_traced_us:>8.1f} us  ({mean_traced_us / 1000:.3f} ms)")
print(f"  P50:   {p50_traced_us:>8.1f} us  ({p50_traced_us  / 1000:.3f} ms)")

# ── Cleanup ────────────────────────────────────────────────────────────────────

ttnn.release_trace(device, trace_id)
ttnn.close_device(device)
```

---

## Stage 4: Validate Trace Replay Produces Numerically Identical Outputs

Correctness must be verified before speed is compared. A traced decode step that produces wrong outputs silently is worse than no optimization.

The validation strategy is to compare the capture-phase output (which is a live device execution identical to the untraced baseline) to the replay output after one or more replay steps. If the replay changes the output in any way, the trace is operating on stale or misaligned buffer state.

```python
# Run one replay step with the same input that was used during capture.
ttnn.copy_host_to_device_tensor(
    ttnn.from_torch(capture_input_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
    input_tensor,
)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
replay_output = ttnn.to_torch(output_tensor)

# Compare replay output to the reference captured during Stage 3.
abs_diff = (replay_output - capture_output_ref).abs()
max_abs_diff = abs_diff.max().item()
max_rel_diff = (abs_diff / (capture_output_ref.abs() + 1e-8)).max().item()

print(f"Numerical validation")
print(f"  Max absolute difference: {max_abs_diff:.6f}")
print(f"  Max relative difference: {max_rel_diff:.6f}")

# Tolerance thresholds for bfloat16 arithmetic.
# bfloat16 has ~2 decimal digits of precision; expect small rounding differences
# if your model uses intermediate float32 accumulation but stores bfloat16.
ABS_TOL = 1e-2
REL_TOL = 1e-2

if max_abs_diff > ABS_TOL or max_rel_diff > REL_TOL:
    raise AssertionError(
        f"Trace replay output differs from capture output: "
        f"max_abs={max_abs_diff:.6f}, max_rel={max_rel_diff:.6f}. "
        f"Check that input tensor addresses are stable and no ops changed "
        f"tensor shapes or buffer bindings since capture."
    )
else:
    print("  PASS: Replay output matches capture output within tolerance.")
```

> **Warning:** The tolerances above (1e-2 absolute and relative) are appropriate for bfloat16 models where the trace records bfloat16 kernel arguments. If your model uses float32 kernels throughout, tighten the tolerance to 1e-5. If you see differences larger than the expected rounding error, the trace is either operating on wrong buffer addresses or a tensor was reallocated after capture — review the address-fixity constraint described in Chapter 3 (`trace_internals.md`).

---

## Stage 5: Benchmark and Compare

With both the baseline timings (Stage 2) and the traced timings (Stage 3) in hand, compute and report the measured speedup:

```python
# Values from Stage 2 and Stage 3 measurements:
baseline_mean_us = 2400.0   # replace with your Stage 2 measurement
traced_mean_us   = 1810.0   # replace with your Stage 3 measurement

measured_speedup = baseline_mean_us / traced_mean_us
latency_saved_us = baseline_mean_us - traced_mean_us
latency_saved_pct = 100.0 * latency_saved_us / baseline_mean_us

print(f"Speedup comparison")
print(f"  Baseline mean:  {baseline_mean_us:>8.1f} us")
print(f"  Traced mean:    {traced_mean_us:>8.1f} us")
print(f"  Speedup:        {measured_speedup:.3f}x")
print(f"  Latency saved:  {latency_saved_us:.1f} us  ({latency_saved_pct:.1f}%)")
```

Compare `measured_speedup` to the predicted speedup from the formula in `estimating_trace_speedup.md`. The measured value should be within 10–20% of the prediction. A measured speedup significantly below prediction typically indicates that:

- Some dispatch overhead was already hidden by async pipelining (the formula is an upper bound; async mode may have already eliminated some of it).
- Synchronization points inside or near the traced region are limiting the achievable latency.
- Python overhead outside the trace boundary has become the new bottleneck after dispatch overhead is removed.

A measured speedup significantly above prediction is a sign that your `dispatch_overhead` measurement was lower than the true overhead (for example, warm-up was insufficient and cold-path costs inflated the first few measurement samples).

---

## Stage 6: Regression Testing

A captured trace is a snapshot of the model's exact op sequence, kernel arguments, and buffer bindings at the time of capture. If any of the following changes, the trace becomes invalid and will produce wrong outputs on replay:

- A new op is added or removed from the decode step.
- An op's kernel arguments change (e.g., a scaling factor that was previously constant becomes step-dependent).
- A tensor is reallocated at a different device address (e.g., due to a change in model initialization order).
- The device is opened with a different configuration (e.g., different number of CQs).

Regression tests must detect trace invalidation before bad outputs reach production.

### Regression test structure

```python
import pytest
import ttnn
import torch

@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, num_hw_cqs=1)
    dev.enable_async(True)
    yield dev
    ttnn.close_device(dev)

def test_trace_replay_matches_live(device):
    """
    Verify that one trace replay step produces numerically identical output
    to one live (untraced) dispatch step, given the same input.
    """
    # --- Load model and allocate tensors ---
    model = load_model(device)
    input_data = torch.randn(1, 1, 1, 512, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device)

    # Live (untraced) reference execution.
    live_output_tensor = model.decode_step(input_tensor)
    ttnn.synchronize_device(device)
    live_output = ttnn.to_torch(live_output_tensor)

    # Trace capture (same input).
    ttnn.synchronize_device(device)
    ttnn.begin_trace_capture(device, cq_id=0)
    traced_output_tensor = model.decode_step(input_tensor)
    trace_id = ttnn.end_trace_capture(device, cq_id=0)
    ttnn.synchronize_device(device)
    capture_output = ttnn.to_torch(traced_output_tensor)

    # Replay.
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    replay_output = ttnn.to_torch(traced_output_tensor)

    # Assert correctness.
    torch.testing.assert_close(
        replay_output, live_output,
        atol=1e-2, rtol=1e-2,
        msg="Trace replay output does not match live dispatch output."
    )
    torch.testing.assert_close(
        capture_output, live_output,
        atol=1e-2, rtol=1e-2,
        msg="Capture output does not match live dispatch output."
    )

    # Cleanup.
    ttnn.release_trace(device, trace_id)


def test_trace_speedup_regression(device):
    """
    Verify that the traced decode step is faster than the untraced baseline.
    Fail if the speedup drops below a minimum threshold, which would indicate
    that the trace is no longer effectively eliminating dispatch overhead.
    """
    import time, statistics

    model = load_model(device)
    input_data = torch.randn(1, 1, 1, 512, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=device)

    WARMUP = 5
    NUM_STEPS = 30
    MIN_SPEEDUP = 1.05  # fail if speedup drops below 5%

    # Warm-up.
    for _ in range(WARMUP):
        model.decode_step(input_tensor)
    ttnn.synchronize_device(device)

    # Baseline timing.
    baseline_times = []
    for _ in range(NUM_STEPS):
        t0 = time.perf_counter_ns()
        model.decode_step(input_tensor)
        ttnn.synchronize_device(device)
        baseline_times.append((time.perf_counter_ns() - t0) / 1_000.0)

    # Trace capture.
    ttnn.synchronize_device(device)
    ttnn.begin_trace_capture(device, cq_id=0)
    model.decode_step(input_tensor)
    trace_id = ttnn.end_trace_capture(device, cq_id=0)

    # Traced timing.
    traced_times = []
    for _ in range(NUM_STEPS):
        t0 = time.perf_counter_ns()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        traced_times.append((time.perf_counter_ns() - t0) / 1_000.0)

    speedup = statistics.mean(baseline_times) / statistics.mean(traced_times)
    assert speedup >= MIN_SPEEDUP, (
        f"Trace speedup {speedup:.3f}x is below minimum threshold {MIN_SPEEDUP}x. "
        f"Check that the decode step op sequence matches the captured trace."
    )

    ttnn.release_trace(device, trace_id)
```

### When to re-capture

Run these tests in CI on every commit that touches the decode step, tensor allocation, or device initialization.

> **Note:** It is acceptable to commit code that intentionally invalidates the captured trace as long as you also update the capture in the same commit. The regression test is a safety net against accidental invalidation — model changes that were not intended to affect the traced region but did so anyway (e.g., a refactoring that reordered tensor allocations and changed device addresses).

---

**Next:** [Chapter 6 — Putting It All Together](../ch6_reference_implementation/index.md)
