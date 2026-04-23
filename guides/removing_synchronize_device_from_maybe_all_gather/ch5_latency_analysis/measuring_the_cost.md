# Measuring the Cost of synchronize_device

This file describes two practical procedures for measuring the host-blocking latency of `ttnn.synchronize_device()` inside `_maybe_all_gather`: a coarse Python wall-clock method that any engineer can run immediately, and a precise Tracy-based method that separates the synchronize cost from surrounding Python overhead. By the end of this file you will have a concrete measurement plan and know where to record the values needed to validate the throughput improvement estimate in [`throughput_improvement_estimate.md`](./throughput_improvement_estimate.md).

---

## Method 1 — Python Wall-Clock Bracket (Quick Approximation)

The fastest way to get an order-of-magnitude estimate is to bracket the `ttnn.synchronize_device()` call with `time.perf_counter()`:

```python
import time

def _maybe_all_gather(self, x, cluster_axis, ...):
    if self.num_devices > 1:
        x = ttnn.all_gather(x, ...)

        t0 = time.perf_counter()
        ttnn.synchronize_device(self.mesh_device)
        t1 = time.perf_counter()

        sync_latency_ms = (t1 - t0) * 1000.0
        print(f"[_maybe_all_gather] synchronize_device latency: {sync_latency_ms:.3f} ms")

    return x
```

Run a decode step in a loop (at least 20 iterations, discarding the first 3 as warm-up) and collect the distribution of `sync_latency_ms` values. Record:

- Minimum (best-case, closest to pure PCIe round-trip)
- Median (representative steady-state cost)
- 95th percentile (captures OS scheduling jitter spikes)

> **Note:** This measurement includes Python function call overhead for `time.perf_counter()` itself (typically 0.1–0.5 µs), which is negligible relative to the expected 100–500 µs synchronize cost. Do not apply any correction for it.

> **Warning:** `time.perf_counter()` measures wall-clock time on the host. If the OS preempts the measuring thread between `t0` and `t1` for an unrelated reason, the measurement will be inflated. Discard outliers more than 3× the median when computing the representative cost.

---

## Method 2 — Tracy-Based Measurement (Precise, with Device Timeline)

Tracy provides a more precise measurement by correlating host-side events with device-side kernel completion timestamps. This method is necessary to separate the PCIe round-trip component from the kernel completion component as described in [`synchronize_device_latency_model.md`](./synchronize_device_latency_model.md).

### Environment Setup

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_TRACE_TRACKING=1
```

Run the decode test:

```bash
python3 -m tracy -r -- pytest tests/test_qwen3_attention_decode.py::test_maybe_all_gather_latency \
    -k "batch1 and T3K" \
    --no-header -rN
```

### What to Look For in Tracy Output

1. Open the Tracy output in the Tracy GUI or export the ops CSV.
2. Search for the `synchronize_device` event row. Tracy records the host-side call as a named zone; the zone duration is the observable `synchronize_device` latency.
3. On the device timeline, identify the last kernel dispatched before the `synchronize_device` call (this should be the `all_gather` kernel or its final sub-kernel). Note its completion timestamp.
4. The gap between the `all_gather` kernel's device-side completion and the Tracy zone start for the first op enqueued *after* `synchronize_device` returns is the net host stall time attributable to the synchronize call.

### Interpreting the Tracy Gap

- If the `all_gather` kernel completes before the host reaches the `synchronize_device` Python call: the Tracy zone duration equals approximately 1× PCIe round-trip latency (expected: 10–30 µs) under preemption-free conditions; OS thread preemption can inflate this to hundreds of µs (up to ~500 µs), yielding the practical 100–500 µs wall-clock range observed in production.
- If the `all_gather` kernel completes *inside* the `synchronize_device` zone: the zone duration includes remaining all_gather execution time. This indicates the host is submitting `synchronize_device` while device work is still in flight — the steady-state condition is worse than the pure PCIe lower bound.

> **Note:** The `METAL TRACE REPLAY SESSION ID >= 2` rows in the ops CSV are relevant only for the full-trace latency measurement in [Chapter 7](../ch7_validation/latency_measurement.md). When measuring `synchronize_device` cost in non-traced mode, ignore those rows and focus on the per-decode-step timeline.

---

## Method 3 — TTNN Op Timer Infrastructure

For a middle ground between Python wall-clock and full Tracy, set:

```bash
export TT_METAL_PROFILER_SYNC=1
```

This causes the TTNN runtime to emit timing annotations around each API call to the profiler log. Look for the `synchronize_device` entry in the log; its reported duration reflects the host-side wait time. This method is less precise than Tracy (no device-side correlation) but requires no Tracy GUI and can be used in CI environments where GUI tools are unavailable.

---

## Expected Results

At decode batch=1 on T3K with the current `_maybe_all_gather` implementation:

| Measurement | Expected value | Notes |
|---|---|---|
| Python wall-clock median | 0.1–0.5 ms | Per call; two calls per hybrid attention layer |
| Tracy host zone duration | 10–60 µs (structural) | Upper tail (to ~500 µs) reflects OS preemption; preemption-free runs stay near 10–60 µs |
| Per-step total (N=16 layers) | 3.2–16 ms | Sum across all `_maybe_all_gather` call sites |

> **Note:** These figures are estimates. The implementing engineer must replace them with measured values obtained from the procedures above. Record the measured values in a comment block in this file alongside the date and T3K firmware version used, so future engineers have a calibrated baseline.

---

## Recording Measured Values

Add a section to this file after running the measurement:

```
## Measured Values (TODO: fill in)

- Date measured: TODO
- T3K firmware version: TODO
- Model: TTNNQwen3LinearAttention + TTNNQwen3FullAttention, N=? hybrid layers
- Batch size: 1
- Method used: TODO (wall-clock / Tracy / TTNN op timer)

| Call site | Median latency | 95th-pct latency |
|---|---|---|
| TTNNQwen3FullAttention._maybe_all_gather | TODO ms | TODO ms |
| TTNNQwen3LinearAttention._maybe_all_gather | TODO ms | TODO ms |
| Total per step (sum across N layers) | TODO ms | TODO ms |
```

These measured values feed directly into the throughput improvement estimate in [`throughput_improvement_estimate.md`](./throughput_improvement_estimate.md).
