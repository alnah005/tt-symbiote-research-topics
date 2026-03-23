# Measuring Dispatch Overhead

Before applying the speedup model from `estimating_trace_speedup.md`, you need measured values — not estimates — for how much of your decode step's latency is host dispatch overhead. This file explains how to use Tracy and the TTNN built-in profiler to capture those values, which metrics to extract, how to interpret the host-device timeline in a Tracy trace to identify dispatch-dominated ops, and what reference numbers look like on current Tenstorrent hardware.

---

## Two Profiling Tools

TTNN provides two complementary mechanisms for measuring dispatch overhead: a built-in CSV profiler that logs per-op host and device times at the TTNN runtime level, and a Tracy integration that produces a graphical timeline showing host thread activity and device kernel execution side by side. Both tools surface the same underlying measurements, but they are suited to different analysis tasks.

**TTNN built-in profiler** — best for automated analysis. It emits a structured CSV that can be parsed programmatically to compute aggregate dispatch overhead for a step. Use this tool when you need to integrate overhead measurement into a benchmarking script or when you want to average overhead across many steps.

**Tracy** — best for understanding structure. The graphical timeline makes it immediately visible which ops have large encoding gaps, which ops complete instantly on the device but block the host for tens of microseconds, and where synchronization points interrupt the host-device pipeline. Use Tracy when diagnosing a specific anomaly or auditing a new model for trace candidacy.

---

## Using the TTNN Built-In Profiler

### Enabling Profiling

The TTNN built-in profiler is controlled by environment variables. Set the following before launching your Python script:

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH=1
```

`TT_METAL_DEVICE_PROFILER=1` enables device-side profiling: each kernel execution is timestamped using the device's on-chip cycle counters, and those timestamps are written to a buffer that is read back to the host when you call `ttnn.synchronize_device()` or close the device.

`TT_METAL_DEVICE_PROFILER_DISPATCH=1` enables host-side dispatch profiling: each of the four dispatch phases (argument validation, kernel selection, command encoding, CQ submission) is individually timestamped on the host using `std::chrono::steady_clock`. Both timestamps are emitted into the same output CSV alongside the device kernel timestamps.

> **Warning:** Profiling adds measurement overhead. The act of recording timestamps for each op introduces roughly 2–5 us of additional host latency per op. This overhead is present in all profiled runs and does not appear in unprofile runs. Always compare profiled numbers to other profiled numbers, and unprofile numbers to other unprofile numbers. Never subtract a profiled dispatch time from an unprofile step time — the denominator must come from the same execution mode.

### Running a Profiled Baseline

```python
import ttnn
import torch
import time

device = ttnn.open_device(device_id=0, num_hw_cqs=1)
device.enable_async(True)

# Load model weights and allocate input tensors here.
# ...

# Warm-up run: the cold-path kernel selection cost (described in
# Chapter 1) is paid during warm-up so it does not inflate your
# baseline measurement. Run at least 3–5 warm-up steps.
for _ in range(5):
    output = model.decode_step(input_tensor)
ttnn.synchronize_device(device)

# Baseline measurement: record wall-clock time around a fixed
# number of decode steps.
NUM_STEPS = 50
t0 = time.perf_counter()
for _ in range(NUM_STEPS):
    output = model.decode_step(input_tensor)
ttnn.synchronize_device(device)
t1 = time.perf_counter()

avg_step_ms = (t1 - t0) * 1000.0 / NUM_STEPS
print(f"Average step latency (baseline): {avg_step_ms:.3f} ms  ({avg_step_ms * 1000:.1f} us)")

ttnn.close_device(device)
```

> **Note:** The `ttnn.synchronize_device()` call after the loop is required to ensure the device has completed all in-flight work before the wall-clock stop. Without it, `t1` captures the time the host submitted all commands but not the time the device finished executing them. Step latency is the host-to-device round-trip time, not the time to submit.

### Reading the Profiler CSV

When `TT_METAL_DEVICE_PROFILER=1` is set, TTNN writes a CSV file to the path specified by the `TT_METAL_PROFILER_OUTPUT` environment variable (default: `./profile_log_device.csv` in the working directory). Each row corresponds to one op dispatch event and contains the following columns (among others):

| Column | Description |
|---|---|
| `OP_CODE` | TTNN op name (e.g., `Matmul`, `Softmax`) |
| `HOST_SIDE_START_NS` | Host `steady_clock` nanoseconds at start of phase 1 |
| `HOST_SIDE_END_NS` | Host `steady_clock` nanoseconds at end of phase 4 |
| `DEVICE_START_CYCLE` | Device cycle counter at kernel start |
| `DEVICE_END_CYCLE` | Device cycle counter at kernel end |
| `HOST_DISPATCH_LATENCY_US` | Computed: `(HOST_SIDE_END_NS - HOST_SIDE_START_NS) / 1000` |
| `DEVICE_KERNEL_LATENCY_US` | Computed: device cycles converted to microseconds at device clock frequency |

The `HOST_DISPATCH_LATENCY_US` column is the per-op dispatch overhead value you need for the speedup calculation.

<details>
<summary>Python snippet: parse the CSV to compute total dispatch overhead per step</summary>

```python
import csv
import statistics

def parse_dispatch_overhead(csv_path: str, ops_per_step: int) -> dict:
    """
    Parse the TTNN profiler CSV and compute per-step dispatch overhead.

    Args:
        csv_path:      Path to the profiler output CSV.
        ops_per_step:  Number of ops in one decode step.

    Returns:
        A dict with 'per_op_us' (list), 'total_overhead_us' (float),
        'mean_per_op_us' (float), and 'max_per_op_us' (float).
    """
    dispatch_times_us = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("HOST_DISPATCH_LATENCY_US", "").strip()
            if val:
                dispatch_times_us.append(float(val))

    # Slice to one step's worth of ops (skip warm-up rows if needed).
    step_slice = dispatch_times_us[-ops_per_step:]
    total = sum(step_slice)
    return {
        "per_op_us": step_slice,
        "total_overhead_us": total,
        "mean_per_op_us": statistics.mean(step_slice),
        "max_per_op_us": max(step_slice),
    }

# Example usage:
result = parse_dispatch_overhead("./profile_log_device.csv", ops_per_step=32)
print(f"Total dispatch overhead: {result['total_overhead_us']:.1f} us")
print(f"Mean per-op overhead:    {result['mean_per_op_us']:.1f} us")
print(f"Max per-op overhead:     {result['max_per_op_us']:.1f} us")
```
</details>

---

## Using Tracy for Visual Overhead Inspection

Tracy is an open-source frame profiler with a graphical UI. TTNN includes Tracy instrumentation hooks; when enabled, all four dispatch phases are annotated as named Tracy zones, and device kernel timestamps are emitted as Tracy frame markers. The result is a unified host+device timeline you can navigate interactively.

### Building with Tracy Support

Tracy support requires a build-time flag:

```bash
# When building tt-metal from source, enable Tracy instrumentation:
cmake -B build -DENABLE_TRACY=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo .
cmake --build build --target ttnn -j$(nproc)
```

### Capturing a Tracy Trace

Launch the Tracy capture server in one terminal, then run your script in another:

```bash
# Terminal 1: start the Tracy capture server (records to a .tracy file)
./tracy/capture/build/unix/capture -o decode_baseline.tracy

# Terminal 2: run your script with Tracy output enabled
export TT_METAL_ENABLE_TRACY=1
python your_decode_script.py
```

After the script exits, open `decode_baseline.tracy` in the Tracy profiler UI:

```bash
./tracy/profiler/build/unix/Tracy-release decode_baseline.tracy
```

### Reading a Tracy Trace to Identify Dispatch-Dominated Ops

In the Tracy UI, the main timeline shows horizontal swimlanes. The TTNN runtime occupies two swimlanes: one for the host dispatch thread and one for device kernel execution. Each op produces:

- A named zone in the dispatch thread lane labeled with the op name and phase (e.g., `Matmul::encode`, `Matmul::submit`).
- A marker in the device lane showing when the kernel started and ended.

**What to look for:**

1. **Idle gaps on the device lane.** A device idle gap immediately following a kernel completion indicates that the host has not yet finished encoding the next op. The width of the gap in microseconds is the dispatch overhead for the next op. Select the gap with the mouse to read the exact duration in the Tracy tooltip.

2. **Long encoding zones.** Sort the dispatch thread lanes by zone duration. Ops with encoding zones longer than 40 us are outliers — they contribute disproportionately to total dispatch overhead and are the highest-value targets for tracing.

3. **Phase breakdown within a zone.** Tracy records each of the four dispatch phases as sub-zones. Expand an op's dispatch zone to see whether the overhead is concentrated in argument validation, kernel selection, command encoding, or CQ submission. Encoding dominates for large or multi-core ops; validation dominates for ops with many argument checks.

> **Example:** A `ttnn.matmul` on an 8x8 core grid with 64 runtime arguments shows a 38 us encoding sub-zone and a 12 us gap on the device after the preceding kernel. A `ttnn.add` with a fixed scalar shows a 6 us encoding sub-zone and a 2 us device gap. In a 32-op step where 8 ops are matmuls of this scale, the 8 x 38 us matmul encoding cost (304 us) dominates the total dispatch overhead.

---

## Key Metrics to Extract

Whether you use the CSV profiler or Tracy, the following three metrics characterize dispatch overhead for the speedup calculation:

**1. Op dispatch latency per call (`d_i`).** The host time from the start of argument validation to the end of CQ submission for op `i`. This is `HOST_DISPATCH_LATENCY_US` in the CSV, or the total zone width of the dispatch lane entry for op `i` in Tracy.

**2. Total host dispatch time per decode step (`D`).** The sum of `d_i` over all ops in one step: `D = sum(d_1 + d_2 + ... + d_n)`. This is the `dispatch_overhead` value used in the speedup formula.

**3. Kernel occupancy (`K / T`).** The fraction of total step time (`T`) during which the device is actively executing a kernel vs. idle. Formally: `K = sum of all kernel execution durations`; `occupancy = K / T`. High kernel occupancy (>90%) means dispatch overhead is small relative to device execution — trace will produce a modest speedup. Low occupancy (<70%) means the device is frequently idle waiting for the host, and dispatch overhead is large — trace will produce a significant speedup.

---

## Reference Numbers for Current Hardware

The following values are representative of current Tenstorrent hardware generations running production LLM decode steps. They are provided as sanity-check references; always measure your own workload.

| Metric | Typical range | Notes |
|---|---|---|
| Argument validation (per op) | 5–15 us | Scales with number of checks, not tensor size |
| Kernel selection, warm path (per op) | 1–3 us | Cold path: 50–200 us; paid once per session |
| Command encoding (per op, small) | 6–12 us | Elementwise ops, single-core |
| Command encoding (per op, large) | 20–50 us | Multi-core matmuls, large core grids |
| CQ submission (per op) | 1–5 us | Dominated by cache coherence latency |
| Total dispatch per op (warm) | 17–63 us | As established in Chapter 1 |
| Total dispatch for 32-op decode step | 544 us–2,016 us (2.0 ms) | Full warm-path range: 32 × 17 us = 544 us; 32 × 63 us = 2,016 us |
| Typical decode step latency (small model) | 2–8 ms (2,000–8,000 us) | Dispatch is 7–100% of step |
| Typical decode step latency (large model) | 10–50 ms (10,000–50,000 us) | Dispatch is 1–20% of step |

> **Note:** The "large model" range shows that dispatch overhead can drop below 5% of step time when kernels are long-running. Chapter 4 establishes a rule of thumb of >5% dispatch fraction as the threshold where trace becomes worthwhile; Chapter 5 (`estimating_trace_speedup.md`) quantifies this with the exact formula.

---

**Next:** [`estimating_trace_speedup.md`](./estimating_trace_speedup.md)
