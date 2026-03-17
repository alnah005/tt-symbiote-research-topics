# Device Profiler API

## Overview

The TTNN device profiler instruments every TTNN operation with hardware cycle counters that run directly on the Tensix cores. This gives you a measurement of true on-device kernel execution time, independent of host-side dispatch overhead or Python interpreter latency. The profiler writes its output to a CSV (comma-separated values) file that you post-process with a bundled script to produce a human-readable timeline.

This section covers how to enable the profiler, what it records, how TTNN op names appear in the output, and how to convert raw cycle counts to nanoseconds correctly.

---

## Enabling the Device Profiler

The device profiler is controlled by the `TT_METAL_DEVICE_PROFILER` environment variable. This variable must be set **before** the tt-metal runtime initializes — it cannot be toggled at runtime after the device is opened.

```bash
# Preferred method: set in the shell before launching your script
TT_METAL_DEVICE_PROFILER=1 python run_moe_forward.py
```

```python
# Alternative: set via os.environ before any tt-metal import
# This only works if tt-metal has not yet been imported in the same process
import os
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

import ttnn  # profiler is now enabled at device open time
```

When enabled, the runtime instructs every compiled kernel program to write start and end cycle counts to a reserved memory region on the device. After each program completes, the driver reads these counts back to the host and appends them to `ops_perf_results_<timestamp>.csv` in the current working directory.

> **Note:** The device profiler buffer auto-flushes after 1000 ops. For workloads with more than 1000 ops, call `ttnn.ReadDeviceProfiler(device)` explicitly before reaching this limit to ensure data is captured at a predictable point; the call reads and resets the buffer.

> **Warning:** Setting `TT_METAL_DEVICE_PROFILER=1` increases memory traffic between the device and host after each op. This adds a small but nonzero overhead to every dispatch. Do not measure absolute performance with the profiler enabled — use it for relative comparisons and to understand the ratio of hardware execution time to total latency.

---

## What the Profiler Records Per Op

Each row in `ops_perf_results_<timestamp>.csv` corresponds to one TTNN operation dispatch. The profiler records the following per op:

| Field | Description |
|---|---|
| `OP TYPE` | The TTNN op class name (see naming convention below) |
| `PROGRAM ID` | Integer identifier for the compiled kernel program; stable across runs when program cache is enabled |
| `DEVICE ID` | Which Tenstorrent device the op ran on (relevant for multi-device setups) |
| `CORE GRID` | The Tensix core grid dimensions used by this op (e.g., `8x8`) |
| `DEVICE KERNEL START CYCLE` | Raw Tensix cycle counter value at kernel start |
| `DEVICE KERNEL END CYCLE` | Raw Tensix cycle counter value at kernel end |
| `DEVICE KERNEL DURATION [ns]` | Duration converted from cycles to nanoseconds using the actual AICLK (AI Clock) frequency |
| `OP TO OP LATENCY [ns]` | Device-side time between consecutive op boundaries on the device timeline — includes dispatch overhead measured at the device, not a host CPU wall-clock measurement |

The cycle-to-nanosecond conversion uses the actual AICLK frequency reported by the device driver, which `process_ops_logs.py` reads automatically. See the warning at the end of this section about not hardcoding this value.

---

## TTNN Op Names in the CSV

TTNN op names in the CSV follow the C++ namespace path of the underlying operation class. Learning the naming convention lets you match CSV rows back to specific calls in your Python source code.

### Naming convention

```
tt::operations::primary::<OpClassName>
ttnn::<op_name>
```

Common examples you will encounter in MoE forward pass analysis:

| Python call | CSV op name |
|---|---|
| `ttnn.matmul(...)` | `tt::operations::primary::matmul` |
| `ttnn.linear(...)` | `tt::operations::primary::matmul` (linear lowers to matmul) |
| `ttnn.softmax(...)` | `tt::operations::primary::moreh_softmax` |
| `ttnn.silu(...)` | `ttnn::silu` |
| `ttnn.topk(...)` | `tt::operations::primary::topk` |
| `ttnn.all_gather(...)` | `ttnn::ccl::all_gather` |

> **Tip:** If you are unsure of the CSV name for a given Python call, run a minimal script with just that one op and `TT_METAL_DEVICE_PROFILER=1`. The resulting CSV will have exactly one data row, making the mapping unambiguous.

### Matching CSV rows to source code calls

When multiple ops of the same type appear in a forward pass (e.g., several matmul calls), use the `PROGRAM ID` and `CORE GRID` columns together with call order to disambiguate. Ops are written to the CSV in dispatch order, so the nth occurrence of a given op name in the CSV corresponds to the nth dispatch of that op at runtime.

For MoE layers, the typical dispatch order is:

```
topk → softmax (routing weights) → all_gather → matmul (gate proj) →
silu → matmul (up proj) → matmul (down proj) → reduce_scatter/all_reduce
```

---

## Interaction with `ttnn.enable_program_cache()`

Program cache caches compiled kernel programs by op type and tensor shape. When the cache is enabled:

- The **first call** to an op compiles the kernel program from scratch, which includes LLVM/SFPU compilation and dispatch. This first call has significantly higher latency than steady-state execution.
- **Subsequent calls** with the same op type and tensor shapes reuse the cached compiled program, incurring only dispatch and execution latency.

When profiling, the first-call compilation latency inflates `DEVICE KERNEL DURATION` because the cycle counters start before compilation completes in some code paths.

```python
import ttnn

# Enable program cache before any ops run
ttnn.enable_program_cache()

# Warm-up: run at least 2 iterations before measuring
# Iteration 0: compilation + dispatch (do not measure)
# Iteration 1: first cached dispatch (may still be slightly elevated)
for i in range(2):
    output = model.forward(input_tensor)

# Measurement iterations: cache is warm, results are stable
for i in range(10):
    output = model.forward(input_tensor)
    # collect timing here
```

> **Warning:** Always run at least 2 warm-up iterations before collecting profiler data when `ttnn.enable_program_cache()` is enabled. Profiling the first iteration measures compilation cost, not steady-state hardware execution. For MoE models with many unique op shapes per layer, warm-up may require more than 2 iterations until all programs are cached.

---

## Post-Processing with `process_ops_logs.py`

The raw `ops_perf_results_<timestamp>.csv` contains cycle counts and requires post-processing to be readable. The bundled script `tt_metal/tools/profiler/process_ops_logs.py` converts cycle counts to nanoseconds, computes op-to-op latencies, and generates a formatted HTML or ODS (OpenDocument Spreadsheet) report.

### Basic usage

```bash
# Run from the tt-metal repository root
python tt_metal/tools/profiler/process_ops_logs.py --csv ops_perf_results_<timestamp>.csv
```

The script reads the AICLK frequency from the device driver log embedded in the CSV header (or queries `tt-smi` if available) and uses it for the cycle-to-nanosecond conversion. It outputs an HTML report and an ODS spreadsheet to the same directory as the input CSV.

For the complete end-to-end workflow from run to report, see [Reading and Interpreting Profiler Output](./reading_op_timing_output.md).

---

## AICLK and the Cycle-to-Nanosecond Conversion

AICLK (AI Clock) is the clock domain that drives the Tensix compute cores on Wormhole hardware. The device profiler measures time in Tensix cycles, so an accurate cycle-to-nanosecond conversion requires knowing the actual AICLK frequency at the time of the run.

> **Warning:** Do not hardcode 1 GHz (1 ns/cycle) for the AICLK conversion on Wormhole hardware. The actual AICLK varies depending on thermal state, power limits, and driver configuration. Hardcoding 1 GHz can introduce errors of 5–15% in your nanosecond estimates. Always use `process_ops_logs.py` or query `tt-smi` directly for the actual frequency.

To check the current AICLK without running a full profiling pass:

```bash
# Query current AICLK from the Tenstorrent System Management Interface
tt-smi | grep AICLK
# Example output:  AICLK: 1202 MHz
```

`process_ops_logs.py` reads this value automatically and applies it to all cycle-count conversions in the generated report.

---

---

**Next:** [`annotating_your_code.md`](./annotating_your_code.md)
