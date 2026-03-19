# TTNN Device-Level Profiling with Tracy

This guide teaches ML engineers and kernel developers how to run the Tracy-based device profiler against a TTNN pytest, interpret the resulting `ops_perf_results` CSV, and use that data to diagnose whether an op is compute-bound or bandwidth-bound — and what to do when it is neither. Readers are assumed to know TTNN ops and basic Tenstorrent hardware but have no prior experience with Tracy or the device profiler CSV.

---

## How to Use This Guide

| Goal | Recommended path | Direct entry point |
|---|---|---|
| I need to run the profiler for the first time | Ch 1 → Ch 2 | [Ch 1 — Tracy Profiler Fundamentals](ch1_tracy_fundamentals/index.md) |
| I want to understand what FPU UTIL means | Ch 3, then Ch 5 | [Ch 3 — Reading the ops_perf_results CSV](ch3_csv_reference/index.md) |
| My op is slow — I don't know if it's compute-bound or bandwidth-bound | Ch 3 → Ch 4 | [Ch 4 — Compute-Bound vs. Bandwidth-Bound Analysis](ch4_compute_vs_bandwidth/index.md) |
| I need to reduce dispatch overhead for decode inference | Ch 3 → Ch 6 | [Ch 6 — Host Dispatch Overhead vs. Device Kernel Time](ch6_host_dispatch_overhead/index.md) |
| I want to understand all CSV columns | Ch 3 | [Ch 3 — Reading the ops_perf_results CSV](ch3_csv_reference/index.md) |
| FPU UTIL is low — I need to diagnose why | Ch 3 → Ch 5 | [Ch 5 — Low FPU Utilization: Causes and Remediation](ch5_low_fpu_util/index.md) |

---

## Chapter Index

| # | Chapter | Description | Key concepts |
|---|---|---|---|
| 1 | [Ch 1 — Tracy Profiler Fundamentals](ch1_tracy_fundamentals/index.md) | Architecture of the Tracy integration in tt-metal: how host and device events are captured, timestamped, and correlated | Tracy zones, device-side instrumentation, `TT_METAL_DEVICE_PROFILER`, `tt_metal_tracy.hpp`, profiler sync |
| 2 | [Ch 2 — Invoking the Profiler for a TTNN Pytest](ch2_invocation/index.md) | Step-by-step: environment variables, `tracy-capture`, `process_ops_logs.py`, and verifying that a run was profiled correctly | `TT_METAL_DEVICE_PROFILER=1`, `TT_METAL_PROFILER_SYNC`, tracy-capture, `process_ops_logs.py`, output file layout |
| 3 | [Ch 3 — Reading the ops_perf_results CSV](ch3_csv_reference/index.md) | Complete column-by-column reference for the `ops_perf_results` CSV: what each field measures, its units, and common pitfalls | DEVICE KERNEL DURATION, FPU UTIL, PM IDEAL, NOC BW UTIL, OP CODE, CORE COUNT, INPUT/OUTPUT shapes |
| 4 | [Ch 4 — Compute-Bound vs. Bandwidth-Bound Analysis](ch4_compute_vs_bandwidth/index.md) | How to classify an op using the roofline model applied to CSV data, and what classification implies for optimization strategy | Arithmetic intensity, roofline model, PM IDEAL comparison, DRAM bandwidth ceiling, FPU peak |
| 5 | [Ch 5 — Low FPU Utilization: Causes and Remediation](ch5_low_fpu_util/index.md) | Systematic diagnosis of low FPU UTIL: tile size misalignment, suboptimal program configs, L1 spill, and dispatch stalls | FPU UTIL, tile alignment, `MatmulMultiCoreReuseProgramConfig`, L1 pressure, data format choice |
| 6 | [Ch 6 — Host Dispatch Overhead vs. Device Kernel Time](ch6_host_dispatch_overhead/index.md) | Measuring and reducing the gap between host op launch and device kernel start, with focus on decode-regime inference | `ttnn.execute_trace`, dispatch latency, trace capture, op fusion, async dispatch, DEVICE KERNEL DURATION vs. end-to-end wall time |

---

## Quick Reference

| Concept / column | What it measures | Learn more |
|---|---|---|
| `DEVICE KERNEL DURATION` | Cycles the device kernel was active from first core start to last core end; the primary measure of device-side op cost | [Ch 3 — CSV Reference](ch3_csv_reference/index.md) |
| `FPU UTIL` | Fraction of kernel cycles during which the FPU (Matrix Engine) was performing useful computation; the key compute-efficiency signal | [Ch 3 — CSV Reference](ch3_csv_reference/index.md), [Ch 5 — Low FPU Utilization](ch5_low_fpu_util/index.md) |
| `PM IDEAL` | Theoretical minimum duration based on peak FPU throughput and the op's MAC count; used as the denominator for FPU UTIL | [Ch 3 — CSV Reference](ch3_csv_reference/index.md), [Ch 4 — Compute vs. Bandwidth](ch4_compute_vs_bandwidth/index.md) |
| `NOC BW UTIL` | Fraction of peak NoC bandwidth consumed; distinguishes bandwidth-bound ops from compute-bound ops | [Ch 3 — CSV Reference](ch3_csv_reference/index.md), [Ch 4 — Compute vs. Bandwidth](ch4_compute_vs_bandwidth/index.md) |
| `TT_METAL_DEVICE_PROFILER` | Environment variable (set to `1`) that enables device-side event recording; profiling is a no-op without it | [Ch 2 — Invocation](ch2_invocation/index.md) |
| `TT_METAL_PROFILER_SYNC` | Environment variable that forces a host-device timestamp synchronization at profiler startup, ensuring accurate cross-boundary latency measurements | [Ch 1 — Tracy Fundamentals](ch1_tracy_fundamentals/index.md), [Ch 2 — Invocation](ch2_invocation/index.md) |
| `tracy-capture` | Command-line tool that collects the Tracy event stream from a running process and writes a `.tracy` file or triggers CSV post-processing | [Ch 1 — Tracy Fundamentals](ch1_tracy_fundamentals/index.md), [Ch 2 — Invocation](ch2_invocation/index.md) |
| `ttnn.execute_trace` | TTNN API that replays a pre-captured op sequence on device with minimal host involvement, reducing dispatch overhead in latency-sensitive loops | [Ch 6 — Host Dispatch Overhead](ch6_host_dispatch_overhead/index.md) |

---

## Prerequisites

Readers should be comfortable with the following before using this guide:

- **TTNN programming model:** You can write a TTNN op forward pass, call `ttnn.to_device` / `ttnn.from_device`, and understand the role of memory configs and program configs.
- **Tenstorrent Wormhole hardware basics:** You know what a Tensix core is, that the Matrix Engine (FPU) performs tile-level `MxK x KxN` multiplications, and that data moves between DRAM and L1 over the NoC.
- **Python and pytest:** You can run a pytest invocation from the command line and read tracebacks.
- **Basic Linux CLI:** You are comfortable setting environment variables, piping commands, and reading CSV files in a terminal.

No prior Tracy experience is required. No prior knowledge of the device profiler CSV format is assumed.

---

## Source Code Location

The profiler infrastructure lives under `tt_metal/tools/profiler/` in the tt-metal repository. Key paths:

| Path | Contents |
|---|---|
| `tt_metal/tools/profiler/` | Root directory for all profiler tooling: capture scripts, post-processing scripts, and C++ instrumentation headers |
| `tt_metal/tools/profiler/process_ops_logs.py` | Post-processing script that converts raw Tracy / device log output into the `ops_perf_results` CSV consumed by this guide |
| `tt_metal/tools/profiler/tt_metal_tracy.hpp` | C++ header that defines device-side Tracy zone macros used to instrument TTNN op kernels; the primary source for understanding which events appear in a profile |
