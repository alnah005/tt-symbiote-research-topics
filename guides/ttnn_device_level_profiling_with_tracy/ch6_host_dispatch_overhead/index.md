# Chapter 6 — Host Dispatch Overhead vs. Device Kernel Time

## Overview

This chapter addresses the boundary between host and device in TTNN performance analysis. Every TTNN op call pays two costs: the time the host spends constructing and enqueueing the work, and the time the device spends actually executing the kernel. For large ops on large tensors, device kernel time dominates and host dispatch is a rounding error. For small ops — elementwise additions on decode-batch tensors, softmax on a single sequence, or matmul on a [32, 32] tile — the host dispatch overhead can be 10–20× larger than the device kernel time. This inversion is the central diagnostic insight of the chapter.

> **Key insight:** For small ops, `host_dispatch_time` dominates `device_kernel_time`. For large ops, `device_kernel_time` dominates `host_dispatch_time`. The crossover point — where dispatch and kernel time are roughly equal — falls around a device kernel duration of ~100 µs for a warm-cache TTNN op call.

Chapter 6 closes the latency decomposition introduced in Chapter 1: `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead`. See [`what_is_dispatch_overhead.md`](./what_is_dispatch_overhead.md) for the full term definitions and a worked numeric example.

You now have all three tools to measure every term: Tracy for `host_dispatch_time` (Chapter 1), the device CSV for `device_kernel_time` (Chapter 3), and `TT_METAL_PROFILER_SYNC` to serialize the pipeline so measurements align correctly (Chapter 2).

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe the full TTNN dispatch pipeline from Python call to kernel launch, and identify which stages contribute to `host_dispatch_time` and which to `device_kernel_time`.
2. Quantify dispatch overhead for a cache-hit call (~5–50 µs) vs. a cache-miss call (~100–500 µs), and explain the difference.
3. Use Tracy zones alongside the device CSV to separate `host_dispatch_time` from `device_kernel_time` for a given op invocation.
4. Use `TT_METAL_PROFILER_SYNC=1` correctly and explain why it is required for accurate per-op latency decomposition.
5. Identify which workloads are dispatch-bound and apply mesh trace capture to eliminate host dispatch overhead from the hot path.
6. Verify that `ttnn.execute_trace` replay has not altered device kernel durations by comparing CSV rows from traced and non-traced runs.

---

## Prerequisites

- **Chapter 1** — Tracy zones for host-side measurement; the latency decomposition equation; Tracy's blind spots regarding on-device kernel internals.
- **Chapter 2** — Tracy capture workflow; `TT_METAL_PROFILER_SYNC` env var; `TT_METAL_DEVICE_PROFILER=1` for CSV generation.
- **Chapter 3** — `DEVICE KERNEL DURATION [ns]` as the wall-clock span from first core start to last core end; the distinction between `DEVICE KERNEL DURATION` and individual RISC duration columns.

---

## Chapter Contents

| File | Description |
|---|---|
| [`what_is_dispatch_overhead.md`](./what_is_dispatch_overhead.md) | The full TTNN dispatch pipeline, where host time and device pre-kernel time are spent, typical overhead magnitudes for cache-hit and cache-miss calls, and why this dominates for small ops. |
| [`measuring_dispatch_vs_kernel.md`](./measuring_dispatch_vs_kernel.md) | How to use Tracy and the device CSV together to measure `host_dispatch_time` and `device_kernel_time` per op; the role of `TT_METAL_PROFILER_SYNC=1`; worked data table across four representative ops and shapes. |
| [`eliminating_dispatch_overhead.md`](./eliminating_dispatch_overhead.md) | Mesh trace capture (`ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`): how it eliminates dispatch overhead, expected speedup for decode-regime workloads, constraints, and how to verify correctness via the device profiler. |

---

## Navigation

- **Previous:** [Chapter 5 — Low FPU Utilization: Causes and Remediation](../ch5_low_fpu_util/index.md)
- **Guide Index:** [`../index.md`](../index.md)
