# Two Profilers Compared: Tracy and the Device Profiler

## The Two Complementary Tools

TTNN performance analysis requires two distinct profiling instruments because the system spans two fundamentally different execution domains: the host CPU (running Python, C++ dispatch logic, and command queue management) and the Tensix device (executing compiled kernels on an array of RISC processors with hardware cycle counters).

**Tracy** operates entirely on the host side. It records named CPU zones with nanosecond timestamps using instrumentation macros compiled into the tt-metal C++ runtime. Tracy sees everything the host does: when a Python call enters the C++ dispatch layer, when a program object is created or retrieved from the cache, when the command buffer is written, and when the enqueue function returns.

**The device profiler** operates entirely on the Tensix device. It uses hardware cycle counters built into each Tensix core to record when kernel code begins and ends execution on each RISC processor (NCRISC, TRISC0, TRISC1, TRISC2). BRISC is excluded from per-cycle profiling because it does not have per-cycle counters and does not appear in `ops_perf_results.csv`. These cycle counts are written to a reserved region of L1 memory at kernel boundaries, then read back to the host after the op completes and post-processed by `process_ops_logs.py` into `ops_perf_results.csv`.

## What Each Tool Answers

The fundamental diagnostic question each tool is designed to answer:

| Tool | Primary question answered |
|---|---|
| Tracy | "When did the host enqueue this op, and how long did host-side dispatch take?" |
| Device profiler | "How long did the kernel actually run on Tensix cores, broken down by RISC processor?" |

## Why Both Are Needed: The Latency Decomposition

The total observed latency for a TTNN op invocation decomposes into three additive terms: `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead`. See Ch6 for the full latency decomposition.

Neither tool alone captures all three terms. Tracy captures `host_dispatch_time` precisely. The device profiler captures `device_kernel_time` precisely. `sync_overhead` requires combining both: you measure it as the residual after subtracting the other two from the total wall-clock time observed by a Python `time.perf_counter()` call.

> **Tip:** For large compute-bound ops (e.g., matmul on a [1024, 4096] × [4096, 1024] tensor in BF16), `device_kernel_time` typically dominates — it is on the order of hundreds of µs while `host_dispatch_time` is on the order of 10–20 µs. For small ops (e.g., a [32, 32] elementwise add), `host_dispatch_time` can be 10–20× larger than `device_kernel_time`. Understanding which term dominates is the first step in knowing where to spend optimization effort.

## When to Use Each Tool

| Scenario | Recommended tool(s) | Reason |
|---|---|---|
| Diagnosing unexpected host-side latency between ops | Tracy alone | Device profiler has no visibility into inter-op host activity |
| Tracing op ordering and identifying dispatch serialization | Tracy alone | Tracy zones show exact enqueue sequence and timestamps |
| Measuring TRISC1 (FPU) utilization or PM IDEAL efficiency | Device profiler alone | Tracy does not see on-device kernel internals |
| Analyzing per-RISC duration breakdown (NCRISC, TRISC0, TRISC1, TRISC2) | Device profiler alone | This data lives entirely on the device |
| Attributing a gap between expected and observed throughput | Both together | Need `host_dispatch_time` (Tracy) and `device_kernel_time` (device profiler) to account for all time |
| Verifying that trace replay eliminates dispatch overhead | Both together | Tracy confirms host-side zones collapse; device profiler confirms kernel durations are unchanged |
| Classifying an op as compute-bound vs. bandwidth-bound | Device profiler alone | Classification depends on `FPU UTIL`, `NOC BW UTIL`, and `PM IDEAL` — all device-profiler columns |

> **Note:** "Both together" means running both profilers simultaneously in a single pass — not running the workload twice. Running twice is unreliable because op timing and ordering shift between runs, making the measurements incomparable. The correct procedure is to enable both instruments at the same time: set `TT_METAL_DEVICE_PROFILER=1` and build with `TRACY_ENABLE`, then launch `tracy-capture` before starting your workload. Set `TT_METAL_PROFILER_SYNC=1` to align Tracy zone ends with op completion on the device, which ensures clean decomposition of `host_dispatch_time` and `device_kernel_time` from the same execution. See Chapter 6 for the full simultaneous-capture measurement procedure.

## Known Blind Spots

Understanding what each tool cannot see is as important as understanding what it can see.

### Tracy's blind spots

- **On-device kernel internals** — Tracy has no instrumentation inside Tensix kernel code. Once the enqueue write completes on the host side, Tracy's zone ends. It cannot tell you whether the kernel ran for 1 µs or 1 ms, which RISC was the bottleneck, or whether the FPU was stalled waiting for data.
- **Device-side queueing latency** — there is a delay between when the host writes to the command queue and when the device firmware actually decodes and launches the kernel. Tracy sees the host side of this handoff but not the device side.
- **Multi-core kernel parallelism** — Tracy sees one zone per op dispatch, not one zone per participating Tensix core. It cannot show whether cores started and finished simultaneously or staggered.

### Device profiler's blind spots

- **Host-side dispatch time** — `ops_perf_results.csv` contains no column for the host time that preceded kernel execution. A row with a 0.5 µs `DEVICE KERNEL DURATION [ns]` gives no indication of whether the host spent 5 µs or 500 µs dispatching that op.
- **Inter-op gaps** — the device profiler records per-op rows in isolation. It cannot show you that there was a 50 µs idle gap on the device between op N and op N+1 because the host was slow to enqueue op N+1.
- **Host synchronization cost** — the time the host spends waiting for the device to complete (`Finish` calls, event waits) is not recorded in the CSV. Sync overhead can dominate total latency in tightly synchronized loops.
- **Program cache misses** — a first-call kernel recompilation is entirely host-side work; device cycle counters only begin once the kernel is executing on Tensix, so `DEVICE KERNEL DURATION` is unaffected by compilation time. A cache miss inflates the host dispatch time visible as a Tracy zone, not the `DEVICE KERNEL DURATION` CSV column. The CSV gives no flag or column indicating that host dispatch was slow due to recompilation; you must infer this from Tracy data by comparing first-call and steady-state host dispatch durations.

> **Warning:** A common diagnostic mistake is to look only at `DEVICE KERNEL DURATION` when debugging a slow workload and conclude the kernel itself is slow, when in fact the dominant cost is host dispatch overhead that the device profiler does not record. Always check Tracy data alongside the CSV before concluding a kernel needs optimization.

---

**Next:** [Chapter 2 — Invoking the Profiler for a TTNN Pytest](../ch2_invocation/index.md)
