# Tracy vs. the TTNN Device Profiler

Two distinct profiling systems are available when analyzing a TTNN workload running on Tenstorrent hardware. They measure fundamentally different things, produce different output formats, and answer different questions. Using only one of them to investigate a latency gap will almost always leave part of the picture invisible. This file defines each tool precisely, explains what each one cannot see, and describes the combined workflow that Chapter 5 uses to attribute the 16 ms MoE forward-pass gap.

---

## The Two Tools at a Glance

### Tracy: CPU-Side Op Dispatch Instrumentation

Tracy measures what the **host CPU** is doing. When a TTNN op such as `ttnn.matmul` is called from Python, a chain of function calls executes on the host: Python C extension entry, argument validation, kernel program selection, `EnqueueProgram` (the call that writes the kernel and its arguments into the device command queue). All of this work happens on the host CPU before the device has executed a single instruction of the kernel.

Tracy zones wrap this host-side work. A Tracy zone named `ttnn::matmul` that starts at time T₀ and ends at time T₁ means: "the host spent T₁ − T₀ nanoseconds doing everything required to hand this matmul to the device command queue." It does not mean the matmul kernel ran for T₁ − T₀. The kernel may not have even started running on the device yet when the Tracy zone ends.

### TTNN Device Profiler: On-Device Kernel Execution Timing

The TTNN device profiler measures what **Tensix cores** are doing. It instruments Tensix compute kernels with reads of the hardware cycle counter register at the beginning and end of each kernel. When the host reads back the profiler data (at the end of the program or after a synchronization point), it retrieves per-core cycle-count pairs, converts them to wall-clock time using the device's clock frequency, and writes the result to `profile_log_device.csv`.

> **Note on cycle-to-nanosecond conversion:** The device's clock frequency (AICLK) on Wormhole hardware is not a fixed constant — it varies and must be read from the driver at runtime. The "~1 ns per cycle at 1 GHz" figure that appears in quick-reference tables is illustrative only. Do not hardcode 1 ns/cycle in any analysis. Accurate cycle-to-nanosecond conversion requires the device's actual reported AICLK frequency, which can be obtained from `tt-smi` or is read automatically by the `tt_metal/tools/profiler/process_ops_logs.py` post-processing script (the recommended approach).

A row in `profile_log_device.csv` for a matmul kernel means: "Tensix core at grid coordinate (r, c) executed the matmul kernel from cycle X to cycle Y, a duration of Y − X cycles." This is genuine on-device kernel execution time — the time the silicon spent doing the computation — with no host overhead included.

The device profiler is activated at runtime by setting `TT_METAL_DEVICE_PROFILER=1` in the environment before launching the workload. Unlike Tracy, the device profiler does not require a separate server process; profiler data is written into device DRAM during kernel execution and read back to the host after the program completes.

---

## What Each Tool Answers

The following table maps questions an engineer might ask to the tool that can answer them:

| Question | Tracy | Device Profiler |
|---|---|---|
| When did the host start dispatching this op? | Yes — zone start timestamp | No |
| How long did host-side kernel selection take? | Yes — zone duration | No |
| How long did the kernel actually run on Tensix? | No (without explicit correlation) | Yes — cycle count duration |
| Which Tensix core grid was used? | No | Yes — per-core rows in CSV |
| Is there a gap between two consecutive ops? | Yes — whitespace between consecutive zones on the same thread | Only if both kernels are in the profiler output |
| Is the host waiting for the device? | Partially — a synchronization call appears as a Tracy zone if annotated | No — device profiler does not record idle time |
| What is the kernel's FLOPs/cycle efficiency? | No | Yes — compare cycle count against theoretical FLOPs |
| How long does Python overhead between two ops take? | Yes — gap between consecutive Tracy zones | No |
| Is a CCL collective the source of a gap? | Yes if the CCL dispatch is annotated; the gap is visible even if not annotated | Partially — kernel execution time of the CCL kernel is visible, but not the dispatch overhead |

---

## Architectural Difference: Where Timing Happens

The root reason these tools see different things is that they instrument different points in the dispatch-execute pipeline:

```
Host (CPU)                                       Device (Tensix)
──────────────────────────────────────────────────────────────────
Python: ttnn.matmul(a, b, ...)
  │
  ▼
[Tracy zone start: "ttnn::matmul"]
  │
  ▼
Argument validation
  │
  ▼
Kernel program selection / program cache lookup
  │
  ▼
EnqueueProgram → writes to command queue FIFO
  │
[Tracy zone end: "ttnn::matmul"]
  │
  ▼  (host is now free to dispatch the next op)
                     │
                     ▼  (device dispatcher reads command queue)
                     │
                     ▼
            [Device profiler: cycle count written at kernel entry]
                     │
                     ▼
            Tensix cores execute matmul kernel
                     │
                     ▼
            [Device profiler: cycle count written at kernel exit]
                     │
                     ▼
            Result written to output tensor in DRAM
```

The Tracy zone ends when the host has finished writing to the command queue. The device profiler measurement begins when the Tensix cores start executing. The interval between these two points — between the Tracy zone end and the device profiler start — is the **host-to-device dispatch latency**: the time the command queue instruction spent waiting to be processed by the device dispatcher before the kernel ran.

This interval is invisible to both tools individually. It is only visible by correlating their outputs, which is the Method 1 gap attribution technique described in Chapter 5.

---

## When to Use Each Tool

### Use Tracy Alone When:

- You want to understand **host-side overhead** between ops: Python interpreter time, argument validation time, or program cache lookup time.
- You want to identify **gaps between consecutive ops** at the host level: periods where the Python thread was not dispatching any work to the device.
- You are analyzing a **trace-capture workload** (`ttnn.execute_trace`) and want to see how long each trace replay takes from the host's perspective.
- You want to compare **wallclock time** across runs or configurations without the overhead of the device profiler.

### Use the Device Profiler Alone When:

- You want to measure **kernel execution efficiency**: actual FLOPs/second, Tensix utilization, or memory bandwidth utilization.
- You are **comparing two kernel implementations** (e.g., two different `ProgramConfig` settings for `ttnn.matmul`) and want to know which executes faster on device.
- You suspect a **compute-bound vs. memory-bound** boundary and want per-core cycle counts to verify.

### Use Both Together When:

- You have identified a **latency gap** (by wallclock measurement or by Tracy) that you cannot attribute to a known op.
- You need to determine whether the gap is in **host dispatch**, **device execution**, or **host-device synchronization**.
- You are performing the **16 ms gap investigation** that this guide is built around.

---

## The Combined Workflow

The combined workflow is the primary methodology for gap attribution in Chapter 5. It proceeds in three stages:

**Stage 1: Identify the gap with Tracy.**
Run the workload with Tracy enabled and the device profiler disabled. Open the resulting `.tracy` file and locate the MoE forward-pass timeline. Find the gap: a period of horizontal whitespace between two consecutive named zones that is longer than expected. Measure the gap duration using the Tracy GUI cursor or by exporting to CSV and computing `start[i+1] - end[i]` for consecutive zones.

**Stage 2: Characterize the gap's host content with Tracy.**
Zoom into the gap in the Tracy GUI. If the gap contains child zones (nested beneath the enclosing thread's timeline), those zones represent host work that was not attributed to a named op — for example, Python-level index tensor construction between `ttnn.topk` and `ttnn.gather`. If the gap contains no child zones, the host thread was truly idle during that period — it was either waiting for a device synchronization point or was blocked in a kernel-side operation (e.g., an `EnqueueProgram` call that blocks until the command queue drains).

**Stage 3: Confirm device-side content with the device profiler.**
Run the workload again with both Tracy and `TT_METAL_DEVICE_PROFILER=1` enabled. Locate the device profiler CSV rows that correspond to the ops bracketing the gap. Compare the Tracy zone end timestamp for the op before the gap with the cycle-count start timestamp for the op after the gap. The relationship between these two measurements indicates the gap's origin:

| Observation | Interpretation |
|---|---|
| Tracy zone end for op A is well before device profiler kernel start for op B | Gap is dominated by host-to-device dispatch latency for op B, or by a host-side synchronization wait |
| Tracy zone end for op A is approximately simultaneous with device profiler kernel end for op A | Gap begins when the host is already free; the device was still running op A's kernel during part of the gap |
| Device profiler shows no kernel execution during the gap interval | Gap is entirely host-side: Python overhead, synchronization barrier, or CCL dispatch overhead |
| Device profiler shows a kernel executing during the gap (a kernel with no corresponding Tracy zone) | A kernel was dispatched via a code path that lacks Tracy instrumentation; add a zone at that call site |

---

## Known Blind Spots

### Tracy Blind Spots

**On-device kernel execution time.** A Tracy zone for `ttnn::matmul` ends when the host has enqueued the kernel — not when the kernel finishes running on Tensix. If you see a 200 µs Tracy zone for a matmul, that is 200 µs of host work; the actual kernel could take 50 µs or 2 ms on device, and Tracy alone cannot tell you which.

**Concurrent device execution.** After the host enqueues several ops in rapid succession, the device may be executing them concurrently (if their data dependencies allow). Tracy shows the host dispatch timeline as sequential zones, which looks like the ops ran sequentially even if device-side they overlapped. Do not read Tracy zone durations as device execution durations.

**Code paths without instrumentation.** Not every function in tt-metal has a Tracy zone. A host-side computation that runs between two Tracy zones but is not itself annotated will appear as whitespace (a gap) in the Tracy timeline, even though it is not a synchronization wait. This is a primary source of false-positive gaps that Chapter 5 teaches you to distinguish from genuine synchronization gaps.

**Inter-device communication (CCL) host latency.** The CCL (Collective Communication Library) operations that implement all-to-all and reduce-scatter across T3K chips involve both a host dispatch phase and an on-device execution phase. Tracy may capture the host dispatch call as a zone (if annotated), but the time the CCL collective spends executing on Ethernet cores is only visible via the device profiler, not Tracy.

### Device Profiler Blind Spots

**Host-side Python and dispatch overhead.** The device profiler records nothing about what the host was doing. A 5 ms Python loop that constructs index tensors between two TTNN ops will be completely invisible in the `profile_log_device.csv` output.

**Gaps between kernel launches.** The device profiler records per-kernel start and end times, but it does not record the time between kernel completions and the next kernel start. A period where the device was idle waiting for the next `EnqueueProgram` call to arrive from the host is not captured.

**Post-processing requirement.** The raw `profile_log_device.csv` output requires `tt_metal/tools/profiler/process_ops_logs.py` to produce a human-readable per-op summary. The raw CSV contains per-core rows keyed by cycle counts; without post-processing, it is difficult to identify which rows correspond to which TTNN op.

**Version sensitivity.** The device profiler output format (column names, CSV structure) may change between tt-metal versions. Always use the `process_ops_logs.py` script from the same tt-metal checkout as the binary that produced the CSV.

> **Warning:** Do not compare device profiler cycle counts from two different runs that used different `TT_METAL_DEVICE_PROFILER` settings (one with, one without). The device profiler adds a small amount of overhead to kernel execution (cycle counter reads). Comparing profiler-enabled and profiler-disabled runs to measure profiler overhead is valid, but mixing the two within a single analysis will introduce systematic error.

---

## Summary

Tracy measures what the host CPU is doing: op dispatch, program enqueue, Python-level gaps, and trace lifecycle events. The TTNN device profiler measures what Tensix cores are doing: kernel execution time in hardware cycles. Neither tool alone is sufficient to attribute a latency gap: Tracy cannot see device execution, and the device profiler cannot see host overhead. The combined workflow — identify the gap with Tracy, then confirm its device-side content with the device profiler — is the methodology used in every gap attribution exercise in this guide.

---

---

**Next:** [Chapter 2 — Setting Up Tracy Profiling](../ch2_tracy_setup/index.md)
