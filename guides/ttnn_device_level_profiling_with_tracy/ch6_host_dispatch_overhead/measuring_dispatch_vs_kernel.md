# Measuring Dispatch Overhead vs. Kernel Time

Tracy measures `host_dispatch_time`; the device CSV measures `device_kernel_time`.

## Tracy: Measuring Host-Side Dispatch

Tracy records the host-side dispatch zone for each TTNN op. The zone begins when the Python call enters the C++ dispatch function (via the Python bindings) and ends when the enqueue command is written to the device command queue and the enqueue function returns to the caller.

For `ttnn.matmul`, this Tracy zone will appear in the `.tracy` file under a name like `ttnn::operations::matmul::Matmul` or the op's registered dispatch zone name. The zone duration is the `host_dispatch_time` for that specific invocation.

> **Tip:** In the Tracy GUI, look for op dispatch zones nested inside the TTNN Python binding call. The outermost zone covering the full Python-to-enqueue path is the one to use for `host_dispatch_time`. Inner zones (for sub-stages like program cache lookup or command buffer serialization) give you finer breakdown but are not needed for the basic dispatch vs. kernel comparison.

### What the Tracy zone does NOT include

The Tracy zone ends when enqueue returns — that is, when the host finishes writing to the command queue. It does not include:

- Device firmware command decode time.
- Core grid configuration time on the device.
- NoC descriptor setup on the device.
- The time from kernel launch to kernel completion (`device_kernel_time`).

This is why Tracy alone cannot tell you whether an op is slow because of host overhead or because of slow kernel execution. You need the CSV to see `device_kernel_time`.

---

## Device CSV: Measuring Kernel Time

`DEVICE KERNEL DURATION [ns]` in `ops_perf_results.csv` is the authoritative measure of `device_kernel_time`. See Ch3 for the `DEVICE KERNEL DURATION` definition.

To enable the device CSV:

```bash
export TT_METAL_DEVICE_PROFILER=1
```

Run your workload, then post-process the per-core logs:

```bash
python tt_metal/tools/profiler/process_ops_logs.py --output-dir ./results/
```

The resulting `ops_perf_results.csv` will have one row per op dispatch. Match rows to Tracy zones by op type and invocation order within the same run.

---

## `TT_METAL_PROFILER_SYNC=1`: Why It Is Required

Without `TT_METAL_PROFILER_SYNC=1`, tt-metal pipelines host dispatch and device execution for maximum throughput: while the device is executing op N, the host is already dispatching op N+1. This pipelining means that from Tracy's perspective, the host dispatch zone for op N+1 overlaps with the device kernel execution of op N.

In this pipelined mode:

- The Tracy zone for op N accurately records the host-side enqueue cost — it ends when enqueue returns, exactly as in the serialized case. The zone itself is not distorted by pipelining.
- However, because the host begins dispatching op N+1 while the device is still executing op N, Tracy zones for consecutive ops overlap in wall-clock time. This makes it impossible to decompose total latency additively: you cannot sum `host_dispatch_time` and `device_kernel_time` across ops and expect the result to equal total elapsed time.

Setting `TT_METAL_PROFILER_SYNC=1` inserts an explicit host–device synchronization barrier after each op. The host blocks after enqueueing op N until the device signals completion of op N before proceeding to dispatch op N+1. This serializes the pipeline and guarantees that:

1. No two op dispatch zones overlap in wall-clock time — each op is fully dispatched and device-completed before the next op is enqueued.
2. `total_op_latency` for op N can be measured end-to-end with `time.perf_counter()` on the Python side.
3. The three terms of the decomposition are temporally non-overlapping and can be added.

> **Warning:** `TT_METAL_PROFILER_SYNC=1` reduces throughput by eliminating dispatch-execution overlap. Use it only for latency measurement and dispatch overhead analysis. Never use it to measure tokens/second or ops/second throughput — the result will be artificially low and does not reflect deployed performance.

### Using sync to isolate dispatch overhead

With `TT_METAL_PROFILER_SYNC=1` active, the procedure to isolate `host_dispatch_time` from `device_kernel_time` for a single op is:

```python
import time
import ttnn

# Ensure program cache is warm (run the op once before measuring)
_ = ttnn.matmul(a, b)
ttnn.synchronize_device(device)

# Measure total op latency at the Python level
t0 = time.perf_counter()
c = ttnn.matmul(a, b)
ttnn.synchronize_device(device)  # TT_METAL_PROFILER_SYNC=1 does this implicitly per op,
                                  # but an explicit sync here anchors the Python timer
t1 = time.perf_counter()

total_op_latency_us = (t1 - t0) * 1e6
# host_dispatch_time: read from Tracy zone duration for this invocation
# device_kernel_time: read from DEVICE KERNEL DURATION [ns] in CSV, convert to µs
# sync_overhead = total_op_latency_us - host_dispatch_time_us - device_kernel_time_us
```

Then open the `.tracy` file in the Tracy GUI (or use `tracy-csvexport`) and find the dispatch zone matching this invocation. The Tracy zone duration directly equals `host_dispatch_time` — the zone spans only the host-side enqueue work and contains no device kernel time. With `TT_METAL_PROFILER_SYNC=1` active, simply read the Tracy zone duration to obtain `host_dispatch_time`; no subtraction is needed.

---

## Worked Data Table

The following measurements illustrate how `host_dispatch_time` and `device_kernel_time` scale differently with tensor size. All measurements are warm-cache (program cache enabled, measured on steady-state calls, not first calls). Device: Wormhole B0. Data format: BF16. `TT_METAL_PROFILER_SYNC=1` active.

| Op | Shape | Device Kernel (µs) | Host Dispatch (µs) | Ratio (dispatch/kernel) |
|---|---|---|---|---|
| `ttnn.matmul` | [32, 32] × [32, 32] | ~0.5 | ~8 | ~16× |
| `ttnn.matmul` | [1024, 4096] × [4096, 1024] | ~200 | ~12 | ~0.06× |
| `ttnn.add` | [32, 4096] | ~0.3 | ~6 | ~20× |
| `ttnn.softmax` | [1, 1, 32, 4096] | ~2 | ~7 | ~3.5× |

Key observations from the table:

**Small matmul [32, 32] × [32, 32]:** Kernel optimization is irrelevant here.

**Large matmul [1024, 4096] × [4096, 1024]:** Dispatch overhead is negligible; kernel optimization is the right lever.

**Small add [32, 4096]:** This is one of the most dispatch-dominated op shapes you will encounter in transformer decode inference.

**Softmax [1, 1, 32, 4096]:** Dispatch still dominates, but the margin is narrower. As sequence length grows (e.g., softmax on [1, 1, 32, 32768]), kernel time scales while dispatch stays roughly flat, and the crossover is reached.

### The dispatch/kernel crossover

Dispatch overhead becomes negligible when `device_kernel_time` is roughly 10× or more of `host_dispatch_time`. For a warm-cache TTNN op with ~10 µs dispatch overhead, this crossover occurs around `device_kernel_time ≈ 100 µs`. Ops with kernel durations above ~100 µs are not dispatch-bound; ops with kernel durations below ~10 µs almost certainly are.

> **Note:** The exact crossover depends on the op's dispatch complexity. A multi-input op with complex memory config resolution may have a higher baseline dispatch overhead (~15–25 µs) and therefore require a larger tensor before the crossover is reached. A simple elementwise op may have ~5 µs dispatch and cross over at smaller kernel durations.

---

## Reading Both Outputs Together

To interpret a workload's dispatch vs. kernel balance:

1. Run with `TT_METAL_DEVICE_PROFILER=1` and Tracy capture active, with `TT_METAL_PROFILER_SYNC=1`.
2. For each op of interest, find the Tracy zone duration (= `host_dispatch_time`).
3. Find the corresponding row in `ops_perf_results.csv` and read `DEVICE KERNEL DURATION [ns]` (= `device_kernel_time`).
4. Compute the ratio. If `host_dispatch_time / device_kernel_time > 1`, the op is dispatch-bound.
5. If the op is dispatch-bound and is called in a repeated loop (e.g., inference decode steps), it is a candidate for trace capture elimination (see [`eliminating_dispatch_overhead.md`](./eliminating_dispatch_overhead.md)).

> **Tip:** When profiling a full model forward pass, sort ops by Tracy zone duration descending to find the worst dispatch offenders. Then cross-reference with `DEVICE KERNEL DURATION` from the CSV to distinguish dispatch-bound ops (long zone, short kernel) from compute-bound ops (long zone, long kernel). Only dispatch-bound ops benefit from trace capture.

---

**Next:** [`eliminating_dispatch_overhead.md`](./eliminating_dispatch_overhead.md)
