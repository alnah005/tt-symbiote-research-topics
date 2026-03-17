# Gap Attribution

This file describes the three methods for attributing a measured latency gap to a root cause,
presents a hypothesis table for the 16ms gap, and explains how to rule out measurement noise
before drawing conclusions.

---

## Method 1: Compare Tracy CPU Zone End Times with Device Profiler Kernel Start Times

This method detects **host dispatch latency**: time the CPU spends between finishing one TTNN
op dispatch and starting the next, during which the device may be idle.

### How It Works

The Tracy CPU zone end time marks the moment the host thread returned from the TTNN Python
call that dispatched the previous op. The device profiler CSV records the hardware clock
timestamp at which the next kernel began executing. The difference is the host dispatch
latency for that op.

```
Host timeline:  [zone A end] ----gap---- [zone B start]
Device timeline:     [kernel A end] --- [kernel B start]

                              ^^^^
                     This interval is host dispatch latency
                     if kernel B starts significantly after
                     zone A ended on the host.
```

### Steps

1. Export the Tracy CSV with `tracy-csvexport -u output.tracy > zones.csv`.
2. Load `ops_perf_results_<timestamp>.csv` from the device profiler.
3. Match Tracy zones to device profiler rows by op name. The naming convention
   (from Chapter 3, `reading_op_timing_output.md`) maps `MoE/dispatch/gather` on the Tracy
   side to `ttnn::gather` or similar in the device profiler CSV.
4. For each matched pair, compute:

```python
# CSV loading setup follows the pattern in reading_tracy_traces.md
# (tracy_zones and device_ops are already loaded as shown there)

# Example: find Tracy zone end for MoE/dispatch and device kernel start for matmul
dispatch_zone = next(z for z in tracy_zones if z["name"] == "MoE/dispatch")
matmul_op = next(
    op for op in device_ops
    if "matmul" in op.get("OP TYPE", "").lower()
    and float(op.get("DEVICE KERNEL START [ns]", 0)) > float(dispatch_zone["end_ns"])
)

host_zone_end_ns = float(dispatch_zone["end_ns"])
device_kernel_start_ns = float(matmul_op["DEVICE KERNEL START [ns]"])

host_dispatch_latency_ms = (device_kernel_start_ns - host_zone_end_ns) / 1e6
print(f"Host dispatch latency: {host_dispatch_latency_ms:.2f} ms")
```

5. If `host_dispatch_latency_ms` is large (e.g., 10–16ms) relative to the Tracy gap, the
   gap is dominated by host-side overhead: the host is not immediately enqueuing the next op
   after the previous one completes.

> **Warning:** Tracy host timestamps and device profiler timestamps are in different clock
> domains on T3K. The device profiler anchors to the host clock via dispatch core timestamps
> recorded at program enqueue time (see `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES`). A clock
> skew of a few microseconds is normal; skew larger than 100µs indicates a misconfigured
> profiler setup. Do not draw conclusions from Method 1 if clock alignment has not been
> verified.

---

## Method 2: Look for Explicit Synchronization Calls

This method detects **device synchronization barriers**: explicit host calls that block until
the device finishes all enqueued work.

### How It Works

`ttnn.synchronize_device(device)` and `ttnn.wait_for_event(event)` both block the calling
Python thread. If either appears in the code path between two MoE ops, Tracy will record a
zone for the duration of the blocking call (if the TTNN Python bindings add a Tracy zone for
it, or if you added one manually). The zone duration directly explains the gap.

### Steps

1. In the Tracy GUI, zoom into the gap region identified in `reading_tracy_traces.md`.
2. Look for a zone named `ttnn.synchronize_device`, `synchronize_device`, or
   `ttnn.wait_for_event` on the dispatch thread spanning the gap duration.
3. If no zone appears, search the source code for synchronization calls in the MoE forward
   pass code path:

```python
# grep equivalent: find synchronization calls in the MoE layer implementation
import ast, pathlib

source = pathlib.Path("models/experimental/moe/tt/moe_layer.py").read_text()
tree = ast.parse(source)

sync_calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and hasattr(node.func, "attr")
    and node.func.attr in {"synchronize_device", "wait_for_event"}
]

for call in sync_calls:
    print(f"Line {call.lineno}: {ast.unparse(call)}")
```

4. If a synchronization call is found but has no Tracy zone, wrap it manually:

```python
import tracy

with tracy.zone("ttnn.synchronize_device"):
    ttnn.synchronize_device(device)
```

5. Re-run the trace and re-check. If the gap is now fully inside the new zone, Method 2
   confirms the barrier as the root cause.

> **Tip:** In multi-chip (T3K) configurations, CCL ops such as `all_to_all` and
> `reduce_scatter` sometimes include an implicit synchronization at completion. If the CCL op
> has no Tracy zone but a synchronization zone appears immediately after `ttnn.gather`, the
> synchronization may be clearing a CCL completion event rather than an explicit user call.

---

## Method 3: Check If the Gap Aligns with a CCL Collective

This method detects **CCL collective latency**: time spent in all-to-all or reduce-scatter
communication on the T3K mesh that is not captured in a named Tracy zone.

### How It Works

On T3K, two CCL collectives appear in the MoE forward pass (see Chapter 4):

- All-to-all (or all-gather) after `ttnn.gather` in the dispatch phase: redistributes
  dispatched tokens to the chip holding each expert shard.
- Reduce-scatter (or all-reduce) after the expert matmul phase: aggregates partial expert
  outputs from each chip.

If these CCL ops are not annotated with a Tracy zone, their latency appears as whitespace
between the surrounding zones. The distinguishing property is that CCL latency scales
linearly with message size.

### Steps

1. Estimate the expected CCL all-to-all duration for your configuration:

```python
# Estimate all-to-all latency for token redistribution on T3K
d_model = 7168          # DeepSeek-V3 / Qwen 235B hidden dimension
seq_len = 1024          # tokens in prefill
top_k = 8               # active experts per token
num_chips = 8           # T3K mesh size
bytes_per_element = 2   # BF16

# Each chip sends its share of dispatched tokens to all other chips.
# Total message volume per chip = (seq_len * top_k * d_model * bytes) / num_chips
message_bytes = (seq_len * top_k * d_model * bytes_per_element) / num_chips
# Approximate: 1024 * 8 * 7168 * 2 / 8 = 14,680,064 bytes ≈ 14 MB per chip

# T3K inter-chip ethernet bandwidth ≈ 7 GB/s effective for large all-to-all
effective_bw_bytes_per_s = 7e9  # 7 GB/s

ccl_latency_s = message_bytes / effective_bw_bytes_per_s
print(f"Estimated CCL all-to-all latency: {ccl_latency_s * 1000:.2f} ms")
# At seq_len=1024: ~2.1 ms (well below 16ms; see Pattern C for when this is larger)
```

2. If the observed gap is much larger than the CCL estimate, the gap is not primarily CCL
   latency — return to Method 1 or Method 2.
3. If the gap scales proportionally with `seq_len` across multiple runs (using the sweep
   from Chapter 6), CCL latency is the primary contributor. Annotate the CCL call with a
   Tracy zone to confirm:

```python
with tracy.zone("MoE/dispatch/all_to_all"):
    dispatched_tokens = ttnn.experimental.all_to_all(
        gathered_tokens,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

---

## The 16ms Hypothesis Table

The following table summarizes what each hypothesis predicts about the Tracy and device
profiler evidence for a consistent 16ms gap observed between `MoE/dispatch` and
`MoE/expert_matmul` at seq_len=1024 on T3K.

| Hypothesis | Tracy Zone Pattern | Device Profiler Evidence | Repeatability |
|---|---|---|---|
| **H1: Host-side Python overhead** | Gap on CPU thread with no zone present; CPU is active (not waiting) but not inside any profiled scope | `OP TO OP LATENCY [ns]` between gather and first matmul is ~16ms; `DEVICE KERNEL DURATION [ns]` for the matmul is normal | Consistent across runs (same Python code path executes every iteration); stddev < 1ms |
| **H2: Device synchronization barrier** | `ttnn.synchronize_device` or `ttnn.wait_for_event` zone spans ~16ms on the CPU thread | Device profiler shows a ~16ms gap between kernel end times; device was idle during the interval | Highly consistent; stddev < 0.5ms because synchronization waits for a deterministic device queue drain |
| **H3: CCL collective latency** | Gap between `MoE/dispatch/gather` and `MoE/expert_matmul` zones; no CCL zone present (unannotated); gap widens as seq_len increases | `OP TO OP LATENCY [ns]` shows a large inter-op interval; CCL kernel may appear in device profiler under a generic name (`eth_txq_sender` or similar) | Scales with seq_len; stddev moderate (5–15% of mean) due to network-level jitter on ethernet links |
| **Noise / OS jitter** | Gap is intermittent; present in some iterations, absent in others | `OP TO OP LATENCY [ns]` is normal in runs without the gap | High stddev (> 5ms); gap disappears under CPU affinity pinning or with `TRACY_NO_EXIT=1` preventing GC during measurement |

> **Hypothesis (unconfirmed):** Based on the model dimensions and T3K link bandwidth, the
> 16ms gap at seq_len=1024 is too large to be explained by CCL all-to-all latency alone
> (estimated ~2.1ms; see Method 3 calculation). The gap most likely combines a device
> synchronization barrier (H2) with unannotated host-side index construction (H1). This
> should be confirmed by wrapping the index construction step and the synchronization call
> in Tracy zones and re-profiling.

---

## Ruling Out Measurement Noise

Before concluding that any gap is a real performance issue, verify that the gap is
statistically consistent across multiple iterations.

### Procedure

Run 10 warm inference iterations (after 3 warm-up iterations to prime the program cache)
and record the gap duration for each:

```python
import time
import statistics

# Assume: moe_forward() returns after one complete MoE layer forward pass.
# Tracy zones are active. Record the gap from the CSV after each run.
# Here we show a simplified version using wallclock timing of the gap region.

gap_durations_ms = []

# Warm-up
for _ in range(3):
    moe_forward(hidden_states, router_weights)

# Timed runs
for i in range(10):
    # Start measurement around the suspected gap region
    t_start = time.perf_counter_ns()
    output = moe_forward(hidden_states, router_weights)
    t_end = time.perf_counter_ns()

    # Total forward pass time includes more than just the gap;
    # use device profiler CSV or Tracy CSV for the gap-specific value.
    # This is a placeholder to show the iteration structure.
    total_ms = (t_end - t_start) / 1e6
    gap_durations_ms.append(total_ms)  # replace with gap-specific measurement

mean_ms = statistics.mean(gap_durations_ms)
stddev_ms = statistics.stdev(gap_durations_ms)
cv = stddev_ms / mean_ms  # coefficient of variation

print(f"Mean gap: {mean_ms:.2f} ms")
print(f"Stddev:   {stddev_ms:.2f} ms")
print(f"CV:       {cv:.2%}")

if cv < 0.05:
    print("Gap is consistent (CV < 5%). Likely a real performance bottleneck.")
elif cv < 0.20:
    print("Gap has moderate variability. Check for OS scheduling interference.")
else:
    print("Gap is highly variable. Pin CPU affinity and retest before concluding.")
```

### Interpretation

| Coefficient of Variation (CV) | Interpretation |
|---|---|
| < 5% | Gap is deterministic. Root cause is inside the tt-metal stack (barrier, kernel, CCL). |
| 5%–20% | Moderate jitter. May be OS scheduling or DRAM refresh. Increase sample size to 20. |
| > 20% | High jitter. Likely OS preemption or GC. Pin process with `taskset`, disable swap, and retest. |

A consistent gap of 16ms (mean=16.3ms, stddev=0.2ms, CV=1.2%) is a real bottleneck.
A variable gap (mean=8ms, stddev=4ms, CV=50%) is likely OS noise and should not drive
optimization decisions without further filtering.

---

---

**Next:** [`common_gap_patterns.md`](./common_gap_patterns.md)
