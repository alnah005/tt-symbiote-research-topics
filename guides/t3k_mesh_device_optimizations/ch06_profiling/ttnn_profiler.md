# TTNN Profiler

This file covers the TTNN profiler: how to enable it, what it measures, how to read its output,
and how to use Tracy for timeline visualization. The profiler is the primary tool for Step 2 and
Step 3 of the 5-step workflow described in `index.md`.

---

## Quick Reference: Profiler API Symbols

| Symbol | Type | Description |
|---|---|---|
| `TTNN_ENABLE_PROFILER` | Environment variable | Enable per-op device timing [VERIFY exact name] |
| `ttnn.set_profiler_enabled()` | Python API | Programmatic profiler toggle [VERIFY] |
| `ttnn.get_profiler_report()` | Python API | Retrieve profiler records as a list [VERIFY] |
| Tracy trace flag | Build/env option | Emit Tracy-compatible timeline [VERIFY Tracy integration] |

---

## Section 1: Enabling the Profiler

### Environment Variable

The simplest way to enable per-op profiling is via an environment variable before launching the
Python process:

```bash
export TTNN_ENABLE_PROFILER=1   # [VERIFY exact environment variable name]
python run_moe_decode.py
```

The profiler records every TTNN operation dispatched to the device. There is a small host-side
overhead per op for recording timestamps, so do not leave the profiler enabled in production
builds.

### Programmatic Enable

For finer control — for example, to profile only the MoE layer and exclude warm-up passes — use
the programmatic API [VERIFY API names]:

```python
import ttnn

# Enable profiling for a specific region
ttnn.set_profiler_enabled(True)   # [VERIFY]

output = moe_layer_forward(mesh_device, hidden_states, router_weights)

ttnn.set_profiler_enabled(False)  # [VERIFY]
report = ttnn.get_profiler_report()  # [VERIFY]
```

### Scope and Overhead

The profiler operates at the **op level**: each dispatched kernel produces one profiler record.
Host submission time and device execution time are measured separately, so host overhead (Python
dispatch, buffer allocation) does not inflate device timing. The profiler does not currently
support sub-op or per-core breakdown without Tracy (see Section 4).

> **Tip:** Run at least 10 warmup iterations with the profiler disabled before enabling it.
> Wormhole B0 Ethernet links and DRAM channels require a short steady-state period after the
> first dispatches before bandwidth measurements stabilize.

---

## Section 2: Op-Level Timing

### What Is Measured

Each op produces a profiler record containing:

- **Op name** — the TTNN kernel identifier (e.g., `ttnn.all_to_all`, `ttnn.matmul`)
- **Device ID** — which of the 8 T3K devices (0–7) executed the op
- **Device time** — elapsed device cycles from kernel launch to completion; convert to nanoseconds
  using the device clock frequency [UNVERIFIED exact Hz]
- **Host submission time** — wall-clock time from Python dispatch call to kernel launch;
  represents the host overhead

Device time in cycles is the authoritative metric for comparing across runs because it is
independent of host scheduling noise.

### Key Operations in MoE Profiling

When profiling a Qwen3.5-35B MoE layer on T3K, focus on the following operations:

| TTNN Op | Phase | Expected Latency Range (Decode, B=32) | Notes |
|---|---|---|---|
| `ttnn.all_to_all` (dispatch) | Decode, Prefill | < 1 ms per call | Dispatch token → expert device; see ch02 `collective_primitives.md` |
| `ttnn.all_to_all` (combine) | Decode, Prefill | < 1 ms per call | Gather expert output back to token owner |
| `ttnn.matmul` (expert FFN) | Decode, Prefill | Varies with $B$, $C$ | Per-expert [C, H] × [H, D] matmul |
| `ttnn.topk` | Both | < 0.1 ms | Router top-$k$ selection over $E=256$ experts |
| `ttnn.softmax` / `ttnn.sigmoid` | Both | < 0.1 ms | Router gate activation |

> **Warning:** The expected latency ranges above assume steady-state operation with warm caches.
> First-iteration latency will be higher due to weight loading from DRAM. Always discard warmup
> iterations when computing means.

### All-to-All Volume Reference

At decode with $B=32$ and capacity $C=2$, the all-to-all dispatch volume is approximately
6.4 MB per device. At $B=1$ ($C=1$) the volume is approximately 3.2 MB per device. These numbers
bound the expected transfer time given the T3K Ethernet link bandwidth of ~12.5 GB/s per link
(see `ch01_t3k_topology/ethernet_link_bandwidth.md`).

---

## Section 3: Reading Profiler Output

### CSV Output Format

The TTNN profiler writes a CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `op_name` | string | TTNN operation name |
| `device_id` | int | Device index (0–7 on T3K) |
| `device_time_ns` | int | Device execution time in nanoseconds |
| `host_overhead_ns` | int | Host dispatch overhead in nanoseconds |
| `input_shapes` | string | Serialized input tensor shapes |

The CSV path defaults to `ttnn_profile_output.csv` in the working directory [VERIFY default
output path].

### Python Analysis Snippet

The following script reads the profiler CSV, computes per-op mean latency across all 8 devices,
and prints the fraction of total MoE layer time attributable to each op:

```python
import pandas as pd

# Load profiler output
df = pd.read_csv("ttnn_profile_output.csv")  # [VERIFY default filename]

# Filter to a single MoE layer pass (adjust iteration_id or step filtering as needed)
# Assume the CSV contains a column 'iteration' added by the harness; if not, use row ranges.
layer_df = df.copy()

# Aggregate device_time_ns per op, averaged across all 8 devices
per_op = (
    layer_df
    .groupby("op_name")["device_time_ns"]
    .mean()
    .sort_values(ascending=False)
)

total_time_ns = per_op.sum()

print(f"{'Op Name':<40} {'Mean Device Time (µs)':>22} {'Fraction of Total':>18}")
print("-" * 82)
for op, t_ns in per_op.items():
    print(f"{op:<40} {t_ns / 1e3:>22.1f} {t_ns / total_time_ns:>18.1%}")

# Key diagnostic metric: all-to-all fraction
dispatch_ns = per_op.get("ttnn.all_to_all_dispatch", 0)
combine_ns  = per_op.get("ttnn.all_to_all_combine", 0)
all_to_all_fraction = (dispatch_ns + combine_ns) / total_time_ns

print(f"\nAll-to-all fraction of total MoE layer time: {all_to_all_fraction:.1%}")
if all_to_all_fraction > 0.5:
    print("  → Communication-bound: see bottleneck_diagnosis_guide.md §2")
else:
    print("  → Not communication-bound: check expert matmul DRAM BW next")
```

> **Tip:** Op names in the CSV may include suffixes or be fully qualified (e.g.,
> `tt::operations::all_to_all`). Print `df["op_name"].unique()` first to find the exact strings
> present in your profiler output.

### Identifying Top-5 Expensive Ops

Sort by `device_time_ns` descending and inspect the top 5 entries. In a communication-bound
decode run you will typically see:

1. `ttnn.all_to_all` (dispatch or combine)
2. `ttnn.all_to_all` (the other direction)
3. `ttnn.matmul` (expert FFN)
4. `ttnn.topk`
5. `ttnn.softmax` or `ttnn.matmul` (router projection)

In a compute-bound prefill run the ordering inverts: expert FFN matmul moves to the top.

---

## Section 4: Tracy Timeline Visualization

### Overview

TTNN supports emitting Tracy-compatible traces for timeline visualization [VERIFY Tracy
integration]. Tracy displays all 8 T3K devices as parallel timelines, making it straightforward
to identify:

- **Gaps** — periods where Tensix cores are idle, typically waiting for an all-to-all transfer
  to complete
- **Compute phases** — dense blocks where all cores are executing expert FFN matmuls
- **Overlap** — whether dispatch, compute, and combine phases can be pipelined

### Enabling Tracy Output

```bash
# Build tt-metal with Tracy support [VERIFY build flags]
cmake -DENABLE_TRACY=ON ...

# Run with Tracy profiling
export TRACY_NO_EXIT=1   # [VERIFY Tracy env var]
python run_moe_decode.py
```

Connect the Tracy GUI to the running process or open the recorded `.tracy` file after the run.

### What to Look for

On a communication-bound decode run, the Tracy timeline shows:

- All-to-all dispatch block on all 8 devices (Ethernet transfers active, Tensix idle)
- Short expert FFN compute block
- All-to-all combine block (Ethernet active again)
- Compute utilization is low; gaps dominate

On a compute-bound prefill run the pattern inverts: long dense compute blocks with short
all-to-all gaps between them.

> **Tip:** Use Tracy's "frame" markers to isolate a single MoE layer iteration. Without frame
> markers the timeline becomes difficult to interpret at scale.

---

## Section 5: Comparing Prefill vs. Decode Profiles

The MoE layer operates in two fundamentally different regimes. Profile both before making any
tuning decisions.

### Expected Op Fraction Ranges

| Op | Decode (B=32, C=2) | Prefill (B=4, S=512) |
|---|---|---|
| `ttnn.all_to_all` dispatch + combine | 50–70% | 10–20% |
| Expert FFN `ttnn.matmul` | 20–40% | 60–80% |
| `ttnn.topk` | 1–5% | < 1% |
| Router projection `ttnn.matmul` | 2–6% | 1–3% |
| Other (softmax, accumulation) | < 5% | < 5% |

These ranges are approximate and will vary with hardware configuration, `num_links` setting, and
expert placement. They are provided as a sanity check, not as absolute targets.

### Regime Crossover

As batch size increases from $B=1$ to $B=32$, the expert matmul time grows proportionally to
$C$ (expert capacity), while all-to-all transfer time grows roughly proportionally to payload
volume. The crossover point — where expert matmul time exceeds all-to-all time — depends on
`num_links` and the T3K Ethernet bandwidth. Profile at $B=1$ and $B=32$ to locate this crossover
for your specific configuration.

---

## References

- TTNN profiler source: `ttnn/tools/profiler/` in the `tt-metal` repository [VERIFY path]
- Tracy profiler documentation: https://github.com/wolfpld/tracy
- `ch01_t3k_topology/ethernet_link_bandwidth.md` — Ethernet link bandwidth baseline (~12.5 GB/s per link)
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API reference
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` and bandwidth utilization

---

**Next:** [device_perf_counters.md](./device_perf_counters.md)
