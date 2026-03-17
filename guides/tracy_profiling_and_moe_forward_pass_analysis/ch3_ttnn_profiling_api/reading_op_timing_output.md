# Reading and Interpreting Profiler Output

## Overview

This section walks through the complete workflow for reading profiler output: from running your model with the device profiler enabled, to opening `ops_perf_results_<timestamp>.csv`, to running `process_ops_logs.py`, to interpreting the key columns in the report. It then explains how to reconstruct total MoE (Mixture of Experts) forward pass time from per-op kernel durations and how to reason about the gaps between hardware execution time and wallclock measurement.

---

## Step-by-Step Workflow

Follow these five steps to go from source code to a readable timing report.

### Step 1 — Enable the device profiler

Enable `TT_METAL_DEVICE_PROFILER=1` in your shell before launching the process (see [device_profiler_api.md](./device_profiler_api.md#enabling-the-device-profiler) for the full options and the no-runtime-toggle caveat).

### Step 2 — Run the model

```bash
python run_moe_forward.py
```

The TTNN runtime writes `ops_perf_results_<timestamp>.csv` to the current working directory as the model runs. The file grows incrementally — each op dispatch appends one or more rows.

### Step 3 — Locate the CSV file

```bash
ls -lh ops_perf_results_*.csv
# Example output:
# -rw-r--r-- 1 user group 2.4M Mar 17 14:22 ops_perf_results_2024_03_17_14_22_00.csv
```

If the file is not present, verify that `TT_METAL_DEVICE_PROFILER=1` was set in the environment *before* the Python process started — setting it after import has no effect.

### Step 4 — Run the post-processing script

```bash
# Run from the tt-metal repository root
python tt_metal/tools/profiler/process_ops_logs.py --csv ops_perf_results_<timestamp>.csv
```

The script reads the AICLK (AI Clock) frequency from the CSV header or from `tt-smi`, converts all cycle counts to nanoseconds, computes op-to-op latencies, and writes:
- `ops_perf_results_<timestamp>.html` — a browser-viewable HTML timeline report
- `ops_perf_results_<timestamp>.ods` — an ODS (OpenDocument Spreadsheet) for sorting and filtering

### Step 5 — Read the output report

```bash
# Open the HTML report in a browser
xdg-open ops_perf_results_<timestamp>.html    # Linux
open ops_perf_results_<timestamp>.html        # macOS
```

In the ODS spreadsheet, sort by `DEVICE KERNEL DURATION [ns]` descending to immediately see the most expensive ops.

---

## Key CSV Columns

For a description of each CSV column, see the [CSV column reference in device_profiler_api.md](./device_profiler_api.md#what-the-profiler-records-per-op).

> **Tip:** `DEVICE KERNEL DURATION [ns]` is the hardware execution time only. `OP TO OP LATENCY [ns]` is the device-side time between consecutive op boundaries (including dispatch overhead as seen from the device). Add both to get the total contribution of each op to the forward pass wallclock time.

---

## Finding MoE Ops in the CSV

A complete MoE layer forward pass produces rows for several op types. Filter the `OP TYPE` column for the following strings to isolate MoE-relevant ops:

| Filter term | Ops it matches |
|---|---|
| `moe` | Any MoE-specific fused op if present |
| `matmul` | Gate projection, up projection, down projection expert matmuls |
| `topk` | Routing top-k selection |
| `all_gather` | Cross-device token dispatch (CCL collective) |
| `reduce_scatter` | Cross-device output aggregation (CCL collective) |
| `scatter` | Expert token scatter after routing |
| `gather` | Expert output gather after dispatch |
| `softmax` | Routing weight normalization |
| `silu` | SiLU (Sigmoid Linear Unit) activation between gate and up projections |

> **Note:** `all_to_all` is not a confirmed canonical TTNN/CCL op name and may match nothing in the CSV. The exact CCL op names can vary by TTNN version — use `all_gather` and `reduce_scatter` as the reliable filter terms. Verify the exact op names in your version by running a minimal CCL script with `TT_METAL_DEVICE_PROFILER=1` and inspecting the `OP TYPE` column directly.

For the mapping from Python call to CSV op name (e.g., that `ttnn.linear` lowers to `tt::operations::primary::matmul`), see [TTNN Op Names in the CSV](./device_profiler_api.md#ttnn-op-names-in-the-csv).

In a spreadsheet application, filter the `OP TYPE` column to contain any of these terms to produce a MoE-only view of the timeline.

---

## Reconstructing Total MoE Time

To reconstruct the total hardware execution time for one MoE layer:

```python
import csv

moe_filter_terms = ["matmul", "topk", "softmax", "silu", "all_gather",
                    "reduce_scatter", "scatter", "gather", "moe"]
# Note: "scatter" covers expert token scatter ops. "all_to_all" is not a
# confirmed TTNN CCL op name — use "all_gather" and "reduce_scatter" instead.
# Verify op names for your TTNN version by inspecting OP TYPE in the CSV directly.

def is_moe_op(op_type: str) -> bool:
    return any(term in op_type.lower() for term in moe_filter_terms)

total_kernel_ns = 0
with open("ops_perf_results_<timestamp>.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if is_moe_op(row["OP TYPE"]):
            # DEVICE KERNEL DURATION [ns] is the on-device execution time
            duration = float(row.get("DEVICE KERNEL DURATION [ns]", 0))
            total_kernel_ns += duration

print(f"Total MoE hardware kernel time: {total_kernel_ns / 1e6:.3f} ms")
```

Compare this against the wallclock time for the same forward pass:

```python
import time
import ttnn

ttnn.synchronize_device(device)  # ensure device is idle before starting
t_start = time.perf_counter()

output = model.moe_forward(hidden_states)

ttnn.synchronize_device(device)  # wait for all device work to complete
t_end = time.perf_counter()

wallclock_ns = (t_end - t_start) * 1e9
gap_ns = wallclock_ns - total_kernel_ns

print(f"Wallclock:          {wallclock_ns / 1e6:.3f} ms")
print(f"Hardware kernels:   {total_kernel_ns / 1e6:.3f} ms")
print(f"Unaccounted gap:    {gap_ns / 1e6:.3f} ms")
```

The difference between wallclock and summed kernel time is the **gap** — overhead that does not appear as a named TTNN op in the CSV.

---

## Gap Interpretation

Not all time in a forward pass appears as hardware kernel execution. The gap between summed `DEVICE KERNEL DURATION` and wallclock measurement consists of real overhead that must be diagnosed to close it.

### Gap size interpretation table

| Gap size | Most likely cause | Investigation approach |
|---|---|---|
| **> 5 ms** | Host-side Python overhead or a missing `ttnn.synchronize_device()` that causes the wallclock measurement to include time from a previous async operation | Add Tracy zones around Python sections; verify synchronization points bracket the measurement correctly |
| **1 – 5 ms** | Host dispatch overhead accumulating across many ops, or CCL (Collective Communication Library) collective latency not attributed to a named op row | Check `OP TO OP LATENCY [ns]` for large inter-op gaps; look for CCL ops that appear with zero kernel duration |
| **< 1 ms** | Normal — cache lookup overhead, minor Python interpreter cost, and measurement noise | No action required |

### Known gap sources in MoE forward passes

The following sources of unaccounted time are common in MoE workloads:

**Python operations between TTNN calls**

Routing logic often includes Python-side index construction, tensor masking, and shape manipulation that runs on the CPU and does not appear in the device profiler CSV. These show up as `OP TO OP LATENCY` on the previous op row rather than as their own rows.

```python
# This Python code is not a TTNN op — it does not appear in the CSV
# but it does consume time between ops
expert_mask = (top_k_indices == expert_id).float()  # Python/CPU operation
```

**`ttnn.synchronize_device()` barriers**

Explicit synchronization barriers block the host until all previously dispatched device work completes. Synchronization time appears as wall time but not as a named op in the CSV.

```python
ttnn.synchronize_device(device)  # this is a synchronization point, not a compute op
```

**CCL collective latency between op boundaries**

`ttnn.all_gather()`, `ttnn.reduce_scatter()`, and related CCL collectives may report a `DEVICE KERNEL DURATION` that is shorter than their true end-to-end latency. The remaining inter-chip communication latency appears in the `OP TO OP LATENCY` of the following op.

> **Tip:** For CCL-heavy MoE models on T3K (eight-chip Tenstorrent systems), the collective communication gap is often the dominant source of unaccounted time. If you see large `OP TO OP LATENCY` values on the op immediately following an `all_gather` or `reduce_scatter` row, the gap is most likely cross-chip communication latency.

> **Warning:** The device profiler CSV covers only ops that execute on the local device. Cross-device communication latency is not fully captured in a single device's CSV. On multi-device setups, collect CSV files from all devices and align their timestamps by matching shared `PROGRAM ID` values to get a complete picture.

---

## Complete Worked Example

The following script demonstrates the full workflow in one place.

```bash
#!/bin/bash
# collect_and_analyze_moe_profile.sh

# Step 1: run the model with profiling enabled
TT_METAL_DEVICE_PROFILER=1 python run_moe_forward.py

# Step 2: post-process the CSV
python tt_metal/tools/profiler/process_ops_logs.py --csv ops_perf_results_<timestamp>.csv

# Step 3: print a quick summary of total MoE kernel time
python - <<'EOF'
import csv

moe_terms = ["matmul", "topk", "softmax", "silu",
             "all_gather", "reduce_scatter", "scatter", "gather", "moe"]

total_ns = 0
op_counts = {}

with open("ops_perf_results_<timestamp>.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        op = row.get("OP TYPE", "")
        if any(t in op.lower() for t in moe_terms):
            dur = float(row.get("DEVICE KERNEL DURATION [ns]", 0))
            total_ns += dur
            op_counts[op] = op_counts.get(op, 0) + 1

print(f"\nMoE op breakdown:")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count} dispatches")

print(f"\nTotal MoE hardware kernel time: {total_ns / 1e6:.3f} ms")
EOF
```

---

---

**Next:** [Chapter 4 — MoE Forward Pass Op Breakdown](../ch4_moe_op_breakdown/index.md)
