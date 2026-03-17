# Output File Formats

A combined profiling run produces two distinct output files: a `.tracy` binary database from the Tracy host-side profiler, and a `profile_log_device.csv` from the on-device cycle-counter profiler. This file explains the structure of each, how to post-process them, and how their timestamps relate to each other.

---

## The `.tracy` Binary File

The `.tracy` file is a compressed binary database written by `tracy-capture` when the profiled process disconnects. It is not human-readable and cannot be inspected with a text editor or standard Unix tools.

### Opening in the Tracy GUI

Open `moe_trace.tracy` in `tracy-profiler` (the Tracy GUI) for an interactive timeline view that shows all recorded zones, their durations, thread assignment, and call hierarchy. This is the most efficient way to visually locate large gaps or unexpected serialization.

### Exporting to CSV with `tracy-csvexport`

For scripted analysis, convert the binary file to a flat CSV:

```bash
./tracy-csvexport -u output.tracy > zones.csv
```

The `-u` flag outputs timestamps in nanoseconds (as opposed to microseconds). The resulting CSV has one row per recorded zone event with the following columns:

| Column | Description |
|---|---|
| `name` | Zone name as passed to the Tracy macro |
| `src_file` | Source file where the zone was defined |
| `src_line` | Line number in the source file |
| `ns_since_start` | Nanoseconds elapsed since the start of the trace (not an absolute wall-clock timestamp) |
| `exec_time_ns` | Zone execution duration in nanoseconds |
| `thread` | OS thread ID of the thread that recorded the zone |

Note: there is no `ns_end` column. If you need the zone end time, compute it as `ns_since_start + exec_time_ns`.

### Zone Naming Conventions in tt-metal

tt-metal zone names follow recognizable patterns:

- `TracyTTMetalOp_<op_name>`: op dispatch zones created by the tt-metal dispatch path when an op is enqueued to the device.
- Custom user zones use whatever string literal was passed to the Tracy zone macro in the source.

To find MoE-related zones in a large CSV:

```bash
grep -i "moe\|matmul\|expert\|dispatch" zones.csv
```

### Computing Inter-Zone Gaps

A common analysis task is finding gaps between consecutive zone end and the next zone start — these gaps represent time where no tracked work was happening. Gaps larger than 1 ms are typically worth investigating.

```python
import pandas as pd

# Gap computation using actual column names
df = pd.read_csv("zones.csv")
df = df.sort_values("ns_since_start")
df["gap_to_next_ns"] = df["ns_since_start"].shift(-1) - (df["ns_since_start"] + df["exec_time_ns"])
gaps = df[df["gap_to_next_ns"] > 1_000_000]  # gaps > 1ms
print(gaps[["name", "ns_since_start", "exec_time_ns", "gap_to_next_ns"]].to_string())
```

> **Tip:** Sort by `gap_to_next_ns` descending to immediately surface the largest unexplained pauses. In MoE forward passes, the largest gap is often between the routing softmax zone and the first expert matmul zone — this is the dispatch overhead that Chapter 5 examines in detail.

---

## The `profile_log_device.csv` File

`profile_log_device.csv` is written to the working directory when `TT_METAL_DEVICE_PROFILER=1` is set. It contains raw cycle-counter readings from Tensix cores. The raw file is not directly interpretable; it must be post-processed by `process_ops_logs.py`.

### Running the Post-Processor

```bash
python3 tt_metal/tools/profiler/process_ops_logs.py \
    --device-log profile_log_device.csv \
    --output-dir profiler_output/
```

This produces:
- An ODS spreadsheet in `profiler_output/` with one sheet per op type.
- An HTML timeline visualization in `profiler_output/` showing kernel execution across Tensix cores over time.

### Key Columns After Post-Processing

| Column | Description |
|---|---|
| `OP TYPE` | Op category (e.g., `matmul`, `softmax`) |
| `OP CODE` | Specific kernel variant dispatched |
| `DEVICE ID` | Target device index |
| `CORE X`, `CORE Y` | Tensix core grid coordinates |
| `RISC PROCESSOR TYPE` | Which RISC-V processor (BRISC, NCRISC, TRISC0–2) |
| `TIMER ID` | Marker identifier within the kernel's cycle-counter sequence |
| `TIME[cycles]` | Raw hardware cycle count |
| `DURATION[ns]` | Cycle count divided by the actual AICLK read from the device at runtime |

> **Warning:** Do not assume AICLK is 1 GHz. The `DURATION[ns]` column uses the actual AICLK queried at runtime; using a fixed divisor will produce incorrect nanosecond values for any device running at a different frequency.

### TTNN Visualizer

For interactive exploration of post-processed device profiler output, install the TTNN Visualizer:

```bash
pip install ttnn-visualizer
```

Point it at the `profiler_output/` directory for a browser-based timeline view with per-core drill-down. This is particularly useful for MoE workloads where you need to verify that expert kernels are distributing work evenly across the Tensix core grid.

---

## Correlating Tracy Timestamps with Device Profiler Cycle Counts

Tracy records host wall-clock time using `CLOCK_MONOTONIC` on Linux, producing nanosecond-precision timestamps anchored to process start. The device profiler records hardware cycle counts from Tensix cycle-counter registers, converted to nanoseconds via AICLK.

These two clocks are not automatically synchronized. To anchor device time to host time, tt-metal emits dispatch timing markers that appear in both the Tracy host trace and the device profiler output. The `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` environment variable controls which cores emit these anchor markers.

Direct cross-tool correlation — using the dispatch markers to compute a clock offset and map device events onto the Tracy timeline — is a multi-step process that requires care around AICLK variability and host-device round-trip latency. The full methodology is covered in Chapter 5 (gap attribution), which builds on the output formats described here.

---

---

**Next:** [Chapter 3 — TTNN Op-Level Profiling API](../ch3_ttnn_profiling_api/index.md)
