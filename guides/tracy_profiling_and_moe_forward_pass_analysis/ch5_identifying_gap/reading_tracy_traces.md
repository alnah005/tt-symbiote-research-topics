# Reading Tracy Traces

This file explains how to orient yourself in the Tracy GUI when investigating the MoE forward
pass, and how to compute gap durations programmatically using `tracy-csvexport`.

---

## Tracy GUI: Key Panels

Open a `.tracy` file in `tracy-profiler` (the GUI binary). Four panels are relevant for gap
analysis:

### Timeline View

The timeline view occupies the center of the GUI. Each horizontal lane is one thread. The
main Python thread and the TTNN dispatch thread are the two lanes of interest for MoE
profiling. Each colored rectangle is a Tracy zone: its width is proportional to its duration
and its horizontal position reflects its start time.

Zones nest visually: a parent zone spans its children, and children appear as colored bars
inside the parent rectangle. The zone name appears inside the rectangle when the zone is wide
enough to render text.

Use the mouse scroll wheel to zoom on the horizontal axis. Hold `Ctrl` to scroll vertically.
Right-click on a zone to jump to the zone statistics panel for that zone name.

### Zone Statistics Panel

Open via `View > Zone Statistics` or by right-clicking any zone in the timeline. For a
selected zone name it shows: total call count, mean duration, median duration, min, max, and
standard deviation. Use this panel to check whether a gap is consistent (low stddev) or noisy.

> **Tip:** Sort the zone statistics table by mean duration descending to identify which custom
> zones account for the most time. Any gap between the sum of child zone durations and the
> parent zone duration is unannotated time.

### Frame Graph

Displayed at the top of the GUI. Each bar is one profiled frame (in the tt-metal context, one
TTNN trace replay or one forward pass iteration). Bar height represents frame duration. Click
a bar to jump the timeline to that frame. Use this to select a representative iteration (avoid
the first two, which may include program cache cold-start overhead).

### Statistics Panel

Open via `View > Statistics`. Displays a flat table of all zone names with aggregate timing.
Useful for confirming that `MoE/forward`, `MoE/dispatch`, `MoE/expert_matmul`, and
`MoE/combine` zones are present and accounted for. If a zone is missing it means the
corresponding code path was not annotated (see Chapter 3, `annotating_your_code.md`).

---

## Zooming into the MoE Forward Pass

The custom zones added in Chapter 3 create a navigable hierarchy in the timeline:

```
MoE/forward                         (outermost: one complete MoE layer)
  MoE/dispatch                      (router → topk → gather → all-to-all)
    MoE/dispatch/router
    MoE/dispatch/topk
    MoE/dispatch/index_construction
    MoE/dispatch/gather
    MoE/dispatch/all_to_all         (T3K only)
  MoE/expert_matmul                 (gate/up/down projections)
  MoE/combine                       (scatter → weighted sum → reduce-scatter)
    MoE/combine/scatter
    MoE/combine/weighted_sum
    MoE/combine/reduce_scatter      (T3K only)
```

To zoom in on one MoE layer:

1. In the frame graph, click the bar corresponding to a warm iteration (iteration 4 or later).
2. In the timeline, locate the `MoE/forward` zone on the main thread. It will be the widest
   annotated zone spanning the MoE layer.
3. Right-click `MoE/forward` and select "Zoom to zone" to fill the timeline with that zone.
4. The children `MoE/dispatch` and `MoE/combine` will now be visible as colored sub-bars
   inside the `MoE/forward` rectangle.

### Reading Zone Nesting

In Tracy's visual model, a zone is a child of the zone that was open at the moment it was
created. The Python annotation pattern from Chapter 3:

```python
with tracy.zone("MoE/forward"):
    with tracy.zone("MoE/dispatch"):
        with tracy.zone("MoE/dispatch/topk"):
            output_scores, output_indices = ttnn.topk(router_logits, k=top_k)
        # index construction (annotate here if not already)
        with tracy.zone("MoE/dispatch/gather"):
            dispatched_tokens = ttnn.gather(hidden_states, expert_indices)
    with tracy.zone("MoE/expert_matmul"):
        # expert matmuls
    with tracy.zone("MoE/combine"):
        # scatter and weighted sum
```

This produces the three-level nesting shown above. In the GUI, the `MoE/dispatch` bar ends
before `MoE/expert_matmul` begins; any horizontal whitespace between them on the same thread
is an untracked gap.

---

## Finding Gaps: Horizontal Whitespace Between Zones

In the Tracy timeline, horizontal whitespace between two consecutive zones on the same thread
means the thread was active but not inside any profiled zone during that interval. This
whitespace is the gap under investigation.

To measure a gap manually:

1. Zoom into the region between the end of `MoE/dispatch` and the start of `MoE/expert_matmul`.
2. Hold `Ctrl` and click-drag to activate the ruler overlay. The ruler reports the selected
   duration in the status bar at the bottom of the GUI in nanoseconds.
3. Verify the gap is real by checking whether the device profiler CSV shows a corresponding
   interval with no kernel activity.

> **Warning:** Not all whitespace between zones is a true performance gap. If the thread is
> waiting on a `threading.Event` or a Python queue, the wall-clock whitespace is real but may
> be intentional. Always cross-reference with the device profiler CSV to determine whether
> the device was also idle during the interval.

---

## Using `tracy-csvexport` to Measure Gaps Programmatically

Manual gap measurement in the GUI is useful for initial exploration but is not reproducible
and does not aggregate across multiple iterations. Use `tracy-csvexport` to extract a
machine-readable zone table and compute gap statistics in Python.

### Step 1: Export the CSV

```bash
tracy-csvexport -u output.tracy > zones.csv
```

The `-u` flag outputs timestamps in nanoseconds (unsigned). The resulting CSV has columns:
`name`, `src_file`, `src_line`, `ns_per_tick`, `exec_time_ns`, `self_time_ns`,
`start_ns`, `end_ns`, `thread_id`.

> **Tip:** If `tracy-csvexport` is not in your PATH, it is built alongside `tracy-profiler`
> in the Tracy build directory. The version must match the version used to capture the trace.

### Step 2: Filter to the MoE Thread and Measure Gaps

The following script loads the exported CSV, filters to the main TTNN dispatch thread
(identified by the thread ID that owns the `MoE/forward` zone), and reports all consecutive-zone
gaps larger than 1ms:

```python
import csv

# Load the exported zone CSV produced by tracy-csvexport -u
csv_path = "zones.csv"

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Filter to top-level zones only (no nesting depth info in CSV export;
# use zone name prefix as a proxy for the zones of interest)
MOE_ZONES = {
    "MoE/dispatch",
    "MoE/dispatch/router",
    "MoE/dispatch/topk",
    "MoE/dispatch/index_construction",
    "MoE/dispatch/gather",
    "MoE/dispatch/all_to_all",
    "MoE/expert_matmul",
    "MoE/combine",
    "MoE/combine/scatter",
    "MoE/combine/weighted_sum",
    "MoE/combine/reduce_scatter",
}

moe_rows = [r for r in rows if r["name"] in MOE_ZONES]

# Sort by start time (nanoseconds)
zones = sorted(moe_rows, key=lambda r: float(r["start_ns"]))

# Compute inter-zone gaps: gap = start[i+1] - end[i] for consecutive zones
GAP_THRESHOLD_NS = 1_000_000  # 1 ms

for i in range(len(zones) - 1):
    gap_ns = float(zones[i + 1]["start_ns"]) - float(zones[i]["end_ns"])
    if gap_ns > GAP_THRESHOLD_NS:
        print(
            f"Gap: {gap_ns / 1e6:.2f} ms "
            f"between '{zones[i]['name']}' and '{zones[i + 1]['name']}'"
        )
```

Sample output for a trace with the 16ms gap:

```
Gap: 16.34 ms between 'MoE/dispatch/all_to_all' and 'MoE/expert_matmul'
Gap:  0.23 ms between 'MoE/dispatch/topk' and 'MoE/dispatch/index_construction'
```

The first gap is the primary target. The second gap (0.23ms) is below the threshold for most
investigations but may be worth annotating to confirm the index construction step is covered.

### Step 3: Aggregate Across Iterations

To compute mean and standard deviation across multiple MoE forward pass iterations, group
rows by iteration index before computing gaps. The simplest approach is to insert a Tracy
message at the start of each forward pass iteration marking the iteration number:

```python
import tracy
# Inside the inference loop:
tracy.message(f"iteration={i}")
```

After export, the iteration marker rows appear in the CSV with `name = "iteration=<n>"`.
Use these as delimiters to split the zone list into per-iteration segments, then compute
gap statistics per segment.

```python
# Split zone list into iterations using iteration marker rows
iteration_gaps = []
current_iter_zones = []

for row in zones:
    if row["name"].startswith("iteration="):
        if current_iter_zones:
            # Compute gaps for the completed iteration
            gaps = []
            for i in range(len(current_iter_zones) - 1):
                g = float(current_iter_zones[i + 1]["start_ns"]) - float(current_iter_zones[i]["end_ns"])
                gaps.append(g)
            iteration_gaps.append(gaps)
        current_iter_zones = []
    else:
        current_iter_zones.append(row)

import statistics
for idx, gaps in enumerate(iteration_gaps):
    large = [g for g in gaps if g > GAP_THRESHOLD_NS]
    if large:
        mean_ms = statistics.mean(large) / 1e6
        stddev_ms = statistics.stdev(large) / 1e6 if len(large) > 1 else 0.0
        print(f"Iteration {idx}: {len(large)} gap(s) > 1ms, mean={mean_ms:.2f}ms stddev={stddev_ms:.2f}ms")
```

A consistent gap (stddev < 1ms across 10 iterations) is a real performance issue. A variable
gap (stddev > 3ms) may be OS scheduling jitter and requires further filtering.

---

## Device Profiler CSV Companion

The Tracy zone CSV shows host-side zone boundaries. The device profiler CSV
(`ops_perf_results_<timestamp>.csv`) shows device-side kernel execution times.

For the full CSV column reference including `DEVICE KERNEL DURATION [ns]` and
`OP TO OP LATENCY [ns]`, see Chapter 3,
[`device_profiler_api.md`](../ch3_ttnn_profiling_api/device_profiler_api.md#what-the-profiler-records-per-op).

A large `OP TO OP LATENCY [ns]` value between two consecutive ops in the device profiler CSV
is the device-side counterpart to a Tracy CPU timeline gap. Cross-referencing these two
sources is the foundation of the gap attribution methods in `gap_attribution.md`.

---

---

**Next:** [`gap_attribution.md`](./gap_attribution.md)
