# TTNN Op Timer Profiling

## Context

This file addresses **Q8**: What is the per-op latency breakdown for `TTNNMoE.forward` on T3K, obtained programmatically so that results can be aggregated, filtered, and compared across configuration sweeps without a GUI?

Source ranges: `moe.py:L1159–L1343` (`TTNNExperts.forward`), `moe.py:L1412–L1496` (`TTNNMoE.forward`).

---

## How TTNN Op Timers Work

When `TT_METAL_DEVICE_PROFILER=1` is set, TT-Metal records a timing event for every op dispatched to the device command queue. Each event captures:

- Op name (the C++ kernel name, e.g., `"matmul_multi_core_reuse_mcast_1d"`)
- Op type string from the TTNN Python dispatch layer (e.g., `"ttnn.linear"`, `"ttnn.operations.ccl.all_to_all"`)
- Device-side start and end cycle counts from the Tensix free-running timer
- Host-dispatch timestamp (useful for ordering but not for kernel duration)

After profiling completes, TT-Metal writes a CSV to the path specified by `TT_METAL_DEVICE_PROFILER_OUTPUT`. The CSV has one row per op dispatch, with cycle counts that convert to microseconds via `elapsed_cycles / device_clock_mhz`.

The device clock on Wormhole B0 is **1 GHz** (1000 MHz). Converting:

```
elapsed_us = (end_cycle - start_cycle) / 1000.0
```

---

## Step 1: Environment Variables

```bash
# Enable device-side profiling
export TT_METAL_DEVICE_PROFILER=1

# Output directory for the profiling CSV
export TT_METAL_DEVICE_PROFILER_OUTPUT=/tmp/ttnn_op_timers

# Optional: op timer output verbosity (set to 1 to also print a summary to stdout)
export TT_METAL_DEVICE_PROFILER_DISPATCH_VERBOSE=0

# Disable Tracy to avoid interference
export TT_METAL_ENABLE_TRACY=0
```

Create the output directory before running:

```bash
mkdir -p /tmp/ttnn_op_timers
```

---

## Step 2: Minimal Profiling Harness

The harness below runs `TTNNMoE.forward` for `n_profile` passes after warmup and collects the resulting CSV. It writes a per-pass summary and an aggregated per-op-type summary.

```python
# profile_op_timers.py
import os
import csv
import glob
import time
import statistics
from pathlib import Path
from typing import Dict, List

import ttnn
import torch


DEVICE_CLOCK_MHZ = 1000  # Wormhole B0: 1 GHz


def cycles_to_us(cycles: int) -> float:
    return cycles / DEVICE_CLOCK_MHZ


def read_profiler_csv(output_dir: str) -> List[Dict]:
    """
    Parse all CSV files written by TT_METAL_DEVICE_PROFILER into a list of dicts.
    TT-Metal writes one CSV per device; on T3K (8 devices) there will be 8 files.
    """
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(output_dir, "*.csv"))):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = csv_path
                rows.append(row)
    return rows


def extract_op_timings(rows: List[Dict]) -> Dict[str, List[float]]:
    """
    Group rows by op_name and return a dict mapping op_name → list of elapsed_us
    values, one per dispatch.
    """
    timings: Dict[str, List[float]] = {}
    for row in rows:
        try:
            start = int(row.get("device_start_cycle") or row.get("DEVICE_START_CYCLE", 0))
            end   = int(row.get("device_end_cycle")   or row.get("DEVICE_END_CYCLE", 0))
        except (ValueError, KeyError):
            continue
        elapsed = cycles_to_us(end - start)
        if elapsed <= 0:
            continue
        op_name = row.get("op_name") or row.get("OP_NAME") or "unknown"
        timings.setdefault(op_name, []).append(elapsed)
    return timings


def run_and_profile(model, x, mesh_device, output_dir: str, n_warmup=3, n_profile=10):
    """
    Runs TTNNMoE.forward with op timer profiling enabled.
    Returns a dict of op_name → {'count': int, 'mean_us': float, 'min_us': float, 'max_us': float}.
    """
    # Warmup passes: device profiler is active but we discard output
    for _ in range(n_warmup):
        _ = model(x)
        ttnn.synchronize_device(mesh_device)

    # Clear any warmup CSV output
    for f in glob.glob(os.path.join(output_dir, "*.csv")):
        os.remove(f)

    # Profiling passes
    for i in range(n_profile):
        out = model(x)
        ttnn.synchronize_device(mesh_device)

    # Read and aggregate
    rows = read_profiler_csv(output_dir)
    timings = extract_op_timings(rows)

    summary = {}
    for op_name, values in timings.items():
        summary[op_name] = {
            "count": len(values),
            "mean_us": statistics.mean(values),
            "min_us": min(values),
            "max_us": max(values),
            "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary, out


def print_summary_table(summary: Dict, top_n: int = 20):
    """Print the top_n ops by mean latency, descending."""
    sorted_ops = sorted(summary.items(), key=lambda kv: kv[1]["mean_us"], reverse=True)
    print(f"\n{'Op Name':<55} {'Count':>6} {'Mean µs':>9} {'Min µs':>8} {'Max µs':>8} {'Stdev µs':>9}")
    print("-" * 100)
    for op_name, stats in sorted_ops[:top_n]:
        print(
            f"{op_name:<55} "
            f"{stats['count']:>6} "
            f"{stats['mean_us']:>9.1f} "
            f"{stats['min_us']:>8.1f} "
            f"{stats['max_us']:>8.1f} "
            f"{stats['stdev_us']:>9.1f}"
        )
```

---

## Step 3: Mapping CSV Rows to Forward-Pass Stages

The CSV op names do not directly correspond to Python call sites. The table below maps the expected CSV kernel names to the source lines they originate from:

### `TTNNMoE.forward` (moe.py:L1412–L1496)

| CSV op name (partial match) | Source line | Python call |
|---|---|---|
| `all_gather` | L1429–L1436 | `ttnn.experimental.all_gather_async` |
| `matmul_multi_core_reuse_mcast_1d` (first) | L1445–L1455 | `ttnn.linear` gate routing, HiFi4 |
| `topk` | L1466 (inside router) | `route_tokens_to_experts` → `TTNNMoERouterDecode.forward` |
| `matmul_multi_core_reuse_mcast_1d` (×3) | L1471 (inside experts) | `TTNNExperts.forward` w1, w3, w2 |
| `eltwise_mul` (scalar) | L1477 | `ttnn.mul(routed_out, 1.0/float(n_rs))` — pre-norm scale |
| `reduce_scatter` | L1478–L1490 | `ttnn.experimental.reduce_scatter_minimal_async` |
| `matmul_multi_core_reuse_mcast_1d` (last) | L1493 | `shared_experts` linear |
| `eltwise_add` | L1494 | `ttnn.add` |

> The gate linear and expert matmuls share the same kernel name `matmul_multi_core_reuse_mcast_1d`. Distinguish them by dispatch order (the gate linear fires before the three expert matmuls) or by tensor shape annotations if `TT_METAL_DEVICE_PROFILER_DISPATCH_VERBOSE=1`.

### `TTNNExperts.forward` (moe.py:L1159–L1343)

| CSV op name (partial match) | Source lines | Stage |
|---|---|---|
| `pad` or `concat` | L1191–L1212 | Token padding |
| `all_to_all` (first) | L1225–L1230 | `all_to_all_dispatch` |
| `moe_expert_token_remap` | L1238–L1245 | Token remap |
| `matmul_multi_core_reuse_mcast_1d` (1st in experts) | L1250–L1259 | w1 sparse matmul |
| `matmul_multi_core_reuse_mcast_1d` (2nd in experts) | L1260–L1269 | w3 sparse matmul |
| `silu` + `eltwise_mul` | L1271–L1275 | SiLU activation + elementwise mul |
| `matmul_multi_core_reuse_mcast_1d` (3rd in experts) | L1280–L1289 | w2 sparse matmul |
| `all_to_all` (second) | L1307–L1312 | `all_to_all_combine` |
| `repeat` + `permute` + `mul` + `sum` | L1321–L1335 | Weight application |

---

## Step 4: Per-Pass CSV Output and Aggregation

For configuration sweep comparison, write per-run CSVs and compare them across runs:

```python
import pandas as pd


def save_summary_csv(summary: Dict, output_path: str):
    """Write the per-op summary to a CSV file for later comparison."""
    rows = []
    for op_name, stats in summary.items():
        rows.append({"op_name": op_name, **stats})
    df = pd.DataFrame(rows).sort_values("mean_us", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Written: {output_path}")


def compare_summaries(path_a: str, label_a: str, path_b: str, label_b: str):
    """
    Compare two per-op summary CSVs produced by save_summary_csv.
    Prints a diff table showing mean_us for each op under both configs
    and the absolute and relative change.
    """
    df_a = pd.read_csv(path_a).set_index("op_name")[["mean_us"]].rename(columns={"mean_us": label_a})
    df_b = pd.read_csv(path_b).set_index("op_name")[["mean_us"]].rename(columns={"mean_us": label_b})
    df = df_a.join(df_b, how="outer").fillna(0)
    df["delta_us"] = df[label_b] - df[label_a]
    df["delta_pct"] = (df["delta_us"] / df[label_a].replace(0, float("nan"))) * 100
    df = df.sort_values("delta_us")
    print(df.to_string(float_format=lambda x: f"{x:.1f}"))
```

Example sweep usage:

```bash
# Run config A (in0_block_w=2)
python profile_op_timers.py --in0_block_w 2 --output /tmp/sweep_bw2/
# Run config B (in0_block_w=4)
python profile_op_timers.py --in0_block_w 4 --output /tmp/sweep_bw4/
# Compare
python -c "
from profile_op_timers import compare_summaries
compare_summaries(
    '/tmp/sweep_bw2/summary.csv', 'in0_block_w=2',
    '/tmp/sweep_bw4/summary.csv', 'in0_block_w=4',
)
"
```

---

## Step 5: Multi-Pass Aggregation Strategy

At batch=1 decode, individual kernel runtimes exhibit measurement noise of ±5–15% because the device command queue can have variable fill levels between passes. Use the following aggregation rules:

1. **Discard the first 3 warmup passes** (JIT compile, cache cold).
2. **Collect at least 10 profiling passes** (20 preferred for stable statistics).
3. **Report the median, not the mean**, for individual ops. The mean is pulled upward by occasional outliers caused by DRAM bank conflicts. Use `statistics.median` instead of `statistics.mean` in `extract_op_timings`.
4. **Report the sum of medians** as the estimated forward pass latency. This underestimates true latency slightly because it ignores synchronization overhead between ops, but it is the right metric for identifying which ops are worth optimizing.
5. **For CCL ops** (`all_gather`, `reduce_scatter`, `all_to_all`), report min instead of median. CCL latency at low occupancy is dominated by the software path, and the minimum reflects the best-case hardware performance that tuning is trying to approach.

---

## Step 6: Expected Output at Batch=1 Decode (T3K, GLM-4-MoE)

The table below gives representative baseline values. Actual measurements will differ based on mesh configuration and firmware version.

| Op | Stage | Expected median µs |
|---|---|---|
| `all_gather` | `TTNNMoE` L1429–L1436 | 80–150 |
| `matmul_multi_core_reuse_mcast_1d` (gate) | `TTNNMoE` L1445–L1455 | 20–40 |
| `all_to_all` (dispatch) | `TTNNExperts` L1225–L1230 | 60–120 |
| `matmul_multi_core_reuse_mcast_1d` (w1) | `TTNNExperts` L1250–L1259 | 30–60 |
| `matmul_multi_core_reuse_mcast_1d` (w3) | `TTNNExperts` L1260–L1269 | 30–60 |
| `silu` + `eltwise_mul` | `TTNNExperts` L1271–L1275 | 5–15 |
| `matmul_multi_core_reuse_mcast_1d` (w2) | `TTNNExperts` L1280–L1289 | 30–60 |
| `all_to_all` (combine) | `TTNNExperts` L1307–L1312 | 60–120 |
| `repeat` + `permute` + `mul` + `sum` | `TTNNExperts` L1321–L1335 | 20–50 |
| `eltwise_mul` (scalar, pre-norm) | `TTNNMoE` L1477 | 2–8 |
| `reduce_scatter` | `TTNNMoE` L1478–L1490 | 60–120 |
| `matmul_multi_core_reuse_mcast_1d` (shared) | `TTNNMoE` L1493 | 20–40 |

CCL ops (`all_gather`, both `all_to_all`, `reduce_scatter`) collectively account for approximately 55–65% of the total forward-pass device time at batch=1 decode. This is the principal finding that motivates CCL topology tuning (Chapter 2) and expert dispatch pipeline work (Chapter 3).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No CSV files written to `TT_METAL_DEVICE_PROFILER_OUTPUT` | Env var not set before process start, or path does not exist | Verify env var and `mkdir -p` the directory |
| All cycle counts are 0 | Device profiler not compiled in | Rebuild TT-Metal with profiling enabled (check `CMakeLists.txt` for `ENABLE_DEVICE_PROFILER`) |
| CSV has rows but `device_start_cycle` column is missing | CSV schema changed in a newer TT-Metal version | Print `rows[0].keys()` to discover actual column names and update `extract_op_timings` |
| `matmul_multi_core_reuse_mcast_1d` appears only once, not 4× | Kernel fusion or op deduplication in a newer TTNN version | Use `TT_METAL_DEVICE_PROFILER_DISPATCH_VERBOSE=1` to get per-dispatch tensor shape information |
| Very high variance (> 30% stdev) | DRAM contention from concurrent device usage | Ensure no other processes are using the T3K mesh; check `tt-smi` |

---

Next: [`router_latency_profiling.md`](./router_latency_profiling.md)
