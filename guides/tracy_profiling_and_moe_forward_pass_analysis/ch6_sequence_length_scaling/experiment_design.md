# Experiment Design

This file specifies how to design, configure, and automate the `seq_len` scaling sweep for
MoE gap analysis. It covers sweep point selection, confounder controls, warm-up protocol,
measurement targets, and two automation strategies: a parameterized pytest fixture and a
standalone Python sweep script.

---

## Choosing the Sweep Points

The standard sweep for this guide is:

```python
SEQ_LEN_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096]
```

This set spans two decades (64 to 4096, a 64× range) and is chosen to:

1. **Reveal the scaling exponent.** A two-decade span provides enough dynamic range to
   distinguish slope ≈ 0 (constant) from slope ≈ 1 (linear) in log-log space with
   confidence, even with 10–15% measurement noise.
2. **Stay within practical inference ranges.** 64 tokens is typical for short decode
   sequences; 4096 is the practical upper limit for prefill on a single T3K pass before
   chunking is required.
3. **Be power-of-two.** Powers of two align to tile boundaries (32 divides all values
   evenly), which avoids artificial tile-padding overhead that would obscure the true scaling
   trend. The exception — intentional investigation of tile boundary effects — is handled
   separately in `interpreting_scaling_results.md`.

> **Tip:** If you observe non-monotonic behavior at a specific `seq_len` point and need
> to characterize a tile-count discontinuity, run a supplementary fine-grained sweep around
> that point. For example, if the jump occurs between 512 and 1024, add
> `[512, 544, 576, 640, 704, 768, 896, 1024]` to locate the exact boundary.

---

## Controlling Confounders

Hold the following variables constant across all sweep points. Any parameter not in this
list must remain at its default value throughout the sweep.

| Parameter | Fixed Value | Rationale |
|---|---|---|
| `batch_size` | 1 | Avoids introducing a second scaling dimension |
| `num_experts` | 128 | Architecture constant for Qwen 235B / DeepSeek-V3 |
| `top_k` | 8 | Architecture constant; determines `num_active_tokens = seq_len × top_k` |
| `d_model` | 7168 | Architecture constant; determines CCL message size |
| `d_ff` | 2048 | Architecture constant |
| `dtype` | BF16 | Determines bytes per element (2) |
| Hardware | T3K (8-chip mesh) | Do not run part of the sweep on a different mesh or single-chip |
| `ep_degree` | 8 (all chips) | Expert parallelism degree; must match across all runs |
| TTNN program cache | Warm (3 warm-up iterations before measuring) | Prevents program cache miss (Pattern D) from inflating measurements |
| Device power state | Nominal | Do not run sweep while another process is saturating the ethernet links |

> **Warning:** Running the sweep across a reboot or firmware reload boundary will invalidate
> comparisons across sweep points. If the device is reset between points, re-run the full
> sweep from the beginning.

---

## Warm-Up Protocol

The program cache is cold on the first inference call. Pattern D (program cache miss) adds a
fixed compilation cost of 10–30ms to the first forward pass, which would corrupt the gap
measurement at the first sweep point.

Always execute exactly **3 warm-up iterations** before beginning the timed measurement
iterations. This ensures:

1. The program cache is warm for the specific `seq_len` being measured.
2. DRAM pages for weight tensors are already resident in the memory hierarchy.
3. Any one-time initialization in the MoE router or expert selector has completed.

> **Warning:** Warm-up must be performed separately for each `seq_len` value in the sweep.
> The program cache entry for `seq_len=512` is distinct from the entry for `seq_len=1024`
> because the tensor shapes differ. Running 3 warm-ups at `seq_len=512` does not warm the
> cache for `seq_len=1024`.

---

## What to Measure per Sweep Point

For each `seq_len` value, collect three measurements:

### Measurement 1: Wallclock Time (Python `time.perf_counter`)

Total elapsed time from before the MoE forward pass call to after the call, measured on the
host. This is the coarsest measurement and serves as a sanity check. Use `time.perf_counter_ns()`
for nanosecond resolution to avoid floating-point precision loss.

### Measurement 2: Per-Phase Latency (Device Profiler CSV)

Run with `TT_METAL_DEVICE_PROFILER=1` and read the `ops_perf_results_<timestamp>.csv` file
(see Chapter 3 for the CSV schema). Extract `DEVICE KERNEL DURATION [ns]` for each MoE op
and sum by phase (dispatch, expert matmul, combine). This gives device-side timing,
independent of host-side overhead.

### Measurement 3: Gap Duration (Tracy CSV Export)

Run with Tracy enabled and capture a `.tracy` file. Export to CSV with:

```bash
tracy-csvexport -u output.tracy > zones_seq${SEQ_LEN}.csv
```

Load the CSV in Python and compute the gap between the end of the last zone in the dispatch
phase (`MoE/dispatch`) and the start of the first zone in the expert matmul phase
(`MoE/expert_matmul`). This is the gap duration for Pattern C analysis (CCL all-to-all
latency between dispatch and expert compute). Note: Pattern B is the separate gap between
the end of `MoE/expert_matmul` and the start of `MoE/combine`.

---

## Sample Size and Statistical Rigor

Run **20 timed iterations** per `seq_len` value (after the 3 warm-up iterations). Report
the **median** and **p95** (95th percentile) of the gap duration distribution. Do not report
the mean; it is sensitive to outliers from OS scheduling preemption and DRAM page faults.

```python
import statistics

def compute_stats(samples: list[float]) -> dict:
    """Compute median and p95 from a list of gap durations in ms."""
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    median = statistics.median(sorted_samples)
    p95_idx = int(0.95 * n)  # floor index for p95
    p95 = sorted_samples[min(p95_idx, n - 1)]
    return {"median_ms": median, "p95_ms": p95, "n": n}
```

> **Tip:** If `p95 / median > 2.0` for any `seq_len` point, the measurement is dominated by
> OS jitter at that point. Pin the process to a specific CPU core with `taskset -c 0` and
> re-run before including the point in the scaling plot.

---

## Automation Option 1: Parameterized pytest Fixture

This approach integrates with existing tt-metal test infrastructure and produces a structured
CSV output via pytest's built-in reporting.

```python
# tests/ttnn/moe/test_moe_scaling_sweep.py

import csv
import statistics
import time
import pytest
import torch
import ttnn

# ---------------------------------------------------------------------------
# Model configuration constants
# ---------------------------------------------------------------------------
D_MODEL = 7168
D_FF = 2048
NUM_EXPERTS = 128
TOP_K = 8
NUM_CHIPS = 8
DTYPE = ttnn.bfloat16

SEQ_LEN_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_WARMUP = 3
NUM_TIMED = 20

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    """Open a T3K device mesh and yield it for the duration of the test session."""
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, NUM_CHIPS),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.enable_program_cache(mesh)
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="session")
def moe_weights(device):
    """Initialize MoE expert weights once per session and reuse across seq_len values."""
    # Placeholder: in a real test, load from a checkpoint or initialize randomly.
    gate_weight = ttnn.from_torch(
        torch.randn(NUM_EXPERTS, D_MODEL, D_FF, dtype=torch.bfloat16),
        dtype=DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return {"gate_weight": gate_weight}


@pytest.mark.parametrize("seq_len", SEQ_LEN_SWEEP)
def test_moe_gap_scaling(device, moe_weights, seq_len, tmp_path):
    """
    Measure the MoE gap duration at each seq_len sweep point.
    Results are written to a CSV for downstream analysis.
    """
    hidden_states_torch = torch.randn(seq_len, D_MODEL, dtype=torch.bfloat16)
    hidden_states = ttnn.from_torch(
        hidden_states_torch,
        dtype=DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Warm-up: populate program cache for this seq_len
    for _ in range(NUM_WARMUP):
        _ = moe_forward(device, hidden_states, moe_weights)
        ttnn.synchronize_device(device)

    # Timed iterations
    gap_durations_ms = []
    wallclock_ms = []

    for _ in range(NUM_TIMED):
        t0 = time.perf_counter_ns()
        output = moe_forward(device, hidden_states, moe_weights)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter_ns()

        wallclock_ms.append((t1 - t0) / 1e6)
        # NOTE: For the gap-specific measurement, parse the Tracy CSV or device profiler CSV
        # after each run. Here we record wallclock as a proxy; replace with gap-specific
        # extraction in production use.

    stats = compute_stats(wallclock_ms)

    # Write per-seq_len results to CSV
    out_path = tmp_path / "moe_scaling_results.csv"
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "median_ms", "p95_ms", "n"])
        if write_header:
            writer.writeheader()
        writer.writerow({"seq_len": seq_len, **stats})

    # Soft assertion: gap should be < 20ms even at seq_len=4096
    assert stats["median_ms"] < 20.0, (
        f"Median gap {stats['median_ms']:.2f}ms at seq_len={seq_len} exceeds 20ms threshold"
    )


def compute_stats(samples):
    sorted_s = sorted(samples)
    n = len(sorted_s)
    return {
        "median_ms": statistics.median(sorted_s),
        "p95_ms": sorted_s[min(int(0.95 * n), n - 1)],
        "n": n,
    }
```

Run the sweep with:

```bash
pytest tests/ttnn/moe/test_moe_scaling_sweep.py -v --tb=short 2>&1 | tee sweep_output.log
```

---

## Automation Option 2: Standalone Python Sweep Script

Use this option when you need to run the sweep outside pytest, e.g., for rapid iteration
during investigation or when the pytest fixture infrastructure is not yet set up.

```python
#!/usr/bin/env python3
"""
moe_scaling_sweep.py — Standalone MoE seq_len scaling sweep script.

Usage:
    python moe_scaling_sweep.py --output results/moe_gap_scaling.csv

Outputs a CSV with columns: seq_len, median_ms, p95_ms, n
"""

import argparse
import csv
import statistics
import time
import pathlib

import torch
import ttnn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
D_MODEL = 7168
D_FF = 2048
NUM_EXPERTS = 128
TOP_K = 8
NUM_CHIPS = 8
DTYPE = ttnn.bfloat16

SEQ_LEN_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096]
NUM_WARMUP = 3
NUM_TIMED = 20


# ---------------------------------------------------------------------------
# Placeholder forward pass — replace with actual MoE layer call
# ---------------------------------------------------------------------------
def moe_forward(device, hidden_states):
    """
    Placeholder: call the actual TTNN MoE layer forward pass here.
    Returns output tensor. Must include dispatch, expert matmul, and combine.
    """
    raise NotImplementedError(
        "Replace this with the actual MoE forward pass call from your model."
    )


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(samples: list[float]) -> dict:
    sorted_s = sorted(samples)
    n = len(sorted_s)
    return {
        "median_ms": statistics.median(sorted_s),
        "p95_ms": sorted_s[min(int(0.95 * n), n - 1)],
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(output_path: pathlib.Path) -> None:
    print(f"Opening T3K device mesh ({NUM_CHIPS} chips)")
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, NUM_CHIPS),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.enable_program_cache(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        for seq_len in SEQ_LEN_SWEEP:
            print(f"\n[seq_len={seq_len}] Preparing input tensor...")
            hidden_states_torch = torch.randn(seq_len, D_MODEL, dtype=torch.bfloat16)
            hidden_states = ttnn.from_torch(
                hidden_states_torch,
                dtype=DTYPE,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # Warm-up: populate program cache for this seq_len specifically
            print(f"[seq_len={seq_len}] Running {NUM_WARMUP} warm-up iterations...")
            for _ in range(NUM_WARMUP):
                _ = moe_forward(device, hidden_states)
                ttnn.synchronize_device(device)

            # Timed iterations
            print(f"[seq_len={seq_len}] Running {NUM_TIMED} timed iterations...")
            wallclock_ms = []
            for i in range(NUM_TIMED):
                t0 = time.perf_counter_ns()
                _ = moe_forward(device, hidden_states)
                ttnn.synchronize_device(device)
                t1 = time.perf_counter_ns()
                elapsed_ms = (t1 - t0) / 1e6
                wallclock_ms.append(elapsed_ms)
                if (i + 1) % 5 == 0:
                    print(f"  iteration {i+1}/{NUM_TIMED}: {elapsed_ms:.2f} ms")

            stats = compute_stats(wallclock_ms)
            results.append({"seq_len": seq_len, **stats})
            print(
                f"[seq_len={seq_len}] median={stats['median_ms']:.2f} ms  "
                f"p95={stats['p95_ms']:.2f} ms"
            )

    finally:
        ttnn.close_mesh_device(device)

    # Write results CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_len", "median_ms", "p95_ms", "n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE seq_len scaling sweep")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/moe_gap_scaling.csv"),
        help="Path for the output CSV file",
    )
    args = parser.parse_args()
    run_sweep(args.output)
```

Run the standalone script with:

```bash
TT_METAL_DEVICE_PROFILER=1 TRACY_NO_EXIT=1 python moe_scaling_sweep.py \
    --output results/moe_gap_scaling.csv
```

Setting `TT_METAL_DEVICE_PROFILER=1` captures the device profiler CSV alongside the
wallclock measurements. Setting `TRACY_NO_EXIT=1` prevents Tracy from flushing the trace
before the script exits, ensuring the full sweep is captured in one `.tracy` file.

---

## Output CSV Format

Both automation options produce a CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| `seq_len` | int | Sequence length for this sweep point |
| `median_ms` | float | Median latency across 20 iterations (ms) |
| `p95_ms` | float | 95th-percentile latency across 20 iterations (ms) |
| `n` | int | Number of timed iterations (always 20) |

Example output:

```
seq_len,median_ms,p95_ms,n
64,14.21,14.38,20
128,14.35,14.52,20
256,14.58,14.79,20
512,14.94,15.18,20
1024,15.82,16.10,20
2048,17.45,17.88,20
4096,20.71,21.30,20
```

In this hypothetical output, the gap grows from ~14.2ms at `seq_len=64` to ~20.7ms at
`seq_len=4096`. The ~14ms floor at small `seq_len` indicates a large constant term
(synchronization barrier), while the ~6.5ms increase from 64 to 4096 indicates a
smaller linear term (CCL or host Python). `interpreting_scaling_results.md` explains how
to decompose and attribute these two components.

---

---

**Next:** [`interpreting_scaling_results.md`](./interpreting_scaling_results.md)
