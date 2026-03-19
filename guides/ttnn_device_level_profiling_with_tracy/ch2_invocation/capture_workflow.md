# Capture Workflow

## The Three Components

A complete profiling session involves two processes: the profiled process and exactly one of the two Tracy receiver options.

1. **The profiled pytest process** — your test binary, linked against the Tracy client and the tt-metal device profiler runtime. This process produces all profiling data; it is the source of both the Tracy event stream and the device cycle-counter readings. The Tracy client opens a single outbound TCP connection and accepts exactly one receiver.

2. **`tracy-capture` (headless/batch capture)** — a standalone binary built from the Tracy submodule. It listens on a local TCP port, receives the event stream that the Tracy client background thread pushes out of the ring buffer, and writes it to a binary `.tracy` file. It must be started **before** the profiled process initializes or the connection attempt from the Tracy client will be refused. Use this for CI/batch workflows where you want a `.tracy` file for later analysis.

   **OR**

3. **Tracy GUI (live interactive viewing)** — the interactive `tracy` binary, also built from the submodule. It can connect directly to the profiled process for a live view of incoming zones while the test runs.

> **Important:** `tracy-capture` and the Tracy GUI are mutually exclusive — the profiled process accepts only one connection at a time. You cannot run both simultaneously. Choose `tracy-capture` for batch capture to a file, or the Tracy GUI for live interactive inspection.

## Step-by-Step Capture Procedure

### Terminal A — Start `tracy-capture`

```bash
# Navigate to the version-matched binary (built from the tt-metal submodule)
cd tt_metal/third_party/tracy/capture/build/unix

# Start the capture server
# -o profile.tracy  : output file path
# -f                : overwrite output file if it already exists (required for repeated runs)
./tracy-capture -o /tmp/profile.tracy -f
```

You should see output similar to:

```
Tracy Profiler capture - 0.10.0
Listening on port 8086...
```

Leave this terminal open. `tracy-capture` blocks waiting for the profiled process to connect.

### Terminal B — Set environment variables and run pytest

```bash
# Required: activate device CSV profiler
export TT_METAL_DEVICE_PROFILER=1

# Recommended: include dispatch cores for host-device time anchoring
export TT_METAL_DEVICE_PROFILER_DISPATCH_CORES=1

# Required for batch capture: prevent early exit before Tracy drains its buffer
export TRACY_NO_EXIT=1

# Recommended: deterministic starting state between runs
export TT_METAL_CLEAR_L1=1

# Optional: add sync barriers for accurate per-op latency (omit for throughput measurement)
# export TT_METAL_PROFILER_SYNC=1

# Run the test
# -s: do not capture stdout (required to see Tracy connection messages)
pytest tests/ttnn/my_test.py -s -v
```

When the Tracy client in the profiled process initializes, Terminal A will print:

```
Connection from 127.0.0.1
Receiving data...
```

When the profiled process exits (after TRACY_NO_EXIT=1 ensures the drain completes), Terminal A will print:

```
Frames: NNN
Time span: N.NNs
Saving to profile.tracy... done
```

### Verifying Successful Capture

After both processes have exited:

```bash
# 1. Verify the Tracy binary database is non-empty
ls -lh /tmp/profile.tracy
# Expected: file size in the range of hundreds of KB to tens of MB depending on test length

# 2. Verify the device profiler CSV exists and contains data rows
# The CSV is written to generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv
# Use TT_METAL_HOME if set; otherwise fall back to the repo-relative path.
CSV_PATH="${TT_METAL_HOME:-.}/generated_profile_log_${TEST_NAME}/ops_perf_results_${TIMESTAMP}.csv"
ls -lh "$CSV_PATH"

# Count data rows (subtract 1 for the header)
wc -l "$CSV_PATH"
# Expected: at least 2 lines (1 header + 1 data row) for even a minimal single-op test
```

## Output Artifacts

| Artifact | Location | Produced by |
|---|---|---|
| `profile.tracy` | Path given to `tracy-capture -o` | Tracy client + `tracy-capture` server |
| `ops_perf_results.csv` | `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` | Device profiler (`TT_METAL_DEVICE_PROFILER=1`) |
| Per-core cycle logs | `tt_metal/tools/profiler/logs/` (subdirectories) | Device profiler, one file per Tensix core |

The per-core cycle log subdirectory structure is `logs/<device_id>/<op_id>/core_<x>_<y>.log`. These files contain the raw cycle-counter readings before aggregation into the CSV. They are useful for diagnosing load imbalance across cores — if one core's `DEVICE KERNEL DURATION` is significantly longer than the median, the per-core logs identify which physical core is the straggler.

## Common Failure Mode 1: Tracy Client/Server Version Mismatch

**Symptom:** `tracy-capture` prints "Connection from 127.0.0.1" and then immediately prints "Connection dropped" or "Protocol mismatch". The output `profile.tracy` is created but is either zero bytes or contains no events (the Tracy GUI shows an empty timeline).

**Root cause:** The `tracy-capture` binary was built from a different Tracy tag than the Tracy client compiled into tt-metal. Tracy's binary protocol is versioned and incompatible across releases. Even a minor version difference causes the handshake to fail.

**Fix:**

```bash
# Step 1: Identify which Tracy commit is pinned in tt-metal
cd tt_metal/third_party/tracy
git log --oneline -1
# e.g., "abc1234 Tracy v0.10.0"

# Step 2: Rebuild tracy-capture from that exact commit
cd capture/build/unix
make clean && make -j$(nproc)

# Step 3: Confirm versions match
./tracy-capture
# The version string appears in the startup banner (e.g., "Tracy Profiler capture - 0.10.0").
# tracy-capture does not support a --version flag; run it without arguments to see the banner.
```

> **Warning:** Do not install Tracy from your system package manager (`apt install tracy`) and use that binary alongside a tt-metal build. Package manager versions are almost never synchronized with the submodule pin. Always build from `tt_metal/third_party/tracy/`.

## Common Failure Mode 2: `TT_METAL_DEVICE_PROFILER=1` Set but Build Lacks `ENABLE_PROFILER`

**Symptom:** `ops_perf_results.csv` is created and contains the correct number of rows (one per op), but every value in the `DEVICE KERNEL DURATION` column (and all other timing columns) is `0`. The CSV header is present and correctly formed.

**Root cause:** `TT_METAL_DEVICE_PROFILER=1` tells the tt-metal runtime to read cycle-counter values and write them to the CSV. However, if `ENABLE_PROFILER=ON` was not set at build time, `kernel_profiler.hpp` was compiled out of all kernel binaries. There are no cycle-counter reads in the device code; the host simply reads back zero from uninitialized memory regions.

**Fix:**

```bash
# Step 1: Confirm the build is missing profiler symbols
nm -C build/tt_metal/libtt_metal.so | grep -c "tracy::"
# If this prints 0, the build does not have ENABLE_PROFILER=ON

# Step 2: Reconfigure and rebuild
cmake -B build -DENABLE_PROFILER=ON [other flags] ..
cmake --build build --target tt_metal ttnn -j$(nproc)

# Step 3: Re-run the capture session
```

> **Tip:** A quick heuristic: if `ops_perf_results.csv` has the correct number of rows but every `DEVICE KERNEL DURATION` is `0`, the build is the problem. If the CSV is missing entirely or has zero rows, `TT_METAL_DEVICE_PROFILER=1` was not set at runtime.

## Minimal Working Example

The following self-contained pytest demonstrates the complete profiling workflow for a single `ttnn.matmul` call. It verifies that `ops_perf_results.csv` exists and is non-empty before the test passes.

```python
# tests/ttnn/test_matmul_profiled.py
"""
Minimal profiling smoke test: runs ttnn.matmul with both profilers active
and asserts that ops_perf_results.csv contains at least one data row.

Run with:
    export TT_METAL_DEVICE_PROFILER=1
    export TRACY_NO_EXIT=1
    pytest tests/ttnn/test_matmul_profiled.py -s -v
"""
import glob
import os
import csv
import pathlib

import pytest
import torch
import ttnn


# Path pattern: generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv
log_dirs = sorted(pathlib.Path(".").glob("generated_profile_log_*"))
assert log_dirs, "No profiler log directory found"
csv_files = sorted(log_dirs[-1].glob("ops_perf_results_*.csv"))
assert csv_files, f"No ops_perf_results CSV in {log_dirs[-1]}"
CSV_PATH = csv_files[-1]
assert CSV_PATH.exists()


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


def test_matmul_produces_profiler_output(device):
    # Arrange: two small matrices that will produce a meaningful kernel dispatch
    a_torch = torch.randn(128, 256, dtype=torch.bfloat16)
    b_torch = torch.randn(256, 128, dtype=torch.bfloat16)

    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Act: run the op under profiling
    c = ttnn.matmul(a, b)

    # Force host-device synchronization so device profiler data is flushed
    ttnn.synchronize_device(device)

    # Assert: device profiler CSV must exist and contain at least one data row
    assert CSV_PATH.exists(), (
        f"ops_perf_results.csv not found at {CSV_PATH}. "
        "Ensure TT_METAL_DEVICE_PROFILER=1 is set."
    )

    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, (
        "ops_perf_results.csv exists but contains no data rows. "
        "If all DEVICE KERNEL DURATION values are 0, the build may be "
        "missing ENABLE_PROFILER=ON. See build_requirements.md."
    )

    # Verify that timing data is non-zero (catches the ENABLE_PROFILER=OFF case)
    duration_col = "DEVICE KERNEL DURATION [ns]"
    assert duration_col in rows[0], (
        f"Expected column '{duration_col}' not found in CSV. "
        f"Available columns: {list(rows[0].keys())}"
    )

    durations = [int(row[duration_col]) for row in rows if row[duration_col].strip()]
    assert any(d > 0 for d in durations), (
        "All DEVICE KERNEL DURATION values are 0. "
        "Rebuild tt-metal with ENABLE_PROFILER=ON."
    )

    print(f"\nProfiling assertions passed.")
    print(f"  CSV rows: {len(rows)}")
    print(f"  Max DEVICE KERNEL DURATION: {max(durations)} ns")
```

Run the test as follows:

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH_CORES=1
export TRACY_NO_EXIT=1
export TT_METAL_CLEAR_L1=1

# In a separate terminal, start tracy-capture first:
# ./tt_metal/third_party/tracy/capture/build/unix/tracy-capture -o /tmp/profile.tracy -f

pytest tests/ttnn/test_matmul_profiled.py -s -v
```

After the test passes, open `profile.tracy` in the Tracy GUI to inspect the host-side dispatch zones, and inspect `generated_profile_log_{test_name}/ops_perf_results_{timestamp}.csv` for the `DEVICE KERNEL DURATION` of the matmul kernel.

---

**Next:** [Chapter 3 — Reading the ops_perf_results CSV](../ch3_csv_reference/index.md)
