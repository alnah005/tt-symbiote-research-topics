# Profiling Setup: TTNN Op-Level Timing Infrastructure

This file explains how to enable the TTNN (Tenstorrent Neural Network) device profiler, what output it produces, and how to configure your environment and benchmark script to collect valid per-op timing data for `ttnn.silu`.

---

## 1. Enabling the Device Profiler

The TTNN device profiler is controlled by a single environment variable:

```bash
export TT_METAL_DEVICE_PROFILER=1
python your_benchmark.py
```

> **Warning:** The variable must be set in the shell **before** the Python process starts. Setting it inside the script via `os.environ["TT_METAL_DEVICE_PROFILER"] = "1"` after the interpreter has launched will not activate the profiler, because tt-metal reads this variable during its C++ initialization.

> **Tip:** Add `TT_METAL_DEVICE_PROFILER=1` as a prefix on the command line for one-off runs: `TT_METAL_DEVICE_PROFILER=1 python benchmark_silu.py`. This avoids accidentally leaving it set in your shell for unrelated runs.

---

## 2. Profiler Output: CSV File

When profiling is enabled, tt-metal writes raw device traces after each `ttnn.ReadDeviceProfiler(device)` call (or automatically at program boundaries). The post-processing tool converts these into a human-readable CSV:

```bash
python tt-metal/tools/profiler/process_ops_logs.py
```

When no `--output` flag is given, the script uses a default timestamped filename. The output file is named `ops_perf_results_<timestamp>.csv` and is written to the working directory.

### Relevant CSV Columns

| Column | What it measures | Use for benchmarking? |
|---|---|---|
| `DEVICE KERNEL DURATION [ns]` | On-device hardware execution time — the time between the first Tensix core starting and the last core finishing the kernel | **Yes — use this column** |
| `OP TO OP LATENCY [ns]` | Device-side time between consecutive op boundaries on the device timeline, including host dispatch overhead between ops; not a pure hardware execution metric | **No — do not use for hardware comparison** |
| `OP TYPE` | Operation class name (e.g., `Silu`, `Matmul`) | Yes — used to filter rows |
| `INPUT_0_SHAPE` | Shape of the first input tensor | Yes — used to verify correct benchmark input |

> **Warning:** `OP TO OP LATENCY [ns]` is frequently 3–10× higher than `DEVICE KERNEL DURATION [ns]` for short operations like SiLU at small batch sizes because host dispatch overhead dominates. Using it for hardware comparison will give a misleading picture of device efficiency.

---

## 3. Warm-Up Requirement

TTNN compiles and caches device kernels the first time an operation is dispatched with a given configuration (shape, dtype, layout, and target device). This first-run compilation cost can be hundreds of milliseconds and will completely overwhelm the actual kernel execution time (which is typically tens of microseconds).

**Minimum warm-up: 2 iterations before any timed measurement.**

In practice, 3–5 warm-up iterations is safer because some multi-kernel operations populate the cache in stages.

```python
# Warm-up: populate the program cache — do NOT include in measurements
WARMUP_ITERS = 3
for _ in range(WARMUP_ITERS):
    ttnn.silu(input_tensor)
ttnn.synchronize_device(device)  # ensure all warm-up ops have completed
```

> **Warning:** Omitting warm-up is the single most common cause of inflated latency numbers. A single cold-cache run can show SiLU taking 200ms when the true hardware time is 20µs.

---

## 4. ReadDeviceProfiler Limit

The runtime automatically flushes the device profiler buffer after **1000 operations**; if you run more ops without reading, the flush happens automatically. Call `ttnn.ReadDeviceProfiler(device)` explicitly before the 1000-op limit to ensure data collection at a predictable point:

Call `ttnn.ReadDeviceProfiler(device)` explicitly before reaching this limit:

```python
ttnn.ReadDeviceProfiler(device)
```

For a typical benchmark with 20–100 timed iterations this limit is unlikely to be reached, but it matters during development when you may also be running warm-up ops and diagnostic ops in the same session.

---

## 5. Toolchain Overview

| Tool | Location | Purpose |
|---|---|---|
| `process_ops_logs.py` | `tt-metal/tools/profiler/process_ops_logs.py` | Post-processes raw device trace logs into the `ops_perf_results_<timestamp>.csv` file |
| TTNN Visualizer | `tt-metal/tools/profiler/` | Interactive web-based trace inspector; useful for identifying unexpected ops in the dispatch stream |
| `ttnn.ReadDeviceProfiler` | TTNN Python API | Flushes device profiler data to disk; call explicitly in long-running benchmarks |

---

## 6. Complete Setup Code Example

The following example demonstrates the full initialization-to-profiler-read sequence for a SiLU latency benchmark.

```python
import ttnn
import torch

# ── Device initialization ────────────────────────────────────────────────────
# TT_METAL_DEVICE_PROFILER=1 must already be set in the environment.
device = ttnn.open_device(device_id=0)

# ── Tensor allocation ────────────────────────────────────────────────────────
# Use TILE_LAYOUT and bfloat16 — the layout/dtype that gate_proj matmul produces.
# Shape: [1, 1, num_tokens, hidden_dim]
num_tokens = 32
hidden_dim = 4096

torch_input = torch.randn(1, 1, num_tokens, hidden_dim, dtype=torch.bfloat16)
input_tensor = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# ── Warm-up and timed measurement loop ──────────────────────────────────────
# See isolating_silu_from_matmul.md §2 (Strategy 1) for a complete,
# runnable benchmark including warm-up (WARMUP_ITERS=3) and timed
# loop (TIMED_ITERS=20) with correct synchronization.

# ── Flush profiler data to disk ──────────────────────────────────────────────
ttnn.ReadDeviceProfiler(device)

ttnn.close_device(device)
```

---

---

**Next:** [`isolating_silu_from_matmul.md`](./isolating_silu_from_matmul.md)
