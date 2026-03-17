# Isolating SiLU Latency from Matmul Operations

This file explains why isolating `ttnn.silu` from surrounding matrix multiplication operations requires deliberate benchmark design, and provides two concrete strategies for doing so.

---

## 1. The Measurement Challenge

In a real MoE (Mixture of Experts) forward pass, the SiLU activation is sandwiched between two other operations:

```
gate_proj matmul  →  ttnn.silu  →  element-wise multiply (SiLU output × up_proj output)
```

If you time the entire sequence end-to-end, the profiler CSV will contain one row per operation, but you cannot cleanly attribute the total wall-clock time to any single op without reading individual `DEVICE KERNEL DURATION [ns]` values. More importantly:

- The shapes and dtypes used in a full forward pass may differ from the shapes you want to study in isolation.
- Async dispatch means operations can overlap in the host dispatch queue; `OP TO OP LATENCY [ns]` for one op can be inflated by the cost of the next op's dispatch.
- Fusing or reordering in a real model can change the apparent cost of any single operation.

The two strategies below give you clean, reproducible SiLU hardware time.

---

## 2. Isolation Strategy 1 — Standalone Benchmark

Allocate a pre-filled input tensor that matches the shape and format that gate_proj matmul would produce, then call `ttnn.silu` on it alone.

**When to use:** When you want the pure, unconditional hardware cost of SiLU for a given input shape — the number to report in a latency table or compare against matmul.

### Tensor Requirements

> **Warning:** The input tensor **must** use `TILE_LAYOUT` and `ttnn.bfloat16`. Using `ROW_MAJOR_LAYOUT` or `ttnn.float32` triggers format-conversion kernels that run before the SiLU kernel itself. The CSV will still show a single row for the SiLU op, but its `DEVICE KERNEL DURATION` will include the conversion overhead, inflating the result.

| Requirement | Correct value | Incorrect value (and consequence) |
|---|---|---|
| Layout | `ttnn.TILE_LAYOUT` | `ttnn.ROW_MAJOR_LAYOUT` → layout conversion kernel prepended |
| dtype | `ttnn.bfloat16` | `ttnn.float32` → dtype conversion kernel prepended |
| Shape | `[1, 1, num_tokens, hidden_dim]` where `hidden_dim` is the gate_proj output width | Arbitrary shape → different core grid, different DRAM access pattern |

### Code Example

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

# Input shape matches gate_proj matmul output for this benchmark point.
# hidden_dim is typically 4 * d_model for the FFN intermediate dimension.
num_tokens = 32
hidden_dim = 4096  # adjust to the gate_proj output width under test

torch_input = torch.randn(1, 1, num_tokens, hidden_dim, dtype=torch.bfloat16)

# Allocate tensor with correct layout and dtype from the start.
input_tensor = ttnn.from_torch(
    torch_input,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
)

# Warm-up: populates the program cache — see profiling_setup.md §3 for rationale.
WARMUP_ITERS = 3
for _ in range(WARMUP_ITERS):
    ttnn.silu(input_tensor)
ttnn.synchronize_device(device)

# Timed loop: these are the rows that appear in the profiler CSV.
TIMED_ITERS = 20
for _ in range(TIMED_ITERS):
    output = ttnn.silu(input_tensor)  # standalone — no matmul before or after

# Flush profiler data.
ttnn.ReadDeviceProfiler(device)
ttnn.close_device(device)

# In the resulting CSV, filter for OP TYPE == "Silu".
# Read DEVICE KERNEL DURATION [ns] for those 20 rows and compute median / p95.
```

---

## 3. Isolation Strategy 2 — Difference Measurement

Benchmark `matmul` alone and `matmul + ttnn.silu` with identical inputs; subtract the matmul-only median from the combined-sequence median. The delta is the SiLU cost in context.

**When to use:** When you want to understand the incremental cost of adding SiLU to a matmul pipeline — for example, to assess whether fusing SiLU into the matmul kernel would be worthwhile. This strategy is also useful as a cross-check against Strategy 1.

> **Tip:** Because each profiler CSV row gives the device kernel time for a single op, you can also achieve the same result by running both ops in the same benchmark and reading the individual `DEVICE KERNEL DURATION` rows instead of subtracting host-side totals. Strategy 2 is most valuable when you want a single combined number for the sequence.

### Async Dispatch Note

TTNN operations are dispatched asynchronously to the device command queue. If you are using **host-side timers** (e.g., `time.perf_counter`) rather than the profiler CSV, you must call `ttnn.synchronize_device(device)` before stopping the timer; otherwise the host timer captures only dispatch latency, not device execution time. This is not needed when reading `DEVICE KERNEL DURATION [ns]` from the profiler CSV, which captures device time directly.

### Code Example

```python
import ttnn
import torch
import statistics

device = ttnn.open_device(device_id=0)

num_tokens = 32
d_model = 1024
hidden_dim = 4 * d_model  # gate_proj output width

# Input to the matmul (token embeddings).
torch_x = torch.randn(1, 1, num_tokens, d_model, dtype=torch.bfloat16)
# Weight matrix for gate_proj.
torch_w = torch.randn(1, 1, d_model, hidden_dim, dtype=torch.bfloat16)

x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w = ttnn.from_torch(torch_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# ── Warm-up both paths (see profiling_setup.md §3 for rationale) ─────────────
WARMUP_ITERS = 3
for _ in range(WARMUP_ITERS):
    gate = ttnn.matmul(x, w)           # matmul alone
    ttnn.silu(gate)                     # matmul + silu
ttnn.synchronize_device(device)

# ── Timed loop: matmul only ──────────────────────────────────────────────────
# Read DEVICE KERNEL DURATION for the Matmul rows in the CSV.
TIMED_ITERS = 20
for _ in range(TIMED_ITERS):
    gate = ttnn.matmul(x, w)

ttnn.ReadDeviceProfiler(device)  # flush matmul-only measurements

# ── Timed loop: matmul + silu ────────────────────────────────────────────────
# Read DEVICE KERNEL DURATION for the Silu rows in the CSV.
for _ in range(TIMED_ITERS):
    gate = ttnn.matmul(x, w)
    ttnn.silu(gate)

ttnn.ReadDeviceProfiler(device)  # flush combined measurements

ttnn.close_device(device)

# Post-processing (conceptual — use actual CSV values):
#   matmul_median_ns  = median of Matmul DEVICE KERNEL DURATION rows
#   combined_median_ns = median of (Matmul + Silu) DEVICE KERNEL DURATION rows
#   silu_delta_ns = combined_median_ns - matmul_median_ns
#
# Cross-check: silu_delta_ns should match the median of Silu rows directly.
```

---

## 4. Choosing Between the Two Strategies

| Consideration | Strategy 1 (Standalone) | Strategy 2 (Difference) |
|---|---|---|
| Measures pure hardware time | Yes | Approximately (small noise from matmul variation) |
| Captures in-context scheduling effects | No | Yes |
| CSV interpretation | Read Silu rows directly | Read both Matmul and Silu rows; subtract |
| Recommended for latency tables | Yes | Use as cross-check |
| Recommended for fusion analysis | No | Yes |

---

## Next Steps

Proceed to [`measurement_methodology.md`](measurement_methodology.md) for the recommended input shapes, statistical protocol (median and p95 over 20+ iterations), and a complete pitfalls table covering all common measurement errors.
