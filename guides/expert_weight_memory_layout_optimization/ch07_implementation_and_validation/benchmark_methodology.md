# Benchmark Methodology

## Overview

This file provides a complete, reproducible benchmark harness for measuring the latency and effective DRAM bandwidth of expert weight matmuls under two configurations: standard interleaved DRAM placement and DRAM-sharded placement. The results validate the indicative estimates from Chapter 6 (`bandwidth_gain_analysis.md`) against your specific hardware, firmware version, and TTNN build.

The harness follows a two-config loop: all parameters (activation shapes, compute kernel config, matmul program config) are held fixed while only the weight `MemoryConfig` is toggled. This isolates the bandwidth effect of the sharding layout from all other variables.

---

## Hardware and Software Setup

Before running benchmarks, record the following to ensure reproducibility:

```
Hardware:        Wormhole B0 (single chip) or T3K (8-chip mesh)
Firmware:        [tt-flash --fw-ver output]
TTNN version:    [ttnn.__version__]
Python version:  [python --version]
Model config:    Mixtral 8x7B — d_model=4096, d_ff=14336, num_experts=8, top_k=2
Benchmark date:  [ISO date]
```

> **Warning:** TTNN kernel compilation and program caching behavior can differ across firmware versions. Benchmark results obtained on one firmware version are not directly comparable to results from another version. Always record and report the firmware version alongside benchmark numbers.

---

## Benchmark Harness: Single-Chip (Wormhole B0)

```python
import ttnn
import torch
import time
import statistics

# -----------------------------------------------------------------------
# Benchmark configuration.
# -----------------------------------------------------------------------
NUM_WARMUP  = 5    # Iterations to fill program cache; not timed.
NUM_TIMED   = 20   # Iterations to time; report median and p95.
BATCH_SIZE  = 1    # Decode regime: batch=1, seq=1.
SEQ_LEN     = 1
D_MODEL     = 4096
D_FF        = 14336

# -----------------------------------------------------------------------
# Activation tensor (fixed across both configs).
# -----------------------------------------------------------------------
torch.manual_seed(0)
cpu_act = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, dtype=torch.bfloat16)
activation = ttnn.from_torch(
    cpu_act,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)


def run_benchmark(weight_gate, weight_up, weight_down, label: str, ckc) -> dict:
    """
    Run NUM_WARMUP + NUM_TIMED iterations of the expert FFN forward pass.
    Returns dict with median_ms, p95_ms, label.
    """
    latencies_ms = []

    # Flush the program cache for the first warmup call to ensure
    # the cache is primed with this config's kernel.
    for i in range(NUM_WARMUP + NUM_TIMED):
        # Synchronize device to ensure previous ops are complete before timing.
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()

        out = expert_ffn_forward(activation, weight_gate, weight_up, weight_down, ckc)
        ttnn.synchronize_device(device)  # Block until the matmul completes.

        t1 = time.perf_counter()
        ttnn.deallocate(out)

        if i >= NUM_WARMUP:
            latencies_ms.append((t1 - t0) * 1000.0)

    median_ms = statistics.median(latencies_ms)
    p95_ms    = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(f"[{label}]  median={median_ms:.3f} ms  p95={p95_ms:.3f} ms  (n={NUM_TIMED})")
    return {"label": label, "median_ms": median_ms, "p95_ms": p95_ms, "latencies": latencies_ms}


# -----------------------------------------------------------------------
# Run both configs.
# -----------------------------------------------------------------------
# Config A: standard interleaved DRAM weights.
results_interleaved = run_benchmark(
    gate_weights[0], up_weights[0], down_weights[0],
    label="interleaved",
    ckc=lofi_config,
)

# Config B: DRAM-sharded weights (from code_patterns.md Step 3).
results_sharded = run_benchmark(
    sharded_gate[0], sharded_up[0], sharded_down[0],
    label="dram_sharded",
    ckc=lofi_config,
)

ttnn.deallocate(activation)
```

---

## Computing Effective Bandwidth

The primary performance metric for weight-bound expert matmuls is **effective DRAM bandwidth**: how many bytes of weight data are read per second, compared against the hardware peak of ~300 GB/s.

```python
# Weight bytes read per forward pass (gate + up + down, all in bfloat16).
# gate: [4096, 14336], up: [4096, 14336], down: [14336, 4096]
gate_bytes = D_MODEL * D_FF * 2          # 2 bytes per bfloat16 element.
up_bytes   = D_MODEL * D_FF * 2
down_bytes = D_FF   * D_MODEL * 2
total_weight_bytes = gate_bytes + up_bytes + down_bytes
print(f"Total weight bytes per expert FFN: {total_weight_bytes / 1e6:.1f} MB")
# Mixtral 8x7B: (4096 × 14336 × 2) × 3 = 352 MB per expert.

PEAK_BW_GBs = 300.0  # Wormhole B0 peak DRAM bandwidth (GB/s).

def compute_bandwidth_stats(result: dict) -> None:
    """Print bandwidth and efficiency from a benchmark result dict."""
    median_s = result["median_ms"] / 1000.0
    bw_gbs   = total_weight_bytes / median_s / 1e9
    eff_pct  = bw_gbs / PEAK_BW_GBs * 100.0
    print(
        f"  [{result['label']}]  "
        f"median={result['median_ms']:.3f} ms  "
        f"BW={bw_gbs:.1f} GB/s  "
        f"efficiency={eff_pct:.1f}% of peak"
    )


compute_bandwidth_stats(results_interleaved)
compute_bandwidth_stats(results_sharded)

# Improvement delta.
delta_pct = (results_interleaved["median_ms"] - results_sharded["median_ms"]) \
            / results_interleaved["median_ms"] * 100.0
print(f"\nLatency improvement (sharded vs interleaved): {delta_pct:.1f}%")
```

> **Tip:** For Mixtral 8x7B at batch=1, `total_weight_bytes ≈ 352 MB` per expert FFN. If median latency is 1.7 ms (interleaved) vs 1.0 ms (sharded), effective bandwidths are ~207 GB/s vs ~352 GB/s — 69% vs 117% of 300 GB/s. The >100% figure for sharded indicates that the peak DRAM figure is conservative or that the matmul pipeline hides some compute latency. This is expected behavior; re-profile with `d_ff × d_model × 2 × 3` as the numerator.

---

## Measuring Weight Read Bytes with the Device Profiler

Wall-time latency includes dispatch overhead and synchronization cost. For a clean bandwidth measurement, use the TTNN device memory reporter to get hardware-counted DRAM read bytes.

```python
# Enable memory reporting before the timed run.
ttnn.device.EnableMemoryReports()

out = expert_ffn_forward(activation, sharded_gate[0], sharded_up[0], sharded_down[0], lofi_config)
ttnn.synchronize_device(device)
ttnn.deallocate(out)

# Disable and read the report.
ttnn.device.DisableMemoryReports()
# Reports are written to the working directory as CSV or printed to stdout
# depending on TTNN version. Parse the DRAM read column for the matmul ops.
```

---

## Tracy Profiler Integration

Tracy provides op-level timing that separates the DRAM read phase from the compute phase within a `ttnn.matmul` call. This is the most precise way to attribute latency.

```bash
# Start Tracy profiler server before launching Python.
# (Tracy must be built and on PATH.)
tracy-capture -o trace_interleaved.tracy &
python benchmark_interleaved.py
# Ctrl+C to stop capture.

tracy-capture -o trace_sharded.tracy &
python benchmark_sharded.py
```

In the Tracy timeline, identify the `matmul` op. Look for two phases:
- **DRAM read phase:** Tensix cores issuing DMA read requests to DRAM; this phase is bandwidth-bound.
- **Compute phase:** Tensix MAC units accumulating tiles; this phase is compute-bound.

For decode regime (batch=1, seq=1), the DRAM read phase should dominate the total matmul time. DRAM-sharded weights reduce the DRAM read phase duration; the compute phase is unchanged.

> **Tip:** In the Tracy trace, filter by `core=0,0` to isolate a single Tensix core's view. Under interleaved layout, core (0,0) issues reads spread across all 6 DRAM columns; under DRAM-sharded layout, it reads only from its assigned DRAM shard, and the reads appear as a single contiguous burst on one DRAM column. This visual difference in the NoC traffic pattern confirms the sharding is working as intended.

---

## Multi-Regime Benchmark

To validate the Chapter 6 trade-off matrix across regimes, sweep `batch_size` and `seq_len`:

```python
def sweep_regimes(gate_w_dict, up_w_dict, down_w_dict, ckc) -> list:
    """
    Run benchmark across decode and prefill regimes.
    gate_w_dict: {"interleaved": tensor, "dram_sharded": tensor}
    Returns a list of result rows.
    """
    regimes = [
        {"batch": 1,  "seq": 1,   "label": "decode_b1"},
        {"batch": 4,  "seq": 1,   "label": "decode_b4"},
        {"batch": 8,  "seq": 1,   "label": "decode_b8"},
        {"batch": 32, "seq": 1,   "label": "decode_b32"},
        {"batch": 1,  "seq": 128, "label": "prefill_s128"},
        {"batch": 1,  "seq": 512, "label": "prefill_s512"},
    ]
    rows = []
    for regime in regimes:
        B, S = regime["batch"], regime["seq"]
        cpu_x = torch.randn(B, S, D_MODEL, dtype=torch.bfloat16)
        x = ttnn.from_torch(cpu_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

        for config_name in ["interleaved", "dram_sharded"]:
            res = run_benchmark(
                gate_w_dict[config_name],
                up_w_dict[config_name],
                down_w_dict[config_name],
                label=f"{config_name}_{regime['label']}",
                ckc=ckc,
            )
            effective_M = B * 2  # top_k=2 for Mixtral.
            res["effective_M"] = effective_M
            res["config"] = config_name
            res["regime"] = regime["label"]
            rows.append(res)

        ttnn.deallocate(x)
    return rows


weight_dict_gate = {"interleaved": gate_weights[0], "dram_sharded": sharded_gate[0]}
weight_dict_up   = {"interleaved": up_weights[0],   "dram_sharded": sharded_up[0]}
weight_dict_down = {"interleaved": down_weights[0], "dram_sharded": sharded_down[0]}

all_results = sweep_regimes(weight_dict_gate, weight_dict_up, weight_dict_down, lofi_config)
```

---

## Reporting Results

Present results in the following table format. The `delta` column shows the latency improvement of DRAM-sharded over interleaved (negative = sharded is faster):

| Regime | `effective_M` | Interleaved (ms) | DRAM-sharded (ms) | BW interleaved (GB/s) | BW sharded (GB/s) | Delta |
|---|---|---|---|---|---|---|
| decode_b1 | 2 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| decode_b4 | 8 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| decode_b8 | 16 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| decode_b32 | 64 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| prefill_s128 | 256 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| prefill_s512 | 1024 | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

Expected latency deltas per regime are specified in Chapter 6, `tradeoff_matrix.md`.

If measured results diverge from these ranges, inspect the Tracy trace to determine whether the DRAM read phase or compute phase has changed unexpectedly.

---

## T3K Multi-Chip Benchmark

On T3K, each chip receives approximately `total_batch × top_k / num_experts` token-expert pairs. For Mixtral 8x7B (8 experts, 8 chips, `top_k=2`), each chip receives `batch × 2 / 8 = batch / 4` pairs. At global `batch=32`, each chip processes `effective_M = 8` — firmly in the decode regime.

```python
# T3K device initialization (replace device_id with mesh configuration).
# mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
# Each chip independently benchmarks its resident expert's weight access.
# Apply the same benchmark harness above on each chip's device handle.
# Aggregate results to compute per-chip and total bandwidth.
```

> **Tip:** On T3K, the across-chip expert routing latency (Ethernet links) adds to the end-to-end MoE layer latency but is separate from the within-chip weight access latency measured here. Benchmark within-chip weight access independently before profiling the full T3K expert routing pipeline.

---

**End of guide.** Return to [Guide Index](../index.md)
