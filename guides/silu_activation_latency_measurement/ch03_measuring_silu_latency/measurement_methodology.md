# Measurement Methodology: Protocol, Pitfalls, and Expected Results

This file specifies the recommended input shapes, statistical protocol, and interpretation guidelines for measuring `ttnn.silu` latency. It also provides a complete pitfalls table and expected result ranges to validate that your measurements are in the right ballpark.

---

## 1. Input Shapes for MoE Expert FFN Benchmarking

MoE (Mixture of Experts) FFN (feed-forward network) benchmarks should sweep over both the token dimension and the hidden dimension to capture the full operating range from autoregressive decode (small batches) to prefill (large batches).

Use tensors of shape `[1, 1, num_tokens, hidden_dim]`:

| `num_tokens` | Regime | Typical use case |
|---|---|---|
| 1 | Decode (single token) | Autoregressive generation, single user |
| 8 | Decode (small batch) | Batched decode, 8 concurrent users |
| 32 | Decode (medium batch) | Batched decode, 32 concurrent users |
| 128 | Prefill (short context) | Prompt processing, short sequences |

| `hidden_dim` | Approximate model scale |
|---|---|
| 2048 | ~7B parameter models (intermediate FFN width) |
| 4096 | ~13B–34B parameter models |
| 8192 | ~70B parameter models |

Run all `num_tokens × hidden_dim` combinations for a complete benchmark matrix (12 data points for the table above).

> **Tip:** Always verify that `hidden_dim` is a multiple of 32 (required for `TILE_LAYOUT`). All values in the table above satisfy this requirement.

---

## 2. Statistical Protocol

Device dispatch exhibits jitter: occasional outlier iterations where the device stalls for a host-side reason (cache invalidation, DRAM refresh collision, OS scheduling) inflate the mean significantly without reflecting typical hardware performance.

**Recommended protocol:**

1. Run **3 warm-up iterations** (minimum 2) before any timed measurement.
2. Run **20 timed iterations** (minimum; 50 is preferred for p95 stability).
3. Collect `DEVICE KERNEL DURATION [ns]` from the profiler CSV for each Silu row in the timed section.
4. Report **median** as the primary latency figure.
5. Report **p95** (95th percentile) as the worst-case figure for scheduling analysis.
6. Do not report mean; outliers caused by dispatch jitter will inflate it by 2–5× in some runs.

```python
import statistics

# Example post-processing of durations read from the CSV.
# durations_ns is a list of DEVICE KERNEL DURATION [ns] values for the timed iterations.
durations_ns = [...]  # populated from CSV

median_us = statistics.median(durations_ns) / 1_000   # convert ns → µs
p95_us    = sorted(durations_ns)[int(0.95 * len(durations_ns)) - 1] / 1_000

print(f"Median SiLU latency: {median_us:.2f} µs")
print(f"p95 SiLU latency:    {p95_us:.2f} µs")
```

---

## 3. Reading Results: DEVICE KERNEL DURATION vs. OP TO OP LATENCY

> The profiler CSV contains two latency columns. **Always use `DEVICE KERNEL DURATION [ns]`** for hardware comparisons; see [`profiling_setup.md` §2](profiling_setup.md) for full column definitions and the rationale for excluding `OP TO OP LATENCY [ns]`.

For a SiLU kernel at `num_tokens=1, hidden_dim=4096`, a typical split illustrates why this matters:

- `DEVICE KERNEL DURATION` ≈ 12 µs
- `OP TO OP LATENCY` ≈ 80–150 µs

The OP TO OP LATENCY is dominated by host dispatch overhead and is not a useful measure of device capability.

---

## 4. Multi-Core Behavior and Grid Normalization

`ttnn.silu` launches on the full Tensix grid by default. The grid size varies with input shape:

- Small inputs (`num_tokens=1, hidden_dim=2048`) may use a partial grid (e.g., 4×4 cores).
- Large inputs (`num_tokens=128, hidden_dim=8192`) use the full grid (e.g., 8×8 cores on Wormhole).

When comparing SiLU latency across different `hidden_dim` values, note the core count in the CSV (`CORE_COUNT` column if available, or infer from `DEVICE KERNEL DURATION` scaling). A 4× increase in `hidden_dim` with the same core count produces ~4× longer kernel time; a 4× increase that also doubles the core count produces ~2× longer kernel time.

> **Tip:** If you are comparing SiLU efficiency across configurations, normalize by `(num_tokens × hidden_dim) / core_count` to get throughput per core per cycle rather than comparing raw kernel durations.

---

## 5. Pitfalls Table

| Pitfall | Symptom | Root cause | Fix |
|---|---|---|---|
| No warm-up before timed iterations | First measurement is 100–1000× higher than subsequent ones | First-run kernel compilation is included in device time | Add at least 3 warm-up iterations before the timed loop; call `ttnn.synchronize_device` after warm-up |
| `ROW_MAJOR_LAYOUT` input tensor | SiLU `DEVICE KERNEL DURATION` is 2–5× higher than expected; a layout-conversion op appears in the CSV before the Silu row | TTNN inserts a layout-conversion kernel when the input is not in `TILE_LAYOUT` | Allocate input with `layout=ttnn.TILE_LAYOUT` from the start |
| DRAM-backed tensor instead of L1-sharded | Throughput is lower than expected for small tensors; does not match L1-bandwidth model | DRAM bandwidth ceiling (~300 GB/s) is lower than L1 bandwidth; small tensors that fit in L1 should be L1-sharded for peak performance | Use `ttnn.MemoryConfig` with `TensorMemoryLayout.INTERLEAVED` and `BufferType.L1` for tensors that fit in L1 |
| Reading `OP TO OP LATENCY` instead of `DEVICE KERNEL DURATION` | Measured SiLU latency matches host dispatch overhead (tens to hundreds of µs) instead of device kernel time (single-digit to tens of µs) | `OP TO OP LATENCY` includes host overhead that dwarfs device time for fast ops | Filter for `DEVICE KERNEL DURATION [ns]` column only |
| dtype mismatch (`float32` instead of `bfloat16`) | A dtype-conversion op appears in the CSV before the Silu row; SiLU duration appears inflated | TTNN inserts a dtype-conversion kernel when input dtype does not match the op's native dtype | Allocate input with `dtype=ttnn.bfloat16` |
| ReadDeviceProfiler limit exceeded | Unexpected flush mid-session; fewer than expected Silu rows in a given CSV batch | The runtime auto-flushes the device profiler buffer at ~1000 ops; if you run more ops without reading, the flush happens automatically | Call `ttnn.ReadDeviceProfiler(device)` explicitly before the 1000-op limit to ensure data collection at a predictable point |

---

## 6. Expected Results Preview

These ranges are approximate and depend on hardware generation (Wormhole n150/n300), clock frequency, and DRAM configuration. Use them only to validate that your measurements are not obviously wrong.

### Decode regime (`num_tokens ∈ {1, 8, 32}`)

At decode batch sizes, both SiLU and gate_proj matmul are memory-bandwidth-bound (activation map reads dominate). SiLU latency is typically **15–40% of gate_proj matmul time**.

- `num_tokens=1, hidden_dim=4096`: SiLU ≈ 10–25 µs; gate_proj matmul ≈ 30–80 µs
- `num_tokens=32, hidden_dim=4096`: SiLU ≈ 15–35 µs; gate_proj matmul ≈ 60–150 µs

The SiLU-to-matmul ratio is relatively stable across decode batch sizes because both ops scale with the same activation map size.

### Prefill regime (`num_tokens ≥ 128`)

At prefill batch sizes, gate_proj matmul becomes compute-bound (matrix multiply reuses weights across many tokens) while SiLU remains memory-bandwidth-bound. SiLU falls to **below 5% of total FFN time**.

- `num_tokens=128, hidden_dim=4096`: SiLU ≈ 20–50 µs; gate_proj matmul ≈ 400–1200 µs

> **Tip:** If your measured SiLU latency is more than 2× the upper bounds above, check for the pitfalls in Section 5 before investigating hardware-specific issues.

> **Warning:** These ranges are measured on Wormhole with tt-metal release builds. Debug builds, non-default memory layouts, or custom SFPU (Special Function Processing Unit) configurations can produce significantly different numbers. Always record your tt-metal version and device model alongside your measurements.

---

---

**Next:** [Chapter 4 — SiLU vs. Matmul Comparison](../ch04_silu_vs_matmul_comparison/index.md)
