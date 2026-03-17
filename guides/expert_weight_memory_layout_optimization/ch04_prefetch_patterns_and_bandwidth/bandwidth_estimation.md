# Bandwidth Estimation

## Simple Bandwidth Model

Effective bandwidth during a kernel execution is:

```
effective_bandwidth (GB/s) = total_bytes_read / kernel_time_s / 1e9
```

`total_bytes_read` is the number of bytes transferred from DRAM to L1 during the kernel. For a matmul consuming a weight tensor of shape `[K, N]` in BF16:

```
total_bytes_read = K * N * 2  (weight bytes)
                + M * K * 2  (activation bytes, if not already in L1)
```

In decode (M=1), activation bytes are negligible relative to weight bytes when K and N are large. For Mixtral w1: K=4096, N=14336 → weight bytes = 4096 * 14336 * 2 = 117,440,512 bytes (~112 MB).

---

## Measurement with TTNN Tools

**Option 1: `ttnn.device.EnableMemoryReports()`**

Enables per-operation memory transfer logging. After calling the matmul, inspect the report for `dram_read_bytes` and `kernel_duration_ns`:

```python
ttnn.device.EnableMemoryReports()
output = ttnn.matmul(activation, weight_sharded, ...)
# Reports printed to stdout or accessible via device report API
```

**Option 2: Tracy profiling**

Tracy captures hardware performance counters including DRAM read bytes and kernel wall time at cycle resolution. Steps:
1. Build tt-metal with `TT_METAL_TRACY=1`.
2. Run the workload; Tracy captures a `.tracy` file.
3. Open in Tracy viewer; filter by kernel name; read `dram_bw` counter.

Tracy is the preferred method for production bandwidth characterization because it captures per-core counters and separates DRAM read time from NoC transit time.

---

## Expected Bandwidth: Interleaved Layout

Under full-grid matmul with interleaved DRAM weight access, empirical results on Wormhole B0:

| Workload | Theoretical Peak | Measured (Interleaved) | Efficiency |
|---|---|---|---|
| Full-grid matmul, large K | 300 GB/s | 180–240 GB/s | 60–80% |
| Decode (M=1), small batch | 300 GB/s | 160–200 GB/s | 53–67% |

The decode regime is worse because M=1 means each core issues fewer tile requests, so the pipeline depth is shallow and NoC stalls are not amortized over many in-flight requests.

---

## Expected Improvement: DRAM-Sharded Layout

DRAM-sharded layout targets effective bandwidth closer to the ~300 GB/s peak by eliminating NoC hotspots. Measured improvement:

| Layout | Effective Bandwidth | vs. Interleaved |
|---|---|---|
| Interleaved | ~180–200 GB/s (decode) | baseline |
| WIDTH_SHARDED, DRAM | ~260–290 GB/s (decode) | +30–50% |

The residual gap from 300 GB/s is due to:
- GDDR6 bank activation overhead (row precharge cycles between non-contiguous accesses).
- L1 write latency after the NoC packet arrives.
- DMA engine scheduling overhead between shard prefetch requests.

---

## Roofline Analysis

The roofline model identifies whether a kernel is compute-bound or memory-bound given its arithmetic intensity.

**Ridge point for Wormhole B0:**
```
ridge_point = peak_compute / peak_bandwidth
            = 131 TFLOP/s / 300 GB/s
            = 131e12 / 300e9 FLOP/byte
            ≈ 437 FLOP/byte
```

A kernel with arithmetic intensity above 437 FLOP/byte is compute-bound. Below 437 FLOP/byte, it is memory-bound and DRAM layout matters.

**Expert FFN matmul arithmetic intensity:**

For a matmul of shape `[M, K] × [K, N]`, floating-point operations = `2 * M * K * N`. Bytes read from DRAM = weight bytes + activation bytes:

```
bytes_read = K * N * 2  (BF16 weights)
           + M * K * 2  (BF16 activations, if cold)

arithmetic_intensity = 2 * M * K * N / (K * N * 2 + M * K * 2)
                     = 2 * M * K * N / (2 * K * (N + M))
                     = M * N / (N + M)   [FLOP/byte, simplified]
```

**Decode regime (M=1, batch=1):**
```
arithmetic_intensity = 1 * N / (N + 1) ≈ 1 FLOP/byte  (for large N)
```

At 1 FLOP/byte, the workload is ~437× below the ridge point. Every optimization must target DRAM bandwidth; compute throughput is irrelevant.

**Prefill regime (M=512, batch=1, K=4096, N=14336):**
```
arithmetic_intensity = 512 * 14336 / (14336 + 512)
                     ≈ 494 FLOP/byte
```

At 494 FLOP/byte, the workload is marginally compute-bound. DRAM layout has diminishing returns here; compute kernel configuration (Chapter — compute_kernel_configuration) dominates.

---

## Summary: When DRAM Sharding Matters Most

| Regime | M | Arithmetic Intensity | Bandwidth-Bound? | Sharding Impact |
|---|---|---|---|---|
| Decode, single token | 1 | ~1 FLOP/byte | Yes (strongly) | High — 30–50% speedup |
| Decode, small batch | 4–16 | 4–14 FLOP/byte | Yes | High |
| Prefill, long sequence | 512+ | >400 FLOP/byte | Marginally compute | Low |

For MoE inference at decode, DRAM-sharded expert weights are the single highest-leverage memory layout change available.

---

**Next:** [Chapter 5 — Tile Size Constraints](../ch05_tile_size_constraints/index.md)
