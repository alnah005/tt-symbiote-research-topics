# On-Device Latency Estimate for the Composed and Fused Forms

This file derives analytic latency estimates for both the composed TTNN form (12 ops per layer, no new kernel) and the fused kernel form (`gdn_full_fused_inplace`, `[REUSABLE_WITH_TUNING]`). Both forms eliminate the PCIe transfer entirely. The estimates show that the composed form is dispatch-dominated — not DRAM-bandwidth-dominated — and that the fused kernel removes the dispatch overhead that limits the composed form.

> **Key Finding:** The composed TTNN form reduces DeltaNet decode latency from 9–21 ms to approximately 1 ms per decode step (dispatch-dominated, not bandwidth-dominated). The fused kernel form further reduces this to approximately 177 µs. Both forms eliminate the PCIe transfer entirely.

**Notation throughout:** `d_k = d_v = 128`, `H = 4` (heads per T3K device after sharding), state `S` per head: `[128, 128]` BF16 = 32 KB; full device state per layer: `[1, 4, 128, 128]` = 128 KB. T3K DRAM bandwidth: 288 GB/s per device.

## 1. Composed TTNN Form — Analytic Estimate

The composed TTNN form executes 12 TTNN ops per DeltaNet layer (6 mathematical ops from the recurrence, each decomposed into primitive TTNN calls). There are 30 DeltaNet layers in Qwen3.6-35B-A3B, giving 360 total op dispatches per decode step.

### 1a. DRAM Bandwidth Component

Each decode step reads the previous state `S_prev` and writes back `S_new`. Per device per layer:

```
read:  128 KB (S_prev, shape [1, 4, 128, 128], BF16)
write: 128 KB (S_new,  shape [1, 4, 128, 128], BF16)
total: 256 KB per layer
```

At T3K DRAM bandwidth of 288 GB/s:

```
256 KB / 288 GB/s = 262,144 bytes / (288 × 10^9 bytes/s) ≈ 0.91 µs per layer
30 layers × 0.91 µs ≈ 27 µs total DRAM time
```

The intermediate tensors (Q̃, K̃, V, `o_t`, error, outer product) are small (`[1, 1, 4, 128]` and `[1, 4, 128, 128]`) and fit in Tensix L1 (1.5 MB per core); tile-streaming from L1 does not stress DRAM. DRAM bandwidth is not the bottleneck.

### 1b. Dispatch Overhead Component

Empirical TTNN op dispatch latency on Wormhole B0 is approximately 1–5 µs per op (command queue write + host-side bookkeeping; does not include kernel execution time, which overlaps with subsequent dispatch in pipelined mode). For 360 dispatches:

```
360 ops × 1 µs = 360 µs  (optimistic)
360 ops × 5 µs = 1,800 µs (conservative)
```

Total estimated latency for the composed TTNN form: **0.36–1.8 ms** for all 30 DeltaNet layers.

### 1c. Why Dispatch Is the Bottleneck

The DeltaNet state update operates on tensors with fewer than 65,536 elements per head (`128 × 128 = 16,384` elements per head, 4 heads per device = 65,536 total). At this scale, each TTNN op launches in microseconds and executes in microseconds. The host overhead of enqueuing 12 commands per layer is proportionally large relative to the actual kernel execution time. This is the canonical "many small ops" dispatch-overhead regime in TTNN.

The composed form is already 5–58× faster than the fallback path (0.36–1.8 ms vs. 9–21 ms). It is the correct starting point for implementation and correctness validation. The fused kernel is the next optimization step.

## 2. Fused Kernel Form — Analytic Estimate

The fused `gdn_full_fused_inplace` kernel (ported from the reference CUDA implementation, `[REUSABLE_WITH_TUNING]` per Chapter 4 analysis) collapses all 6 mathematical ops of the DeltaNet recurrence into a single TT-Metalium kernel dispatch per layer. The 12 TTNN op dispatches per layer become 1.

### 2a. Dispatch Overhead

```
1 dispatch × 30 layers = 30 total dispatches
30 dispatches × ~5 µs = 150 µs dispatch overhead
```

### 2b. DRAM Bandwidth

State I/O is the same as the composed form (256 KB per layer):

```
30 layers × 0.91 µs = 27 µs DRAM time
```

### 2c. Total Estimate

```
150 µs (dispatch) + 27 µs (DRAM) ≈ 177 µs for all 30 DeltaNet layers
```

### 2d. Comparison Against Fallback

| Form | Estimated total (30 layers) | vs. fallback |
|---|---|---|
| Host CPU fallback | 9–21 ms | 1× (baseline) |
| Composed TTNN (12 ops/layer) | ~0.36–1.8 ms | 5–58× faster |
| Fused kernel (1 dispatch/layer) | ~177 µs | 50–120× faster |

The fused kernel form achieves 50–120× improvement over the current fallback. Both on-device forms eliminate the PCIe transfer; the fused kernel additionally eliminates 11 of the 12 per-layer dispatches.

## 3. Prefill Note

Prefill (processing an input sequence of length T) uses `chunk_gated_delta_rule` rather than the single-step recurrence. The prefill path operates as a Python loop over T/64 chunks, each chunk calling TTNN matmuls for the DeltaNet recurrence in parallel across the chunk dimension.

At T=8192 (a representative long-context prompt):

```
128 chunks × ~4 TTNN matmuls per chunk × ~10 µs per matmul = ~5 ms per layer
30 DeltaNet layers × 5 ms = ~150 ms total DeltaNet prefill time
```

This estimate is within the expected prefill latency budget for a 35B-class model (full prefill at T=8192 typically runs on the order of seconds). Prefill DeltaNet latency is not a priority issue; the critical path is the decode recurrence. The prefill path does not require trace compatibility (traces are decode-only), so it is addressed in Task 7 of Chapter 7 with low priority.

## 4. Measurement Instructions

The estimates in this file are analytic. The implementing engineer must re-measure all latency figures on target T3K hardware after implementation. Canonical measurement procedure:

1. Enable the TT-Metalium device profiler: set environment variable `TT_METAL_DEVICE_PROFILER=1` before launching the model.
2. Use Tracy profiler to capture a per-op timeline of the decode loop. Tracy will show per-op dispatch and execution latencies broken down by kernel.
3. Run at least 100 warmup decode steps before recording measurements. Report mean and p99 over 1000 steps.
4. Compare DeltaNet layer contribution (sum of DeltaNet-related op times across all 30 layers) against the full-attention layer contribution to establish the relative balance.
5. After Task 6 (fused kernel), re-run the same measurement and confirm the dispatch count drops from 360 to 30 per decode step for DeltaNet layers.

Update `on_device_latency_estimate.md` with empirical values after measurement to replace the analytic estimates.
