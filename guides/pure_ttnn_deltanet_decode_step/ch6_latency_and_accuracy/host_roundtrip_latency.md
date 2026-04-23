# Host Round-Trip Latency for the Current PyTorch Fallback

This file quantifies the latency cost of the existing `recurrent_gated_delta_rule` fallback path, which runs the DeltaNet state update on the host CPU rather than on the Wormhole device. Understanding this cost in detail — component by component — establishes the baseline that a pure on-device implementation must beat and explains why eliminating the PCIe transfer is the highest-impact single change available.

> **Key Finding:** The dominant individual cost in the host round-trip is not the CPU kernel (10–100 µs) but the 128 KB `S_prev` readback (100–300 µs) and the 128 KB `S_new` upload (100–300 µs). This data needlessly crosses PCIe at every layer: the state tensors fit entirely in DRAM (3.75 MB for all 30 layers per device) and should never leave the device. Across 30 DeltaNet layers, the round-trip moves approximately 7.5 MB of state data over PCIe per decode step, contributing 9–21 ms — for matrix operations that take microseconds on-device.

## 1. Latency Components

The fallback path makes four categories of cross-device data movement and one CPU kernel execution per DeltaNet layer. All figures assume batch size B=1, one token per decode step, and a PCIe 4.0 x16 link at its rated 16 GB/s peak (practical throughput is lower due to protocol overhead and synchronization).

**Notation:** `d_k = d_v = 128`, `num_v_heads = 32` total across the T3K mesh (4 per Wormhole device after tensor-parallel sharding), `H = 4` (heads per device).

### 1a. `ttnn.to_torch` for Q̃, K̃, V (input projections)

Each of Q̃, K̃, and V has shape `[1, 1, H, d_k]` = `[1, 1, 4, 128]`. In BF16, each element is 2 bytes:

```
bytes per tensor = 1 × 1 × 4 × 128 × 2 = 1,024 bytes
total for 3 tensors = 3,072 bytes ≈ 3 KB
```

At 16 GB/s theoretical peak: `3,072 / (16 × 10^9) ≈ 0.19 µs`. In practice, each `ttnn.to_torch` call incurs a command dispatch, a device-side readback fence, and a host-side memcpy. Across all three calls, the expected actual latency is **10–50 µs** — an order of magnitude above the theoretical transfer time.

### 1b. `ttnn.to_torch` for `S_prev` (recurrent state)

The recurrent state tensor per device has shape `[1, H, d_k, d_v]` = `[1, 4, 128, 128]`. In BF16:

```
bytes = 1 × 4 × 128 × 128 × 2 = 131,072 bytes = 128 KB per device
```

On a T3K 1×8 mesh, each device holds 4 of the 32 heads. The single `ttnn.to_torch` call for the device-local state shard transfers 128 KB. At 16 GB/s theoretical: `131,072 / (16 × 10^9) ≈ 8.2 µs`. With synchronization overhead (device fence + DMA + host copy), actual latency is **100–300 µs**.

This is the single largest latency component per layer. At 30 layers, the state readback alone is 3–9 ms per decode step — simply moving read-only data across PCIe to feed a CPU kernel that immediately writes it back.

### 1c. `recurrent_gated_delta_rule` CPU kernel execution

The DeltaNet recurrence (6 mathematical operations: decay `S`, retrieve `o_t`, compute error `e`, outer product write `∆S`, add, project output) runs as a PyTorch-native CPU kernel at B=1 on small matrices (`[d_k, d_v]` = `[128, 128]`). Triton GPU kernels are not applicable on Wormhole inference machines. Expected CPU kernel execution: **10–100 µs** depending on host CPU utilization and BLAS library state.

### 1d. `ttnn.from_torch` for `S_new` and `o_t` (upload)

After the CPU kernel produces the updated state `S_new` (shape `[1, 4, 128, 128]`, 128 KB) and output `o_t` (shape `[1, 1, 4, 128]`, 1 KB), both must be uploaded back to the Wormhole device. Total upload: approximately 129 KB. At 16 GB/s theoretical: ~8 µs. Actual with `ttnn.from_torch` overhead: **100–300 µs**.

### 1e. Per-Layer and End-to-End Totals

| Component | Theoretical | Actual (expected) |
|---|---|---|
| `ttnn.to_torch` Q̃, K̃, V (3 tensors) | ~0.2 µs | 10–50 µs |
| `ttnn.to_torch` `S_prev` (128 KB) | ~8 µs | 100–300 µs |
| `recurrent_gated_delta_rule` CPU | — | 10–100 µs |
| `ttnn.from_torch` `S_new` + `o_t` | ~8 µs | 100–300 µs |
| **Per-layer total** | ~16 µs | **300–700 µs** |
| **30 DeltaNet layers** | ~0.5 ms | **9–21 ms** |

## 2. How to Measure

Instrument the existing fallback code with `time.perf_counter_ns()` around each boundary crossing. A minimal measurement harness:

```python
import time

# Per-layer measurement (inside the DeltaNet layer forward):
t0 = time.perf_counter_ns()
q_torch = ttnn.to_torch(q_tilde)
k_torch = ttnn.to_torch(k_tilde)
v_torch = ttnn.to_torch(v)
t1 = time.perf_counter_ns()

s_torch = ttnn.to_torch(s_prev)
t2 = time.perf_counter_ns()

s_new, o_t = recurrent_gated_delta_rule(q_torch, k_torch, v_torch, s_torch, g_t, beta_t)
t3 = time.perf_counter_ns()

s_new_tt = ttnn.from_torch(s_new, ...)
o_t_tt   = ttnn.from_torch(o_t, ...)
t4 = time.perf_counter_ns()

# Record: (t1-t0), (t2-t1), (t3-t2), (t4-t3) in µs
```

Run **100 warmup steps** (to ensure device caches and PCIe link are warm), then record measurements over **1000 decode steps**. Report:
- Mean and p99 per component per layer
- Total per-layer mean and p99
- End-to-end 30-layer mean and p99

Store results as a CSV with columns `[step, layer, component, latency_us]` for later comparison against the on-device implementation.

## 3. Why This Dominates Decode Latency

At decode B=1, the Wormhole devices execute attention computations in microseconds (matmuls over small tensors are fast; the full-attention layers for the 10 non-DeltaNet layers in Qwen3.6-35B-A3B take on the order of a few hundred microseconds total). The DeltaNet fallback path, by contrast, drains 9–21 ms on PCIe transfers and CPU execution.

This is the primary motivation for implementing the on-device DeltaNet decode. Even the composed TTNN form (Chapter 2, 12 TTNN ops per layer, no new kernel required) eliminates the PCIe transfer entirely. The expected speedup at the DeltaNet contribution alone is **5–58× at B=1**, reducing the 30-layer DeltaNet latency from 9–21 ms to approximately 0.36–1.8 ms (approximately 1 ms at the midpoint estimate).

## 4. Note on `S_prev` Transfer Dominance

The 128 KB `S_prev` transfer per layer is the dominant individual latency component, not the kernel execution. At 30 layers, the model moves:

```
30 layers × 128 KB (read) + 30 layers × 128 KB (write) = 7,680 KB ≈ 7.5 MB
```

of state data across PCIe per decode step — solely to run 30 small matrix operations on the CPU. This data never needed to leave the device. The state tensors fit comfortably in DRAM (128 KB × 30 layers = 3.75 MB per device) and should remain there for the lifetime of a decode sequence. Eliminating this transfer is the single highest-impact change available before writing a single line of new kernel code.
