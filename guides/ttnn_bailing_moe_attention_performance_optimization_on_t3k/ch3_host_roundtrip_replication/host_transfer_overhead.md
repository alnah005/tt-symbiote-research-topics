# Host Transfer Overhead of `_to_replicated`

## Purpose

This file builds a quantitative model for the PCIe round-trip latency introduced by `_to_replicated` at decode batch=1, explains why the cost is overhead-dominated rather than bandwidth-dominated, describes how to measure it empirically using Tracy and TTNN op timers, and analyses how the overhead scales with batch size so that a practical crossover point can be identified.

All figures are for the Ling model in decode mode: `hidden_size=4096`, `num_heads=16`, `num_kv_heads=4`, `head_dim=128`, batch=1, `seq_len=1`, on T3K (8-chip, 1×8 mesh), unless otherwise noted.

## PCIe Throughput Model for T3K Host Transfers

### Physical Configuration

Each Wormhole n300 chip in a T3K connects to the host x86 system via a dedicated PCIe Gen4 x16 link. The theoretical peak bandwidth of a PCIe Gen4 x16 link is:

```
Peak bandwidth (one direction) = 16 lanes × 16 GT/s per lane / 8 bits per byte × (128/130 encoding)
                               = 16 × 16 / 8 × (128/130)
                               ≈ 31.5 GB/s (theoretical peak); practical ~20–25 GB/s after protocol overhead
```

In practice, DMA transfers between host DRAM and device DRAM achieve significantly lower throughput due to:

- PCIe transaction layer framing overhead (TLP headers, credit flow control)
- Host DRAM contention (all 8 PCIe root-complex endpoints share the same memory bus)
- TTNN DMA engine setup time (command queue dispatch, address validation, cache line flushes)
- Non-page-aligned transfers for small tensors (increases TLP count per byte)

Practical observed throughput for small tensor transfers (< 1 MB) on T3K is approximately **20–25 GB/s per chip, unidirectional** [ESTIMATE]. This is consistent with published Wormhole n300 system benchmarks and with experience from comparable PCIe Gen4 x16 DMA workloads on x86 hosts.

### Bidirectional Throughput Constraint

PCIe Gen4 x16 supports simultaneous bidirectional transfers. The `_to_replicated` round-trip requires:

1. Device → Host (all 8 chips in parallel, or serialised at the host memory bus)
2. Host → Device (all 8 chips in parallel)

These two phases are sequential in time (step 3 of `_to_replicated` cannot begin until step 1 completes). They do not overlap. The round-trip is therefore:

```
t_RT = t_device_to_host + t_host_to_device + t_host_cpu_overhead
```

### Per-Chip vs. Aggregate Transfer Bandwidth

A key architectural point: the 8 chips do not share a single PCIe link. Each chip has its own independent PCIe Gen4 x16 root-complex port. This means:

- **Ideally**, all 8 device→host DMA reads can proceed in parallel, completing in the time of one chip's transfer.
- **In practice**, the host side of all 8 links converges on the same host DRAM bus. At 20–25 GB/s practical throughput per chip, the aggregate host DRAM write rate during device→host would be up to 200 GB/s — within or below the host DRAM bandwidth range for T3K's server-class platform (typically ~150–400 GB/s for quad-channel to 8-channel DDR5 [ESTIMATE]), so host DRAM capacity alone is not a guaranteed bottleneck, though contention effects may still reduce effective throughput under simultaneous traffic.

For small transfers like the 6 KB per chip in `_to_replicated`, the **per-chip transaction overhead** (not bandwidth) is the limiting factor regardless of parallelism. Each PCIe DMA transfer, no matter how small the payload, incurs a minimum round-trip overhead from:

- Command queue write on the device (~1–2 µs per DMA command) [ESTIMATE]
- Interrupt or polling-based completion notification on the host (~1–5 µs) [ESTIMATE]
- Cache coherence protocol operations (host DRAM cache invalidation/flush)

The latency model below assumes the QKV tensor is in ROW_MAJOR_LAYOUT at transfer time (6,144 bytes per chip). If the tensor is in TILE_LAYOUT, substitute ~192 KB per chip; see `roundtrip_mechanics.md` for the full breakdown.

Table: PCIe transfer component costs for small payloads on T3K (per chip, unidirectional)

| Component | Estimated cost | Notes |
|---|---|---|
| DMA command queue write | 1–2 µs | Device firmware overhead |
| TLP framing and serialisation | < 0.1 µs | Negligible at 6 KB payload |
| Payload transfer time | ~0.3 µs | 6 KB at 20 GB/s |
| Host completion polling / interrupt | 1–5 µs | Depends on host OS scheduler |
| Host DRAM write finalisation | 0.1–0.5 µs | Cache-line write-back |
| **Per-chip unidirectional subtotal** | **2.4–7.6 µs** | [ESTIMATE] |

## Estimated Round-Trip Latency at Decode Batch=1

### One Direction: Device → Host

TTNN's `ConcatMeshToTensor` initiates DMA reads from all 8 chips. Because the post-all-reduce tensor is numerically identical on every chip, `ConcatMeshToTensor(dim=3)` transfers 8 × 6 KB = 48 KB total across PCIe, concatenates the copies into a (1, 1, 1, 24576) host tensor, and then discards 7/8 of that data when slicing back to (1, 1, 1, 3072) — leaving only 6 KB of unique data on the host. Because each individual chip's payload is small (6 KB per chip), the chips' DMA engines finish nearly simultaneously and the cost is dominated by the per-chip command overhead rather than any serialisation at the host memory bus.

```
t_device_to_host ≈ max(per-chip DMA overhead across 8 chips)
                 ≈ 2.4–7.6 µs  [ESTIMATE]
```

(The `max` rather than `sum` applies because the 8 chips transfer concurrently, and the critical path is the slowest chip.)

### One Direction: Host → Device

`from_torch` with `ReplicateTensorToMesh` initiates 8 independent DMA writes, one per chip. Again, for 6 KB payloads, overhead dominates:

```
t_host_to_device ≈ 2.4–7.6 µs  [ESTIMATE]
```

### Host CPU Overhead

Between the two DMA phases, the host CPU executes the `torch.Tensor` construction in Python. For a 6 KB BF16 tensor, this involves:

- Python object instantiation and reference counting
- `torch.Tensor` descriptor allocation
- No data copy (the buffer from `to_torch` is used directly)

```
t_host_cpu ≈ 2–10 µs  [ESTIMATE]
```

This range is wide because Python GIL contention, garbage collection pauses, and CPU-side NUMA effects can inflate it.

### Total Round-Trip Estimate

```
t_RT = t_device_to_host + t_host_cpu + t_host_to_device
     ≈ (2.4–7.6) + (2–10) + (2.4–7.6)
     ≈ 6.8–25.2 µs  [ESTIMATE]
```

Table: Estimated `_to_replicated` round-trip latency components (Ling, decode batch=1, T3K)

| Phase | Estimated latency | Bottleneck |
|---|---|---|
| Device → Host (`ConcatMeshToTensor`) | 2.4–7.6 µs | PCIe transaction overhead |
| Host CPU (`torch.Tensor` construction) | 2–10 µs | Python runtime + GIL |
| Host → Device (`from_torch` + `ReplicateTensorToMesh`) | 2.4–7.6 µs | PCIe transaction overhead |
| **Total round-trip** | **6.8–25.2 µs** | [ESTIMATE] |

For context, the fused QKV matmul + all-reduce together are estimated at **13–43 µs** (see Chapter 2, [`latency_savings_analysis.md`](../ch2_fused_qkv_projection/latency_savings_analysis.md)). The round-trip therefore represents **16–194% of the QKV projection time** depending on which end of both ranges materialises [ESTIMATE]. At the median of both ranges (decode step ~28 µs for QKV, ~16 µs for round-trip), the round-trip accounts for approximately **57%** of QKV projection time — a highly significant overhead.

## How to Measure Actual Latency

### Using Tracy (Recommended for Wall-Clock Accuracy)

Tracy is the highest-fidelity tool for measuring host-side operations that span PCIe boundaries (see Chapter 7, `tracy_profiling.md` for full setup). To instrument `_to_replicated`:

```python
import tracy

def _to_replicated(qkv_tensor, mesh_device):
    with tracy.zone("to_replicated::device_to_host"):
        qkv_host = ttnn.to_torch(
            qkv_tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
        )

    with tracy.zone("to_replicated::host_cpu"):
        # torch.Tensor is already constructed; this zone captures any
        # Python-side post-processing (slicing, dtype cast, etc.)
        qkv_host = qkv_host[..., :3072].contiguous()

    with tracy.zone("to_replicated::host_to_device"):
        qkv_replicated = ttnn.from_torch(
            qkv_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,  # TILE_LAYOUT required for kernel compatibility, but produces ~192 KB per chip transfer — not the 6 KB assumed in the latency model above
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return qkv_replicated
```

With Tracy instrumentation enabled, each zone will appear as a named span in the Tracy timeline view. The wall-clock duration of `to_replicated::device_to_host` captures the full PCIe DMA plus completion signalling latency; it is not decomposed further without hardware PCIe performance counters.

**Capture procedure:**

1. Run at least 10 warmup decode steps before starting the Tracy capture (the first few steps include JIT compilation overhead that inflates all latencies).
2. Capture exactly 1 decode step to avoid averaging across varying system states.
3. In the Tracy GUI, filter for zones matching `to_replicated::` and read the `mean` and `min` durations. The `min` is closest to the hardware floor; the `mean` reflects realistic scheduling noise.

### Using TTNN Op Timers (for Device-Side Phases Only)

TTNN's built-in op timer (`ttnn.device.dump_device_profiler`) captures device-side kernel execution times but does **not** capture host-side Python execution time or the PCIe transfer latency (the DMA transfer is not a TTNN "op" in the device profiler sense).

For the device→host direction, the TTNN device profiler records when the device DMA engine begins but not when the host receives the data. Therefore, TTNN op timers alone underestimate the round-trip latency. Use them only to bound the device-initiation cost:

```python
import ttnn

ttnn.enable_program_cache()
ttnn.device.enable_profiling(device)

# ... run decode step ...

ttnn.device.dump_device_profiler(device, output_path="profile_ch3.csv")
```

Filter the CSV for `op_name` containing `"ConcatMeshToTensor"` or `"from_torch"` to find the device-side contribution. Subtract this from the Tracy wall-clock measurement to isolate the host-side and PCIe-transfer contribution.

### Quick Python-Level Timing (for Rough Estimates)

For a fast sanity check without Tracy setup:

```python
import time
import ttnn
import torch

# After model warmup (at least 10 decode steps):
t0 = time.perf_counter()
qkv_host = ttnn.to_torch(
    qkv_all_reduced,
    mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3),
)
t1 = time.perf_counter()
qkv_replicated = ttnn.from_torch(
    qkv_host[..., :3072].contiguous(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
t2 = time.perf_counter()

print(f"device→host: {(t1-t0)*1e6:.1f} µs")
print(f"host→device: {(t2-t1)*1e6:.1f} µs")
print(f"round-trip:  {(t2-t0)*1e6:.1f} µs")
```

`time.perf_counter()` is a wall-clock timer with nanosecond resolution. It captures the full Python + PCIe + DMA latency per direction. The results will have higher variance than Tracy due to Python GIL noise; run 100 iterations and report the median.

## Sensitivity to Batch Size

### How the Tensor Grows with Batch Size

The QKV tensor shape at batch size `B` and seq_len=1 is:

```
QKV tensor shape: (B, 1, 1, 3072)
Bytes per chip:   B × 1 × 1 × 3072 × 2 (BF16) = B × 6,144 bytes
```

Byte sizes below assume ROW_MAJOR_LAYOUT (see [Layout Assumption](#layout-assumption) above). For TILE_LAYOUT, multiply per-chip bytes by 32.

Table: QKV tensor size and estimated round-trip latency vs. batch size (Ling, T3K)

| Batch size | Per-chip bytes | Estimated t_RT | Bandwidth-limited? |
|---|---|---|---|
| 1 | 6.1 KB | 6.8–25.2 µs | No — overhead-dominated |
| 4 | 24.6 KB | 7.5–27 µs | No — overhead-dominated |
| 16 | 98.3 KB | 9–32 µs | Transitioning |
| 64 | 393.2 KB | 18–50 µs | Partially bandwidth-limited |
| 256 | 1.57 MB | 70–160 µs | Bandwidth-dominated |
| 512 | 3.15 MB | 130–310 µs | Bandwidth-dominated |

[ESTIMATE] for all rows.

### Where the Overhead Becomes Negligible Relative to Compute

The round-trip cost is relevant to decode performance as long as it is a significant fraction of the total attention layer latency. As batch size increases:

1. The round-trip latency grows (linearly with batch size once bandwidth-limited).
2. The compute time of `paged_sdpa_decode` also grows (linearly with batch size and sequence length for the KV-cache read).
3. The matmul inside `TTNNLinearIColShardedWAllReduced` shifts toward compute-bound at larger batches, also increasing.

The round-trip latency becomes **negligible** (< 5% of total attention latency) only when the total attention compute time exceeds roughly **20× the round-trip cost**. Given that `paged_sdpa_decode` latency at batch=1 is on the order of a few hundred microseconds [ESTIMATE], the round-trip at batch=1 (≈ 7–25 µs) is already a small fraction of SDPA time but not negligible relative to other preprocessing ops.

The round-trip overhead is **most impactful at batch=1 relative to non-SDPA operations** (QKV projection, memory-config transitions, RoPE). It is at batch=1 where `_to_replicated` consumes the largest percentage of non-SDPA decode latency, making it the highest-priority target for device-side elimination.

At batch ≥ 64, the round-trip cost (18–50 µs) begins to be dwarfed by the matmul compute time (which grows to several hundred µs) and becomes progressively less important. There is no clean "crossover batch size" — it depends on the relative optimisation state of the other operations. A practical rule of thumb: **if batch ≥ 32 and SDPA is the dominant latency, the round-trip is not the highest-priority target** [ESTIMATE].

---

**Next:** [Device-Side Alternatives](./device_side_alternatives.md)
