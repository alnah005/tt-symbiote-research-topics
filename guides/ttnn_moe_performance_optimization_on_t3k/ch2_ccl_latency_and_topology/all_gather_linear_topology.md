# All-Gather: Linear Topology and num_links=1

## Context

This file addresses:
- **Q1** — What are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?

Source range: `moe.py:L1429–L1436`

---

## The Call Site

```python
x = ttnn.experimental.all_gather_async(
    x,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
)
```

> **Note — `cluster_axis` is absent from this call site** (verified against `moe.py:L1429–L1436`). The `all_gather_async` call does not pass `cluster_axis`; the parameter is not present in the source. The op therefore uses its default axis behavior (axis 0, i.e., the single row-axis of the 1×8 T3K mesh). Do not add `cluster_axis=1` to this snippet; the source does not include it.

The input tensor `x` arrives sharded along the hidden dimension (`dim=-1`). Before any routing or expert compute can begin, each device must hold the full-width activation. `all_gather_async` assembles these shards, leaving a full-width tensor replicated on every device.

**Message size at batch=1 decode:**

| Model | Hidden dim (`H`) | Per-device shard width | Full gather size per device |
|---|---|---|---|
| GLM-4-MoE | 7168 | 896 | 7168 elements × 2 bytes (bf16) = 14 336 B ≈ 14 KB |
| Bailing | 4096 | 512 | 4096 elements × 2 bytes (bf16) = 8 192 B ≈ 8 KB |

At batch=1, one token, the gather message is tiny (8–14 KB). This is firmly in the latency-dominated, not bandwidth-dominated, regime.

---

## Linear Topology: Mechanics and Latency Model

`ttnn.Topology.Linear` implements a linear pipeline gather. Each device holds a shard. Device 0 sends its shard to device 1; device 1 appends its own shard and forwards the two-shard message to device 2; and so on. The final device (device 7) receives a message containing all 8 shards after 7 sequential forwarding steps.

The key characteristics:

- **No wrap-around link.** Devices 0 and 7 are not connected in the gather direction. This avoids one extra hop compared to a ring, which matters when the bottleneck is per-hop latency rather than bandwidth.
- **Latency grows linearly with hop count.** Device 7 waits behind 7 sequential hops. Earlier devices finish sooner, but all-gather is only complete when the last device finishes.
- **Message size grows per hop.** Each forwarding step sends a progressively larger payload: 1 shard on hop 1, 2 shards on hop 2, ..., 7 shards on hop 7. The final shard (device 7's own shard) is assembled locally; hop 7 therefore carries 7 shards, not 8. For small shards (14 KB total), even the 7-shard message is small relative to Ethernet startup costs.

**First-principles latency estimate for Linear all-gather:**

```
T_linear = N_hops × (T_start + T_xfer(M_per_hop))
         ≈ 7 × (T_start + M_full / BW_link)
```

Where:
- `N_hops = 7` for an 8-device chain
- `T_start` — per-hop startup latency (Ethernet NIC initialization + DMA setup); estimated 2–5 µs per hop on T3K
- `M_full` — message size at the last forwarding hop; hop 7 carries 7 shards (the 8th is assembled locally), so `M_full = 7 × 1 792 B = 12 544 B ≈ 12.3 KB` for GLM-4-MoE
- `BW_link` ≈ 12 GB/s per link

For GLM-4-MoE at batch=1:

```
T_xfer(12.5 KB) = 12 544 B / (12 × 10^9 B/s) ≈ 1.04 µs
T_linear ≈ 7 × (3 µs + 1.04 µs) ≈ 7 × 4.04 µs ≈ 28 µs
```

This is a rough lower bound assuming no contention and ideal pipelining. Real measurements will be higher (typically 40–80 µs observed on T3K) due to DMA latency, PCIe interaction, and semaphore overhead.

---

## Ring Topology: Latency Model for Comparison

Ring topology would yield approximately 22 µs steady-state transfer time (vs. ~28 µs for Linear) by keeping each hop's payload constant at one shard rather than accumulating. However, Ring requires the wrap-around link (device 7 to device 0), whose availability on T3K depends on firmware configuration; Linear is the safe default when that link is unconfirmed. At batch=1, where both topologies are dominated by per-hop startup latency rather than transfer time, the difference is small — see [`ccl_sensitivity_analysis.md`](./ccl_sensitivity_analysis.md) for the topology comparison in context.

---

## Increasing num_links: Would It Help?

`num_links=1` means a single Ethernet link is used per hop. T3K Wormhole chips have multiple Ethernet ports between adjacent chip pairs. Setting `num_links=2` would allow both links to carry traffic in parallel, potentially halving the transfer time per hop.

**Analysis for batch=1 decode:**

The current message size per hop ranges from 1 792 B (first hop, GLM-4-MoE) to 12 544 B (last hop). Transfer time for the largest hop at 12 GB/s:

```
T_xfer(12.5 KB, 1 link) ≈ 1.04 µs
T_xfer(12.5 KB, 2 links) ≈ 0.52 µs
```

The delta is approximately 0.52 µs per hop, or 3.6 µs total across all 7 hops. Given that `T_start` contributes ~3 µs per hop and is not reduced by adding links, the expected improvement from `num_links=2` is modest: perhaps 5–10% reduction in all-gather latency at batch=1.

At larger batch sizes or prefill (where message size grows with sequence length), increasing `num_links` would have proportionally larger impact because the bandwidth term dominates.

**Recommendation for investigation:** Measure `all_gather_async` latency with `num_links ∈ {1, 2}` and compare at batch=1. The cost of the investigation is low; the expected gain is small but not zero, and it establishes a baseline for larger batches.

---

## Isolation and Measurement Methodology

To measure the all-gather cost in isolation:

### Method 1: TTNN Op Timer (Recommended for First Pass)

Enable the TTNN device profiler and capture per-op durations. After a forward pass, the op log will contain an entry for `all_gather_async` with its start and end device timestamps.

```python
import tt_lib.profiler as profiler  # or equivalent TTNN timing API

profiler.start("ag_only")
for _ in range(100):  # warmup + measurement
    x_gathered = ttnn.experimental.all_gather_async(
        x_shard,
        dim=-1,
        multi_device_global_semaphore=...,
        barrier_semaphore=...,
        num_links=1,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(device)  # force completion before next call
profiler.stop("ag_only")
```

Key discipline: insert `ttnn.synchronize_device` after each call to force the async op to complete before measurement ends. Without it, you are measuring enqueue time, not execution time.

### Method 2: Microbenchmark with Synthetic Tensor

Construct a tensor that matches the real all-gather input shape and run the CCL op standalone:

```python
import torch
import ttnn

# GLM-4-MoE example: batch=1, seq=1, hidden=7168 sharded across 8 devices
# Each device holds a shard of width 7168/8 = 896
shard_shape = (1, 1, 1, 896)  # [batch, 1, seq, hidden_shard]

x_shard = ttnn.from_torch(
    torch.randn(*shard_shape, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Warmup
for _ in range(10):
    _ = ttnn.experimental.all_gather_async(x_shard, dim=-1, ...)
    ttnn.synchronize_device(device)

# Timed runs
import time
N = 100
t0 = time.perf_counter()
for _ in range(N):
    x_gathered = ttnn.experimental.all_gather_async(x_shard, dim=-1, ...)
    ttnn.synchronize_device(device)
t1 = time.perf_counter()
print(f"all_gather_async mean latency: {(t1 - t0) / N * 1e6:.1f} µs")
```

> **Note:** This warmup-loop / `perf_counter` / divide-by-N harness is the same measurement approach used for reduce-scatter in [`reduce_scatter_ring_topology.md`](./reduce_scatter_ring_topology.md).

### Method 3: Tracy Profiling (Cross-reference Chapter 5)

Annotate the call site in `moe.py:L1429–L1436` with a Tracy zone marker to measure wall-clock duration across devices in the Tracy timeline. See `ch5_profiling_methodology/tracy_profiling_setup.md` for zone annotation instructions.

---

## Parameter Sweep: All-Gather

Run the following sweep to characterize all-gather sensitivity:

| Parameter | Values to sweep | Expected observation |
|---|---|---|
| `num_links` | `{1, 2}` | Marginal improvement at batch=1; larger improvement at larger batch/seq |
| `topology` | `{Linear, Ring}` | Measure only if Ring topology is available on T3K; expect minimal difference at batch=1 |
| batch size | `{1, 4, 8, 16}` | Bandwidth term grows linearly; num_links benefit increases |

Record results in the format:

| batch | num_links | topology | latency (µs) | message size (KB) | effective BW (GB/s) |
|---|---|---|---|---|---|
| 1 | 1 | Linear | ___ | 14.0 | ___ |
| 1 | 2 | Linear | ___ | 14.0 | ___ |
| ... | | | | | |

The "effective BW" column should be computed as `message_size / latency` and compared against the theoretical 12 GB/s per link to assess how close to peak the current implementation operates.

---

## Interpreting Results

- If measured latency significantly exceeds the first-principles estimate (e.g., > 2×), the overhead is in semaphore management, DMA setup, or host-device synchronization — not in wire transfer. Investigate `ccl_manager.get_and_cycle_ag_semaphore_handles` overhead.
- If `num_links=2` provides less than 5% improvement at batch=1 but > 20% at batch=16, this confirms the startup-latency-dominated regime and argues for a different optimization strategy (e.g., asynchronous overlap rather than raw bandwidth increase).
- If the measured all-gather latency is comparable to or greater than the `TTNNExperts.forward` latency at batch=1 decode, CCL is the bottleneck and topology/link tuning is high priority.

---

**Next:** [`reduce_scatter_ring_topology.md`](./reduce_scatter_ring_topology.md)
