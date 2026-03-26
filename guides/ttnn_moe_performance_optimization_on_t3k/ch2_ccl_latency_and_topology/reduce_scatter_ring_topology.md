# Reduce-Scatter: Ring Topology, chunks_per_sync, and num_workers_per_link

## Context

This file addresses:
- **Q1** — What are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?

Source range: `moe.py:L1478–L1490`

---

## The Call Site

```python
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    persistent_output_buffers=None,
    dim=3,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

This is called immediately after `TTNNExperts.forward` completes and after the pre-normalization multiply (`moe.py:L1477`). The input `routed_out` is a full-width tensor replicated across all 8 devices. The output is a tensor re-sharded along `dim=3` (hidden dimension), where each device holds a 1/8 slice — restoring the tensor-parallel layout needed by the next layer.

**Message size at batch=1 decode:**

| Model | Full hidden dim (`H`) | Output shard size per device | Reduce-scatter payload |
|---|---|---|---|
| GLM-4-MoE | 7168 | 896 | Full tensor: 7168 × 2B ≈ 14 KB in; 896 × 2B ≈ 1.75 KB out |
| Bailing | 4096 | 512 | Full tensor: 4096 × 2B ≈ 8 KB in; 512 × 2B ≈ 1 KB out |

Unlike all-gather, reduce-scatter involves both a reduction (element-wise sum) and a scatter (shard assignment). Each device must send portions of its tensor to all other devices and receive and accumulate the matching portions from all others.

---

## Pre-Scatter Normalization (moe.py:L1477)

```python
n_rs = self.device.shape[1]  # = 8 for T3K
routed_out = routed_output.to_ttnn
if n_rs > 1:
    routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
```

Before the scatter, the tensor is element-wise multiplied by `1/8`. The ring CCL kernel itself performs a **sum** — it accumulates contributions from all 8 devices without any normalization. The `1/8` pre-scaling applied here (at `moe.py:L1477`) is what produces the final normalized (mean) output: each device's contribution is pre-divided before the ring sum, so the accumulated result equals the mean. The pattern matters for profiling: `ttnn.mul` executes as a separate op immediately before `reduce_scatter_minimal_async`. Its cost should be measured independently to avoid conflating element-wise multiply overhead with CCL overhead.

---

## Ring Topology: Mechanics

`ttnn.Topology.Ring` implements a pipelined ring reduce-scatter. The algorithm proceeds as follows:

1. Divide the output dimension (`dim=3`, hidden) into `N = 8` equal chunks, one per device.
2. Each device sends its local chunk destined for device `(i+1) % 8` and begins receiving from device `(i-1) % 8`.
3. At each step, the received chunk is accumulated (summed) into the local buffer for that chunk.
4. After `N-1` steps, each device holds the fully reduced shard corresponding to its assigned output slice.

In steady-state pipeline operation, a device is simultaneously:
- Sending a chunk to the right neighbor
- Receiving a chunk from the left neighbor
- Performing element-wise accumulation of a previously received chunk

This triple overlap — send, receive, compute — is what makes Ring topology efficient for large messages. The throughput in steady state is limited by `min(BW_send, BW_recv, compute_rate)`.

**Why Ring for reduce-scatter but not all-gather:** All-gather accumulates shards additively along a growing front (no reduction needed); Linear topology is simpler and avoids wrap-around link requirements. Reduce-scatter needs all-to-all exchange for the accumulation; Ring amortizes this into a symmetric pipeline. For large messages, Ring's steady-state throughput is superior. For small messages, both are startup-limited and the difference is minimal.

---

## chunks_per_sync: Pipelining Depth

`chunks_per_sync=10` controls how many payload chunks are injected into the pipeline before a synchronization point is inserted.

**What a synchronization point does:** After `chunks_per_sync` chunks have been transmitted, the worker waits for an acknowledgment from the receiving device that the chunks were received and accumulated correctly. This synchronization:
- Prevents sender overrun of the receiver's buffers.
- Provides a checkpoint for error recovery.
- Introduces a stall bubble in the pipeline if the receiver is slower than the sender.

**Effect of increasing `chunks_per_sync`:**
- Fewer synchronization stalls per unit of payload.
- Larger in-flight buffer requirement (`num_buffers_per_channel=2` provides some cushion, but at very high `chunks_per_sync`, buffer exhaustion can cause back-pressure).
- Higher peak latency if an error occurs (more unacknowledged data in flight).

**Effect of decreasing `chunks_per_sync`:**
- More frequent sync points, each adding ~1–2 µs of handshake overhead.
- At `chunks_per_sync=1`, the op degenerates to stop-and-wait, serializing every chunk with an acknowledgment round-trip.

For a 14 KB message split into chunks, the chunk size depends on the internal chunk granularity of the TTNN CCL kernel. If the CCL kernel uses 1 KB chunks, a 14 KB message has 14 chunks; `chunks_per_sync=10` means one synchronization at chunk 10 and the op completes before a second sync is needed. At `chunks_per_sync=1`, 14 synchronizations occur instead of 1.

---

## num_workers_per_link: DMA Parallelism

`num_workers_per_link=2` allocates two worker threads per Ethernet link. Each worker independently manages DMA transfers for its assigned subset of chunks.

**Effect of `num_workers_per_link=1` vs `2`:**

With 1 worker: the worker processes chunks serially — submit DMA for chunk k, wait for transfer, submit DMA for chunk k+1. There is no overlap between DMA setup and data transfer.

With 2 workers: while worker A waits for DMA completion of chunk k, worker B can set up and submit DMA for chunk k+1. This doubles the pipeline depth at the DMA submission level, hiding DMA setup latency behind transfer time.

**At batch=1 decode, with 14 KB total message size:**

The transfer time per chunk is sub-microsecond. DMA setup latency on Wormhole is estimated at 200–500 ns. The benefit of 2 workers versus 1 is a reduction in DMA idle time on the order of:

```
ΔT ≈ (N_chunks - 1) × T_dma_setup × (1 - 1/num_workers)
   ≈ (14 - 1) × 350 ns × 0.5
   ≈ 2.3 µs
```

This is a non-trivial fraction of the total reduce-scatter latency at batch=1. Setting `num_workers_per_link=1` for a direct comparison is the correct experiment.

---

## num_buffers_per_channel: Double-Buffering

`num_buffers_per_channel=2` enables double-buffering at the channel level. While buffer A is being filled by an incoming transfer, buffer B is being consumed (accumulated, passed to the next stage). This hides memory latency behind transfer latency.

At batch=1 with small message sizes, the double-buffering benefit is limited by the fact that both buffers can fill faster than the accumulate-and-forward path can drain them. For larger messages or higher `chunks_per_sync`, `num_buffers_per_channel=2` is more important.

---

## Parameter Sweep Methodology

### Step 1: Establish Baseline

Run the reduce-scatter in isolation with current parameters (`chunks_per_sync=10`, `num_workers_per_link=2`, `num_links=1`, Ring topology). Record mean latency over 100 iterations.

```python
import time
import ttnn

# Construct input matching the real reduce-scatter input shape
# GLM-4-MoE: routed_out is [1, 1, 1, 7168] replicated across 8 devices
routed_out_shape = (1, 1, 1, 7168)

routed_out = ttnn.from_torch(
    torch.randn(*routed_out_shape, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

def measure_rs(cps, nwpl, n_runs=100):
    """Measure reduce_scatter_minimal_async with given chunks_per_sync and num_workers_per_link."""
    # warmup
    for _ in range(10):
        _ = ttnn.experimental.reduce_scatter_minimal_async(
            routed_out, persistent_output_buffers=None, dim=3,
            multi_device_global_semaphore=..., barrier_semaphore=...,
            num_links=1, cluster_axis=1, topology=ttnn.Topology.Ring,
            chunks_per_sync=cps, num_workers_per_link=nwpl, num_buffers_per_channel=2,
        )
        ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = ttnn.experimental.reduce_scatter_minimal_async(
            routed_out, persistent_output_buffers=None, dim=3,
            multi_device_global_semaphore=..., barrier_semaphore=...,
            num_links=1, cluster_axis=1, topology=ttnn.Topology.Ring,
            chunks_per_sync=cps, num_workers_per_link=nwpl, num_buffers_per_channel=2,
        )
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1e6  # µs
```

### Step 2: Sweep chunks_per_sync

```
chunks_per_sync ∈ {1, 2, 5, 10, 20, 50}
```

Hold `num_workers_per_link=2` constant. Record mean latency (µs) and standard deviation.

Expected pattern:
- `chunks_per_sync=1`: highest latency (maximum sync overhead).
- `chunks_per_sync=10` (current): near-optimal for batch=1 message sizes.
- `chunks_per_sync=20` or `50`: may offer marginal improvement if the sync overhead is still measurable; or may plateau once sync cost is negligible relative to transfer time.

| `chunks_per_sync` | mean latency (µs) | std dev (µs) | notes |
|---|---|---|---|
| 1 | ___ | ___ | stop-and-wait |
| 2 | ___ | ___ | |
| 5 | ___ | ___ | |
| 10 | ___ | ___ | **current** |
| 20 | ___ | ___ | |
| 50 | ___ | ___ | |

### Step 3: Sweep num_workers_per_link

```
num_workers_per_link ∈ {1, 2, 4}
```

Hold `chunks_per_sync=10` constant. `num_workers_per_link=4` may not be supported; check TTNN API constraints first.

| `num_workers_per_link` | mean latency (µs) | delta vs baseline (µs) |
|---|---|---|
| 1 | ___ | ___ |
| 2 | ___ | **baseline** |
| 4 | ___ | ___ |

### Step 4: Joint Sweep (Optional)

After establishing the individual optima, confirm the joint optimum with a 3×3 grid:

```
chunks_per_sync ∈ {best - 1 step, best, best + 1 step}
num_workers_per_link ∈ {1, 2}
```

### Step 5: Vary Batch Size

Repeat the `chunks_per_sync` sweep at `batch ∈ {1, 4, 8, 16}`. The optimal `chunks_per_sync` may shift toward lower values (more frequent sync) as messages grow, because pipeline stall time relative to transfer time decreases.

---

## Synchronization Granularity vs. Latency Tradeoff

The core tradeoff governed by `chunks_per_sync` is:

| Property | Low `chunks_per_sync` | High `chunks_per_sync` |
|---|---|---|
| Sync overhead | High (many round-trips) | Low (few round-trips) |
| Buffer pressure | Low (few chunks in flight) | High (many chunks in flight) |
| Pipeline efficiency | Low (frequent stalls) | High (long steady-state run) |
| Sensitivity to receiver speed | Low | High |

For batch=1 decode with 14 KB messages on T3K, the message is small enough that the op may complete within the first `chunks_per_sync=10` chunk window without even hitting a sync point. In that case, `chunks_per_sync` values of 5, 10, and 20 will produce identical latency, and only values of 1 or 2 will show degradation. This is the expected null result — and it is informative: it confirms that sync overhead is not a bottleneck for this message size.

---

## Isolation and Measurement Notes

Several pitfalls to avoid when measuring reduce-scatter in isolation:

1. **Include the pre-normalization multiply.** The `ttnn.mul(routed_out, 1.0 / 8.0)` at `moe.py:L1477` executes on every forward pass when `n_rs > 1`. Its cost should be measured alongside the reduce-scatter to get an accurate CCL-phase total. It is a cheap scalar multiply on a 14 KB tensor (~1 µs), but it is part of the wall-clock path.

2. **Use a warm buffer.** Reuse the same `routed_out` buffer across iterations to avoid cold-cache effects. If the buffer is freshly allocated each iteration, DRAM fetch latency will inflate measurements.

3. **Account for semaphore cycling.** `ccl_manager.get_and_cycle_rs_semaphore_handles(1)` advances an internal pointer. In a tight microbenchmark loop, the cycle depth of the semaphore pool may be exhausted, causing stalls. Pre-allocate sufficient handles (or use a mock semaphore manager that returns handles from a deep pool) when running hundreds of iterations.

4. **Synchronize before starting the timer.** Issue `ttnn.synchronize_device(device)` before `time.perf_counter()` to ensure no prior work is in flight.

---

**Next:** [`ccl_sensitivity_analysis.md`](./ccl_sensitivity_analysis.md)
