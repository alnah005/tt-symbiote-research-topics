# Benchmarking `num_links`: Methodology and Interpretation

This file describes how to set up a minimal benchmark harness, sweep `num_links` values, interpret results, and decide when to re-benchmark. No concrete latency numbers are provided here because those values must be measured on the specific T3K system under test; hardware revision, firmware version, and system load all affect results. The `benchmarking_num_links.md` framework is designed to produce reproducible numbers that you can compare across TTNN versions and configuration changes.

---

## Goal

Find the `num_links` value that minimizes all-to-all latency (or maximizes throughput, depending on your optimization target) for a specific combination of:

- **Tensor size** — determined by batch size $B$, sequence length $S$, expert capacity $C$, and hidden dimension $H$
- **Inference regime** — prefill (large batch, throughput-bound) vs. decode (small batch, latency-bound)
- **Concurrency** — single isolated collective vs. pipelined with concurrent operations

The benchmark must be run separately for each combination of interest. A value of `num_links` that is optimal for prefill will generally not be optimal for decode.

---

## Step 1: Set Up the MeshDevice

```python
import ttnn
import torch

# Open a T3K MeshDevice with the standard (1, 8) mesh shape.
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 8),
    # Device IDs 0–7 in row-major order.
    # Verify the physical ordering matches your T3K board's wiring before use.
)
```

Verify the device count before running the benchmark:

```python
assert mesh_device.get_num_devices() == 8, (
    f"Expected 8 devices on T3K; found {mesh_device.get_num_devices()}. "
    "Check MeshDevice initialization and device ordering."
)
```

---

## Step 2: Construct a Representative All-to-All Tensor

The benchmark tensor should represent the dispatch tensor for the target workload. The dispatch tensor shape is $(N \times C, H)$ per device (where $N = 8$), laid out so that row group $d$ contains the tokens to be sent to device $d$.

For a decode scenario with $B = 8$ sequences (one example of a representative small-batch decode):

```python
# Parameters — adjust to match your target workload.
N = 8           # number of devices
k = 8           # top-K experts per token [D UNVERIFIED]
E = 256         # total experts [D UNVERIFIED]
H = 7168        # hidden dimension [D UNVERIFIED]
B = 8           # batch size for this benchmark run
S = 1           # decode step: one new token per sequence

import math
C = math.ceil(k * B * S / E)   # expert capacity per device per expert
# For B=8, S=1: C = ceil(8*8*1/256) = ceil(0.25) = 1

# Shape of the all-to-all input tensor: (N, C, H) per the grouped-by-destination layout.
dispatch_shape = (N, C, H)   # = (8, 1, 7168) for this example

# Create a host-side tensor with random BF16 data.
host_tensor = torch.randn(dispatch_shape, dtype=torch.bfloat16)

# Place the tensor on the mesh. The tensor is replicated across devices here
# for simplicity; in production the tensor is already resident on the mesh
# from the prior layer's output. Adjust as needed.
mesh_tensor = ttnn.from_torch(
    host_tensor,
    dtype=ttnn.bfloat16,
    device=mesh_device,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
```

For a prefill scenario, adjust $B$ and $S$ to match the target sequence length (e.g., $B = 32$, $S = 2048$).

**Tile alignment.** Verify that $C \times H$ is divisible by the TTNN tile size (32 elements per dimension). If $C$ or $H$ is not tile-aligned, pad before the benchmark and unpad after, as you would in production.

---

## Step 3: Warm-Up

Run 10 warm-up iterations of the all-to-all at the `num_links` value under test before recording any measurements. Warm-up is essential because:

1. **JIT compilation.** TTNN compiles kernels on first use for a given (shape, dtype, topology, num_links) combination. The first few iterations carry compilation overhead that inflates measured latency.
2. **Cache priming.** L1 and DRAM caches should be in a steady state consistent with production access patterns before timing begins.
3. **Firmware link training.** Ethernet link state machines may take a few transfers to reach steady-state throughput after being initialized or after a period of inactivity.

```python
WARMUP_ITERS = 10

for _ in range(WARMUP_ITERS):
    _ = ttnn.all_to_all(
        mesh_tensor,
        cluster_axis=1,
        mesh_device=mesh_device,
        num_links=num_links_under_test,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)  # wait for device to complete before next iteration
```

---

## Step 4: Measure Latency Over Many Iterations

After warm-up, record per-iteration latency for at least 100 iterations. Use host-side timestamps bracketing a `ttnn.synchronize_device` call to measure device-side completion time.

```python
import time

MEASURE_ITERS = 100
latencies_us = []

for _ in range(MEASURE_ITERS):
    t_start = time.perf_counter()

    output = ttnn.all_to_all(
        mesh_tensor,
        cluster_axis=1,
        mesh_device=mesh_device,
        num_links=num_links_under_test,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    ttnn.synchronize_device(mesh_device)

    t_end = time.perf_counter()
    latencies_us.append((t_end - t_start) * 1e6)  # convert to microseconds
```

**Host-side timestamp caveats.** Host-side timestamps include host scheduling overhead (OS context switches, Python interpreter overhead). This overhead introduces variability of up to tens of microseconds per measurement. For sub-100-µs operations (decode at small batch), consider using the TTNN device profiler for device-side timing, which excludes host overhead.

If using the TTNN profiler:

```bash
# Set environment variable before launching the Python process.
export TT_METAL_ENABLE_PROFILER=1
```

Then read the profiler output CSV to extract op-level device-side timing. See Chapter 6, `ttnn_profiler.md`, for details on reading the profiler output.

---

## Step 5: Sweep `num_links`

Repeat Steps 3 and 4 for each `num_links` value under test. At minimum, sweep: 1, 2, and the maximum available value. If the maximum is 4, sweep {1, 2, 4}.

```python
import statistics

valid_num_links = [1, 2, 4]  # verify max against T3K documentation
results = {}

for nl in valid_num_links:
    # Warm up this num_links configuration.
    for _ in range(WARMUP_ITERS):
        _ = ttnn.all_to_all(
            mesh_tensor,
            mesh_device,
            num_links=nl,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ttnn.synchronize_device(mesh_device)

    # Measurement.
    lats = []
    for _ in range(MEASURE_ITERS):
        t0 = time.perf_counter()
        _ = ttnn.all_to_all(
            mesh_tensor, cluster_axis=1, mesh_device=mesh_device,
            num_links=nl, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        ttnn.synchronize_device(mesh_device)
        lats.append((time.perf_counter() - t0) * 1e6)

    results[nl] = {
        "median_us": statistics.median(lats),
        "p95_us": sorted(lats)[int(0.95 * MEASURE_ITERS) - 1],
        "min_us": min(lats),
    }
    print(f"num_links={nl}: median={results[nl]['median_us']:.1f} µs  "
          f"p95={results[nl]['p95_us']:.1f} µs  min={results[nl]['min_us']:.1f} µs")
```

---

## Interpreting Results

### The Latency-vs-`num_links` Curve

Plot or tabulate median latency against `num_links`. Three outcomes are possible:

1. **Monotonically decreasing.** Latency falls with each additional link. The workload is throughput-bound and the maximum `num_links` is optimal.
2. **Decreasing then flat (knee).** Latency falls up to some value $n_l^*$ and then levels off. Adding more links beyond $n_l^*$ does not help because the workload has saturated the links it already has or because the bottleneck has shifted to something other than link bandwidth (e.g., device-side buffer management). Use $n_l^*$.
3. **Decreasing then increasing (U-shape).** Latency falls initially but rises at higher `num_links`. The workload is small enough that link setup overhead $\tau_{\text{setup}}$ dominates at high `num_links`. The optimal value is the minimum of the U-curve.

The model from `num_links_parameter.md` predicts outcomes (1) and (3) based on payload size relative to the crossover volume $V^*$. Outcome (2) indicates a non-bandwidth bottleneck at high `num_links` and is common when the per-device buffer allocation or the collective coordination protocol becomes the bottleneck.

### Use Median or 95th Percentile, Not Mean

The mean latency across 100 iterations is sensitive to outliers caused by host-side OS scheduling delays, which can be 100–1000× the median latency. A single 1 ms outlier among 99 measurements of 20 µs each inflates the mean by ~10 µs but does not represent steady-state performance.

Use **median** latency as the primary comparison metric for choosing between `num_links` values. Use **95th percentile (p95)** to characterize tail latency, which affects user-perceived responsiveness in interactive inference. The minimum latency across the run is useful as an upper bound on achievable best-case performance but is susceptible to anomalously fast outliers and should not be used alone.

### Run Three Independent Sweeps

Hardware and OS noise can cause one sweep run to produce anomalously low or high results for one `num_links` value. Run the full sweep at least 3 times (each with its own warm-up) and take the minimum median across runs for each `num_links` value. If the three runs disagree by more than ~10%, investigate the source of variability before drawing conclusions.

---

## Reference Results Table Structure

The table below should be populated with measurements from your specific T3K system. Until measured, all cells are marked as placeholder.

Rows represent tensor sizes (characterized by the per-hop volume in bytes), columns represent `num_links` values, and cells contain median latency in µs.

| Per-hop volume | Regime | `num_links=1` (µs) | `num_links=2` (µs) | `num_links=4` (µs) |
|---|---|---|---|---|
| ~14 KiB ($B=1$, $S=1$) | decode | [placeholder] | [placeholder] | [placeholder] |
| ~14 KiB ($B=8$, $S=1$) | decode | [placeholder] | [placeholder] | [placeholder] |
| ~14 KiB ($B=32$, $S=1$) | decode | [placeholder] | [placeholder] | [placeholder] |
| ~112 KiB ($B=1$, $S=256$) | prefill | [placeholder] | [placeholder] | [placeholder] |
| ~28 MiB ($B=32$, $S=2048$) | prefill | [placeholder] | [placeholder] | [placeholder] |

**Note:** Per-hop volumes above are computed using $H = 7168$ and $C = \lceil k \times B \times S / E \rceil$ with $k = 8$ [D UNVERIFIED] and $E = 256$ [D UNVERIFIED]. All cell values require measurement on the target T3K system.

---

## When to Re-Benchmark

The optimal `num_links` for a given workload can change after any of the following events:

- **TTNN version upgrade.** Collective implementation changes, kernel recompilation, or link training improvements can alter the crossover volume $V^*$ and the effective $\tau_{\text{setup}}$.
- **Firmware or driver update.** Wormhole Ethernet firmware updates can change link initialization overhead and steady-state bandwidth. Re-benchmark after any firmware update that mentions Ethernet performance.
- **Significant workload change.** A change in batch size, sequence length, or expert count shifts the per-hop volume and may move the operating point across the crossover volume $V^*$.
- **Multi-board or multi-host deployment.** The link characteristics and contention behavior on inter-board or inter-host links differ from intra-board links. Re-benchmark for each new topology.
- **Concurrent workload changes.** If the number or type of concurrent collectives changes (e.g., adding a pipelined combine all-to-all), link contention behavior changes and the optimal `num_links` for each concurrent operation may need to be re-evaluated.

---

## Teardown

After benchmarking, close the `MeshDevice` to release hardware resources:

```python
ttnn.close_mesh_device(mesh_device)
```

Do not leave the `MeshDevice` open between benchmark runs or across Python sessions. Stale device state can cause the first warm-up iteration of a subsequent run to exhibit anomalously high latency, biasing results.

---

## References

- Chapter 1, `ethernet_link_bandwidth.md` — per-link bandwidth ($\approx 12.5\ \text{GB/s}$) and saturation thresholds
- Chapter 2, `mesh_device_setup.md` — `MeshDevice` construction parameters and teardown sequence
- Chapter 2, `collective_primitives.md` — `ttnn.all_to_all` API signature and `num_links` parameter
- Chapter 3, `num_links_parameter.md` — bandwidth model and the $T(n_l, V)$ formula with $\tau_{\text{setup}}$
- Chapter 3, `all_to_all_in_moe.md` — per-hop volume derivation for Qwen3.5-35B
- Chapter 6, `ttnn_profiler.md` — TTNN device-side profiler for sub-µs timing accuracy
