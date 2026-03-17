# Throughput Profiling

## Purpose

Step 4 of the workflow uses TTNN's device profiler and Tracy traces to measure per-
operation latency in the expert FFN (feed-forward network). The goals are:

1. Identify which projection is the latency bottleneck under each precision configuration.
2. Quantify the decode memory-bandwidth improvement from bfloat4_b gate/up vs. bfloat16.
3. Compare decode and prefill throughput across bfloat16, bfloat8_b all, and the mixed
   bfloat4_b gate/up + bfloat8_b down configurations.
4. Understand T3K multi-chip communication overhead relative to expert FFN compute.

---

## Using TTNN's Device Profiler and Tracy

TTNN provides two complementary profiling mechanisms: a lightweight program-cache-aware
device profiler that logs per-op device cycles, and Tracy integration for full system
traces including host overhead.

### Program Cache and Device Profiler Setup

```python
import ttnn
import os

def setup_profiler(device):
    """Enable program caching and device profiler for a profiling run.

    The program cache amortizes kernel compilation across the 20-iteration benchmark
    loop. Enable profiler output to the default trace directory.

    Args:
        device: TTNN device handle (single chip or T3K mesh device).
    """
    ttnn.device.enable_program_cache(device)
    # Set environment variable before device creation for Tracy trace output path
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
```

### Tracy Trace Collection

Tracy (https://github.com/wolfpld/tracy) provides a flame-graph view of both device
kernel times and host dispatch overhead. To collect a Tracy trace:

```bash
# Build with Tracy support (requires TT_METAL_TRACY=1 at build time)
TT_METAL_TRACY=1 python profiling_script.py
```

The Tracy profiler captures device kernel start/end timestamps alongside host Python
dispatch times. Use it to identify cases where host overhead (tensor copy, kernel
dispatch) dominates over device compute. For expert FFN matmuls, device time should
dominate; if host overhead exceeds 20% of total latency, check that program cache is
enabled and that tensors are already on device before the timed loop.

> **Tip:** Always warm up the program cache before benchmarking. Run at least 3
> un-timed iterations before starting the 20-iteration timed benchmark. The first
> iteration compiles and caches the kernel; subsequent iterations reuse the compiled
> program and reflect true device latency.

---

## Expert FFN Latency Breakdown

The SwiGLU expert FFN consists of five operations. Profile each separately to identify
the bottleneck:

| Operation | Inputs | Output | Notes |
|---|---|---|---|
| Gate matmul (w1) | `[T, d_model]` × `[d_ff, d_model]` | `[T, d_ff]` | bfloat4_b + LoFi |
| SiLU activation | `[T, d_ff]` | `[T, d_ff]` | Elementwise; SiLU: x × sigmoid(x) |
| Up matmul (w3) | `[T, d_model]` × `[d_ff, d_model]` | `[T, d_ff]` | bfloat4_b + LoFi |
| Elementwise mul | `[T, d_ff]` × `[T, d_ff]` | `[T, d_ff]` | gate × up |
| Down matmul (w2) | `[T, d_ff]` × `[d_model, d_ff]` | `[T, d_model]` | bfloat8_b + HiFi2 |

Where `T` is the number of tokens routed to this expert (decode: T=1; prefill: T up to
`seq_len × top_k / num_experts`).

```python
import time
import torch
import ttnn

NUM_WARMUP = 3
NUM_TIMED  = 20   # benchmark: 20 timed iterations; report median + p95

def benchmark_expert_ffn(x_torch, w1_tt, w3_tt, w2_tt, device, label=""):
    """Benchmark all five FFN operations and report per-op latency.

    Args:
        x_torch: Input activations, torch.bfloat16, shape [num_tokens, d_model].
        w1_tt: Gate weight, TTNN bfloat4_b on device.
        w3_tt: Up weight, TTNN bfloat4_b on device.
        w2_tt: Down weight, TTNN bfloat8_b on device.
        device: TTNN device handle.
        label: Description string for output header.

    Returns:
        dict mapping operation name -> list of latencies in microseconds.
    """
    # Configs: see index.md (COMPUTE_KERNEL_CONFIG_LOFI, COMPUTE_KERNEL_CONFIG_HIFI2)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)

    timings = {op: [] for op in ["gate_matmul", "silu", "up_matmul", "mul", "down_matmul"]}

    for iteration in range(NUM_WARMUP + NUM_TIMED):
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        gate_pre = ttnn.linear(x_tt, w1_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()

        gate_out = ttnn.silu(gate_pre)
        ttnn.synchronize_device(device)
        t2 = time.perf_counter()

        up_out = ttnn.linear(x_tt, w3_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
        ttnn.synchronize_device(device)
        t3 = time.perf_counter()

        inter = ttnn.mul(gate_out, up_out)
        ttnn.synchronize_device(device)
        t4 = time.perf_counter()

        out = ttnn.linear(inter, w2_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)
        ttnn.synchronize_device(device)
        t5 = time.perf_counter()

        if iteration >= NUM_WARMUP:  # skip warmup iterations
            timings["gate_matmul"].append((t1 - t0) * 1e6)
            timings["silu"].append((t2 - t1) * 1e6)
            timings["up_matmul"].append((t3 - t2) * 1e6)
            timings["mul"].append((t4 - t3) * 1e6)
            timings["down_matmul"].append((t5 - t4) * 1e6)

    import statistics
    print(f"\nExpert FFN latency breakdown — {label}")
    print(f"{'Operation':<20} {'Median (µs)':>14} {'p95 (µs)':>10}")
    print("-" * 46)
    for op, vals in timings.items():
        vals_sorted = sorted(vals)
        median_us = statistics.median(vals_sorted)
        p95_us    = vals_sorted[int(0.95 * len(vals_sorted))]
        print(f"{op:<20} {median_us:>14.1f} {p95_us:>10.1f}")

    return timings
```

> **Tip:** Report median and p95, not mean. On Wormhole B0 with noisy workloads, a few
> high-latency outliers can inflate the mean significantly. Median reflects the typical
> case; p95 reveals tail latency relevant to interactive inference SLAs.

---

## Expected Latency Profile Under Mixed Precision

For Qwen 235B-A22B dimensions (`d_model=7168`, `d_ff=2048`) at decode (T=1) on a single
Wormhole B0 chip, the expected per-operation approximate ordering is:

- **Gate matmul (bfloat4_b + LoFi):** Fastest weight read; memory-bound on a single
  vector. Lower latency than bfloat8_b equivalent by ~2× DRAM read time.
- **Up matmul (bfloat4_b + LoFi):** Same as gate; these two operations are symmetric.
- **SiLU + elementwise mul:** Elementwise operations; compute time negligible relative
  to matmuls in decode mode.
- **Down matmul (bfloat8_b + HiFi2):** Wider mantissa and higher fidelity make this the
  controlled accuracy bottleneck. HiFi2 adds a small compute overhead vs. LoFi but
  produces higher PCC for the residual output.

At prefill (T=2048), all three matmuls become closer to the compute-bound regime as
arithmetic intensity rises. The latency gap between bfloat4_b and bfloat8_b narrows
because DRAM read time is amortized over more tokens.

---

## Decode vs. Prefill Throughput Comparison

Run the benchmark at both decode and prefill token counts to quantify the memory-
bandwidth benefit of bfloat4_b gate/up in the memory-bound decode regime.

```python
def compare_decode_prefill(w1_bf16, w3_bf16, w2_bf16, device):
    """Compare gate+up latency for bfloat16 vs bfloat8_b vs bfloat4_b across token counts.

    Runs three weight configurations at decode (T=1) and prefill (T=2048).
    """
    configs = [
        ("bfloat16",          ttnn.bfloat16,   ttnn.bfloat16),
        ("bfloat8_b all",     ttnn.bfloat8_b,  ttnn.bfloat8_b),
        ("bfloat4_b gate/up", ttnn.bfloat4_b,  ttnn.bfloat8_b),
    ]

    for token_count, regime in [(1, "decode"), (2048, "prefill")]:
        x_torch = torch.randn(token_count, 7168, dtype=torch.bfloat16)
        for label, gate_up_dtype, down_dtype in configs:
            w1_tt = ttnn.as_tensor(w1_bf16, dtype=gate_up_dtype,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w3_tt = ttnn.as_tensor(w3_bf16, dtype=gate_up_dtype,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w2_tt = ttnn.as_tensor(w2_bf16, dtype=down_dtype,
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
            benchmark_expert_ffn(
                x_torch, w1_tt, w3_tt, w2_tt, device,
                label=f"{label} | {regime} T={token_count}"
            )
```

For the gate projection alone, the expected decode latency ratios relative to bfloat16
(using n300 effective bandwidth ~461 GB/s, `d_model=7168`, `d_ff=2048`) are:

For the decode latency comparison across quantization formats, see `../ch06_comparative_study/qwen_bfloat16_baseline.md` Table X.

Use measured device latency from `benchmark_expert_ffn` as ground truth.

> **Warning:** On-chip bandwidth from the memory config can differ from theoretical peak.
> Use measured device latency as the ground truth, not the theoretical estimate. If
> measured latency is more than 20% above the theoretical floor, check whether the weight
> tensor is correctly placed in DRAM (`memory_config=ttnn.DRAM_MEMORY_CONFIG`) and whether
> the program cache is active.

---

## T3K Multi-Chip Considerations

On a T3K system (8 Wormhole B0 chips), the expert FFN compute is preceded and followed
by all-to-all communication for expert routing: tokens are dispatched to their assigned
expert chips and results are gathered back. In small-batch decode, this communication
latency often exceeds the expert FFN compute latency.

```python
def profile_t3k_expert_with_communication(mesh_device, x_per_chip, w1_tt, w3_tt, w2_tt):
    """Estimate communication overhead relative to FFN compute on T3K.

    Measures:
      - all_gather latency (simulated by timing a cross-chip tensor operation)
      - Expert FFN compute latency per chip
      - Ratio of communication to compute

    Args:
        mesh_device: TTNN MeshDevice for T3K (8 chips).
        x_per_chip: Per-chip input tensor, shape [T_per_chip, d_model].
        w1_tt, w3_tt, w2_tt: Expert weight tensors on mesh device.
    """
    # Time the FFN compute component on a single chip
    single_device = mesh_device.get_devices()[0]
    ffn_timings = benchmark_expert_ffn(x_per_chip, w1_tt, w3_tt, w2_tt,
                                       single_device, label="T3K single chip FFN")

    ffn_total_median = sum(
        sorted(v)[len(v) // 2] for v in ffn_timings.values()
    )
    print(f"\nT3K per-chip FFN total (median): {ffn_total_median:.1f} µs")
    print("Note: cross-chip all-to-all latency for 8-chip T3K is typically "
          "200–500 µs at small batch — compare against FFN compute above.")
```

The key insight for T3K profiling: if the measured FFN compute is, for example, 50 µs
per expert and the all-to-all routing overhead is 300 µs, then quantization improvements
to FFN compute (reducing it from 50 µs to 30 µs) produce only a ~5% reduction in total
latency. The memory footprint reduction remains fully beneficial regardless — it directly
increases available DRAM for KV cache, reducing eviction pressure.

> **Tip:** On T3K, report both FFN compute latency and estimated communication overhead
> separately. The communication floor determines whether quantization's latency savings
> are visible at the user level. For memory-footprint-dominated decisions (e.g., fitting
> more KV cache), the throughput analysis is secondary to the footprint analysis in
> Chapter 3.

---

## Next Steps

Proceed to `iterative_tuning_guide.md` to apply the profiling results and PCC reports
to the tuning decision tree, run calibration perplexity, and lock in the final per-
projection configuration for regression testing.
