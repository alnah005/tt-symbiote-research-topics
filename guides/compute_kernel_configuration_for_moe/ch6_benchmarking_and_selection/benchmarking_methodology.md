# Benchmarking Methodology for MoE Expert Matmul

## What to Measure and Why

Before sweeping kernel configs, establish what you are measuring and from which source. Two common traps corrupt MoE benchmarks:

1. **Wall-clock Python time** includes Python dispatch overhead, device sync latency, and host-side scheduling jitter. For a single expert matmul that may execute in 20–200 µs on-device, Python overhead can be the same order of magnitude.
2. **Program cache warm-up effects** cause the first one to three invocations of any op to appear slower due to JIT compilation and kernel dispatch caching. Timing those iterations produces pessimistic latency numbers that do not represent steady-state performance.

The correct measurement source is the on-device op trace from **Tracy** or **`ttnn.device.profiler`**. Both report the time between the op's start and end events on the device, independent of host-side Python scheduling. All latency numbers in this chapter are on-device microseconds.

---

## Constructing a Standalone Benchmark

A standalone benchmark isolates a single expert matmul — one projection, one config — from the full model forward pass. This eliminates data-dependent dispatch ordering, reduces the risk of measuring pipeline-stall artifacts from neighboring ops, and makes it easy to iterate quickly.

### Minimal Benchmark Template

```python
import torch
import ttnn

# -------------------------------------------------------------------------
# Hardware setup
# -------------------------------------------------------------------------
device = ttnn.open_device(device_id=0)
ttnn.enable_program_cache(device)  # enable caching so warm-up iters compile once

# -------------------------------------------------------------------------
# Synthetic input tensors
# Use random bfloat16 inputs; scale to match typical activation magnitudes
# (~1.0 for layer-normed hidden states, ~0.1 for gate logits post-projection)
# -------------------------------------------------------------------------
BATCH = 32          # decode batch size; set to 1 for single-token decode
SEQ   = 1           # decode: 1; change to 512+ for prefill regime
D_MODEL = 7168      # DeepSeek-V3; adjust for your model
D_FF    = 2048      # per-expert intermediate dim

# Gate projection: [batch*seq, d_model] x [d_model, d_ff]
a_torch = torch.randn(BATCH * SEQ, D_MODEL, dtype=torch.bfloat16) * 0.1
w_torch = torch.randn(D_MODEL, D_FF,        dtype=torch.bfloat16) * 0.02

# Move to device with appropriate memory layout
a_ttnn = ttnn.from_torch(
    a_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
w_ttnn = ttnn.from_torch(
    w_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # weights in DRAM for realistic bandwidth pressure
)

# -------------------------------------------------------------------------
# Config under test — swap this object to sweep different configs
# -------------------------------------------------------------------------
config_under_test = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# -------------------------------------------------------------------------
# Warm-up: run WARMUP_ITERS iterations to populate program cache and
# steady-state device state; do NOT record timing for these
# -------------------------------------------------------------------------
WARMUP_ITERS = 3
for _ in range(WARMUP_ITERS):
    _ = ttnn.matmul(a_ttnn, w_ttnn, compute_kernel_config=config_under_test)
ttnn.synchronize_device(device)  # wait for all warm-up ops to complete

# -------------------------------------------------------------------------
# Timed run: use ttnn.device.profiler to capture per-op device timestamps
# Do NOT use time.perf_counter() here — it measures host time, not device time
# -------------------------------------------------------------------------
TIMED_ITERS = 20

with ttnn.device.profiler.enabled():
    for _ in range(TIMED_ITERS):
        output = ttnn.matmul(a_ttnn, w_ttnn, compute_kernel_config=config_under_test)
    ttnn.synchronize_device(device)

# Retrieve per-op latency from the profiler trace (µs)
# The profiler returns one entry per dispatched op; take the median across TIMED_ITERS
op_latencies_us = ttnn.device.profiler.get_op_latencies()
median_latency_us = sorted(op_latencies_us)[len(op_latencies_us) // 2]
print(f"Median on-device matmul latency: {median_latency_us:.1f} µs")

ttnn.close_device(device)
```

> **Warning:** Do not time the first invocation after opening a device. Device firmware initialization and kernel compilation happen on the first dispatch and inflate the measured latency by 10–100x. Always separate warm-up iterations from timed iterations.

> **Tip:** If you do not have access to `ttnn.device.profiler`, use Tracy: set the environment variable `TT_METAL_DEVICE_PROFILER=1` before running and open the resulting `.tracy` file with the Tracy profiler UI to read per-op timestamps directly from the timeline.

---

## Sweep Dimensions

To characterize the Pareto frontier of latency vs. accuracy for a given projection, sweep the following dimensions:

| Dimension | Values to sweep | Notes |
|---|---|---|
| `math_fidelity` | `LoFi`, `HiFi2`, `HiFi4` | Sweep all three; `HiFi3` is rarely used for MoE |
| `packer_l1_acc` | `True`, `False` | Always keep one axis fixed to isolate the effect |
| `fp32_dest_acc_en` | `True`, `False` | Only relevant at HiFi4 or when PCC > 0.9995 is required |
| `math_approx_mode` | Hold at `False` | For pure matmul benchmarks, `math_approx_mode` has no effect on the FPU path (Chapter 4); fixing it at `False` removes one free variable without loss of generality |

This gives 3 × 2 × 2 = 12 combinations (with `math_approx_mode` fixed). In practice, many combinations are dominated and you will only measure 4–6 meaningfully distinct operating points.

```python
import itertools

fidelities  = [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]
packer_vals = [True, False]
fp32_vals   = [True, False]

sweep_results = []  # list of (config_dict, latency_us, pcc)

for fidelity, packer, fp32 in itertools.product(fidelities, packer_vals, fp32_vals):
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,  # held constant for pure matmul sweep
        fp32_dest_acc_en=fp32,
        packer_l1_acc=packer,
    )
    # ... run benchmark and PCC measurement (see sections below) ...
    sweep_results.append({
        "math_fidelity":    fidelity,
        "packer_l1_acc":    packer,
        "fp32_dest_acc_en": fp32,
        "latency_us":       median_latency_us,
        "pcc":              pcc_value,
    })
```

---

## Isolating the `packer_l1_acc` Effect

The cleanest way to measure the pure bandwidth benefit of `packer_l1_acc` is a controlled comparison: two configs that are identical in every field except `packer_l1_acc`.

```python
# Config A: packer_l1_acc disabled (baseline)
config_packer_off = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,  # <-- only difference
)

# Config B: packer_l1_acc enabled
config_packer_on = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,   # <-- only difference
)

latency_off = run_benchmark(a_ttnn, w_ttnn, config_packer_off)  # µs
latency_on  = run_benchmark(a_ttnn, w_ttnn, config_packer_on)   # µs

bandwidth_speedup_pct = (latency_off - latency_on) / latency_off * 100
print(f"packer_l1_acc speedup: {bandwidth_speedup_pct:.1f}%")
```

The latency difference between `config_packer_off` and `config_packer_on` is attributable entirely to bandwidth — no numerical change is introduced. The theoretical upper bound on the speedup comes from the bandwidth reduction formula derived in Chapter 3:

```
Redundant read reduction = (K_t−1)/K_t
```

where K_t = K/32. For a decode-mode expert matmul with K=2048 (K_t=64), the theoretical maximum redundant read reduction is (64−1)/64 = 98.4%. For K=7168 (K_t=224), the reduction is (224−1)/224 = 99.6%. See `ch3_packer_l1_acc/throughput_impact.md` for the full derivation and table — do not re-derive here.

The observed end-to-end speedup is lower than the theoretical bandwidth reduction because DRAM bandwidth is not the only bottleneck: compute cycles, NoC routing, and dispatch overhead all contribute to total latency. For decode-mode MoE matmuls (M=1–32), the observed speedup is typically in the 10–40% range.

> **Tip:** Run the packer isolation test first, before sweeping fidelity. If `packer_l1_acc=True` alone gives a 20–30% speedup in your regime, that is the dominant optimization. Fidelity sweep results should be interpreted on top of a `packer_l1_acc=True` baseline, not the `False` default.

---

## Quantifying PCC Impact

PCC (Pearson Cross-Correlation) is the standard correctness metric in tt-metal CI. The target threshold for bfloat16 MoE outputs is typically > 0.999 (99.9% correlation with a float32 reference).

### PCC Measurement Recipe

```python
import torch

def compute_pcc(ttnn_output: ttnn.Tensor, torch_ref: torch.Tensor) -> float:
    """
    Compute PCC between a TTNN output tensor and a float32 PyTorch reference.
    Both tensors are flattened before correlation; this gives token-level PCC
    over the full output.
    """
    # Move TTNN output to CPU as float32
    tt_out_cpu = ttnn.to_torch(ttnn_output).to(torch.float32).flatten()
    ref_flat   = torch_ref.to(torch.float32).flatten()

    # torch.corrcoef expects a [2, N] matrix; row 0 = first tensor, row 1 = second
    stacked = torch.stack([tt_out_cpu, ref_flat], dim=0)
    corr_matrix = torch.corrcoef(stacked)
    return corr_matrix[0, 1].item()  # off-diagonal entry is the cross-correlation


# Compute PyTorch float32 reference
ref_output = torch.matmul(
    a_torch.to(torch.float32),
    w_torch.to(torch.float32),
)  # shape: [batch*seq, d_ff]

# Measure PCC for each config in the sweep
for result in sweep_results:
    tt_out = run_matmul_with_config(a_ttnn, w_ttnn, result["config"])
    result["pcc"] = compute_pcc(tt_out, ref_output)
```

### Token-Level vs. Layer-Level PCC

Token-level PCC flattens the output across the full [M, N] shape and computes a single scalar. This is the correct granularity for benchmarking:

- It captures systematic rounding bias across the K dimension (amplified by large K_t).
- It is not sensitive to the statistical variance of a single output token, which is unreliable for small M (batch=1 decode).

Layer-level PCC is measured by running the full expert FFN layer (gate + up + SiLU + element-mul + down) and comparing the layer output to a float32 reference. This is the final arbiter: a config that passes token-level PCC on each projection in isolation may still fail layer-level PCC if rounding errors compound through the SiLU and element-multiply.

> **Warning:** Do not rely solely on per-element error metrics (e.g., max absolute error, mean squared error) when selecting kernel configs. A matmul output can have high max absolute error on outlier elements while maintaining PCC > 0.9999 — and conversely, a small systematic offset can slightly degrade PCC while keeping per-element errors small. PCC is the correct metric because it is what tt-metal CI gates on.

> **Warning:** PCC results from a single decode token (M=1) are statistically unreliable. The correlation between two [1, N] vectors with random input noise is highly variable. Use seq >= 128 for PCC measurements, or batch size >= 32 for decode-mode benchmarks.

---

## Reporting Benchmark Results

After running the full sweep, report results in a table that allows direct Pareto comparison:

| `math_fidelity` | `packer_l1_acc` | `fp32_dest_acc_en` | Latency (µs) | PCC vs float32 |
|---|---|---|---|---|
| LoFi  | False | False | — (baseline) | — |
| LoFi  | True  | False | — | — |
| HiFi2 | True  | False | — | — |
| HiFi4 | True  | False | — | — |
| HiFi4 | True  | True  | — | — |

Fill in your measured values. The canonical production configs target the row that achieves PCC >= 0.999 at the lowest latency. For most gate/up projections this is LoFi + packer_l1_acc=True. For most down projections this is HiFi2 + packer_l1_acc=True.

---

**Next:** [`config_decision_matrix.md`](./config_decision_matrix.md)
