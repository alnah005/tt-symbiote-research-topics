# CPU Baseline Profiling

This file establishes the CPU latency baseline for the Gemma 4 vision encoder. It covers profiling methodology, expected per-module breakdown, and latency estimates across token budgets and batch sizes. All estimates assume server-class CPUs typically paired with Wormhole cards in production deployments.

## Profiling Methodology

### Tools

Use PyTorch's built-in profiling infrastructure to measure the vision encoder in isolation:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import Gemma4VisionModel, Gemma4VisionConfig

config = Gemma4VisionConfig.from_pretrained("google/gemma-4-31b")
model = Gemma4VisionModel(config).eval().to(torch.bfloat16)

# Synthetic input: 280-token budget => ~840 patches before pooling
# For a roughly square image: ~29x29 patch grid => 841 patches
num_patches = 841
pixel_values = torch.randn(1, num_patches, 768, dtype=torch.bfloat16)
position_ids = torch.stack([
    torch.arange(29).unsqueeze(1).expand(29, 29).reshape(-1),
    torch.arange(29).unsqueeze(0).expand(29, 29).reshape(-1),
], dim=-1).unsqueeze(0)  # [1, 841, 2]

# Warm up
for _ in range(5):
    with torch.no_grad():
        model(pixel_values, position_ids)

# Profile
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with torch.no_grad():
        for _ in range(20):
            model(pixel_values, position_ids)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
```

### Measurement Protocol

1. **Isolate the vision encoder.** Do not profile the full Gemma 4 model — measure `Gemma4VisionModel` alone.
2. **Use BF16 throughout.** This matches the dtype used for TTNN inference and gives the CPU the benefit of reduced memory traffic.
3. **Warm up for at least 5 forward passes** to ensure JIT compilation, memory allocation, and cache warming are complete.
4. **Average over 20+ iterations** and report the median to reduce variance from OS scheduling.
5. **Pin to a single NUMA node** using `numactl --cpunodebind=0 --membind=0` to avoid cross-socket memory access noise.
6. **Disable MKL threading for single-stream measurement** or set `OMP_NUM_THREADS` to the target production value.

### Hardware Assumptions

Server-class CPUs typically paired with Wormhole B0 cards:

| CPU Family | Typical Config | Estimated BF16 TOPS |
|-----------|---------------|---------------------|
| Intel Xeon Sapphire Rapids | 48 cores, AVX-512 + AMX | ~200 TOPS (AMX BF16) |
| AMD EPYC Genoa | 96 cores, AVX-512 | ~150 TOPS (AVX-512 BF16) |

> **Tip:** The AMX BF16 instructions on Sapphire Rapids provide substantially better matmul throughput than pure AVX-512. If the deployment target uses Sapphire Rapids, CPU latency will be closer to the optimistic end of the ranges below. However, PyTorch eager mode often does not fully utilize AMX, so real-world performance is typically 20-40% of peak.

## Per-Module Latency Breakdown

The 570M-parameter vision encoder has a highly uneven compute distribution. The following breakdown applies to the default 280-token budget (841 patches, sequence length ~841) at batch size 1:

### FLOP Estimation by Module

For a single forward pass at sequence length $S = 841$, hidden dimension $H = 1152$, intermediate size $I = 4304$, and $L = 27$ layers:

**Patch embedding (linear projection):**
- FLOPs: $2 \times S \times 768 \times H = 2 \times 841 \times 768 \times 1152 \approx 1.49 \times 10^9$

**Per encoder layer:**
- QKV projections: $2 \times S \times H \times 3H = 2 \times 841 \times 1152 \times 3456 \approx 6.70 \times 10^9$
- Attention score matmul: $2 \times \text{num\_heads} \times S^2 \times \text{head\_dim} = 2 \times 16 \times 841^2 \times 72 \approx 1.63 \times 10^9$
- Attention value matmul: $2 \times \text{num\_heads} \times S^2 \times \text{head\_dim} \approx 1.63 \times 10^9$
- Output projection: $2 \times S \times H \times H = 2 \times 841 \times 1152^2 \approx 2.23 \times 10^9$
- MLP gate-projection: $2 \times S \times H \times I = 2 \times 841 \times 1152 \times 4304 \approx 8.35 \times 10^9$
- MLP up-projection: $2 \times S \times H \times I \approx 8.35 \times 10^9$
- MLP down-projection: $2 \times S \times I \times H \approx 8.35 \times 10^9$
- **Total per layer:** ~$37.3 \times 10^9$ FLOPs

**27 encoder layers total:** $27 \times 37.3 \times 10^9 \approx 1{,}007 \times 10^9$ FLOPs

**Pooling + projection:**
- Adaptive average pooling: negligible FLOPs (mean reductions)
- RMSNorm: $\sim 3 \times S \times H \approx 2.9 \times 10^6$
- Linear projection to LM dim: $2 \times 280 \times 1152 \times 5376 \approx 3.47 \times 10^9$

**Total vision encoder:** ~$1{,}010 \times 10^9$ FLOPs

### Breakdown Percentages

| Module | FLOPs (GFLOPs) | Share of Total |
|--------|----------------|---------------|
| Patch embedding | 1.5 | 0.1% |
| 27 encoder layers (attention + MLP) | 1,007 | 99.5% |
| &mdash; of which attention matmuls | 329 | 32.6% |
| &mdash; of which MLP matmuls | 678 | 67.1% |
| Pooling + projection | 3.5 | 0.3% |
| **Total** | **1,010** | **100%** |

> **Warning:** The encoder layers dominate at ~99% of total FLOPs. This means CPU profiling will show virtually all time in the 27-layer stack. Optimizing the patch embedder or pooler on CPU is irrelevant — only accelerating the encoder layers matters.

## CPU Latency Estimates

### Methodology for Estimates

The estimates below use two bounds:

- **Optimistic:** assumes 30% of peak CPU BF16 throughput (good AMX utilization via MKL/oneDNN) = ~60 effective TOPS on Sapphire Rapids
- **Conservative:** assumes 15% of peak throughput (eager-mode PyTorch, mixed op types, memory-bound attention) = ~30 effective TOPS

These utilization rates are consistent with published benchmarks for PyTorch eager-mode transformer inference on server CPUs.

### Single-Image Latency by Token Budget

The table below shows estimated wall-clock latency for a single image (batch=1) across the five supported token budgets. Sequence length before pooling is approximately $3 \times \text{token\_budget}$ (since `pooling_kernel_size=3` reduces the count by ~3x per spatial dimension, i.e., ~9x total, but the budget-to-patch mapping is more nuanced — see Chapter 1).

| Token Budget | Approx. Patches (seq_len) | Total GFLOPs | CPU Latency (Optimistic) | CPU Latency (Conservative) |
|-------------|---------------------------|-------------|------------------------|---------------------------|
| 70 | ~210 | 235 | 3.9 ms | 7.8 ms |
| 140 | ~420 | 482 | 8.0 ms | 16.1 ms |
| 280 | ~840 | 1,010 | 16.8 ms | 33.7 ms |
| 560 | ~1680 | 2,192 | 36.5 ms | 73.1 ms |
| 1120 | ~3360 | 5,087 | 84.8 ms | 169.6 ms |

> **Tip:** The FLOPs scale super-linearly with sequence length because attention has $O(S^2)$ cost. At 1120 tokens, the attention matmuls become a larger fraction of the total, pushing overall FLOPs beyond a simple 4x multiple of the 280-token case.

### Batch Scaling

CPU matmul throughput generally improves with larger batches up to a point (better AMX tile utilization, amortized overhead). However, for vision encoder inference, batching increases both the compute and the memory footprint.

| Batch Size | Token Budget | Total GFLOPs | CPU Latency (Optimistic) | CPU Latency (Conservative) |
|-----------|-------------|-------------|------------------------|---------------------------|
| 1 | 280 | 1,010 | 16.8 ms | 33.7 ms |
| 4 | 280 | 4,040 | 64.1 ms | 128.3 ms |
| 8 | 280 | 8,080 | 120.2 ms | 240.5 ms |
| 1 | 1120 | 5,087 | 84.8 ms | 169.6 ms |
| 4 | 1120 | 20,348 | 323.0 ms | 645.9 ms |
| 8 | 1120 | 40,696 | 605.6 ms | 1,211.2 ms |

> **Warning:** At batch=8, token budget=1120, the CPU vision encoder alone could consume 606-1,211 ms. For context, the Gemma 4 31B language model on Wormhole completes a decode step in roughly 15-25 ms. The vision encoder would represent a catastrophic pipeline stall.

### Scaling Notes

- Batch latency does not scale perfectly linearly — at larger batches, better utilization of CPU vector units provides a modest efficiency gain (roughly 10-15% better FLOPs/s at batch=8 vs. batch=1). The estimates above already assume a small efficiency improvement.
- Memory becomes a concern at large batch sizes. At batch=8, 1120 tokens, the encoder activations alone require roughly $8 \times 3360 \times 1152 \times 2 \times 27 \approx 1.7$ GB in BF16. This fits comfortably in server DRAM but may pressure L3 cache.
- PyTorch's `torch.compile` with `mode="reduce-overhead"` can improve CPU throughput by 20-40% over eager mode for fixed-shape inputs. If the decision is to remain on CPU, this should be evaluated.

## Profiling Tips

### Isolating Attention vs. MLP

To determine whether the encoder layers are attention-bound or MLP-bound on a specific CPU:

```python
# Wrap the attention and MLP in record_function for fine-grained profiling
for layer in model.encoder.layers:
    original_attn_forward = layer.self_attn.forward
    original_mlp_forward = layer.mlp.forward

    def make_attn_wrapper(attn_fn, idx):
        def wrapper(*args, **kwargs):
            with record_function(f"attn_layer_{idx}"):
                return attn_fn(*args, **kwargs)
        return wrapper

    def make_mlp_wrapper(mlp_fn, idx):
        def wrapper(*args, **kwargs):
            with record_function(f"mlp_layer_{idx}"):
                return mlp_fn(*args, **kwargs)
        return wrapper

    layer.self_attn.forward = make_attn_wrapper(original_attn_forward, i)
    layer.mlp.forward = make_mlp_wrapper(original_mlp_forward, i)
```

### Checking for Memory-Bound Behavior

If profiled latency significantly exceeds the compute-bound estimate, the workload is likely memory-bound. Indicators:

- Attention at long sequence lengths (S > 1000) may become memory-bound due to the $O(S^2)$ intermediate tensor
- MLP matmuls at `1152 x 4304` are moderately sized and typically compute-bound on server CPUs
- Monitor memory bandwidth utilization with `perf stat` or Intel VTune to confirm

## Key Takeaway

The 27 encoder layers are responsible for ~99% of the vision encoder's compute. At the default 280-token budget with batch=1, CPU latency is roughly 17-34 ms. This is non-trivial but may be acceptable depending on the deployment scenario. At higher token budgets or batch sizes, CPU latency grows rapidly and becomes a clear bottleneck. The next file, [`ttnn_latency_projection.md`](./ttnn_latency_projection.md), quantifies the alternative.

---

**Next:** [`ttnn_latency_projection.md`](./ttnn_latency_projection.md) — First-principles TTNN latency estimation and break-even analysis.
