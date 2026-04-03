# TTNN Latency Projection

This file estimates the Gemma 4 vision encoder latency on Wormhole B0 hardware using TTNN, working from first principles (FLOP counts, hardware throughput, memory bandwidth) and calibrating against the existing Gemma 3 SigLIP TTNN encoder performance.

## Wormhole B0 Hardware Specs

Key performance parameters for a single Wormhole B0 chip:

| Parameter | Value |
|----------|-------|
| Compute grid | 8x8 = 64 Tensix cores |
| BF16 matmul peak throughput | ~262 TOPS (BF16 accumulate) |
| FP32 matmul peak throughput | ~33 TOPS |
| SRAM per core | 1.5 MB (L1) |
| Total SRAM | 96 MB |
| DRAM capacity | 12 GB |
| DRAM bandwidth | 288 GB/s |
| Host-to-device PCIe bandwidth | ~12.8 GB/s (PCIe Gen4 x16) |

> **Tip:** TTNN matmul performance is highly dependent on tensor shapes and tiling efficiency. The 32x32 tile size means dimensions not divisible by 32 require padding, which reduces effective throughput. For Gemma 4's `hidden_size=1152 = 36 x 32`, the hidden dimension tiles perfectly. However, `head_dim=72` and `intermediate_size=4304` require padding to 96 and 4320 respectively.

## FLOP Counts Recap

From [cpu_baseline_profiling.md](./cpu_baseline_profiling.md), the total vision encoder FLOPs at the default 280-token budget (sequence length ~841):

| Component | GFLOPs |
|-----------|--------|
| Patch embedding | 1.5 |
| 27 encoder layers | 1,007 |
| Pooling + projection | 3.5 |
| **Total** | **1,010** |

## First-Principles TTNN Latency Estimate

### Matmul-Bound Estimate

Assuming the vision encoder is matmul-dominated (which it is — see FLOP breakdown above), the theoretical minimum latency is:

$$t_{\text{compute}} = \frac{\text{Total FLOPs}}{\text{Effective Throughput}}$$

Effective throughput depends on utilization. For moderately sized matmuls like those in the Gemma 4 vision encoder:

| Utilization Scenario | Effective TOPS | Basis |
|---------------------|---------------|-------|
| Peak (theoretical) | 262 | Datasheet; unreachable in practice |
| High utilization (large matmuls, well-tiled) | 150-180 | Observed for language model matmuls with dim >= 2048 |
| Moderate utilization (Gemma 4 vision shapes) | 80-120 | `hidden_size=1152`, `intermediate_size=4304` — moderately sized; some padding overhead |
| Low utilization (small/irregular shapes) | 40-60 | Very short sequences or poorly tiled dimensions |

For the Gemma 4 vision encoder at 280-token budget, the shapes are moderately sized:
- QKV projection: `[841, 1152] x [1152, 3456]` — good utilization
- Attention matmul: 16 heads of `[841, 72] x [72, 841]` — `head_dim=72` pads to 96, some overhead
- MLP (gate, up, down): `[841, 1152] x [1152, 4304]` (x2) and `[841, 4304] x [4304, 1152]` — `4304` pads to `4320`, minimal overhead
- Sequence length 841 pads to 864 (27 x 32) — modest padding

**Estimated effective throughput: 90-130 TOPS** for these shapes.

| Estimate | Effective TOPS | Latency |
|----------|---------------|---------|
| Optimistic | 130 | 7.8 ms |
| Mid-range | 110 | 9.2 ms |
| Conservative | 90 | 11.2 ms |

### Memory-Bandwidth Estimate

For each encoder layer, the dominant memory traffic is loading weights from DRAM:

- QKV weights: $1152 \times 3456 \times 2 = 7.96$ MB
- Output projection: $1152 \times 1152 \times 2 = 2.65$ MB
- MLP gate: $1152 \times 4304 \times 2 = 9.92$ MB
- MLP up: $1152 \times 4304 \times 2 = 9.92$ MB
- MLP down: $4304 \times 1152 \times 2 = 9.92$ MB
- **Total per layer:** ~40.4 MB
- **27 layers:** ~1,091 MB

At 288 GB/s DRAM bandwidth:

$$t_{\text{bandwidth}} = \frac{1{,}091 \text{ MB}}{288 \text{ GB/s}} = 3.8 \text{ ms}$$

Since $t_{\text{compute}} > t_{\text{bandwidth}}$ at batch=1, the workload is **compute-bound** at batch=1. At larger batch sizes, activations grow but weight loading stays constant, so it remains compute-bound.

> **Tip:** The compute-bound nature of this workload is good news for TTNN — it means the speedup over CPU scales with the compute throughput ratio, not the memory bandwidth ratio.

### Non-Matmul Overhead

Beyond matmuls, the encoder includes:
- **RoPE application:** element-wise multiply of cos/sin tables with Q/K tensors. For the CPU-precomputed strategy (recommended for initial bringup), this is a device-side element-wise op. Estimated overhead: ~0.05 ms per layer, ~1.4 ms total.
- **RMSNorm:** two norms per layer (pre-attention, pre-MLP) plus one for the projector. Estimated: ~0.02 ms per norm, ~1.1 ms total.
- **Residual additions:** two per layer, negligible.
- **Softmax in attention:** ~0.03 ms per layer, ~0.8 ms total.
- **Op dispatch overhead:** each layer has ~15-20 TTNN op calls. At ~5-10 us per dispatch, this adds ~2-5 ms for 27 layers.

**Total non-matmul overhead estimate: 5-8 ms.**

### Combined TTNN Latency Estimate (280 Tokens, Batch=1)

| Component | Latency Estimate |
|-----------|-----------------|
| Matmul compute | 7.8 - 11.2 ms |
| Non-matmul ops | 5.0 - 8.0 ms |
| **Total (no tracing)** | **12.8 - 19.2 ms** |
| **Total (with tracing)** | **9.0 - 14.5 ms** |

> **Tip:** Tracing eliminates per-op dispatch overhead and enables the TTNN runtime to pipeline ops more aggressively. For a 27-layer encoder with ~400+ ops, tracing can reduce latency by 30-40%. However, tracing requires fixed tensor shapes, which conflicts with variable-resolution input. The recommendation is to trace the default 280-token budget first, then extend to other budgets.

## Reference: Gemma 3 SigLIP TTNN Performance

The existing Gemma 3 SigLIP encoder on TTNN provides a calibration point. While exact numbers depend on the firmware version and optimization level, the following are representative:

| Parameter | Gemma 3 SigLIP | Gemma 4 Vision |
|----------|----------------|----------------|
| hidden_size | 1152 | 1152 |
| num_layers | 27 | 27 |
| num_heads | 16 | 16 |
| head_dim | 72 | 72 |
| intermediate_size | 4304 | 4304 |
| Input | Fixed 896x896, seq_len=4096 | Variable, seq_len~841 (280 budget) |
| Position encoding | Absolute (learned) | 2D RoPE + 2D learned |
| Approx. GFLOPs | ~3,800 | ~1,010 |

The Gemma 3 SigLIP encoder processes a much longer sequence (4096 vs. ~841) because it uses patch_size=14 on 896x896 images ($64 \times 64 = 4096$ patches). At 4096 tokens, the matmuls are larger and achieve better utilization. However, the per-layer weight sizes are identical, so the weight-loading cost is the same.

**Implication:** If Gemma 3 SigLIP achieves X ms for 4096 tokens, the Gemma 4 encoder at 280 tokens should be substantially faster because:
1. Sequence length is ~5x shorter, reducing attention FLOPs by ~25x
2. MLP FLOPs scale linearly with sequence length, reducing by ~5x
3. Total FLOPs are ~1,010 vs. ~3,800 GFLOPs (roughly 3.8x fewer)

The speedup will not be a full 3.8x because smaller matmuls have lower utilization, but a 2-3x reduction in TTNN latency relative to the Gemma 3 SigLIP encoder is a reasonable expectation.

## TTNN Latency Across Token Budgets and Batch Sizes

| Token Budget | Seq Len | GFLOPs | TTNN Latency (Optimistic) | TTNN Latency (Conservative) |
|-------------|---------|--------|--------------------------|----------------------------|
| 70 | ~210 | 235 | 5.0 ms | 9.0 ms |
| 140 | ~420 | 482 | 6.5 ms | 12.0 ms |
| 280 | ~840 | 1,010 | 9.5 ms | 14.5 ms |
| 560 | ~1680 | 2,192 | 15.0 ms | 23.0 ms |
| 1120 | ~3360 | 5,087 | 26.0 ms | 40.0 ms |

| Batch Size | Token Budget | GFLOPs | TTNN Latency (Optimistic) | TTNN Latency (Conservative) |
|-----------|-------------|--------|--------------------------|----------------------------|
| 1 | 280 | 1,010 | 9.5 ms | 14.5 ms |
| 4 | 280 | 4,040 | 18.0 ms | 28.0 ms |
| 8 | 280 | 8,080 | 29.0 ms | 45.0 ms |
| 1 | 1120 | 5,087 | 26.0 ms | 40.0 ms |
| 4 | 1120 | 20,348 | 70.0 ms | 112.0 ms |
| 8 | 1120 | 40,696 | 120.0 ms | 190.0 ms |

> **Tip:** TTNN batch scaling is significantly better than CPU batch scaling. At batch=8, 280 tokens, TTNN latency is roughly 3x the batch=1 latency (vs. ~7x on CPU) because the larger effective matmul sizes improve Tensix utilization.

## Break-Even Analysis

TTNN execution incurs a fixed overhead for host-to-device data transfer that CPU execution avoids. The break-even point is where TTNN's compute advantage overcomes this transfer cost.

### Host-to-Device Transfer Cost

Input data to transfer for the vision encoder:

| Data | Size (batch=1, 280 tokens) | Transfer Time (PCIe Gen4 x16) |
|------|---------------------------|------------------------------|
| Pixel values `[1, 841, 768]` in BF16 | 1.29 MB | 0.10 ms |
| Position IDs `[1, 841, 2]` in INT32 | 0.007 MB | ~0 ms |
| **Total** | **~1.3 MB** | **~0.1 ms** |

At batch=8, 1120 tokens: ~41 MB, ~3.2 ms transfer time.

> **Warning:** If the vision encoder weights must be loaded to device at the start of each request (no persistent weight placement), the weight transfer cost is ~1.1 GB (570M params in BF16), taking ~86 ms over PCIe Gen4. This would dominate latency. **Weights must be pre-loaded and persistent on device.**

### Break-Even Comparison

| Scenario | CPU Latency (Mid) | TTNN Latency (Mid) + Transfer | TTNN Speedup |
|----------|-------------------|------------------------------|-------------|
| Batch=1, 70 tokens | 5.9 ms | 7.0 ms + 0.03 ms = 7.0 ms | 0.8x (CPU wins) |
| Batch=1, 140 tokens | 12.1 ms | 9.3 ms + 0.05 ms = 9.3 ms | 1.3x |
| Batch=1, 280 tokens | 25.3 ms | 12.0 ms + 0.1 ms = 12.1 ms | 2.1x |
| Batch=1, 560 tokens | 54.8 ms | 19.0 ms + 0.3 ms = 19.3 ms | 2.8x |
| Batch=1, 1120 tokens | 127.2 ms | 33.0 ms + 0.8 ms = 33.8 ms | 3.8x |
| Batch=4, 280 tokens | 96.2 ms | 23.0 ms + 0.4 ms = 23.4 ms | 4.1x |
| Batch=8, 280 tokens | 180.4 ms | 37.0 ms + 0.8 ms = 37.8 ms | 4.8x |
| Batch=8, 1120 tokens | 908.4 ms | 155.0 ms + 3.2 ms = 158.2 ms | 5.7x |

**Key findings:**

1. **At 70 tokens, batch=1, TTNN may not win.** The matmuls are small, utilization is low, and non-matmul overhead dominates. CPU is competitive here.
2. **At 140+ tokens, batch=1, TTNN starts to pull ahead.** The speedup grows with sequence length because TTNN handles the quadratic attention cost much better.
3. **At batch >= 4, TTNN wins decisively at all token budgets.** The speedup factor reaches 4.1-5.7x.
4. **Transfer overhead is negligible** for pixel data. It is less than 1% of total latency in all scenarios except the very smallest.

## Key Factors Affecting TTNN Speedup

### 1. Attention Matmuls

Shape: 16 heads of `[S, 72] x [72, S]` (score) and `[S, S] x [S, 72]` (value).

At `head_dim=72`, each head's matmul is small. TTNN can batch heads along the batch dimension, yielding an effective matmul of `[16, S, 72] x [16, 72, S]`. This maps reasonably to the 8x8 grid but does not saturate it at short sequence lengths. At S=210 (70 tokens), utilization will be low; at S=3360 (1120 tokens), utilization will be high.

### 2. MLP Matmuls

Shapes: `[S, 1152] x [1152, 4304]` (gate and up projections) and `[S, 4304] x [4304, 1152]` (down projection). The gated MLP has three weight matrices total.

These are the largest matmuls per layer and achieve the best utilization. `1152 = 36 x 32` tiles perfectly; `4304` pads to `4320 = 135 x 32` with only 0.4% overhead. These matmuls will be close to peak throughput for S >= 400.

### 3. RoPE Overhead

For the CPU-precomputed approach (Strategy 1 from [Chapter 3](../ch03_2d_factored_rope/index.md)), the cos/sin tables are computed on the host and transferred to the device. The on-device cost is two element-wise multiplies and an addition per Q and K tensor. This is bandwidth-bound and adds ~1.4 ms across 27 layers.

For the composed-from-TTNN-ops approach (Strategy 2), the overhead may be slightly higher due to the split/concat operations. For initial bringup, Strategy 1 is recommended to minimize TTNN latency uncertainty.

### 4. Variable Shapes and Program Cache

Variable input shapes across different image resolutions cause TTNN to recompile kernels for each new shape. This adds a one-time cost per shape but does not affect steady-state throughput if the shapes repeat.

**Mitigation strategies:**
- Pre-compile all five token budgets at startup
- Pad all inputs within a budget to the maximum sequence length for that budget
- If tracing is used, pre-trace all five budgets

The program cache overhead is a latency concern only for the first inference at each resolution. Subsequent inferences at the same resolution hit the cache and execute at full speed.

## Summary

| Metric | CPU (Mid Estimate) | TTNN (Mid Estimate) | Speedup |
|--------|-------------------|--------------------|---------|
| Batch=1, 280 tokens | 25.3 ms | 12.1 ms | 2.1x |
| Batch=1, 1120 tokens | 127.2 ms | 33.8 ms | 3.8x |
| Batch=8, 280 tokens | 180.4 ms | 37.8 ms | 4.8x |

The TTNN port provides a meaningful speedup across nearly all deployment scenarios. The only exception is the smallest configuration (single image, 70 tokens), where CPU remains competitive. The next file synthesizes these numbers into deployment-specific recommendations.

---

**Next:** [`decision_matrix.md`](./decision_matrix.md) — Deployment-scenario recommendations for CPU vs. TTNN execution.
