# Qwen bfloat16 Baseline

## Current Implementation State

The Qwen 235B-A22B TTNN implementation uses `ttnn.bfloat16` for all expert weight tensors
across all three projections in every MoE layer. This is the conservative starting point:
no accuracy risk from quantization, no per-layer PCC validation needed, and straightforward
weight loading from the original Hugging Face checkpoint. It is also the most expensive
option in terms of DRAM memory and decode bandwidth.

The current configuration for every expert projection:

| Projection | Dtype | Notes |
|---|---|---|
| Gate (w1) | `ttnn.bfloat16` | Full precision |
| Up (w3) | `ttnn.bfloat16` | Full precision |
| Down (w2) | `ttnn.bfloat16` | Full precision |

No custom `WormholeComputeKernelConfig` is required when all weights are bfloat16 — the
default kernel config handles this case. This simplicity is part of the appeal of the
bfloat16 baseline during initial bringup.

## Memory Cost Analysis

### Per-Expert Footprint

Tile size for bfloat16: 2048 bytes per 32×32 tile (2 bytes per element × 1024 elements).

For a single expert with `d_model=7168` and `d_ff=2048`:

The per-expert BF16 footprint is 84.0 MB (derived in `deepseek_v3_quantization_design.md` in this chapter).

### System-Level Footprint on T3K

With 128 experts distributed across 8 T3K chips (16 experts per chip):

For the T3K system-level memory totals, see the table in `deepseek_v3_quantization_design.md` in this chapter.

The bfloat16 system total of ~10.5 GB is a significant fraction of available DRAM on a T3K,
leaving less headroom for KV cache, activations, and other model components. The mixed-
precision equivalent at ~3.5 GB frees roughly 7 GB system-wide — capacity that translates
directly into larger batch sizes or longer context windows during inference.

> **Tip:** On a single n150 Wormhole B0 chip with ~288 GB/s DRAM bandwidth, the 10.5 GB
> expert weight footprint at bfloat16 is not just a storage problem — it sets a hard lower
> bound on decode latency. Every forward pass must stream all active expert weights from
> DRAM, and the volume of bytes read is proportional to the dtype bit-width.

## Throughput Cost

### Arithmetic Intensity at Decode

During decode, the expert FFN gate projection computes:

```
[batch, d_model] × [d_model, d_ff]  →  [batch, d_ff]
```

For `d_model=7168`, `d_ff=2048`, and bfloat16 weights (2 bytes per element):

```
Arithmetic intensity (batch=1) = (2 × 1 × 7168 × 2048) / (7168 × 2048 × 2)
                               = 1.0 FLOP/byte
```

Wormhole B0's ridge point is approximately 437 FLOP/byte. At 1.0 FLOP/byte, the bfloat16
gate projection is over 400× below the compute-bound regime. Every cycle the Tensix compute
units are idle, waiting for weight data from DRAM.

### DRAM Read Volume Per Projection Pass

For a single bfloat16 gate projection at `d_model=7168`, `d_ff=2048`:

```
Bytes read = 7168 × 2048 × 2 = 29,360,128 bytes ≈ 28.0 MB
```

At effective DRAM bandwidth of ~230 GB/s (80% of 288 GB/s n150 peak accounting for
controller overhead):

```
Latency ≈ 29,360,128 / 230e9 ≈ 127.7 µs per projection per expert
```

Gate and up together: ~255 µs. Down: ~127.7 µs. Total per active expert: ~383 µs.

With `top_k=8` experts active per token, and assuming ideal parallelism across Tensix cores,
the expert FFN decode cost is dominated by DRAM streaming time.

### Throughput Headroom Left on the Table

The Wormhole B0 compute throughput at bfloat16 is approximately 131 TFLOP/s (n150) and
262 TFLOP/s (n300 with 2 chips). These peak figures are not reachable for decode because
the operation is memory-bound. The practical implication is that the hardware's FMA units
are underutilised by a large factor — quantization is the mechanism that raises arithmetic
intensity and allows more of that compute capacity to be used.

> **Warning:** bfloat16 throughput figures (131/262 TFLOP/s) describe peak compute capacity,
> not the achieved throughput for memory-bound decode matmuls. In memory-bound regime, the
> effective throughput is determined by DRAM bandwidth, not FLOP/s. Quoting TFLOP/s for
> decode expert FFN without an arithmetic intensity qualifier is misleading.

## Why Qwen Uses bfloat16

The bfloat16 choice for Qwen is deliberate and understandable in context:

**1. No quantization-aware training.** Qwen 235B-A22B was trained in standard bfloat16.
The weight distributions contain outliers and dynamic range characteristics that have not
been shaped for low-bit storage. Applying bfloat4_b to an unprepared model risks per-layer
PCC drops below acceptable thresholds — and on a 128-expert model, one outlier layer can
degrade generation quality noticeably.

**2. Conservative initial deployment.** When bringing up a new model on TTNN, establishing
a correct bfloat16 baseline before attempting quantization is sound engineering practice.
Debugging mixed-precision accuracy failures is harder when the baseline itself is not yet
validated. The bfloat16 implementation provides the reference PCC > 0.999 that all
quantized variants are measured against.

**3. Simpler weight loading.** bfloat16 weights load directly from the Hugging Face
checkpoint without any dtype conversion step. This reduces bringup friction and eliminates
a class of weight-loading bugs (tile alignment errors, shared exponent computation mistakes)
during initial development.

**4. Accuracy risk aversion.** For a model not trained with quantization awareness, the
accuracy impact of bfloat4_b on gate and up projections is empirically unknown until
validated. The safe default is bfloat16, with quantization as a subsequent optimization pass
backed by rigorous per-layer PCC measurement.

The cost of this conservatism is the 16ms gap described in the next section.

## The 16ms Gap Motivation

The Qwen MoE optimization analysis identified a per-layer latency gap of approximately 16ms
between the measured decode latency and the target latency for production throughput. This
gap has multiple contributing factors, but DRAM bandwidth consumption from bfloat16 expert
weights is a primary contributor.

The mechanism is direct: during decode, each active expert's three weight matrices (~84 MB
total at bfloat16) must be streamed from DRAM every forward pass. With `top_k=8` experts
active per token, the total weight data read per MoE layer is:

```
8 active experts × 84.0 MB = 672 MB per MoE layer per decode step (bfloat16)
8 active experts × 28.0 MB = 224 MB per MoE layer per decode step (mixed precision)
```

At 288 GB/s DRAM bandwidth (n150 peak):

```
bfloat16:     672 MB / 288 GB/s ≈ 2.33 ms lower bound per MoE layer
mixed (3×):   224 MB / 288 GB/s ≈ 0.78 ms lower bound per MoE layer
```

The difference of ~1.55 ms is a lower bound assuming perfect bandwidth utilisation. In
practice, with multiple MoE layers, non-ideal bandwidth utilisation, and routing overhead,
the cumulative impact across all layers contributes substantially to the observed 16ms gap.

Switching to `bfloat8_b` for all projections (the recommended first step for Qwen) would
halve the per-expert weight bytes to 42 MB, reducing the per-MoE-layer DRAM read volume
by 2× and providing immediate decode latency relief without the accuracy risk of bfloat4_b.

---

**Next:** [`recommendations_and_decision_framework.md`](./recommendations_and_decision_framework.md)
