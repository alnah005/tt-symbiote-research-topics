# Memory Budget

This file provides the complete per-device DRAM budget for Gemma 4 31B on T3K,
covering weight memory, KV cache memory, activation memory, and the total DRAM
footprint at various sequence lengths and quantization levels. Each Wormhole
chip has 12 GB of DRAM.

## Weight Memory

### Per-Layer Weight Sizes (Per Device, TP=8)

Weight sizes are derived from the sharding analysis in
[Chapter 6 --- Weight Sharding](../ch6_tp_sharding/weight_sharding.md).

#### Sliding Layer Weights (Per Device)

| Projection | Full Shape | Per-Device Shape (TP=8) | BF16 | BFP8 | BFP4 |
|------------|-----------|------------------------|------|------|------|
| Q | [5376, 8192] | [5376, 1024] | 11.0 MB | 5.5 MB | 2.8 MB |
| K | [5376, 4096] | [5376, 512] | 5.5 MB | 2.8 MB | 1.4 MB |
| V | [5376, 4096] | [5376, 512] | 5.5 MB | 2.8 MB | 1.4 MB |
| O | [8192, 5376] | [1024, 5376] | 11.0 MB | 5.5 MB | 2.8 MB |
| Gate | [5376, 21504] | [5376, 2688] | 28.9 MB | 14.5 MB | 7.2 MB |
| Up | [5376, 21504] | [5376, 2688] | 28.9 MB | 14.5 MB | 7.2 MB |
| Down | [21504, 5376] | [2688, 5376] | 28.9 MB | 14.5 MB | 7.2 MB |
| Norms (6) | --- | --- | ~0.04 MB | ~0.04 MB | ~0.04 MB |
| **Total per sliding layer** | | | **~119.7 MB** | **~60.1 MB** | **~30.0 MB** |

#### Global Layer Weights (Per Device)

| Projection | Full Shape | Per-Device Shape (TP=8) | BF16 | BFP8 | BFP4 |
|------------|-----------|------------------------|------|------|------|
| Q | [5376, 16384] | [5376, 2048] | 22.0 MB | 11.0 MB | 5.5 MB |
| K (replicated) | [5376, 2048] | [5376, 2048] | 22.0 MB | 11.0 MB | 5.5 MB |
| V | N/A (K=V) | N/A | 0 | 0 | 0 |
| O | [16384, 5376] | [2048, 5376] | 22.0 MB | 11.0 MB | 5.5 MB |
| Gate | [5376, 21504] | [5376, 2688] | 28.9 MB | 14.5 MB | 7.2 MB |
| Up | [5376, 21504] | [5376, 2688] | 28.9 MB | 14.5 MB | 7.2 MB |
| Down | [21504, 5376] | [2688, 5376] | 28.9 MB | 14.5 MB | 7.2 MB |
| Norms (6) | --- | --- | ~0.04 MB | ~0.04 MB | ~0.04 MB |
| **Total per global layer** | | | **~152.8 MB** | **~76.5 MB** | **~38.1 MB** |

The global layers are larger per-layer due to the replicated K projection
(full `[5376, 2048]` on every device) and the larger Q and O projections
(head_dim=512 vs 256).

### Total Weight Memory Across All Layers (Per Device)

| Component | BF16 | BFP8 | BFP4 |
|-----------|------|------|------|
| 50 sliding layers | 5,985 MB | 3,005 MB | 1,500 MB |
| 10 global layers | 1,528 MB | 765 MB | 381 MB |
| **Subtotal (60 layers)** | **7,513 MB** | **3,770 MB** | **1,881 MB** |
| Embedding (262144 x 5376, BF16, TP=8) | ~336 MB | ~336 MB | ~336 MB |
| Norms, cos/sin tables, misc | ~50 MB | ~50 MB | ~50 MB |
| **Total weights** | **~7,899 MB** | **~4,143 MB** | **~2,267 MB** |

> **Note on BFP8 rounding:** The per-layer BFP8 totals above (60.1 MB sliding,
> 76.5 MB global) are rounded from per-projection values. Multiplying rounded
> per-layer totals gives 3,770 MB for layers alone. Ch6's bottom-up calculation
> from exact per-projection bytes yields a layer subtotal of ~3,757 MB and a
> grand total of ~4,143 MB, which is the authoritative figure used throughout.

The embedding table is kept at BF16 regardless of weight quantization because
it is the LM head (tied weights) and must preserve output precision for logit
computation.

### Quantization Requirement

At BF16, the model weights alone consume ~7.9 GB of the 12 GB per-device
budget, leaving only ~4.1 GB for KV cache, activations, and runtime buffers.
This is insufficient for contexts beyond ~8K tokens with BF16 KV cache.

**BFP8 weight quantization is required for practical deployment.** At BFP8,
weights consume ~4.1 GB, leaving ~7.9 GB for KV cache and activations ---
sufficient for up to ~65K context with BF16 KV cache.

BFP4 weights (~2.3 GB) would provide even more headroom but may impact model
quality. BFP4 is worth evaluating for memory-constrained scenarios (very long
context or larger batch sizes).

## KV Cache Memory

KV cache sizes are derived from
[Chapter 6 --- KV Cache Sharding](../ch6_tp_sharding/kv_cache_sharding.md).

### Sliding KV Cache (50 Layers)

The sliding window caps KV cache at 1,024 tokens per layer, making it
constant regardless of sequence length.

```math
\text{Sliding KV per device} = 50 \times 2 \times 2_{\text{heads/dev}} \times 1024 \times 256 \times \text{bpe}
```

| Dtype | Per Device (B=1) |
|-------|-----------------|
| BF16 | 100.0 MB |
| BFP8 | 50.0 MB |

### Global KV Cache (10 Layers)

Global layers store the full sequence with all 4 KV heads replicated on every
device.

```math
\text{Global KV per device} = 10 \times 2 \times 4_{\text{heads}} \times S \times 512 \times \text{bpe}
```

| Seq Length (S) | BF16 (per device) | BFP8 (per device) |
|----------------|-------------------|-------------------|
| 2,048 | 160 MB | 80 MB |
| 4,096 | 320 MB | 160 MB |
| 8,192 | 640 MB | 320 MB |
| 16,384 | 1,280 MB | 640 MB |
| 32,768 | 2,560 MB | 1,280 MB |
| 65,536 | 5,120 MB | 2,560 MB |
| 131,072 | 10,240 MB | 5,120 MB |

### Combined KV Cache

| Seq Length | Sliding (BF16) | Global (BF16) | **Total (BF16)** | **Total (BFP8)** |
|------------|---------------|---------------|-----------------|-----------------|
| 2,048 | 100 MB | 160 MB | 260 MB | 130 MB |
| 8,192 | 100 MB | 640 MB | 740 MB | 370 MB |
| 32,768 | 100 MB | 2,560 MB | 2,660 MB | 1,330 MB |
| 65,536 | 100 MB | 5,120 MB | 5,220 MB | 2,610 MB |
| 131,072 | 100 MB | 10,240 MB | 10,340 MB | 5,170 MB |

The global KV cache dominates at all practical sequence lengths. The sliding
KV cache, despite spanning 50 layers, is negligible due to the 1,024-token
window.

## Activation Memory

### Per-Layer Peak Activations (Decode, B=1, S=1)

During decode, activation memory is minimal because only a single token is
processed. The peak occurs during the FFN when two intermediate tensors are
held simultaneously:

| Activation | Shape | BF16 Size |
|-----------|-------|-----------|
| FFN gate output (activated) | [1, 1, 2688] | ~5.3 KB |
| FFN up output | [1, 1, 2688] | ~5.3 KB |
| Attention Q (after projection) | [1, 1, 1024] (sliding) / [1, 1, 2048] (global) | ~2--4 KB |
| Attention KV (after projection) | [1, 1, 512--2048] | ~1--4 KB |
| Hidden states (residual) | [1, 1, 5376] | ~10.5 KB |
| **Peak per layer** | | **~30 KB** |

Note: These shapes are per-device after TP=8 sharding.

Total activation memory across 60 layers at decode: negligible (< 2 MB).
Activations are reused between layers, so only one layer's activations are
live at a time.

### Prefill Activation Memory

Prefill processes a full chunk of tokens simultaneously. At chunk size C:

| Component | Shape per Device | BF16 Size (C=2048) |
|-----------|-----------------|-------------------|
| FFN gate + up (simultaneous) | 2 x [1, C, 2688] | ~22 MB |
| Attention Q | [1, C, 1024--2048] | ~4--8 MB |
| Hidden states | [1, C, 5376] | ~22 MB |
| **Peak per layer** | | **~52 MB** |

Prefill activation memory is bounded by the chunk size, not the total sequence
length. A chunk size of 2048 keeps activation memory well under 100 MB per
device.

## Total DRAM Budget Per Device

### BFP8 Weights + BF16 KV Cache (Recommended Configuration)

| Seq Length | Weights | KV Cache | Activations | **Total** | **Headroom** |
|------------|---------|----------|-------------|-----------|-------------|
| 2,048 | 4,143 MB | 260 MB | ~2 MB | 4,405 MB | 7,883 MB |
| 4,096 | 4,143 MB | 420 MB | ~2 MB | 4,565 MB | 7,723 MB |
| 8,192 | 4,143 MB | 740 MB | ~2 MB | 4,885 MB | 7,403 MB |
| 16,384 | 4,143 MB | 1,380 MB | ~2 MB | 5,525 MB | 6,763 MB |
| 32,768 | 4,143 MB | 2,660 MB | ~2 MB | 6,805 MB | 5,483 MB |
| 65,536 | 4,143 MB | 5,220 MB | ~2 MB | 9,365 MB | 2,923 MB |
| 131,072 | 4,143 MB | 10,340 MB | ~2 MB | **14,485 MB** | **EXCEEDS** |

### BFP8 Weights + BFP8 KV Cache (Extended Context)

| Seq Length | Weights | KV Cache | **Total** | **Headroom** |
|------------|---------|----------|-----------|-------------|
| 65,536 | 4,143 MB | 2,610 MB | 6,755 MB | 5,533 MB |
| 131,072 | 4,143 MB | 5,170 MB | 9,315 MB | 2,973 MB |
| 262,144 | 4,143 MB | 10,290 MB | **14,435 MB** | **EXCEEDS** |

### Maximum Context Length Summary

| Configuration | Max Context (B=1) | Headroom at Max |
|---------------|-------------------|-----------------|
| BF16 weights + BF16 KV | ~8K | ~3.6 GB |
| BFP8 weights + BF16 KV | ~65K | ~2.9 GB |
| BFP8 weights + BFP8 KV | ~131K | ~2.9 GB |
| BFP4 weights + BFP8 KV | ~131K+ | ~4.6 GB |

The practical sweet spot for T3K is **BFP8 weights with BF16 KV cache at
8K--32K context**, which provides 5--7 GB of headroom per device for the Metal
Trace buffer, page tables, temporary buffers, and runtime overhead.

For contexts beyond 65K, BFP8 KV cache is mandatory. Full 256K context is
infeasible with replicated global KV heads even at BFP8 KV --- it would
require either BFP4 KV cache, a non-replicated KV strategy (accepting CCL
overhead), or host-side KV offloading.

---

**Next:** [`decode_latency_analysis.md`](./decode_latency_analysis.md)
