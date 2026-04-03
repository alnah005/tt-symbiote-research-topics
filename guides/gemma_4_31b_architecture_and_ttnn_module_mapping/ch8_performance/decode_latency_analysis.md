# Decode Latency Analysis

This file provides a per-layer breakdown of expected decode latency for
Gemma 4 31B on T3K (TP=8), compares sliding and global layer costs, quantifies
CCL overhead, and estimates the total decode latency per token.

All estimates assume batch=1, single-token decode, BFP8 weights, and BF16 KV
cache unless otherwise stated. Latency numbers are approximate projections
based on Wormhole DRAM bandwidth (~300 GB/s per chip) and compute throughput
characteristics. Actual numbers will depend on kernel tuning, program configs,
and DRAM-sharded optimization.

## Per-Layer Latency Components

### Linear Projection Latency (Memory-Bound)

At batch=1 decode, every linear projection is a matrix-vector multiply. The
latency is dominated by reading the weight matrix from DRAM, not by compute.
The formula for memory-bound latency:

```math
t_{\text{matmul}} \approx \frac{\text{weight bytes per device}}{\text{DRAM bandwidth per chip}}
```

With ~300 GB/s DRAM bandwidth per Wormhole chip:

#### Sliding Layer Projections (BFP8 Weights)

| Projection | Per-Device Weight (BFP8) | Estimated Latency |
|------------|-------------------------|-------------------|
| Q | 5.5 MB | ~18 us |
| K | 2.8 MB | ~9 us |
| V | 2.8 MB | ~9 us |
| O | 5.5 MB | ~18 us |
| Gate | 14.5 MB | ~48 us |
| Up | 14.5 MB | ~48 us |
| Down | 14.5 MB | ~48 us |
| **Total projections** | **60.1 MB** | **~198 us** |

With fused QKV (single `[5376, 2048]` matmul after TP=8 sharding,
combining Q+K+V into `[5376, 2048]`), the three separate reads become one,
saving kernel launch overhead (~5--10 us per avoided launch). Similarly, fused
gate+up reduces two matmuls to one (~29 MB combined weight read at ~96 us
bandwidth time, but saving one kernel launch of ~5 us). Estimated savings
from all fusions: ~15--20 us.

#### Global Layer Projections (BFP8 Weights)

| Projection | Per-Device Weight (BFP8) | Estimated Latency |
|------------|-------------------------|-------------------|
| Q | 11.0 MB | ~37 us |
| K (replicated) | 11.0 MB | ~37 us |
| O | 11.0 MB | ~37 us |
| Gate | 14.5 MB | ~48 us |
| Up | 14.5 MB | ~48 us |
| Down | 14.5 MB | ~48 us |
| **Total projections** | **76.5 MB** | **~255 us** |

Global layers are ~29% more expensive for projections due to the larger Q/K/O
weights (head_dim=512 vs 256) and the replicated K projection (full weight on
each device).

### RoPE Latency

RoPE is an element-wise operation applied to Q and K after projection.

| Layer Type | Operation | Estimated Latency |
|------------|-----------|-------------------|
| Sliding | Full RoPE on Q `[1, 4, 1, 256]` + K `[1, 2, 1, 256]` | ~2 us |
| Global | Partial RoPE (128/512 dims) on Q `[1, 4, 1, 512]` + K `[1, 4, 1, 512]` | ~3 us |

Global partial RoPE is slightly more expensive due to the split-apply-concat
pattern (split first 128 dims, apply RoPE, concatenate with passthrough dims).
However, the tensors are small at decode, so RoPE is negligible.

### Norm Latency

Each decoder layer has 3 RMSNorm operations: pre-attention, post-attention
(pre-FFN), and post-FFN. Additionally, V-norm is applied to value vectors, and
K-norm to key vectors.

| Norm | Shape (per device) | Estimated Latency |
|------|-------------------|-------------------|
| Pre-attention RMSNorm | [1, 1, 5376] | ~1 us |
| K-norm | [1, kv_heads, 1, head_dim] | < 1 us |
| V-norm | [1, kv_heads, 1, head_dim] | < 1 us |
| Post-attention RMSNorm | [1, 1, 5376] | ~1 us |
| Post-FFN RMSNorm | [1, 1, 5376] | ~1 us |
| **Total norms per layer** | | **~4 us** |

### SDPA Latency

Scaled dot-product attention latency depends heavily on the KV cache length.

#### Sliding Layers (Window = 1,024 Tokens)

The sliding window bounds the KV cache to 1,024 tokens regardless of the total
sequence length. At decode, each device computes attention over 4 Q heads and
2 KV heads with head_dim=256.

```text
Per-device KV cache read: 2 tensors x 2 heads x 1024 tokens x 256 dims x 2 bytes = 2.0 MB
```

Estimated SDPA latency: **~8--12 us** (bounded, sequence-length-independent).

#### Global Layers (Full Causal, Grows with S)

Global layers attend over the full sequence. Each device computes attention over
4 Q heads and 4 replicated KV heads with head_dim=512.

```text
Per-device KV cache read: 2 tensors x 4 heads x S tokens x 512 dims x 2 bytes
```

| Seq Length (S) | KV Cache Read per Device | Estimated SDPA Latency |
|----------------|-------------------------|----------------------|
| 2,048 | 16.0 MB | ~53 us |
| 4,096 | 32.0 MB | ~107 us |
| 8,192 | 64.0 MB | ~213 us |
| 16,384 | 128.0 MB | ~427 us |
| 32,768 | 256.0 MB | ~853 us |

Global SDPA grows linearly with sequence length and becomes a dominant cost at
long contexts. At S=32,768, a single global SDPA call (~853 us) is more
expensive than all projections in the same layer (~255 us).

### PLE Latency

In the 31B config, PLE (`hidden_size_per_layer_input=0`) is disabled. The PLE
injection is a no-op, contributing zero latency.

If PLE were active, it would add a small embedding lookup and linear projection
per layer (~5--10 us), but this is not relevant for the current config.

### CCL Latency (All-Reduce)

Each layer requires 2 `ttnn.all_reduce` operations (after O projection and
after down projection). The payload at B=1 is:

```text
Per all-reduce: 1 x 1 x 5376 x 2 bytes = 10,752 bytes (~10.5 KB)
```

This tiny payload means the all-reduce is entirely latency-bound, not
bandwidth-bound. The per-call latency on T3K with `ttnn.Topology.Linear`
is estimated at **5--10 us**.

Per layer: 2 x 5--10 us = **10--20 us**.
Per decode step (60 layers): 120 x 5--10 us = **600--1,200 us** (0.6--1.2 ms).

## Per-Layer Latency Summary

### Sliding Layer (BFP8 Weights, B=1, S-Independent)

| Component | Estimated Latency |
|-----------|-------------------|
| Q+K+V projections (fused) | ~31 us |
| K-norm + V-norm + RoPE | ~3 us |
| SDPA (window=1024) | ~10 us |
| O projection | ~18 us |
| Pre-attention norm | ~1 us |
| Post-attention norm | ~1 us |
| Gate+Up projection (fused) | ~96 us |
| GELU + elementwise multiply | ~2 us |
| Down projection | ~48 us |
| Post-FFN norm | ~1 us |
| CCL (2 all-reduce) | ~15 us |
| **Total per sliding layer** | **~226 us** |

### Global Layer (BFP8 Weights, B=1)

| Component | S=2,048 | S=8,192 | S=32,768 |
|-----------|---------|---------|----------|
| Q projection | ~37 us | ~37 us | ~37 us |
| K projection (replicated) | ~37 us | ~37 us | ~37 us |
| K-norm + V-norm + partial RoPE | ~4 us | ~4 us | ~4 us |
| SDPA (full causal) | ~53 us | ~213 us | ~853 us |
| O projection | ~37 us | ~37 us | ~37 us |
| Pre-attention norm | ~1 us | ~1 us | ~1 us |
| Post-attention norm | ~1 us | ~1 us | ~1 us |
| Gate+Up projection (fused) | ~96 us | ~96 us | ~96 us |
| GELU + elementwise multiply | ~2 us | ~2 us | ~2 us |
| Down projection | ~48 us | ~48 us | ~48 us |
| Post-FFN norm | ~1 us | ~1 us | ~1 us |
| CCL (2 all-reduce) | ~15 us | ~15 us | ~15 us |
| **Total per global layer** | **~332 us** | **~492 us** | **~1,132 us** |

## Total Decode Latency (60 Layers)

```math
T_{\text{decode}} = 50 \times T_{\text{sliding}} + 10 \times T_{\text{global}}
```

| Seq Length | Sliding (50 layers) | Global (10 layers) | **Total** | **Tokens/s** |
|------------|--------------------|--------------------|-----------|-------------|
| 2,048 | 11.3 ms | 3.3 ms | **14.6 ms** | ~68 |
| 4,096 | 11.3 ms | 3.9 ms | **15.2 ms** | ~66 |
| 8,192 | 11.3 ms | 4.9 ms | **16.2 ms** | ~62 |
| 16,384 | 11.3 ms | 7.1 ms | **18.4 ms** | ~54 |
| 32,768 | 11.3 ms | 11.3 ms | **22.6 ms** | ~44 |

### Key Observations

1. **Sliding layers dominate at short contexts.** At S=2,048, the 50 sliding
   layers contribute 77% of the total latency. Sliding layer latency is
   constant regardless of sequence length.

2. **Global layers dominate at long contexts.** At S=32,768, the 10 global
   layers contribute 50% of the total latency, despite being only 1/6 of the
   layer count. This is entirely due to SDPA growing with sequence length.

3. **The crossover point is around S=32K.** Below 32K tokens, optimizing
   sliding layers (projections, CCL) has more impact. Above 32K, optimizing
   global SDPA is the priority.

4. **CCL overhead is significant.** At 0.6--1.2 ms across 120 calls, CCL
   accounts for 5--10% of total decode latency. Multi-CQ overlap can reduce
   the effective CCL cost.

## Comparison with Similar Models

| Model | Parameters | Architecture | T3K Decode (est.) | Notes |
|-------|------------|-------------|-------------------|-------|
| LLaMA 3 70B | 70B | 80 layers, GQA 8:1 | ~20--25 ms | 2x params, TP=8 |
| Qwen2.5 32B | 32B | 64 layers, GQA 8:1 | ~12--15 ms | Homogeneous attention |
| **Gemma 4 31B** | **30.7B** | **60 layers, heterogeneous** | **~16 ms (S=8K)** | Sliding + global |
| Mistral 7B | 7B | 32 layers, sliding window | ~4--5 ms | Much smaller model |

Gemma 4 31B should achieve competitive decode performance with similarly-sized
models. The sliding window design for 50/60 layers is a significant advantage
--- it bounds SDPA cost for the majority of layers and caps sliding KV cache
memory at a fixed 100 MB. The 10 global layers add context-dependent overhead
but keep the total latency reasonable at moderate sequence lengths.

---

**Next:** [`optimization_roadmap.md`](./optimization_roadmap.md)
