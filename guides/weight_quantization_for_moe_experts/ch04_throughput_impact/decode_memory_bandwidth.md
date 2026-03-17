# Decode Memory Bandwidth

## Regime Definition

During decode, a single new token is generated per forward pass (batch=1–32, seq=1).
The expert FFN matmuls degenerate to matrix-vector products:

```
[batch, d_model] × [d_model, d_ff]  →  [batch, d_ff]
```

FLOP count: `2 × batch × d_model × d_ff`
Weight bytes read: `d_model × d_ff × bytes_per_element`

For batch=1, Qwen MoE gate projection (d_model=4096, d_ff=2048, BF16):

```
AI = (2 × 1 × 4096 × 2048) / (4096 × 2048 × 2)
   = 1 FLOP/byte
```

This is far below Wormhole's ridge point of ~437 FLOP/byte. The operation is
memory-bound: compute units are starved waiting for weight data from DRAM.

## DRAM Read Volume Reduction

Each weight element is read once per forward pass. Quantization reduces bytes read
proportionally to bit-width reduction:

| Dtype | Bits/element | DRAM read (gate proj) | Reduction vs BF16 |
|-------|--------------|-----------------------|-------------------|
| bfloat16 | 16 | 16,777,216 bytes | 1× |
| bfloat8_b | 8 | 8,388,608 bytes | 2× less |
| bfloat4_b | 4 | 4,194,304 bytes | 4× less |

For memory-bound matmuls, DRAM read volume directly determines latency:

```
Latency ≈ DRAM_bytes / DRAM_bandwidth
        = 16,777,216 / 300e9  ≈ 55.9 µs  (BF16, gate proj)
        = 8,388,608  / 300e9  ≈ 27.9 µs  (bfloat8_b)
        = 4,194,304  / 300e9  ≈ 14.0 µs  (bfloat4_b)
```

These are lower bounds assuming full DRAM bandwidth utilization across all 6 DRAM
controllers and 12 GDDR6 banks.

## L1 vs DRAM Placement

Wormhole B0 Tensix local SRAM (L1) is small relative to expert weight tensors. For
decode:

- **Activation tensors** `[batch, d_model]`: small enough to reside in L1 across
  the token's routing path. Reused for gate and up projections.
- **Weight tensors** `[d_model, d_ff]`: too large for L1 at BF16; streamed from
  DRAM every forward pass.

Quantized weights load in fewer cache lines per tile:

| Dtype | Cache lines per 32×32 tile (64-byte lines) |
|-------|--------------------------------------------|
| bfloat16 | 32 cache lines (2,048 bytes) |
| bfloat8_b | 16 cache lines (1,024 bytes) |
| bfloat4_b | 8 cache lines (512 bytes) |

Fewer cache line fetches reduce DRAM controller pressure and improve effective
bandwidth utilization. With bfloat4_b, a Tensix core can sustain higher tile
throughput before stalling on DRAM latency.

## Arithmetic Intensity Crossover Formula

The matmul transitions from memory-bound to compute-bound when AI exceeds the ridge
point. The crossover batch size for each dtype:

```
AI(batch) = (2 × batch × d_model × d_ff) / (d_model × d_ff × bytes_per_element)
           = (2 × batch) / bytes_per_element

Crossover when AI = ridge_point = 437 FLOP/byte:
  batch_crossover = (437 × bytes_per_element) / 2
```

| Dtype | bytes/element | batch_crossover |
|-------|--------------|-----------------|
| bfloat16 | 2 | ~437 |
| bfloat8_b | 1 | ~219 |
| bfloat4_b | 0.5 | ~109 |

At batch=32 (common decode scenario), all dtypes remain memory-bound. This confirms
that bandwidth reduction is the primary mechanism for decode speedup — compute
throughput improvements are secondary.

## Bandwidth Utilization in Practice

Real-world DRAM bandwidth is below peak due to:
- Non-contiguous access patterns from tensor sharding
- DRAM controller scheduling overhead
- Burst length mismatches

Effective bandwidth is typically 60–80% of peak (~180–240 GB/s). Bandwidth reduction
from quantization is still proportional, so decode latency ratios closely follow the
2× and 4× theoretical predictions.

## Summary

- Decode expert FFN is memory-bound for all practical batch sizes (batch ≤ 109 for
  bfloat4_b; ≤ 219 for bfloat8_b; ≤ 437 for bfloat16).
- bfloat8_b halves DRAM read volume; bfloat4_b quarters it.
- Bandwidth reduction directly translates to proportional latency reduction.
- Activation tensors stay in L1; weight tensors stream from DRAM each pass.
