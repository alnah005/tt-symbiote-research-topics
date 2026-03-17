# Prefill Compute Throughput

## Regime Definition

Prefill processes all prompt tokens in a single forward pass. At sequence lengths of
2048 and above, the expert FFN matmuls become compute-bound on Wormhole B0. The
arithmetic intensity for a matmul of shape `[S, d_model] × [d_model, d_ff]` is:

```
AI = (2 × S × d_model × d_ff) / (bytes_weights + bytes_activations)
   ≈ 2 × S  [FLOP/byte]  when S >> 1 and weight bytes dominate
```

For S=2048, d_model=4096, d_ff=2048 (Qwen MoE expert): AI ≈ 2×2048 = 4096 FLOP/byte,
which far exceeds Wormhole's ridge point of ~437 FLOP/byte. The matmul is firmly
compute-bound.

## Dtype Throughput Ratios

Wormhole B0 compute ceilings by dtype:

| Dtype | Bits/element | Compute ceiling | Ratio vs BF16 |
|-------|-------------|-----------------|---------------|
| bfloat16 | 16 | 131 TFLOP/s | 1× |
| bfloat8_b | 8 | ~262 TFLOP/s (effective) | ~2× |
| bfloat4_b | 4 | ~524 TFLOP/s (effective) | ~4× |

"Effective" compute ceiling accounts for hardware unpack throughput. bfloat8_b and
bfloat4_b dequantize to bfloat16 in hardware (no software loop), so the FPU operates
at native BF16 rates on a higher density of tiles per unit time.

## MathFidelity Overhead

For MathFidelity level definitions and throughput overhead, see Chapter 1 and `tile_compute_efficiency.md` in this chapter.

HiFi2 with `fp32_dest_acc_en=True` adds approximately 20–30% latency relative to LoFi
for the same tile count. This cost is paid in compute cycles, not memory bandwidth.

For prefill (compute-bound), MathFidelity selection directly affects end-to-end FFN
latency. Using LoFi maximizes throughput; HiFi2 is appropriate when output PCC
requirements demand tighter accumulation fidelity.

## Expert FFN Dimensions: Qwen MoE

Qwen MoE expert weights (single expert, per-projection):

```
Gate projection:  [d_model=4096, d_ff=2048]   → 8,388,608 elements
Up   projection:  [d_model=4096, d_ff=2048]   → 8,388,608 elements
Down projection:  [d_ff=2048,   d_model=4096] → 8,388,608 elements
```

For per-projection memory footprints by dtype, see Chapter 1 format files.

bfloat4_b doubles effective tile density: a 32×32 tile holds the same 1,024 elements
but in 512 bytes instead of 2,048, so twice as many tiles fit per L1 SRAM cycle and
four times the arithmetic work is extracted per DRAM byte transferred.

## Practical Bottleneck Note

In a full MoE forward pass, all-to-all dispatch and combine operations route tokens
between tensor-parallel ranks before and after expert computation. These collective
operations are latency-bound by inter-chip interconnect, not by expert FFN compute.

**Consequence:** quantization gains from lower dtype are most visible when the expert
FFN itself is the measured bottleneck. Profile with dispatch/combine isolated before
attributing prefill latency entirely to FFN dtype. In microbenchmarks of the FFN in
isolation, bfloat4_b + LoFi achieves close to the theoretical 4× throughput improvement
over bfloat16 + HiFi2.

## Summary

- Prefill at S≥2048 is compute-bound for Qwen MoE expert FFNs.
- bfloat8_b yields ~2× throughput; bfloat4_b yields ~4× when fully compute-bound.
- MathFidelity overhead (HiFi2 vs LoFi) adds 20–30% latency; choose based on PCC
  requirements.
- Always measure with all-to-all overhead isolated to observe true FFN speedup.
