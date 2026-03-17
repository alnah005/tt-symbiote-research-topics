# Tile Compute Efficiency

## Fixed Tile Geometry

Wormhole B0 operates on 32×32 element tiles regardless of dtype. The logical tile shape
is invariant; only the byte footprint changes with bit-width:

| Dtype | Bits/element | Bytes per 32×32 tile | Tiles per 1 MB L1 |
|-------|-------------|----------------------|-------------------|
| bfloat16 | 16 | 2,048 | 512 |
| bfloat8_b | 8 | 1,024 | 1,024 |
| bfloat4_b | 4 | 512 | 2,048 |

The FPU always receives and emits bfloat16 values: the hardware unpack stage converts
bfloat8_b or bfloat4_b tiles to bfloat16 before arithmetic, and the pack stage
converts outputs back. Effective compute per byte of DRAM traffic is therefore 2× or
4× higher for denser dtypes — no software dequantization loop needed.

## Wormhole Math Engine: Block-FP Hardware Unpack

The Wormhole B0 FPU/SFPU implements block-floating-point unpack in silicon:

1. Load compressed tile from DRAM or L1 (512 or 1,024 bytes).
2. Hardware reads shared block exponent from tile header.
3. Each mantissa field is sign-extended and shifted by block exponent in one cycle.
4. Output: 1,024 bfloat16 values ready for FPU accumulation.

This pipeline adds negligible latency compared to a software dequantization loop
(which would require ~1,024 individual SFPU ops per tile). The unpack throughput is
matched to the FPU's tile consumption rate, so the math engine is not stalled by
format conversion.

## MathFidelity and Throughput

MathFidelity selects the number of FPU accumulation passes per output element. The
projection-level use cases within Chapter 4 are:

- **LoFi** — gate/up projections; decode throughput (single-pass, fastest)
- **HiFi2** — down projection; residual stream (adds ~20–30% latency vs LoFi)
- **HiFi4** — dense MLP; highest numerical fidelity

Pass counts and relative throughput figures are defined in Chapter 2,
`compute_kernel_config.md` § math_fidelity.

Pass count is a compile-time selection encoded in the kernel configuration; it cannot
be changed per-tile at runtime.

## Grid Utilization

Wormhole B0 has 80 Tensix cores arranged in an 8×10 grid. Each Tensix core holds its
own L1 SRAM. Expert weight tiles are distributed across cores by the tensor parallel
sharding scheme.

With quantized weights, more expert weight tiles fit in L1 simultaneously:

```
L1 per Tensix core ≈ 1.5 MB (approximate; architecture-dependent)

Tiles per core in L1 (gate projection shard, 1/80 of total):
  BF16:      8,192 elements × 2 bytes = 16,384 bytes → 8 tiles
  bfloat8_b: 8,192 elements × 1 byte  =  8,192 bytes → 8 tiles (half bandwidth)
  bfloat4_b: 8,192 elements × 0.5 byte = 4,096 bytes → 8 tiles (quarter bandwidth)
```

Even though each shard is small, quantization allows the activation tiles and weight
tiles to coexist in L1 with lower eviction pressure, enabling better pipelining of
DRAM prefetch against compute.

## Why bfloat4_b + LoFi for Gate/Up Projections

Gate and up projections feed into the SwiGLU activation:

```
FFN(x) = (gate(x) ⊙ σ(up(x))) · W_down
```

The SwiGLU nonlinearity absorbs gate/up quantization errors before they reach the residual stream. See Chapter 3, `projection_sensitivity.md` for the full mechanistic analysis.

bfloat4_b + LoFi is the preferred configuration for gate/up because:
- 4× tile density relative to BF16 → 4× fewer DRAM reads in decode
- LoFi single pass → highest FPU throughput in prefill
- SwiGLU path acts as a natural noise filter for quantization error

Down projection writes directly to the residual stream. Accumulated errors are not
filtered before the next layer's attention, making bfloat8_b + HiFi2 the preferred
choice, delivering PCC ~0.97–0.98.

## Summary

- 32×32 tile is fixed; bfloat8_b is 1,024 bytes; bfloat4_b is 512 bytes.
- Hardware unpack converts to BF16 in the FPU pipeline — no software overhead.
- LoFi is optimal for gate/up; HiFi2 for down projection/residual stream; HiFi4 for dense MLP. See Chapter 2, `compute_kernel_config.md` for pass counts and throughput figures.
- 80-core grid benefits from lower per-tile byte footprint via reduced L1 pressure.
- bfloat4_b + LoFi is optimal for gate/up; bfloat8_b + HiFi2 for down projection.

---

**Next:** [`bandwidth_vs_accuracy_tradeoff.md`](./bandwidth_vs_accuracy_tradeoff.md)
