# Chapter 4: Throughput and Memory Bandwidth Impact

## Overview

This chapter quantifies the throughput and memory bandwidth consequences of weight
quantization for MoE expert layers on Wormhole B0 hardware. Chapters 1–3 established
what block-floating-point formats are and how they are stored; this chapter covers what
happens at runtime when the math engine operates on them.

## Prerequisites

- Chapter 1: Block-floating-point format encoding (bfloat8_b, bfloat4_b)
- Chapter 2: Wormhole B0 memory hierarchy and Tensix architecture
- Chapter 3: MoE expert weight layout and tensor parallel sharding

## Learning Objectives

By the end of this chapter you should be able to:

1. Explain why prefill and decode sit on opposite sides of the arithmetic intensity
   roofline and why quantization affects them differently.
2. Compute the DRAM read volume reduction from bfloat16 → bfloat8_b → bfloat4_b for
   a given expert FFN weight tensor.
3. State the MathFidelity pass counts (LoFi / HiFi2 / HiFi4) and their throughput cost.
4. Identify which projection types (gate, up, down) benefit most from bfloat4_b and
   explain why down projection warrants more conservative quantization.
5. Locate each quantized format on the roofline and read off the expected speedup
   relative to bfloat16.

## Roofline Sketch — Wormhole B0

```
Throughput
(TFLOP/s)
  131 |......................._________ compute ceiling (BF16)
      |                   /
      |                  /   ← ridge point ~437 FLOP/byte
      |                 /
      |                /
    0 |_______________/_________________________ Arithmetic Intensity
      0              437                       (FLOP/byte)

  Memory wall: 300 GB/s DRAM bandwidth
  Compute ceiling: 131 TFLOP/s BF16

  Decode expert FFN (batch 1–32): far left of ridge → memory-bound
  Prefill expert FFN (seq 2048+): approaches or crosses ridge → compute-bound
```

Quantization moves the effective FLOP/byte ratio upward (more ops per byte of weights
read), shifting memory-bound workloads rightward toward the ridge.

## File Map

| File | Contents |
|------|----------|
| `prefill_compute_throughput.md` | Prefill regime: compute-bound analysis, dtype throughput ratios |
| `decode_memory_bandwidth.md` | Decode regime: memory-bound analysis, bandwidth reduction formulas |
| `tile_compute_efficiency.md` | Tile-level efficiency: byte counts, MathFidelity passes, grid utilization |
| `bandwidth_vs_accuracy_tradeoff.md` | Joint Pareto analysis: accuracy vs bandwidth for all projection types |

## Key Hardware Constants (Reference)

| Parameter | Value |
|-----------|-------|
| DRAM bandwidth | ~300 GB/s |
| BF16 compute | 131 TFLOP/s |
| Ridge point | ~437 FLOP/byte |
| Tensix cores | 80 (8×10 grid) |
| Tile size (all dtypes) | 32×32 elements |

MathFidelity pass counts (LoFi=1, HiFi2=2, HiFi4=4) are defined in Chapter 2, `compute_kernel_config.md` § math_fidelity.

Tile memory footprints: BF16 = 2,048 bytes, bfloat8_b = 1,024 bytes, bfloat4_b = 512 bytes per 32×32 tile. See Chapter 1 for derivation.
