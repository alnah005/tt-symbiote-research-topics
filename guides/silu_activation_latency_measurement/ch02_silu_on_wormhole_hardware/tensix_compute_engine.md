# Tensix Compute Engine Architecture

Each Wormhole B0 chip contains a grid of Tensix cores. Understanding what is inside a single
Tensix core is the foundation for reasoning about latency. Every matmul and every activation op
you launch ultimately executes on one of these cores.

---

## The Three Execution Pipelines

A Tensix core contains three distinct execution pipelines that run as a coordinated but separate
set of engines:

| Pipeline | Role | Granularity |
|---|---|---|
| RISC-V (unpacker/packer) | Control: moves data between DRAM/L1 and the compute engines | byte / tensor level |
| FPU (matrix engine) | Tile-level matrix multiply-accumulate (FMA) | 32×32 tile |
| SFPU (vector engine) | Element-wise scalar transformations | 32 elements (one LReg lane) |

These pipelines are **not interchangeable**. Matmuls run on the FPU. Element-wise activations
like SiLU run on the SFPU. The RISC-V cores orchestrate data movement but do not perform float
arithmetic.

---

## FPU: The Matrix Engine

The FPU (Floating Point Unit, sometimes called the math engine or matrix engine) is designed
for dense tile multiply-accumulate operations:

- Input: two 32×32 BF16 tiles (A and B)
- Output: one 32×32 BF16 (or FP32) accumulated tile in the destination register
- Peak throughput: one tile-FMA per cycle in the ideal fully-pipelined case
- All TTNN matmul ops — `ttnn.linear`, `ttnn.matmul` — compile down to sequences of FPU
  tile-FMAs

The FPU is the reason Wormhole achieves high arithmetic intensity on large matmuls. Because it
processes 32×32 = 1024 multiply-adds in a single tile operation, it amortizes the cost of
loading A and B over many FLOPs.

### Destination Register

After each tile-FMA, the result lands in the FPU destination register. This register is shared
between the FPU and the SFPU, and this sharing is the root cause of the sequential constraint
described below.

---

## SFPU: The Vector Engine

The SFPU (Special Function Processing Unit) is a 32-wide SIMD vector unit. Its job is to apply
nonlinear scalar functions element-wise across a tile:

- **Register file (LReg):** 32 elements × 32 bits each. This is the SFPU's working memory.
- **Input:** elements are loaded from the FPU destination register into the LReg, 32 at a time.
- **Operation:** applies a sequence of LLK (Low-Level Kernel) instructions to the 32 loaded
  elements.
- **Output:** results are written back to the destination register before packing.

A BF16 tile's 1024 elements require 32 SFPU passes; see `silu_sfpu_execution.md` for the full derivation in the context of SiLU.

---

## The Sequential Pipeline Constraint

**FPU and SFPU cannot operate simultaneously on the same output tile.**

The sequence for a matmul-then-activation pattern (e.g., gate_proj followed by SiLU) is:

```
[FPU] tile-FMA for output tile T
      → T lands in FPU destination register
      → [SFPU] 32 passes over T (SiLU instruction sequence)
              → T written back to destination register
              → [Packer] T packed and written to L1
```

Within a single tile, the FPU matmul pass and the SFPU activation pass are sequential — the SFPU cannot begin until the FPU has written the tile to the destination register. The destination register is the bottleneck: the FPU needs it to write matmul output; the SFPU needs it to read activation input. They share the same physical register, so for any given tile they must take turns.

This is why `ttnn.silu` appears as a sequential latency contribution after `ttnn.linear` in
op-level profiler traces — it is not a scheduling artifact, it reflects real hardware.

---

## Data Type Considerations

| Data Type | Notes |
|---|---|
| BF16 | Standard path. FPU operates natively on BF16 tiles; SFPU reads from FPU dest register. |
| BFP8 | Blocked floating point; requires unpacking to a wider format before SFPU processing. Precision path differs from BF16. |
| FP32 | SFPU operates at 32 bits per lane; BF16 inputs are effectively upcast internally during SFPU processing. |

For Qwen3 MoE and similar models running in BF16, the BF16 matmul output flows directly from
the FPU destination register into the SFPU — there is no precision conversion overhead beyond
what the LLK sequence itself performs.

---

## Why This Matters for SiLU in MoE

For the quantitative matmul vs. SiLU cost comparison — including roofline placement and the estimated SiLU fraction of FFN time — see `cycles_vs_matmul.md`.

---

**Next Steps:** Read `silu_sfpu_execution.md` to see exactly which SFPU LLK instructions
implement SiLU and why the polynomial approximation for sigmoid is the dominant cost.
