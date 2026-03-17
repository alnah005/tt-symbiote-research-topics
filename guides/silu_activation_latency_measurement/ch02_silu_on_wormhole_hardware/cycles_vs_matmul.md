# SiLU SFPU Cost versus Matmul FPU Cost

This file builds a quantitative picture of how SiLU's SFPU cost relates to the FPU cost of
the matmul that precedes it. The goal is not exact cycle counts — Chapter 3 provides
measurement methodology for that — but rather calibrated intuition for where SiLU falls on
a roofline and what fraction of FFN time it plausibly consumes.

---

## FPU Throughput for a Matmul

In the ideal pipelined case, the Wormhole FPU produces one 32×32 BF16 output tile per cycle.
For a matmul with dimensions M × K × N (in tiles: M_t × K_t × N_t):

```
FPU tile-FMAs = M_t × K_t × N_t
```

The inner dimension K_t (number of tiles along the reduction axis) is the critical factor:

- **Deeper K_t → more FPU cycles** for the same M_t × N_t output, but also more reuse of
  the loaded A and B tiles, which increases arithmetic intensity.
- **Shallow K_t** (e.g., small sequence lengths) → the matmul is short, potentially
  memory-bound, and the FPU is not saturated.

For a decode-phase matmul with 32 tokens and hidden_dim=4096:

```
M_t = 32/32 = 1 tile (tokens)
K_t = 4096/32 = 128 tiles (hidden_dim)
N_t = 4096*8/3 / 32 ≈ 341 tiles (gate_proj output, ~10922 columns)

FPU tile-FMAs = 1 × 128 × 341 = 43,648 tile-FMAs
```

This is the gate_proj matmul. At one tile-FMA per cycle per core, this is 43,648 FPU cycles
if the computation is not distributed across cores (it is, in practice — see Chapter 4 for
multi-core sharding discussion).

---

## SFPU Budget for SiLU

For each output tile from gate_proj, SiLU requires:

As derived in `silu_sfpu_execution.md`, a SiLU tile requires 160–256 SFPU instructions.

The gate_proj output has M_t × N_t = 1 × 341 = 341 tiles (for 32 tokens, ~10922 columns).
Total SFPU instruction budget for SiLU over the full gate_proj output:

```
341 tiles × 160–256 instructions/tile = ~54,560–87,296 SFPU instructions
```

The SFPU does not run at the same throughput as the FPU. The FPU is pipelined for tile-FMAs;
the SFPU executes a serial instruction sequence per pass. The relative throughput ratio is
workload-dependent, but as a rough model: the SFPU adds latency roughly proportional to
instruction depth × number of passes × number of tiles, and this latency is not hidden by
the FPU (because they share the destination register).

---

## Arithmetic Intensity Comparison

The roofline model characterizes whether an operation is compute-bound or memory-bound by
comparing its arithmetic intensity (FLOPs per byte of memory traffic) against the machine's
ridge point.

### Wormhole B0 Reference Parameters

| Resource | Value |
|---|---|
| Peak FPU throughput (BF16) | ~131 TFLOP/s aggregate |
| DRAM bandwidth (aggregate) | ~300 GB/s |
| L1 bandwidth (est. aggregate) | ~900+ GB/s |
| Ridge point (DRAM) | ~131e12 / 300e9 ≈ **437 FLOP/byte** |

The ridge point is the arithmetic intensity at which an op exactly saturates FPU while
fully utilizing DRAM bandwidth. Below the ridge point, the op is memory-bound (DRAM is the
bottleneck). Above it, the op is compute-bound (FPU is the bottleneck).

### Gate_proj Matmul Arithmetic Intensity

For the 32-token gate_proj example above ([32, 4096] × [4096, ~10922]):

```
FLOPs  = 2 × 32 × 4096 × 10922 ≈ 2.86 × 10^9 FLOPs
Bytes  = (32×4096 + 4096×10922 + 32×10922) × 2 bytes/BF16
       ≈ (131,072 + 44,736,512 + 349,504) × 2 ≈ 90.4 MB

Arithmetic intensity ≈ 2.86e9 / 90.4e6 ≈ 31.6 FLOP/byte
```

At 32 tokens, the gate_proj matmul is **memory-bound** (31.6 << 437). The FPU is underutilized.

As token count grows (prefill or larger batches), the M_t dimension grows, FLOPs scale as
M × K × N while bytes scale as M×K + K×N + M×N, and the K×N term dominates for large K×N.
The matmul transitions to compute-bound when token count is large enough that M×K×N >>
(K×N) × ridge_point.

### SiLU Arithmetic Intensity

SiLU reads each element once and writes it once:

```
FLOPs  ≈ 2 per element (one multiply, plus the sigmoid approximation treated as ~1 FLOP
          for the roofline; note the actual instruction count is higher, but FLOPs ≈ 2)
Bytes  = 4 per element (2 bytes read BF16 + 2 bytes write BF16)

Arithmetic intensity ≈ 2 / 4 = 0.5 FLOP/byte
```

SiLU is **deeply memory-bound at all practical tensor sizes** (0.5 << 437). There is no batch
size or hidden_dim that makes SiLU compute-bound on Wormhole B0.

---

## Roofline Summary

```
Arithmetic Intensity (FLOP/byte)

1000 ┤
     │                                          ← compute-bound (FPU saturated)
     │
437  ┤ ─────────────────────────────────────── ridge point (DRAM)
     │
 32  ┤  ← gate_proj matmul @ 32 tokens (memory-bound)
     │
  0.5┤  ← SiLU (always memory-bound)
     └──────────────────────────────────────────────────────────
```

Both gate_proj and SiLU are memory-bound at 32 tokens. In this regime their latencies are
determined by how many bytes they read/write relative to DRAM (or L1) bandwidth, not by FPU
or SFPU throughput.

---

## Estimated SiLU Fraction of FFN Time

When both gate_proj matmul and SiLU are in the memory-bound regime, their latency ratio
is approximately proportional to their byte footprints:

| Operation | Bytes (32 tokens, hidden=4096, ~10922 cols) |
|---|---|
| gate_proj matmul | ~90.4 MB (computed above) |
| SiLU on gate output | 32 × 10922 × 2 bytes × 2 (read+write) ≈ 1.36 MB |

In pure memory-bandwidth terms, SiLU reads/writes ~1.4% of the bytes that gate_proj does.
However, SiLU also has per-element SFPU instruction overhead that does not appear in a naive
byte-count model. Accounting for the 160–256 SFPU instructions per tile and the SFPU's
lower effective throughput, a more realistic estimate is:

**SiLU ≈ 15–40% of gate_proj matmul latency when both are memory-bound.**

The range is wide because it depends on SFPU clock, `math_approx_mode`, core sharding, and
L1 vs DRAM data placement. Chapter 4 provides measured values that narrow this range. Use
the 15–40% figure only as a prior for experimental design — for example, it justifies
measuring SiLU as a standalone op rather than assuming it is negligible.

---

## A Note on Multi-Core Sharding

The estimates above assume a single Tensix core. In practice, TTNN distributes both the
matmul and the SiLU across many cores. The byte counts and FPU/SFPU budgets per core
scale down proportionally, but the memory-bound/compute-bound classification does not change:
sharding reduces absolute latency but does not change arithmetic intensity.

---

**Next:** [Chapter 3 — Measuring SiLU Latency](../ch03_measuring_silu_latency/index.md)
