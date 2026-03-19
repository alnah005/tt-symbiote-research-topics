# Roofline Model Primer

## What the Roofline Model Says

The roofline model gives a tight upper bound on achievable throughput for any kernel by considering two independent hardware ceilings simultaneously.

For a kernel with arithmetic intensity `AI` (FLOPs per byte transferred), the maximum achievable throughput `T` is:

```
T_achievable = min(T_peak_FPU,  AI × BW_peak)
```

Where:
- `T_peak_FPU` is the peak floating-point throughput of the hardware (FLOPs/s or FLOPs/cycle).
- `BW_peak` is the peak memory bandwidth (bytes/s or bytes/cycle).
- `AI × BW_peak` is the bandwidth-limited ceiling: the FPU cannot be fed faster than the memory system can supply data.

The roofline model gets its name from the shape of this curve: it rises linearly with `AI` on the left (bandwidth-bound region), then flattens at `T_peak_FPU` on the right (compute-bound region). A kernel's actual measured throughput always sits at or below this roof.

---

## Arithmetic Intensity

**Definition:**

```
AI = total_FLOPs / total_bytes_transferred
```

Units: FLOPs/byte.

"Total bytes transferred" means all bytes read from and written to the memory level that is the bottleneck. For a NoC-bound analysis, this is the number of bytes crossing the NoC (i.e., bytes read from other cores' L1 banks or DRAM plus bytes written back). For a DRAM-bound analysis it is DRAM traffic only.

**Example — BF16 matrix multiplication (M, K, N):**

```
FLOPs            = 2 × M × K × N            (multiply-accumulate counts as 2 FLOPs)
bytes_read_A     = M × K × 2                 (BF16 = 2 bytes per element)
bytes_read_B     = K × N × 2
bytes_written_C  = M × N × 2
total_bytes      = 2MK + 2KN + 2MN

AI = (2MKN) / (2MK + 2KN + 2MN)
   = MKN / (MK + KN + MN)
```

For large square matrices where M = K = N = D, this simplifies to:

```
AI = D³ / (3D²) = D / 3   FLOPs/byte
```

So a 1024×1024×1024 BF16 matmul has AI ≈ 341 FLOPs/byte, which is very high. A 32×32×32 matmul has AI ≈ 10.7 FLOPs/byte, which is much lower.

**Example — Elementwise op (`silu` on a [1024, 4096] BF16 tensor):**

`silu(x) = x × sigmoid(x)` requires roughly 4–6 FLOPs per element (one sigmoid, one multiply).

```
FLOPs        ≈ 5 × 1024 × 4096  =  20,971,520
bytes_read   =  1024 × 4096 × 2 =   8,388,608
bytes_write  =  1024 × 4096 × 2 =   8,388,608
total_bytes  =                     16,777,216

AI ≈ 20,971,520 / 16,777,216 ≈ 1.25 FLOPs/byte
```

This is extremely low. Almost all data-movement cost is spent just reading and writing the tensor, with negligible compute relative to bandwidth.

> **Tip:** A quick mental shortcut: elementwise ops almost always have AI < 5 FLOPs/byte. Reductions and batch-norm-style ops are in the 2–20 range. Dense matmuls with large tiles are in the 100–1000+ range.

---

## The Ridge Point

The ridge point is the arithmetic intensity at which the compute ceiling and the bandwidth ceiling are exactly equal:

```
AI_ridge = T_peak_FPU / BW_peak
```

- Ops with `AI > AI_ridge`: compute-bound. The FPU is the bottleneck; adding bandwidth does not help.
- Ops with `AI < AI_ridge`: bandwidth-bound. The FPU is starved for data; adding more compute units does not help.

---

## Wormhole B0 Hardware Ceilings

### FPU Throughput

The Wormhole B0 Tensix FPU is a matrix engine, not a traditional SIMD unit. For BF16 and FP16:

- The Wormhole B0 Tensix FPU executes **128 BF16 FMA operations per cycle** = 256 FLOPs/cycle (1 multiply + 1 add per FMA).
- Peak FPU throughput: **128 FMA ops/cycle = 256 FLOPs/cycle** for all op paths (matmul and element-wise alike). In practice, element-wise ops (add, multiply) are nearly always memory-bound and reach the 256 FLOPs/cycle FPU ceiling only in contrived fully-L1-resident scenarios; real workloads are limited by bandwidth long before FPU saturation.

For matrix operations the hardware operates on 32×32 element tiles. A full tile-triplet ((32×32) × (32×32) → (32×32)) involves 32 × 32 × 32 = 32,768 multiply-accumulate operations = **65,536 FLOPs per tile-triplet**. The matmul engine sustains one tile-triplet every 256 cycles when fully pipelined (65,536 FLOPs ÷ 256 FLOPs/cycle = 256 cycles per tile-triplet), yielding a practical throughput ceiling of **128 FP16/BF16 FMA ops per cycle per core**. Since each FMA = 1 multiply + 1 add = 2 FLOPs, this equals **256 FLOPs/cycle per core**.

### NoC Bandwidth

The Network-on-Chip (NoC) provides two independent links per core (one for reads, one for writes):

- **Per-link bandwidth:** 32 bytes/cycle.
- Read and write links are independent, so in theory up to 64 bytes/cycle of combined traffic is possible when both are saturated simultaneously.
- In practice, a single matmul core reading two input tiles and writing one output tile will primarily saturate the read link.

> **Warning:** `NOC BW UTIL` in the CSV reflects utilization of the NoC link(s) relevant to the op. A value near 1.0 means the NoC is the bottleneck, not that 100% of all available link bandwidth is consumed. Check whether the op is predominantly a read-heavy or write-heavy operation to identify which link is saturated.

### DRAM Bandwidth

- **Aggregate DRAM bandwidth (system-level):** approximately **300 GB/s** across all DRAM channels on a Wormhole B0 board.
- For a single core reading from DRAM, the effective bandwidth is far lower because DRAM bandwidth is shared across all active cores and the NoC must carry the data from the DRAM controller to the core.
- When modeling single-core DRAM-bound ops, use the NoC bandwidth as the tighter ceiling; the 300 GB/s figure applies when reasoning about system-wide, multi-core DRAM pressure.

### Ridge Point for a Single Matmul Core

Using the single-core ceilings:

```
T_peak_FPU  = 256 FLOPs/cycle   (BF16 matmul, 128 FMA ops/cycle × 2 FLOPs/FMA)
BW_NoC_read = 32  bytes/cycle   (read link)

AI_ridge = 256 / 32 = 8.0 FLOPs/byte
```

Any matmul or op with AI above 8.0 FLOPs/byte is in the compute-bound region when analyzed at the single-core NoC level. Given that even a modest 64×64×64 BF16 matmul has AI ≈ 21 FLOPs/byte, large matmuls are almost always firmly compute-bound at the per-core level — the bottleneck analysis then shifts to whether the FPU itself is fully utilized (see `FPU UTIL`).

---

## Why Tile Size Matters

All compute on Tensix cores operates on **32×32 element tiles** regardless of the logical tensor shape. This has two important consequences for roofline analysis:

1. **Sub-tile shapes incur overhead.** If a tensor dimension is not a multiple of 32, the hardware pads the tile with zeros but still executes the full 32×32 operation. The FLOPs performed are the same, but the effective arithmetic intensity on useful data is lower.

2. **Tile counts drive parallelism.** A matmul with shape (M, K, N) performs `M_t × K_t × N_t` tile multiplications, where `M_t = M/32`, `K_t = K/32`, `N_t = N/32`. The more tiles, the more the FPU pipeline can be kept full, and the closer measured throughput approaches the theoretical ceiling.

> **Note:** Throughout this chapter and the following chapters, tile counts are written as `M_t = M/32`, `K_t = K/32`, `N_t = N/32`. When a dimension is not divisible by 32, ceiling division is used: `M_t = ⌈M/32⌉`.

---

## L1 vs. DRAM Access

Each Tensix core has **~1.5 MB of L1 SRAM** (1,536 KB, scratch memory). Operands resident in L1 can be read by the FPU at NoC speed (up to 32 bytes/cycle on the local link), while operands in DRAM must first traverse the NoC and then the DRAM controller, adding latency and competing for shared bandwidth.

The practical implication for roofline analysis:

- If all operands of a kernel fit in L1 (~1.5 MB total, shared between inputs and outputs), the effective bandwidth is determined by the NoC read link (32 bytes/cycle) — the tight single-core ceiling.
- If operands spill to DRAM, DRAM latency and shared DRAM bandwidth become the bottleneck, and the effective bandwidth available to the kernel drops significantly below 32 bytes/cycle per core.

For a BF16 matmul, the L1 working set is:

```
L1_size_needed ≈ (M_t × K_t + K_t × N_t + M_t × N_t) × 32 × 32 × 2 bytes
```

Kernels that keep their working set in L1 will exhibit higher `NOC BW UTIL` (because the NoC is doing real work at its peak rate), while those that spill to DRAM will show lower `NOC BW UTIL` alongside unexpectedly long `DEVICE KERNEL DURATION`.

---

**Next:** [`classification_method.md`](./classification_method.md)
