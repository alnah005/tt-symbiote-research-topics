# Wormhole Architecture

This file covers the hardware structures on Wormhole B0 that directly affect how TTNN kernels perform. The focus is on what matters for matrix multiplication and memory management in MoE workloads. All hardware quantities refer to **Wormhole B0** unless stated otherwise.

---

## 1. The Tensix Core

The fundamental programmable unit on Wormhole is the **Tensix core**. Each Tensix core contains five RISC-V processors, a Math engine, and its own local SRAM. Understanding how these components interact is the key to understanding why certain kernel configurations are faster than others.

### 1.1 RISC-V Processors

A Tensix core contains five RISC-V scalar processors, which Tenstorrent calls **BRISC**, **NCRISC**, **TRISC0**, **TRISC1**, and **TRISC2**. In typical kernel operation these divide labor along the following lines:

- **BRISC (Blank RISC)** — general-purpose control processor; handles kernel setup and orchestration.
- **NCRISC (NoC RISC)** — manages NoC DMA transfers into and out of this core's L1. Responsible for issuing read and write transactions to DRAM or other cores.
- **TRISC0 / TRISC1 / TRISC2** — three processors dedicated to the data path. TRISC0 and TRISC1 typically run the "unpack" phase (preparing tiles from L1 source buffers into register files for the Math engine). TRISC2 runs the "pack" phase (writing Math engine outputs from registers back into L1 destination buffers).

At the Metalium level, you write separate kernel programs that execute on each of these processors. TTNN's op library provides pre-written kernels; the program config parameters you pass to ops like `ttnn.matmul` control how those kernels parameterize the work across these processors.

### 1.2 Math Engines: FPU and SFPU

The Math engine inside each Tensix core contains two compute units:

**FPU (Floating-Point Unit)** — a systolic array-style unit designed for matrix operations on tiles. The FPU operates on 32×32 tiles as its native atomic unit. A single FPU can perform a 32×32 × 32×32 matrix multiply-accumulate in one pass. When you perform a matmul in TTNN, the FPU is doing the actual tile-level multiply-accumulate work.

**SFPU (Special-Function FPU)** — a SIMD-style unit for element-wise operations: activation functions (GELU, SiLU, ReLU), softmax, type conversions, and other per-element math. The SFPU can process 32 elements per cycle. For MoE workloads the SFPU handles the router's softmax and the top-k selection logic, as well as the activation functions inside expert FFNs.

The FPU and SFPU share register files but operate sequentially on a tile: the FPU does matrix math, then the SFPU can apply element-wise transforms on the same tile in-place. This pipelining is why fused ops (e.g., a matmul followed immediately by a GELU) can be more efficient than separate op calls.

> **Tip:** When profiling, look for the FPU utilization metric in tt-metal's performance profiler. Low FPU utilization almost always means the core is stalling on memory — either waiting for tiles to arrive from DRAM or waiting for NoC transfers from other cores.

### 1.3 The Network-on-Chip (NoC)

Each Tensix core connects to every other Tensix core (and to DRAM controllers and ethernet endpoints) through the **NoC** — a 2D mesh network running at high bandwidth on-die. The NoC carries:

- Tile data reads/writes between cores during multicast or unicast transfers
- DRAM reads and writes issued by NCRISC
- Synchronization signals and semaphore updates

The NoC is the mechanism that enables the "multicast" in `MatmulMultiCoreReuseMultiCastProgramConfig`. When a weight tile is multicast, NCRISC on one source core issues a single NoC write that fans out to a row or column of destination cores simultaneously, rather than requiring each destination core to issue its own DRAM read. This is a critical bandwidth optimization for MoE expert matmuls, which typically have large static weight matrices and small, variable activation batches.

> **Warning:** NoC congestion is a real bottleneck in large multi-expert configurations. If many cores simultaneously try to stream data to the same column of destination cores, NoC arbitration delays can erode the theoretical bandwidth advantage. Chapter 4 covers strategies for staggering transfers to reduce NoC hotspots.

---

## 2. L1 SRAM vs. DRAM: The Memory Hierarchy

### 2.1 L1 SRAM

Each Tensix core has **1.5 MB of L1 SRAM** (on Wormhole B0). This scratchpad is the only memory the FPU and SFPU can operate on directly. Before the Math engine can process a tile, that tile must reside in L1. After computation, the output tile sits in L1 before being written back to DRAM or forwarded to another core.

The critical constraint for matmul performance is that **all active input and output buffers for a kernel must fit in L1**. A typical double-buffered matmul kernel on a single core needs to hold:

- One circular buffer for the A-matrix tiles being streamed in (input activations)
- One circular buffer for the B-matrix tiles being streamed in (weight tiles, possibly multicast)
- One output buffer for accumulating result tiles before packing out

Each 32×32 tile of bfloat16 data occupies `32 × 32 × 2 bytes = 2 KB`. A bfloat8_b tile occupies 1024 value bytes + 64 shared exponent bytes = **1088 bytes** (one exponent byte per 16-value block, 64 blocks per tile). With 1.5 MB total L1, a single core can hold roughly 768 bfloat16 tiles (1,572,864 / 2,048 = 768) or approximately 1,445 bfloat8_b tiles (1,572,864 / 1,088 ≈ 1,445.5). In practice, kernel metadata, stack space, and synchronization structures consume some L1, so the effective capacity for data buffers is somewhat lower.

> **Tip:** Buffer size calculations are the most common source of L1 overflow errors. When TTNN raises a "not enough L1" error, the fix is usually one of: reducing `per_core_M` or `per_core_N` (fewer output tiles per core), switching from bfloat16 to bfloat8_b (~1.88× tile size reduction, from 2048 bytes to 1088 bytes), or reducing the number of double-buffer slots.

### 2.2 DRAM Bandwidth

Wormhole B0 has **12 GDDR6 DRAM channels**, providing a theoretical aggregate bandwidth of approximately **288 GB/s**. The DRAM channels are distributed around the die and accessed through DRAM controller tiles that sit at fixed positions in the core grid.

Practical DRAM bandwidth available to a single matmul kernel is a fraction of this peak, because:

- Multiple kernels running concurrently compete for DRAM bandwidth
- DRAM access efficiency depends on access patterns (sequential vs. strided)
- NCRISC-issued reads have latency that must be hidden by overlapping compute with the next tile's fetch (double-buffering)

For MoE workloads, the dominant DRAM access pattern is reading expert weight matrices. These matrices are large (often `[d_model, d_ff]` where `d_ff` can be 4× or 8× `d_model`) and static between forward passes. The TTNN program cache and, more importantly, weight-sharding strategies (covered in Chapter 5) exist specifically to manage this weight-loading bottleneck.

### 2.3 Impact on Tile Scheduling

The interplay between L1 capacity and DRAM bandwidth determines the optimal tile scheduling strategy:

1. **Double-buffering**: While the FPU processes batch N of tiles, NCRISC prefetches batch N+1. This requires two full sets of input buffers in L1 but hides DRAM latency behind compute.

2. **Tile reuse (weight multicasting)**: For a matmul where B (the weight matrix) is the same across many A rows, multicast a single copy of each B tile to all relevant cores via the NoC rather than having each core fetch its own copy from DRAM. This trades NoC bandwidth for DRAM bandwidth — a favorable trade when the NoC is less congested than the DRAM channels.

3. **K-loop blocking**: When K is large (e.g., `K_t = d_model / 32 = 128` for `d_model = 4096`), a single core cannot hold all K weight tiles in L1 at once. The kernel iterates over K in blocks, loading a block of B tiles, computing partial sums, then loading the next block. `per_core_M` and the inner K loop size are the main levers for tuning this schedule.

---

## 3. Core Grid Layouts

### 3.1 Logical vs. Physical Grid

In TTNN, you specify compute grids as `ttnn.CoreGrid(y, x)` or `ttnn.CoreRange` objects. These are **logical** coordinates. The TTNN runtime maps them to physical Tensix core coordinates on the die.

On Wormhole B0 the full NOC grid is **10×12** (10 rows, 12 columns), but not all of those positions are Tensix compute cores. The 10×12 count includes NOC router tiles, DRAM controller tiles, ethernet tiles, and PCIe tiles alongside the programmable Tensix cores. The actual programmable Tensix core array is **8×10 = 80 Tensix cores**.

The effective programmable grid available to TTNN workloads on a single Wormhole B0 chip is therefore **80 Tensix cores** (8×10). When you request a `CoreGrid(y=8, x=8)` in TTNN, you are using 64 of those 80 available cores; `CoreGrid(y=8, x=10)` uses the full 80-core programmable array.

> **Warning:** Requesting a grid larger than the available programmable cores will cause a runtime error. If you see errors about core coordinates out of range, verify your grid dimensions against the available core count for your specific chip and system configuration.

### 3.2 Grid Shape and Work Distribution

How you shape the logical core grid affects performance:

- **Tall grids (large Y, small X)**: More rows of cores. Favorable when M is large (many output rows to distribute) and N is small.
- **Wide grids (small Y, large X)**: More columns of cores. Favorable when N is large (many output columns to distribute) and M is small.
- **Square grids**: Often a good starting point; balance compute and NoC traffic evenly.

For MoE expert matmuls where M varies dynamically (token dispatch sends different numbers of tokens to different experts), the grid shape should be chosen based on the *maximum expected* M rather than the mean, to avoid stalls when a popular expert receives a large batch.

### 3.3 Core Grid Specialization in MoE

A single Wormhole B0 chip can run multiple expert computations simultaneously by assigning disjoint subgrids to different experts. For example, with 8 experts and an 80-core grid, you might assign a 10-core subgrid to each expert. This approach is discussed in detail in Chapter 3 (Expert Tensor Parallelism); the key concept to internalize here is that **a CoreRange in TTNN is a rectangular subgrid**, and non-rectangular or irregular assignments require multiple CoreRange objects combined into a CoreRangeSet.

---

## 4. T3K Multi-Chip Mesh

### 4.1 Physical Topology

The **T3K** is Tenstorrent's 8-chip Wormhole B0 system. The eight chips are connected in a **2×4 mesh** topology using high-speed ethernet links. Each chip connects to its immediate neighbors (left, right, up, down in the 2D mesh). Ethernet links between chips have a bandwidth of approximately **12.5 GB/s bidirectional per link**, substantially lower than on-chip NoC bandwidth.

The consequence is that cross-chip transfers are expensive relative to on-chip transfers. Algorithms that require frequent all-to-all or all-reduce communication across all 8 chips will be ethernet-bandwidth-bound. Algorithms that can organize computation so that cross-chip communication is infrequent (e.g., each chip handles a disjoint subset of experts and only aggregates at the end) are far more efficient on T3K.

### 4.2 Tensor Parallelism Across the Mesh

TTNN exposes mesh-level operations through its distributed tensor primitives. A tensor can be sharded across the mesh so that each chip holds a slice:

```python
import ttnn

# Create a mesh device spanning all 8 chips in a 2x4 arrangement
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))

# A weight tensor sharded across the 8-chip mesh along dim 0 (row sharding)
# Each chip holds [d_model / 8, d_ff] of the full [d_model, d_ff] weight matrix
weight_mesh = ttnn.as_tensor(
    weight_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),  # shard along dim 0
)
```

For MoE models, the two primary parallelism strategies on T3K are:

**Expert parallelism**: Different chips own different experts. No cross-chip communication is needed during expert FFN forward passes; cross-chip all-to-all is only required at the token dispatch and token gather steps. This strategy works well when the number of experts is a multiple of 8 and each expert fits on one chip.

**Tensor parallelism within an expert**: A single large expert's weight matrices are sharded across multiple chips. Each chip computes a partial matmul and an all-reduce (or reduce-scatter + all-gather) is performed across chips to combine partial results. This is necessary when a single expert is too large to fit in DRAM of one chip, or when the expert computation is the throughput bottleneck and you want to parallelize within it.

> **Tip:** On T3K, expert parallelism generally outperforms intra-expert tensor parallelism for standard MoE configurations (8–64 experts, d_model up to 8192) because it avoids the ethernet all-reduce communication on every forward pass. Intra-expert tensor parallelism becomes necessary for very large experts (d_ff > 32768 or larger) where DRAM capacity on one chip is insufficient.

### 4.3 Ethernet Link Latency and Bandwidth Budgeting

The T3K ethernet links operate at approximately **100 Gb/s (12.5 GB/s) per link**. For an all-reduce across 8 chips using a ring algorithm, the effective bandwidth is:

```
All-reduce bandwidth (ring, N chips) = N / (2(N-1)) × link_bandwidth
All-reduce bandwidth (ring, 8 chips) = 8 / (2×7) × 12.5 GB/s ≈ 7.1 GB/s
```

For a typical MoE all-to-all (dispatching tokens from all chips to the owning expert chips), the volume transferred is:

```
Transfer volume = num_tokens × d_model × dtype_bytes
               = 2048 × 4096 × 2 bytes (bfloat16)
               = ~16 MB per forward pass (per chip)
```

At ~7.1 GB/s effective ethernet bandwidth, this is on the order of 2–3 ms per forward pass — small relative to the expert FFN compute time at typical batch sizes, but non-trivial at small batch sizes where compute is fast. Chapters 6 and 7 discuss techniques for overlapping this communication with compute.

---

## Summary

| Concept | Key Number / Fact |
|---------|-------------------|
| RISC-V cores per Tensix | 5 (BRISC, NCRISC, TRISC0/1/2) |
| L1 SRAM per Tensix core | 1.5 MB |
| Tile sizes | See ttnn_programming_model.md Section 1.2 |
| DRAM channels | 12 GDDR6 channels |
| DRAM peak bandwidth | ~288 GB/s aggregate |
| Programmable core grid | 8×10 = 80 Tensix cores (Wormhole B0); full NOC grid is 10×12 including non-compute tiles |
| T3K chip count | 8 Wormhole B0 chips |
| T3K topology | 2×4 mesh via ethernet |
| T3K ethernet link bandwidth | ~12.5 GB/s per link |

---

---

**Next:** [ttnn_programming_model.md](./ttnn_programming_model.md)
