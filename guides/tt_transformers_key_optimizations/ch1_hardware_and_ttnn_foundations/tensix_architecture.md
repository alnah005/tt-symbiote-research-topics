# Tensix Architecture

This file describes the internal structure of a Tensix core — the fundamental compute tile on a Tenstorrent chip. Every optimization in tt-transformers traces back to constraints or opportunities in the Tensix microarchitecture. Understanding this substrate will make later discussions of tile sizing, output subblock dimensions, L1 sharding, and math fidelity immediately interpretable rather than arbitrary.

---

## The Tensix Core at a Glance

A Tenstorrent chip (Wormhole B0) contains a grid of Tensix cores. Each core is a self-contained, heterogeneous processing unit. Unlike a CUDA SM, which is primarily a SIMD lane bundle, a Tensix core is organized as a small pipeline with distinct functional units:

- **Matrix FPU** (also called FPU or MATMUL engine): performs tile-level matrix multiplication
- **SFPU** (Scalar FPU): performs element-wise scalar operations — activations, transcendentals, comparisons
- **Packer**: serializes computation results from the Dst register file to L1 SRAM or DRAM
- **Unpacker**: deserializes tile data from L1 SRAM into the SrcA/SrcB register files for the FPU
- Five **RISC-V processors**: **BRISC** (Reader kernel — loads input tiles from DRAM/NoC into L1), **NCRISC** (Writer kernel — writes output tiles from L1 to DRAM/NoC), and **TRISC0**, **TRISC1**, **TRISC2** (Compute kernels — the three stages of the compute pipeline that orchestrate the FPU, SFPU, Packer, and Unpacker)

These units can operate concurrently within one Tensix core, which enables the pipelining strategies described in Chapter 4.

---

## Matrix FPU: The 16×16 × 16×16 Primitive

The matrix FPU computes one matrix-multiply-accumulate per clock cycle over its native tile primitive. The input tiles fed to SrcA and SrcB must conform to:

- **SrcA tile**: 16 rows × 16 columns (a "face")
- **SrcB tile**: 16 rows × 16 columns

A full 32×32 TTNN tile is composed of four 16×16 faces arranged in a 2×2 grid. The matrix engine processes a 32×32 × 32×32 matmul by internally iterating over the 16×16 face sub-tiles in the required sequence. From a TTNN programmer's perspective, the atomic unit of computation is the 32×32 tile, but the hardware executes it as four 16×16 face-level operations.

The relevance for tt-transformers is direct: weight tensors and activations are always padded to multiples of 32 in each dimension before being placed on device, and all program configs specify blocking in units of 32×32 tiles. Attempting to run on non-tile-aligned shapes incurs padding overhead; model weights that are naturally tile-aligned (e.g., 4096×4096 for Llama 3 8B's Q projection) run without waste.

---

## SFPU: Element-Wise Scalar Operations

The SFPU operates on the Dst register file (see next section) and handles operations that the matrix engine cannot: `gelu`, `silu`, `softmax`, `exp`, `rsqrt`, `tanh`, square root, and comparisons. In LLM inference, the SFPU is responsible for:

- Softmax in attention score normalization
- SiLU (sigmoid linear unit) in SwiGLU MLP blocks
- RMSNorm's reciprocal square root (`rsqrt`)
- Elementwise masking and clipping

When `math_approx_mode=True` is set in `WormholeComputeKernelConfig`, the SFPU uses polynomial approximations for transcendental functions (e.g., `rsqrt`, `exp`). This trades a small amount of numeric fidelity for meaningfully higher SFPU throughput — typically the right choice for RMSNorm.

---

## Dst Register File

The **Dst register file** is the output accumulation space for the matrix FPU and the input/output space for the SFPU. It sits between the FPU and the Packer:

```
SrcA ─┐
      ├─→ [Matrix FPU] ─→ [Dst] ─→ [SFPU] ─→ [Packer] ─→ L1
SrcB ─┘
```

### Capacity: fp16 vs fp32 Mode

The Dst register file has a **fixed physical size**. Its capacity in terms of 32×32 tiles depends on the accumulation data type:

| Accumulation Mode | Tiles in Dst |
|---|---|
| fp16 (BF16 accumulation) | 8 tiles |
| fp32 (FP32 accumulation) | 4 tiles |

This directly constrains the **output subblock dimensions** used in matmul program configs. The output subblock is the rectangular block of output tiles that a core accumulates into Dst before the Packer writes them to L1. For a matmul output subblock of height `out_subblock_h` and width `out_subblock_w`, the product `out_subblock_h × out_subblock_w` must not exceed the Dst capacity:

- fp16 accumulation: `out_subblock_h × out_subblock_w ≤ 8`
- fp32 accumulation: `out_subblock_h × out_subblock_w ≤ 4`

Larger subblocks reduce the number of Packer invocations and improve compute efficiency — but they are bounded by Dst capacity. When `fp32_dest_acc_en=True` is set in `WormholeComputeKernelConfig`, the accumulation happens in FP32 (important for accuracy when using BF16 operands), but the Dst capacity halves from 8 to 4 tiles.

### Practical Subblock Sizing

Common subblock configurations for tt-transformers matmuls:

| Config | `out_subblock_h` | `out_subblock_w` | Product | fp16 valid? | fp32 valid? |
|---|---|---|---|---|---|
| 4×2 | 4 | 2 | 8 | Yes | No |
| 2×4 | 2 | 4 | 8 | Yes | No |
| 1×8 | 1 | 8 | 8 | Yes | No |
| 2×2 | 2 | 2 | 4 | Yes | Yes |
| 1×4 | 1 | 4 | 4 | Yes | Yes |
| 1×2 | 1 | 2 | 2 | Yes | Yes |

Chapter 3 discusses how to set these values in the `MatmulMultiCoreReuseProgramConfig` and related configs.

---

## NoC: Network-on-Chip

The **NoC (Network-on-Chip)** is the intra-chip interconnect that moves data between Tensix cores, DRAM controllers, and Ethernet ports. On Wormhole B0:

- The NoC is a 2D mesh connecting all cores on the chip
- Each NoC link provides approximately **32 bytes/cycle** per direction
- Aggregate intra-chip NoC bandwidth (summed across all links of both NoC meshes simultaneously) is significantly higher than DRAM bandwidth, making it practical to saturate DRAM without becoming NoC-bound on most workloads; a precise aggregate figure is architecture-version-dependent and should be taken from Tenstorrent's official documentation

In practice, the NoC is used for:

1. **Unicast reads/writes**: a core fetches a tile from DRAM or from another core's L1
2. **Multicast**: a single source core broadcasts the same tile to a rectangular region of destination cores simultaneously — the key mechanism behind `MatmulMultiCoreReuseMultiCastProgramConfig` (Chapter 3)
3. **Reduction trees**: partial sums from multiple cores collected to a root core

The NoC operates concurrently with the compute pipeline. In a well-optimized kernel, the Reader kernel on a core's RISC-V processor issues NoC DMA requests to pre-fetch the next tile from DRAM into L1 while the Compute kernel is still processing the current tile. This overlap is the basis for double-buffering (Chapter 4).

### Tile Routing

TTNN tiles are addressed by their physical location in the tensor buffer. For an interleaved buffer, TTNN distributes tile addresses round-robin across DRAM banks. When a core needs tile `i`, the Unpacker issues a NoC read to the bank holding that tile's data. For a sharded buffer, each core's slice is pre-placed in that core's L1 — no NoC traffic needed to fetch the shard itself, only to write outputs.

---

## L1 SRAM vs DRAM

This is perhaps the most important hardware concept for understanding tt-transformers performance optimization.

### L1 SRAM

- **120 KB per Tensix core**, exclusive to that core
- **Low latency**: single-cycle access from the Unpacker and Packer
- Used as staging area for tiles flowing through the compute pipeline
- Managed via **circular buffers** (CB): named slots that Reader/Compute/Writer kernels communicate through
- **Limit**: 120 KB is small. A single 32×32 BF16 tile is 2 KB. L1 can hold at most ~60 tiles simultaneously (in practice fewer, because circular buffer slots, kernel binary, and stack space also consume L1).

### DRAM

- **Gigabytes shared across the chip** (12 GB on a single N150 card)
- Connected to Tensix cores through NoC + DRAM controllers
- Higher latency than L1: tens to hundreds of nanoseconds per access, depending on banking and queuing
- Bandwidth on Wormhole: approximately **~256–300 GB/s aggregate** across all DRAM banks

### Why L1 Is a Performance Goal

The performance gap between L1 and DRAM access is fundamental:

- A matmul that keeps all intermediate tiles in L1 never stalls waiting for DRAM — it runs at compute throughput
- A matmul that must read input tiles from DRAM on every use is **memory-bandwidth-bound** — the matrix FPU idles waiting for data
- **Sharding activations to L1** means each core's working set fits entirely in its 120 KB, turning memory-bound ops into compute-bound ops

For decode-phase inference (one token at a time, small batch), the activation tensor is tiny — often just a few tiles per core. Sharding it to L1 is straightforward. For prefill (large sequence length), the activation tensor is large and may not fully fit in L1 across all cores; in that case, interleaved DRAM with double-buffering is used to hide latency.

This is the foundational reason the sharding strategies in Chapters 3 and 4 prioritize L1 placement for activations.

---

## Wormhole Peak Throughput Numbers

The following throughput figures are for a single Wormhole B0 device (one chip), varying by math fidelity mode:

| Math Fidelity | Peak Throughput (matmul) |
|---|---|
| LoFi | ~262 TOPS |
| HiFi2 | ~148 TOPS |
| HiFi3 | ~111 TOPS |
| HiFi4 | ~74 TOPS |

TOPS = Tera Operations Per Second (for INT/FP multiply-add operations).

Key observations:

- LoFi is approximately **3.5× faster** than HiFi4 for matmul-dominated workloads
- HiFi2 is approximately **2× faster** than HiFi4 — a common sweet spot for LLM inference
- These numbers are **peak** values; real models achieve a fraction of peak depending on memory access efficiency, tensor shapes, and pipeline utilization

The math fidelity system is explained in detail in [math_fidelity_and_data_formats.md](./math_fidelity_and_data_formats.md). The practical selection guide for LLM layers is also in that file.

---

## Key Takeaways

- A Tensix core contains a matrix FPU (operating on 32×32 tiles built from 16×16 faces), an SFPU for element-wise ops, a Packer, an Unpacker, and five RISC-V processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) — all capable of concurrent operation.
- The Dst register file holds 8 tiles in fp16 accumulation mode and 4 tiles in fp32 accumulation mode; this hard constraint drives all output subblock dimension choices in matmul program configs.
- L1 SRAM (120 KB per core) is fast and local; DRAM is large but shared and high-latency. Sharding tensors to L1 is the primary technique for turning memory-bound ops into compute-bound ops.
- The NoC supports multicast, enabling weight broadcast to many cores simultaneously — the key to `MatmulMultiCoreReuseMultiCast` configs.
- Wormhole peak throughput scales from ~74 TOPS (HiFi4) to ~262 TOPS (LoFi) — a 3.5× range that directly impacts inference token rate.

---

## Further Reading

- Tenstorrent Wormhole Architecture Overview (tenstorrent.com/technology) — chip die layout and NoC topology diagrams
- TT-Metalium kernel programming guide (github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_guide.md) — detailed CircularBuffer and RISC-V kernel dispatch documentation
- [ttnn_tensor_model.md](./ttnn_tensor_model.md) — how TTNN abstracts the memory hierarchy described here into tensor layout and memory config objects
- [math_fidelity_and_data_formats.md](./math_fidelity_and_data_formats.md) — the precision system behind the throughput numbers above

---

**Next:** [`ttnn_tensor_model.md`](./ttnn_tensor_model.md)
