# `gdn_full_fused_inplace` Kernel Analysis

This file documents what the `gdn_full_fused_inplace` kernel computes, where its source is expected to live, the key parameters that govern its behavior, and the architecture-specific assumptions that determine how much work is required to run it on Wormhole T3K. The conclusion is that the kernel is classified `REUSABLE_WITH_TUNING`: its algorithmic structure is portable, but CB size constants calibrated for Blackhole's 2 MB L1 must be verified against Wormhole's 1.5 MB before the kernel can be trusted to run correctly.

> **Key Finding:** The composed TTNN form (Chapter 2) is sufficient for Metal Trace compatibility and should be wired first. `gdn_full_fused_inplace` is a latency optimization — it eliminates 11 of the 12 TTNN dispatches per layer by fusing all six DeltaNet decode operations into one TT-Metalium program. It is not a prerequisite for unblocking trace.

---

## 1. What the Kernel Computes

`gdn_full_fused_inplace` implements the complete gated DeltaNet recurrent decode step as a single fused TT-Metalium kernel. The six operations it fuses match exactly the Chapter 2 TTNN decomposition:

| Step | Math | Fused kernel role |
|---|---|---|
| 1. Decay | `S_decayed = g_t * S_{t-1}` | Scalar broadcast multiply over state tiles; state held in L1 CB during execution |
| 2. Retrieval | `retrieval = S_{t-1}^T @ k̃_t` | Matrix-vector multiply using pre-decay state (independent read; both ops read `S_{t-1}`) |
| 3. Error | `error = β_t * (v_t - retrieval)` | Element-wise subtract and scalar multiply; all vectors held in L1 |
| 4. Write | `write = k̃_t ⊗ error` | Outer product via `matmul_tiles` with transposed dimensions; result is a `[d_k, d_v]` rank-1 matrix |
| 5. New state | `S_t = S_decayed + write` | Element-wise tile add; result overwrites the state CB in L1 |
| 6. Output | `o_t = S_t^T @ q̃_t` | Matrix-vector multiply; output tile written to output buffer |

The defining feature of the fused kernel is that the state matrix `S` — shape `[d_k, d_v] = [128, 128]` per head — is loaded from DRAM into L1 once at the start of the kernel, manipulated entirely in L1 through all six operations, and written back to DRAM once at the end. This eliminates the intermediate DRAM round-trips that would occur in the composed TTNN form (where each op dispatches independently and intermediate tensors may be written to DRAM between ops).

---

## 2. Source Location

> **Note:** The exact file path must be verified by searching the tt-metal and tt-symbiote repositories. The expected location, based on the Qwen3.5-27B Blackhole implementation, is one of:
>
> - `models/experimental/tt_symbiote/ops/gdn_full_fused_inplace.cpp` (or a `.hpp` / `.py` op wrapper alongside it)
> - `models/experimental/gdn/` (a standalone op directory if the kernel is factored out)
>
> The kernel may also be registered under a different identifier in the TTNN op registry. Grep for `gdn_full_fused_inplace` across `models/experimental/` and `ttnn/cpp/ttnn/operations/experimental/` to confirm the location before beginning the port.

The kernel is composed of at least three files following the standard TT-Metalium custom op pattern:

- **Op definition and dispatch** (`.cpp` / `.py`): registers the op, validates tensor shapes, selects the core grid, and launches the RISCV programs.
- **RISCV data movement programs** (reader and writer): issue NOC DMA reads for the state and input tensors into L1 CBs; issue NOC DMA writes for the updated state and output tensor from L1 to DRAM.
- **RISCV compute program**: executes the six fused operations on the tile data already present in L1 CBs, using the TT-Metalium compute API (`tile_regs_acquire`, `matmul_tiles`, `add_tiles`, `mul_tiles`, `mul_tiles_bcast_scalar`, etc.).

---

## 3. Key Parameters to Document During Source Review

When locating the kernel source, record the following values. These are the parameters that determine Wormhole compatibility.

### 3.1 Tile Size

The kernel must be written for 32x32 BF16 tiles to be compatible with Wormhole. Blackhole B0 supports 64x64 tiles in some configurations. Any `constexpr` like `TILE_WIDTH`, `TILE_HEIGHT`, or equivalent that is hardcoded to 64 must be changed to 32.

### 3.2 Heads Per Core

The kernel processes one or more attention heads per Tensix core. The expected assignment for the Blackhole implementation is one head per core, which matches the Wormhole adaptation plan (see `wormhole_t3k_adaptation.md`). Confirm whether the `heads_per_core` value is a `constexpr` or a runtime parameter.

### 3.3 CB (Circular Buffer) Layout

Record the CB indices, sizes, and roles:

| CB index | Expected role | Expected size (per head) |
|---|---|---|
| CB0 | State S `[d_k, d_v]` | 32 KB (128 × 128 × 2 bytes BF16) |
| CB1 | k̃ input `[d_k, 1]` padded to tile | 256 B padded to 2 KB (one 32×32 tile) |
| CB2 | v input `[d_v, 1]` padded to tile | 256 B padded to 2 KB (one 32×32 tile) |
| CB3 | g and β scalars, broadcast | 2 KB (one tile; scalars broadcast over full tile) |
| CB4 / CBOUT | Output o_t `[d_v, 1]` | 256 B padded to 2 KB (one 32×32 tile) |

Total CB usage per core: approximately 40 KB (32 KB + 2 KB + 2 KB + 2 KB + 2 KB). This is the figure to recheck against Wormhole's 1.5 MB L1 (see Section 4.1).

### 3.4 Data Format

Record the `DataFormat` enum values used in the CB configurations. Blackhole supports `FP32_DEST_ACC` in the FPU accumulator register. Wormhole supports BF16 accumulation in the FPU. If the kernel requests `FP32_DEST_ACC`, this path must be verified on Wormhole — either Wormhole's BF16 accumulator is sufficient (and the flag is dropped), or the kernel must be restructured to avoid it.

### 3.5 State Layout

Confirm whether the state tensor `S` is stored as `TILE_LAYOUT` or `ROW_MAJOR` in DRAM. `TILE_LAYOUT` (32×32 BF16 tiles) is strongly preferred: it enables tile-aligned DMA reads from DRAM into the L1 CB without per-element address arithmetic. `[128, 128]` in TILE layout is exactly 16 tiles (4 columns × 4 rows of 32×32 tiles), which maps cleanly onto the CB0 allocation.

### 3.6 Core Grid

Record the core grid used in the Blackhole implementation. The expected value is a single row or column of cores, with one core per head. Under the head-parallel sharding on Blackhole (which may differ from T3K's 4 heads per device), the number of heads per device determines the grid size. See `wormhole_t3k_adaptation.md` for the T3K-specific grid.

---

## 4. Architecture-Specific Concerns for the Blackhole to Wormhole Port

### 4.1 CB Size Constants vs. Wormhole L1

> **Key concern:** L1 per Tensix core is 2 MB on Blackhole and 1.5 MB on Wormhole. Any CB size constant that was set to a value that fits in 2 MB but exceeds 1.5 MB will fail silently or cause kernel hang on Wormhole.

The expected total CB usage for the DeltaNet fused kernel is approximately 40 KB (see Section 3.3). At this size, there is no risk: 40 KB is well within both 2 MB and 1.5 MB. However, the Blackhole implementation may have allocated additional working memory in L1 for scratch tiles, double-buffering the input CBs, or prefetching the next state tile. All such allocations must be audited.

**Action:** Extract all `CreateCircularBuffer` calls from the kernel source. Sum the `total_size` fields. Verify that the total does not exceed 1.5 MB. If it does, reduce double-buffering depth or eliminate scratch tiles that were over-provisioned for Blackhole's 2 MB budget.

### 4.2 FPU Tile Dimensions

Wormhole's FPU matrix engine operates on 32×32 tiles in BF16. Blackhole B0's FPU can operate on 64×64 tiles in some configurations. Any compute program that hardcodes the tile dimension as 64 (in loop bounds, accumulator register counts, or CB size calculations) is incorrect on Wormhole.

**Action:** Search for any literal `64` that appears in the compute kernel source in a context related to tile dimensions. Replace with `TILE_HEIGHT` and `TILE_WIDTH` compile-time constants (which resolve to 32 on Wormhole).

### 4.3 FP32_DEST_ACC Usage

Blackhole B0 supports FP32 accumulation in the FPU destination register natively. Wormhole's FPU accumulates in BF16 by default; FP32 accumulation is available on Wormhole but may require different compile-time flags or explicit conversion steps.

**Action:** Search the compute kernel source for `FP32_DEST_ACC` or `DEST_ACCUM_EN` or equivalent. If present, verify that the accumulation path is correct on Wormhole by running the kernel with a reference BF16 input and comparing the state update output against the PyTorch FP32 reference. A PCC drop below 0.999 would indicate that the accumulation format is causing significant numerical error.

### 4.4 NOC Routing and Address Offset Calculations

The data movement RISCV programs issue NOC reads and writes using core coordinates and address offsets. Blackhole and Wormhole have different NOC topologies and Tensix core numbering schemes.

**Action:** Review the `noc_async_read` and `noc_async_write` calls in the reader and writer RISCV programs. Confirm that core coordinates are computed programmatically from the op dispatch (not hardcoded). Address offsets for DRAM buffers should be computed from tensor metadata passed at dispatch time; if they are computed statically, they must be verified against Wormhole's memory map.

---

## 5. Reuse Classification

| Property | Assessment |
|---|---|
| Algorithmic structure | Portable — the 6 ops and their L1 state-in-kernel pattern are not architecture-specific |
| CB total size | Expected ~40 KB — well within 1.5 MB Wormhole L1; no fundamental obstacle |
| FPU tile dimensions | Potential issue if hardcoded to 64; must be verified and replaced with constants |
| FP32_DEST_ACC | Potential flag mismatch; must verify Wormhole accumulation path |
| NOC routing | Likely programmatic; must verify no hardcoded Blackhole-specific coordinates |
| Data movement pattern | DRAM-to-L1 stream via NOC — same pattern on both architectures |

> **Key Finding:** The kernel is classified **`REUSABLE_WITH_TUNING`**. The algorithmic structure — six fused DeltaNet decode operations with state held in L1 — is portable between Blackhole and Wormhole. The expected work is: (1) audit and correct CB size constants, (2) replace any hardcoded 64-tile FPU dimension with 32-tile constants, (3) verify the FP32_DEST_ACC accumulation path, (4) confirm NOC routing is programmatic. No rewrite of the compute logic or DMA streaming pattern is required.

The specific constant changes and the verification test are detailed in `wormhole_t3k_adaptation.md`.

---

## 6. Relationship to the Composed TTNN Form

The fused kernel and the composed TTNN form (Chapter 2) implement the same six mathematical operations and must produce identical numerical results (PCC > 0.999 against the PyTorch reference). Their difference is dispatch overhead:

| Implementation | Dispatches per layer | Estimated dispatch latency (30 layers) |
|---|---|---|
| Composed TTNN (Chapter 2) | 12 ops | ~360 dispatches at 1–5 µs each = 0.36–1.8 ms |
| Fused kernel (this chapter) | 1 kernel | ~30 dispatches at 1–5 µs each = 30–150 µs |

The DRAM bandwidth cost (state read + write, 256 KB per device per layer round-trip) is identical for both forms because both must load the state from DRAM and write it back. The only cost difference is kernel dispatch overhead.

For the initial trace compatibility goal — unblocking `ttnn.begin_trace_capture` / `ttnn.execute_trace` for the hybrid decoder — the composed form is sufficient. The fused kernel becomes relevant when profiling (Chapter 6) shows that dispatch overhead, not DRAM bandwidth, is the dominant DeltaNet decode cost.
