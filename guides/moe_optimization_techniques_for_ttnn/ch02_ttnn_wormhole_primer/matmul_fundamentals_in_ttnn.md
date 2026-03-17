# Matmul Fundamentals in TTNN

This file explains how TTNN maps matrix multiplication dimensions onto the Tensix grid, establishes the 32×32 tile as the atomic compute unit, defines the output subblock parameters in full, and guides selection between the two primary matmul program configs.

The definitions here — particularly `out_subblock_h` and `out_subblock_w` — are used without re-explanation in Chapters 3, 4, and 7. Read this file completely before proceeding.

---

## 1. M, K, N Dimensions and Their Mapping to the Grid

### 1.1 Matmul Dimensions

Every matrix multiplication in TTNN computes `C = A × B` where:

- `A` has shape `[M, K]`
- `B` has shape `[K, N]`
- `C` has shape `[M, N]`

In tile units:

- `M_t = M / 32` — number of tiles along the M (output row) dimension
- `K_t = K / 32` — number of tiles along the K (contraction) dimension
- `N_t = N / 32` — number of tiles along the N (output column) dimension

For a transformer expert FFN's first linear layer: if `d_model = 4096` and `d_ff = 16384`, and the token batch for one expert has `tokens_for_expert = 64` tokens, then:

```
A = [tokens_for_expert, d_model] = [64, 4096]   → M_t = 2,   K_t = 128
B = [d_model, d_ff]              = [4096, 16384] → K_t = 128, N_t = 512
C = [tokens_for_expert, d_ff]   = [64, 16384]   → M_t = 2,   N_t = 512
```

### 1.2 Distributing Work Across the Core Grid

TTNN assigns the output tile space `[M_t, N_t]` across the core grid. Each core is responsible for computing a rectangular block of output tiles:

- `per_core_M` — number of output tile rows assigned to each core (in the M_t dimension)
- `per_core_N` — number of output tile columns assigned to each core (in the N_t dimension)

Given a core grid of `grid_y × grid_x` cores:

```
M_t = per_core_M × grid_y
N_t = per_core_N × grid_x
```

For example, with `M_t = 2`, `N_t = 512`, and a grid of `2 × 8`:

```
per_core_M = M_t / grid_y = 2 / 2 = 1
per_core_N = N_t / grid_x = 512 / 8 = 64
```

Each of the 16 cores (2 rows × 8 columns) computes a `[per_core_M, per_core_N] = [1, 64]` block of output tiles, meaning 64 individual 32×32 output tiles per core.

> **Warning:** `M_t` must be exactly divisible by `grid_y`, and `N_t` must be exactly divisible by `grid_x`. If they are not, you must either pad the tensor dimensions (adding zero rows/columns to the next multiple of `32 × grid_y` or `32 × grid_x`), or use a smaller grid that divides evenly. Mismatches cause TTNN to raise a shape/grid incompatibility error.

### 1.3 The K Reduction Loop

Each core must sum contributions from all `K_t` tiles along the contraction dimension. The core iterates over `K_t / in0_block_w` steps, loading `in0_block_w` A tiles and the corresponding `in0_block_w` B tiles per step, accumulating partial results. `in0_block_w` is the number of K tiles processed per inner loop step and is set in the program config.

> **Constraint:** `K_t` must be exactly divisible by `in0_block_w` — i.e., `K_t % in0_block_w == 0`. If this does not hold, the inner K-loop step count is non-integer: the kernel will either silently drop the remaining K tiles (producing incorrect results) or raise a runtime error. Always verify this divisibility when choosing `in0_block_w`.

For large K (e.g., `K_t = 128`), the inner K loop dominates kernel runtime. For small K (e.g., `K_t = 4`), the kernel is often L1-bandwidth-bound on loading the output buffer rather than DRAM-bandwidth-bound on K tiles.

---

## 2. The 32×32 Tile as the Atomic Compute Unit

### 2.1 Why 32×32?

The FPU in each Tensix core is designed around a 32×32 systolic array. One FPU operation consumes three tiles: one 32×32 A tile, one 32×32 B tile, and one 32×32 accumulator C tile. The operation is:

```
C_tile += A_tile × B_tile   (32×32 multiply-accumulate)
```

This is the smallest unit of compute the FPU can perform. You cannot ask the FPU to compute a 16×16 submatrix — it always operates on full 32×32 tiles.

Consequences:

1. **All tensor dimensions must be multiples of 32.** Dimensions that are not multiples of 32 must be padded before entering tile layout. TTNN handles this padding automatically when you call `ttnn.as_tensor(..., layout=ttnn.TILE_LAYOUT)`.

2. **Tile counts `M_t`, `K_t`, `N_t` must be positive integers.** A tensor with `M = 64` gives `M_t = 2`. A tensor with `M = 16` (which is less than 32) cannot exist in tile layout without padding to `M = 32`, giving `M_t = 1` with 16 padding rows.

3. **Compute intensity scales with the cube of tile count.** Doubling both `M_t` and `N_t` (e.g., by doubling sequence length and d_ff) quadruples the FLOPs but only doubles the weight-loading volume. Larger tile counts are more compute-efficient, not less.

### 2.2 Tile Size and L1 Budgeting

Refer to `wormhole_architecture.md` for L1 capacity (1.5 MB per core). The practical L1 budget for a matmul kernel's buffers is roughly 1 MB per core after reserving space for kernel code and metadata.

A double-buffered matmul on one core needs:

```
L1 for A input = 2 × per_core_M × in0_block_w × (tile_size bytes)      (2 = double-buffer depth; in0_block_w K tiles per step)
L1 for B input = 2 × in0_block_w × per_core_N × (tile_size bytes)      (B tiles for one K step × block width)
L1 for C output = per_core_M × per_core_N × (tile_size bytes)

Total ≈ 2 × per_core_M × in0_block_w × 2KB + 2 × in0_block_w × per_core_N × 2KB + per_core_M × per_core_N × 2KB
      (Use 2048 bytes/tile for BF16 and 1088 bytes/tile for BFP8; in the canonical mixed config — BFP8 weights, BF16 activations/output — only the B buffer uses 1088 bytes/tile; all three use 1088 bytes/tile for pure BFP8.)
```

For the example from Section 1.2 (`per_core_M = 1`, `per_core_N = 64`, `in0_block_w = 4`, BF16):

```
A buffer: 2 × 1 × 4 × 2 KB = 16 KB
B buffer: 2 × 4 × 64 × 2 KB = 1024 KB
C buffer: 1 × 64 × 2 KB = 128 KB
Total: ~1168 KB  ← exceeds ~1 MB budget; reduce in0_block_w or per_core_N
```

If `per_core_N = 512` (all N on one core) the C buffer alone would be `1 × 512 × 2 KB = 1024 KB = 1 MB`, leaving no room for A or B buffers. Distributing across more cores (increasing `grid_x`) is necessary.

---

## 3. Output Subblock Constraints

### 3.1 What Are `out_subblock_h` and `out_subblock_w`?

The output subblock is a rectangular block of output tiles that a single TRISC2 packing thread accumulates in the FPU's register file before writing back to L1. It is defined by two parameters:

- **`out_subblock_h`**: The height of the output subblock in tiles (number of output tile rows held in registers simultaneously).
- **`out_subblock_w`**: The width of the output subblock in tiles (number of output tile columns held in registers simultaneously).

Together, `out_subblock_h × out_subblock_w` tiles are kept in the FPU register file and accumulated across the entire K loop before being written to the L1 output buffer. Only after all `K_t` partial sums have been accumulated into this subblock does TRISC2 write the completed tiles to L1.

> **Note:** The packer iterates over `(per_core_M / out_subblock_h)` subblock row-groups and `(per_core_N / out_subblock_w)` subblock column-groups. Both divisions must produce exact integers — this is the source of the divisibility requirements described in Section 3.3. Chapters 3, 4, and 7 rely on this loop structure when reasoning about packer scheduling and output write patterns.

This is why `out_subblock_h` and `out_subblock_w` matter: **the more tiles you accumulate in registers before writing back, the fewer L1 write transactions you issue, and the less the output write latency interrupts the FPU's multiply-accumulate pipeline.**

### 3.2 The Register File Constraint

The FPU register file on Tensix has a fixed size. The constraint is:

```
out_subblock_h × out_subblock_w ≤ 8
```

This is a hard limit. Exceeding it causes a kernel error (TTNN will raise an exception at program config validation time).

Common valid `(out_subblock_h, out_subblock_w)` pairs:

| out_subblock_h | out_subblock_w | Product | Valid? |
|:--------------:|:--------------:|:-------:|:------:|
| 1 | 1 | 1 | Yes |
| 1 | 2 | 2 | Yes |
| 1 | 4 | 4 | Yes |
| 1 | 8 | 8 | Yes |
| 2 | 1 | 2 | Yes |
| 2 | 2 | 4 | Yes |
| 2 | 4 | 8 | Yes |
| 4 | 1 | 4 | Yes |
| 4 | 2 | 8 | Yes |
| 8 | 1 | 8 | Yes |
| 2 | 5 | 10 | **No** — exceeds 8 |
| 3 | 3 | 9 | **No** — exceeds 8 |

### 3.3 Divisibility Constraints

In addition to the register file constraint, the subblock dimensions must divide evenly into the per-core output block:

```
per_core_M must be divisible by out_subblock_h
per_core_N must be divisible by out_subblock_w
```

If these do not divide evenly, TTNN will either raise a validation error or fall back to `out_subblock_h = 1, out_subblock_w = 1` (which is always valid but suboptimal).

For example, with `per_core_M = 4` and `per_core_N = 8`:
- `out_subblock_h = 2, out_subblock_w = 4` → product = 8 ≤ 8; 4 divisible by 2; 8 divisible by 4. **Valid and efficient.**
- `out_subblock_h = 4, out_subblock_w = 2` → product = 8 ≤ 8; 4 divisible by 4; 8 divisible by 2. **Valid and efficient.**
- `out_subblock_h = 3, out_subblock_w = 2` → product = 6 ≤ 8, but 4 is not divisible by 3. **Invalid.**

### 3.4 Choosing Subblock Dimensions for Performance

The goal is to maximize `out_subblock_h × out_subblock_w` (maximize register utilization) subject to the product ≤ 8 and the divisibility constraints.

General heuristics:

1. **Prefer wider subblocks over taller ones** when `per_core_N >> per_core_M`. Wide subblocks allow the FPU to accumulate more N-dimension work per register file flush, which aligns with the natural access pattern of weight matrices (wide in N).

2. **Use `out_subblock_w ≤ floor(8 / out_subblock_h)` as the general safe upper bound** (enforcing the register file constraint from Section 3.2). The shortcut `out_subblock_w = per_core_N` when `per_core_N ≤ 8` is only valid when `out_subblock_h = 1` (i.e., when `per_core_M = 1` or when the product constraint forces `out_subblock_h = 1`). When `out_subblock_h ≥ 2`, the maximum safe `out_subblock_w` is `floor(8 / out_subblock_h)` — not `per_core_N`. For example, with `out_subblock_h = 2`, the maximum safe `out_subblock_w` is 4, regardless of `per_core_N`. Setting `out_subblock_w = per_core_N` unconditionally when `out_subblock_h > 1` will exceed the product limit of 8 for any `per_core_N ≥ 5` and cause a TTNN validation error.

3. **Avoid `out_subblock_h = out_subblock_w = 1` unless forced**. This is the minimum valid configuration; the FPU writes back to L1 after every single tile accumulation, causing frequent L1 write stalls.

```python
import ttnn

# Example: per_core_M=2, per_core_N=4 → out_subblock_h=2, out_subblock_w=4 (product=8, max efficiency)
program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 2),  # (x, y) — note: x first in this API
    in0_block_w=4,                          # number of K tiles processed per inner loop step
    out_subblock_h=2,                       # 2 output tile rows accumulated in registers
    out_subblock_w=4,                       # 4 output tile cols accumulated in registers
    per_core_M=2,                           # total output tile rows per core
    per_core_N=4,                           # total output tile cols per core
    transpose_mcast=False,
    fused_activation=None,
)
```

> **Tip:** When you are unsure what subblock dimensions to use, start with `out_subblock_h=1, out_subblock_w=min(per_core_N, 8)` and verify it does not exceed the register file limit. Then try larger `out_subblock_h` values to see if performance improves. Always re-profile — the interaction between subblock size, K loop length, and DRAM latency means theoretical maxima do not always translate to measured speedups.

---

## 4. Program Config Selection

### 4.1 `MatmulMultiCoreReuseMultiCastProgramConfig`

This config is for **large matmuls where the weight matrix (B) can be multicast across a row or column of cores**. It is the primary config for MoE expert FFN computations where the weight matrix is large and static per forward pass.

Key characteristics:
- Weight tiles are loaded once from DRAM and **multicast via the NoC** to all cores in the relevant row or column of the grid.
- Each row of cores shares the same B tiles; each column of cores shares the same A tiles.
- Requires a minimum 2×2 core grid (both M and N dimensions ≥ 2 cores); otherwise multicast degenerates to unicast and you should use `MatmulMultiCoreProgramConfig` instead.
- Works best when `K_t` is large (the weight matrix is tall in K), because multicast amortizes the DRAM read cost across many cores.

```python
import ttnn

# MatmulMultiCoreReuseMultiCastProgramConfig: appropriate for
# large expert weight matmuls (e.g., d_model=4096, d_ff=16384)
program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # (grid_x, grid_y) — x first
    in0_block_w=4,        # process 4 K tiles per inner loop step; trades L1 for compute overlap
    out_subblock_h=1,     # 1 output tile row in registers (M is small here)
    out_subblock_w=4,     # 4 output tile cols in registers (maximize width utilization)
    per_core_M=1,         # each core computes 1 tile row of output (M_t=4, grid_y=4)
    per_core_N=16,        # each core computes 16 tile cols of output (N_t=128, grid_x=8)
    transpose_mcast=False,
    fused_activation=None,
)

output = ttnn.matmul(
    activation,    # [batch_tokens, d_model], tile layout
    weight,        # [d_model, d_ff], tile layout
    program_config=program_config,
    dtype=ttnn.bfloat16,
)
```

**When to use:**
- `M_t` × `N_t` is large enough to fill a minimum 2×2 core grid (both M and N dimensions ≥ 2 cores)
- The weight matrix is reused across multiple tokens (it is the same for all tokens in the expert batch)
- `K_t` is large (K ≥ 1024, i.e., `K_t ≥ 32`), so the multicast amortization is worth the setup cost

### 4.2 `MatmulMultiCoreProgramConfig`

This config is for **small or irregular matmuls where multicasting would provide little benefit** or where the tensor shapes do not fit the multicast grid pattern.

Key characteristics:
- Each core independently loads its own slice of both A and B from DRAM (or L1). No multicast.
- Simpler scheduling, lower setup overhead.
- More flexible in accepting irregular shapes that would not align with multicast row/column patterns.
- Better for small M (e.g., single-token inference where `M_t = 1`) or narrow N.

```python
import ttnn

# MatmulMultiCoreProgramConfig: appropriate for small batches
# (e.g., single-token decode, M_t=1) or irregular shapes
program_config = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 1),  # (grid_x, grid_y); 8 cores in a single row
    in0_block_w=4,
    out_subblock_h=1,
    out_subblock_w=8,     # per_core_N=8, out_subblock_w=8; saturates register file width
    per_core_M=1,
    per_core_N=8,
)

output = ttnn.matmul(
    activation,    # [1, d_model], single token
    weight,        # [d_model, d_ff]
    program_config=program_config,
    dtype=ttnn.bfloat16,
)
```

**When to use:**
- `M_t` is 1 or very small (decode-time single-token batches)
- The grid is effectively 1D (a single row or column)
- Weight matrices are small enough that per-core DRAM reads are not a bottleneck
- You need a quick fallback config while tuning — `MatmulMultiCoreProgramConfig` is more forgiving of shape irregularities than the multicast variant

### 4.3 Decision Flowchart

```
Is M_t × N_t large enough to fill a minimum 2×2 core grid (both M and N dimensions ≥ 2 cores)?
├── Yes → Is K_t ≥ 32 (K ≥ 1024)?
│         ├── Yes → Use MatmulMultiCoreReuseMultiCastProgramConfig
│         └── No  → Either config may work; profile both.
│                   Multicast may not amortize for short K.
└── No  → Use MatmulMultiCoreProgramConfig
          (shapes too small for multicast to help)
```

> **Warning:** TTNN's auto-selection heuristic (passing no `program_config` argument) will choose a config based on shape, but the auto-selected config is not always optimal for MoE workloads where token counts are irregular. Always specify the program config explicitly in production code and verify performance with the TTNN profiler.

---

## 5. Worked Example: Expert FFN Matmul Configuration

To tie everything together, here is a complete configuration example for the first linear layer of a MoE expert FFN:

**Setup:**
- `d_model = 4096`, `d_ff = 16384`
- `tokens_for_expert = 128` (padded to ensure divisibility)
- Weight dtype: BFP8 (to reduce L1 and DRAM pressure)
- Activation dtype: BF16
- Grid: `4 × 8 = 32 cores`

**Tile counts:**
```
M_t = 128 / 32 = 4
K_t = 4096 / 32 = 128
N_t = 16384 / 32 = 512
```

**Per-core assignment:**
```
per_core_M = M_t / grid_y = 4 / 4 = 1
per_core_N = N_t / grid_x = 512 / 8 = 64
```

**Subblock selection:**
```
per_core_M = 1 → out_subblock_h must divide 1 → out_subblock_h = 1
per_core_N = 64 → out_subblock_w must divide 64 and out_subblock_h × out_subblock_w ≤ 8
             → out_subblock_w = 8 (product = 1×8 = 8, valid)
```

**L1 budget check (per core, with BFP8 B tiles and BF16 A tiles, `in0_block_w=4`):**
```
A buffer (BF16): 2 × per_core_M × in0_block_w × 2048 bytes = 2 × 1 × 4 × 2048 = 16 KB
B buffer (BFP8): 2 × in0_block_w × per_core_N × 1088 bytes = 2 × 4 × 64 × 1088 bytes ≈ 544 KB
C buffer (BF16): per_core_M × per_core_N × 2048 bytes = 1 × 64 × 2048 = 128 KB
Total ≈ 688 KB  ← fits within ~1 MB budget
```

**Final config:**

```python
import ttnn

# Expert FFN first linear layer: [128, 4096] × [4096, 16384] = [128, 16384]
program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # (grid_x=8, grid_y=4) → 32 cores total
    in0_block_w=4,        # process 4 K tiles (128 elements) per inner loop step
    out_subblock_h=1,     # 1 tile row in registers (per_core_M=1, forced)
    out_subblock_w=8,     # 8 tile cols in registers (saturates 1×8=8 limit)
    per_core_M=1,         # each core owns 1 tile row of output (32 rows)
    per_core_N=64,        # each core owns 64 tile cols of output (2048 cols)
    transpose_mcast=False,
    fused_activation=None,  # add ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) to fuse SiLU here
)

expert_output = ttnn.matmul(
    expert_tokens,            # [128, 4096], BF16, TILE_LAYOUT
    expert_weight_gate,       # [4096, 16384], BFP8, TILE_LAYOUT
    program_config=program_config,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,  # use LoFi for BFP8 weight matmuls
        math_approx_mode=True,
        fp32_dest_acc_en=False,                # BF16 accumulation is sufficient for BFP8 weights
        packer_l1_acc=True,                    # pack directly to L1 accumulator for speed
    ),
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

> **Tip:** The `packer_l1_acc=True` flag in `WormholeComputeKernelConfig` enables the L1 accumulator, which keeps partial sums in L1 between inner loop steps rather than writing to DRAM. This significantly reduces DRAM write traffic during the K reduction loop and is almost always beneficial for large-K matmuls. Always profile with and without it — occasionally the extra L1 usage causes capacity issues with very large `per_core_N`.

> **Warning:** Performance numbers in this example are indicative for a specific token count and grid size. Always re-profile when changing `tokens_for_expert`, `d_model`, `d_ff`, dtype, or hardware configuration. The optimal `in0_block_w`, subblock sizes, and grid shape can change significantly with different tensor dimensions.

---

## Summary

| Parameter | Definition | Constraint |
|-----------|-----------|------------|
| `M_t` | `M / 32` — output tile rows total | M must be a multiple of 32 |
| `K_t` | `K / 32` — contraction tile count | K must be a multiple of 32 |
| `N_t` | `N / 32` — output tile cols total | N must be a multiple of 32 |
| `per_core_M` | Output tile rows per core | `per_core_M × grid_y = M_t` |
| `per_core_N` | Output tile cols per core | `per_core_N × grid_x = N_t` |
| `out_subblock_h` | Output tile rows accumulated in FPU registers before L1 write | Divides `per_core_M`; `out_subblock_h × out_subblock_w ≤ 8` |
| `out_subblock_w` | Output tile cols accumulated in FPU registers before L1 write | Divides `per_core_N`; `out_subblock_h × out_subblock_w ≤ 8` |
| `in0_block_w` | K tiles per inner loop step | Must divide `K_t`; limited by A-tile buffer in L1 |

---

## Next Steps

You have now completed Chapter 2. The hardware model (Tensix cores, L1 vs. DRAM, NoC, T3K mesh), programming abstractions (tensor shapes, dtypes, memory configs, op dispatch, program caching), and matmul configuration vocabulary (`M_t`/`K_t`/`N_t`, `out_subblock_h`, `out_subblock_w`, program config selection) are fully established.

Proceed to **Chapter 3 — Expert Tensor Parallelism** to see how these primitives combine to distribute MoE expert computations across the Tensix grid and across the T3K multi-chip mesh.
