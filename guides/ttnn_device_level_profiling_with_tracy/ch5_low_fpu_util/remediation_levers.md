# Remediation Levers

This file documents the configuration parameters and API calls that address each cause of low `FPU UTIL` identified in [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md). For the CSV signatures that tell you which lever to apply, see [`csv_signatures.md`](./csv_signatures.md).

---

## `compute_with_storage_grid_size` — Core Count Control

Controls how many Tensix cores are allocated for a matmul. This is the primary lever for both Cause 1 (insufficient tile count) and Cause 5 (NoC contention).

**Usage:**

```python
import ttnn

matmul_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(4, 4),   # 4×4 = 16 cores
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=4,
    per_core_N=4,
)

output = ttnn.matmul(a, b, program_config=matmul_config)
```

**Guideline:** Choose `compute_with_storage_grid_size` such that:

```
M_t × N_t / core_count ≥ 4
```

Where `M_t = ⌈M/32⌉`, `N_t = ⌈N/32⌉`, and `core_count = grid_rows × grid_cols`.

**Worked example:** For a matmul with M=128, N=128 (i.e., M_t=4, N_t=4, output_tiles=16):

- Grid (8, 8) = 64 cores → 16/64 = 0.25 tiles/core → well below guideline (Cause 1 active).
- Grid (2, 2) = 4 cores → 16/4 = 4 tiles/core → meets guideline.
- Grid (2, 4) = 8 cores → 16/8 = 2 tiles/core → below guideline (2.0 < 4) — Cause 1 tile starvation active; reduce to 4 cores or fewer.

> **Tip:** There is no single universally optimal grid size. Profile at a few candidate grids (e.g., halving from the default until `FPU UTIL` peaks) and use the grid that maximizes `FPU UTIL` for your target shape.

---

## `math_fidelity` Parameter — Precision vs. Throughput

Controls the number of internal FMA iterations per tile in the TRISC1 math engine. This is the primary lever for Cause 3.

**Usage:**

```python
matmul_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    ...,
    math_fidelity=ttnn.MathFidelity.LoFi,
)
```

**Available values and tradeoffs:**

| Value | FMA iterations per tile | Relative throughput | Use case |
|---|---|---|---|
| `ttnn.MathFidelity.LoFi` | 1× | 1.0 (fastest) | BF16 inference; most transformer ops |
| `ttnn.MathFidelity.HiFi2` | 2× | ~0.5× | When `LoFi` causes measurable accuracy degradation |
| `ttnn.MathFidelity.HiFi4` | 4× | ~0.25× | FP32 accumulation; numerically sensitive ops |

> **Warning:** Lowering fidelity without validating accuracy is risky. Always run an accuracy check (e.g., compare top-1 accuracy or perplexity against a FP32 reference) when changing math fidelity. Start with `HiFi2` if `LoFi` shows degradation; move to `LoFi` only when accuracy is confirmed to be acceptable.

---

## Data Format Selection — `ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.float32`

Selecting the appropriate data format addresses Cause 2. Format affects FPU throughput, memory footprint, and NoC traffic simultaneously.

**Format comparison for Wormhole B0:**

| Format | Bits/element | FPU throughput (relative) | NoC traffic (relative) | Use case |
|---|---|---|---|---|
| `ttnn.float32` | 32 | 0.5× BF16 | 2× BF16 | Optimizer state; precision-critical ops |
| `ttnn.bfloat16` | 16 | 1.0× (baseline) | 1× (baseline) | Activations and weights in standard inference/training |
| `ttnn.bfloat8_b` | 8 | 1.0× BF16 (same TRISC1 FMA rate as BF16; gain is reduced TRISC0/NCRISC load — unpacker reads half the bytes, cutting bandwidth stalls) | 0.5× BF16 | Weight tensors in memory-bandwidth-limited ops |

**Usage — explicit typecast:**

```python
x_bf16 = ttnn.typecast(x_fp32, ttnn.bfloat16)
w_bf8  = ttnn.typecast(w_bf16, ttnn.bfloat8_b)
```

**Usage — specify output dtype in op config:**

```python
output = ttnn.matmul(
    a, b,
    dtype=ttnn.bfloat16,
    program_config=matmul_config,
)
```

> **Note:** `ttnn.bfloat8_b` is a block-float format where a shared exponent is stored per block of 8 elements. It is well-suited for weight tensors in inference where the dynamic range per block is small and predictable. It is generally not suitable for accumulator tensors or intermediate activations with high dynamic range.

---

## Sharded Memory Configs — Eliminating DRAM Reads for Stationary Weights

Sharded memory keeps tensor tiles resident in L1 rather than DRAM, eliminating TRISC0 stalls caused by DRAM fetch latency (Cause 4). It also reduces per-call NoC traffic, which helps with Cause 5.

**Key `TensorMemoryLayout` options:**

| Layout | Description | Best for |
|---|---|---|
| `HEIGHT_SHARDED` | Each core holds a horizontal slab (contiguous rows) of the tensor | Activations distributed across cores by sequence position |
| `BLOCK_SHARDED` | Each core holds a rectangular block of the tensor | Weight matrices in large matmuls; 2D output grids |
| `WIDTH_SHARDED` | Each core holds a vertical slab (contiguous columns) | Less common; useful for column-parallel patterns |

**Usage:**

```python
shard_spec = ttnn.ShardSpec(
    core_grid,
    [shard_height, shard_width],
    ttnn.ShardOrientation.ROW_MAJOR,
)
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    shard_spec,
)

# Pre-place weight tensor in sharded L1
w_sharded = ttnn.to_device(w, device, memory_config=memory_config)

# Matmul with sharded weight
output = ttnn.matmul(activation, w_sharded, program_config=matmul_config)
```

> **Warning:** Sharded layouts require that tensor shapes are compatible with the chosen shard dimensions. Mismatched shapes cause runtime errors. Validate that `shard_height × core_rows = M` and `shard_width × core_cols = N` (for `BLOCK_SHARDED`) before running. Also verify total L1 usage: sharding increases per-core L1 occupancy and can cause an L1 overflow if the tensor is large.

---

## `ttnn.enable_program_cache()` — Eliminating Recompilation

Enables the program cache, which prevents repeated kernel compilation on subsequent calls with the same parameters. This is the fix for Cause 6.

**Usage:**

```python
ttnn.enable_program_cache(device)

# Warmup pass (cache miss, not timed)
_ = ttnn.matmul(a, b, program_config=config)

# All subsequent calls hit the cache (no recompilation)
for _ in range(n_iters):
    output = ttnn.matmul(a, b, program_config=config)
```

**Scope:** The cache is keyed on op type, tensor shapes, data formats, device configuration, and program config parameters. Changing any of these fields — including `math_fidelity` or `compute_with_storage_grid_size` — produces a new cache key and triggers recompilation.

> **Tip:** In benchmarking scripts, always call `ttnn.enable_program_cache(device)` immediately after device initialization and before any timed regions. Pair it with an explicit warmup pass to ensure the first timed call uses a cached binary.

---

## `in0_block_w` — Double-Buffering Depth for Unpacker Prefetch

Controls how many input tiles TRISC0 prefetches into the double-buffer ahead of the math engine. Increasing this value hides DRAM fetch latency and reduces TRISC0/TRISC1 pipeline stalls (Cause 4).

**Usage:**

```python
matmul_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(4, 4),
    in0_block_w=4,      # Prefetch 4 tiles of input A per block
    out_subblock_h=1,
    out_subblock_w=1,
    per_core_M=4,
    per_core_N=4,
)
```

**Guideline:**

- `in0_block_w=1`: minimal buffering; suitable when L1 is nearly full.
- `in0_block_w=2`: standard double-buffering; reduces stalls without large L1 cost.
- `in0_block_w=4` or higher: aggressive prefetch; most effective when DRAM latency is the dominant bottleneck; requires proportionally more L1.

The L1 cost of the double buffer is approximately:

```
extra_L1_bytes = per_core_M × in0_block_w × 32 × 32 × bytes_per_element
```

> **Warning:** Increasing `in0_block_w` beyond what L1 can accommodate causes a hard L1 overflow error at runtime. Always compute the expected L1 footprint before increasing this parameter. If L1 is already near capacity (approaching the ~1.5 MB per-core limit), prefer sharded memory layouts (described above) over increasing `in0_block_w`.

---

**Next:** [Chapter 6 — Host Dispatch Overhead vs. Device Kernel Time](../ch6_host_dispatch_overhead/index.md)
