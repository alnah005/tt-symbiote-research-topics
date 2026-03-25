# MatMul Program Configs

## Why Explicit Program Configs Are Required

The default `ttnn.matmul` auto-heuristic selects a program config based on tensor shapes alone. For general-purpose tensor computation this is a reasonable starting point, but it cannot match hand-tuned configs for LLM layers because LLM matmuls have irregular aspect ratios that the heuristic is not calibrated for.

Decode workloads have M=1–32 (one token per user in a batch), with very large K and N (hidden dimensions of 4096–8192+ for modern models). The bottleneck is weight bandwidth: the FPU is underutilized because there is almost no reuse of the weight tiles across M rows. Prefill workloads have very large M (hundreds to thousands of tokens) with the same large K and N — here the bottleneck shifts toward compute and activation bandwidth.

These two regimes call for fundamentally different memory access patterns and NoC strategies. The auto-heuristic cannot distinguish them reliably, and does not have access to the layer-level context (e.g., "this is an MLP FF1 during decode") that would let it make the optimal choice. In practice, tt-transformers model implementations explicitly construct and pass a program config for every major linear layer.

---

## `MatmulMultiCoreReuseProgramConfig` (Standard 2D)

This is the baseline 2D tiled matmul config. It distributes output tiles across a 2D core grid: core (i, j) computes a block of output rows and columns determined by `per_core_M` and `per_core_N`. Each core fetches its required activation and weight tiles from DRAM independently.

### Parameters

- **`compute_with_storage_grid_size`**: A `(rows, cols)` tuple specifying the 2D Tensix core grid to use. The total number of cores available is `rows × cols`. Larger grids parallelize more output tiles but must divide the output shape cleanly.

- **`in0_block_w`**: The K-dimension tile block size. Each iteration of the inner loop loads `in0_block_w` tiles along K from the activation matrix (in0) before accumulating into the output. Larger values reduce NoC transaction overhead (fewer round-trips per K sweep) but increase L1 pressure because more tiles must be live simultaneously. Must evenly divide the total K tile count.

- **`out_subblock_h`** and **`out_subblock_w`**: The output tile block that is kept live in the Dst register during accumulation. Must satisfy:
  - `out_subblock_h * out_subblock_w <= 8` in fp16/bf16 mode (Dst holds 8 tiles)
  - `out_subblock_h * out_subblock_w <= 4` in fp32 mode (Dst holds 4 tiles)

  Larger subblocks are strictly better when they fit: more output tiles accumulate in register across K without being written back to L1, reducing pack/unpack traffic. Valid combinations for fp16/bf16 mode include (1×8), (2×4), (4×2), (8×1), (2×2), (1×4), (4×1), (1×1), and others satisfying the constraint.

- **`per_core_M`** and **`per_core_N`**: The number of output tile rows and columns assigned to each core. Together with the grid size, these determine total output coverage. `per_core_M * grid_rows` must equal the total M tile count; `per_core_N * grid_cols` must equal the total N tile count.

### Use Case

`MatmulMultiCoreReuseProgramConfig` is the correct default for square-ish matmuls and for any workload where M, K, and N are all moderate and no specialized sharding layout is already in place upstream. It is the baseline to compare all other configs against when profiling.

---

## `MatmulMultiCoreReuseMultiCastProgramConfig` (Multicast)

In the standard 2D config, each core fetches its weight tiles from DRAM independently. For large-M workloads (prefill), this creates redundant DRAM reads: every core that processes a different activation row against the same weight column range is re-reading identical weight tiles from DRAM.

The multicast config eliminates this redundancy. The weight matrix is broadcast via the Tenstorrent NoC to all cores in the grid simultaneously, rather than each core fetching independently. One core (or a designated source) reads the weight tiles from DRAM once and multicasts them to every other core in a single NoC transaction. Multiple cores then process different rows of the activation against the same in-flight weight tiles.

### When to Use

Prefill workloads where M is large (many tokens simultaneously) and the weight matrix is significantly smaller than the activation. The weight fits in L1 across all receiving cores, and all cores process different activation rows against the same weight. This eliminates most of the per-core DRAM read overhead for the weight matrix.

### Key Difference from `MatmulMultiCoreReuseProgramConfig`

In `MatmulMultiCoreReuseProgramConfig`, weight tiles are fetched independently by each core from DRAM — O(n_cores) DRAM reads for the weight matrix. In `MatmulMultiCoreReuseMultiCastProgramConfig`, weight tiles are read once from DRAM and multicast via NoC — O(1) DRAM reads for the weight matrix, with NoC broadcast delivering the tiles to all cores in parallel.

For prefill with large M, the saving is substantial: if 64 cores would each independently fetch a 512 MB weight matrix, multicast cuts that to one fetch plus 63 NoC deliveries.

---

## `MatmulMultiCoreReuseMultiCast1DProgramConfig` (1D Sharded)

This config is designed specifically for 1D-sharded activation tensors — typically tensors that are already height-sharded or width-sharded in L1 from the output of a preceding op. Because the activation shard is already resident in the core's L1, the activation DRAM read is eliminated entirely. Only the weight DRAM read (or multicast) remains.

### When to Use

Fused MLP layers where FF1's output is already height-sharded across cores and a fused activation (SiLU or GELU) is applied before the FF2 matmul. Because the FF1 output shard sits in L1, the 1D config feeds it directly into FF2 compute without a DRAM round-trip.

### Fused Activation

The `fused_activation` parameter applies an elementwise activation function inside the Packer unit during the matmul, without a separate op dispatch. For example:

```python
fused_activation=(ttnn.UnaryOpType.SILU, True)
```

This fuses the SiLU activation into the matmul output packing step, saving one full op dispatch and one L1 read/write cycle. The second element of the tuple enables the fast approximation where available.

---

## `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (DRAM-Sharded)

This config addresses the primary bottleneck in decode: weight bandwidth. In decode, M is small (1–32 tokens), so the FPU finishes each output tile quickly. The bottleneck is the time spent reading the weight matrix from DRAM. For example, a 4096×16384 BFP8 weight matrix is approximately 68 MB, while the activation batch is tiny (32 tokens × 4096 hidden = 256 KB in BF16) — the matmul is entirely bandwidth-bound on the weight read.

The DRAM-sharded config addresses this by distributing weight columns directly across DRAM banks so that each core reads exclusively from its own DRAM bank partition. All banks are accessed in parallel, delivering maximum aggregate DRAM bandwidth with no inter-bank contention. Activation rows are height-sharded in L1.

For the full per-bank mechanism, shape constraint (N-divisibility requirement), and throughput comparison table, see [L1 Sharding for MatMul — DRAM-Sharded Matmul](l1_sharding_for_matmul.md#dram-sharded-matmul-decode).

---

## Key Takeaways

- The Dst register constraint — `out_subblock_h * out_subblock_w <= 8` in fp16/bf16 mode, `<= 4` in fp32 mode — is a hard limit. Always maximize subblock size within this limit for the best register reuse.
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` is the first config to reach for in decode (small M, large N): it maximizes DRAM bank parallelism and is the primary throughput lever for bandwidth-bound decode layers.
- `MatmulMultiCoreReuseMultiCastProgramConfig` is the first config to reach for in prefill (large M): multicasting the weight matrix eliminates per-core redundant DRAM reads.
- `in0_block_w` is the primary L1 pressure knob: increase it to reduce NoC transactions and improve inner-loop efficiency; decrease it if L1 capacity is exceeded and tiles spill.

## Further Reading

- TTNN matmul documentation: `ttnn.matmul` API reference in the tt-metal repository under `ttnn/cpp/ttnn/operations/matmul/`
- Example layer configs in tt-transformers: `models/demos/llama3/tt/llama_mlp.py` and `models/demos/llama3/tt/llama_attention.py`
- Program config dataclasses: `ttnn/cpp/ttnn/operations/matmul/device/matmul_op.hpp`

---

[Back to Chapter 3 Index](index.md)
