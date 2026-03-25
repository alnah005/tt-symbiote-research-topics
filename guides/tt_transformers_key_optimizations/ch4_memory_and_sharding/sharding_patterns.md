# Sharding Patterns

Sharding determines where each tensor's tiles live — in which core's L1, or in which DRAM bank — and how they are accessed during compute. Choosing the right sharding pattern for each op is not just about that op in isolation: the output shard layout of op N must be compatible with the input shard layout expected by op N+1. When they match, the output stays in L1 and the next op reads directly from it. When they do not match, a re-shard step moves tiles across the NoC to the new layout, adding latency. The four patterns below cover the full range used in tt-transformers.

---

## Height Sharding — Tensor Parallelism for Attention Prefill

In height sharding, the activation tensor is split along the batch × sequence dimension. Core k holds rows `k × shard_height` through `(k+1) × shard_height - 1`. Each core computes attention (or a linear projection) for its slice of rows independently, and results are gathered across cores after the op.

**What it enables:** Height sharding scales the sequence dimension across the full core grid. For prefill — where the sequence length is hundreds or thousands of tokens — a single core's L1 cannot hold the full Q, K, and V tensors, but the per-core shard does fit.

**Corresponding config:** `MatmulMultiCoreReuseMultiCast1DProgramConfig` with height-sharded `in0`. The activation shard is already resident in each core's L1 before the kernel starts, eliminating the activation DRAM read. Weight tiles are broadcast via NoC multicast from a single DRAM source.

**Typical layers:**
- QKV projection during prefill: `[batch, seq, hidden]` height-sharded, weight multicast
- Attention output projection during prefill: height-sharded activation from the attention op feeds directly into the next linear layer
- Decode attention output projection: the output of `scaled_dot_product_attention_decode` is height-sharded across the batch dimension; feeding it into a height-sharded-input linear avoids a re-shard

**L1 continuity:** When consecutive ops both consume and produce height-sharded tensors with the same shard spec, no re-sharding is needed between them. This is the primary op-fusion benefit: the output of op N remains in L1 as the input to op N+1 without touching DRAM.

---

## Block Sharding — Intermediate Activations

Block sharding distributes a tensor in 2D: core `(i, j)` holds the tile block at row group `i`, column group `j`. This maps activation rows to grid rows and activation columns to grid columns, spreading the full tensor footprint across the entire core grid.

**What it enables:** A large intermediate activation tensor — such as the output of a wide linear projection in a large-batch prefill — may be too large to hold in one core's L1, but fits easily when divided across a 4×8 or 8×8 core grid in 2D tiles. Block sharding avoids the alternative: writing the full intermediate to DRAM and re-reading it for the next op.

**Corresponding config:** `MatmulMultiCoreReuseProgramConfig` (standard 2D) or `MatmulMultiCoreReuseMultiCastProgramConfig` when the weight is shared across row blocks. The output of the matmul is block-sharded in L1 matching the core grid layout.

**Typical layers:**
- Large intermediate activations in prefill MLP (hidden × ff_dim, where ff_dim = 4 × hidden for standard transformers)
- Any tensor that cannot fit height-sharded in L1 but must stay off DRAM for the subsequent elementwise op (e.g., SwiGLU gate computation between FF1 and FF2)

**Re-sharding consideration:** Block-sharded output is not compatible with a height-sharded input without a re-shard step. If the next op expects height-sharded input (e.g., a 1D decode matmul following a prefill layer in a chunked-prefill pipeline), a Reduce-Scatter or explicit re-shard op is inserted at the phase boundary. The cost of this re-shard is paid once at the boundary rather than per-element across every K-block.

---

## Width-Sharded / DRAM-Sharded — Weight Tensors for Decode

During decode, M=1 per sequence and the batch is typically 32. The bottleneck is not compute but weight bandwidth: reading the full weight matrix once per decode step dominates the wall-clock time. DRAM-sharded matmul addresses this by giving each core its own DRAM bank, eliminating inter-bank contention.

**Mechanism:** Weight columns are distributed contiguously across DRAM banks — not interleaved round-robin as in the default layout. Core k owns the column tiles for its output range, stored in DRAM bank k. Activation rows are height-sharded in L1 (one row per core for batch=32 on a 32-core grid). Each core reads exclusively from its own DRAM bank while computing its weight column range. All banks are active simultaneously; aggregate bandwidth scales linearly with bank count.

Activations (shape `[1, batch, K]`, e.g., M=32, K=4096, ~256 KB in BF16) are broadcast to all cores via NoC multicast, so each core has the full activation in L1 before starting the weight-column matmul.

**Shape constraint:** N must be divisible by the product of DRAM bank count and core grid column count. Standard transformer hidden dimensions (4096, 8192) satisfy this constraint in practice. See [l1_sharding_for_matmul.md](../ch3_matmul_optimizations/l1_sharding_for_matmul.md) for exact derivation.

**Corresponding config:** `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`

**Typical layers:** All large linear layers in the decode path:
- QKV projection: `[batch, 1, hidden] @ [hidden, 3 × n_heads × head_dim]`
- MLP FF1 and FF3: `[batch, 1, hidden] @ [hidden, ff_dim]`
- MLP FF2: `[batch, 1, ff_dim] @ [ff_dim, hidden]`
- Output projection (`wo`): `[batch, 1, n_heads × head_dim] @ [n_heads × head_dim, hidden]`

---

## 1D Ring-Sharded — K-Dimension Reduction

In 1D ring-sharded layout, the input tensor is partitioned along the K (reduction) dimension. Core k holds columns `k × shard_K` through `(k+1) × shard_K - 1` of the input. Each core computes a partial dot product against its K slice of the weight, then an all-reduce aggregates the partial sums across cores.

**What it enables:** When K is large — for example, in a very wide hidden dimension or in a row-parallel tensor-parallel variant — distributing the K dimension across cores parallelizes the reduction. Each core contributes a partial result; a reduce-scatter (implemented via `ttnn.reduce_scatter` using Ethernet for multi-chip or NoC for intra-chip) produces the final distributed output.

**Corresponding config:** [INFERRED] Typically handled by `MatmulMultiCoreReuseMultiCastProgramConfig` with an explicit reduce-scatter step, or by fused collective matmul kernels where the reduction is pipelined with the matmul. The exact program config class depends on whether the K-reduction is intra-chip (NoC) or inter-chip (Ethernet CCL).

**Typical layers:** K-parallel reductions in tensor-parallel MLP and attention output projections, particularly in large-model configurations where both the weight K dimension and the core/device count are large. In multi-device TP, this pattern is the foundation of row-parallel linear layers: each device computes a partial `[batch, partial_K] @ [partial_K, N]` matmul and `ttnn.reduce_scatter` combines and distributes the results.

---

## Keeping Activations in L1 Across Layers

The four patterns above are most valuable when chained. The core principle: design the sharding spec of each op's output so that the next op can consume it from L1 without any reformatting or DRAM round-trip.

**The condition for L1 continuity:** Op N's output shard spec (layout, shard shape, core grid) must exactly match op N+1's expected input shard spec. If they match, TTNN can fuse or pipeline the two ops with the output CB of op N serving as the input CB of op N+1, all within L1.

**In practice for a decode MLP block:**
1. FF1 produces a height-sharded output in L1 (DRAM-sharded matmul, output in L1 per core)
2. SiLU activation is applied elementwise on the height-sharded L1 tensor — no layout change, no DRAM access
3. The gate tensor from FF3 (also height-sharded) is multiplied elementwise in L1
4. FF2 consumes the height-sharded result directly, using the same DRAM-sharded config

Each of steps 2–4 reads from and writes to L1 only. The only DRAM accesses in the entire MLP block are the weight reads for FF1, FF2, and FF3 — not the activations.

When a layout mismatch is unavoidable (e.g., transitioning from block-sharded prefill output to height-sharded decode input), insert the re-shard at the natural phase boundary rather than between tightly-coupled ops.

---

## Summary Table

| Pattern | Config class | Splits along | Best for | Typical layers |
|---|---|---|---|---|
| Height-sharded | `MatmulMultiCoreReuseMultiCast1DProgramConfig` | Batch × sequence rows | Prefill attention, decode batch | QKV proj (prefill), attn output proj |
| Block-sharded | `MatmulMultiCoreReuseProgramConfig` | 2D: rows and columns | Large intermediate tensors (prefill) | MLP intermediates, wide projections |
| DRAM-sharded (width) | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | Weight columns across DRAM banks | Decode linear layers (bandwidth-bound) | All decode linear layers |
| 1D ring-sharded | [INFERRED] K-parallel matmul + reduce-scatter | K (reduction) dimension | Large-K reductions, TP row-parallel | TP output projection, row-parallel MLP |

---

## Key Takeaways

- Height sharding keeps the activation batch in L1 and multicasts weights via NoC; it is the foundation for L1-sharded prefill and the enabling condition for DRAM-sharded decode.
- DRAM-sharded is the decode-optimal pattern: each core reads from its own dedicated DRAM bank in parallel, achieving maximum aggregate DRAM bandwidth with no bank contention; activation multicast from L1 adds negligible overhead.
- Block sharding handles large intermediate tensors that exceed per-core L1 when held whole by distributing them 2D across the core grid; it avoids DRAM spill at the cost of a re-shard step when transitioning to a 1D-sharded downstream op.
- 1D ring-sharded distributes the K reduction across cores and aggregates via reduce-scatter (`ttnn.reduce_scatter`); it is the intra-chip or inter-chip complement to column-parallel and row-parallel tensor parallelism.
- The highest-value optimization is matching output shard specs to input shard specs across consecutive ops so activations stay in L1 end-to-end through the attention block and MLP block with no intervening DRAM round-trips.

---

## Further Reading

- L1 sharding API: `ttnn.MemoryConfig`, `ttnn.TensorMemoryLayout`, `ttnn.ShardSpec`, `ttnn.create_sharded_memory_config` in the TTNN Python reference
- DRAM-sharded config construction and shape constraint: [l1_sharding_for_matmul.md](../ch3_matmul_optimizations/l1_sharding_for_matmul.md)
- Height-sharded 1D config: [matmul_program_configs.md](../ch3_matmul_optimizations/matmul_program_configs.md)
- Decode layer sharding in Llama 3: `models/demos/llama3/tt/llama_attention.py` and `models/demos/llama3/tt/llama_mlp.py`
- Multi-device tensor parallelism and row/column-parallel patterns: Chapter 5

---

[Back to Chapter 4 Index](index.md) | [Previous: Tensix Memory Hierarchy](tensix_memory_hierarchy.md)
