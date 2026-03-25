# L1 Sharding for MatMul

## When L1 Sharding Helps

The default memory layout for an activation tensor is DRAM-interleaved: each 32×32 tile is placed on a different DRAM bank in round-robin order. When a core needs a tile, it issues a DRAM read, waits for the tile to arrive in L1, and then proceeds to compute. For bandwidth-bound workloads, this wait is the dominant cost.

L1 sharding changes the activation tensor's home location from DRAM to L1. Instead of reading activation tiles from DRAM at compute time, the activation tiles are already resident in each core's local SRAM before the matmul kernel begins. The DRAM read for the activation is eliminated. Only the weight DRAM read remains as a bandwidth bottleneck.

This matters most in two scenarios:

- **Decode**: M is small (1–32 tokens), so the full activation batch is tiny relative to L1 capacity distributed across cores. The activation can be fully resident in L1 with headroom to spare for weight tiles.
- **Prefill**: M is large but the shard per core may still fit in L1 when divided across enough cores. Whether L1 sharding helps in prefill depends on whether the activation shard fits in L1 alongside the working set for the weight tiles.

L1 sharding also enables op fusion: when op A produces a height-sharded output and op B consumes a height-sharded input, the output of A can be left in L1 as the input to B without any DRAM round-trip between ops. This is the primary source of end-to-end latency improvement in fused MLP and fused attention implementations.

---

## Height Sharding Activations for 1D Matmul

The most common L1 sharding pattern in tt-transformers decode is height sharding along the batch dimension.

An activation tensor of shape `[batch, seq=1, hidden]` (decode: one token per user) is distributed so that each core holds `batch // n_cores` rows. For a batch of 32 users and a 4×8 core grid (32 cores), each core holds exactly one activation row — a single `[1, hidden]` slice in L1.

The weight matrix is broadcast via NoC multicast from one DRAM-resident source to all cores simultaneously. Each core receives the same weight tiles, multiplies them against its local activation row, and produces its portion of the output. Because all cores receive the weight via multicast (one DRAM read, one broadcast), DRAM read traffic for the weight is O(1) regardless of core count.

Each core produces a full output row for its batch slice. No inter-core communication is needed during the matmul because each core computes an independent output row. The output can remain height-sharded in L1 for the next op.

This pattern is used in tt-transformers for:

- Decode attention output projection: `[batch, 1, n_heads * head_dim] @ [n_heads * head_dim, hidden]`
- Decode MLP FF2: `[batch, 1, ff_dim] @ [ff_dim, hidden]`

---

## Block Sharding for 2D Matmul

Block sharding distributes both activation rows and weight columns across a 2D core grid. Core at position `(i, j)` in the grid holds:

- Activation rows `i * tile_H` through `(i+1) * tile_H - 1` (the height shard for row block i)
- Responsibility for output columns `j * tile_W` through `(j+1) * tile_W - 1` (the width shard for column block j)

The core fetches the corresponding weight column range from DRAM (or receives it via multicast) and computes its output block locally. No activation data needs to move between cores during the matmul.

Block sharding is the preferred layout when both M and N are large and a 2D core grid is available. It distributes both the activation memory footprint and the output computation across the full grid, achieving higher parallelism than height-only sharding.

### Output Sharding Compatibility

After a block-sharded matmul, the output is block-sharded in L1. If the next op expects a different shard format — for example, a height-sharded input for a subsequent 1D matmul — a Reduce-Scatter or re-shard op is required. Re-sharding incurs NoC traffic and a short latency penalty.

To minimize re-sharding overhead, design the op sequence so that consecutive ops use compatible shard layouts. In practice, this means:

- Chain block-sharded ops together when possible
- Place re-shard ops at natural phase boundaries (e.g., between prefill and decode, or between attention and MLP)
- Use `MatmulMultiCoreReuseMultiCast1DProgramConfig` for subsequent ops if the upstream op produces a height-sharded output naturally

---

## DRAM-Sharded Matmul (Decode)

DRAM-sharded is the decode-optimal configuration for all large linear layers. It combines height-sharded activations in L1 with weight columns distributed directly across DRAM banks, achieving maximum DRAM bandwidth utilization.

### How It Works

Weight columns are not interleaved across DRAM banks in the usual round-robin fashion. Instead, weight columns are sharded contiguously: all column tiles belonging to core `k`'s assigned output range are placed in DRAM bank `k`. When core `k` begins its matmul, it reads exclusively from DRAM bank `k` — no other core reads from that bank simultaneously. All cores read from their respective banks in parallel.

The result is that all DRAM banks are utilized simultaneously with no bank contention among cores. This is the highest achievable DRAM throughput: aggregate bandwidth = single-bank bandwidth × number of banks.

Activation rows are height-sharded in L1 as described above. Each core holds one or a few activation rows and multiplies them against the local-bank weight slice.

### Which Layers Use DRAM-Sharded Decode

This is the decode-optimal pattern for all large linear layers in tt-transformers:

- **QKV projection (decode)**: DRAM-sharded weight, height-sharded activation in L1
- **MLP FF1/FF2 (decode)**: same
- **Output projection (decode)**: same

### Throughput Comparison

The following numbers are illustrative for Llama 3.1 8B decode on N150. Actual results vary by layer size, batch size, and data type.

| Config | Approx decode throughput |
|---|---|
| Interleaved DRAM matmul | baseline |
| L1-sharded (height) | ~1.5–2x baseline |
| DRAM-sharded | ~2–4x baseline |

The DRAM-sharded improvement is largest when the weight matrix is large relative to DRAM bank width (i.e., large N) and when M is small enough that compute is not the bottleneck. Both conditions hold for all decode-phase linear layers in standard transformer architectures.

### Shape Constraint

For DRAM-sharded to apply, the weight N dimension must be divisible by the product of DRAM bank count and core grid column count; standard transformer hidden dimensions (4096, 8192, etc.) satisfy this constraint in practice.

---

## Key Takeaways

- Height sharding for 1D decode is the default sharding pattern in tt-transformers for attention and MLP linear layers during decode; it eliminates the activation DRAM read and is the prerequisite for DRAM-sharded weight fetching.
- DRAM-sharded gives the best decode throughput for bandwidth-bound layers (large N, small M) by placing each core's required weight columns in its adjacent DRAM bank, eliminating all inter-bank contention.
- Block sharding is used when 2D parallelism is needed and M is large enough that height sharding alone does not fully utilize the core grid; it is the preferred layout for prefill when both activation rows and output columns can be cleanly distributed.
- Sharding adds re-sharding overhead at shard boundaries; minimize DRAM round-trips between ops by designing consecutive ops with compatible shard layouts and inserting re-shard steps only at natural phase boundaries.

## Further Reading

- TTNN memory config API: `ttnn.MemoryConfig`, `ttnn.TensorMemoryLayout`, `ttnn.BufferType` in the TTNN Python API reference
- Shard spec construction: `ttnn.ShardSpec` and `ttnn.create_sharded_memory_config` utility function
- tt-transformers decode attention sharding: `models/demos/llama3/tt/llama_attention.py`, decode path configuration
- DRAM-sharded config construction: search for `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` in `models/demos/llama3/tt/`
- Chapter 3, matmul program configs: [`matmul_program_configs.md`](matmul_program_configs.md)

---

[Back to Chapter 3 Index](index.md)
