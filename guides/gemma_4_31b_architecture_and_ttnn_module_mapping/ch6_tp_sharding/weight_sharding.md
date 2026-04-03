# Weight Sharding for Tensor-Parallel Execution

## Sharding Principles

Tensor-parallel inference on T3K uses two complementary sharding patterns for
linear projections:

1. **Column-parallel (shard output dim):** The weight matrix `[in_features,
   out_features]` is split along `out_features` across 8 devices. Each device
   computes a slice of the output. The input activation is replicated (or
   already available) on all devices.

2. **Row-parallel (shard input dim):** The weight matrix `[in_features,
   out_features]` is split along `in_features` across 8 devices. Each device
   holds a column-sharded input and a row-sharded weight, producing a partial
   output that must be summed across devices via `ttnn.all_reduce`.

The general pattern for a transformer attention + FFN block is:

```text
Input (replicated or gathered)
  │
  ├─ Column-parallel: Q, K, V, gate, up projections
  │    └─ Each device holds a slice of the output
  │
  ├─ Local compute: RoPE, norms, SDPA, activation, elementwise multiply
  │
  ├─ Row-parallel: O, down projections
  │    └─ Each device holds a slice of the input, full output
  │    └─ Followed by ttnn.all_reduce
  │
  └─ Output (replicated across all devices after all-reduce)
```

This pattern ensures that all-reduce is needed only twice per decoder layer
(once after O projection, once after down projection), minimizing CCL overhead.

## Sliding Layer Weight Sharding (TP=8)

All 50 sliding layers use standard TP=8 column/row-parallel sharding.

### Column-Parallel Projections (Shard Output Dim)

| Projection | Full Shape | Shard Dim | Per-Device Shape | Per-Device Bytes (BF16) |
|------------|-----------|-----------|-----------------|------------------------|
| Q | `[5376, 8192]` | dim=-1 | `[5376, 1024]` | 11.0 MB |
| K | `[5376, 4096]` | dim=-1 | `[5376, 512]` | 5.5 MB |
| V | `[5376, 4096]` | dim=-1 | `[5376, 512]` | 5.5 MB |
| Gate | `[5376, 21504]` | dim=-1 | `[5376, 2688]` | 28.9 MB |
| Up | `[5376, 21504]` | dim=-1 | `[5376, 2688]` | 28.9 MB |

The Q weight shard `[5376, 1024]` corresponds to 4 Q heads x 256 head_dim =
1024. The K and V shards `[5376, 512]` correspond to 2 KV heads x 256
head_dim = 512.

### Row-Parallel Projections (Shard Input Dim)

| Projection | Full Shape | Shard Dim | Per-Device Shape | Per-Device Bytes (BF16) |
|------------|-----------|-----------|-----------------|------------------------|
| O | `[8192, 5376]` | dim=-2 | `[1024, 5376]` | 11.0 MB |
| Down | `[21504, 5376]` | dim=-2 | `[2688, 5376]` | 28.9 MB |

The O weight shard `[1024, 5376]` takes the 4-head attention output from
this device and projects to hidden_size. The partial results from all 8
devices are summed via `ttnn.all_reduce` to produce the correct full output.

### Per-Device Sliding Layer Weight Total

| Component | Per-Device Bytes (BF16) |
|-----------|------------------------|
| Q | 11.0 MB |
| K | 5.5 MB |
| V | 5.5 MB |
| O | 11.0 MB |
| Gate | 28.9 MB |
| Up | 28.9 MB |
| Down | 28.9 MB |
| Norms (6 total) | ~0.04 MB |
| **Total** | **~119.7 MB** |

Across 50 sliding layers: **~5,985 MB per device** (BF16).
At BFP8 (1 byte/element): **~2,993 MB per device**.

## Global Layer Weight Sharding (TP=8, Replicated KV)

Global layers use the recommended strategy from
[`sharding_strategy_analysis.md`](./sharding_strategy_analysis.md): Q and O
projections are column/row-parallel sharded as usual, but the K projection
weight is **replicated** on all devices (not sharded).

### Column-Parallel Projections

| Projection | Full Shape | Shard Dim | Per-Device Shape | Per-Device Bytes (BF16) |
|------------|-----------|-----------|-----------------|------------------------|
| Q | `[5376, 16384]` | dim=-1 | `[5376, 2048]` | 22.0 MB |
| Gate | `[5376, 21504]` | dim=-1 | `[5376, 2688]` | 28.9 MB |
| Up | `[5376, 21504]` | dim=-1 | `[5376, 2688]` | 28.9 MB |

The Q weight shard `[5376, 2048]` corresponds to 4 Q heads x 512 head_dim =
2048.

### Replicated Projection (K=V Shared Weight)

| Projection | Full Shape | Sharding | Per-Device Shape | Per-Device Bytes (BF16) |
|------------|-----------|----------|-----------------|------------------------|
| K (= V) | `[5376, 2048]` | Replicated | `[5376, 2048]` | 22.0 MB |

The K weight is **not sharded** --- each device holds the full `[5376, 2048]`
matrix and computes all 4 KV heads locally. Because K=V sharing is active in
global layers, this single weight serves both K and V paths. No separate V
weight exists.

### Row-Parallel Projections

| Projection | Full Shape | Shard Dim | Per-Device Shape | Per-Device Bytes (BF16) |
|------------|-----------|-----------|-----------------|------------------------|
| O | `[16384, 5376]` | dim=-2 | `[2048, 5376]` | 22.0 MB |
| Down | `[21504, 5376]` | dim=-2 | `[2688, 5376]` | 28.9 MB |

### Per-Device Global Layer Weight Total

| Component | Per-Device Bytes (BF16) |
|-----------|------------------------|
| Q | 22.0 MB |
| K (replicated) | 22.0 MB |
| O | 22.0 MB |
| Gate | 28.9 MB |
| Up | 28.9 MB |
| Down | 28.9 MB |
| Norms (6 total) | ~0.04 MB |
| **Total** | **~152.8 MB** |

Across 10 global layers: **~1,528 MB per device** (BF16).
At BFP8 (1 byte/element): **~764 MB per device**.

## All-Reduce After Row-Parallel Matmuls

Both the O projection and the down projection are row-parallel. After each,
an `ttnn.all_reduce` sums the partial results across all 8 devices to produce
the correct hidden_size output.

```text
Device 0: O_partial_0 = x_local_0 @ W_O_shard_0   shape: [B, 1, 5376]
Device 1: O_partial_1 = x_local_1 @ W_O_shard_1   shape: [B, 1, 5376]
...
Device 7: O_partial_7 = x_local_7 @ W_O_shard_7   shape: [B, 1, 5376]

all_reduce → O_full = sum(O_partial_0 ... O_partial_7)   shape: [B, 1, 5376]
```

The all-reduce payload is `B x 1 x 5376 x 2 = 10,752 bytes` at B=1 (BF16).
This is ~10.5 KB total (~1.3 KB per device contribution), so
the all-reduce is latency-bound rather than bandwidth-bound. At B=32 the
payload is still only ~344 KB, comfortably fitting in L1.

For the T3K linear topology, the all-reduce uses `ttnn.Topology.Linear` (not
Ring, since there is no wrap-around link between device 0 and device 7). The
recommended configuration follows the patterns established in the
[T3K Mesh Device Optimizations guide](../../t3k_mesh_device_optimizations/ch02_ttnn_mesh_api/collective_primitives.md):

```python
output = ttnn.all_reduce(
    partial_output,
    cluster_axis=1,
    mesh_device=mesh_device,
    num_links=1,          # sufficient for small decode tensors
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Linear,
)
```

Per decoder layer, there are exactly **2 all-reduce operations**: one after O
projection, one after down projection. Across 60 layers, this is 120
all-reduce calls per decode step.

## TTNN Linear Module Mapping

### Column-Parallel Projections (Q, K for Sliding, Gate, Up)

These projections have replicated input and sharded output. They map to
`TTNNLinearIReplicatedWColSharded`:

- Weight is sharded on `dim=-1` (output dimension) across the mesh using
  `ttnn.shard_tensor_to_mesh_mapper(device, dim=-1)`.
- Input activation is replicated on all devices (output of the preceding
  all-reduce or embedding lookup).
- No CCL operation after the matmul --- the sharded output is consumed
  locally by subsequent operations (RoPE, norms, SDPA, activation).

```python
# Example: Q projection for sliding layer
q_proj = TTNNLinearIReplicatedWColSharded.from_parameters(
    weight=q_weight,  # [5376, 8192] → sharded to [5376, 1024] per device
)
# forward: q_local = q_proj(hidden_states)  # [B, 1, 1024] per device
```

### Row-Parallel Projections (O, Down)

These projections have sharded input and produce a partial output that must
be reduced. They map to `TTNNLinearIColShardedWRowSharded`:

- Weight is sharded on `dim=-2` (input dimension) across the mesh.
- Input activation is the local shard from the preceding column-parallel
  computation.
- The `forward` method internally performs a reduce-scatter or all-reduce
  after the matmul.

```python
# Example: O projection for sliding layer
o_proj = TTNNLinearIColShardedWRowSharded.from_parameters(
    weight=o_weight,  # [8192, 5376] → sharded to [1024, 5376] per device
)
# forward: o_full = o_proj(attn_out_local)  # includes all-reduce internally
```

The `TTNNLinearIColShardedWRowSharded` class (documented in the
[tt-symbiote built-in modules guide](../../tt_symbiote/ch5_builtin_modules/linear_layers.md))
performs the collective communication as part of its `forward` method, using
`ttnn.experimental.reduce_scatter_minimal_async` with `cluster_axis=1` and
`ttnn.Topology.Linear` (Linear topology, matching the all-reduce, since the
T3K 1x8 mesh has no wrap-around link between device 0 and device 7).

### Replicated K Projection (Global Layers)

The K projection for global layers is replicated, not sharded. This maps to
the base `TTNNLinear` class (or its LLama variants for BFP8):

- Weight is replicated using `ttnn.replicate_tensor_to_mesh_mapper(device)`
  instead of `shard_tensor_to_mesh_mapper`.
- Each device computes the full `[5376, 2048]` matmul independently.
- The output `[B, 1, 2048]` (= 4 heads x 512 dims) is consumed locally for
  both the K and V paths (after cloning for the divergent norm/RoPE
  processing described in
  [Chapter 3](../ch3_kv_sharing_and_vnorm/k_eq_v_mechanism.md)).

```python
# Global layer K projection (replicated)
k_proj = TTNNLinear.from_parameters(
    weight=k_weight,  # [5376, 2048] → replicated on all 8 devices
)
# forward: kv_shared = k_proj(hidden_states)  # [B, 1, 2048] on every device
```

### Complete Per-Layer TTNN Module Map

| Projection | Sliding Layer | Global Layer |
|------------|--------------|-------------|
| Q | `TTNNLinearIReplicatedWColSharded` | `TTNNLinearIReplicatedWColSharded` |
| K | `TTNNLinearIReplicatedWColSharded` | `TTNNLinear` (replicated) |
| V | `TTNNLinearIReplicatedWColSharded` | N/A (K=V sharing) |
| O | `TTNNLinearIColShardedWRowSharded` | `TTNNLinearIColShardedWRowSharded` |
| Gate | `TTNNLinearIReplicatedWColSharded` | `TTNNLinearIReplicatedWColSharded` |
| Up | `TTNNLinearIReplicatedWColSharded` | `TTNNLinearIReplicatedWColSharded` |
| Down | `TTNNLinearIColShardedWRowSharded` | `TTNNLinearIColShardedWRowSharded` |

For BFP8 weight quantization, substitute the LLama variants
(`TTNNLinearLLamaIColShardedWRowSharded` for row-parallel, and the
corresponding BFP8 column-parallel variant for column-parallel).

---

**Next:** [`kv_cache_sharding.md`](./kv_cache_sharding.md)
