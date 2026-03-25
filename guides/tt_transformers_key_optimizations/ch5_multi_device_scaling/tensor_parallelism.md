# Tensor Parallelism

## Introduction

Tensor parallelism (TP) partitions weight matrices across devices so that each device performs a fraction of the total matrix multiply work. On Tenstorrent hardware this means distributing L1-resident weight shards across multiple Wormhole chips and using collective communication operations (CCL) to produce the correct output. This section covers the three canonical TP linear patterns used in transformer models, how they interact with GQA head counts, and the differences between N300 and T3K configurations.

For background on single-device sharding (height, block, DRAM-sharded, 1D ring), see Ch4. For GQA structure (Q/KV head counts, group size), see Ch2.

---

## Hardware Configurations

### N300

[confirmed] The N300 is a 2-chip Wormhole board. The two chips communicate via on-board Ethernet; there is no PCIe path between them. Maximum supported TP degree on N300 is 2.

### T3K

[confirmed] The T3K is an 8-chip Wormhole configuration arranged as an Ethernet ring. Each chip is connected to its two neighbors in the ring via Ethernet links. Maximum supported TP degree on T3K is 8. The ring topology (not a mesh) means collectives traverse the ring; see `ccl_and_ethernet.md` for the latency implications.

| Board | Chips | Interconnect | Max TP |
|---|---|---|---|
| N300 | 2 | On-board Ethernet | 2 |
| T3K | 8 | Ethernet ring | 8 |

[confirmed] Llama 3.1 70B runs on T3K at TP=8 in production.

---

## Column-Parallel Linear

### Pattern

In a column-parallel linear layer the weight matrix `W` of shape `[K, N]` is sharded along the column (output) dimension. Each of the TP devices holds a shard of shape `[K, N/TP]`.

```
Weight on device d:  W_d  shape [K, N/TP]
```

The full activation `X` of shape `[batch, K]` is broadcast to every device. Each device computes a partial output:

```
Y_d = X @ W_d      shape [batch, N/TP]
```

After the local matmul, a `ttnn.all_gather` call concatenates the per-device partial outputs along the N dimension, producing the full output `Y` of shape `[batch, N]` on every device.

```
ttnn.all_gather(Y_d, dim=N_dim)  ->  Y  shape [batch, N]  on every device
```

### When to Use

Column-parallel is used for the first linear in a two-layer MLP block (the "up" or "gate" projection) and for the Q/K/V projection in attention. The all_gather cost is paid once per layer at the boundary between the column-parallel layer and its consumer. [INFERRED] It is also appropriate when the downstream operation requires the full activation tensor before it can proceed.

### Weight Layout and Quantization

[confirmed] For large models such as Llama 3.1 70B, MLP weights use BFP4 quantization (3.56x lower memory bandwidth vs BF16). [confirmed] Attention weights use BFP8 (1.88x lower bandwidth vs BF16). Each per-device shard `W_d` is stored in the quantized format; the bandwidth savings apply per device, independent of TP degree.

---

## Row-Parallel Linear

### Pattern

In a row-parallel linear layer the weight matrix `W` of shape `[K, N]` is sharded along the row (input) dimension. Each device holds a shard of shape `[K/TP, N]`.

```
Weight on device d:  W_d  shape [K/TP, N]
```

The activation `X` of shape `[batch, K]` must be pre-sharded along the K dimension so that device d holds `X_d` of shape `[batch, K/TP]`. This pre-sharding is typically the natural output of the preceding column-parallel layer or an explicit scatter. Each device computes a partial sum:

```
P_d = X_d @ W_d      shape [batch, N]    (partial sum, not the final result)
```

After the local matmul, a `ttnn.reduce_scatter` call sums the partial results and distributes the summed output: each device ends up holding one shard of the final output `[batch, N/TP]`.

```
ttnn.reduce_scatter(P_d, dim=N_dim)  ->  Y_d  shape [batch, N/TP]  on each device
```

The collective for row-parallel is `reduce_scatter`, not `all_reduce`. See `ccl_and_ethernet.md` § ttnn.reduce_scatter for the full explanation of why `all_reduce` would be wasteful here.

### When to Use

Row-parallel is used for the second linear in a two-layer MLP block (the "down" projection) and for the output projection in attention (the linear that maps concatenated head outputs back to the model hidden dimension). It pairs naturally with column-parallel: the column-parallel layer produces sharded activations along N, which become the K-sharded inputs for the subsequent row-parallel layer. This column-then-row pairing means the all_gather from the column-parallel stage can be deferred or avoided when the row-parallel layer immediately follows.

---

## Vocab-Parallel LM Head

### Pattern

The language model head is an embedding lookup or linear projection from hidden dimension to vocabulary size. The weight matrix has shape `[vocab_size, hidden]` (or `[hidden, vocab_size]` depending on convention). Under vocab-parallel TP the weight is sharded along the vocab dimension:

```
Weight on device d:  E_d  shape [vocab_size/TP, hidden]
```

Each device computes logits for its vocabulary shard:

```
L_d = X @ E_d^T      shape [batch, vocab_size/TP]
```

Unlike the row-parallel case, the downstream operation (sampling or loss computation) requires the full logit vector of shape `[batch, vocab_size]` on every device. Therefore the collective is `ttnn.all_reduce`:

```
ttnn.all_reduce(L_d)  ->  L  shape [batch, vocab_size]  on every device
```

`ttnn.all_reduce` is the correct operation here because every device must have the complete result. See `ccl_and_ethernet.md` § ttnn.all_reduce for context on why this is the only place in the standard transformer TP pattern where `all_reduce` is appropriate.

For a consolidated summary of all three collectives and which layer boundary each serves, see `ccl_and_ethernet.md` § Summary and § Per-Layer CCL Summary.

---

## GQA with Tensor Parallelism

### Background

Grouped-query attention (GQA, covered in Ch2) uses fewer KV heads than Q heads. The GQA group size is:

```
G = n_q_heads / n_kv_heads
```

[confirmed] For Llama 3.1 70B: n_q_heads = 64, n_kv_heads = 8, G = 8.

### Sharding Rule

The number of KV heads sets a hard limit on how finely KV weights can be sharded. A KV head is the smallest unit of KV work; it cannot be further divided.

| Condition | KV Weight Treatment |
|---|---|
| TP <= n_kv_heads | KV weights can be sharded: each device holds `n_kv_heads / TP` KV heads |
| TP > n_kv_heads | KV weights must be replicated on every device (cannot be divided below 1 head per device) |

[confirmed] For Llama 3.1 70B at TP=8: TP = n_kv_heads = 8, so TP is exactly equal to the KV head count. Each device receives exactly 1 KV head and 8 Q heads (64 Q heads / 8 devices).

This is the boundary case: TP=8 is the maximum degree at which KV weights can still be sharded for this model. Increasing TP beyond 8 for a model with 8 KV heads would require replicating KV weights.

### Q Head Distribution

Q heads shard straightforwardly because n_q_heads >= TP in all practical configurations:

```
Q heads per device = n_q_heads / TP
```

[confirmed] Llama 3.1 70B on T3K: 64 / 8 = 8 Q heads per device.

### KV Cache Implications

[INFERRED] When KV weights are sharded (TP <= n_kv_heads), the KV cache can also be sharded across devices, reducing per-device KV cache memory proportionally. When KV weights are replicated (TP > n_kv_heads), each device stores a full copy of the KV cache, eliminating the memory benefit of TP for the KV cache. The paged KV cache discussed in Ch2 applies per device in either case.

### Attention Computation Under TP

[INFERRED] With sharded Q and KV heads, each device computes self-attention independently for its subset of heads. The attention outputs (one per Q head per device) are concatenated locally on each device, then fed into the output projection (a row-parallel linear) as described above. The KV replication boundary (TP = n_kv_heads) is therefore an important design point when choosing TP degree for a given model.

---

## N300 vs T3K: Practical Differences

| Consideration | N300 (TP=2) | T3K (TP=8) |
|---|---|---|
| Devices in ring | 2 | 8 |
| CCL hops (ring) | 1 | Up to 4 (half-ring) |
| Weight memory per device | W / 2 per shard | W / 8 per shard |
| KV head sharding (Llama 3.1 70B) | 4 KV heads per device | 1 KV head per device (exactly divisible) |
| Q heads per device (Llama 3.1 70B) | 32 | 8 |
| Use case | Smaller models; latency-sensitive with low CCL overhead | Large models (70B+); throughput-oriented |

[confirmed] Llama 3.1 70B targets T3K at TP=8 as its production configuration.

[INFERRED] For smaller models where the weight matrices fit comfortably within a single device's L1 and DRAM, the CCL overhead of higher TP degrees may outweigh the compute benefit. The trade-off is discussed in `ccl_and_ethernet.md`.
