# CCL and Ethernet

## Introduction

Collective communication library (CCL) operations are the connective tissue of tensor parallelism. Every layer boundary in a TP-distributed transformer requires at least one collective to reconcile partial results across devices. This section covers the three collectives used in TT-Transformers, the Ethernet bandwidth budget that constrains them on Wormhole hardware, the pipelining technique that hides CCL latency, and a framework for deciding when TP helps vs hurts end-to-end performance.

For the three TP patterns that call these collectives, see `tensor_parallelism.md`. For the physical Ethernet ring topology of T3K, see Ch4.

---

## Collective Operations

### ttnn.all_gather

`ttnn.all_gather` is used after column-parallel linear layers.

Each device holds its own output shard of shape `[batch, N/TP]`. `all_gather` exchanges shards so that every device ends up with the full concatenated tensor of shape `[batch, N]`.

```
Input per device:   Y_d   shape [batch, N/TP]
Output per device:  Y     shape [batch, N]      (same on every device)
```

Volume communicated per device: each device sends `N/TP` elements and receives `(TP-1) * N/TP` elements, for a total receive volume of `N` elements minus the local shard. [INFERRED] Total wire volume across all devices scales as `TP * (TP-1)/TP * N * batch * element_size`, which grows with TP degree.

### ttnn.reduce_scatter

`ttnn.reduce_scatter` is used after row-parallel linear layers.

Each device holds a partial sum of shape `[batch, N]` that must be summed with the partial sums from all other devices. After `reduce_scatter`, the summed result is distributed: each device receives one shard of the final output of shape `[batch, N/TP]`. No device holds the full result.

```
Input per device:   P_d   shape [batch, N]      (partial sum)
Output per device:  Y_d   shape [batch, N/TP]   (one shard of the summed result)
```

`reduce_scatter` is not `all_reduce`. `all_reduce` would produce the full result on every device, which is unnecessary when the subsequent operation (for example, the next column-parallel layer) can consume a sharded input directly. Using `reduce_scatter` keeps the output distributed, avoiding redundant replication.

### ttnn.all_reduce

`ttnn.all_reduce` is used after the vocab-parallel LM head.

Conceptually, `all_reduce` is an `all_gather` followed by an elementwise reduction (sum). Every device ends up with the complete reduced tensor.

```
Input per device:   L_d   shape [batch, vocab_size/TP]
Output per device:  L     shape [batch, vocab_size]     (same on every device)
```

`all_reduce` is the correct operation here because downstream sampling or loss computation on every device requires the full logit vector. This is the only place in the standard transformer TP pattern where `all_reduce` is appropriate. Elsewhere, `all_gather` (which concatenates without reducing) or `reduce_scatter` (which reduces and distributes) is the correct choice.

### Summary

| Operation | Input per Device | Output per Device | Use in Transformer |
|---|---|---|---|
| `ttnn.all_gather` | Shard `[batch, N/TP]` | Full `[batch, N]` | After column-parallel linear |
| `ttnn.reduce_scatter` | Partial sum `[batch, N]` | Shard `[batch, N/TP]` | After row-parallel linear |
| `ttnn.all_reduce` | Shard `[batch, vocab_size/TP]` | Full `[batch, vocab_size]` | After vocab-parallel LM head |

---

## Ethernet Bandwidth Budget

### Wormhole Ethernet Interconnect

[confirmed] Both N300 and T3K use Ethernet for inter-chip communication. N300 uses on-board Ethernet between its two chips; T3K arranges 8 chips in an Ethernet ring. Neither configuration uses PCIe for chip-to-chip data transfer.

The Ethernet bandwidth is finite and shared by all CCL traffic in a given time step. Every collective operation places a demand on the available Ethernet bandwidth. Specific MB/s or GB/s figures for Wormhole Ethernet links are not reproduced here because confirmed numbers are not available in the source material used for this guide.

### Ring Topology and Hop Count

[confirmed] T3K is an Ethernet ring, not a mesh. A ring topology means that a message traveling from device 0 to device 4 (the furthest point in an 8-device ring) must traverse 4 hops. [INFERRED] Collective operations that require every device to communicate with every other device (such as `all_gather` and `all_reduce`) must route data around the ring, and the effective latency grows with the number of hops traversed. In a ring-based `all_gather` with TP=8, each device receives TP-1 = 7 additional shards in a pipelined ring pass.

For N300 (2 chips, 1 hop), the ring traversal cost is minimal — a single Ethernet transfer suffices. For T3K, the multi-hop nature of the ring means collective latency is materially higher than on N300.

### Interaction with Weight Quantization

The per-device weight bandwidth is reduced by quantization as described in Ch3:

- [confirmed] BFP4 weights (used for MLP layers in large models such as Llama 3.1 70B): 3.56x lower bandwidth demand vs BF16.
- [confirmed] BFP8 weights (attention layers): 1.88x lower bandwidth demand vs BF16.

Quantization reduces the DRAM-to-L1 weight load cost on each device but does not reduce the CCL communication volume, which is determined by the activation tensor sizes (typically BF16 or BFP8), not by weight precision. [INFERRED] Therefore, as weight quantization makes the local matmul faster (less weight bandwidth), CCL overhead becomes a relatively larger fraction of total layer time.

---

## CCL Pipelining

### The Latency Problem

In a naive implementation, the execution sequence for a column-parallel linear layer is:

1. Local matmul: `Y_d = X @ W_d`
2. Wait for CCL: `ttnn.all_gather(Y_d)`
3. Next operation begins using the full `Y`

Step 2 is pure communication; the compute units on every device are idle during `all_gather`. At small batch sizes, where the matmul itself is fast (few tokens to process), the CCL wait can dominate total layer time.

[confirmed] For very small batch sizes, CCL latency accumulates per decode step and can dominate decode inter-token latency.

### Pipelining CCL with Compute

[confirmed] Overlapping the CCL operation with the next matmul computation reduces effective CCL cost. The idea is to start the `all_gather` for layer L while simultaneously beginning the matmul for the next sub-operation that does not depend on the all_gather result.

A concrete example in an MLP block with a column-parallel "up" projection followed by a row-parallel "down" projection:

1. Column-parallel matmul: `Y_d = X @ W_up_d`
2. Launch `ttnn.all_gather(Y_d)` (asynchronous)
3. While all_gather is in flight, begin loading `W_down_d` weights into L1 (or performing any compute that does not yet depend on the full `Y`)
4. Consume completed `Y` once all_gather finishes

[INFERRED] The degree to which pipelining hides CCL cost depends on the arithmetic intensity of the next compute phase relative to the CCL transfer time. At large batch sizes (prefill or batched decode), compute dominates and pipelining benefit is high. At small batch (single-token decode), compute is fast and CCL is harder to hide.

---

## When Tensor Parallelism Helps vs Hurts

### Factors That Make TP Beneficial

1. **Weight memory**: A weight matrix that does not fit in a single device's combined L1 + DRAM can be split across devices with TP. [INFERRED] For Llama 3.1 70B at BFP4/BFP8 quantization, the 70B parameter count still requires multiple devices to hold all weights at inference time; TP=8 on T3K is the solution.

2. **Compute throughput**: TP divides the arithmetic work of each matmul by TP. At large batch sizes where the local matmul is the bottleneck, this provides near-linear throughput scaling. [INFERRED] Ideal scaling is reduced by CCL overhead and the broadcast/gather cost of activation distribution.

3. **Memory bandwidth per device**: Each device loads only `1/TP` of the weight matrix per forward pass. At the bandwidth-bound regime (common in decode with small batch), this directly reduces per-step weight load time.

### Factors That Make TP Harmful or Neutral

1. **CCL overhead at small batch**: [confirmed] At very small batch sizes, CCL latency can dominate inter-token latency. The matmul itself finishes quickly (few tokens, low arithmetic), but the `all_gather` or `reduce_scatter` must still traverse the full ring regardless of batch size. The CCL cost is roughly constant per layer per decode step; the compute savings from TP shrink with batch size. Below a model-specific batch threshold, adding more TP degrees increases total step time.

2. **Increased hop count on T3K**: [INFERRED] Going from TP=4 to TP=8 on T3K increases the number of CCL participants and the volume of data traversing the ring. If the model's weight matrices already fit on 4 devices and compute is not the bottleneck, the extra CCL cost of TP=8 may not pay off.

3. **KV head replication overhead**: As described in `tensor_parallelism.md`, when TP > n_kv_heads, KV weights are replicated rather than sharded. The per-device KV cache is not reduced, and additional KV cache memory pressure offsets some of the benefit of sharding Q and MLP weights.

4. **Quantization interaction**: [INFERRED] BFP4 MLP weights already reduce the weight bandwidth bottleneck by 3.56x on a single device. If the per-device weight load is fast enough that compute is no longer bandwidth-bound, the bandwidth argument for TP is weaker. The CCL cost must then be justified by arithmetic throughput gains alone.

### Decision Framework

| Situation | Likely Decision |
|---|---|
| Model weights do not fit on one device | TP required (no alternative) |
| Large batch (prefill or high-throughput decode) | TP beneficial; compute savings outweigh CCL |
| Single-token decode, small model, fits on one device | TP may hurt; CCL dominates |
| TP degree would exceed n_kv_heads | Avoid; KV replication wastes memory and negates KV cache benefits |
| T3K available, model is 70B+ | TP=8 is the confirmed production choice for Llama 3.1 70B |

[INFERRED] The optimal TP degree for a given model and serving configuration is found empirically by profiling CCL time vs matmul time at the target batch size. The framework above provides the directional guidance; specific thresholds depend on model size, weight precision, and batch distribution.

---

## Per-Layer CCL Summary for a Standard Transformer Block

The following table summarizes which collective is called at each layer boundary in a fully TP-distributed transformer block.

| Layer | TP Pattern | Collective | Notes |
|---|---|---|---|
| Q/K/V projection | Column-parallel | `ttnn.all_gather` | Produces full QKV on each device |
| Attention output projection | Row-parallel | `ttnn.reduce_scatter` | Produces sharded result |
| MLP up/gate projection | Column-parallel | `ttnn.all_gather` | Full activation for activation function |
| MLP down projection | Row-parallel | `ttnn.reduce_scatter` | Produces sharded result for next layer |
| LM head (vocab-parallel) | Vocab-parallel | `ttnn.all_reduce` | Full logits required on every device |

[INFERRED] In practice, the `all_gather` after the Q/K/V column-parallel projection and the row-parallel reduce_scatter after the attention output projection may be fused or reordered with adjacent operations to maximize CCL/compute overlap, depending on the implementation.
