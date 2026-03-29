# Batched Projections: 2D Matmul for Prefill

The single largest source of prefill overhead in the baseline implementation was per-token projection dispatch. During decode, each projection uses a DRAM-sharded matmul with `M=1` -- a configuration tuned for the bandwidth-bound regime where a single token's activations are multiplied against large weight matrices. When this same pattern is used for prefill, each of the `seq_len` tokens triggers a separate kernel launch with its own dispatch overhead, DRAM weight reads, and synchronization. For a 96-token prompt through 64 layers, this means thousands of individual DRAM-sharded dispatches.

The optimized prefill replaces these with **2D multicast matmuls** that process the entire `[1, 1, seq_len, dim]` activation tensor in a single dispatch. This changes the computational character from bandwidth-bound to compute-bound: the weight matrix is read from DRAM once and multicast across an 8x8 compute grid, while the M dimension (seq_len) is parallelized across grid rows.

## Decode vs. Prefill Matmul Configuration

| Property | Decode (DRAM-sharded) | Prefill (2D multicast) |
|----------|----------------------|------------------------|
| Program config | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | `MatmulMultiCoreReuseMultiCastProgramConfig` |
| Compute grid | Variable (weight-dependent) | 8x8 (64 cores) |
| M dimension | 1 (single token) | seq_len (full prompt) |
| Weight location | WIDTH_SHARDED across 8 DRAM cores | DRAM interleaved |
| Bottleneck | DRAM bandwidth (weight read) | Compute (matmul FLOPs) |
| Dispatches per layer | seq_len | 1 |

## Attention Layer Prefill Projections

In `Qwen35Attention.forward_prefill()` (`attention.py`, lines 302-484), three batched projections compute the full sequence's Q+gate, K, and V:

```
x_dram: [1, 1, seq_len, dim=5120]

Q+gate:  x_dram @ wqkv -> [1, 1, seq_len, n_local_heads*HD*2 = 6*256*2 = 3072]
K:       x_dram @ wk   -> [1, 1, seq_len, n_local_kv_heads*HD = 1*256 = 256]
V:       x_dram @ wv   -> [1, 1, seq_len, n_local_kv_heads*HD = 1*256 = 256]
```

Each projection uses `create_prefill_matmul_program_config(seq_len, in_dim, out_dim)` to compute tile-aligned `per_core_M` and `per_core_N` values for the 8x8 grid. The config also respects the FP32 DST register limit: `out_subblock_h * out_subblock_w <= 4` when FP32 accumulation is enabled.

The input `x` is first moved to DRAM interleaved via `ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)`. All three projections share this same `x_dram` tensor, which is deallocated after the last projection completes.

After the projections, the attention layer continues with reshape, partial RoPE, KV cache fill, flash SDPA, sigmoid gating, and a batched output projection -- all operating on the full `[1, NH, seq_len, HD]` tensor. The output projection is also a 2D matmul:

```
gated_flat: [1, 1, seq_len, n_local_heads*HD = 6*256 = 1536] @ wo -> [1, 1, seq_len, dim = 5120]
```

## GDN Layer Prefill Projections

In `TtGatedDeltaNet.forward_prefill()` (`gdn.py`, lines 578-726), two batched projections compute QKVZ and AB for the full sequence:

```
x_dram: [1, 1, seq_len, dim=5120]

QKVZ:  x_dram @ wqkvz -> [1, 1, seq_len, qkvz_dim_tp = 4096]
AB:    x_dram @ wab   -> [1, 1, seq_len, Nv_TP*2    = 24]
```

The QKVZ projection is the dominant compute cost. It produces a fused tensor containing Q, K, V, and Z gate values for all tokens, which are later sliced per-token during the sequential recurrence loop. The AB projection produces the two scalar gates (a and b) per value head, also for all tokens at once.

After the per-token recurrence loop (covered in [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md)), the per-token outputs are concatenated and processed through a batched output projection:

```
gated_seq: [1, 1, seq_len, value_dim_tp] @ wout -> [1, 1, seq_len, dim = 5120]
```

This output projection uses `create_prefill_matmul_program_config(seq_len, self.value_dim_tp, dim)` and is followed by an all-reduce across TP devices.

## Why This Matters for TTFT

The dispatch overhead reduction is multiplicative. Consider a single GDN layer processing a 96-token prompt:

- **Baseline:** 96 QKVZ dispatches + 96 AB dispatches + 96 output projection dispatches = 288 kernel launches
- **Optimized:** 1 QKVZ dispatch + 1 AB dispatch + 1 output projection dispatch = 3 kernel launches

Each dispatch involves host-device communication, program cache lookup, and synchronization. Eliminating 285 dispatches per GDN layer, multiplied across 48 GDN layers and 16 attention layers, accounts for a significant portion of the 5.3x TTFT improvement.

Beyond dispatch count, the 2D matmul configuration is inherently more efficient for `M > 1`. The DRAM-sharded decode matmul reads the full weight matrix from DRAM for each token -- the weight read cost is constant regardless of M. The 2D multicast matmul reads the weight once and distributes it across grid rows, amortizing the DRAM read across all `seq_len` tokens. When `seq_len = 96`, the weight is read once instead of 96 times, converting a bandwidth-bound operation into a compute-bound one where the 8x8 grid is fully utilized.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md)
