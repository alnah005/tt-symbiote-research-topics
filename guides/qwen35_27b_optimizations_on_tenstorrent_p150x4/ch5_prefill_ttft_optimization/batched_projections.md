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

The decode program config is built by `create_dram_sharded_matmul_program_config` (`model_config.py`, line 95) and stored as `M=1` configs on `Qwen35ModelArgs` at init time (`model_config.py`, lines 291-296). The prefill config is built on-demand by `create_prefill_matmul_program_config(m, k, n, grid_size=(8, 8))` (`model_config.py`, lines 146-172), which computes tile-aligned `per_core_M` and `per_core_N` values and finds the largest valid `out_subblock_w` satisfying the FP32 DST register constraint `out_subblock_h * out_subblock_w <= 4`.

## Attention Layer Prefill Projections

In `Qwen35Attention.forward_prefill()` (`attention.py`, lines 302-484), three batched projections compute the full sequence's Q+gate, K, and V. The projections operate on per-device TP-sharded weights, so output dimensions reflect local head counts (`n_local_heads = 6`, `n_local_kv_heads = 1` with TP=4):

```
x_dram: [1, 1, seq_len, dim=5120]

Q+gate:  x_dram @ wqkv -> [1, 1, seq_len, n_local_heads*HD*2 = 6*256*2 = 3072]
K:       x_dram @ wk   -> [1, 1, seq_len, n_local_kv_heads*HD = 1*256 = 256]
V:       x_dram @ wv   -> [1, 1, seq_len, n_local_kv_heads*HD = 1*256 = 256]
```

The Q+gate and K projections each call `create_prefill_matmul_program_config(seq_len, dim, out_dim)` directly (`attention.py`, lines 336 and 344). The V projection does not create a new config; it reuses `k_progcfg` from line 344 (`attention.py`, line 352). The input `x` is first moved to DRAM interleaved via `ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)` at line 334. All three projections share this same `x_dram` tensor, which is deallocated at line 358 after the last projection completes.

After the projections, the attention layer continues with reshape, partial RoPE, KV cache fill, flash SDPA, and sigmoid gating -- all operating on the full `[1, NH, seq_len, HD]` tensor. The output projection is also a 2D matmul (`attention.py`, lines 474-480):

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

The QKVZ projection (`gdn.py`, lines 611-617) uses `create_prefill_matmul_program_config(seq_len, dim, qkvz_dim_tp)` and produces a fused tensor containing Q, K, V, and Z gate values for all tokens, which are later sliced per-token during the sequential recurrence loop. The AB projection (`gdn.py`, lines 620-626) uses `create_prefill_matmul_program_config(seq_len, dim, Nv_TP * 2)` and produces the two scalar gates (a and b) per value head for all tokens at once. Both share the same `x_dram` tensor, which is deallocated at line 628.

After the per-token recurrence loop (covered in [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md)), the per-token outputs are concatenated and processed through a batched output projection (`gdn.py`, lines 716-722):

```
gated_seq: [1, 1, seq_len, value_dim_tp] @ wout -> [1, 1, seq_len, dim = 5120]
```

This output projection uses `create_prefill_matmul_program_config(seq_len, self.value_dim_tp, dim)` and is followed by an all-reduce across TP devices.

## Why This Matters for TTFT

For a 96-token prefill, this reduces GDN kernel dispatches from ~288 (one per token per layer) to 3 — one QKVZ matmul, one AB matmul, and one output matmul.

---

**Next:** [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md)
