# Chapter 6 — RoPE and QK Normalization Overhead

## Scope

This chapter examines two interconnected sources of per-step latency in the Ling decode path that are governed by model hyperparameters rather than by code structure alone: the QK normalization detour imposed by `use_qk_norm=True`, and the non-distributed rotary embedding forced by `partial_rotary_factor < 1.0`. Together they account for a substantial fraction of attention decode latency on T3K.

**Chapter 6 answers Questions 6 and 7 of the guide:**

- **Question 6:** What is the full latency breakdown of the `use_qk_norm=True` path — including the reshape overhead, the L1 move cost, and the `TTNNRMSNorm` kernel cost — and which components are eliminable?
- **Question 7:** Why does `partial_rotary_factor < 1.0` disable `TTNNDistributedRotaryPositionEmbedding` in favor of the non-distributed `TTNNRotaryPositionEmbedding`, what is the performance cost of running non-distributed RoPE on T3K, and are there avoidance strategies?

## Prerequisites

Readers should be familiar with the following material from earlier chapters:

- **Memory-config transition cost model** — The cost formula `t_trans = t_overhead + (B / BW)` and the per-transition estimates for Q (≈21 µs) and K (≈11 µs) are established in Chapter 4, [`transition_cost_model.md`](../ch4_memory_config_transitions/transition_cost_model.md). All T_norm_in and T_norm_out figures used in this chapter are derived from that model.
- **Decode tensor lifecycle** — The sequence of memory-config states traversed by Q and K before reaching the QK norm and SDPA stages is established in Chapter 4, [`decode_tensor_lifecycle.md`](../ch4_memory_config_transitions/decode_tensor_lifecycle.md). The norm path begins immediately after T2a/T2b (Q and K post-RoPE eviction to DRAM INTERLEAVED).
- **SDPA input layout** — Chapter 5 establishes the memory config expected by `paged_sdpa_decode` for Q input. The output memory config produced by the QK norm path must be compatible with SDPA's expectations (see Chapter 5, [`paged_sdpa_chunk_sizes.md`](../ch5_sdpa_and_compute_config/paged_sdpa_chunk_sizes.md)).
- **Fused QKV latency reference** — The fused QKV matmul costs approximately 10–40 µs [ESTIMATE] (Chapter 2, [`fusion_mechanics.md`](../ch2_fused_qkv_projection/fusion_mechanics.md)). This figure is the calibration point for assessing whether QK norm and non-distributed RoPE overhead is negligible or dominant relative to the prior compute stage.

## Ling Configuration Recap

The relevant Ling hyperparameters for this chapter:

| Parameter | Value | Consequence |
|---|---|---|
| `num_heads` | 16 | Q tensor: 16 heads × head_dim=128 → 4 KB at BF16 (seq_len=1); 128 KB with TILE_LAYOUT padding (seq_dim padded to 32) |
| `num_kv_heads` | 4 | K tensor: 4 heads × head_dim=128 → 1 KB at BF16 (seq_len=1); 32 KB with TILE_LAYOUT padding (seq_dim padded to 32) |
| `head_dim` | 128 | Shard shape per RoPE core: `(32, 128)` |
| `partial_rotary_factor` | 0.5 | `rotary_dim = 64`; forces `TTNNRotaryPositionEmbedding` (non-distributed) |
| `use_qk_norm` | `True` | Activates per-token `TTNNRMSNorm` on Q and K after RoPE |
| `hidden_size` | 4096 | Relevant for fused QKV matmul comparison (Chapter 2) |

## Reading Order

Work through the files in this order:

1. [`partial_rotary_rope.md`](./partial_rotary_rope.md) — Why `partial_rotary_factor < 1.0` disables the distributed RoPE kernel, performance cost of non-distributed execution on T3K, and concrete alternative strategies including cos/sin table padding and post-embedding slicing.
2. [`qk_norm_latency.md`](./qk_norm_latency.md) — The complete QK norm code path when `use_qk_norm=True`: the reshape sequence, the L1 move, the `TTNNRMSNorm` call, and the corresponding latency breakdown. Compares norm overhead to fused QKV matmul latency and evaluates elimination options.

## Key Symbols Used in This Chapter

| Symbol | Meaning |
|---|---|
| `N_q` | Number of Q heads = 16 |
| `N_kv` | Number of KV heads = 4 |
| `H` | `head_dim` = 128 elements |
| `T` | `TILE_SIZE` = 32 elements |
| `rotary_dim` | `int(partial_rotary_factor × head_dim)` = 64 elements |
| `T_norm_in_Q` | DRAM→L1 transition cost for Q entering norm ≈ 21 µs [ESTIMATE] |
| `T_norm_in_K` | DRAM→L1 transition cost for K entering norm ≈ 11 µs [ESTIMATE] |
| `T_norm_out_Q` | L1→DRAM transition cost for Q exiting norm ≈ 21 µs [ESTIMATE] |
| `T_norm_out_K` | L1→DRAM transition cost for K exiting norm ≈ 11 µs [ESTIMATE] |
| `TTNNRMSNorm` | TTNN RMS normalization kernel used for QK norm |
| `TTNNRotaryPositionEmbedding` | Non-distributed RoPE kernel (selected when `partial_rotary_factor < 1.0`) |
| `TTNNDistributedRotaryPositionEmbedding` | Mesh-parallel RoPE kernel (requires `partial_rotary_factor == 1.0`) |

**Start reading:** [Partial Rotary RoPE](partial_rotary_rope.md)
