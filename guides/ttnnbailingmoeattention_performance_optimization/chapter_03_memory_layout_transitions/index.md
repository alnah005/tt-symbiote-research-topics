# Chapter 3 — Memory Layout Transitions and L1 Pressure

## Background: The Cost of Layout Transitions on Wormhole

Every `ttnn.to_memory_config` call in the decode path is a data-movement operation that consumes NoC bandwidth and stalls downstream compute until the copy completes. On Wormhole, tensors at rest live in DRAM (off-chip SRAM banks accessible over the NoC). L1 is per-core on-chip SRAM — fast and low-latency for the core that owns it, but strictly sized and not shared across cores. Moving a tensor into L1 means the NoC must carry each tile from a DRAM bank to the target core's L1 buffer.

Two distinct costs apply:

**DRAM → L1 (interleaved).** An interleaved DRAM tensor has its tiles striped across DRAM banks. Moving it to `ttnn.L1_MEMORY_CONFIG` (interleaved L1) causes every tile to traverse the NoC from DRAM to a worker core's L1. For a bfloat16 tensor of shape `[1, B, H, D]` the volume is `B * H * D * 2` bytes (before tile padding). At decode with B=32, H=16, D=128 that is `32 * 16 * 128 * 2 = 131,072` bytes for Q and `32 * 4 * 128 * 2 = 32,768` bytes for K per device. These numbers are small in absolute terms but the copy still serializes with the next op unless pipelined.

**Interleaved L1 → HEIGHT_SHARDED L1 (reshard).** Resharding within L1 is an L1-to-L1 copy over the NoC. The bandwidth cost is the same as the tensor volume, but the hop count is usually short (intra-chip NoC). The latency is lower than a DRAM read but is still a blocking dependency: `rotary_embedding_llama` with `is_decode_mode=True` requires its Q and K inputs in a specific HEIGHT_SHARDED config with shard shape `(TILE_SIZE=32, head_dim=128)` per core. Any tensor that arrives from a different layout must pass through a reshard before the RoPE kernel can launch.

**HEIGHT_SHARDED → different HEIGHT_SHARDED (reshard).** The most wasteful case: K is already HEIGHT_SHARDED after RoPE, but the shard spec differs from what `paged_update_cache` requires. This forces a second HEIGHT_SHARDED copy of the same tensor for no computational gain.

**Quantitative baseline.** At B=32 (padded to TTNN tile boundary) the total data moved by all 9 `ttnn.to_memory_config` calls in `_forward_decode_paged` accounts for roughly:

| Tensor | Shape | Volume per copy (bfloat16) |
|---|---|---|
| Q | `[1, 32, 16, 128]` | 131,072 bytes |
| K | `[1, 32, 4, 128]` | 32,768 bytes |
| cos | `[1, 32, 1, 128]` | 8,192 bytes |
| sin | `[1, 32, 1, 128]` | 8,192 bytes |
| Q (reshard) | `[1, 32, 16, 128]` | 131,072 bytes |
| K (reshard) | `[1, 32, 4, 128]` | 32,768 bytes |
| K (re-reshard) | `[1, 32, 4, 128]` | 32,768 bytes |
| V | `[1, 32, 4, 128]` | 32,768 bytes |
| attn_output | `[1, 32, 16, 128]` | 131,072 bytes |

Total: approximately 528 KB of NoC data movement per decode step, just from layout transitions. The two copies of Q each contribute 131 KB, making Q transitions the dominant overhead.

---

## Inventory of All `ttnn.to_memory_config` Calls in `_forward_decode_paged`

The table below lists all 9 calls in execution order. Line numbers refer to `attention.py` at `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/modules/attention.py`.

| Step | Line | Tensor | From layout / memory | To layout / memory | Reason | Avoidable? |
|---|---|---|---|---|---|---|
| 8a | 2656 | `query_states` | DRAM interleaved (post-all-gather + reshape) | L1 interleaved (`L1_MEMORY_CONFIG`) | `ttnn.rms_norm` inside `TTNNRMSNorm` requires interleaved non-sharded input; the reshape on line 2644 produces an interleaved DRAM tensor | Yes — see [avoidable_transitions.md](avoidable_transitions.md) |
| 8b | 2657 | `key_states` | DRAM interleaved (post-all-gather + reshape) | L1 interleaved (`L1_MEMORY_CONFIG`) | Same as step 8a; K also needs QK norm | Yes — same path |
| 12a | 2708 | `cos_ttnn` | DRAM interleaved (output of `ttnn.embedding` + transpose) | L1 HEIGHT_SHARDED (`rope_shard_mem`, shard `(TILE_SIZE, D)`) | `rotary_embedding_llama` with `is_decode_mode=True` requires HEIGHT_SHARDED cos input | Partially — cos lives in DRAM because `ttnn.embedding` always writes to DRAM; could pipeline with Q/K shard |
| 12b | 2709 | `sin_ttnn` | DRAM interleaved (output of `ttnn.embedding` + transpose) | L1 HEIGHT_SHARDED (`rope_shard_mem`, shard `(TILE_SIZE, D)`) | Same as step 12a for sin | Partially — same constraint |
| 12c | 2718 | `query_states` | L1 interleaved (post-QK norm output) | L1 HEIGHT_SHARDED (`q_shard_mem`, shard `(TILE_SIZE, D)`) | `rotary_embedding_llama` with `is_decode_mode=True` requires HEIGHT_SHARDED Q | Yes — if QK norm ran before all_gather, Q could emerge already HEIGHT_SHARDED |
| 12d | 2727 | `key_states` | L1 interleaved (post-QK norm output) | L1 HEIGHT_SHARDED (`k_shard_mem`, shard `(TILE_SIZE, key_states.shape[-1])`) | `rotary_embedding_llama` requires HEIGHT_SHARDED K | Yes — same as step 12c |
| 16a | 2753 | `key_states` | L1 HEIGHT_SHARDED (RoPE shard spec: `(TILE_SIZE=32, D=128)`) | L1 HEIGHT_SHARDED (`kv_mem`, shard `(kv_vol, padded_D)`) | `paged_update_cache` requires different shard spec; second HEIGHT_SHARDED copy of K | Yes — potentially if RoPE output shard spec can be matched to paged_update requirement |
| 16b | 2754 | `value_states` | DRAM interleaved (V skipped QK norm and RoPE; arrived from all-gather + reshape) | L1 HEIGHT_SHARDED (`kv_mem`, shard `(kv_vol, padded_D)`) | `paged_update_cache` requires HEIGHT_SHARDED V | No — V must come from DRAM because it received no prior L1 ops |
| 20 | 2783 | `attn_output` | DRAM interleaved (paged SDPA decode output) | L1 HEIGHT_SHARDED (`sdpa_output_memcfg`, shard `(32, D)`, `CoreGrid(y=1, x=B)`) | `nlp_concat_heads_decode` requires HEIGHT_SHARDED input | Potentially — if `paged_sdpa_decode` could be configured to output directly in this shard spec |

---

## Per-file deep dives

- [transition_analysis.md](transition_analysis.md) — per-transition analysis of all 9 transitions: data volume, direction, and avoidability rationale
- [avoidable_transitions.md](avoidable_transitions.md) — deep-dive on the three categories of avoidable transitions with concrete proposals

---

**Next:** [transition_analysis.md](transition_analysis.md)
