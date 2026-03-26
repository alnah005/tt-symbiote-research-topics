# Chapter 1 — Decode Path Architecture and Op Sequence

## Overview

`TTNNBailingMoEAttention._forward_decode_paged` is the performance-critical hot path
executed at every single decode step of the Bailing MoE model on T3K. It accepts the
hidden states from the preceding MoE layer, projects them into query, key, and value
tensors, applies QK normalization and rotary position embedding, writes K and V into the
on-device paged KV cache, performs paged scaled dot-product attention over the full
cached context, and projects the output back to `d_model` dimensions.

Understanding the complete op sequence of `_forward_decode_paged` is a prerequisite for
every later chapter in this guide for three reasons:

1. **Collective communication ops are structurally determined by the projection
   types.** Chapter 2's analysis of all-gather and reduce-scatter costs flows directly
   from the order and types of projections performed here; the reduce-scatter + gather pair nets zero data distribution change.

2. **Every `ttnn.to_memory_config` call in this path is a potential optimization
   target.** Chapter 3 enumerates nine distinct memory layout transitions in this
   function. Each is driven by a downstream kernel's layout constraint. The transitions
   are only meaningful once their position in the execution sequence is established.

3. **Host-device round-trips block device execution.** Chapter 4 identifies two
   categories of host touches in this path: the `cur_pos_tt` tensor construction (lines
   2663–2685 of `attention.py`) and the position-tensor re-upload inside
   `BailingRotarySetup.get_cos_sin_for_decode`. Both can only be understood and
   eliminated with reference to the op sequence that surrounds them.

### Conventions

Throughout this chapter and the rest of the guide the following symbolic names are used:

| Symbol  | Meaning                          | Concrete value (Bailing MoE on T3K) |
|---------|----------------------------------|--------------------------------------|
| `B`     | Batch size                       | up to 32 (padded to 32 if `B < 32`) |
| `S`     | Sequence length in decode        | 1                                    |
| `H`     | Number of query attention heads  | 16                                   |
| `Hkv`  | Number of KV attention heads     | 4                                    |
| `D`     | Head dimension                   | 128                                  |
| `d_model` | Hidden size (`H * D`)          | 2048                                 |
| `N`     | Number of devices (T3K 1×8 mesh) | 8                                    |

The decode tensor layout `[1, B, H, D]` (referred to as **S B H D** throughout this
guide) is the format expected by `ttnn.experimental.rotary_embedding_llama` in decode
mode and by the paged attention kernels. All Q/K/V tensors are reshaped into this format
at step 6.

---

## Summary Table

The following table lists all ops across 24 numbered steps in `_forward_decode_paged` in execution order. Line
number references point to
`models/experimental/tt_symbiote/modules/attention.py`.

| Step | Op / Action | Input layout / memory | Output layout / memory | Touches host? |
|------|-------------|-----------------------|------------------------|---------------|
| 1 | `ttnn.to_layout` → TILE_LAYOUT (conditional) | ROW_MAJOR, DRAM | TILE, DRAM | No |
| 2 | `q_proj(hidden_states)` — `TTNNLinearIColShardedWRowSharded` | TILE, DRAM, col-sharded | TILE, DRAM, col-sharded (reduce-scatter output) | No |
| 3 | `ttnn.all_gather(hidden_states, dim=-1, num_links=1)` | `hidden_states`, col-sharded | TILE, DRAM, replicated | No |
| 4a | `k_proj(hidden_states_replicated)` — `TTNNLinearIReplicatedWColSharded` | TILE, DRAM, replicated | TILE, DRAM, col-sharded | No |
| 4b | `v_proj(hidden_states_replicated)` — `TTNNLinearIReplicatedWColSharded` | TILE, DRAM, replicated | TILE, DRAM, col-sharded | No |
| 5a | `_maybe_all_gather(query_states)` | TILE, DRAM, col-sharded | TILE, DRAM, replicated (bfloat16) | No |
| 5b | `_maybe_all_gather(key_states)` | TILE, DRAM, col-sharded | TILE, DRAM, replicated (bfloat16) | No |
| 5c | `_maybe_all_gather(value_states)` | TILE, DRAM, col-sharded | TILE, DRAM, replicated (bfloat16) | No |
| 6 | `ttnn.reshape` Q/K/V → `[1, B, H, D]` / `[1, B, Hkv, D]` | TILE, DRAM, replicated | TILE, DRAM, replicated (S B H D) | No |
| 7 | `ttnn.typecast` Q/K/V → bfloat16 (conditional) | TILE, DRAM | TILE, DRAM, bfloat16 | No |
| 8a | `ttnn.to_memory_config(query_states, L1_MEMORY_CONFIG)` | TILE, DRAM | TILE, L1 interleaved | No |
| 8b | `ttnn.to_memory_config(key_states, L1_MEMORY_CONFIG)` | TILE, DRAM | TILE, L1 interleaved | No |
| 9 | `_apply_qk_norm` — reshape, `ttnn.rms_norm`, reshape back | TILE, L1 interleaved | TILE, L1 interleaved, bfloat16 | No |
| 10 | `cache_position_tensor` construction + `ttnn.from_torch` → `cur_pos_tt` | host torch.int32 | ROW_MAJOR, DRAM, replicated, int32 | **Yes** |
| 11 | `BailingRotarySetup.get_cos_sin_for_decode(cache_position_tensor)` | host torch.int32 → `ttnn.embedding` lookup | TILE, DRAM, replicated, `[1, B, 1, D]` | **Yes (always)**; a conditional device-drain may also occur if `position_ids` arrives as a `ttnn.Tensor` |
| 12a | `ttnn.to_memory_config(cos_ttnn, rope_shard_mem)` | TILE, DRAM | TILE, L1 HEIGHT_SHARDED | No |
| 12b | `ttnn.to_memory_config(sin_ttnn, rope_shard_mem)` | TILE, DRAM | TILE, L1 HEIGHT_SHARDED | No |
| 12c | `ttnn.to_memory_config(query_states, q_shard_mem)` | TILE, L1 interleaved | TILE, L1 HEIGHT_SHARDED | No |
| 12d | `ttnn.to_memory_config(key_states, k_shard_mem)` | TILE, L1 interleaved | TILE, L1 HEIGHT_SHARDED | No |
| 13 | `BailingRotarySetup.get_trans_mat_decode_sharded(batch_size)` | (lazy cache lookup) | TILE, L1 HEIGHT_SHARDED (cached) | No |
| 14 | `ttnn.experimental.rotary_embedding_llama(query_states, …, is_decode_mode=True)` | TILE, L1 HEIGHT_SHARDED | TILE, L1 HEIGHT_SHARDED | No |
| 15 | `ttnn.experimental.rotary_embedding_llama(key_states, …, is_decode_mode=True)` | TILE, L1 HEIGHT_SHARDED | TILE, L1 HEIGHT_SHARDED | No |
| 16a | `ttnn.to_memory_config(key_states, kv_mem)` — re-shard to paged-update spec | TILE, L1 HEIGHT_SHARDED | TILE, L1 HEIGHT_SHARDED (different shard spec) | No |
| 16b | `ttnn.to_memory_config(value_states, kv_mem)` | TILE, DRAM replicated | TILE, L1 HEIGHT_SHARDED | No |
| 17 | `past_key_values.paged_update_on_device(key_states, value_states, …)` | TILE, L1 HEIGHT_SHARDED | written into on-device DRAM KV cache | No |
| 18 | `ttnn.deallocate(key_states)` + `ttnn.deallocate(value_states)` | — | freed L1 | No |
| 19 | `past_key_values.paged_sdpa_decode(query_states, …)` | TILE, L1 HEIGHT_SHARDED (Q); DRAM KV cache | TILE, DRAM or L1, `[1, B, H, D]` | No |
| 20 | `ttnn.to_memory_config(attn_output, sdpa_output_memcfg)` | TILE (various) | TILE, L1 HEIGHT_SHARDED `(32, D)` per core | No |
| 21 | `ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=H)` | TILE, L1 HEIGHT_SHARDED | TILE, `[1, 1, B, d_model]` | No |
| 22 | `dense(attn_output)` — `TTNNLinearIReplicatedWColSharded` | TILE, DRAM | TILE, DRAM, col-sharded | No |
| 23 | `ttnn.slice(attn_output, …)` (conditional, `B < 32`) | TILE, DRAM | TILE, DRAM, `[1, 1, B, d_model]` | No |
| 24 | `ttnn.reshape(attn_output, (B, S, -1))` | TILE, DRAM | TILE, DRAM, `[B, 1, d_model]` | No |

Steps 10 and 11 are the only two that touch the host CPU. Step 11 always touches the
host because `get_cos_sin_for_decode` unconditionally calls `ttnn.from_torch` on every
decode step. In addition, if `position_ids` arrives as a `ttnn.Tensor`, a `ttnn.to_torch`
call is also triggered, producing a device-drain synchronization on top of the
unconditional host touch.

---

**Next:** [op_sequence.md](./op_sequence.md) — step-by-step annotation of every
operation with line numbers, tensor shapes, and performance notes.
