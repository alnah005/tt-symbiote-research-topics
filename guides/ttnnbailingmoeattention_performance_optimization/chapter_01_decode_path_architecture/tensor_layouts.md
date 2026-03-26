# Tensor Layouts Reference — `_forward_decode_paged`

This document is a reference table of tensor shapes, memory locations, sharding
strategies, and data types at each step of `_forward_decode_paged`. It also explains
why the `[1, B, H, D]` S B H D layout is required by paged attention kernels, and
provides a diagram of the memory location transitions across the decode path.

Symbols follow the conventions defined in [index.md](./index.md). `TILE` denotes the TTNN tile size: 32 elements per edge.

## Step-by-Step Tensor Layout Table

### Input and Projection Phase (Steps 1–5)

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| Input | `hidden_states` | `[B, 1, 2048]` | `[B, 1, 256]` | DRAM | col-sharded on dim -1 | bfloat16 |
| 1 (out) | `hidden_states` | `[B, 1, 2048]` | `[B, 1, 256]` | DRAM | col-sharded | bfloat16; TILE_LAYOUT |
| 2 (out) | `query_states` | `[B, 1, 2048]` | `[B, 1, 256]` | DRAM | col-sharded (reduce-scatter output) | bfloat16 |
| 3 (out) | `hidden_states_replicated` | `[B, 1, 2048]` | `[B, 1, 2048]` (full) | DRAM | replicated | bfloat16 |
| 4a (out) | `key_states` | `[B, 1, 512]` | `[B, 1, 64]` | DRAM | col-sharded on dim -1 | bfloat16 |
| 4b (out) | `value_states` | `[B, 1, 512]` | `[B, 1, 64]` | DRAM | col-sharded on dim -1 | bfloat16 |
| 5a (out) | `query_states` | `[B, 1, 2048]` | `[B, 1, 2048]` (full) | DRAM | replicated | bfloat16 |
| 5b (out) | `key_states` | `[B, 1, 512]` | `[B, 1, 512]` (full) | DRAM | replicated | bfloat16 |
| 5c (out) | `value_states` | `[B, 1, 512]` | `[B, 1, 512]` (full) | DRAM | replicated | bfloat16 |

### Reshape and Pre-Norm Phase (Steps 6–8)

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| 6 (out) Q | `query_states` | `[1, B, 16, 128]` | `[1, B, 16, 128]` | DRAM | replicated | bfloat16 |
| 6 (out) K | `key_states` | `[1, B, 4, 128]` | `[1, B, 4, 128]` | DRAM | replicated | bfloat16 |
| 6 (out) V | `value_states` | `[1, B, 4, 128]` | `[1, B, 4, 128]` | DRAM | replicated | bfloat16 |
| 7 (out) | Q / K / V | same shapes as step 6 | same | DRAM | replicated | bfloat16 (enforced) |
| 8a (out) | `query_states` | `[1, B, 16, 128]` | `[1, B, 16, 128]` | **L1** | interleaved | bfloat16 |
| 8b (out) | `key_states` | `[1, B, 4, 128]` | `[1, B, 4, 128]` | **L1** | interleaved | bfloat16 |

Note: `value_states` is **not** moved to L1 at step 8 — it remains in DRAM until step 16b.

### QK Norm Phase (Step 9)

The norm operates on reshaped 2D views; the tensors return to 4D afterward.

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| 9 (internal Q reshape) | `q_reshaped` | `[B*16, 128]` = `[16B, 128]` | `[16B, 128]` | L1 | interleaved | bfloat16 |
| 9 (internal K reshape) | `k_reshaped` | `[B*4, 128]` = `[4B, 128]` | `[4B, 128]` | L1 | interleaved | bfloat16 |
| 9 (after rms_norm, Q) | `q_normed` | `[16B, 128]` | `[16B, 128]` | L1 | interleaved | bfloat16 |
| 9 (after rms_norm, K) | `k_normed` | `[4B, 128]` | `[4B, 128]` | L1 | interleaved | bfloat16 |
| 9 (out Q) | `query_states` | `[1, B, 16, 128]` | `[1, B, 16, 128]` | L1 | interleaved | bfloat16 |
| 9 (out K) | `key_states` | `[1, B, 4, 128]` | `[1, B, 4, 128]` | L1 | interleaved | bfloat16 |

### Position / RoPE Setup Phase (Steps 10–13)

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| 10 (host) | `cache_position_tensor` | `[B]` | host only | — | — | torch.int32 |
| 10 (out) | `cur_pos_tt` | `[B]` | `[B]` (full) | DRAM | replicated | int32; ROW_MAJOR |
| 11 (internal) | `pos_ttnn` | `[1, B]` | `[1, B]` | DRAM | replicated | uint32; ROW_MAJOR |
| 11 (out) | `cos_ttnn` | `[1, B, 1, 128]` | `[1, B, 1, 128]` | DRAM | replicated | bfloat16; TILE |
| 11 (out) | `sin_ttnn` | `[1, B, 1, 128]` | `[1, B, 1, 128]` | DRAM | replicated | bfloat16; TILE |
| 12a (out) | `cos_ttnn` | `[1, B, 1, 128]` | shard `(32, 128)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |
| 12b (out) | `sin_ttnn` | `[1, B, 1, 128]` | shard `(32, 128)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |
| 12c (out) | `query_states` | `[1, B, 16, 128]` | shard `(32, 128)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |
| 12d (out) | `key_states` | `[1, B, 4, 128]` | shard `(32, 128)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |
| 13 (out) | `trans_mat` | `[1, 1, B*32, 32]` | shard `(32, 32)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |

### RoPE and KV Cache Write Phase (Steps 14–18)

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| 14 (out) | `query_states` | `[1, B, 16, 128]` | shard `(32, 128)` per core | L1 | HEIGHT_SHARDED | bfloat16 |
| 15 (out) | `key_states` | `[1, B, 4, 128]` | shard `(32, 128)` per core | L1 | HEIGHT_SHARDED | bfloat16 |
| 16a (out) | `key_states` | `[1, B, 4, 128]` | shard `[kv_vol, 128]` per core | L1 | HEIGHT_SHARDED (`kv_mem`) | bfloat16 |
| 16b (out) | `value_states` | `[1, B, 4, 128]` | shard `[kv_vol, 128]` per core | L1 | HEIGHT_SHARDED (`kv_mem`) | bfloat16 |
| 17 (write) | KV cache (DRAM) | per-layer paged storage | — | **DRAM** | paged | bfloat16 |
| 18 | `key_states`, `value_states` | freed | — | — | — | — |

`kv_vol` = `key_states.volume() // padded_head_dim // num_cores` = `(1 * B * 4 * 128) / 128 / B` = 4 elements (batch-invariant).

### SDPA and Output Phase (Steps 19–24)

| Step | Tensor | Logical shape | Per-device shape | Memory | Sharding | Dtype |
|------|--------|---------------|------------------|--------|----------|-------|
| 19 (in Q) | `query_states` | `[1, B, 16, 128]` | shard `(32, 128)` per core | L1 | HEIGHT_SHARDED | bfloat16 |
| 19 (KV read) | KV cache | paged DRAM per layer | — | DRAM | paged | bfloat16 |
| 19 (out) | `attn_output` | `[1, B, 16, 128]` | depends on SDPA config | DRAM or L1 | — | bfloat16 |
| 20 (out) | `attn_output` | `[1, B, 16, 128]` | shard `(32, 128)` per core | **L1** | HEIGHT_SHARDED | bfloat16 |
| 21 (out) | `attn_output` | `[1, 1, B, 2048]` | `[1, 1, B, 2048]` | L1 or DRAM | interleaved | bfloat16 |
| 22 (out) | `attn_output` | `[1, 1, B, 2048]` | `[1, 1, B, 256]` | DRAM | col-sharded | bfloat16 |
| 23 (out) | `attn_output` | `[1, 1, B, 2048]` (trimmed) | `[1, 1, B, 256]` | DRAM | col-sharded | bfloat16 |
| 24 (out) | `attn_output` | `[B, 1, 2048]` | `[B, 1, 256]` | DRAM | col-sharded | bfloat16 |

---

## Why `[1, B, H, D]` (S B H D) Layout is Required for Paged Attention Kernels

The `[1, B, H, D]` layout — called S B H D throughout this guide — places dimensions in
the order `[seq_len=1, batch, num_heads, head_dim]`. This is the standard "decode token"
layout expected by three kernels in the decode path:

### 1. `ttnn.experimental.rotary_embedding_llama` in decode mode

When called with `is_decode_mode=True`, the RoPE kernel interprets dimension 1 as batch
and dimension 2 as the head index. It applies per-position cos/sin values broadcasted
over heads via the `[1, B, 1, D]` cos/sin tensors. If the tensor were in the prefill
layout `[B, H, S, D]`, the kernel would receive `is_decode_mode=False` and interpret
dimensions differently — broadcasting would be over the sequence dimension rather than
the batch dimension.

### 2. `past_key_values.paged_update_on_device`

The paged KV cache kernel writes one K or V slice per batch element per head to the
cache. It expects its input in `[1, B, Hkv, D]` so that it can iterate over batch
dimension 1 and head dimension 2 independently, writing each head's data to the
appropriate paged cache slot determined by `cur_pos_tt[b]`.

### 3. `past_key_values.paged_sdpa_decode`

The paged SDPA decode kernel reads Q in `[1, B, H, D]` format and K/V from the paged
DRAM cache. It attends each query head against the corresponding KV head (GQA: every
`H/Hkv = 4` Q heads attend to the same KV head). Having batch on dimension 1 allows the
kernel to process different batch elements independently on different cores.

### Why not `[B, H, 1, D]`?

The prefill layout `[B, H, S, D]` is the standard for multi-token prefill where S > 1.
In decode, S=1, and paged attention kernels tile over the batch dimension (not the
sequence dimension), allocating one core per batch element. The `[1, B, H, D]` layout
makes the batch dimension the second dimension (`dim=1`), matching how HEIGHT_SHARDED
memory is distributed: when B cores are allocated in a row and the tensor is sharded on
the height dimension, each core receives exactly `[1, 1, H, D]` of data (one decode
token for all heads belonging to that batch element).

The reshape from `[B, 1, H*D]` to `[1, B, H, D]` at step 6 (lines 2644–2646) performs
this conversion in a single metadata-only operation with no data movement, which is one
of the key architectural advantages of the `TTNNBailingMoEAttention` implementation
versus earlier implementations that used `nlp_create_qkv_heads_decode` (which involved
actual data rearrangement).

---

## Memory Location Transition Diagram

The following diagram traces the memory location of each principal tensor through the
decode path. Arrows show transitions; brackets show the triggering op.

```
HIDDEN STATES (input)
  [DRAM, col-sharded]
        |
        |─── [step 1: to_layout, no-op if TILE] ──────────────────────── (unchanged)
        |
        |─── [step 2: q_proj reduce-scatter] ───────────────────────────→ Q [DRAM, col-sharded]
        |                                                                         |
        |─── [step 3: all_gather] ─────────────────────────────────────→ HIDDEN_REPL [DRAM, replicated]
        |                                                                         |
        |   [step 4a: k_proj]  ─────────────────────────────────────────→ K [DRAM, col-sharded]
        |   [step 4b: v_proj]  ─────────────────────────────────────────→ V [DRAM, col-sharded]
        |   [step 4 done: deallocate HIDDEN_REPL]
        |
        |─── [step 5a: all_gather Q] ──────────────────────────────────→ Q [DRAM, replicated]
        |─── [step 5b: all_gather K] ──────────────────────────────────→ K [DRAM, replicated]
        |─── [step 5c: all_gather V] ──────────────────────────────────→ V [DRAM, replicated]
        |
        |─── [step 6: reshape] ────────────────────────────────────────→ Q [DRAM, replicated, 1,B,H,D]
        |                                                                  K [DRAM, replicated, 1,B,Hkv,D]
        |                                                                  V [DRAM, replicated, 1,B,Hkv,D]
        |
Q ──────┤─── [step 8a: to_memory_config L1] ──────────────────────────→ Q [L1, interleaved]
K ──────┤─── [step 8b: to_memory_config L1] ──────────────────────────→ K [L1, interleaved]
V ──────┘    (V stays in DRAM)
        |
Q,K ───── [step 9: _apply_qk_norm (rms_norm in L1)] ─────────────────→ Q [L1, interleaved, normed]
                                                                          K [L1, interleaved, normed]
        |
        |─── [steps 10,11: host torch ops + ttnn.from_torch] ─────────→ cur_pos_tt [DRAM, replicated, int32]
        |                                                                  cos/sin   [DRAM, replicated]
        |
cos ────── [step 12a: to_memory_config HEIGHT_SHARDED] ───────────────→ cos [L1, HEIGHT_SHARDED]
sin ────── [step 12b: to_memory_config HEIGHT_SHARDED] ───────────────→ sin [L1, HEIGHT_SHARDED]
Q ──────── [step 12c: to_memory_config HEIGHT_SHARDED] ───────────────→ Q   [L1, HEIGHT_SHARDED]
K ──────── [step 12d: to_memory_config HEIGHT_SHARDED] ───────────────→ K   [L1, HEIGHT_SHARDED]
        |
Q ──────── [step 14: rotary_embedding_llama Q] ───────────────────────→ Q   [L1, HEIGHT_SHARDED, RoPE applied]
K ──────── [step 15: rotary_embedding_llama K] ───────────────────────→ K   [L1, HEIGHT_SHARDED, RoPE applied]
        |
K ──────── [step 16a: to_memory_config kv_mem] ──────────────────────→ K   [L1, HEIGHT_SHARDED (kv_mem spec)]
V (DRAM)── [step 16b: to_memory_config kv_mem] ──────────────────────→ V   [L1, HEIGHT_SHARDED (kv_mem spec)]
        |
K,V ─────── [step 17: paged_update_on_device] ───────────────────────→ KV CACHE [DRAM, paged storage]
K,V ─────── [step 18: deallocate] ───────────────────────────────────→ freed from L1
        |
Q ──────── [step 19: paged_sdpa_decode] ──────────────────────────────→ attn_output [DRAM or L1]
  KV CACHE (read) ──────────────────────────────────────────────────────↗
        |
attn_output ─── [step 20: to_memory_config sdpa_output_memcfg] ──────→ [L1, HEIGHT_SHARDED (32,D) per core]
        |
        |─── [step 21: nlp_concat_heads_decode] ──────────────────────→ [1, 1, B, d_model]
        |─── [step 22: dense] ─────────────────────────────────────────→ [1, 1, B, d_model], DRAM, col-sharded
        |─── [step 23: slice (conditional)] ──────────────────────────→ [1, 1, B, d_model], DRAM
        |─── [step 24: reshape] ───────────────────────────────────────→ [B, 1, d_model], DRAM, col-sharded
```

### Summary of Memory Location Transitions

The diagram reveals the following pattern:

1. **DRAM → DRAM** (steps 1–7): All projection and all-gather ops work in DRAM. No L1
   involvement.

2. **DRAM → L1 interleaved** (steps 8a/8b): Q and K move to L1 to support QK norm's
   reshape requirement. V stays in DRAM throughout this phase.

3. **L1 interleaved → L1 HEIGHT_SHARDED** (steps 12c/12d): Q and K reshard within L1
   for the RoPE kernel. This is the second within-L1 transition for Q and K.

4. **DRAM → L1 HEIGHT_SHARDED** (step 16b): V makes its first and only L1 transition
   directly into the paged-update shard layout — skipping the interleaved intermediate
   that Q and K passed through.

5. **L1 → DRAM KV cache** (step 17): K and V are written to the persistent on-device
   KV cache. The KV cache is DRAM-resident and persists across decode steps.

6. **L1 freed** (step 18): K and V L1 buffers are returned immediately after the cache
   write.

7. **L1 → DRAM → L1 HEIGHT_SHARDED** (steps 19–20): SDPA reads Q from L1, reads K/V
   from the DRAM KV cache, and produces output that is re-moved to L1 HEIGHT_SHARDED
   for `nlp_concat_heads_decode`.

8. **L1 → DRAM** (steps 21–24): The concat-heads output flows through the dense
   projection and final reshape back to DRAM as the col-sharded output.

---

**Next:** [Chapter 2 — Collective Communication Costs and Sharding Strategy](../chapter_02_collective_communication/index.md)
