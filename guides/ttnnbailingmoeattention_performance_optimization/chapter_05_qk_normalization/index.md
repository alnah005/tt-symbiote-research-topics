# Chapter 5 — QK Normalization: Cost Analysis and Distributed Alternatives

## Role of QK Normalization in Bailing MoE

QK normalization is an element of the Bailing MoE attention architecture that applies RMSNorm independently to each query and key head vector after the Q and K projections and before RoPE is applied. Its purpose is to prevent attention logit explosion: in deep mixture-of-experts architectures where token representations pass through multiple expert layers, the scale of hidden-state activations can grow large enough that the raw QK dot product produces logits far outside the numerically stable range of softmax. Normalizing each head's Q and K to unit scale (modulated by the learned weight) keeps the logit distribution stable throughout training and inference.

The placement in the decode path is as follows (see Chapter 1 `op_sequence.md`):

1. Q and K are projected and all-gathered to produce full `[1, B, H, D]` and `[1, B, Hkv, D]` tensors.
2. Tensors are moved from DRAM to L1 (`ttnn.to_memory_config` at lines 2655–2657 of `attention.py`).
3. `_apply_qk_norm` is called (line 2659 of `attention.py`).
4. RoPE is applied to the normalized Q and K.

The normalization is controlled by the `use_qk_norm` flag read from `config.use_qk_norm` (line 2333 of `attention.py`). When the flag is `False`, `_apply_qk_norm` returns immediately (line 2456–2457). For the Bailing MoE model that is the subject of this guide, `use_qk_norm=True`.

---

## Performance Relevance

At batch=1 decode, Q and K are small: a single new token means the Q tensor has only H=16 head vectors of D=128 elements each, and K has Hkv=4 head vectors of D=128 elements. The norm computation itself is therefore cheap in terms of arithmetic. The performance cost comes not from the compute of `ttnn.rms_norm` but from the memory layout constraint it imposes:

- `ttnn.rms_norm` requires its input to be in L1 interleaved layout (not DRAM, not HEIGHT_SHARDED).
- After the all-gathers that produce Q and K, both tensors reside in DRAM.
- The reshape inside `_apply_qk_norm` from `[1, B, H, D]` to `[B*H, D]` does not work on sharded tensors (line 2655: comment "reshape doesn't work on sharded tensors").

These two requirements together force a DRAM→L1 interleaved copy of both Q and K before the norm can run. This transition is analyzed in depth in Chapter 3 (`transition_analysis.md`, step 1 and step 2). Chapter 5 focuses on the norm operation itself and whether the layout constraint can be removed.

---

## Tensor Shape Reference

Bailing MoE parameters: H=16, Hkv=4, D=128, d_model=2048, N=8 devices.

The shapes below reflect the S B H D layout (`[seq, batch, heads, head_dim]`) that the paged attention kernels require, established by the reshape at line 2644–2646 of `attention.py`.

### Full (logical) tensor shapes entering `_apply_qk_norm`

| Tensor | Full shape | Elements (B=32) | Bytes (bfloat16, B=32) |
|---|---|---|---|
| `query_states` (Q) | `[1, B, 16, 128]` | 1 × B × 16 × 128 = B × 2048 | B × 4096 = **131,072 bytes** |
| `key_states` (K) | `[1, B, 4, 128]` | 1 × B × 4 × 128 = B × 512 | B × 1024 = **32,768 bytes** |

At B=1 (single-token decode) the tensors are 4,096 bytes (Q, 2 tiles) and 1,024 bytes (K, below one tile). At B=32 (maximum typical batch) they are 128 KB and 32 KB respectively.

### Per-device shapes (post-all-gather, N=8 devices)

After `_maybe_all_gather`, each of the N=8 devices holds the **full** Q and K tensors — the all-gather has replicated data from all devices onto every device. There is no per-device sharding of Q or K at this point in the decode path.

| Tensor | Shape per device | Bytes per device (B=32, bfloat16) |
|---|---|---|
| `query_states` per device | `[1, B, 16, 128]` | 131,072 bytes |
| `key_states` per device | `[1, B, 4, 128]` | 32,768 bytes |

The norm weights (`query_layernorm.tt_weight` and `key_layernorm.tt_weight`) are shape `[32, D]` = `[32, 128]` (the first dimension is expanded to TILE=32 for tile-layout compatibility; see `TTNNRMSNorm.preprocess_weights_impl`, line 85 of `normalization.py`). They are replicated to all N=8 devices during `move_weights_to_device_impl` (line 90 of `normalization.py`). Each device independently runs the full norm on the full tensor.

### Reshape dimensions inside `_apply_qk_norm` (decode mode)

Inside `_apply_qk_norm` (lines 2464–2468 of `attention.py`), the reshape flattens the batch and head dimensions:

| Tensor | Shape before reshape | Shape after reshape (norm input) |
|---|---|---|
| Q | `[1, B, 16, 128]` | `[B*16, 128]` = `[B*H, D]` |
| K | `[1, B, 4, 128]` | `[B*4, 128]` = `[B*Hkv, D]` |

At B=32: Q norm input is `[512, 128]`, K norm input is `[128, 128]`. The norm operates over the last dimension (D=128), independently for each of the B*H or B*Hkv rows. This is equivalent to applying per-head RMSNorm to every head vector in the batch simultaneously.

---

## Files in This Chapter

- [**`current_implementation.md`**](current_implementation.md) — Walk through `_apply_qk_norm` (lines 2454–2493), `TTNNRMSNorm` internals, the "head_dim too small to shard" constraint, the DRAM→L1 transition cost, and the typecast ops.
- [**`distributed_alternative.md`**](distributed_alternative.md) — Why `TTNNDistributedRMSNorm` does not apply here, the pre-all-gather Q norm alternative, the K norm constraint, fusion opportunities, and a summary comparison table.
