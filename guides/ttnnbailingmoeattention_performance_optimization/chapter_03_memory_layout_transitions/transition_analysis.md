# Transition Analysis — All 9 `ttnn.to_memory_config` Calls

This file provides a per-transition analysis of every `ttnn.to_memory_config` call in `_forward_decode_paged` (lines 2610–2799 of `attention.py`). For each transition the analysis covers: the data volume, the direction of movement (DRAM→L1, L1→L1, or DRAM→DRAM), the kernel constraint that forces it, and whether it is avoidable.

Conventions used throughout: B = batch size (up to 32 after TTNN tile-boundary padding), H = 16 (num Q heads), Hkv = 4 (num KV heads), D = 128 (head_dim), TILE = 32.

---

## Steps 8a and 8b — Q and K from DRAM to L1 interleaved (lines 2656–2657)

```python
query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)   # line 2656
key_states   = ttnn.to_memory_config(key_states,   ttnn.L1_MEMORY_CONFIG)   # line 2657
```

**Context.** Q and K arrive here after three prior operations: `_maybe_all_gather` (line 2631–2632) which reconstitutes full Q/K from sharded projections; `ttnn.reshape` (lines 2644–2645) which reorganizes them to decode layout `[1, B, H, D]` and `[1, B, Hkv, D]`; and `ttnn.typecast` (lines 2649–2653) which ensures bfloat16. The all-gather and reshape produce interleaved DRAM tensors.

**Direction.** DRAM → L1 interleaved.

**Data volume.**
- Q: `1 * B * H * D * 2 bytes` = `1 * 32 * 16 * 128 * 2` = **131,072 bytes** per device
- K: `1 * B * Hkv * D * 2 bytes` = `1 * 32 * 4 * 128 * 2` = **32,768 bytes** per device

**Why it is required.** The QK norm is implemented in `TTNNRMSNorm.forward` (normalization.py, line 92–96), which calls `ttnn.rms_norm`. The reshape inside `_apply_qk_norm` (attention.py, line 2467–2468) flattens Q to `[B*H, D]` = `[512, 128]` before the norm call. `ttnn.rms_norm` does not support sharded or DRAM-resident input for this flatten+norm pattern: the kernel requires the tensor to be in L1 interleaved layout. The comment on line 2655 reads "Move to L1 for QK norm (reshape doesn't work on sharded tensors)", confirming that the actual blocker is the reshape inside `_apply_qk_norm`, not the norm kernel itself. Sharded tensors cannot be reshaped by `ttnn.reshape` because the shard boundaries would be violated by a rank-changing reshape.

**Avoidability.** These two transitions are avoidable. The root cause is architectural: Q and K enter the QK norm path already fully replicated (post-all-gather, interleaved DRAM) and must be moved to L1 to enable the reshape. If QK norm were applied before the all-gather — while Q and K are still per-device col-sharded reduce-scatter outputs — the norm could run on smaller per-device tensors without a DRAM→L1 copy. See [avoidable_transitions.md](avoidable_transitions.md) for the detailed proposal.

---

## Steps 12a and 12b — cos and sin from DRAM to HEIGHT_SHARDED L1 (lines 2708–2709)

```python
cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)   # line 2708
sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)   # line 2709
```

where `rope_shard_mem` is created just above (lines 2701–2707) with:
- `shape=(TILE_SIZE, head_dim)` = `(32, 128)` per core
- `strategy=ttnn.ShardStrategy.HEIGHT`
- `core_grid` = `batch_grid` = `num_cores_to_corerangeset(batch_size, ...)`

**Context.** cos and sin are produced by `BailingRotarySetup.get_cos_sin_for_decode` (rope.py, lines 420–472). That method calls `ttnn.embedding` against the pre-cached row-major cos/sin tables (`cos_cache_row_major`, `sin_cache_row_major`) using the current position indices, then reshapes and transposes to `[1, B, 1, D]`. Both `ttnn.embedding` calls specify `memory_config=ttnn.DRAM_MEMORY_CONFIG` (rope.py, lines 456–463), so the outputs land in DRAM. The subsequent `ttnn.unsqueeze_to_4D` and `ttnn.transpose` do not change the memory location.

**Direction.** DRAM → L1 HEIGHT_SHARDED.

**Data volume.**
- cos: `[1, B, 1, D]` = `1 * 32 * 1 * 128 * 2` = **8,192 bytes** per device
- sin: same = **8,192 bytes** per device

**Why it is required.** `ttnn.experimental.rotary_embedding_llama` with `is_decode_mode=True` requires its cos and sin inputs to be HEIGHT_SHARDED in L1 with the same shard spec as Q and K. This is a hard kernel precondition: the decode-mode RoPE kernel dispatches one tile per core, reading the corresponding cos/sin shard directly from each core's L1. Presenting DRAM-interleaved cos/sin would require the kernel to perform its own gather, which is not supported in `is_decode_mode=True`.

**Avoidability.** Partially avoidable. The cos/sin tensors are small (8 KB each) so the copy cost is minimal. The DRAM residency comes from `ttnn.embedding` always writing to DRAM; this could be changed if `ttnn.embedding` supported an output memory config parameter. Alternatively, the two cos/sin copies could be fused into the same dispatch as the Q/K reshards (steps 12c/12d) to reduce scheduling overhead, but they cannot be eliminated entirely unless the RoPE kernel is changed to accept DRAM input in decode mode.

---

## Steps 12c and 12d — Q and K from L1 interleaved to HEIGHT_SHARDED (lines 2718, 2727)

```python
query_states = ttnn.to_memory_config(query_states, q_shard_mem)   # line 2718
key_states   = ttnn.to_memory_config(key_states,   k_shard_mem)   # line 2727
```

`q_shard_mem` uses `shape=(TILE_SIZE, head_dim)` = `(32, 128)`. `k_shard_mem` uses `shape=(TILE_SIZE, key_states.shape[-1])` — since K's last dim after `ttnn.rms_norm` remains 128, this evaluates to `(32, 128)` as well.

**Context.** Q and K are now in L1 interleaved layout, having passed through `_apply_qk_norm` (line 2659). The norm output from `TTNNRMSNorm.forward` is L1 interleaved (ttnn.rms_norm writes its output to the same L1 space). The reshapes back to `[1, B, H, D]` and `[1, B, Hkv, D]` inside `_apply_qk_norm` (lines 2487–2488) keep the tensor in L1 interleaved.

**Direction.** L1 interleaved → L1 HEIGHT_SHARDED. This is an intra-L1 reshard over the NoC.

**Data volume.**
- Q: **131,072 bytes** per device (same shape as step 8a; the reshard copies the full tensor)
- K: **32,768 bytes** per device

**Why it is required.** `ttnn.experimental.rotary_embedding_llama` with `is_decode_mode=True` requires HEIGHT_SHARDED input for Q and K. This is the same kernel constraint as steps 12a/12b. The shard spec `(TILE_SIZE, head_dim)` maps each decode token's head slice to one core; the RoPE kernel then processes each core's shard independently using the corresponding cos/sin shard.

**Avoidability.** Yes. This transition is the second of a two-step redundancy (steps 8a/8b followed by steps 12c/12d). Q is copied from DRAM to L1 at step 8a, normalized, then copied from L1 to HEIGHT_SHARDED at step 12c. If the QK norm could accept HEIGHT_SHARDED input directly, steps 8a/8b would be replaced by a single DRAM→HEIGHT_SHARDED transition that feeds both the norm and RoPE without an intermediate L1 copy. Alternatively, if QK norm is relocated before the all-gather (as discussed in [avoidable_transitions.md](avoidable_transitions.md)), Q and K could emerge from the norm already in a shard-compatible layout, making step 12c/12d unnecessary or reducible to a zero-copy reshape.

---

## Steps 16a and 16b — K and V to kv_mem HEIGHT_SHARDED (lines 2753–2754)

```python
key_states   = ttnn.to_memory_config(key_states,   kv_mem)   # line 2753
value_states = ttnn.to_memory_config(value_states, kv_mem)   # line 2754
```

`kv_mem` is constructed just above (lines 2743–2751):

```python
kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
kv_shard = ttnn.ShardSpec(shard_grid, [kv_vol, key_states.padded_shape[-1]], ttnn.ShardOrientation.ROW_MAJOR)
kv_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_shard)
```

With `num_cores = batch_size = B` and K shape `[1, B, Hkv, D]`:
- `key_states.volume()` = `B * Hkv * D` (the leading 1 is a virtual batch dim)
- `key_states.padded_shape[-1]` = `128` (D, already tile-aligned)
- `kv_vol` = `(B * Hkv * D) / D / B` = `Hkv` = **4**

So each core's shard is `[4, 128]` — four KV heads worth of data for one batch element. Compare this to the RoPE shard spec at step 12d: `(TILE_SIZE=32, D=128)` = `[32, 128]` — one full TILE height × head_dim, which spans multiple batch elements' heads. These two shard specs are incompatible: the RoPE shard is organized around TILE_SIZE rows per core, while the paged_update shard is organized around `Hkv` rows per core (which equals 4, less than TILE_SIZE=32).

**Direction (step 16a).** L1 HEIGHT_SHARDED → L1 HEIGHT_SHARDED (reshard).

**Direction (step 16b).** DRAM interleaved → L1 HEIGHT_SHARDED.

**Data volume.**
- K (step 16a): **32,768 bytes** — third copy of K this decode step (after steps 8b and 12d)
- V (step 16b): **32,768 bytes** — first and only L1 copy of V

**Why step 16a is required.** The `paged_update_on_device` kernel (called at line 2756) has a fixed expectation about the shard spec of its K input: each core must hold exactly `Hkv` rows of the `[1, B, Hkv, D]` tensor. The RoPE output at step 12d uses `(32, D)` shards — this aligns TILE_SIZE rows per core regardless of Hkv. When `Hkv=4` and TILE_SIZE=32, the two shard specs differ by a factor of 8.

**Why step 16b is required.** V received no prior L1 ops. It entered `_forward_decode_paged` col-sharded, was gathered and reshaped to DRAM interleaved at lines 2633 and 2646, but was never moved to L1 because QK norm and RoPE do not apply to V. Step 16b is V's only layout transition and is unavoidable in the current architecture.

**Avoidability of step 16a.** Yes, conditionally. If the RoPE output shard spec for K could be configured to `(kv_vol=Hkv, D)` instead of `(TILE_SIZE, D)`, then K at the end of step 12d would already be in `kv_mem` format and step 16a would be a no-op. The feasibility depends on whether `rotary_embedding_llama` in decode mode accepts a shard height that is not equal to TILE_SIZE. See [avoidable_transitions.md](avoidable_transitions.md) for the detailed analysis.

---

## Step 20 — attn_output from DRAM to HEIGHT_SHARDED (line 2783)

```python
sdpa_output_memcfg = ttnn.create_sharded_memory_config(
    shape=(32, self.head_dim),
    core_grid=ttnn.CoreGrid(y=1, x=batch_size),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)   # line 2783
```

**Context.** `paged_sdpa_decode` (line 2766) produces its output in DRAM. The output shape is `[1, B, H, D]` = `[1, 32, 16, 128]`. The shard spec here uses `CoreGrid(y=1, x=batch_size)` rather than the `num_cores_to_corerangeset` grid used for RoPE — this explicitly maps one core per batch element in a 1×B grid arrangement.

**Direction.** DRAM → L1 HEIGHT_SHARDED.

**Data volume.** `1 * B * H * D * 2` = **131,072 bytes** per device — same volume as Q in steps 8a and 12c.

**Why it is required.** `ttnn.experimental.nlp_concat_heads_decode` (line 2785) requires its input in HEIGHT_SHARDED L1. This kernel concatenates attention heads for all batch elements into `[1, 1, B, d_model]` and expects each core to hold one batch element's complete head data contiguously.

**Avoidability.** Potentially. If `paged_sdpa_decode` could be configured to write its output directly into `sdpa_output_memcfg` HEIGHT_SHARDED L1 rather than DRAM, step 20 would be eliminated. This would require `paged_sdpa_decode` to accept an output memory config parameter analogous to the `memory_config` argument that many other TTNN ops support. The SDPA output is the largest tensor copied in any single step 20 transition (131 KB), so this is the highest-value single elimination after the Q transition at step 8a/12c.

---

## Symmetry Summary: The Three Lives of K

K undergoes the most layout transitions of any tensor in the decode path:

K is the most-copied tensor: three complete copies totalling 98,304 bytes (steps 8b → 12d → 16a). V is copied once; Q is copied twice (steps 8a, 12c), and the SDPA output `attn_output` adds a third 131 KB DRAM→L1 move at step 20.

---

**Next:** [avoidable_transitions.md](avoidable_transitions.md)
