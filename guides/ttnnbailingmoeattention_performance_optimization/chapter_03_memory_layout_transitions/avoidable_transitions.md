# Avoidable Transitions — Deep-Dive

This file takes the three categories of avoidable transitions identified in [transition_analysis.md](transition_analysis.md) and works through each in detail: what makes them avoidable, what would be required to eliminate them, and what trade-offs or risks are involved.

---

## 1. The QK-Norm-Induced DRAM → L1 Transitions (Steps 8a / 8b)

### Current path

After the all-gather and reshape, Q and K arrive in DRAM interleaved with shapes `[1, B, H, D]` and `[1, B, Hkv, D]`. Two explicit copies move them to L1:

```
_maybe_all_gather(query_states)   → DRAM interleaved [1, B, 16, 128]
ttnn.reshape(..., (1, B, 16, 128))
ttnn.typecast(..., bfloat16)
ttnn.to_memory_config(query_states, L1_MEMORY_CONFIG)     # line 2656 — copy 1
_apply_qk_norm(query_states, key_states)                  # rms_norm in L1
ttnn.to_memory_config(query_states, q_shard_mem)          # line 2718 — copy 2
```

The comment on line 2655 states: "Move to L1 for QK norm (reshape doesn't work on sharded tensors)". The actual constraint is that `ttnn.reshape` in `_apply_qk_norm` (line 2467) flattens Q from `[1, B, H, D]` to `[B*H, D]` = `[512, 128]`. Sharded tensors cannot be rank-changed by `ttnn.reshape` because the shard grid mapping becomes undefined when the number of dimensions changes. Interleaved L1 does support this reshape; hence the DRAM→L1 copy.

### Why distributed RMSNorm is not the solution

`TTNNDistributedRMSNorm` (normalization.py, lines 99–151) uses `rms_norm_pre_all_gather` → `all_gather_async` → `rms_norm_post_all_gather`. This design targets the case where the input tensor is split across devices along the reduction dimension (typically the hidden-size dimension of a full hidden state `[B, S, d_model/N]`). After the all-gather in `_forward_decode_paged`, Q and K are already fully replicated on every device — each device holds the complete `[1, B, H, D]` tensor. Re-splitting Q after an all-gather to use distributed norm would undo the gather, producing no net benefit.

### Pre-all-gather QK norm: feasibility analysis

The alternative is to apply QK norm before the all-gather, while Q is still in the col-sharded reduce-scatter output from `TTNNLinearIColShardedWRowSharded.q_proj`. At that point, per-device Q has shape:

```
[B, 1, (H * D) / N]  per device  =  [B, 1, (16 * 128) / 8]  =  [B, 1, 256]
```

The reduce-scatter output from `q_proj` is col-sharded, meaning each device holds 2 of the 16 query heads worth of data concatenated along the last dimension: `H/N * D` = 2 × 128 = 256 elements per token per batch element.

RMSNorm for QK normalization reduces over `head_dim=128` (it normalizes each head independently). With 2 heads per device each of width 128, the per-head norm is entirely intra-device — no cross-device reduction is needed. The reshape before the norm would be:

```python
# Per-device pre-all-gather Q shape: [B, 1, 256]  (2 heads × head_dim=128)
# For QK norm: flatten to [B * (H/N), D] = [B * 2, 128] = [2B, 128]
q_reshaped = ttnn.reshape(query_states, (batch_size * (num_heads // num_devices), head_dim))
```

This reshape is rank-preserving (3D → 2D from 3D input), but the input is col-sharded, not rank-2. The same reshape-on-sharded constraint applies unless the col-sharded tensor is first moved to interleaved L1.

However, the data volume is now much smaller. Pre-all-gather, each device holds 256 elements per batch element (2 heads) versus 2048 elements (16 heads) post-all-gather. Moving `[B, 1, 256]` to L1 costs:

```
B * 1 * 256 * 2  =  32 * 256 * 2  =  16,384 bytes  (per device)
```

versus the current `B * H * D * 2 = 131,072 bytes` for full Q post-all-gather. This is an 8× reduction in data volume for the DRAM→L1 copy, matching the factor-of-N reduction from tensor parallelism.

For K the situation requires additional analysis. K is produced by `TTNNLinearIReplicatedWColSharded.k_proj`. After the K/V all-gather (`_maybe_all_gather(key_states)` at line 2632), each device holds the full `[B, 1, Hkv * D] = [B, 1, 512]` K tensor. Before the all-gather, K from `k_proj` is col-sharded with `Hkv * D / N = 4 * 128 / 8 = 64` elements per token per batch element per device — only 64 raw elements. However, because `Hkv=4 < N=8` (there are more devices than KV heads), col-sharding does not produce complete per-head slices on each device: under head-aligned sharding only 4 of 8 devices would hold any head data at all; under element-aligned sharding all 8 devices hold partial head data (64 of 128 elements per head). In neither case is the pre-all-gather slice a semantically complete KV head. RMSNorm must normalize over the full `head_dim=128` for each head independently; operating on a 64-element partial-head slice would not preserve per-head normalization semantics. Therefore, while the pre-all-gather QK norm proposal is straightforwardly feasible for Q (H=16 > N=8 means each device holds 2 complete Q heads and the per-head norm is entirely intra-device), it is non-trivial for K without architectural changes: the norm would operate on partial head slices, which changes the normalization semantics and may produce incorrect results. Eliminating steps 8b and 12d for K requires either a K-specific pre-all-gather norm design that accounts for partial heads, or a different sharding strategy for `k_proj`.

### The `head_dim too small` comment

Line 2380 of `attention.py` reads:

```python
# QK norms use non-distributed version (head_dim too small to shard across devices)
```

This comment concerns `TTNNDistributedRMSNorm`, not the pre-all-gather approach described here. `TTNNDistributedRMSNorm` shards the reduction dimension (`head_dim=128`) across N=8 devices, giving `128 / 8 = 16` elements per device — below the minimum tile width of TILE=32. This is why distributed norm is not used for QK norm. The pre-all-gather approach does not shard `head_dim` across devices; it simply runs the norm on a smaller per-device tensor where each device already holds complete heads. There is no tile-width violation.

### Proposed change summary

1. Before calling `_maybe_all_gather(query_states)`, move Q to L1 interleaved (cost: 16,384 bytes per device instead of 131,072).
2. Apply QK norm on the col-sharded/interleaved per-device Q.
3. Proceed with `_maybe_all_gather` on the normalized Q.
4. After the all-gather, Q is fully replicated and ready for RoPE; steps 8a and 12c reduce from two copies to one (the post-norm, pre-all-gather L1 copy).

The key risk is that `TTNNRMSNorm`'s reshape from `[B, 1, 256]` to `[2B, 128]` must work on L1 interleaved tensors, which it does. The norm weights are already replicated on all devices (`move_weights_to_device_impl` in normalization.py uses `ttnn.to_device` with no mesh decomposition), so they remain valid before and after the all-gather restructuring.

---

## 2. The Two-Step K Re-Sharding After RoPE (Steps 12d → 16a)

### Current path

K passes through two consecutive HEIGHT_SHARDED configurations:

```
ttnn.to_memory_config(key_states, k_shard_mem)    # line 2727 — RoPE shard spec (32, 128)
rotary_embedding_llama(key_states, ...)            # RoPE in decode mode
ttnn.to_memory_config(key_states, kv_mem)          # line 2753 — paged_update spec (4, 128)
```

### Why the two shard specs differ

**RoPE shard spec (`k_shard_mem`, line 2720–2726):**

```python
k_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, key_states.shape[-1]),   # (32, 128)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)
```

`ttnn.TILE_SIZE = 32`. Each core receives a `[32, 128]` shard. For K with shape `[1, B, Hkv, D]` = `[1, 32, 4, 128]`, the total element count is `32 * 4 * 128 = 16,384` elements. With `batch_size=32` cores, each core holds `16,384 / 32 = 512` elements = `[32, 128]` = one TILE row × head_dim. This gives each core exactly one tile row regardless of Hkv.

**paged_update shard spec (`kv_mem`, lines 2743–2751):**

```python
kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
# = (32 * 4 * 128) // 128 // 32  =  16384 // 128 // 32  =  128 // 32  =  4
kv_shard = ttnn.ShardSpec(shard_grid, [kv_vol, key_states.padded_shape[-1]], ...)
# = ShardSpec(..., [4, 128], ...)
```

Each core receives `[4, 128]` — the four KV heads for one batch element. This shard spec is physically meaningful for paged cache update: each core writes Hkv contiguous rows of the KV cache for the corresponding batch element, matching the paged cache layout.

The incompatibility is fundamental: TILE_SIZE=32 versus Hkv=4. The RoPE kernel uses TILE_SIZE as the natural processing unit (tiles are 32×32 in TTNN on Wormhole), while the paged cache update kernel organizes data by KV-head count per batch element.

### Whether paged_update's shard spec can be matched at RoPE output time

For the two shard specs to be compatible, `rotary_embedding_llama` in decode mode would need to accept `(Hkv, D)` = `(4, 128)` as the shard shape instead of `(TILE_SIZE, D)` = `(32, 128)`. The decode-mode RoPE kernel processes one TILE_SIZE × head_dim block per core; a shard height of 4 is smaller than one tile and would require the kernel to handle sub-tile shards or pad the shard to TILE_SIZE internally.

Whether `rotary_embedding_llama` supports non-tile-aligned shard heights in decode mode is a kernel implementation detail. If it does — or can be made to — then the following single shard spec would work for both RoPE and paged_update:

```python
# Unified shard spec for K: (Hkv, D) per core
# Works if rotary_embedding_llama accepts shard_height < TILE_SIZE
k_unified_shard = ttnn.create_sharded_memory_config(
    shape=(Hkv, D),   # (4, 128)
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)
```

If the kernel does not support this, an alternative is to configure paged_update to accept the TILE-aligned `(32, 128)` shard. This is more likely to require changes on the cache-update kernel side. Either way, one of the two HEIGHT_SHARDED copies of K could be eliminated.

**Net saving.** Eliminating step 16a saves **32,768 bytes** of L1-to-L1 NoC traffic per decode step for K.

---

## 3. Step 20 — attn_output SDPA Output Reshard

### Current path

```
paged_sdpa_decode(query_states, ...)   → DRAM interleaved [1, B, H, D]
ttnn.to_memory_config(attn_output, sdpa_output_memcfg)   # line 2783 — DRAM → L1 HEIGHT_SHARDED
nlp_concat_heads_decode(attn_output, ...)
```

The `sdpa_output_memcfg` uses a `CoreGrid(y=1, x=batch_size)` rather than `num_cores_to_corerangeset`, creating a 1×B grid. The shard shape is `(32, head_dim)` = `(32, 128)`.

### Why it is potentially avoidable

`paged_sdpa_decode` currently writes its output unconditionally to DRAM interleaved. If the kernel accepted an `output_memory_config` argument — analogous to the `memory_config` parameter available on many standard TTNN ops — it could write directly to the `sdpa_output_memcfg` HEIGHT_SHARDED L1 buffer, eliminating the 131 KB DRAM→L1 copy.

Eliminating all avoidable transitions saves approximately 491,520 bytes (~480 KB) of NoC data movement per decode step per device — roughly 91% of the total 528 KB. Step 20 is the single highest-value elimination (131 KB, DRAM→L1).

The `paged_sdpa_decode` output config change is self-contained: it requires only that `past_key_values.paged_sdpa_decode` pass an output memory config to the underlying `ttnn.paged_scaled_dot_product_attention_decode` kernel. No changes to surrounding ops are needed. This makes it the lowest-risk avoidable transition to address.

---

## Combined Optimization Impact

Applying all three sets of changes in order of complexity:

1. **Step 20 (attn_output reshard)** — Add `output_memory_config=sdpa_output_memcfg` to `paged_sdpa_decode` call. No architectural change. Risk: low. Saves 131 KB.

2. **Step 16a (K re-reshard)** — Unify RoPE and paged_update shard specs for K. Requires verifying that `rotary_embedding_llama` accepts `(Hkv, D)` shards or that paged_update accepts `(TILE_SIZE, D)` shards. Risk: medium (kernel precondition). Saves 32 KB.

3. **Steps 8a/8b and 12c/12d (QK norm path)** — Move QK norm before the all-gather. Requires restructuring the `_forward_decode_paged` op ordering, validating that `TTNNRMSNorm` operates correctly on pre-all-gather col-sharded tensors after an L1 interleaved copy of the smaller pre-gather tensor. Risk: medium (correctness validation needed; norm must produce identical numerics). Saves 320 KB across Q and K (steps 8a + 8b + 12c + 12d combined), reduced to 20 KB if the pre-gather copies substitute for the post-gather ones.

Together, items 1 and 3 account for the bulk of the savings (steps 8a, 8b, 12c, 12d, and 20 together account for 448 KB of the 528 KB total).

---

**Next:** [Chapter 4 — Host-Device Round-Trips and On-Device Alternatives](../chapter_04_host_device_roundtrips/index.md)
