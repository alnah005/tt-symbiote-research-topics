# Ling Attention Layer 10 Investigation

## Problem Statement

The Ling-mini-2.0 (BailingMoeV2) model shows accuracy degradation specifically after layer 10 during decode. Layer 0 maintains stable PCC (0.97), but layer 10 degrades first (min PCC 0.53), with layer 19 showing secondary degradation.

**GQA Configuration:**
- Q heads: 16
- KV heads: 4
- Group size: 4 (16 / 4)
- Head dimension: 128
- Total layers: 20

---

## Executive Summary

The layer 10 degradation appears to be caused by **cumulative error accumulation** combined with potential **GQA tensor shape handling differences** between tt-symbiote and tt-transformers. The key differences are:

1. **QKV Head Creation Path**: tt-transformers uses `nlp_create_qkv_heads_decode` directly on fused QKV tensor, while tt-symbiote uses manual reshape/permute operations after separate projections
2. **Memory Configuration**: tt-symbiote creates custom sharding configs while tt-transformers uses optimized pre-configured memory layouts
3. **Tensor Shape Flow**: The SBHD vs BHSD permutations in tt-symbiote may introduce intermediate precision loss

---

## 1. Similar GQA Implementations Found

### tt_transformers/tt/attention.py (Reference Implementation)

The production-quality attention implementation handles GQA with:

```python
# Decode path flow:
# 1. DRAM-sharded QKV matmul
xqkv_fused_sharded = ttnn.linear(x, self.wqkv, ...)

# 2. Convert to L1 interleaved (required for nlp_create_qkv_heads_decode)
xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

# 3. Create QKV heads using dedicated op
q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.n_local_heads,
    num_kv_heads=self.n_local_kv_heads,
    memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),
)

# 4. RoPE using llama fused kernel
q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(...)
k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(...)

# 5. Paged KV cache update (K/V have nkv heads, not nh)
ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)
ttnn.experimental.paged_update_cache(values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)

# 6. Paged SDPA decode
attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q_heads_1BQD, keys, values, page_table_tensor=page_table, cur_pos_tensor=current_pos, ...
)
```

**Key tensor shapes:**
- Q heads: `[1, batch, num_q_heads, head_dim]`
- K/V heads: `[1, batch, num_kv_heads, head_dim]`
- Paged cache: `[max_num_blocks, num_kv_heads, block_size, head_dim]`
- Cache update input: `[batch, num_kv_heads, 1, head_dim]`

### models/common/modules/attention/attention_1d.py

Unified attention for 1D topologies (N150, N300, T3K):

```python
# Uses identical nlp_create_qkv_heads_decode pattern
q_heads_pre_rot, k_heads_pre_rot, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=n_local_heads,
    num_kv_heads=n_local_kv_heads,
    memory_config=cfg.decode_create_qkv_head_memcfg,
)
```

### models/demos/gpt_oss/tt/attention/decode.py

```python
# Same pattern
tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=num_local_heads,
    num_kv_heads=num_local_kv_heads,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
```

---

## 2. Key Differences: tt-symbiote vs tt-transformers

### tt-symbiote/modules/attention.py (TTNNBailingMoEAttention)

```python
# Decode path flow - DIFFERENT from tt-transformers:

# 1. Separate Q, K, V projections (not fused QKV)
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states_replicated)
value_states = self.v_proj(hidden_states_replicated)

# 2. All-gather and reshape manually
query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))

# 3. Permute to BHSD format (not using nlp_create_qkv_heads_decode)
query_states = ttnn.permute(query_states, (0, 2, 1, 3))  # [B, H, S, D]
key_states = ttnn.permute(key_states, (0, 2, 1, 3))

# 4. RoPE (different from rotary_embedding_llama)
query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

# 5. Permute to SBHD for paged kernels
query_states = ttnn.permute(query_states, (2, 0, 1, 3))  # S, B, H, D
key_states = ttnn.permute(key_states, (2, 0, 1, 3))

# 6. Convert to replicated for multi-device (extra round-trip through host)
query_states = self._to_replicated(query_states)
key_states = self._to_replicated(key_states)

# 7. Custom sharding config for KV cache update
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size  # Pads 4 -> 32
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
```

### Critical Differences

| Aspect | tt-transformers | tt-symbiote |
|--------|-----------------|-------------|
| QKV projection | Fused `wqkv` matmul | Separate Q, K, V projections |
| Head creation | `nlp_create_qkv_heads_decode` | Manual reshape + permute |
| RoPE | `rotary_embedding_llama` | `TTNNRotaryPositionEmbedding` |
| Memory config | Pre-optimized configs | Dynamic sharding creation |
| Multi-device | Native mesh support | Host round-trip via `_to_replicated` |
| Permutation count | 1 (after SDPA) | 4+ permutations in decode path |

---

## 3. Layer-Specific Patterns

### No Layer-Specific Logic in Paged Cache

The `TTNNPagedAttentionKVCache` implementation uses layer index only for:
- Indexing into cache lists: `self._tt_key_cache[layer_idx]`
- Sequence length tracking: `self._seq_lengths[layer_idx]`

There is **no layer-dependent computation** that would cause layer 10 specifically to behave differently.

### Cumulative Error Hypothesis

The layer 10 degradation is likely **cumulative error** from:

1. **Repeated permutations**: Each ttnn.permute can introduce small precision loss
2. **Host round-trips**: `_to_replicated` converts tensor to host and back for multi-device
3. **Memory config changes**: Transitioning between sharding strategies
4. **Lack of dedicated decode kernels**: Manual reshape vs optimized `nlp_create_qkv_heads_decode`

Error accumulates through 10 layers of:
```
proj -> reshape -> permute -> rope -> permute -> replicate -> shard -> update -> sdpa -> permute -> reshape -> proj
```

### Why Layer 10 Specifically?

With 20 total layers:
- **Layer 0**: Fresh input, minimal accumulated error - PCC 0.97
- **Layer 10**: ~50% through model, error crosses threshold - PCC drops to 0.53
- **Layer 19**: Near end, maximum accumulated error - secondary degradation

This follows a typical cumulative error pattern where the first half of layers show gradual degradation until a "cliff" is reached.

---

## 4. Specific Issues Identified

### Issue 1: Excessive Permutations

tt-symbiote does 4+ permutations per layer:
```python
# After projection
query_states = ttnn.permute(query_states, (0, 2, 1, 3))  # B,S,H,D -> B,H,S,D
key_states = ttnn.permute(key_states, (0, 2, 1, 3))

# Before paged update
query_states = ttnn.permute(query_states, (2, 0, 1, 3))  # B,H,S,D -> S,B,H,D
key_states = ttnn.permute(key_states, (2, 0, 1, 3))

# After SDPA
attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # S,B,H,D -> B,S,H,D
```

tt-transformers uses `nlp_create_qkv_heads_decode` which outputs the correct shape directly.

### Issue 2: Host Round-Trip for Multi-Device

```python
def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Convert all-gathered topology -> replicated for paged kernels."""
    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
    t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)  # HOST ROUND-TRIP
    return ttnn.from_torch(t_torch, device=self.device, mesh_mapper=ttnn.ReplicateTensorToMesh(self.device), ...)
```

This loses precision and adds latency at every layer.

### Issue 3: Dynamic Sharding Config

tt-symbiote creates sharding config dynamically:
```python
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size  # 4 -> 32
shard_cfg = ttnn.create_sharded_memory_config(shape=(shard_h, self.head_dim), ...)
```

This pads KV heads from 4 to 32 for tile alignment. The guide on "Silent Failure Modes" warns:
> If `nkv_padded / nh_padded != nkv / nh`, the effective group_size changes and every `kv_head_idx = q_head_idx // group_size` lookup returns the wrong KV head.

However, the actual paged kernels should handle this correctly if num_kv_heads is passed correctly.

### Issue 4: RoPE Implementation Differences

tt-transformers uses `rotary_embedding_llama` which is optimized for TTNN tensor layouts.
tt-symbiote uses `TTNNRotaryPositionEmbedding` or `TTNNDistributedRotaryPositionEmbedding`.

Partial rotary (Ling uses 0.5 factor) may behave differently between implementations.

---

## 5. Recommended Fix Approach

### Short-Term: Reduce Permutations

Replace the manual reshape/permute pattern with `nlp_create_qkv_heads_decode`:

```python
# Current (multiple permutes):
query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
query_states = ttnn.permute(query_states, (0, 2, 1, 3))
# ... later
query_states = ttnn.permute(query_states, (2, 0, 1, 3))

# Proposed (single optimized op):
# Fuse Q, K, V projections into single QKV matmul for decode
xqkv_fused = ttnn.linear(hidden_states, wqkv, ...)
q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
```

### Medium-Term: Eliminate Host Round-Trips

Replace `_to_replicated` with native TTNN tensor replication:

```python
# Current (host round-trip):
def _to_replicated(self, tensor):
    t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)  # HOST!
    return ttnn.from_torch(t_torch, device=self.device, mesh_mapper=ttnn.ReplicateTensorToMesh(...))

# Proposed (stay on device):
# Use ttnn.replicate_to_mesh_from_device or ensure tensor topology is correct from the start
```

### Long-Term: Align with tt-transformers Architecture

Refactor `TTNNBailingMoEAttention` to follow the tt-transformers pattern:

1. Use fused QKV projection
2. Use `nlp_create_qkv_heads_decode` for shape transformation
3. Use `rotary_embedding_llama` for RoPE
4. Use pre-configured memory configs
5. Minimize permutations and memory config changes

---

## 6. Validation Tests

### Test 1: Permutation Impact

Compare PCC with and without intermediate permutations:
```python
# A: With multiple permutes (current)
# B: With nlp_create_qkv_heads_decode (proposed)
# Compare PCC at layer 0, 5, 10, 15, 19
```

### Test 2: Host Round-Trip Impact

Compare PCC with and without `_to_replicated`:
```python
# A: Multi-device with host round-trip (current)
# B: Single device (no round-trip)
# If B has higher PCC, the round-trip is contributing to degradation
```

### Test 3: Layer-by-Layer Isolation

Run decode on individual layers with fresh KV cache:
```python
for layer_idx in range(20):
    # Create fresh cache, run single layer
    # Compare TTNN vs PyTorch reference
    # Identify if any specific layer has inherent issues
```

---

## 7. Related Files

- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` - Current problematic implementation
- `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py` - Reference production implementation
- `/home/ttuser/salnahari/tt-metal/models/common/modules/attention/attention_1d.py` - Unified 1D attention
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_decode_diagnostics.py` - Diagnostic tests

---

## 8. Conclusion

The layer 10 accuracy degradation is most likely caused by **cumulative precision loss** from:

1. Multiple ttnn.permute operations (4+ per layer)
2. Host round-trips for multi-device tensor replication
3. Dynamic memory config creation vs pre-optimized configs
4. Lack of dedicated decode kernels (nlp_create_qkv_heads_decode, rotary_embedding_llama)

The fix requires aligning the tt-symbiote attention implementation with the tt-transformers pattern, particularly using:
- Fused QKV projection
- `nlp_create_qkv_heads_decode` for shape transformation
- Native TTNN operations instead of host round-trips
- Pre-configured memory layouts

This is **not** a layer-specific bug in the paged cache or SDPA kernel, but rather an accumulation of small errors through the decode path that crosses a threshold around layer 10.
