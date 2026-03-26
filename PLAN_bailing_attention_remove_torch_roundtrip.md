# Plan: Remove ttnn<->torch Round-Trip Conversions from TTNNBailingMoEAttention

## 1. Problem Description

`TTNNBailingMoEAttention._forward_decode_paged` (in `models/experimental/tt_symbiote/modules/attention.py`) has two unnecessary host round-trips:

### Round-trip 1: `_to_replicated` (line 2648)
After all-gathering Q/K/V and fusing them into `xqkv_fused`, the method calls `_to_replicated()` which:
1. `ttnn.to_torch(t, mesh_composer=ConcatMeshToTensor)` -- device to host
2. `ttnn.from_torch(t_torch, mesh_mapper=ReplicateTensorToMesh)` -- host to device

This exists because `nlp_create_qkv_heads_decode` and downstream paged attention kernels require "replicated" mesh topology metadata, but `ttnn.all_gather` produces "all-gathered" topology metadata (the data is identical on every device, but the metadata differs).

### Round-trip 2: `cache_position` handling (lines 2666-2688)
When `cache_position` arrives as a `ttnn.Tensor` (from the model runner), it is:
1. Converted to torch via `ttnn.to_torch(cp, mesh_composer=ConcatMeshToTensor)`
2. Sliced to `[:batch_size]`
3. Converted back via `ttnn.from_torch(..., mesh_mapper=ReplicateTensorToMesh)`

## 2. Root Cause Analysis

### Why Round-trip 1 exists
The sharding strategy in `TTNNBailingMoEAttention` differs fundamentally from tt-transformers:

**tt-transformers approach (no round-trip needed):**
- QKV weights are **column-parallel** (sharded along output dim across 8 devices)
- Each device computes only its **local** heads: `num_local_heads = mesh_config.shard_size(num_heads)`
- E.g., device 0 computes 2 of 16 Q heads, 0.5 of 4 KV heads
- `nlp_create_qkv_heads_decode(xqkv, num_heads=num_local_heads, ...)` works directly
- Paged attention operates on local heads -- no topology issue

**tt-symbiote BailingMoEAttention approach (causes round-trip):**
- Q uses `TTNNLinearIColShardedWRowSharded` (input col-sharded, weight row-sharded, with reduce_scatter)
- K/V use `TTNNLinearIReplicatedWColSharded` (input replicated, weight col-sharded)
- Then `_maybe_all_gather` collects **all** Q/K/V data on every device
- `ttnn.all_gather` output has "all-gathered" topology metadata
- `nlp_create_qkv_heads_decode` gets `num_heads=16` (all heads, not local)
- Paged attention kernels reject the all-gathered topology, requiring replicated

**Core issue:** The current sharding strategy produces outputs with all-gathered topology, while paged attention kernels require replicated topology. There is no TTNN API to change topology metadata without a host round-trip.

### Why Round-trip 2 exists
`cache_position` arrives as `Optional[torch.LongTensor]` per the type hint, but in practice the model runner may pass it as a `ttnn.Tensor`. The code defensively handles both cases. When it is ttnn, the only way to slice it and re-upload with replicated topology is via host.

## 3. Step-by-Step Implementation Plan

### Approach: Adopt tt-transformers Column-Parallel QKV Pattern

The recommended approach is to **restructure QKV projection to use column-parallel sharding** (like tt-transformers), so each device computes only its local heads. This eliminates the need for all-gather and the topology mismatch entirely.

### Step 1: Change QKV Weight Sharding in `from_torch` (lines 2374-2376)

**Current:**
```python
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)  # reduce-scatter output
new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)  # col-sharded weight
new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)  # col-sharded weight
```

**Target:** Use a single fused QKV linear with column-parallel output sharding (matching tt-transformers pattern):
```python
# Fuse Q/K/V into single weight, sharded along output dim across devices
qkv_linear = nn.Linear(hidden_size, q_size + 2 * kv_size, bias=has_bias)
qkv_linear.weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)
if has_bias:
    qkv_linear.bias.data = torch.cat([q_bias, k_bias, v_bias], dim=0)
new_attn.qkv_proj = TTNNLinearColumnParallel.from_torch(qkv_linear)
```

**Risk:** A new `TTNNLinearColumnParallel` class may be needed, or the existing `TTNNLinearIReplicatedWColSharded` can be reused if it shards the output dim correctly. The weight is [out_features, in_features], and col-sharding on dim=-1 of the transposed weight means each device gets `out_features/num_devices` rows.

**Alternative (lower risk):** Keep the existing three separate projections but use column-parallel sharding for all three, so each device produces local heads directly. This avoids fusing QKV but still eliminates the all-gather.

### Step 2: Remove `_maybe_all_gather` calls (lines 2631-2633)

With column-parallel QKV, each device already has its local heads. Remove:
```python
query_states = self._maybe_all_gather(query_states)
key_states = self._maybe_all_gather(key_states)
value_states = self._maybe_all_gather(value_states)
```

### Step 3: Update `nlp_create_qkv_heads_decode` call (line 2651)

**Current:**
```python
query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.num_heads,        # 16 (all heads)
    num_kv_heads=self.num_kv_heads,  # 4 (all KV heads)
)
```

**Target:**
```python
num_devices = self.device.get_num_devices()
num_local_heads = self.num_heads // num_devices      # 2 per device
num_local_kv_heads = self.num_kv_heads // num_devices  # 0.5 per device -> need care

query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=num_local_heads,
    num_kv_heads=num_local_kv_heads,
)
```

**IMPORTANT CAVEAT:** Ling 2.0 has `num_kv_heads=4` with 8 devices, giving 0.5 KV heads per device. This means KV heads cannot be evenly sharded across 8 devices. There are two sub-options:
- **Option A:** Run on fewer devices (e.g., 4) where KV heads divide evenly
- **Option B:** Use GQA-aware sharding where some devices share KV heads (replicate KV heads across device pairs). tt-transformers handles this via `mesh_config.shard_size()` which may round up.

### Step 4: Remove `_to_replicated` call (line 2648)

With local heads per device (column-parallel output), the topology is already correct. Remove:
```python
xqkv_fused = self._to_replicated(xqkv_fused)
```

### Step 5: Update `nlp_concat_heads_decode` (line 2788)

**Current:**
```python
attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.num_heads)
```

**Target:**
```python
attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=num_local_heads)
```

### Step 6: Add all-reduce after output projection

With column-parallel attention, the output projection `dense` (currently `TTNNLinearIReplicatedWColSharded`) needs to become row-parallel, followed by an all-reduce to combine partial sums from each device. This matches the tt-transformers pattern:
```python
tt_out = ttnn.linear(tt_sdpa_out, weights.o_proj, ...)
tt_out = apply_allreduce(tt_out, ...)  # Sum partial results across devices
```

### Step 7: Fix `cache_position` handling (lines 2666-2688)

**Option A (preferred):** Ensure the model runner always passes `cache_position` as `torch.Tensor`. Then the code simplifies to:
```python
if cache_position is None:
    cur_pos = past_key_values.get_seq_length(layer_idx)
    cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
else:
    cache_position_tensor = cache_position.flatten()[:batch_size].to(torch.int32)

cur_pos_tt = ttnn.from_torch(
    cache_position_tensor,
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**Option B:** If `cache_position` must sometimes be ttnn, keep the conversion but note this is a tiny scalar tensor so the overhead is negligible. Focus optimization effort on the much larger `_to_replicated` round-trip.

### Step 8: Update HEIGHT_SHARDED memory configs

The shard sizes for RoPE and SDPA depend on `num_heads`. Update all memory config calculations to use `num_local_heads` instead of `self.num_heads`.

### Step 9: Update `_apply_qk_norm` for local heads

If QK norm is applied per-head, the reshape dimensions need to use `num_local_heads`/`num_local_kv_heads` instead of the global counts.

## 4. Success Criteria

1. **No `ttnn.to_torch` or `ttnn.from_torch` calls in `_forward_decode_paged`** (except for the one-time `cache_position_tensor -> cur_pos_tt` upload which is a tiny scalar and uses `from_torch` with `ReplicateTensorToMesh` -- this is acceptable as it is host-to-device only, not a round-trip)
2. **`_to_replicated` method is not called** in the BailingMoEAttention decode path
3. **Both tests pass with correct text generation:**
   - `test_ling_mini_2_0.py`
   - `test_bailing_attention_accuracy.py`
4. **Decode latency improves or stays the same** (no regression from restructuring)

## 5. Risk Assessment

### High Risk: KV Head Sharding with 4 KV Heads on 8 Devices
- Ling 2.0 has 4 KV heads. With 8 devices, each device would get 0.5 KV heads, which is not possible.
- **Mitigation:** This is the same problem tt-transformers solves with its `mesh_config.shard_size()`. Investigate how tt-transformers handles sub-1 KV heads per device. It likely replicates KV heads across device pairs. Alternatively, run with tensor parallelism on 4 devices (TP=4) rather than 8 for the attention module.

### Medium Risk: Output Projection Restructuring
- Changing from `TTNNLinearIReplicatedWColSharded` (dense) to a row-parallel linear + all-reduce changes the communication pattern.
- **Mitigation:** This is a well-understood pattern from tt-transformers. The all-reduce replaces the current reduce_scatter in Q projection + all-gather pattern. Net communication should be similar or better.

### Medium Risk: Prefill Path Compatibility
- The prefill path (`_forward_prefill_paged`, lines 2524-2608) uses the same Q/K/V projections. Changing to column-parallel will affect prefill too.
- **Mitigation:** Verify prefill path also works with local heads. The prefill path uses `ttnn.experimental.nlp_create_qkv_heads` (not decode variant) which also accepts `num_heads`/`num_kv_heads`. Update both paths together.

### Low Risk: cache_position as torch
- Ensuring callers pass torch tensors is a minor API contract change.
- **Mitigation:** Add a defensive conversion at the top of `_forward_decode_paged` if needed.

### Alternative Approach (Lower Risk, Smaller Gain)
If the column-parallel restructuring is too risky, a **minimal approach** is:
1. Keep the current sharding strategy
2. Keep `_to_replicated` but optimize it: instead of going through host, try using `ttnn.experimental.all_gather_async` with a specific topology parameter, or create a device-side copy operation that sets replicated metadata
3. Fix `cache_position` to stay as torch (easy win)

This would only fix round-trip 2 fully and might partially improve round-trip 1, but would not eliminate the fundamental architecture issue.

## 6. Recommended Implementation Order

1. **Phase 1 (Quick Win):** Fix `cache_position` handling (Step 7) -- ensures callers pass torch, removes defensive ttnn->torch conversion. Low risk, easy to validate.
2. **Phase 2 (Main Work):** Restructure QKV to column-parallel (Steps 1-6, 8-9). This is the high-value change that eliminates the `_to_replicated` host round-trip.
3. **Phase 3 (Validation):** Run both test suites, profile decode latency, verify text generation correctness.
