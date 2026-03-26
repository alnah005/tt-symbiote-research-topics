# Optimization Plan: BailingAttention Decode Path

**Date:** 2026-03-26
**Target:** `TTNNBailingMoEAttention._forward_decode_paged` in
`models/experimental/tt_symbiote/modules/attention.py` (lines 2610-2799)

**Current decode latency:** ~9.5ms (no profiler), ~12.8ms (with profiler)
**Target decode latency:** 3-5ms
**Profile summary:** 72% data-movement-bound

---

## Current Data Flow (Decode Path)

```
hidden_states: [B, 1, hidden_size/8] column-sharded across 8 devices

Step 1: Q projection (IColShardedWRowSharded)
  matmul(hidden_shard, W_q_shard) -> partial_q
  reduce_scatter(partial_q) -> q_local  [B, 1, q_size/8]
  all_gather(q_local) -> q_full  [B, 1, q_size]          # ALL_GATHER #1

Step 2: K/V projection (IReplicatedWColSharded)
  all_gather(hidden_states) -> hidden_replicated          # ALL_GATHER #2
  matmul(hidden_replicated, W_k_shard) -> k_shard
  matmul(hidden_replicated, W_v_shard) -> v_shard
  all_gather(k_shard) -> k_full                           # ALL_GATHER #3
  all_gather(v_shard) -> v_full                           # ALL_GATHER #4

Step 3: Reshape Q/K/V from [B,1,dim] -> [1,B,heads,head_dim]  # 3x RESHAPE

Step 4: to_memory_config Q,K to L1 for QK norm                # 2x TO_MEM_CONFIG

Step 5: QK norm (reshape -> RMSNorm -> reshape)                # 4x RESHAPE

Step 6: to_memory_config Q,K,cos,sin for RoPE sharding        # 4x TO_MEM_CONFIG

Step 7: RoPE (rotary_embedding_llama x2)

Step 8: to_memory_config K,V for paged cache                   # 2x TO_MEM_CONFIG

Step 9: paged_update_cache + paged_sdpa_decode

Step 10: to_memory_config for nlp_concat_heads_decode          # 1x TO_MEM_CONFIG
Step 11: nlp_concat_heads_decode
Step 12: dense projection (IReplicatedWColSharded) -- NEEDS ALL_GATHER OF INPUT?
         Currently dense takes the concat'd output. Since output is replicated
         after concat_heads, the dense proj takes replicated input.
Step 13: reshape output
```

**Op count summary (current):**
- AllGather: 4 (lines 2626, 2631, 2632, 2633)
- ReduceScatter: 1 (inside q_proj forward, line 158 of linear.py)
- Reshape: ~7-11 (lines 2644-2646, plus QK norm reshapes 2467-2468 + 2487-2488, plus output 2797)
- to_memory_config: ~9 (lines 2656-2657, 2708-2709, 2718, 2727, 2753-2754, 2783)
- Matmul: 4 (Q, K, V, Dense projections)

---

## Optimization 1: Fused Column-Parallel QKV (Eliminate 3 AllGathers)

### Problem
The current design uses two different linear sharding strategies:
- Q: `IColShardedWRowSharded` -- input column-sharded, weight row-sharded, with reduce_scatter
- K/V: `IReplicatedWColSharded` -- input replicated, weight column-sharded

This forces:
1. An all_gather of hidden_states for K/V input (ALL_GATHER #2)
2. An all_gather of Q output after reduce_scatter (ALL_GATHER #1)
3. All_gathers of K and V outputs (ALL_GATHER #3, #4)

### Solution: Column-Parallel QKV (Like LLaMA-70B in tt-transformers)

Switch ALL projections (Q, K, V) to **column-parallel** strategy:
- Each device holds `W_q[:, hidden_size/8]`, `W_k[:, hidden_size/8]`, `W_v[:, hidden_size/8]`
- Input hidden_states is column-sharded (already is)
- Each device computes a **partial sum** for all of Q, K, V
- A single `reduce_scatter` on the fused QKV output replaces all 4 all_gathers + 1 reduce_scatter

But wait -- reduce_scatter gives each device 1/8 of the output. For Q that is fine (16 heads / 8 = 2 heads per device). For K/V with 4 KV heads, that is only 0.5 heads per device, which does not divide evenly.

### Revised Solution: Column-Parallel with AllReduce (not ReduceScatter)

Alternative approach that handles GQA (num_kv_heads < num_devices):

**Option A: Fused QKV matmul + all_reduce**
- Fuse Q, K, V weights into a single column-sharded weight matrix:
  `W_qkv = [W_q; W_k; W_v]` with shape `[(num_heads + 2*num_kv_heads) * head_dim, hidden_size]`
- Shard on the input dimension (dim=-2 of transposed weight): each device holds `W_qkv[:, hidden_size/8]`
- `matmul(hidden_shard, W_qkv_shard)` produces partial sums `[B, 1, (num_heads + 2*num_kv_heads)*head_dim]`
- Single `all_reduce` gives the full result replicated on all devices
- Split into Q, K, V by slicing on the last dimension

**Op savings:**
- Before: 4x all_gather + 1x reduce_scatter + 4x matmul = 9 CCL ops + 4 matmuls
- After: 1x all_reduce + 1x matmul = 2 ops (1 CCL + 1 matmul)

**Expected latency savings:** 2-3ms (eliminating 3 all_gathers at ~37us each on device, but host dispatch overhead is the real cost at ~0.5ms per all_gather call including dispatch)

**Option B: Column-parallel Q with row-parallel K/V (keep current K/V strategy but fuse)**

Since K/V are small (only 4 heads), another approach:
- Keep Q as IColShardedWRowSharded (produces local Q heads after reduce_scatter, no all_gather needed since each device only needs its own Q heads for SDPA)
- Fuse K and V into a single matmul with IReplicatedWColSharded
- This still needs 1 all_gather for hidden_states, but eliminates the separate K/V all_gathers

**Op savings (Option B):**
- Before: 4x all_gather + 1x reduce_scatter + 4x matmul
- After: 1x all_gather + 1x reduce_scatter + 2x matmul (fused_kv + Q + Dense)
- Net: eliminate 3 all_gathers, eliminate 2 matmuls

### Recommended: Option A (Fused QKV + AllReduce)

This is the tt-transformers proven pattern. The key insight: for decode with batch=1-8, the tensors are tiny (few KB), so the matmul compute is negligible compared to dispatch. Reducing from 4 matmuls + 5 CCL ops to 1 matmul + 1 CCL op saves ~4-5 host dispatch round-trips.

### Code Changes Required

#### `from_torch()` (lines 2316-2404)

```python
# CURRENT: Split fused QKV into separate Q, K, V and create 3 linear layers
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)
new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)

# NEW: Keep fused QKV weight, shard on input dimension
# Use TTNNLinearIColShardedWRowSharded for fused QKV
# After matmul, all_reduce (not reduce_scatter) to get full replicated result
qkv_weight_permuted = torch.cat([q_weight, k_weight, v_weight], dim=0)
qkv_linear = nn.Linear(hidden_size, q_size + 2*kv_size, bias=has_bias)
qkv_linear.weight.data = qkv_weight_permuted
if has_bias:
    qkv_linear.bias.data = torch.cat([q_bias, k_bias, v_bias])
new_attn.qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(qkv_linear)
```

This requires a **new linear class** `TTNNLinearIColShardedWAllReduced`:
- Input: column-sharded `[B, 1, hidden_size/8]`
- Weight: row-sharded `[(Q+K+V)_size, hidden_size/8]` per device
- Forward: `matmul` -> `all_reduce` (instead of reduce_scatter)
- Output: replicated `[B, 1, (Q+K+V)_size]`

Alternatively, modify the forward path to call `ttnn.experimental.all_reduce_async` after the matmul instead of `reduce_scatter_minimal_async`.

#### `_forward_decode_paged()` (lines 2610-2799)

```python
# CURRENT (lines 2624-2633):
query_states = self.q_proj(hidden_states)
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
key_states = self.k_proj(hidden_states_replicated)
value_states = self.v_proj(hidden_states_replicated)
ttnn.deallocate(hidden_states_replicated)
query_states = self._maybe_all_gather(query_states)
key_states = self._maybe_all_gather(key_states)
value_states = self._maybe_all_gather(value_states)

# NEW (replaces all of the above):
qkv = self.qkv_proj(hidden_states)  # all_reduce inside
# Split Q, K, V
q_size = self.num_heads * self.head_dim
kv_size = self.num_kv_heads * self.head_dim
query_states = ttnn.slice(qkv, [..., 0:q_size])
key_states = ttnn.slice(qkv, [..., q_size:q_size+kv_size])
value_states = ttnn.slice(qkv, [..., q_size+kv_size:])
```

### Risks and Correctness Concerns

1. **all_reduce availability**: Verify `ttnn.experimental.all_reduce_async` exists and supports the same semaphore/topology API as reduce_scatter. If not, use `all_gather` after reduce_scatter (effectively doing reduce_scatter then all_gather, which is equivalent to all_reduce but in 2 steps -- still saves 3 CCL ops).

2. **Weight layout correctness**: The fused QKV weight must be sharded on dim=-2 (the input dimension after transpose). Each device's shard computes partial sums that must be summed (not concatenated). Verify this matches `TTNNLinearIColShardedWRowSharded`'s existing sharding via `shard_tensor_to_mesh_mapper(device, dim=-2)`.

3. **Bias handling**: If QKV has bias, bias should NOT be sharded -- it should be added after the all_reduce. Currently `TTNNLinearIColShardedWRowSharded` adds bias after reduce_scatter. With all_reduce the bias addition remains the same.

4. **Incremental approach**: Can be done in 2 steps:
   - Step 1: Fuse K+V into single matmul (keep Q separate) -- removes 1 matmul, 1 all_gather
   - Step 2: Fuse Q+K+V and switch to all_reduce -- removes remaining all_gathers

---

## Optimization 2: ReshapeView Reduction (Target: 328 ops -> ~100 ops)

### Problem
328 ReshapeView ops account for 20.5% of device kernel time. These come from:

1. **Linear layer reshapes** (2 per linear call x 4 linears = 8): Lines 155-156 and 175 in `TTNNLinearIColShardedWRowSharded.forward()` and `TTNNLinearIReplicatedWColSharded.forward()` -- pad input to 4D, then reshape output back.

2. **QKV head reshapes** (3): Lines 2644-2646 -- reshape flat Q/K/V to `[1, B, heads, head_dim]`.

3. **QK norm reshapes** (4): Lines 2467-2468, 2487-2488 -- flatten for norm, unflatten after.

4. **Output reshapes** (2): Lines 2797 and inside dense linear.

5. **Fused QKV would replace items 1-3** with just slice ops (zero-copy if contiguous).

### Solutions

#### 2a: Eliminate Linear Layer 4D Padding Reshapes

The `TTNNLinearIColShardedWRowSharded.forward()` and `TTNNLinearIReplicatedWColSharded.forward()` both do:
```python
while len(input_shape) < 4:
    input_shape.insert(1, 1)
input_tensor = ttnn.reshape(input_tensor, input_shape)  # RESHAPE #1
tt_output = ttnn.linear(...)
tt_output = ttnn.reshape(tt_output, ...)                 # RESHAPE #2
```

If we ensure inputs are already 4D (e.g., `[1, 1, B, hidden_size/8]` for decode), these reshapes become unnecessary.

**Code change:** In `_forward_decode_paged`, pre-reshape hidden_states to 4D once before calling any linear:
```python
hidden_states_4d = ttnn.reshape(hidden_states, [1, 1, batch_size, hidden_size_local])
```
Then modify the linear classes to skip reshape when input is already 4D. This saves 8 reshapes per decode (2 per linear x 4 linears).

With fused QKV (Optimization 1), this drops to 4 reshapes eliminated (2 for qkv_proj, 2 for dense).

**Expected impact:** -8 ReshapeView ops per iteration. At 33.8us avg, saves ~270us.

#### 2b: Eliminate QKV Head Reshapes

Currently (lines 2644-2646):
```python
query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))
```

With fused QKV + slice, the slice output is already contiguous and can potentially be reshaped with zero-cost view. But on TTNN, `reshape` may still trigger a kernel if padding or layout changes.

Alternative: Use `nlp_create_qkv_heads_decode` on the fused QKV output (like LLaMA-70B does):
```python
query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv_output,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
```
This single op replaces 3 reshapes AND produces HEIGHT_SHARDED output in L1, which is already what RoPE needs. This eliminates the subsequent `to_memory_config` calls for RoPE sharding.

**Expected impact:** -3 ReshapeView + -2 to_memory_config (Q and K sharding for RoPE). Saves ~300-400us.

#### 2c: Eliminate QK Norm Reshapes

The `_apply_qk_norm` method (lines 2454-2493) does:
```python
# Decode mode
q_reshaped = ttnn.reshape(query_states, (batch_size * num_heads, head_dim))      # RESHAPE
k_reshaped = ttnn.reshape(key_states, (batch_kv * num_kv_heads, head_dim_k))     # RESHAPE
q_normed = self.query_layernorm(q_reshaped)
k_normed = self.key_layernorm(k_reshaped)
query_states = ttnn.reshape(q_normed, (seq_len, batch_size, num_heads, head_dim)) # RESHAPE
key_states = ttnn.reshape(k_normed, (seq_len, batch_kv, num_kv_heads, head_dim)) # RESHAPE
```

If `ttnn.rms_norm` can operate on the last dimension of a 4D tensor (normalizing each `[head_dim]` vector independently), these 4 reshapes can be eliminated. Check if `ttnn.rms_norm` supports 4D input with normalization on dim=-1.

Alternatively, if the norm weights are 1D `[head_dim]`, RMSNorm should broadcast correctly across `[1, B, H, head_dim]` without flattening.

**Expected impact:** -4 ReshapeView ops. Saves ~135us.

### Total ReshapeView Reduction

| Source | Current | After Opt | Saved |
|--------|---------|-----------|-------|
| Linear 4D padding | 8 | 0-2 | 6-8 |
| QKV head reshape | 3 | 0 (use nlp_create_qkv_heads) | 3 |
| QK norm flatten/unflatten | 4 | 0 | 4 |
| Output reshape | 2 | 1-2 | 0-1 |
| **Total per iteration** | **~17** | **~2-4** | **~13-15** |

At 33.8us avg per reshape, saving 13 reshapes = **~440us per iteration**.

---

## Optimization 3: ReduceScatter Optimization

### Problem
1 reduce_scatter per iteration (inside Q projection) accounts for 14.1% of device kernel time (72.8us avg device time, but much higher with host dispatch overhead).

### Analysis

With Optimization 1 (Fused QKV + AllReduce), the reduce_scatter is **replaced** by all_reduce. The all_reduce is a reduce_scatter + all_gather fused into one CCL operation, which is:
- 1 CCL op instead of reduce_scatter(1) + all_gather(4) = 5 CCL ops
- The all_reduce device kernel time is roughly reduce_scatter + all_gather time combined, but with only 1 host dispatch instead of 5

If all_reduce is not available as a single fused op, the fallback is:
```python
# reduce_scatter on fused QKV, then all_gather
qkv_partial = ttnn.linear(hidden_shard, W_qkv_shard)
qkv_reduced = reduce_scatter(qkv_partial, dim=3)  # each device gets 1/8 of QKV
qkv_full = all_gather(qkv_reduced, dim=3)          # replicate to all devices
```
This is 2 CCL ops instead of 5, saving 3 host dispatch round-trips (~1.5ms).

### Alternative: Keep ReduceScatter for Q, Eliminate for K/V

If fused QKV is not feasible (e.g., GQA head count issues), an intermediate optimization:
- Q: keep IColShardedWRowSharded (reduce_scatter produces local Q heads)
- Skip the all_gather of Q entirely -- each device only needs its local Q heads for paged_sdpa_decode
- K/V: use a different strategy (see below)

**Key insight:** For paged SDPA decode, each device only needs:
- Its local Q heads (num_heads / num_devices = 2 heads per device)
- ALL K/V heads (since KV is grouped/replicated)

So Q does NOT need all_gather if we keep it per-device after reduce_scatter. Only K/V need to be fully replicated.

This means:
- Q: IColShardedWRowSharded -> reduce_scatter -> done (no all_gather needed)
- K/V: need replicated output

For K/V, instead of all_gather(hidden) + matmul, we can use:
- `matmul(hidden_shard, W_k_row_shard)` -> partial K -> `all_reduce` -> full K
- Same for V

This pattern (IColShardedWRowSharded + all_reduce instead of reduce_scatter) gives replicated output.

**Net CCL ops:**
- Before: 1 reduce_scatter + 4 all_gather = 5
- After: 1 reduce_scatter (Q) + 1 all_reduce (fused KV) = 2
- Or with separate K/V all_reduces: 1 reduce_scatter + 2 all_reduce = 3

---

## Optimization 4: Eliminate `_to_replicated` Host Round-Trip

### Problem
The `_to_replicated` method (lines 2288-2313) does a device -> host -> device round-trip to change mesh topology metadata from "all_gathered" to "replicated". This is called 3x for Q, K, V in the Glm4MoeLite attention class (lines 1895-1897).

Note: In `TTNNBailingMoEAttention._forward_decode_paged`, `_to_replicated` is NOT called. The code at lines 2644-2646 directly reshapes the all_gathered output. However, the all_gather output's mesh topology may still cause issues with paged kernels.

### Analysis of Current BailingMoE Path

Looking at the current code:
- Line 2631-2633: `_maybe_all_gather` uses `ttnn.all_gather(t, dim=-1, num_links=1)` (NOT the async version)
- The output of `ttnn.all_gather` should already be replicated (each device has the full tensor)
- No explicit `_to_replicated` call in `_forward_decode_paged`

If paged attention kernels are working correctly without `_to_replicated`, this is already eliminated for BailingMoE. If not, the fix is to use `ttnn.distributed.change_mesh_topology()` (if it exists) or ensure the all_reduce approach from Optimization 1 produces replicated topology natively.

**Expected impact:** 0ms if already not called, or ~1-2ms if still needed.

---

## Optimization 5: to_memory_config Reduction

### Problem
~9 `to_memory_config` calls per decode iteration for resharding between DRAM, L1, and various sharding configs.

### Current to_memory_config calls:
1. Lines 2656-2657: Q, K to L1 for QK norm
2. Lines 2708-2709: cos, sin to HEIGHT_SHARDED for RoPE
3. Line 2718: Q to HEIGHT_SHARDED for RoPE
4. Line 2727: K to HEIGHT_SHARDED for RoPE
5. Lines 2753-2754: K, V to HEIGHT_SHARDED for paged cache update
6. Line 2783: attn_output to HEIGHT_SHARDED for nlp_concat_heads

### Solution: Produce Tensors in Target Memory Config

With `nlp_create_qkv_heads_decode` (Optimization 2b), Q and K are already produced in L1_HEIGHT_SHARDED. This eliminates calls 1, 3, 4 (and possibly 5 if the shard config matches).

The cos/sin sharding (calls 2) can be pre-computed and cached on device in the correct sharded format during `move_weights_to_device_impl`, since the batch_size is known at setup time.

**Expected savings:** 4-5 to_memory_config ops eliminated, saving ~200-300us.

---

## Summary: Optimization Priority and Impact

| # | Optimization | Ops Eliminated | Est. Savings | Effort | Risk |
|---|-------------|---------------|-------------|--------|------|
| 1 | Fused QKV + AllReduce | 3 all_gather, 3 matmul, 1 reduce_scatter -> 1 all_reduce + 1 matmul | 2-3ms | High | Medium (new linear class) |
| 2b | nlp_create_qkv_heads_decode | 3 reshape, 2 to_memory_config | 0.4-0.5ms | Low | Low |
| 2a | Eliminate linear 4D reshapes | 6-8 reshape | 0.2-0.3ms | Low | Low |
| 2c | Eliminate QK norm reshapes | 4 reshape | 0.1-0.2ms | Low | Low (verify RMSNorm 4D) |
| 5 | Cache cos/sin sharded | 2 to_memory_config | 0.1ms | Low | None |
| 4 | Eliminate _to_replicated | 0 (already not called) | 0ms | None | None |
| 3 | ReduceScatter -> AllReduce | Subsumed by Opt 1 | (included in Opt 1) | -- | -- |

**Total estimated savings: 3-4ms** (from ~9.5ms to ~5.5-6.5ms)

### Combined with Trace Capture

Adding trace capture (from prior research) would save an additional 2-3ms of host dispatch overhead, bringing the total to **~3-4ms per decode** -- hitting the target range.

---

## Implementation Order (Incremental)

### Phase 1: Low-Hanging Fruit (no architectural changes)
1. Use `nlp_create_qkv_heads_decode` after concatenating Q+K+V (Opt 2b)
2. Eliminate linear 4D padding reshapes by passing 4D input (Opt 2a)
3. Pre-cache cos/sin in sharded format (Opt 5)
4. Test QK norm with 4D input to eliminate flatten/unflatten (Opt 2c)

**Expected savings: 0.7-1.0ms**

### Phase 2: Fused QKV (architectural change)
1. Create `TTNNLinearIColShardedWAllReduced` linear class (or modify existing)
2. Update `from_torch()` to produce fused QKV weight
3. Update `_forward_decode_paged()` to use fused QKV + split
4. Update `_forward_prefill()` accordingly

**Expected savings: 2-3ms additional**

### Phase 3: Trace Capture
1. Enable TTNN trace capture for the decode path
2. Pre-allocate persistent buffers for intermediate tensors

**Expected savings: 2-3ms additional**

---

## Appendix: LLaMA-70B Reference Pattern

From `models/demos/t3000/llama2_70b/tt/llama_attention_optimized.py`:

```python
# Weight setup: Fused QKV, column-sharded across devices
# Each device holds its local Q heads + local KV heads
qkv_interleaved = [W_q_local, W_k_local, W_v_local]  # interleaved per KV group
qkv = torch.cat(qkv_interleaved, dim=-1)
# Sharded with ShardTensorToMesh(device, dim=3) -- each device holds its shard

# Decode forward:
fused_qkv = ttnn.matmul(xs, self.qkv, ...)  # Single matmul
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(fused_qkv, ...)  # Single split op
# RoPE applied directly (already in correct sharded format)
# No all_gather needed for Q or K/V
```

Key difference: LLaMA uses tensor-parallel where each device holds its own heads. For BailingMoE with only 4 KV heads across 8 devices, this requires either:
- Head replication (some devices share KV heads)
- All_reduce instead of reduce_scatter (replicate KV to all devices)
- GQA-aware head distribution

The all_reduce approach is the simplest path forward for BailingMoE's head configuration.
