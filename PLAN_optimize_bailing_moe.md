# Optimization Plan: BailingMoE (TTNNBailingMoE) Decode Path

**Date:** 2026-03-26
**Target:** `TTNNMoE.forward()` in `models/experimental/tt_symbiote/modules/moe.py`
(TTNNBailingMoE inherits from TTNNMoE, no forward override)

**Model:** Ling-mini-2.0
**Architecture parameters:**
- 64 routed experts (`n_routed_experts=64`)
- top-4 routing (`num_experts_per_tok=4`)
- Sigmoid activation (not softmax)
- `n_group=1`, `topk_group=1` (no group-based routing)
- `hidden_size=2048`, `moe_intermediate_size=1536`
- `routed_scaling_factor=1.8`, `norm_topk_prob=True`
- T3K mesh: 8 devices, 8 experts per device

---

## Current Data Flow (TTNNMoE.forward, decode path)

```
Input x: [B, 1, seq_len, hidden_size/8] column-sharded across 8 devices

Step 1: all_gather_async(x, dim=-1)
  -> x_full: [B, 1, seq_len, hidden_size] replicated

Step 2: Gate routing
  2a. typecast x_full to float32
  2b. ttnn.linear(x_f32, gate_weight_bf16) -> router_logits_f32   [HiFi4, fp32 acc]
  2c. typecast router_logits_f32 to bf16
  2d. reshape to (T, n_routed_experts)

Step 3: TTNNMoERouterDecode.forward(router_logits)
  3a. to_layout(TILE_LAYOUT) if needed
  3b. reshape to (1,1,T,64)
  3c. typecast to float32
  3d. sigmoid(logits_f32) -> scores_f32
  3e. repeat bias -> to_layout(TILE) -> typecast to f32
  3f. add(scores_f32, bias_f32)
  --- 3-PASS TOPK (n_group=1 <= topk_group=1 branch) ---
  3g. typecast scores_with_bias_f32 to bf16
  3h. topk(scores_bf16, k=5) -> rough_vals          [PASS 1]
  3i. slice(rough_vals, [top_k]) -> rough_thr_bf16
  3j. typecast rough_thr to f32
  3k. sub(scores_with_bias_f32, rough_thr_f32) -> scores_c1
  3l. typecast scores_c1 to bf16
  3m. topk(scores_bf16, k=5) -> refined_vals         [PASS 2]
  3n. slice(refined_vals, [top_k]) -> refined_thr_bf16
  3o. typecast refined_thr to f32
  3p. sub(scores_c1, refined_thr_f32) -> scores_c2
  3q. typecast scores_c2 to bf16
  3r. topk(scores_bf16, k=4) -> topk_expert_idx      [PASS 3]
  --- END 3-PASS ---
  3s. gather(scores_f32, topk_expert_idx) -> topk_weights
  3t. sum(topk_weights, dim=3) -> denom
  3u. div(topk_weights, denom)
  3v. repeat scale -> to_layout(TILE) -> typecast to f32
  3w. mul(topk_weights, scale_f32)
  3x. reshape topk_expert_idx to (T, 4)
  3y. reshape topk_weights to (T, 4)
  --- NO CPU SORT (TTNNMoERouterDecode does NOT have the sort) ---

Step 4: TTNNExperts.forward(x, topk_idx, topk_weights)
  4a. typecast topk_idx to uint16 (via to_layout(TILE) + typecast)
  4b. pad to SPARSITY_BLOCK_SIZE=32 if needed (x, indices, weights)
  4c. typecast x to bf16
  4d. to_layout(x, ROW_MAJOR)
  4e. reshape x_rm
  4f. to_layout(topk_idx, ROW_MAJOR)
  4g. reshape topk_idx_rm
  4h. all_to_all_dispatch(x_rm, topk_idx_rm, expert_mapping)
  4i. reshape post_dispatch
  4j. to_layout(post_dispatch, TILE_LAYOUT)
  4k. repeat remap_topk_mask
  4l. moe_expert_token_remap -> sparsity_t
  4m. reshape x_sparse
  4n. sparse_matmul(x_sparse, w1, sparsity_t) -> w1_out     [GATE]
  4o. sparse_matmul(x_sparse, w3, sparsity_t) -> w3_out     [UP]
  4p. silu(w1_out)
  4q. mul(w1_activated, w3_out) -> intermediate
  4r. squeeze(intermediate, 0), squeeze(intermediate, 1)
  4s. sparse_matmul(intermediate, w2, sparsity_t) -> expert_output  [DOWN]
  4t. permute(expert_output, (1,0,2,3))
  4u. reshape expert_output
  4v. to_layout(expert_output, ROW_MAJOR)
  4w. reshape expert_output
  4x. all_to_all_combine(expert_output, metadata, expert_mapping)
  4y. reshape combined_output
  4z. to_layout(combined_output, TILE_LAYOUT)

  --- WEIGHT APPLICATION ---
  4aa. to_layout(topk_weights, ROW_MAJOR)
  4ab. unsqueeze(topk_weights, 0)
  4ac. unsqueeze(topk_weights, 0)
  4ad. repeat(topk_weights, (hidden_size, 1, 1, 1))    # <-- EXPENSIVE
  4ae. permute(topk_weights, (3, 1, 2, 0))
  4af. to_layout(topk_weights, TILE_LAYOUT)
  4ag. mul(combined_output, topk_weights)
  4ah. sum(weighted_output, dim=0)

  4ai. slice (remove padding if added)

Step 5: reduce_scatter_minimal_async(routed_output, dim=3)
  (with 1/n_rs scaling)

Step 6: shared_experts(residual) -> shared_output
  6a. gate_proj(residual) [TTNNLinearSilu]
  6b. up_proj(residual) [TTNNLinearIColShardedWRowSharded]
  6c. mul(gate, up)
  6d. down_proj(intermediate) [TTNNLinearIColShardedWRowSharded]

Step 7: add(routed_output, shared_output)
Step 8: squeeze(output, 1)
```

---

## Bottleneck Analysis

### Bottleneck 1: Router Overhead (Steps 2-3)

The router performs **9 typecast operations** and **3 topk passes** even though Ling-mini-2.0 has `n_group=1, topk_group=1`, meaning no group-based routing is needed. The 3-pass topk centering was designed to work around bf16 precision limitations in topk, but with only 64 experts (vs 256), the dynamic range is much smaller.

**Op count in router (current):**
- typecast: 9 (bf16->f32, f32->bf16 round-trips)
- topk: 3 (k+1, k+1, k)
- slice: 2
- sub: 2
- sigmoid: 1
- add: 1
- gather: 1
- sum: 1
- div: 1
- mul: 1
- reshape: 4+
- to_layout: 3+
- repeat: 3+

**Total: ~30+ ops** for routing 1 token to 4 experts out of 64.

### Bottleneck 2: Weight Application Pattern (Steps 4aa-4ah)

The current weight application does:
```python
topk_weights = ttnn.to_layout(topk_weights, ROW_MAJOR)           # layout convert
topk_weights = ttnn.unsqueeze(topk_weights, 0)                    # reshape
topk_weights = ttnn.unsqueeze(topk_weights, 0)                    # reshape
topk_weights = ttnn.repeat(topk_weights, (hidden_size, 1, 1, 1))  # MASSIVE repeat
topk_weights = ttnn.permute(topk_weights, (3, 1, 2, 0))           # permute
topk_weights = ttnn.to_layout(topk_weights, TILE_LAYOUT)          # layout convert
weighted = ttnn.mul(combined_output, topk_weights)                 # broadcast mul
final = ttnn.sum(weighted, dim=0)                                  # reduce
```

The `repeat(topk_weights, (hidden_size, 1, 1, 1))` with `hidden_size=2048` creates a tensor that is 2048x larger than necessary. For decode with 1 token: shape goes from `(1, 1, 1, 4)` to `(2048, 1, 1, 4)` then permuted to `(4, 1, 1, 2048)`. This is a bandwidth-bound operation that copies 2048 * 4 * 2 bytes = 16KB of bf16 data -- but the repeat itself allocates a new tensor and copies weight values 2048 times.

The multiply + sum pattern `sum(combined * weights, dim=0)` is effectively a weighted reduction that could be done more efficiently.

### Bottleneck 3: Layout Conversions (ROW_MAJOR <-> TILE_LAYOUT)

The expert forward path has **at least 6 explicit layout conversions:**
1. `to_layout(x, ROW_MAJOR)` -- before all_to_all_dispatch (Step 4d)
2. `to_layout(post_dispatch, TILE_LAYOUT)` -- before sparse_matmul (Step 4j)
3. `to_layout(expert_output, ROW_MAJOR)` -- before all_to_all_combine (Step 4v)
4. `to_layout(combined_output, TILE_LAYOUT)` -- after combine (Step 4z)
5. `to_layout(topk_weights, ROW_MAJOR)` -- weight application (Step 4aa)
6. `to_layout(topk_weights, TILE_LAYOUT)` -- weight application (Step 4af)

Conversions 1-4 are required by all_to_all_dispatch/combine (ROW_MAJOR) and sparse_matmul (TILE_LAYOUT). These are structural constraints.
Conversions 5-6 for weight application could be avoided with a different approach.

### Bottleneck 4: Sequential Shared Expert Execution (Steps 5-6)

Currently:
```
routed_output = experts(x, ...)          # Step 4
routed_output = reduce_scatter(routed_output)  # Step 5
shared_output = shared_experts(residual) # Step 6  <-- SEQUENTIAL
output = add(routed_output, shared_output)
```

The shared expert computation (gate_proj, up_proj, mul, down_proj -- 4 matmuls) is independent of the routed expert output and could start as soon as the residual is available (after Step 1's all_gather).

### Bottleneck 5: Gate Projection Precision Chain (Step 2)

The gate projection does:
```python
x_f32 = typecast(x, float32)                     # bf16 -> f32
router_logits_f32 = linear(x_f32, gate_weight)   # f32 matmul with HiFi4
router_logits_bf16 = typecast(router_logits_f32, bf16)  # f32 -> bf16
```

Then the router immediately does:
```python
logits_f32 = typecast(logits_bf16, float32)  # bf16 -> f32 again!
scores_f32 = sigmoid(logits_f32)
```

This round-trips through bf16 unnecessarily. The gate could output f32 directly to the router.

---

## Phase 1: Router Simplification (High Impact, Medium Effort)

### Optimization 1.1: Eliminate bf16 Round-Trip in Gate -> Router Handoff

**Problem:** Gate outputs f32 logits, typecasts to bf16, router typecasts back to f32.

**Solution:** Pass f32 logits directly from gate to router. Remove the intermediate typecast.

**Code changes:**
- In `TTNNMoE.forward()` (line ~1458-1460): Remove `typecast(router_logits_f32, bf16)`. Pass `router_logits_f32` directly to `route_tokens_to_experts()`.
- In `TTNNMoERouterDecode.forward()` (lines 897-901): The input is already f32, skip the `typecast(logits, float32)` step.

**Saves:** 2 typecast ops per iteration.

### Optimization 1.2: Simplify 3-Pass Topk to 1-Pass for n_group <= topk_group

**Problem:** When `n_group <= topk_group` (true for Ling-mini-2.0 where both equal 1), all groups are selected, so no group masking is needed. The 3-pass topk centering exists to handle bf16 precision issues, but for 64 experts with sigmoid scores in [0,1], the precision is adequate for a single-pass topk.

**Analysis:** Sigmoid outputs are in [0,1]. With 64 experts and top-4 selection, the decision boundary is typically around the 4th-5th ranked score. Bf16 has ~3 decimal digits of precision near 0.5, meaning scores differing by < 0.001 could be misranked. However:
- The correction bias (e_score_correction_bias) spreads scores apart deliberately
- The scoring function uses sigmoid (not softmax), so scores are independent per expert
- For 64 experts, the top-4 margin is typically >0.01 after training with load-balancing loss

**Solution:** Replace the 3-pass centering with a single f32 topk:
```python
# Instead of 3-pass centering:
# Just do topk on scores_with_bias_f32 directly
topk_values_f32, topk_expert_idx = ttnn.topk(scores_with_bias_f32, k=top_k, dim=3)
```

**Caveat:** ttnn.topk may require bf16 input. If so, the single-pass approach is:
```python
scores_bf16 = typecast(scores_with_bias_f32, bf16)
_, topk_expert_idx = ttnn.topk(scores_bf16, k=top_k, dim=3)
```

This is 1 typecast + 1 topk instead of 6 typecasts + 3 topks + 2 slices + 2 subs.

**Risk:** If the bf16 precision limitation genuinely causes incorrect topk for some edge-case tokens, accuracy could degrade. Mitigation: Run the test suite (`test_moe.py`) comparing 1-pass vs 3-pass topk outputs across 1000+ random inputs to measure agreement rate.

**Saves:** ~20 ops per router call (6 typecasts, 2 topks, 2 slices, 2 subs, several deallocates).

### Optimization 1.3: Reduce Remaining Typecasts in Router

After Optimizations 1.1 and 1.2, the remaining typecast operations in the router are:
1. `scores_f32 = sigmoid(logits_f32)` -- already f32, no cast needed
2. `bias: bf16 -> f32` for addition with scores
3. `scale: bf16 -> f32` for multiplication with weights

**Solution:** Pre-compute bias and scale in f32 during `preprocess_weights_impl()` and `move_weights_to_device_impl()` instead of storing them in bf16 and casting at runtime.

**Code changes:**
- In `TTNNMoERouterDecode.preprocess_weights_impl()`: Store `_bias_torch` and `_scale_torch` as float32 tensors
- In `TTNNMoERouterDecode.move_weights_to_device_impl()`: Use `dtype=ttnn.float32` when creating device tensors

**Saves:** 2 typecast ops per iteration (bias bf16->f32, scale bf16->f32).

### Optimization 1.4: Eliminate Unnecessary repeat + to_layout for Bias and Scale

**Problem:** The bias and scale tensors are stored as shape `(1,1,1,64)` and `(1,1,1,4)` respectively, then `_safe_repeat`-ed along the T dimension every forward call. For decode with T=1, the repeat is `(1,1,1,1)` -- a no-op that triggers a `ttnn.clone()`.

**Solution:** For decode mode (T=1), skip the repeat entirely since the bias shape `(1,1,1,64)` already broadcasts with `(1,1,1,64)` scores, and scale `(1,1,1,4)` broadcasts with `(1,1,1,4)` weights.

**Code changes:** Add a T=1 fast path:
```python
if T == 1:
    bias = self._bias_dev  # Already (1,1,1,64), broadcasts with (1,1,1,64)
else:
    bias = _safe_repeat(self._bias_dev, ttnn.Shape((1,1,T,1)))
```

Similarly for scale.

**Saves:** 2 repeat/clone ops + 2 to_layout calls per iteration for decode.

### Phase 1 Summary

| Optimization | Ops Removed | Description |
|-------------|------------|-------------|
| 1.1 Gate->Router handoff | 2 typecast | Pass f32 directly |
| 1.2 1-pass topk | ~20 ops | Eliminate 2 extra topk passes + centering |
| 1.3 Pre-store f32 constants | 2 typecast | Bias/scale already f32 on device |
| 1.4 Skip repeat for T=1 | 2 repeat + 2 to_layout | Broadcasting suffices |
| **Total** | **~28 ops** | Router: ~30 ops -> ~5-7 ops |

**Estimated savings:** 2-4ms per MoE layer (host dispatch of ~28 eliminated ops at ~100-200us each for small tensors).

---

## Phase 2: Expert Pipeline Optimization (Medium Impact, Medium Effort)

### Optimization 2.1: Optimize Weight Application Pattern

**Problem:** The current pattern for applying routing weights is:
```python
weights -> ROW_MAJOR -> unsqueeze -> unsqueeze -> repeat(hidden_size,1,1,1) -> permute -> TILE_LAYOUT -> mul -> sum
```

This creates a temporary tensor of shape `(hidden_size, 1, T, top_k)` = `(2048, 1, 1, 4)` just to do a weighted sum over the expert dimension.

**Solution A: Use ttnn.matmul for weighted sum**

Reshape combined_output from `(top_k, 1, T, hidden_size)` to `(1, T, top_k, hidden_size)`, treat the expert dimension as a dot-product dimension:

```python
# combined_output: (4, 1, T, 2048) -> (1, 1, T*4, 2048) via reshape is wrong
# Better: reshape to (1, T, 4, 2048), weights as (1, T, 4, 1)
# Then: output = sum(combined_output * weights, dim=2) == (1, T, 2048)
```

Actually, the most efficient approach: loop over the 4 experts and accumulate:
```python
# topk_weights shape: (T, 4)
# combined_output shape: (4, 1, T, hidden_size)
result = ttnn.zeros((1, 1, T, hidden_size))
for k in range(top_k):
    w_k = ttnn.slice(topk_weights, [0, k], [T, k+1])  # (T, 1)
    w_k = ttnn.reshape(w_k, (1, 1, T, 1))  # broadcasts over hidden_size
    e_k = ttnn.slice(combined_output, [k, 0, 0, 0], [k+1, 1, T, hidden_size])
    result = ttnn.add(result, ttnn.mul(e_k, w_k))
```

This uses 4 slices + 4 reshapes + 4 muls + 4 adds = 16 ops, but avoids the massive repeat(2048) and the permute. Each mul is a simple broadcast (1,1,T,1) * (1,1,T,2048).

**Solution B: Use in-place weighted accumulation on device**

If `ttnn.addcmul` or similar fused multiply-accumulate exists:
```python
result = ttnn.zeros(...)
for k in range(top_k):
    result = ttnn.addcmul(result, combined_output[k], weights[k])
```

**Solution C: Restructure as batch matmul**

Reshape routing weights to `(1, 1, 1, 4)` and combined output to `(1, 1, 4, hidden_size)`, then:
```python
# weights: (1, 1, 1, 4) @ combined: (1, 1, 4, 2048) -> (1, 1, 1, 2048)
result = ttnn.matmul(weights_reshaped, combined_reshaped)
```

This is a single matmul op that does the weighted sum. For T=1 decode, this is a `(1,1,1,4) x (1,1,4,2048)` matmul -- very efficient.

**Recommended: Solution C (batch matmul)** for decode. For T>1, extend to `(1,T,1,4) x (1,T,4,2048)`.

**Code changes in TTNNExperts.forward():**
```python
# Replace lines 1321-1335 with:
topk_weights_reshaped = ttnn.reshape(topk_experts_weights, (1, 1, T, self.num_experts_per_tok))
topk_weights_reshaped = ttnn.permute(topk_weights_reshaped, (0, 2, 1, 3))  # (1, T, 1, 4)
combined_permuted = ttnn.permute(combined_output, (1, 2, 0, 3))  # (1, T, 4, hidden_size)
final_output = ttnn.matmul(topk_weights_reshaped, combined_permuted)  # (1, T, 1, hidden_size)
final_output = ttnn.reshape(final_output, (1, 1, T, hidden_size))
```

**Saves:** Eliminates repeat(2048), 2 to_layout conversions, 2 unsqueezes, 1 permute, 1 broadcast mul, 1 sum. Replaces with 2 reshapes + 1 permute + 1 matmul.
Net reduction: ~4 ops eliminated, major bandwidth savings from removing repeat(2048).

### Optimization 2.2: Reduce Layout Conversions in Expert Pipeline

**Problem:** 6 layout conversions between ROW_MAJOR and TILE_LAYOUT.

**Analysis:**
- Conversions 1-2 (x to ROW_MAJOR for dispatch, back to TILE for sparse_matmul): Required by API constraints. all_to_all_dispatch requires ROW_MAJOR input; sparse_matmul requires TILE.
- Conversions 3-4 (expert output to ROW_MAJOR for combine, back to TILE after): Same structural constraint.
- Conversions 5-6 (weight application): Eliminated by Optimization 2.1.

**Possible improvement for conversions 1-4:** Check if all_to_all_dispatch/combine can accept TILE_LAYOUT input. If a newer TTNN version supports this, 4 conversions can be eliminated. If not, these are structural and must remain.

**Fallback optimization:** Ensure layout conversions use optimal memory configs. Currently they default to DRAM. For small decode tensors (T=1), L1 would be faster:
```python
x_rm = ttnn.to_layout(x, ROW_MAJOR, memory_config=ttnn.L1_MEMORY_CONFIG)
```

**Saves:** Potentially 4 to_layout ops if API supports TILE all_to_all, otherwise minor latency improvement from L1 memory config.

### Optimization 2.3: Eliminate Redundant Padding for Decode

**Problem:** For decode with T=1, the code pads to SPARSITY_BLOCK_SIZE=32:
```python
if num_tokens % SPARSITY_BLOCK_SIZE != 0:
    pad_amount = SPARSITY_BLOCK_SIZE - (num_tokens % 32)  # = 31 for T=1
```

This pads 31 zero tokens, making the effective batch 32x larger for sparse_matmul. The sparse_matmul then processes a 32-token block where 31 tokens are zeros.

**Analysis:** This padding is required by the sparse_matmul tile size constraint. The padding itself is unavoidable, but the slice at the end to remove padding could be eliminated if downstream ops can handle the padded tensor and the padding naturally disappears through reduce_scatter.

**Investigation needed:** Whether the padding survives through all_to_all_combine and affects the reduce_scatter result. If padding zeros don't contribute to the reduction, the final slice can be deferred or eliminated.

### Phase 2 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 2.1 Weight application via matmul | Replace repeat+permute+mul+sum with single matmul | ~1-2ms (bandwidth savings) |
| 2.2 L1 memory config for layout conversions | Faster to_layout for small tensors | ~0.2-0.5ms |
| 2.3 Eliminate redundant slice | Defer/remove final padding slice | ~0.1ms |
| **Total** | | **~1-2.5ms** |

---

## Phase 3: Shared Expert Overlap (Medium Impact, High Effort)

### Optimization 3.1: Overlap Shared Expert with Routed Expert Computation

**Problem:** Shared experts run after reduce_scatter completes, adding their full latency serially.

**Current:**
```
all_gather -> gate -> router -> experts(dispatch+compute+combine) -> reduce_scatter -> shared_experts -> add
                                                                       |               |
                                                                    sequential      sequential
```

**Proposed:**
```
all_gather -> gate -> router -> experts(dispatch+compute+combine) -> reduce_scatter ---> add
         \                                                                               /
          \--> shared_experts(residual) ----------------------------------------->------/
```

The shared expert computation depends only on `residual` (= the original input `x` before all_gather), which is available at the very beginning. The shared expert output is only needed at the final add.

**Implementation:**

Using TTNN async command queues:
```python
# In TTNNMoE.forward():
residual = x  # column-sharded input

# Start shared experts on a separate command queue (if supported)
# Or simply reorder: compute shared experts BEFORE routed experts
shared_output = self.shared_experts(residual)  # Can start immediately

# Then do routed experts
x = ttnn.experimental.all_gather_async(x, dim=-1, ...)
# ... gate, router, experts ...
routed_output = ttnn.experimental.reduce_scatter_minimal_async(routed_out, ...)

# Overlap happens naturally if using async ops
output = ttnn.add(routed_output, shared_output.to_ttnn)
```

**Simpler approach: Compute shared experts first**

Since shared_experts takes column-sharded input (no all_gather needed) and routed experts need the all_gather result, compute shared_experts BEFORE the all_gather:

```python
residual = x
# Shared experts can run on column-sharded input directly
shared_output = self.shared_experts(residual)

# Then all_gather and do routed path
x = all_gather_async(x, dim=-1, ...)
# ... gate, router, experts, reduce_scatter ...
output = add(routed_output, shared_output.to_ttnn)
```

**Wait:** The shared_experts' `TTNNGlm4MoeMLP` uses `TTNNLinearIColShardedWRowSharded` which internally does reduce_scatter. So shared_experts already operates on column-sharded input.

However, the shared expert weights are currently set up to operate on column-sharded input producing column-sharded output (with reduce_scatter inside each linear). The routed output after reduce_scatter is also column-sharded. So the add is column-sharded + column-sharded = column-sharded. This should work correctly.

**The key insight:** Computing shared_experts(residual) before the all_gather means the shared expert matmuls can potentially overlap with the all_gather's data transfer on the Ethernet links (matmul uses Tensix cores, all_gather uses Ethernet cores).

**Estimated savings:** 0.5-1ms (partial or full overlap of shared expert computation with routed expert pipeline).

### Phase 3 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 3.1 Reorder shared experts before all_gather | Allow matmul/Ethernet overlap | ~0.5-1ms |

---

## Phase 4: Trace Capture (High Impact, Medium Effort)

### Optimization 4.1: Enable TTNN Trace Capture for MoE Decode

**Problem:** Every iteration, the host dispatches ~50+ ops to the device, paying host dispatch overhead (~100-200us per op for small tensors). With trace capture, the op sequence is recorded once and replayed from device memory, eliminating host dispatch entirely.

**Requirements for trace capture:**
1. Deterministic tensor shapes across iterations (satisfied for decode: T=1, batch=1)
2. No data-dependent control flow (satisfied: n_group=1 path is fixed)
3. No CPU fallbacks (the TTNNMoERouterDecode does NOT have the CPU sort -- it was only in TTNNGlm4MoeRouteTokenToExperts. Verified by code reading.)
4. Pre-allocated persistent output buffers

**Note on CPU sort:** The `_to_torch_any` CPU sort (lines 806-813) exists in `TTNNGlm4MoeRouteTokenToExperts.forward()`, NOT in `TTNNMoERouterDecode.forward()`. Since `TTNNMoE` uses `TTNNMoERouterDecode`, there is **no CPU fallback** in the path we are optimizing. This is good -- trace capture is feasible without fixing the sort.

However, if the older `TTNNGlm4MoeMoE` class is used instead of `TTNNMoE`, the CPU sort IS present. For that path, the sort would need to be replaced with:
```python
# Device-side sort replacement:
# topk with sorted=True already returns sorted indices
# The sort in TTNNGlm4MoeRouteTokenToExperts is sorting by WEIGHT descending
# This can be done with another topk call:
sorted_weights, sort_idx = ttnn.topk(topk_weights, k=top_k, dim=1, largest=True, sorted=True)
sorted_indices = ttnn.gather(topk_expert_idx, dim=1, index=sort_idx)
```

**Implementation steps:**
1. Ensure the `@disable_trace` decorator is NOT present on `TTNNMoE.forward()` (it is not, based on code reading -- only `TTNNGlm4MoeExpertLayers.forward` has it)
2. Add trace capture wrapper at the model level (not the MoE module level) following the tt-transformers pattern:
   ```python
   # First iteration: record trace
   trace_id = ttnn.begin_trace_capture(device)
   output = model.moe_forward(x)
   ttnn.end_trace_capture(device, trace_id)

   # Subsequent iterations: replay trace
   ttnn.execute_trace(device, trace_id)
   ```

**Estimated savings:** 2-3ms per MoE layer (eliminates host dispatch overhead for ~50+ ops).

### Phase 4 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 4.1 Trace capture for MoE decode | Eliminate host dispatch overhead | ~2-3ms |

---

## Summary: Optimization Priority and Impact

| Phase | Optimization | Est. Savings | Effort | Risk |
|-------|-------------|-------------|--------|------|
| **1** | Router simplification (1-pass topk, eliminate typecasts, skip repeats) | 2-4ms | Medium | Low-Medium (validate accuracy) |
| **2** | Expert pipeline (matmul weight application, L1 layout, reduce padding) | 1-2.5ms | Medium | Low |
| **3** | Shared expert overlap (reorder before all_gather) | 0.5-1ms | Low | Low |
| **4** | Trace capture | 2-3ms | Medium | Low (no CPU fallback in this path) |
| **Total** | | **5.5-10.5ms** | | |

Note: Some savings overlap (trace capture subsumes some of the per-op dispatch overhead savings from Phases 1-2). Realistic combined savings: **5-8ms per MoE layer**.

---

## Implementation Order

### Step 1: Validate Baseline (no code changes)
- Run `test_moe.py` with timing instrumentation to measure current MoE decode latency
- Capture Tracy profile of MoE forward pass to get per-op breakdown
- Confirm Ling-mini-2.0 uses `TTNNMoE` (not `TTNNGlm4MoeMoE`) and thus has no CPU sort

### Step 2: Phase 1 (Router) -- Highest ROI
1. **1.1:** Remove bf16 round-trip in gate->router handoff
2. **1.4:** Add T=1 fast path to skip repeat/to_layout for bias and scale
3. **1.3:** Pre-store bias and scale in f32 on device
4. **1.2:** Replace 3-pass topk with 1-pass topk (requires accuracy validation)

### Step 3: Phase 3 (Shared Expert Reorder) -- Lowest Effort
1. Move `shared_experts(residual)` before `all_gather_async(x)`
2. Verify correctness with test suite

### Step 4: Phase 2 (Expert Pipeline) -- Medium Effort
1. **2.1:** Replace weight application with matmul-based approach
2. **2.2:** Switch layout conversions to L1 memory config
3. Run performance comparison

### Step 5: Phase 4 (Trace Capture) -- Requires all above to stabilize
1. Ensure no data-dependent shapes in MoE decode path
2. Implement trace capture at model level
3. Measure end-to-end improvement

---

## Correctness Validation

For each optimization, run:
1. `test_moe.py` -- unit test comparing TTNN vs PyTorch outputs
2. `test_ling_mini_2_0.py` -- end-to-end generation test (128 tokens, coherent text)
3. Compare topk agreement rate between old and new router (for Phase 1.2)

Acceptance criteria:
- Unit test PCC (Pearson correlation) > 0.99
- End-to-end text generation produces coherent output
- Topk agreement > 99.5% across 1000 random inputs (for router changes)

---

## Appendix: Ling-mini-2.0 MoE Dimensions

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| moe_intermediate_size | 1536 |
| n_routed_experts | 64 |
| num_experts_per_tok | 4 |
| n_group | 1 |
| topk_group | 1 |
| n_shared_experts | 1 |
| routed_scaling_factor | 1.8 |
| norm_topk_prob | True |
| Activation | sigmoid (not softmax) |
| T3K devices | 8 |
| Experts per device | 8 |
| SPARSITY_BLOCK_SIZE | 32 |
| Gate weight shape | (64, 2048) bf16 |
| Expert w1 shape per device | (8, 2048, 1536) bf16 |
| Expert w3 shape per device | (8, 2048, 1536) bf16 |
| Expert w2 shape per device | (8, 1536, 2048) bf16 |
