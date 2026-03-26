# Optimization Plan: BailingMoE (TTNNBailingMoE) Decode Path

**Date:** 2026-03-26 (Updated with DeepSeek V3 analysis)
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

## DeepSeek V3 Reference Implementation Comparison

The DeepSeek V3 (DS V3) official implementation in tt-metal provides a production-validated reference for MoE optimization on Tenstorrent hardware. Key differences are analyzed below to inform optimization priorities.

### Router (moe_gate.py vs TTNNMoERouterDecode)

| Aspect | Symbiote (Current) | DeepSeek V3 | Implication |
|--------|-------------------|-------------|-------------|
| Topk approach | 3-pass centering (topk + center + topk + center + topk) | Single-pass topk per stage | **Confirms single-pass is production-viable** |
| Data types | f32 throughout router, 9 bf16<->f32 typecasts | bf16 throughout, bias stored f32 on device | **bf16 is sufficient for routing** |
| Gate precision | HiFi4 with fp32 accumulation | HiFi2 precision | Lower precision is acceptable |
| Sort fallback | No CPU sort in TTNNMoERouterDecode | `topk_fallback` option with bitonic sort on CPU, `use_bitonic_sort=True` default | DS V3 acknowledges ttnn.topk can be unreliable |
| Memory config | DRAM (default) | L1_MEMORY_CONFIG everywhere in decode | **L1 is the production standard for decode** |

### Expert Pipeline (experts.py vs TTNNExperts)

| Aspect | Symbiote (Current) | DeepSeek V3 | Implication |
|--------|-------------------|-------------|-------------|
| Matmul type | sparse_matmul | Regular ttnn.linear | DS V3 does NOT use sparse_matmul |
| SiLU activation | Separate `ttnn.silu(w1_out)` then `ttnn.mul(w1_activated, w3_out)` | Fused: `ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])` | **Fused SiLU eliminates one op** |
| Weight quantization | bf16 for all projections | bfloat4_b (gate/down_proj), bfloat8_b (up_proj) | Quantization possible but not priority for Ling |
| Compute kernel | Default / HiFi2 | COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc | LoFi is production choice |
| Memory config | DRAM (default) | L1_MEMORY_CONFIG | **L1 is the production standard** |

### Weight Application (moe.py)

| Aspect | Symbiote (Current) | DeepSeek V3 | Implication |
|--------|-------------------|-------------|-------------|
| Pattern | repeat(hidden_size) -> permute -> to_layout -> mul -> sum | **IDENTICAL**: repeat(hidden_size) -> permute -> to_layout -> mul -> sum | **This is the accepted production approach** |
| Function | Inline in TTNNExperts.forward() | `_fwd_repeat_permute_expert_weights()` | Same algorithm, just extracted |

**Key finding:** The repeat(hidden_size) weight application pattern is NOT a symbiote bug -- it is the accepted production approach in DS V3. Optimization effort should be redirected to higher-impact areas.

### MoE Decoder Block (moe_decoder_block_2d.py vs TTNNBailingMoEBlock)

| Aspect | Symbiote (Current) | DeepSeek V3 | Implication |
|--------|-------------------|-------------|-------------|
| Shared expert input | Uses residual (column-sharded) | Uses x_gathered (same as routed experts) | Different approach, both valid |
| Reduce pattern | reduce_scatter(routed_out), shared_experts has internal reduce_scatter, then add | **add(moe_out, shared_out) -> single reduce_scatter** | **KEY OPTIMIZATION: one RS instead of two** |
| Dispatch cluster_axis | cluster_axis=1 (default?) | cluster_axis=0 | May affect performance on T3K |
| num_links | Default | num_links=4 for all_to_all ops | **Higher link count = more bandwidth** |

### Custom Kernel (deepseek_moe_gate.hpp)

DS V3 b1 variant has a **custom Tensix kernel** that fuses: sigmoid + bias add + sorting + normalization into a single kernel. This eliminates ALL individual routing ops. This is the most aggressive optimization path but requires kernel development expertise.

### Memory Strategy

DS V3 decode uses:
- **L1_MEMORY_CONFIG** everywhere (gate, experts, combine, routing)
- **Sharded memory config** for input/output: `create_sharded_memory_config(shape=(32, hidden_size//8), core_grid=CoreGrid(y=7, x=4), strategy=WIDTH)`

Symbiote currently uses DRAM (default) for most operations.

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

**DS V3 validation:** DS V3 uses single-pass topk per stage and works in bf16 throughout the router. This confirms single-pass topk is production-viable and f32 typecasts are unnecessary.

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

**DS V3 validation:** DS V3 uses the **IDENTICAL pattern** in `_fwd_repeat_permute_expert_weights()`. This is the accepted production approach. **De-prioritized** -- optimization effort should go elsewhere.

### Bottleneck 3: Layout Conversions (ROW_MAJOR <-> TILE_LAYOUT)

The expert forward path has **at least 6 explicit layout conversions:**
1. `to_layout(x, ROW_MAJOR)` -- before all_to_all_dispatch (Step 4d)
2. `to_layout(post_dispatch, TILE_LAYOUT)` -- before sparse_matmul (Step 4j)
3. `to_layout(expert_output, ROW_MAJOR)` -- before all_to_all_combine (Step 4v)
4. `to_layout(combined_output, TILE_LAYOUT)` -- after combine (Step 4z)
5. `to_layout(topk_weights, ROW_MAJOR)` -- weight application (Step 4aa)
6. `to_layout(topk_weights, TILE_LAYOUT)` -- weight application (Step 4af)

Conversions 1-4 are required by API constraints. Conversions 5-6 are part of the accepted weight application pattern (same in DS V3).

### Bottleneck 4: Two Reduce-Scatters Instead of One (Steps 5-7)

**This is a newly identified HIGH-IMPACT bottleneck based on DS V3 analysis.**

Currently:
```
routed_output = experts(x, ...)                      # Step 4
routed_output = reduce_scatter(routed_output)         # Step 5 -- reduce_scatter #1
shared_output = shared_experts(residual)              # Step 6 -- contains internal reduce_scatter #2
output = add(routed_output, shared_output)            # Step 7
```

The shared_experts module (`TTNNGlm4MoeMLP`) uses `TTNNLinearIColShardedWRowSharded` for down_proj, which internally does a reduce_scatter. So there are **two reduce_scatter operations** total.

DS V3 approach:
```
routed_output = experts(x_gathered, ...)
shared_output = shared_experts(x_gathered)
combined = add(routed_output, shared_output)           # Add BEFORE reduce_scatter
output = reduce_scatter(combined)                      # Single reduce_scatter
```

By adding the routed and shared outputs BEFORE reduce_scatter, DS V3 uses a **single reduce_scatter** instead of two. This saves one full CCL operation per MoE layer.

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

**DS V3 validation:** DS V3 does the entire gate in bf16 with HiFi2. No f32 typecasts at all.

### Bottleneck 6: Separate SiLU Call (Step 4p-4q)

Currently:
```python
w1_out = sparse_matmul(x, w1, sparsity)   # gate projection
w3_out = sparse_matmul(x, w3, sparsity)   # up projection
w1_activated = ttnn.silu(w1_out)            # separate SiLU op
intermediate = ttnn.mul(w1_activated, w3_out)
```

DS V3 fuses SiLU into the multiply:
```python
intermediate = ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
```

This eliminates one op and one intermediate tensor allocation.

### Bottleneck 7: DRAM Memory Config (All Steps)

Symbiote uses DRAM (default memory config) for most operations. DS V3 uses **L1_MEMORY_CONFIG** everywhere in decode mode. For small decode tensors (T=1), L1 eliminates DRAM round-trips and reduces latency for every op.

---

## Phase 1: Router Simplification (High Impact, Medium Effort)

### Optimization 1.1: Eliminate bf16 Round-Trip in Gate -> Router Handoff

**Problem:** Gate outputs f32 logits, typecasts to bf16, router typecasts back to f32.

**DS V3 approach:** Gate operates entirely in bf16. No f32 at all.

**Solution (conservative):** Pass f32 logits directly from gate to router. Remove the intermediate typecast.
**Solution (aggressive, DS V3 style):** Run gate in bf16 with HiFi2. Eliminate all f32 from the gate path.

**Code changes (conservative):**
- In `TTNNMoE.forward()` (line ~1458-1460): Remove `typecast(router_logits_f32, bf16)`. Pass `router_logits_f32` directly to `route_tokens_to_experts()`.
- In `TTNNMoERouterDecode.forward()` (lines 897-901): The input is already f32, skip the `typecast(logits, float32)` step.

**Saves:** 2 typecast ops per iteration (conservative), or 4+ typecasts (aggressive).

### Optimization 1.2: Simplify 3-Pass Topk to 1-Pass

**Problem:** When `n_group <= topk_group` (true for Ling-mini-2.0 where both equal 1), all groups are selected. The 3-pass topk centering exists to handle bf16 precision issues, but DS V3 demonstrates single-pass topk works in production with 256 experts (more than our 64).

**DS V3 validation:** DS V3 uses single-pass topk per stage. With 64 experts and sigmoid scores in [0,1], bf16 precision is more than adequate.

**Solution:** Replace the 3-pass centering with a single bf16 topk (following DS V3's approach):
```python
# Instead of 3-pass centering:
scores_bf16 = typecast(scores_with_bias_f32, bf16)  # or keep in bf16 throughout
_, topk_expert_idx = ttnn.topk(scores_bf16, k=top_k, dim=3)
```

This is 1 typecast + 1 topk instead of 6 typecasts + 3 topks + 2 slices + 2 subs.

**Risk:** Low. DS V3 validates this approach in production with 4x more experts. Mitigation: Run the test suite comparing 1-pass vs 3-pass topk outputs across 1000+ random inputs.

**Saves:** ~20 ops per router call (6 typecasts, 2 topks, 2 slices, 2 subs, several deallocates).

### Optimization 1.3: Pre-Store Bias and Scale as f32 on Device

**Problem:** Bias and scale stored as bf16, typecast to f32 at runtime.

**DS V3 approach:** Score correction bias stored in **float32 on device** (scores stay bf16).

**Solution:** Pre-compute bias and scale in f32 during `preprocess_weights_impl()` and `move_weights_to_device_impl()`.

**Code changes:**
- In `TTNNMoERouterDecode.preprocess_weights_impl()`: Store `_bias_torch` and `_scale_torch` as float32 tensors
- In `TTNNMoERouterDecode.move_weights_to_device_impl()`: Use `dtype=ttnn.float32` when creating device tensors

**Saves:** 2 typecast ops per iteration.

### Optimization 1.4: Eliminate Unnecessary repeat + to_layout for Bias and Scale

**Problem:** For decode with T=1, the repeat is `(1,1,1,1)` -- a no-op that triggers `ttnn.clone()`.

**Solution:** For decode mode (T=1), skip the repeat entirely since broadcasting suffices.

```python
if T == 1:
    bias = self._bias_dev  # Already (1,1,1,64), broadcasts with (1,1,1,64)
else:
    bias = _safe_repeat(self._bias_dev, ttnn.Shape((1,1,T,1)))
```

**Saves:** 2 repeat/clone ops + 2 to_layout calls per iteration for decode.

### Phase 1 Summary

| Optimization | Ops Removed | Description |
|-------------|------------|-------------|
| 1.1 Gate->Router handoff | 2-4 typecast | Pass f32 directly (or go full bf16 like DS V3) |
| 1.2 1-pass topk | ~20 ops | Eliminate 2 extra topk passes + centering (DS V3 validated) |
| 1.3 Pre-store f32 constants | 2 typecast | Bias/scale already f32 on device (matches DS V3) |
| 1.4 Skip repeat for T=1 | 2 repeat + 2 to_layout | Broadcasting suffices |
| **Total** | **~28 ops** | Router: ~30 ops -> ~5-7 ops |

**Estimated savings:** 2-4ms per MoE layer (host dispatch of ~28 eliminated ops at ~100-200us each for small tensors).

---

## Phase 2: Fused Shared Expert + Single Reduce-Scatter (HIGH Impact, Medium Effort)

**This is the highest-impact structural optimization, derived from DS V3 analysis.**

### Optimization 2.1: Combine Routed + Shared Output Before Reduce-Scatter

**Problem:** Currently there are TWO reduce_scatter operations per MoE layer:
1. `reduce_scatter(routed_output)` -- explicit after expert combine
2. `reduce_scatter` inside `shared_experts.down_proj` (TTNNLinearIColShardedWRowSharded)

**DS V3 approach:** Both routed and shared experts use the gathered (replicated) input. Their outputs are added together, then a SINGLE reduce_scatter is performed.

**Solution:** Restructure the MoE forward pass to match DS V3:
```python
# Current (two reduce-scatters):
x_gathered = all_gather(x, dim=-1)
routed_output = experts(x_gathered, ...)
routed_output = reduce_scatter(routed_output)       # RS #1
shared_output = shared_experts(residual)             # Contains internal RS #2
output = add(routed_output, shared_output)

# Proposed (single reduce-scatter, DS V3 style):
x_gathered = all_gather(x, dim=-1)
routed_output = experts(x_gathered, ...)
shared_output = shared_experts_no_rs(x_gathered)     # NEW: shared experts without reduce_scatter
combined = add(routed_output, shared_output)          # Add in replicated space
output = reduce_scatter(combined)                     # Single RS
```

**Implementation:**
1. Create a variant of shared_experts that outputs replicated (not column-sharded) result. This means using `TTNNLinear` (without reduce_scatter) for down_proj instead of `TTNNLinearIColShardedWRowSharded`.
2. Feed `x_gathered` (replicated) to shared_experts instead of `residual` (column-sharded). This changes the shared expert input from column-parallel to replicated.
3. Add routed + shared outputs, then do a single reduce_scatter.

**Challenges:**
- Shared expert weights currently assume column-parallel input (gate_proj, up_proj split across devices). For replicated input, weights need to be replicated too, or the shared expert needs to be re-sharded.
- Alternative: Keep shared expert column-parallel but delay its reduce_scatter. The shared expert does: gate_proj (col-parallel) -> up_proj (col-parallel) -> mul -> down_proj (row-parallel, no RS). Then add the row-parallel shared output with the routed output (also replicated), then RS once.

**Simpler variant (recommended):** Modify only the reduce_scatter placement:
```python
x_gathered = all_gather(x, dim=-1)
routed_output = experts(x_gathered, ...)              # replicated output
shared_output = shared_experts_inner(x_gathered)      # replicated output (no RS)
combined = add(routed_output, shared_output)
output = reduce_scatter(combined)                     # Single RS
```

Where `shared_experts_inner` is the shared expert MLP without the final reduce_scatter in down_proj. This requires using a non-reducing linear for down_proj.

**Estimated savings:** 1-2ms (eliminates one full reduce_scatter CCL operation + its host dispatch overhead).

### Optimization 2.2: Use num_links=4 for all_to_all Operations

**Problem:** Symbiote uses default num_links for all_to_all_dispatch/combine.

**DS V3 approach:** Explicitly sets `num_links=4` for all all_to_all operations.

**Solution:** Add `num_links=4` to all_to_all_dispatch and all_to_all_combine calls.

**Code change:**
```python
# In TTNNExperts.forward():
dispatched = ttnn.experimental.all_to_all_dispatch(x, indices, mapping, num_links=4)
# ...
combined = ttnn.experimental.all_to_all_combine(output, metadata, mapping, num_links=4)
```

**Estimated savings:** 0.3-0.5ms (more Ethernet bandwidth for dispatch/combine).

### Phase 2 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 2.1 Single reduce_scatter (DS V3 pattern) | Add routed+shared before RS | ~1-2ms |
| 2.2 num_links=4 for all_to_all | More Ethernet bandwidth | ~0.3-0.5ms |
| **Total** | | **~1.5-2.5ms** |

---

## Phase 3: L1 Memory Config for Decode (Medium Impact, Low Effort)

**Derived entirely from DS V3 analysis. DS V3 uses L1_MEMORY_CONFIG everywhere in decode.**

### Optimization 3.1: Switch All Decode Ops to L1_MEMORY_CONFIG

**Problem:** Symbiote uses DRAM (default) for most operations. For decode with T=1, tensors are small enough to fit in L1. DRAM round-trips add latency for every op.

**DS V3 approach:** All decode ops use `ttnn.L1_MEMORY_CONFIG`:
- Gate projection output
- Router intermediate tensors
- Expert dispatch/combine
- Weight application
- Shared expert intermediates

**Solution:** Add `memory_config=ttnn.L1_MEMORY_CONFIG` to all ops in decode path:
```python
# Examples:
router_logits = ttnn.linear(x, gate_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
scores = ttnn.sigmoid(logits, memory_config=ttnn.L1_MEMORY_CONFIG)
x_rm = ttnn.to_layout(x, ROW_MAJOR, memory_config=ttnn.L1_MEMORY_CONFIG)
```

**Implementation:** Systematic pass through all ops in:
- `TTNNMoE.forward()` -- gate projection, router call, expert call
- `TTNNMoERouterDecode.forward()` -- all intermediate ops
- `TTNNExperts.forward()` -- layout conversions, weight application
- Shared expert MLP -- all matmul outputs

**Estimated savings:** 0.5-1ms (eliminates DRAM round-trips for ~50+ small-tensor ops).

### Optimization 3.2: Sharded Memory Config for Input/Output

**DS V3 approach:** Uses width-sharded memory config for MoE block input/output:
```python
create_sharded_memory_config(
    shape=(32, hidden_size // 8),
    core_grid=CoreGrid(y=7, x=4),
    strategy=WIDTH
)
```

This distributes the tensor across 28 cores (7x4 grid) for parallel processing.

**Solution:** Investigate applying sharded memory config to BailingMoE decode input/output. This requires:
1. Understanding the core grid constraints for Ling-mini-2.0 dimensions
2. Ensuring upstream/downstream ops can consume sharded tensors
3. May require `ttnn.to_memory_config()` calls at boundaries

**Estimated savings:** 0.2-0.5ms (reduced L1 bank conflicts, better parallelism).

### Phase 3 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 3.1 L1_MEMORY_CONFIG everywhere | Eliminate DRAM round-trips | ~0.5-1ms |
| 3.2 Sharded memory config | Width-sharded I/O | ~0.2-0.5ms |
| **Total** | | **~0.7-1.5ms** |

---

## Phase 4: Fused SiLU and Expert Compute Optimizations (Medium Impact, Low Effort)

### Optimization 4.1: Fuse SiLU into Multiply

**Problem:** Current code has two separate ops:
```python
w1_activated = ttnn.silu(w1_out)
intermediate = ttnn.mul(w1_activated, w3_out)
```

**DS V3 approach:** Fused into a single op:
```python
intermediate = ttnn.mul(w1_out, w3_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
```

**Solution:** Replace the separate silu + mul with fused mul in `TTNNExperts.forward()`.

**Saves:** 1 op + 1 intermediate tensor allocation per expert computation.

**Estimated savings:** 0.1-0.3ms.

### Optimization 4.2: Evaluate LoFi Compute Kernel for Expert Matmuls

**DS V3 approach:** Uses COMPUTE_KERNEL_CONFIG_LOFI with `packer_l1_acc=True` for expert computation.

**Solution:** Test LoFi compute config for expert matmuls:
```python
lofi_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

**Risk:** Potential accuracy impact. Requires validation against reference outputs.

**Estimated savings:** 0.2-0.5ms (faster compute, though experts may already be bandwidth-bound).

### Phase 4 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 4.1 Fused SiLU | Eliminate separate silu op | ~0.1-0.3ms |
| 4.2 LoFi compute kernel | Faster expert matmuls | ~0.2-0.5ms |
| **Total** | | **~0.3-0.8ms** |

---

## Phase 5: Trace Capture (High Impact, Medium Effort)

### Optimization 5.1: Enable TTNN Trace Capture for MoE Decode

**Problem:** Every iteration, the host dispatches ~50+ ops to the device, paying host dispatch overhead (~100-200us per op for small tensors). With trace capture, the op sequence is recorded once and replayed from device memory, eliminating host dispatch entirely.

**Requirements for trace capture:**
1. Deterministic tensor shapes across iterations (satisfied for decode: T=1, batch=1)
2. No data-dependent control flow (satisfied: n_group=1 path is fixed)
3. No CPU fallbacks (the TTNNMoERouterDecode does NOT have the CPU sort -- verified by code reading)
4. Pre-allocated persistent output buffers

**Note:** DS V3's `topk_fallback` with CPU bitonic sort would break trace capture. Our path (TTNNMoERouterDecode) does not have this issue.

**Implementation steps:**
1. Ensure the `@disable_trace` decorator is NOT present on `TTNNMoE.forward()`
2. Add trace capture wrapper at the model level following the tt-transformers pattern:
   ```python
   trace_id = ttnn.begin_trace_capture(device)
   output = model.moe_forward(x)
   ttnn.end_trace_capture(device, trace_id)
   # Subsequent iterations:
   ttnn.execute_trace(device, trace_id)
   ```

**Estimated savings:** 2-3ms per MoE layer (eliminates host dispatch overhead for ~50+ ops).

### Phase 5 Summary

| Optimization | Description | Est. Savings |
|-------------|-------------|-------------|
| 5.1 Trace capture for MoE decode | Eliminate host dispatch overhead | ~2-3ms |

---

## Summary: Optimization Priority and Impact (Revised with DS V3 Findings)

| Phase | Optimization | Est. Savings | Effort | Risk | DS V3 Validated |
|-------|-------------|-------------|--------|------|-----------------|
| **1** | Router simplification (1-pass topk, eliminate typecasts, skip repeats) | 2-4ms | Medium | Low (DS V3 validates) | Yes |
| **2** | Single reduce_scatter (add before RS) + num_links=4 | 1.5-2.5ms | Medium | Low-Medium | Yes |
| **3** | L1 memory config everywhere in decode | 0.7-1.5ms | Low | Low | Yes |
| **4** | Fused SiLU + LoFi compute kernel | 0.3-0.8ms | Low | Low-Medium | Yes |
| **5** | Trace capture | 2-3ms | Medium | Low | N/A (arch level) |
| **Total** | | **6.5-11.8ms** | | | |

Note: Some savings overlap (trace capture subsumes some per-op dispatch overhead from Phases 1-4). Realistic combined savings: **5-9ms per MoE layer**.

### Key changes from original plan:

1. **Weight application de-prioritized:** DS V3 uses the identical repeat+permute pattern. This is the accepted production approach, not a bug to fix. Original Phase 2.1 (matmul-based weight application) is removed.
2. **Single reduce_scatter elevated to Phase 2:** DS V3's `add(moe_out, shared_out) -> single RS` pattern is a high-impact structural optimization that replaces the original "overlap shared expert" approach.
3. **L1 memory config added as Phase 3:** DS V3 uses L1_MEMORY_CONFIG everywhere in decode. This is a systematic, low-effort optimization.
4. **Fused SiLU added as Phase 4:** DS V3's fused `mul(..., activations=[SiLU])` eliminates one op per expert computation.
5. **num_links=4 added to Phase 2:** DS V3 explicitly uses 4 links for all_to_all operations.
6. **Router risk lowered:** DS V3 validates single-pass topk in production with 256 experts, making our 64-expert simplification very safe.

---

## Implementation Order (Revised)

### Step 1: Validate Baseline (no code changes)
- Run `test_moe.py` with timing instrumentation to measure current MoE decode latency
- Capture Tracy profile of MoE forward pass to get per-op breakdown
- Confirm Ling-mini-2.0 uses `TTNNMoE` (not `TTNNGlm4MoeMoE`)

### Step 2: Phase 1 (Router) -- Highest ROI, DS V3 Validated
1. **1.1:** Remove bf16 round-trip in gate->router handoff
2. **1.4:** Add T=1 fast path to skip repeat/to_layout for bias and scale
3. **1.3:** Pre-store bias and scale in f32 on device
4. **1.2:** Replace 3-pass topk with 1-pass topk (DS V3 validates this approach)

### Step 3: Phase 3 (L1 Memory Config) -- Lowest Effort, DS V3 Pattern
1. Add `memory_config=ttnn.L1_MEMORY_CONFIG` to all decode-path ops
2. Systematic pass through TTNNMoE, TTNNMoERouterDecode, TTNNExperts, shared experts
3. Verify correctness with test suite

### Step 4: Phase 4 (Fused SiLU + LoFi) -- Low Effort
1. **4.1:** Replace separate `silu()` + `mul()` with fused `mul(..., activations=[SiLU])`
2. **4.2:** Test LoFi compute kernel config for expert matmuls
3. Validate accuracy

### Step 5: Phase 2 (Single Reduce-Scatter) -- Medium Effort, Structural Change
1. **2.1:** Restructure shared expert to not do internal reduce_scatter
2. **2.1:** Feed gathered input to shared expert (like DS V3)
3. **2.1:** Add routed + shared before single reduce_scatter
4. **2.2:** Add num_links=4 to all_to_all operations
5. Run full test suite

### Step 6: Phase 5 (Trace Capture) -- Requires all above to stabilize
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

## Appendix A: Ling-mini-2.0 MoE Dimensions

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

## Appendix B: DeepSeek V3 Reference Files

| File | Location in tt-metal | Key Content |
|------|---------------------|-------------|
| moe_gate.py | models/demos/deepseek_v3/tt/moe_gate.py | Single-pass topk, bf16 router, f32 bias on device |
| experts.py | models/demos/deepseek_v3/tt/experts.py | Regular ttnn.linear (not sparse_matmul), fused SiLU, bfloat4_b/8_b weights, LoFi |
| moe.py | models/demos/deepseek_v3/tt/moe.py | Weight application (repeat+permute, identical to symbiote), num_links=4 |
| moe_decoder_block_2d.py | models/demos/deepseek_v3/tt/moe_decoder_block_2d.py | add(routed, shared) -> single reduce_scatter |
| deepseek_moe_gate.hpp | tt_metal/hw/ckernels/.../deepseek_moe_gate.hpp | Custom Tensix kernel fusing entire gate (b1 variant) |
