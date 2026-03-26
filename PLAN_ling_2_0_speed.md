# Plan: Speed Up Ling-mini-2.0 and Ensure Correctness

**Date:** 2026-03-26
**Author:** Architect Agent
**Current state:** All tests pass, text generation is correct
**Target:** Reduce per-token decode latency from ~935ms to <100ms (10x improvement)

---

## 1. Current State (All Tests Passing)

### Test Results (2026-03-26)

| Test | Status | Notes |
|------|--------|-------|
| `test_ling_mini_2_0.py` | PASSED | Correct text: "As an AI, I don't have personal preferences..." |
| `test_full_forward_prefill_with_paged_cache` | PASSED | PCC verified |
| `test_full_forward_decode_with_paged_cache` | PASSED | PCC verified |
| `test_multi_token_decode_sequence` | PASSED | PCC 0.999453-0.999636 across 5 decode steps |
| `test_weight_sharing_verification` | PASSED | Weight sharing correct |
| `test_integration_with_model_layer` | PASSED | PCC 0.999748 |

**No correctness work needed.** All 6 tests pass with high PCC (>0.999).

### Current Performance

- **Per-token decode:** ~935ms (85 tokens generated in ~79.5s)
- **20 decoder layers** (all with attention, 19 with MoE, 1 with dense MLP)
- Running on T3K (1x8 mesh, 8 Wormhole devices)

### Time Breakdown Per Token (ms)

| Component | Time (ms) | Per Layer (ms) | % of Total |
|-----------|-----------|----------------|------------|
| Attention (20 layers) | 362 | 18.1 | 38.7% |
| MoE (19 layers) | 461 | 24.3 | 49.3% |
| - Router | 150 | 7.9 | 16.0% |
| - Experts (sparse_matmul + dispatch/combine) | 110 | 5.8 | 11.8% |
| - Shared experts | 125 | 6.6 | 13.3% |
| Layer norms (torch, non-TTNN) | 11 | 0.5 | 1.1% |
| Other (embedding, lm_head, overhead) | ~101 | -- | 10.8% |

### Standalone Attention Profiling (from ANALYSIS_bailing_attention_profiling.md)

Standalone BailingAttention decode = 9.7ms/iter. The full model shows 18.1ms/layer because the TT-Symbiote wrapper overhead (wrap_to_torch, _set_distributed_config, _unwrap) adds ~8ms per attention layer. This wrapper overhead is:
- `wrap_to_torch_ttnn_tensor__set_device_wrap`: 0.27ms/call x ~594 calls/iter = 159ms/iter
- `_unwrap_to_torch`: 0.99ms/call x ~80 calls/iter = 79ms/iter
- `wrap_to_torch_ttnn_tensor`: 0.20ms/call x ~282 calls/iter = 58ms/iter

Total wrapper overhead: ~315ms/token (34% of total time).

---

## 2. Root Cause Analysis: Why 935ms/token?

The bottlenecks fall into 5 categories, ranked by impact:

### Category A: TT-Symbiote Wrapper Overhead (~315ms/token, 34%)

Every TTNN op result passes through `TorchTTNNTensor` wrappers that:
1. Convert between torch and TTNN tensor metadata representations
2. Set distributed configuration on every call
3. Run Python-level dispatch through module `__call__` overhead

**With ~600+ wrapper calls per token at ~0.2-1.0ms each, this is the single largest bottleneck.**

### Category B: MoE Router on Host (~150ms/token across 19 layers, 16%)

`TTNNMoERouterDecode.forward()` performs many small TTNN ops on device:
- 3-pass topk centering (3x ttnn.topk, 3x typecast, 3x sub, slices)
- Group masking with scatter, repeat, reshape
- Weight normalization with gather, sum, div
- ~40+ individual TTNN ops per router call
- Each op has host dispatch overhead >> device compute for batch=1

### Category C: Attention Data Movement (~137ms/token across 20 layers, 15%)

Per the standalone profiling, within each attention decode:
- 4 all_gather ops: ~2-3ms
- 9 to_memory_config resharding ops: ~1.5-2.5ms
- Reshapes and typecasts: ~0.5ms
- Host round-trips (cache_position via torch): ~0.5ms

### Category D: Expert Compute + Dispatch/Combine (~110ms/token, 12%)

- `all_to_all_dispatch` + `all_to_all_combine`: token routing across devices
- 3x `sparse_matmul` (gate_up, up, down projections)
- Multiple reshape/permute/to_layout conversions between ROW_MAJOR and TILE_LAYOUT
- Weight application with repeat + permute

### Category E: Shared Experts (~125ms/token across 19 layers, 13%)

`TTNNGlm4MoeMLP.forward()` runs 3 linear projections (gate+silu, up, down) using
`TTNNLinearIColShardedWRowSharded`. Each linear has wrapper + reduce_scatter overhead.

---

## 3. Optimization Plan

### Phase 1: Eliminate Wrapper Overhead (Target: -300ms/token, from ~935ms to ~635ms)

**Priority: HIGHEST. Estimated effort: 2-3 days.**

The TorchTTNNTensor wrapper system adds ~315ms/token of pure Python overhead. This must be eliminated or bypassed for the decode hot path.

#### 1a. Direct TTNN Decode Loop (Bypass TT-Symbiote Wrappers)

Write a dedicated `model.decode_one_token(input_ids, cache_position)` method that:
- Calls TTNN ops directly without going through `TorchTTNNTensor` wrapping/unwrapping
- Keeps all intermediate tensors as raw `ttnn.Tensor` objects
- Only converts to/from torch at the model boundary (input embedding -> ... -> lm_head output)

This is the pattern used in tt-transformers: the decode loop is a direct sequence of TTNN calls with no wrapper overhead.

**Implementation:**
1. In `TTNNBailingMoEAttention`, create `_forward_decode_paged_raw()` that takes/returns raw `ttnn.Tensor`
2. In `TTNNBailingMoE` / `TTNNMoE`, create `_forward_raw()` similarly
3. Wire them together in a top-level `decode_forward()` that skips the TT-Symbiote module dispatch

#### 1b. Reduce Wrapper Calls

Alternatively (less invasive), modify the wrapper to:
- Cache distributed config (skip `_set_distributed_config` when config hasn't changed)
- Make `wrap_to_torch_ttnn_tensor` a no-op when the input is already wrapped
- Lazy-evaluate the `to_ttnn` / `to_torch` conversions

**Expected impact:** Eliminating 315ms of the 935ms per token.

### Phase 2: Optimize Attention Decode (Target: -6ms/layer = -120ms/token)

**Priority: HIGH. Estimated effort: 3-5 days.**

From the detailed profiling analysis (PLAN_optimize_bailing_attention.md):

#### 2a. Fused QKV + AllReduce (Saves 2-3ms/layer)

Replace the current 4 all_gather + 1 reduce_scatter + 4 matmul pattern with:
- 1 fused QKV matmul + 1 all_reduce (or reduce_scatter + all_gather)
- Creates `TTNNLinearIColShardedWAllReduced` linear class
- Eliminates 3 all_gathers, 3 matmuls, reduces CCL ops from 5 to 1-2

**Current data flow:**
```
hidden -> q_proj -> reduce_scatter -> all_gather(Q)     # 1 matmul + 1 RS + 1 AG
hidden -> all_gather -> k_proj -> all_gather(K)          # 1 AG + 1 matmul + 1 AG
                     -> v_proj -> all_gather(V)          # 1 matmul + 1 AG
```

**Optimized data flow:**
```
hidden -> fused_qkv_proj -> all_reduce -> split(Q,K,V)   # 1 matmul + 1 AR
```

#### 2b. Use nlp_create_qkv_heads_decode (Saves 0.4-0.5ms/layer)

Replace 3 manual reshapes + 2 to_memory_config with a single `ttnn.experimental.nlp_create_qkv_heads_decode` call that produces HEIGHT_SHARDED output directly.

#### 2c. Eliminate QK Norm Reshapes (Saves 0.1-0.2ms/layer)

Test if `ttnn.rms_norm` can operate on 4D `[1, B, H, head_dim]` tensors directly, eliminating 4 reshape ops per layer.

#### 2d. Pre-cache cos/sin in Sharded Format (Saves 0.1ms/layer)

Move the cos/sin RoPE tensors to HEIGHT_SHARDED format during `move_weights_to_device_impl` instead of resharding every decode step.

**Total Phase 2 savings: ~3-4ms/layer x 20 layers = 60-80ms/token**

### Phase 3: Optimize MoE Router (Target: -5ms/layer = -95ms/token)

**Priority: HIGH. Estimated effort: 2-3 days.**

The router at 7.9ms/layer is dominated by many small TTNN ops with host dispatch overhead.

#### 3a. Fuse Router Ops on Host (Torch Fallback for Small Tensors)

For batch=1, the router operates on tiny tensors (1x1x1x64 experts). The TTNN op dispatch overhead (kernel launch, synchronization) far exceeds the actual compute. Options:

1. **Run router on CPU with torch:** For decode (batch=1, 64 experts), the topk + sigmoid + normalization is trivial on CPU (~0.01ms). Transfer the router logits to host (tiny: 64 floats), compute routing on CPU, transfer indices/weights back to device.

2. **Batch the router ops:** Accumulate router inputs across layers and batch-process them. (Not practical due to layer-sequential nature.)

3. **Pre-compute a fused router kernel:** Create a custom TTNN kernel that does sigmoid + bias + 3-pass-topk + normalize in a single kernel launch.

**Recommendation:** Option 1 (torch fallback for decode router) is the fastest to implement and most impactful. The tensor transfer cost (~0.1ms for 64 floats) is negligible compared to the 7.9ms of TTNN op dispatch overhead.

#### 3b. Simplify Group Selection

The current 3-pass topk centering approach (used to work around BF16 precision in topk) could be simplified:
- Use float32 topk if available
- Or use a single-pass approach with larger k and post-filtering

### Phase 4: Optimize Expert Computation (Target: -2ms/layer = -38ms/token)

**Priority: MEDIUM. Estimated effort: 2-3 days.**

#### 4a. Reduce Layout Conversions in Expert Pipeline

The expert pipeline does multiple `to_layout(ROW_MAJOR)` -> `to_layout(TILE_LAYOUT)` conversions:
- Before all_to_all_dispatch: TILE -> ROW_MAJOR
- After dispatch: ROW_MAJOR -> TILE
- Before all_to_all_combine: TILE -> ROW_MAJOR
- After combine: ROW_MAJOR -> TILE

Investigate if all_to_all_dispatch/combine can operate on TILE_LAYOUT tensors directly.

#### 4b. Fuse Weight Application

The expert weight application (lines 1320-1335) does:
```python
weights -> to_layout(ROW_MAJOR) -> unsqueeze -> unsqueeze -> repeat -> permute -> to_layout(TILE)
output = mul(combined, weights)
final = sum(output, dim=0)
```

This is 7 ops that could be replaced with a single weighted reduction kernel or at minimum fewer reshapes.

### Phase 5: Trace Capture for Decode (Target: -200ms/token additional)

**Priority: HIGH (but depends on Phases 1-4). Estimated effort: 3-5 days.**

After eliminating wrapper overhead and optimizing individual ops, trace capture can eliminate remaining host dispatch overhead:

1. **Capture a TTNN trace** for the full decode forward pass (all 20 layers)
2. **Replay the trace** for each decode step, only updating:
   - Input hidden_states
   - cache_position
   - KV cache pointers (already on device)

**Prerequisites:**
- All tensor shapes must be fixed across decode iterations (they are for batch=1)
- No Python-level conditionals in the hot path (need to verify)
- All intermediate buffers must be pre-allocated

**Expected impact:** Eliminates all host dispatch overhead, which includes Python interpreter time, op compilation checks, and command queue submission. Based on tt-transformers benchmarks, this typically provides 2-5x speedup for small-batch decode.

### Phase 6: Shared Expert Optimization (Target: -2ms/layer = -38ms/token)

**Priority: MEDIUM. Estimated effort: 1-2 days.**

#### 6a. Overlap Shared Experts with Routed Experts

Currently the shared experts run sequentially after routed experts:
```python
routed_output = self.experts(x, ...)     # 5.8ms
shared_output = self.shared_experts(residual)  # 6.6ms
output = ttnn.add(routed_output, shared_output)
```

Since shared experts operate on the `residual` (not the routed output), they can run **in parallel** with the routed expert computation using separate command queues or async dispatch.

#### 6b. Fuse Gate+Up in Shared MLP

If `TTNNGlm4MoeMLP` uses separate gate and up projections, fuse them into a single matmul (gate_up_proj) followed by chunk + silu + mul.

---

## 4. Implementation Priority and Expected Results

| Phase | Optimization | Est. Savings | Cumulative | Effort | Risk |
|-------|-------------|-------------|------------|--------|------|
| 1 | Eliminate wrapper overhead | ~315ms | ~620ms | 2-3 days | Low (proven pattern) |
| 2 | Optimize attention decode | ~80ms | ~540ms | 3-5 days | Medium |
| 3 | Optimize MoE router | ~95ms | ~445ms | 2-3 days | Low |
| 5 | Trace capture | ~200ms | ~245ms | 3-5 days | Medium |
| 4 | Optimize expert compute | ~38ms | ~207ms | 2-3 days | Medium |
| 6 | Shared expert optimization | ~38ms | ~170ms | 1-2 days | Low |

**Realistic target with Phases 1-5:** ~200-300ms/token (3-5x improvement)
**Stretch target with all phases + trace:** ~100-170ms/token (5-9x improvement)

Note: These estimates assume the optimizations do not overlap. In practice, trace capture (Phase 5) subsumes much of the host dispatch overhead that Phases 2-4 address, so the cumulative benefit may be less than the sum.

---

## 5. Recommended Execution Order

1. **Phase 1a** (raw TTNN decode loop) -- highest ROI, unlocks all subsequent optimizations
2. **Phase 3a** (torch fallback for router) -- quick win, independent of Phase 1
3. **Phase 2a** (fused QKV + all_reduce) -- the main attention optimization
4. **Phase 5** (trace capture) -- requires Phases 1-3 to be stable
5. **Phases 4, 6** (expert + shared expert optimizations) -- incremental gains

---

## 6. Key Files to Modify

| File | Changes |
|------|---------|
| `modules/attention.py` | Fused QKV, nlp_create_qkv_heads_decode, pre-cached cos/sin, raw forward |
| `modules/moe.py` | Router torch fallback, expert layout optimizations, shared expert overlap |
| `modules/linear.py` | New `TTNNLinearIColShardedWAllReduced` class |
| `tests/test_ling_mini_2_0.py` | Raw decode loop, perf benchmarking |
| `core/` (TT-Symbiote framework) | Wrapper overhead reduction if Phase 1b chosen |

---

## 7. Correctness Safeguards

Every optimization must maintain:
1. **PCC > 0.999** on `test_bailing_attention_accuracy.py` (all 5 tests)
2. **Correct text generation** on `test_ling_mini_2_0.py`
3. **Incremental validation:** Each phase should be tested independently before combining

The existing test suite provides sufficient coverage. No new tests needed unless the decode loop structure changes fundamentally (Phase 1a).
