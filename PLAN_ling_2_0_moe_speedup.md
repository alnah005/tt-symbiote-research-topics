# Plan: Speed Up MoE Module for Ling-mini-2.0 Decode

**Date:** 2026-03-26
**Model:** Ling-mini-2.0 (inclusionAI/Ling-mini-2.0)
**Module:** `TTNNBailingMoE` (subclass of `TTNNMoE`) in `models/experimental/tt_symbiote/modules/moe.py`
**Test:** `pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0`
**Environment:** MESH_DEVICE=T3K (1x8), Wormhole B0
**Success Criteria:** Correct text generation + measurable decode speedup

---

## 1. Problem Description

The MoE (Mixture of Experts) module is the dominant bottleneck in Ling-mini-2.0 decode. Each of the 19 MoE layers (layers 1-19; layer 0 is dense MLP) executes the full `TTNNMoE.forward` pipeline per decode token:

1. `all_gather_async` (Linear topology, 1 link) -- revert tensor parallelism
2. Float32 gate matmul (HiFi4) -- compute router logits
3. `TTNNMoERouterDecode.forward` -- 3-pass topk routing with CPU fallback sort
4. `TTNNExperts.forward` -- all_to_all_dispatch, 3x sparse_matmul, silu, all_to_all_combine, weight application
5. `reduce_scatter_minimal_async` (Ring topology) -- reduce routed output
6. `TTNNGlm4MoeMLP` -- shared expert computation (3 matmuls + silu)
7. `ttnn.add` -- combine routed + shared expert outputs

With 19 layers, MoE executes 19 times per decode token.

---

## 2. Current Architecture Analysis

### Ling-mini-2.0 MoE Configuration

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 3072 |
| `moe_intermediate_size` | 1024 |
| `n_routed_experts` | 64 |
| `num_experts_per_tok` | 6 |
| `n_group` | 1 |
| `topk_group` | 1 |
| `n_shared_experts` | 1 |
| `routed_scaling_factor` | 1.0 |
| `norm_topk_prob` | True |
| Experts per device (T3K) | 8 (64/8) |

### Key Observation: `n_group=1, topk_group=1`

Since `n_group <= topk_group`, the router takes the **single-pass topk** branch (moe.py L927-932), NOT the 3-pass centering branch. However, the current single-pass path still:
- Typecasts f32 -> bf16 for topk
- Then gathers f32 scores for weights
- Then normalizes and scales in f32
- Then performs a **CPU fallback sort** (moe.py L806-813) via `_to_torch_any` + `torch.sort`

### TTNNMoE.forward Data Flow (moe.py L1390-1472)

```
Input x: (1, 1, 1, hidden_size/8=384) per device [sharded on last dim]
    |
    v
all_gather_async -> (1, 1, 1, 3072) [replicated]
    |
    v
typecast bf16->f32 -> linear(x_f32, gate_weight) -> (1, 64) router logits [f32, HiFi4]
    |
    v
TTNNMoERouterDecode.forward -> topk_indices (1,6), topk_weights (1,6)
    |--- sigmoid (f32)
    |--- add bias (f32)
    |--- typecast f32->bf16
    |--- topk(k=6) [bf16]
    |--- gather raw f32 scores
    |--- normalize + scale [f32]
    |--- CPU FALLBACK: _to_torch_any + torch.sort + ttnn.from_torch  <--- BOTTLENECK
    |
    v
TTNNExperts.forward:
    |--- typecast + pad to SPARSITY_BLOCK_SIZE=32
    |--- to_layout TILE->ROW_MAJOR (x and indices)
    |--- reshape
    |--- all_to_all_dispatch
    |--- reshape + to_layout ROW_MAJOR->TILE
    |--- moe_expert_token_remap (sparsity generation)
    |--- 3x sparse_matmul (w1, w3, w2) + silu + mul
    |--- permute + reshape
    |--- to_layout TILE->ROW_MAJOR
    |--- all_to_all_combine
    |--- reshape + to_layout ROW_MAJOR->TILE
    |--- WEIGHT APPLICATION: unsqueeze + repeat(hidden_size,1,1,1) + permute + to_layout + mul + sum  <--- BOTTLENECK
    |--- slice (remove padding)
    |
    v
reduce_scatter_minimal_async -> (1, 1, 1, 384) [sharded]
    |
    v
shared_experts(residual): gate_proj(silu) * up_proj -> down_proj
    |
    v
ttnn.add(routed_output, shared_output) -> squeeze -> output
```

---

## 3. Root Causes of Slowness

### 3.1 CPU Fallback Sort in Router (HIGH IMPACT)

**Location:** `TTNNGlm4MoeRouteTokenToExperts.forward`, moe.py L806-813

```python
topk_idx_t = _to_torch_any(topk_expert_idx).to(torch.int64)
topk_w_t = _to_torch_any(topk_weights).to(torch.float32)
sorted_w, sorted_pos = torch.sort(topk_w_t, dim=1, descending=True)
sorted_idx = torch.gather(topk_idx_t, 1, sorted_pos)
topk_expert_idx = ttnn.from_torch(sorted_idx.to(torch.int32))
topk_weights = ttnn.from_torch(sorted_w.to(torch.bfloat16))
```

This triggers:
- Device-to-host transfer (`_to_torch_any` calls `ttnn.to_torch`)
- CPU sort on tiny tensor (1x6)
- Host-to-device transfer (`ttnn.from_torch`)

Each round-trip forces a full device synchronization. This is executed per MoE layer (19 times per token).

**Note:** `TTNNMoERouterDecode.forward` (moe.py L891-1002) does NOT have this CPU sort -- it returns topk results directly. But `TTNNGlm4MoeRouteTokenToExperts.forward` (moe.py L719-814) does. The `TTNNBailingMoE` uses `TTNNMoERouterDecode` (see moe.py L1506), so the CPU sort at L806-813 is NOT in the active path. However, `TTNNMoERouterDecode.forward` at L891-1002 still has correctness concerns we should verify.

**CORRECTION after re-reading:** `TTNNBailingMoE.from_torch` (L1506) creates `TTNNMoERouterDecode` and stores it as `module.route_tokens_to_experts`. The parent `TTNNMoE.forward` (L1442) calls `self.route_tokens_to_experts(router_logits)`. So the ACTIVE router is `TTNNMoERouterDecode`, NOT `TTNNGlm4MoeRouteTokenToExperts`. `TTNNMoERouterDecode.forward` does NOT have the CPU sort fallback. This means the CPU sort is NOT a current bottleneck.

### 3.2 Expensive Weight Application Pattern (MEDIUM-HIGH IMPACT)

**Location:** `TTNNExperts.forward`, moe.py L1299-1313

```python
topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, repeat_dims=(self.hidden_size, 1, 1, 1))
topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
weighted_output = ttnn.mul(combined_output, topk_experts_weights_tile)
final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
```

This executes 7 separate ops (to_layout, 2x unsqueeze, repeat, permute, to_layout, mul) plus a sum. Each op has host dispatch overhead. The `repeat(hidden_size, 1, 1, 1)` with hidden_size=3072 is particularly expensive -- it replicates a (1,1,1,6) tensor to (3072,1,1,6), then permutes to (6,1,1,3072). This is wasteful for a broadcast multiply.

**Alternative:** Reshape weights to (6, 1, 1, 1) and use broadcast multiply with combined_output (6, 1, 1, 3072). TTNN should broadcast the last dimension automatically. Then sum over dim=0.

### 3.3 Sequential Shared Expert Computation (MEDIUM IMPACT)

**Location:** `TTNNMoE.forward`, moe.py L1469-1470

```python
shared_output = self.shared_experts(residual)
output = ttnn.add(routed_output, shared_output.to_ttnn)
```

The shared expert MLP runs AFTER the entire routed expert pipeline completes (after reduce_scatter). The shared expert computation is independent of the routed experts and depends only on `residual` (the input before all_gather). It could overlap with the routed expert pipeline.

### 3.4 Layout Conversions (MEDIUM IMPACT)

Multiple TILE_LAYOUT <-> ROW_MAJOR_LAYOUT conversions in TTNNExperts.forward:
- L1193: x TILE -> ROW_MAJOR (for all_to_all_dispatch)
- L1197: topk_experts_indices TILE -> ROW_MAJOR
- L1212: post_dispatch ROW_MAJOR -> TILE (for sparse_matmul)
- L1279: expert_output TILE -> ROW_MAJOR (for all_to_all_combine)
- L1296: combined_output ROW_MAJOR -> TILE
- L1299: topk_experts_weights TILE -> ROW_MAJOR
- L1304: topk_experts_weights_rm ROW_MAJOR -> TILE

That is 7 layout conversions per TTNNExperts.forward call, 19 per token.

### 3.5 Unnecessary Typecast in Gate (LOW IMPACT)

**Location:** `TTNNMoE.forward`, moe.py L1419-1422

```python
if x.dtype != ttnn.float32:
    x_f32 = ttnn.typecast(x, ttnn.float32)
```

The all_gather output is bf16. The gate matmul uses HiFi4 with fp32_dest_acc_en=True. This means the matmul accumulates in f32 internally even with bf16 input. The typecast to f32 BEFORE the matmul adds overhead. The matmul could take bf16 input directly and produce f32 output with `dtype=ttnn.float32` output parameter, eliminating the pre-matmul typecast.

### 3.6 Wrapper/Host Overhead (HIGH IMPACT -- addressed separately)

Per the wrapper overhead analysis (PLAN_wrapper_optimizations.md), 50% of decode latency is wrapper/Python overhead. This is addressed by a separate plan (P1-P5). The MoE-specific optimizations in THIS plan are additive to those wrapper optimizations.

### 3.7 No Trace Capture for MoE (MEDIUM-HIGH IMPACT)

The entire MoE forward pass runs without trace capture. Each TTNN op requires a host dispatch. At batch=1 decode, host dispatch overhead likely dominates device compute time. Trace capture would eliminate host dispatch overhead for the steady-state decode path.

---

## 4. Step-by-Step Optimization Plan

### Phase 1: Low-Hanging Fruit (No Architectural Changes)

#### Step 1.1: Simplify Weight Application in TTNNExperts.forward

**File:** `modules/moe.py`, `TTNNExperts.forward`, L1299-1313

**Current:** 7-op sequence: to_layout -> unsqueeze -> unsqueeze -> repeat(3072,1,1,1) -> permute -> to_layout -> mul, then sum.

**Proposed:** Replace with broadcast multiply + sum:

```python
# topk_experts_weights shape: (num_tokens, num_experts_per_tok)
# combined_output shape: (num_experts_per_tok, 1, num_tokens, hidden_size)

# Reshape weights for broadcasting: (num_experts_per_tok, 1, num_tokens, 1)
w = ttnn.reshape(topk_experts_weights, ttnn.Shape((1, 1, topk_experts_weights.shape[0], topk_experts_weights.shape[1])))
w = ttnn.permute(w, (3, 1, 2, 0))  # (num_experts_per_tok, 1, num_tokens, 1)
if w.layout != ttnn.TILE_LAYOUT:
    w = ttnn.to_layout(w, ttnn.TILE_LAYOUT)

# Broadcast multiply: (num_experts_per_tok, 1, num_tokens, hidden_size)
weighted_output = ttnn.mul(combined_output, w)

# Sum over experts dimension
final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
```

This replaces 7 ops with 3 ops (reshape, permute, to_layout) + mul + sum. The key savings come from eliminating `repeat(3072, 1, 1, 1)` which copies 3072x more data than needed.

**Expected Savings:** 15-40us per layer x 19 layers = 0.3-0.8ms/token
**Risk:** Low. Pure data format change; same mathematical result.
**Test:** Compare output tensor values before/after with assert_allclose.

#### Step 1.2: Eliminate Pre-Gate Typecast to f32

**File:** `modules/moe.py`, `TTNNMoE.forward`, L1419-1435

**Current:**
```python
if x.dtype != ttnn.float32:
    x_f32 = ttnn.typecast(x, ttnn.float32)
router_logits = ttnn.linear(x_f32, self._gate_weight_tt, ...)
```

**Proposed:** Keep x as bf16, let the linear op handle f32 accumulation internally:
```python
router_logits = ttnn.linear(
    x,  # bf16 input
    self._gate_weight_tt,  # bf16 weight
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.float32,  # f32 output
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)
```

**Expected Savings:** Small (5-15us per layer, ~0.1-0.3ms/token total). The typecast itself is cheap but every eliminated op reduces host dispatch count.
**Risk:** Low. The linear op with fp32_dest_acc_en already accumulates in f32. Output dtype parameter controls output format. Verify `ttnn.linear` supports `dtype` output parameter; if not, keep the current approach.
**Test:** Compare router_logits values before/after.

#### Step 1.3: Reduce Layout Conversions in TTNNExperts

**File:** `modules/moe.py`, `TTNNExperts.forward`

Some layout conversions may be avoidable if we keep data in ROW_MAJOR throughout the dispatch/combine pipeline and only convert to TILE for sparse_matmul.

**Current flow:** TILE -> ROW_MAJOR (dispatch) -> TILE (sparse_matmul) -> ROW_MAJOR (combine) -> TILE (weight app)

The conversions at L1296 (combined_output to TILE) and L1299/1304 (weights to ROW_MAJOR then TILE) can potentially be eliminated if weight application is done in ROW_MAJOR using elementwise ops.

**Expected Savings:** 2-6us per conversion x 2-3 eliminated x 19 layers = 0.1-0.3ms/token
**Risk:** Low. Layout is a data format concern, not a correctness concern.
**Test:** Same output values.

### Phase 2: Shared Expert Overlap (Moderate Complexity)

#### Step 2.1: Move Shared Expert Computation Before Reduce-Scatter

**File:** `modules/moe.py`, `TTNNMoE.forward`, L1404-1471

**Current order:**
```
1. all_gather(x)
2. gate + router
3. experts.forward(x, indices, weights)  -- includes dispatch, compute, combine
4. reduce_scatter(routed_output)
5. shared_experts(residual)              -- SEQUENTIAL, after reduce_scatter
6. add(routed, shared)
```

**Proposed order:**
```
1. all_gather(x)
2. gate + router
3. shared_experts(residual)              -- START EARLY (uses residual = original x)
4. experts.forward(x, indices, weights)
5. reduce_scatter(routed_output)
6. add(routed, shared)
```

The shared expert computation depends on `residual` (the input before all_gather), which is available immediately. By moving it before or alongside the routed expert pipeline, the shared expert matmuls can overlap with routed expert CCL operations (all_to_all_dispatch, all_to_all_combine) through TTNN's async dispatch.

**Key insight:** The shared experts operate on `residual` which has shape `(1, 1, 1, 384)` (already sharded). There is no dependency on the routed expert output. The only constraint is that `shared_output` and `routed_output` must both be ready before the final `ttnn.add`.

**Caveat:** Async overlap only works if the shared expert ops run on different cores or use different NOC channels than the CCL ops. On T3K, this is plausible since CCL ops use the NOC while matmuls use Tensix compute cores.

**Expected Savings:** 0.5-1ms/token if shared expert MLP (~3 matmuls) overlaps with routed expert all_to_all + sparse_matmul.
**Risk:** Medium. Async overlap is not guaranteed -- depends on TTNN scheduler. If no overlap occurs, there is zero savings but also zero regression.
**Test:** Same output values. Timing comparison with/without reordering.

### Phase 3: Router Optimization (Moderate Complexity)

#### Step 3.1: Eliminate Router f32 Typecast Chain

**File:** `modules/moe.py`, `TTNNMoERouterDecode.forward`, L891-1002

**Current (n_group <= topk_group path, L927-932):**
```python
scores_bf16 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
ttnn.deallocate(scores_with_bias_f32)
_, topk_expert_idx = ttnn.topk(scores_bf16, k=top_k, dim=3, largest=True, sorted=True)
ttnn.deallocate(scores_bf16)
```

Then later:
```python
topk_weights = ttnn.gather(scores_f32, dim=3, index=topk_expert_idx)  # gather from f32 scores
denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
topk_weights = ttnn.div(topk_weights, denom)
```

The f32 -> bf16 typecast before topk is necessary because ttnn.topk only supports bf16. However, the subsequent gather, sum, and div all happen in f32. The chain is:
- sigmoid(logits) in f32
- add bias in f32
- typecast to bf16 for topk
- topk in bf16 -> indices
- gather from f32 scores using indices
- normalize in f32
- scale in f32

This is correct and cannot easily be simplified further without f32 topk support. However, for Ling-mini-2.0 with only 64 experts (not 256), the precision risk from bf16 topk is lower. We could consider eliminating the f32 path entirely and doing everything in bf16, but this is a quality/speed tradeoff that needs validation.

**Alternative optimization:** The gate matmul (Step 1.2) already produces f32 logits. If we skip the sigmoid+bias in f32 and instead do sigmoid in bf16, we save the typecast. But sigmoid precision in bf16 near 0.5 is poor. Keep f32 sigmoid.

**Expected Savings:** Minimal for this step alone. The real savings come from trace capture (Phase 4).
**Risk:** N/A if no change made.

#### Step 3.2: Consider Removing Router Sort (if present in active path)

After re-analysis, `TTNNMoERouterDecode` (the active router for TTNNBailingMoE) does NOT have the CPU sort fallback. The CPU sort is only in `TTNNGlm4MoeRouteTokenToExperts` (L806-813), which is used by `TTNNGlm4MoeMoE` (the GLM-4 path), not by `TTNNBailingMoE`.

**Action:** No change needed for Ling-mini-2.0. The `ttnn.topk` with `sorted=True` already returns sorted indices.

### Phase 4: Trace Capture (High Impact, Higher Complexity)

#### Step 4.1: Enable Trace Capture for TTNNBailingMoE

**File:** `modules/moe.py`, `TTNNBailingMoE` class definition

**Current:** `TTNNBailingMoE` is not decorated with `@trace_enabled`. The `TTNNMoE.forward` method has a `@disable_trace` decorator on `TTNNGlm4MoeExpertLayers.forward` (L532) but TTNNMoE.forward and TTNNExperts.forward do not.

Actually, looking more carefully: TTNNMoE.forward at L1389 is decorated with `@run_on_devices(DeviceArch.T3K)` but NOT `@disable_trace`. And TTNNExperts.forward at L1137 also has `@run_on_devices(DeviceArch.T3K)` but not `@disable_trace`.

**Proposed:** Add `@trace_enabled` decorator to `TTNNBailingMoE`:

```python
from models.experimental.tt_symbiote.core.run_config import trace_enabled

@trace_enabled
class TTNNBailingMoE(TTNNMoE):
    ...
```

Then in the test, use `TracedRun` mode:
```python
from models.experimental.tt_symbiote.core.run_config import set_run_mode, TracedRun
set_run_mode("TRACED")
TracedRun.configure(device=mesh_device)
```

**Challenge:** MoE forward has dynamic control flow:
- `if topk_experts_indices.dtype != ttnn.uint16` (L1161) -- data-dependent but constant after first call
- `if num_tokens % SPARSITY_BLOCK_SIZE != 0` (L1172) -- shape-dependent, constant for fixed batch size
- `if pad_amount > 0` (L1316) -- same as above
- `if T == 1` (L910, L986 in router) -- constant for decode

For decode with batch=1, all these branches are deterministic after the first call. Trace capture should work if the first call establishes the branch directions and subsequent calls follow the same path.

**Key risk:** The trace captures a fixed sequence of TTNN ops. If any branch changes between trace capture and replay (e.g., different padding due to different batch size), the trace will produce incorrect results. For fixed batch=1 decode, this should not occur.

**Additional risk:** `TTNNExperts.forward` creates `all_to_all_dispatch_metadata` dynamically. If this metadata varies between calls (due to different expert selections), trace replay may fail. Need to verify that `all_to_all_dispatch` and `all_to_all_combine` are trace-compatible.

**Expected Savings:** 2-3ms/token. Host dispatch overhead for the ~50+ TTNN ops in TTNNMoE.forward is eliminated.
**Risk:** High. Trace capture for MoE with dynamic routing metadata is complex. May require changes to all_to_all_dispatch/combine to be trace-compatible.
**Test:** Run 128 tokens, verify output matches non-traced output exactly.

#### Step 4.2: Trace Capture for Full Decoder Layer (requires PLAN_wrapper_optimizations P1)

If `TTNNBailingMoEDecoderLayer` is implemented (from wrapper optimizations P1), the entire decoder layer (attention + MoE + residuals + norms) can be trace-captured as a single unit. This eliminates host dispatch overhead for ALL ops in the layer, not just MoE.

**Depends on:** PLAN_wrapper_optimizations P1 (TTNNBailingMoEDecoderLayer).

### Phase 5: Expert Compute Tuning (Low-Medium Impact)

#### Step 5.1: Sweep sparse_matmul Program Config

**File:** `modules/moe.py`, `TTNNExperts.move_weights_to_device_impl`, L1118-1129

**Current config:**
```python
in0_block_w = min(4, hidden_tiles)  # hidden_tiles = 3072/32 = 96 -> in0_block_w = 4
per_core_M = 1
```

For Ling-mini-2.0:
- Gate/up projections: (32, 3072) x (3072, 1024) -- hidden_tiles=96, intermediate_tiles=32
- Down projection: (32, 1024) x (1024, 3072) -- hidden_tiles=32, intermediate_tiles=96

**Sweep grid:**
```
in0_block_w: {2, 4, 8, 16}
per_core_M: {1, 2}
```

**Expected Savings:** 10-30% on sparse_matmul time. At 10-30us per matmul x 3 matmuls x 19 layers, this is 0.1-0.5ms/token.
**Risk:** Low. Program config affects performance, not correctness.
**Test:** Benchmark each config; select lowest latency.

#### Step 5.2: Evaluate LoFi Math Fidelity for Expert Matmuls

**File:** `modules/moe.py`, L1130-1135

**Current:** `math_fidelity=ttnn.MathFidelity.HiFi2`

**Test:** Run with LoFi, compare output quality:
- Cosine similarity >= 0.999 vs HiFi2 reference
- Max absolute error <= 0.5% of output norm
- Generated text quality (subjective)

**Expected Savings:** 10-20% on sparse_matmul compute time.
**Risk:** Medium. LoFi may degrade quality for some expert weight distributions. Must validate per-model.

### Phase 6: CCL Parameter Tuning (Low Impact for MoE-specific)

#### Step 6.1: Sweep reduce_scatter Parameters

**File:** `modules/moe.py`, `TTNNMoE.forward`, L1454-1466

**Current:** `chunks_per_sync=10, num_workers_per_link=2`

**Sweep:** Same as recommended in the guide:
```
chunks_per_sync: {1, 5, 10, 20}
num_workers_per_link: {1, 2}
```

**Expected Savings:** 5-15% on reduce_scatter latency. Small absolute impact since reduce_scatter operates on small tensors at batch=1.
**Risk:** Low.

---

## 5. Implementation Order (by expected impact)

| Priority | Step | Expected Savings | Complexity | Risk |
|----------|------|-----------------|------------|------|
| **1** | Step 1.1: Simplify weight application | 0.3-0.8ms | Low | Low |
| **2** | Step 2.1: Shared expert overlap | 0.5-1.0ms | Medium | Medium |
| **3** | Step 4.1: Trace capture for MoE | 2-3ms | High | High |
| **4** | Step 1.2: Eliminate pre-gate typecast | 0.1-0.3ms | Low | Low |
| **5** | Step 1.3: Reduce layout conversions | 0.1-0.3ms | Low | Low |
| **6** | Step 5.1: Sweep sparse_matmul config | 0.1-0.5ms | Low | Low |
| **7** | Step 5.2: LoFi math fidelity | 0.05-0.2ms | Low | Medium |
| **8** | Step 6.1: CCL parameter sweep | 0.05-0.1ms | Low | Low |
| **Total** | | **~3.2-6.2ms/token** | | |

**Note:** These savings are per-token and are in addition to the ~440ms wrapper overhead savings from PLAN_wrapper_optimizations.md. The MoE-specific savings compound across all 19 MoE layers.

---

## 6. Success Criteria

1. **Correctness:** Test generates coherent, non-empty English text (same test passes: `pytest test_ling_mini_2_0.py --timeout=0`)
2. **Performance:** Measurable reduction in per-token decode time, verified via DispatchManager timing CSV
3. **Regression:** No regression in prefill path
4. **Quality:** For Steps 5.2 (LoFi), verify cosine similarity >= 0.999 vs baseline

---

## 7. Risk Assessment Summary

| Risk | Steps Affected | Likelihood | Impact | Mitigation |
|------|---------------|-----------|--------|------------|
| Trace capture fails due to dynamic routing metadata | 4.1 | Medium | High | Verify all_to_all ops are trace-compatible; fall back to non-traced |
| Async overlap does not materialize | 2.1 | Medium | None | No regression; just no savings |
| Broadcast mul shape mismatch | 1.1 | Low | Medium | Test with reference comparison |
| ttnn.linear dtype parameter unsupported | 1.2 | Low | None | Keep current typecast approach |
| LoFi degrades output quality | 5.2 | Medium | Medium | Validate per-model; keep HiFi2 as fallback |

---

## 8. Dependencies

- **PLAN_wrapper_optimizations.md** (P1-P5): Should be applied first or in parallel. The wrapper overhead savings (440ms) dwarf the MoE-specific savings (3-6ms). However, the MoE optimizations reduce the TTNN op count, which makes trace capture more effective.
- **TTNNBailingMoEDecoderLayer** (from PLAN_wrapper_optimizations P1): Enables Step 4.2 (full decoder layer trace).
- **Profiling data:** Steps 5.1, 5.2, and 6.1 require per-op timing data. Collect baseline timing first using the DispatchManager CSV output.

---

## 9. How to Collect Baseline Measurements

```bash
cd /home/ttuser/salnahari
source tt_bashrc
cd tt-metal
MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0 -xvs
```

The test automatically saves timing to `ling_mini_2_0_paged_attention_timing_stats.csv`. Analyze the CSV to identify:
1. Total MoE forward time per layer
2. Per-op breakdown within TTNNExperts.forward
3. Router forward time
4. Shared expert time
5. CCL op times (all_gather, reduce_scatter, all_to_all_dispatch, all_to_all_combine)

---

## 10. Quick Reference: Key Source Locations

| Component | File | Lines |
|-----------|------|-------|
| TTNNBailingMoE class | `modules/moe.py` | L1475-1628 |
| TTNNMoE.forward | `modules/moe.py` | L1389-1472 |
| TTNNExperts.forward | `modules/moe.py` | L1137-1321 |
| TTNNMoERouterDecode.forward | `modules/moe.py` | L891-1002 |
| Weight application (target for Step 1.1) | `modules/moe.py` | L1299-1313 |
| Pre-gate typecast (target for Step 1.2) | `modules/moe.py` | L1419-1435 |
| Shared experts call (target for Step 2.1) | `modules/moe.py` | L1469-1470 |
| sparse_matmul config (target for Step 5.1) | `modules/moe.py` | L1118-1135 |
| reduce_scatter params (target for Step 6.1) | `modules/moe.py` | L1454-1466 |
| Test file | `tests/test_ling_mini_2_0.py` | Full file |
