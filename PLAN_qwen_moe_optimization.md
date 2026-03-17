# Qwen3.5-35B MoE Optimization Plan

## Current Performance Baseline

| Component | Decode (1 tok) | Prefill (32 tok) | Prefill (128 tok) |
|-----------|---------------|------------------|-------------------|
| **Full MoE** | 30.15 ms | 31.68 ms | 34.47 ms |
| Experts | 6.47 ms | 6.63 ms | 7.13 ms |
| Router | 8.31 ms | 7.97 ms | 7.90 ms |

**The 16ms Gap** = Full MoE (~30ms) - Experts (~6.5ms) - Router (~8ms) = ~16ms unaccounted

---

## 16ms Gap Analysis

### Code Path Analysis

Based on analysis of `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`:

#### What "Experts" Baseline Measures (TTNNQwenExperts.forward, lines 313-581):
- Padding operations
- Layout conversion: input to ROW_MAJOR
- **all_to_all_dispatch** operation
- Layout conversion: to TILE_LAYOUT
- Sparsity tensor generation (moe_expert_token_remap)
- 3x sparse_matmul (w1, w3, w2) + SiLU + multiply
- Layout conversion: to ROW_MAJOR
- **all_to_all_combine** operation
- Weight application and expert sum

#### What "Router" Baseline Measures (TTNNQwenMoERouterDecode):
- Router forward computation (softmax, topk, etc.)

#### Operations Contributing to 16ms Gap (TTNNQwen3MoE.forward, lines 828-984):

These operations are **OUTSIDE** the Experts and Router measurements:

| Operation | Code Location | Estimated Impact |
|-----------|---------------|------------------|
| **all_gather_async** | lines 846-853 | **HIGH** - CCL operation across 8 devices |
| Gate linear (float32) | lines 862-872 | MEDIUM - [batch, hidden] x [hidden, 256_experts] |
| Typecasts bf16<->f32 | lines 858-859, 876-877 | LOW - memory bound |
| **reduce_scatter_minimal_async** | lines 944-956 | **HIGH** - CCL operation across 8 devices |
| **shared_experts MLP** | line 959 | **HIGH** - Full MLP forward pass |
| shared_expert_gate | lines 967-978 | LOW - linear + sigmoid + mul |
| Final add | lines 977-980 | LOW - element-wise |

### Key Findings

1. **CCL Operations (all_gather + reduce_scatter)**: These are collective communication operations across 8 devices. They bookend the MoE computation and likely contribute significantly to the 16ms gap.

2. **Shared Experts MLP**: This is a FULL MLP forward pass that runs in PARALLEL with the routing. It includes its own gate/up/down projections, which could be ~5-8ms.

3. **Gate Linear in float32**: The router logits computation uses float32 accumulation (fp32_dest_acc_en=True) for accuracy, but requires typecast overhead.

4. **Layout Conversions**: Multiple ROW_MAJOR <-> TILE_LAYOUT conversions occur throughout the forward pass.

### Estimated Gap Breakdown

| Component | Estimated Time | Notes |
|-----------|---------------|-------|
| all_gather_async | 3-5ms | Gather hidden dim across 8 devices |
| Gate linear (router logits) | 2-3ms | Float32 matmul for accuracy |
| Typecasts | 0.5-1ms | bf16 <-> f32 conversions |
| reduce_scatter_minimal_async | 3-5ms | Scatter+reduce across 8 devices |
| shared_experts MLP | 5-8ms | Full MLP with gate/up/down projections |
| shared_expert_gate | 0.5-1ms | Small linear + sigmoid |
| **Total Estimated** | **14-23ms** | Consistent with 16ms gap |

---

## Rejected Optimizations (Do Not Pursue)

The following approaches have been evaluated and rejected:

1. **CCL Link Count Changes**: Already tuned; changing num_links is hardware-dependent
2. **Async Overlap**: Already using async CCL operations
3. **L1 Sharding for Activations**: Memory constraints on Wormhole
4. **Prefill Chunking**: Does not address decode bottleneck

---

## Optimization Proposals

### Priority 1: Shared Expert Fusion

**Target**: shared_experts MLP (estimated 5-8ms)

**Current Implementation** (line 959):
```python
shared_output = self.shared_experts(residual)
```

**Problem**: Shared expert runs as a completely separate MLP, duplicating computation that could be fused or optimized differently.

**Proposed Optimization**:
1. **Option A - Fused Gate/Up Projection**: If using the same compute pattern as routed experts, fuse shared_expert gate/up into a single matmul with concatenated weights.
2. **Option B - Parallel Execution**: Ensure shared_expert computation overlaps with all_gather_async (which currently blocks).

**Implementation Steps**:
1. Profile shared_experts independently to get exact timing
2. Evaluate if shared expert weights can be fused with routed expert weights
3. If separate, ensure CCL ops and shared_expert run in parallel using async ops

---

### Priority 2: Router Logits Computation Optimization

**Target**: Gate linear + typecasts (estimated 2.5-4ms)

**Current Implementation** (lines 858-877):
```python
x_f32 = ttnn.typecast(x, ttnn.float32)  # bf16 -> f32
router_logits_f32 = ttnn.linear(x_f32, self._gate_weight_tt, ...)  # f32 matmul
router_logits = ttnn.typecast(router_logits_f32, ttnn.bfloat16)  # f32 -> bf16
```

**Problem**: Double typecast overhead for float32 accumulation.

**Proposed Optimization**:
Use bfloat16 input with fp32 destination accumulation (already configured with `fp32_dest_acc_en=True`), but avoid the input typecast:

```python
# Remove x_f32 = ttnn.typecast(x, ttnn.float32)
# Use bf16 input directly with fp32 accumulation
router_logits_f32 = ttnn.linear(x, self._gate_weight_tt, ...)  # bf16 in, f32 accum
```

**Implementation Steps**:
1. Test if bf16 input + fp32 accumulation produces equivalent accuracy
2. Remove input typecast if accuracy is preserved
3. Measure latency reduction

---

### Priority 3: Layout Conversion Reduction

**Target**: Multiple ROW_MAJOR <-> TILE_LAYOUT conversions

**Current Issue** (TTNNQwenExperts.forward):
- Line 389: to_layout ROW_MAJOR (for all_to_all_dispatch)
- Line 410: to_layout TILE_LAYOUT (for sparse_matmul)
- Line 517: to_layout ROW_MAJOR (for all_to_all_combine)
- Lines 543, 551, 557: Additional layout conversions for weight application

**Proposed Optimization**:
Investigate if all_to_all_dispatch/combine can work with TILE_LAYOUT directly, eliminating 2 layout conversions per forward pass.

**Implementation Steps**:
1. Check TTNN all_to_all ops layout requirements
2. If TILE_LAYOUT supported, modify dispatch/combine to use it
3. Measure reduction in conversion overhead

---

### Priority 4: Weight Quantization for Shared Expert

**Target**: shared_experts MLP weights

**Current Implementation**: bfloat16 weights for shared expert

**Proposed Optimization**:
Apply bfloat8_b quantization to shared_expert weights (similar to DeepSeek-V3's approach for routed experts).

**Rationale**: Shared expert computation contributes significantly to the 16ms gap. Quantizing weights reduces memory bandwidth and may improve throughput.

**Implementation Steps**:
1. Implement weight quantization for shared_expert (gate_up_proj, down_proj)
2. Run accuracy tests to validate acceptable loss
3. Measure performance improvement

---

### Priority 5: Persistent Output Buffers for reduce_scatter

**Target**: reduce_scatter_minimal_async (estimated 3-5ms)

**Current Implementation** (lines 944-956):
```python
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    persistent_output_buffers=None,  # <-- No buffer reuse
    ...
)
```

**Problem**: Allocating new output buffers per forward pass adds overhead.

**Proposed Optimization**:
Pre-allocate persistent output buffers and reuse across forward passes:

```python
# In __init__ or preprocess:
self._rs_output_buffer = ttnn.allocate_tensor(...)

# In forward:
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    persistent_output_buffers=self._rs_output_buffer,
    ...
)
```

**Implementation Steps**:
1. Determine output buffer shape requirements
2. Pre-allocate buffers in weight preprocessing
3. Pass buffers to reduce_scatter
4. Measure allocation overhead reduction

---

## Accuracy Test Suite Requirements

Any optimization must pass these tests before deployment:

### Test Categories

1. **Numerical Accuracy Tests**
   - Compare TTNN output to PyTorch reference (existing `test_qwen_moe_accuracy.py`)
   - Maximum absolute error < 0.01 for decode
   - Maximum relative error < 1% for prefill

2. **End-to-End Generation Tests**
   - Run full text generation and compare outputs
   - Ensure coherent, sensible text generation
   - Compare perplexity metrics

3. **Regression Tests**
   - Performance must not regress on any path
   - Memory usage must stay within bounds

### Test Commands

```bash
# Accuracy tests
pytest tests/test_qwen_moe_accuracy.py -v

# Generation tests
python models/experimental/tt_symbiote/tests/test_qwen_generation.py

# Performance benchmarks
python models/experimental/tt_symbiote/benchmarks/benchmark_qwen_moe.py
```

---

## Implementation Roadmap

| Phase | Optimization | Estimated Savings | Risk |
|-------|-------------|-------------------|------|
| 1 | Router logits typecast removal | 0.5-1ms | Low |
| 2 | Persistent reduce_scatter buffers | 0.5-1ms | Low |
| 3 | Layout conversion reduction | 1-2ms | Medium |
| 4 | Shared expert optimization | 2-4ms | Medium |
| 5 | Weight quantization (shared expert) | 1-2ms | High (accuracy) |

**Total Estimated Savings**: 5-10ms (33-66% of gap)

---

## Next Steps

1. **Profile CCL operations**: Use Tracy or op-level profiling to get exact timing for all_gather and reduce_scatter
2. **Profile shared_experts**: Isolate shared expert MLP timing
3. **Prototype typecast removal**: Low-risk change to validate accuracy
4. **Investigate persistent buffers**: Check reduce_scatter API for buffer reuse

---

## Appendix: Code References

- `TTNNQwen3MoE.forward()`: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py` lines 828-984
- `TTNNQwenExperts.forward()`: Same file, lines 313-581
- Router decode: `TTNNQwenMoERouterDecode` in same file
