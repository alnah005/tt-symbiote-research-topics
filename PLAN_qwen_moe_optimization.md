# TTNNQwen3MoE Optimization Plan for T3K (Revised)

**Target:** Qwen3.5-35B-A3B MoE module running on T3K (1x8 mesh)
**Goal:** Improve decode and prefill throughput without losing accuracy
**Revision Date:** 2026-03-17
**Revision Note:** This plan removes previously rejected optimizations and proposes NEW approaches

---

## REJECTED OPTIMIZATIONS (DO NOT IMPLEMENT)

The following optimizations were rejected and should NOT be pursued:
- Increasing num_links for CCL ops (all_gather, reduce_scatter)
- Adding num_links to all_to_all_dispatch/combine
- Async overlap between shared/routed experts
- L1 sharded memory configs for activations
- Prefill chunking
- Fused weight broadcasting (repeat+permute)
- Single-pass router optimization

---

## Table of Contents
1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [NEW Proposed Optimizations](#2-new-proposed-optimizations)
3. [Step-by-Step Implementation Guide](#3-step-by-step-implementation-guide)
4. [Accuracy Test Suite Design](#4-accuracy-test-suite-design)
5. [Success Criteria](#5-success-criteria)

---

## 1. Current Implementation Analysis

### 1.1 Architecture Overview

**Qwen3.5-35B-A3B MoE Configuration:**
- 256 routed experts total
- Top-8 routing per token
- 32 experts per device (256 / 8 devices)
- Hidden size: 4096 (typical)
- MoE intermediate size: 18944 (typical for Qwen3)
- Shared expert with optional gating

### 1.2 Current Implementation Gaps vs DeepSeek-V3

| Feature | Qwen MoE (Current) | DeepSeek-V3 (Reference) | Opportunity |
|---------|-------------------|------------------------|-------------|
| Expert weight dtype | bfloat16 | bfloat4_b/bfloat8_b | 2-4x memory BW reduction |
| Compute kernel config | None specified | COMPUTE_KERNEL_CONFIG_LOFI | Faster matmuls |
| Weight memory layout | DRAM interleaved | DRAM sharded | Better prefetch |
| Fused SiLU activation | Separate ops | mul_experts with input_tensor_a_activations | Kernel fusion |
| Matmul program config | Auto-generated | Explicit 1D multicast config | Better core utilization |

---

## 2. NEW Proposed Optimizations

### Ranked by Impact (High to Low)

| Rank | Optimization | Expected Speedup | Risk | Effort |
|------|-------------|------------------|------|--------|
| 1 | Expert weight quantization (bfloat8_b) | 20-40% expert compute | Medium (accuracy) | Medium |
| 2 | Compute kernel config (LoFi + packer_l1_acc) | 10-20% matmul ops | Low | Low |
| 3 | Fused SiLU in mul operation | 5-10% expert compute | Low | Low |
| 4 | DRAM-sharded weight memory config | 10-15% weight loading | Low | Medium |
| 5 | Explicit matmul program configs | 5-15% matmul ops | Low | Medium |
| 6 | Router computation optimization | 5-10% routing | Low | Low |

---

### 2.1 Expert Weight Quantization (bfloat8_b)

**Current State (qwen_moe.py preprocess_weights):**
```python
# Weights stored as bfloat16
self.tt_w1_proj = ttnn.from_torch(
    gate_proj,
    dtype=ttnn.bfloat16,  # Full precision
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    device=device,
)
```

**Reference (DeepSeek-V3 experts.py lines 74-77):**
```python
# Expert weights use quantization
dtype=ttnn.bfloat8_b if hf_name == "up_proj" else ttnn.bfloat4_b,
memory_config=ttnn.DRAM_MEMORY_CONFIG,
```

**Proposed Change:**
```python
# Use bfloat8_b for experts (more conservative than bfloat4_b)
self.tt_w1_proj = ttnn.from_torch(
    gate_proj,
    dtype=ttnn.bfloat8_b,  # 8-bit quantization
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    device=device,
)
```

**Why This Helps:**
- Reduces weight memory bandwidth by 2x (bfloat8_b vs bfloat16)
- Expert weights are naturally tolerant to quantization due to averaging across top-8
- DeepSeek-V3 demonstrates this works in production

**Priority: HIGH** - Significant bandwidth reduction with acceptable accuracy trade-off.

---

### 2.2 Compute Kernel Configuration

**Current State:**
The sparse_matmul and matmul operations in TTNNQwenExperts don't specify compute_kernel_config.

**Reference (DeepSeek-V3 config_helpers.py lines 30-35):**
```python
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,  # Key optimization
)
```

**Proposed Change (qwen_moe.py TTNNQwenExperts):**
```python
# Add compute kernel config for expert matmuls
self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,  # Fastest math mode
    math_approx_mode=False,                 # Maintain accuracy
    fp32_dest_acc_en=False,                 # Faster with bf16 accumulation
    packer_l1_acc=True,                     # Enable L1 accumulation
)

# Apply to sparse_matmul calls
w1_out = ttnn.sparse_matmul(
    post_dispatch,
    self.tt_w1_proj,
    sparsity,
    program_config=self._gate_up_program_config,
    compute_kernel_config=self._expert_compute_cfg,  # ADD THIS
    memory_config=decode_memory_config,
)
```

**Why This Helps:**
- LoFi math fidelity is fastest (fewer cycles per multiply)
- packer_l1_acc reduces DRAM round-trips for partial sums
- DeepSeek-V3 uses this configuration successfully

**Priority: HIGH** - Low risk, immediate speedup.

---

### 2.3 Fused SiLU Activation

**Current State (qwen_moe.py forward lines 457-459):**
```python
w1_activated = ttnn.silu(w1_out, memory_config=decode_memory_config)
# Then multiply separately
intermediate = ttnn.mul(w1_activated, w3_out, memory_config=decode_memory_config)
```

**Reference (DeepSeek-V3 experts.py lines 137-140):**
```python
"mul_experts": MulConfig(
    memory_config=output_memory_config,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],  # Fused activation
),
```

**Proposed Change:**
```python
# Fuse SiLU into the multiply operation
intermediate = ttnn.mul(
    w1_out,
    w3_out,
    memory_config=decode_memory_config,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],  # Fused activation
)
# Remove separate silu call
```

**Why This Helps:**
- Eliminates one kernel launch
- Reduces memory traffic (no intermediate w1_activated tensor)
- Fused operations execute faster than sequential

**Priority: MEDIUM** - Simple change with measurable impact.

---

### 2.4 DRAM-Sharded Weight Memory Configuration

**Current State:**
```python
memory_config=ttnn.DRAM_MEMORY_CONFIG  # Basic interleaved
```

**Reference (DeepSeek-V3 config_helpers.py dram_sharded_weight_config):**
```python
def dram_sharded_weight_config(k, n, dram_grid_size):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = dram_grid_size.x  # WH has 12 dram cores
    shard_spec = ttnn.ShardSpec(
        dram_weight_grid,
        (k, ttnn.core.roundup(ttnn.core.divup(n, dram_cores), ttnn.TILE_SIZE)),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
```

**Proposed Change:**
```python
def _create_dram_sharded_weight_config(self, k: int, n: int):
    """Create DRAM-sharded config for expert weights."""
    dram_grid_size = self.device.dram_grid_size()  # 12 for Wormhole
    dram_weight_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
        )
    })
    shard_spec = ttnn.ShardSpec(
        dram_weight_grid,
        (k, ttnn.core.roundup(ttnn.core.divup(n, dram_grid_size.x), ttnn.TILE_SIZE)),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
```

**Why This Helps:**
- DRAM-sharded layout enables parallel reads from multiple DRAM banks
- Improves weight prefetch patterns for matmuls
- Better utilization of Wormhole's 12 DRAM channels

**Priority: MEDIUM** - Requires careful dimension alignment but provides sustained benefit.

---

### 2.5 Explicit Matmul Program Configurations

**Current State:**
The sparse_matmul operations use _make_sparse_matmul_program_config which auto-generates configs.

**Proposed Enhancement:**
Create optimized 1D multicast program configs based on actual tensor shapes:

```python
def _create_expert_matmul_config(self, m: int, k: int, n: int, is_decode: bool):
    """Create optimized matmul config for expert projections.

    Args:
        m: Batch dimension (num_tokens * num_experts_selected)
        k: Input features
        n: Output features
        is_decode: True for decode mode (small m), False for prefill
    """
    grid = ttnn.CoreGrid(x=8, y=8) if not is_decode else ttnn.CoreGrid(x=8, y=4)

    per_core_m = max(1, m // (ttnn.TILE_SIZE * grid.num_cores))
    per_core_n = ttnn.core.divup(n, ttnn.TILE_SIZE * grid.num_cores)
    in0_block_w = find_largest_divisor(k // ttnn.TILE_SIZE)

    # Find optimal subblock dimensions
    out_subblock_w = max([i for i in range(1, 5) if per_core_n % i == 0])
    out_subblock_h = max([i for i in range(1, 5) if per_core_m % i == 0 and i * out_subblock_w <= 8])

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
```

**Why This Helps:**
- Explicit configs avoid auto-tuning overhead
- Better subblock dimensions improve register utilization
- Mode-specific grids (decode vs prefill) optimize core usage

**Priority: LOW-MEDIUM** - Requires profiling to validate improvements.

---

### 2.6 Router Computation Optimization

**Current State:**
The router uses float32 intermediate for precision:
```python
router_logits_f32 = ttnn.linear(
    x_for_gate,
    self.router_weights,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # High precision
        fp32_dest_acc_en=True,
    ),
)
```

**Proposed Change:**
For decode mode (single token), use faster compute config:
```python
if is_decode_mode:
    router_compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,  # Good accuracy, faster
        math_approx_mode=False,
        fp32_dest_acc_en=False,  # BF16 accumulation is sufficient for decode
        packer_l1_acc=True,
    )
else:
    router_compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Keep high precision for prefill
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
```

**Why This Helps:**
- Decode mode processes single tokens where precision is less critical
- HiFi2 is significantly faster than HiFi4
- Router selection is robust to small numerical differences

**Priority: LOW** - Incremental improvement.

---

## 3. Step-by-Step Implementation Guide

### Phase 1: Compute Kernel Configuration (Day 1)

**Step 1.1: Add Compute Kernel Config Constants**

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`

**Add near top of file (after imports):**
```python
# Compute kernel configurations for expert matmuls
EXPERT_COMPUTE_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

EXPERT_COMPUTE_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

**Step 1.2: Apply to sparse_matmul Calls**

**Location:** `TTNNQwenExperts.forward()` around lines 432-476

**Add `compute_kernel_config=EXPERT_COMPUTE_CONFIG_LOFI` to all sparse_matmul calls:**
```python
w1_out = ttnn.sparse_matmul(
    post_dispatch,
    self.tt_w1_proj,
    sparsity,
    program_config=self._gate_up_program_config,
    compute_kernel_config=EXPERT_COMPUTE_CONFIG_LOFI,  # ADD
    memory_config=decode_memory_config,
)

w3_out = ttnn.sparse_matmul(
    post_dispatch,
    self.tt_w3_proj,
    sparsity,
    program_config=self._gate_up_program_config,
    compute_kernel_config=EXPERT_COMPUTE_CONFIG_LOFI,  # ADD
    memory_config=decode_memory_config,
)

expert_output = ttnn.sparse_matmul(
    intermediate,
    self.tt_w2_proj,
    sparsity,
    program_config=self._down_program_config,
    compute_kernel_config=EXPERT_COMPUTE_CONFIG_LOFI,  # ADD
    memory_config=decode_memory_config,
)
```

### Phase 2: Fused SiLU Activation (Day 1)

**Step 2.1: Replace Sequential SiLU + Mul with Fused Op**

**Location:** `TTNNQwenExperts.forward()` around lines 457-459

**Before:**
```python
w1_activated = ttnn.silu(w1_out, memory_config=decode_memory_config)
intermediate = ttnn.mul(w1_activated, w3_out, memory_config=decode_memory_config)
ttnn.deallocate(w1_activated)
```

**After:**
```python
# Fused SiLU + multiply
intermediate = ttnn.mul(
    w1_out,
    w3_out,
    memory_config=decode_memory_config,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
)
```

**Note:** Also update the non-sparse (batched matmul) path around lines 502-505.

### Phase 3: Weight Quantization (Days 2-3)

**Step 3.1: Add Quantization Option to preprocess_weights**

**Location:** `TTNNQwenExperts.preprocess_weights()`

**Add parameter and implementation:**
```python
# Add environment variable check
TT_QWEN_EXPERT_WEIGHT_DTYPE = os.environ.get("TT_QWEN_EXPERT_WEIGHT_DTYPE", "bfloat16").lower()

def preprocess_weights(self):
    # ... existing code ...

    # Determine dtype based on config
    if TT_QWEN_EXPERT_WEIGHT_DTYPE == "bfloat8_b":
        weight_dtype = ttnn.bfloat8_b
    elif TT_QWEN_EXPERT_WEIGHT_DTYPE == "bfloat4_b":
        weight_dtype = ttnn.bfloat4_b
    else:
        weight_dtype = ttnn.bfloat16  # Default

    # Apply to all expert weight conversions
    self.tt_w1_proj = ttnn.from_torch(
        gate_proj,
        dtype=weight_dtype,  # Use configurable dtype
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    # ... repeat for w2_proj, w3_proj ...
```

**Step 3.2: Add Accuracy Tests Before Enabling**

Run accuracy tests with:
```bash
TT_QWEN_EXPERT_WEIGHT_DTYPE=bfloat8_b pytest tests/test_qwen3_5_35b_a3b.py -v
```

Compare PCC values against baseline.

### Phase 4: DRAM-Sharded Weights (Days 4-5)

**Step 4.1: Add Helper Function**

**Location:** `TTNNQwenExperts` class

```python
def _create_dram_sharded_weight_config(self, k: int, n: int):
    """Create DRAM-sharded config for expert weights.

    Args:
        k: Input dimension (rows)
        n: Output dimension (columns)
    """
    # Get device DRAM grid (12 cores for Wormhole)
    dram_grid_size = self.device.dram_grid_size()
    dram_cores = dram_grid_size.x

    # Create DRAM core range
    dram_weight_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
        )
    })

    # Calculate shard shape (width-sharded)
    padded_n = ttnn.core.roundup(ttnn.core.divup(n, dram_cores), ttnn.TILE_SIZE)

    shard_spec = ttnn.ShardSpec(
        dram_weight_grid,
        (k, padded_n),
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )
```

**Step 4.2: Apply to Weight Storage**

**In preprocess_weights, use DRAM-sharded config when dimensions align:**
```python
# Only use DRAM-sharded for weights where n divides evenly
if n % (dram_cores * ttnn.TILE_SIZE) == 0:
    weight_mem_config = self._create_dram_sharded_weight_config(k, n)
else:
    weight_mem_config = ttnn.DRAM_MEMORY_CONFIG

self.tt_w1_proj = ttnn.from_torch(
    gate_proj,
    dtype=weight_dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=weight_mem_config,
    device=device,
)
```

---

## 4. Accuracy Test Suite Design

### 4.1 Unit Tests for Quantization

```python
@pytest.mark.parametrize("weight_dtype", ["bfloat16", "bfloat8_b", "bfloat4_b"])
def test_expert_weight_quantization_accuracy(weight_dtype, mesh_device):
    """Verify expert computation accuracy with different weight dtypes."""
    os.environ["TT_QWEN_EXPERT_WEIGHT_DTYPE"] = weight_dtype

    # Load reference PyTorch model
    torch_experts = load_pytorch_experts()
    ttnn_experts = TTNNQwenExperts.from_torch(torch_experts)
    ttnn_experts.preprocess_weights()
    ttnn_experts.move_weights_to_device()

    # Test with multiple input patterns
    test_inputs = [
        torch.randn(32, 4096),   # Decode batch
        torch.randn(128, 4096),  # Prefill batch
    ]

    for x in test_inputs:
        torch_out = torch_experts_forward(torch_experts, x)
        ttnn_out = ttnn_experts.forward(to_ttnn(x))

        pcc = pearson_correlation(torch_out, to_torch(ttnn_out))

        if weight_dtype == "bfloat16":
            assert pcc > 0.999, f"PCC {pcc} for bfloat16 weights"
        elif weight_dtype == "bfloat8_b":
            assert pcc > 0.995, f"PCC {pcc} for bfloat8_b weights"
        else:  # bfloat4_b
            assert pcc > 0.990, f"PCC {pcc} for bfloat4_b weights"
```

### 4.2 Compute Config Validation

```python
def test_compute_config_accuracy():
    """Verify LoFi compute config maintains accuracy."""
    # Run same input with different configs
    configs = [
        ("HiFi4", ttnn.MathFidelity.HiFi4),
        ("HiFi2", ttnn.MathFidelity.HiFi2),
        ("LoFi", ttnn.MathFidelity.LoFi),
    ]

    results = {}
    for name, fidelity in configs:
        output = run_expert_with_config(fidelity)
        results[name] = output

    # All should match HiFi4 reference closely
    hifi4_ref = results["HiFi4"]
    for name, output in results.items():
        pcc = pearson_correlation(hifi4_ref, output)
        assert pcc > 0.999, f"{name} PCC {pcc} vs HiFi4"
```

### 4.3 End-to-End Generation Test

```python
def test_generation_with_optimizations():
    """Verify text generation quality with all optimizations enabled."""
    os.environ["TT_QWEN_EXPERT_WEIGHT_DTYPE"] = "bfloat8_b"

    prompts = [
        "What is machine learning?",
        "Explain quantum computing simply.",
    ]

    for prompt in prompts:
        baseline_tokens = generate_baseline(prompt, max_tokens=50)
        optimized_tokens = generate_optimized(prompt, max_tokens=50)

        # Compare token sequences
        match_rate = sum(a == b for a, b in zip(baseline_tokens, optimized_tokens)) / len(baseline_tokens)
        assert match_rate > 0.90, f"Token match rate {match_rate}"
```

---

## 5. Success Criteria

### 5.1 Performance Targets

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| Expert matmul time (decode) | TBD ms | -20% | Kernel profiling |
| Expert matmul time (prefill) | TBD ms | -15% | Kernel profiling |
| Weight memory bandwidth | TBD GB/s | +50% | (via quantization) |
| MoE layer time (decode) | TBD ms | -15% | `DispatchManager` timing |
| MoE layer time (prefill) | TBD ms | -10% | `DispatchManager` timing |

### 5.2 Accuracy Thresholds

| Metric | Threshold | Test |
|--------|-----------|------|
| PCC (bfloat16 baseline) | > 0.999 | test_expert_accuracy |
| PCC (bfloat8_b weights) | > 0.995 | test_quantization_accuracy |
| PCC (LoFi compute config) | > 0.999 | test_compute_config_accuracy |
| Token match rate (generation) | > 90% | test_generation_with_optimizations |

### 5.3 Resource Constraints

| Metric | Constraint |
|--------|------------|
| Expert weight memory | -50% vs baseline (with bfloat8_b) |
| Compile time | < 1.5x baseline |
| Peak L1 memory (decode) | < baseline |

---

## 6. Risk Mitigation

### 6.1 Quantization Accuracy Degradation
- **Mitigation:** Start with bfloat8_b (conservative), only try bfloat4_b if accuracy passes
- **Test:** Run full eval suite before committing
- **Rollback:** Keep TT_QWEN_EXPERT_WEIGHT_DTYPE env var for easy revert

### 6.2 Compute Config Numerical Issues
- **Mitigation:** Test LoFi on diverse inputs (small/large values)
- **Test:** Compare against HiFi4 reference
- **Fallback:** Use HiFi2 if LoFi shows issues

### 6.3 DRAM-Sharded Alignment Issues
- **Mitigation:** Only enable for perfectly aligned dimensions
- **Test:** Verify tensor shapes match shard specs
- **Fallback:** Default to interleaved if alignment fails

---

## Appendix A: Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`
   - Add compute kernel configs
   - Add fused SiLU activation
   - Add weight quantization option
   - Add DRAM-sharded weight configs

2. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_qwen3_5_35b_a3b.py`
   - Add quantization accuracy tests
   - Add compute config validation tests

## Appendix B: Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TT_QWEN_USE_SPARSE_MATMUL` | "1" | Use sparse_matmul vs batched_matmul |
| `TT_QWEN_CPU_EXPERTS` | "0" | CPU fallback for debugging |
| `TT_QWEN_EXPERT_WEIGHT_DTYPE` | "bfloat16" (new) | Expert weight quantization |
| `TT_QWEN_USE_LOFI_COMPUTE` | "1" (new) | Use LoFi compute config |
| `TT_QWEN_FUSED_SILU` | "1" (new) | Use fused SiLU activation |
