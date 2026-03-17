# TTNNQwen3MoE Optimization Plan for T3K

**Target:** Qwen3.5-35B-A3B MoE module running on T3K (1x8 mesh)
**Goal:** Improve decode and prefill throughput without losing accuracy

---

## Table of Contents
1. [Current Implementation Analysis](#1-current-implementation-analysis)
2. [Identified Bottlenecks](#2-identified-bottlenecks)
3. [Proposed Optimizations](#3-proposed-optimizations)
4. [Step-by-Step Implementation Guide](#4-step-by-step-implementation-guide)
5. [Accuracy Test Suite Design](#5-accuracy-test-suite-design)
6. [Success Criteria](#6-success-criteria)

---

## 1. Current Implementation Analysis

### 1.1 Architecture Overview

**Qwen3.5-35B-A3B MoE Configuration:**
- 256 routed experts total
- Top-8 routing per token
- 32 experts per device (256 / 8 devices)
- Hidden size: configurable (typically 4096)
- Intermediate size: configurable (typically 14336)
- Shared expert with optional gating

### 1.2 Current Data Flow

```
Input (x) [batch, 1, seq, hidden/8]
    |
    v
[All-Gather] -- dim=-1, num_links=1, Topology.Linear
    |
    v
x_gathered [batch, 1, seq, hidden]
    |
    ├─────────────────────────────────────┐
    v                                     v
[Router/Gate]                      [Shared Expert]
    |                                     |
    v                                     v
topk_indices, topk_weights          shared_output
    |
    v
[TTNNQwenExperts.forward]
    |
    ├── [All-to-All Dispatch]  -- cluster_axis=1, no num_links specified
    │       |
    │       v
    │   dispatched_tokens [1, 1, total_tokens, hidden]
    │       |
    │       v
    │   [Expert Computation] -- sparse_matmul or batched_matmul
    │       |
    │       v
    │   expert_output
    │       |
    │       v
    └── [All-to-All Combine] -- cluster_axis=1, no num_links specified
            |
            v
        combined_output
            |
            v
        [Weight & Sum]
            |
            v
        routed_output
    |
    v
[Reduce-Scatter] -- dim=3, num_links=1, Topology.Ring
    |
    v
[Add shared_output]
    |
    v
Output [batch, 1, seq, hidden/8]
```

### 1.3 Key Components

| Component | File Location | Current Implementation |
|-----------|---------------|----------------------|
| Router | `TTNNQwenMoERouterDecode` | Softmax + 3-pass topk for precision |
| Experts | `TTNNQwenExperts` | sparse_matmul (default) or batched_matmul |
| MoE Orchestration | `TTNNQwen3MoE` | All-gather -> Route -> Experts -> Reduce-scatter |
| Shared Expert | `TTNNGlm4MoeMLP` | Inherited from GLM base |

---

## 2. Identified Bottlenecks

### 2.1 CRITICAL: Communication Link Underutilization

**Current State:**
```python
# qwen_moe.py lines 846-853, 944-956
num_links=1  # All CCL operations use only 1 link
```

**Reference (DeepSeek-V3):**
```python
# models/demos/deepseek_v3/tt/moe.py lines 127-131
"all_to_all_dispatch": {"num_links": 4},
"all_to_all_combine": {"num_links": 4}
```

**Impact:** T3K supports 4 links per device. Using only 1 link leaves 75% of inter-device bandwidth unused.

**Priority: HIGH** - Estimated 2-4x speedup for communication-bound workloads.

---

### 2.2 Missing num_links for All-to-All Operations

**Current State (qwen_moe.py lines 398-404, 525-531):**
```python
# all_to_all_dispatch - NO num_links parameter
all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
    x_rm,
    topk_experts_indices_rm,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
)

# all_to_all_combine - NO num_links parameter
combined_output = ttnn.all_to_all_combine(
    expert_output,
    all_to_all_dispatch_metadata,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
)
```

**Priority: HIGH** - These are the MoE-specific communication primitives where expert parallelism happens.

---

### 2.3 Inefficient Weight Broadcasting

**Current State (qwen_moe.py lines 551-558):**
```python
# Convert weights via permute and unsqueeze
topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (1, 0))
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 1)
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 3)
topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
```

**Issue:** Multiple layout conversions and reshapes create unnecessary memory copies.

**Reference (DeepSeek-V3 moe.py lines 336-344):**
```python
# Single repeat + permute operation
topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])
topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
```

**Priority: MEDIUM** - Layout conversions are relatively cheap, but the current approach is suboptimal.

---

### 2.4 Synchronous Operations Without Overlap

**Current State:** All operations are sequential:
1. All-gather waits for completion
2. Router computation starts
3. All-to-all dispatch waits
4. Expert computation starts
5. All-to-all combine waits
6. Reduce-scatter waits

**Opportunity:** Overlap shared expert computation with routed expert communication.

**Priority: MEDIUM** - Requires careful dependency analysis but can hide latency.

---

### 2.5 No L1 Sharding for Decode Mode Activations

**Current State:** Memory configs use basic L1_MEMORY_CONFIG or DRAM_MEMORY_CONFIG.

**Reference (DeepSeek-V3 moe.py lines 162-166):**
```python
input_output_memory_config = ttnn.create_sharded_memory_config(
    shape=(USERS_PER_ROW, HIDDEN_SIZE // TP_SIZE),
    core_grid=ttnn.CoreGrid(y=7, x=4),
    strategy=ttnn.ShardStrategy.WIDTH,
)
```

**Priority: LOW-MEDIUM** - May provide additional speedup for decode mode.

---

### 2.6 No Prefill Chunking

**Current State:** No chunking for large prefill sequences.

**Reference (DeepSeek-V3 moe.py lines 254-280):**
```python
chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
if global_tokens > chunk_tokens:
    return cls._forward_chunked_prefill(x, cfg, chunk_size)
```

**Priority: LOW** - Only relevant for large prefill batches; prevents OOM.

---

### 2.7 Router Precision Overhead

**Current State (qwen_moe.py lines 96-128):**
The router uses a 3-pass topk algorithm with float32 centering to achieve high precision:
1. BF16 topk(k+1) for coarse threshold
2. BF16 topk(k+1) on centered scores for refined threshold
3. BF16 topk(k) on doubly-centered scores for final indices

**Trade-off:** This is intentional for accuracy but adds compute overhead.

**Priority: LOW** - Only optimize if accuracy testing confirms single-pass is sufficient.

---

## 3. Proposed Optimizations

### Ranked by Impact (High to Low)

| Rank | Optimization | Expected Speedup | Risk | Effort |
|------|-------------|------------------|------|--------|
| 1 | Increase num_links to 4 for all CCL ops | 2-4x for CCL | Low | Low |
| 2 | Add num_links to all_to_all_dispatch/combine | 1.5-2x for dispatch/combine | Low | Low |
| 3 | Overlap shared expert with routed expert | 10-20% overall | Medium | Medium |
| 4 | L1 sharded memory config for decode | 5-15% for decode | Low | Medium |
| 5 | Prefill chunking | N/A (OOM prevention) | Low | Low |
| 6 | Fused weight broadcasting | 2-5% | Low | Low |
| 7 | Optional single-pass router | 5-10% router time | High (accuracy) | Low |

---

## 4. Step-by-Step Implementation Guide

### Phase 1: CCL Link Optimization (Days 1-2)

#### Step 1.1: Update All-Gather num_links

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`

**Location:** `TTNNQwen3MoE.forward()` around line 846-853

**Change:**
```python
# BEFORE:
x = ttnn.experimental.all_gather_async(
    x,
    dim=-1,
    multi_device_global_semaphore=...,
    barrier_semaphore=...,
    num_links=1,  # <-- CHANGE THIS
    topology=ttnn.Topology.Linear,
)

# AFTER:
x = ttnn.experimental.all_gather_async(
    x,
    dim=-1,
    multi_device_global_semaphore=...,
    barrier_semaphore=...,
    num_links=4,  # <-- Utilize all 4 links on T3K
    topology=ttnn.Topology.Linear,
)
```

#### Step 1.2: Update Reduce-Scatter num_links

**Location:** `TTNNQwen3MoE.forward()` around line 944-956

**Change:**
```python
# BEFORE:
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    ...,
    num_links=1,  # <-- CHANGE THIS
    ...
)

# AFTER:
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    ...,
    num_links=4,  # <-- Utilize all 4 links
    ...
)
```

#### Step 1.3: Add num_links to all_to_all_dispatch

**Location:** `TTNNQwenExperts.forward()` around line 398-404

**Change:**
```python
# BEFORE:
all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
    x_rm,
    topk_experts_indices_rm,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
)

# AFTER:
all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
    x_rm,
    topk_experts_indices_rm,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
    num_links=4,  # <-- ADD THIS
)
```

#### Step 1.4: Add num_links to all_to_all_combine

**Location:** `TTNNQwenExperts.forward()` around line 525-531

**Change:**
```python
# BEFORE:
combined_output = ttnn.all_to_all_combine(
    expert_output,
    all_to_all_dispatch_metadata,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
)

# AFTER:
combined_output = ttnn.all_to_all_combine(
    expert_output,
    all_to_all_dispatch_metadata,
    self.expert_mapping_tensors,
    cluster_axis=1,
    memory_config=decode_memory_config,
    num_links=4,  # <-- ADD THIS
)
```

### Phase 2: Async Overlap (Days 3-4)

#### Step 2.1: Restructure forward() for Overlap

**Goal:** Start shared expert computation while routed tokens are being dispatched.

**Location:** `TTNNQwen3MoE.forward()`

**Approach:**
```python
def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
    # 1. All-gather (blocking)
    x_gathered = ttnn.experimental.all_gather_async(x, ...)

    # 2. Router (blocking)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

    # 3. START shared expert computation (can overlap with dispatch)
    shared_output_future = self.shared_experts(x_gathered)  # If async API exists

    # 4. Routed experts with all-to-all
    routed_output = self.experts(x_gathered, topk_indices, topk_weights)

    # 5. Reduce-scatter
    routed_output = ttnn.experimental.reduce_scatter_minimal_async(...)

    # 6. WAIT for shared expert and combine
    output = ttnn.add(routed_output, shared_output_future)

    return output
```

**Note:** Check if TTNN supports async execution for MLPs. If not, reorder operations to maximize overlap.

### Phase 3: Memory Config Optimization (Days 5-6)

#### Step 3.1: Create Sharded Memory Configs for Decode

**File:** `qwen_moe.py`

**Add new helper function:**
```python
def _create_decode_memory_configs(self):
    """Create optimized sharded memory configs for decode mode."""
    USERS_PER_ROW = 32  # Typical decode batch size
    hidden_size = self.config.hidden_size
    tp_size = self.device.get_num_devices()

    self._decode_io_memory_config = ttnn.create_sharded_memory_config(
        shape=(USERS_PER_ROW, hidden_size // tp_size),
        core_grid=ttnn.CoreGrid(y=7, x=4),  # Wormhole grid
        strategy=ttnn.ShardStrategy.WIDTH,
    )
```

#### Step 3.2: Use Sharded Configs in forward()

**In decode mode, use the sharded config:**
```python
if is_decode_mode:
    memory_config = self._decode_io_memory_config
else:
    memory_config = ttnn.DRAM_MEMORY_CONFIG
```

### Phase 4: Prefill Chunking (Day 7)

#### Step 4.1: Add Chunked Prefill Support

**Add to `TTNNQwen3MoE`:**
```python
def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
    seq_len = x.shape[-2]
    chunk_tokens = 16384  # Configurable
    num_dispatch_devices = self.device.get_num_devices()
    global_tokens = seq_len * num_dispatch_devices

    if global_tokens > chunk_tokens:
        chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
        return self._forward_chunked(x, chunk_size)

    return self._forward_impl(x)

def _forward_chunked(self, x: ttnn.Tensor, chunk_size: int) -> ttnn.Tensor:
    _, _, seq_len, _ = x.shape
    output_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
        output_chunks.append(self._forward_impl(x_chunk))
        ttnn.deallocate(x_chunk)

    if len(output_chunks) == 1:
        return output_chunks[0]

    output = ttnn.concat(output_chunks, dim=2)
    for chunk in output_chunks:
        ttnn.deallocate(chunk)
    return output
```

---

## 5. Accuracy Test Suite Design

### 5.1 Unit Tests (Per Component)

#### Router Accuracy Test
```python
def test_router_accuracy():
    """Verify router produces identical top-k indices and weights vs PyTorch."""
    # Setup
    torch_router = load_pytorch_router()
    ttnn_router = TTNNQwenMoERouterDecode.from_torch(torch_router)

    # Test cases
    test_inputs = [
        torch.randn(1, 256),   # Single token
        torch.randn(32, 256),  # Decode batch
        torch.randn(128, 256), # Prefill batch
    ]

    for x in test_inputs:
        torch_indices, torch_weights = torch_router(x)
        ttnn_indices, ttnn_weights = ttnn_router(to_ttnn(x))

        # Verify indices match exactly
        assert torch.equal(torch_indices, to_torch(ttnn_indices))

        # Verify weights match within tolerance
        assert torch.allclose(torch_weights, to_torch(ttnn_weights), atol=1e-3)
```

#### Expert Computation Accuracy Test
```python
def test_experts_accuracy():
    """Verify expert computation matches PyTorch reference."""
    # Setup
    torch_experts = load_pytorch_experts()
    ttnn_experts = TTNNQwenExperts.from_torch(torch_experts)

    # Test with known routing
    x = torch.randn(32, 4096)  # 32 tokens
    indices = torch.randint(0, 256, (32, 8))  # top-8 routing
    weights = torch.softmax(torch.randn(32, 8), dim=-1)

    torch_out = torch_experts(x, indices, weights)
    ttnn_out = ttnn_experts(to_ttnn(x), to_ttnn(indices), to_ttnn(weights))

    pcc = pearson_correlation(torch_out, to_torch(ttnn_out))
    assert pcc > 0.99, f"PCC {pcc} below threshold"
```

#### Full MoE Accuracy Test
```python
def test_moe_accuracy():
    """Verify full MoE layer matches PyTorch reference."""
    torch_moe = load_pytorch_moe()
    ttnn_moe = TTNNQwen3MoE.from_torch(torch_moe)

    for seq_len in [1, 32, 128, 512]:
        x = torch.randn(1, 1, seq_len, 4096)

        torch_out = torch_moe(x)
        ttnn_out = ttnn_moe(to_ttnn(x))

        pcc = pearson_correlation(torch_out, to_torch(ttnn_out))
        assert pcc > 0.999, f"PCC {pcc} for seq_len={seq_len}"
```

### 5.2 Integration Tests

#### End-to-End Generation Test
```python
def test_generation_accuracy():
    """Verify generated text quality before/after optimization."""
    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI.",
    ]

    for prompt in prompts:
        # Generate with baseline
        baseline_output = generate_baseline(prompt, max_tokens=50)

        # Generate with optimized
        optimized_output = generate_optimized(prompt, max_tokens=50)

        # Token-level comparison (should be identical or very similar)
        token_match_rate = compare_tokens(baseline_output, optimized_output)
        assert token_match_rate > 0.95, f"Token match rate {token_match_rate}"
```

### 5.3 Performance Regression Tests

```python
def test_performance_baseline():
    """Establish and verify performance baselines."""
    test_configs = [
        {"mode": "decode", "batch_size": 32, "seq_len": 1},
        {"mode": "prefill", "batch_size": 1, "seq_len": 128},
        {"mode": "prefill", "batch_size": 1, "seq_len": 512},
    ]

    for cfg in test_configs:
        latency = measure_moe_latency(**cfg)

        # Log for tracking
        log_metric(f"moe_latency_{cfg['mode']}_{cfg['seq_len']}", latency)

        # Assert against baseline (set after initial measurement)
        # assert latency < BASELINE_LATENCIES[str(cfg)] * 1.1
```

---

## 6. Success Criteria

### 6.1 Performance Targets

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| Decode latency (32 users) | TBD ms | -30% | `test_qwen3_5_35b_a3b.py` timing |
| Prefill throughput (128 seq) | TBD tok/s | +50% | `test_qwen3_5_35b_a3b.py` timing |
| MoE layer time (decode) | TBD ms | -40% | `DispatchManager` timing |
| MoE layer time (prefill) | TBD ms | -30% | `DispatchManager` timing |

### 6.2 Accuracy Thresholds

| Metric | Threshold | Test |
|--------|-----------|------|
| PCC (MoE output vs PyTorch) | > 0.999 | `test_moe_accuracy` |
| PCC (Router weights) | > 0.999 | `test_router_accuracy` |
| Token match rate (generation) | > 95% | `test_generation_accuracy` |
| Perplexity change | < 0.1% | Eval on benchmark dataset |

### 6.3 Resource Constraints

| Metric | Constraint |
|--------|------------|
| Peak L1 memory (decode) | < 1 MB per core |
| Peak DRAM usage | < baseline |
| Compile time | < 2x baseline |

---

## 7. Risk Mitigation

### 7.1 Accuracy Degradation
- **Mitigation:** Run accuracy tests after each change
- **Rollback:** Feature flags for each optimization (e.g., `TT_QWEN_MoE_NUM_LINKS`)

### 7.2 OOM on Large Sequences
- **Mitigation:** Implement prefill chunking (Phase 4)
- **Test:** Stress test with seq_len=2048, 4096

### 7.3 CCL Hangs with num_links=4
- **Mitigation:** Test on isolated T3K before production
- **Fallback:** Make num_links configurable via environment variable

---

## Appendix A: Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/qwen_moe.py`
   - `TTNNQwenExperts.forward()` - Add num_links to all-to-all ops
   - `TTNNQwen3MoE.forward()` - Update CCL num_links, add chunking

2. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_qwen3_5_35b_a3b.py`
   - Add accuracy validation tests
   - Add performance benchmarks

## Appendix B: Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TT_QWEN_USE_SPARSE_MATMUL` | "1" | Use sparse_matmul vs batched_matmul |
| `TT_QWEN_CPU_EXPERTS` | "0" | CPU fallback for debugging |
| `TT_QWEN_MoE_NUM_LINKS` | "4" (new) | CCL link count |
| `TT_QWEN_PREFILL_CHUNK_SIZE` | "16384" (new) | Prefill chunking threshold |
