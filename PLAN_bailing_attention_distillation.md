# PLAN: Bailing Attention TTNN Implementation Distillation

## Current State Analysis

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
**Class:** `TTNNBailingMoEAttention` (Lines 2209-2835)
**Lines of Code:** 626 LOC
**Supporting Modules:**
- `TTNNRotaryPositionEmbedding` / `TTNNDistributedRotaryPositionEmbedding` in `rope.py`
- `BailingRotarySetup` in `rope.py`
- `to_replicated_topology()` in `tensor_utils.py`
- `TTNNLinearIColShardedWRowSharded` and `TTNNLinearIReplicatedWColSharded` in `linear.py`

**HuggingFace Reference:** `BailingMoeV2Attention` ~80 LOC

**Complexity Ratio:** 7.8x larger than reference implementation

---

## Identified Simplification Opportunities

### 1. QK Normalization Duplication (Potential Savings: ~35 LOC)
**Current Issue:** Two separate methods for QK normalization:
- `_apply_qk_norm()` for prefill [batch, heads, seq, head_dim]
- `_apply_qk_norm_decode()` for decode [1, batch, heads, head_dim]

**Simplification:** Unify into single method that handles both tensor layouts by detecting shape.

### 2. Topology Conversion Overhead (Potential Savings: ~20 LOC)
**Current Issue:** `_to_replicated()` performs host round-trip for every decode token:
```python
t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
return ttnn.from_torch(t_torch, mesh_mapper=ttnn.ReplicateTensorToMesh(device), ...)
```

**Simplification:** Investigate whether paged attention kernels truly require replicated topology or if this is legacy code. TT Transformers uses local heads per device without round-trips.

### 3. Multiple All-Gather Operations (Potential Savings: ~15 LOC)
**Current Issue:** Three separate all-gather calls in `_project_qkv_t3k()`:
```python
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
query_states = self._maybe_all_gather(query_states)
key_states = self._maybe_all_gather(key_states)
value_states = self._maybe_all_gather(value_states)
```

**Simplification:** Consider using fused QKV projection with single all-gather, similar to TT Transformers pattern.

### 4. Separate Q/K/V Projections vs Fused QKV (Potential Savings: ~80 LOC)
**Current Issue:** Split fused QKV into three separate projections in `from_torch()`:
```python
q_weight = qkv_weight[:q_size, :]
k_weight = qkv_weight[q_size:q_size + kv_size, :]
v_weight = qkv_weight[q_size + kv_size:, :]
```

**Simplification:** Keep fused QKV projection and use `ttnn.experimental.nlp_create_qkv_heads()` to split after projection, matching TT Transformers pattern.

### 5. RoPE Module Selection Complexity (Potential Savings: ~10 LOC)
**Current Issue:** Conditional selection between two RoPE implementations:
```python
if uses_partial_rotary:
    new_attn.rope = TTNNRotaryPositionEmbedding()
else:
    new_attn.rope = TTNNDistributedRotaryPositionEmbedding()
```

**Simplification:** Use single unified RoPE module that handles both cases.

### 6. Decode Position Handling Complexity (Potential Savings: ~25 LOC)
**Current Issue:** Complex cache_position tensor handling in `_forward_decode_paged()`:
```python
if cache_position is None:
    cur_pos = past_key_values.get_seq_length(layer_idx)
    cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
else:
    # 15 lines of TorchTTNNTensor/ttnn.Tensor conversion
```

**Simplification:** Standardize on single tensor type for position tracking.

---

## Step-by-Step Implementation Plan

### Phase 1: Consolidate QK Normalization (Low Risk)
1. Create unified `_apply_qk_norm_unified()` method
2. Detect tensor layout from shape dimensions
3. Replace both existing methods
4. Test prefill and decode paths

### Phase 2: Evaluate Topology Conversion (Medium Risk)
1. Add instrumentation to measure `_to_replicated()` overhead
2. Test if paged attention works without host round-trip
3. If yes, use ttnn-native topology conversion
4. If no, document why host round-trip is required

### Phase 3: Fuse QKV Projection (High Impact)
1. Revert to fused QKV weight storage
2. Use single projection followed by `nlp_create_qkv_heads()`
3. Handle all-gather after split (single operation)
4. Remove separate Q/K/V projection code

### Phase 4: Unify RoPE Handling (Low Risk)
1. Extend `TTNNRotaryPositionEmbedding` to handle distributed case
2. Remove conditional RoPE selection
3. Simplify `BailingRotarySetup` usage

### Phase 5: Simplify Position Handling (Low Risk)
1. Create helper function for cache position conversion
2. Standardize tensor type handling
3. Reduce code duplication

---

## Success Criteria

1. **Functional Correctness:**
   - Test `test_ling_mini_2_0` passes on T3K
   - Generated text remains coherent
   - No regression in numerical accuracy

2. **Code Reduction Target:**
   - Reduce from 626 LOC to ~400 LOC (36% reduction)
   - Minimum viable: 500 LOC (20% reduction)

3. **Performance Preservation:**
   - Prefill latency within 5% of current
   - Decode latency within 5% of current
   - No increase in host-device transfers

---

## Risk Areas to Watch

1. **Paged Attention Topology Requirements:** The `_to_replicated()` host round-trip may be necessary for paged attention kernel compatibility. Test thoroughly before removing.

2. **GQA Head Ratio:** Ling-mini-2.0 has 16 Q heads and 4 KV heads (4:1 ratio). The separate Q/K/V projections exist specifically to handle num_kv_heads < num_devices. Fused QKV must preserve this.

3. **Partial Rotary Factor:** Ling uses `partial_rotary_factor < 1.0`. The decode RoPE path requires manual slice/concat. Unified RoPE must handle this case.

4. **Multi-Device Coordination:** All-gather and reduce-scatter operations must maintain correct synchronization across T3K devices.

---

## Critical Files for Implementation

| File | Purpose |
|------|---------|
| `modules/attention.py` | Core TTNNBailingMoEAttention class (626 LOC to distill) |
| `modules/rope.py` | RoPE implementations to potentially unify |
| `modules/tensor_utils.py` | Topology conversion utilities |
| `tests/test_ling_mini_2_0.py` | Test file to validate correctness |
| `llama3_70b_galaxy/tt/llama_attention.py` | Reference pattern for fused QKV approach |
