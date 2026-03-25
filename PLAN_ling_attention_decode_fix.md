# Plan: Fix Layer 10 Decode Path Accuracy in TTNNBailingMoEAttention

**Date:** 2026-03-25
**Target:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
**Class:** `TTNNBailingMoEAttention._forward_decode_paged()`
**Model:** Ling-mini-2.0 (BailingMoeV2)

---

## 1. Problem Summary

### Symptoms
- Garbled text generation when only TTNN attention is enabled
- PCC degradation from 0.99 to 0.43 over 29 decode iterations
- **Layer 10 specifically degrades first** (min PCC 0.5286) while Layer 0 remains stable (avg PCC 0.9687)
- Prefill path works correctly (all layers PCC > 0.999)
- Problem appears at decode step 1 (not cumulative from prefill)

### Observed Per-Layer Accuracy
| Layer | Avg PCC | Min PCC | Behavior |
|-------|---------|---------|----------|
| Layer 0 | 0.9687 | 0.9069 | Stable |
| Layer 10 | 0.7288 | 0.5286 | **Severe degradation from step 1** |
| Layer 19 | 0.9038 | 0.8318 | Secondary degradation |

### Model Configuration
- **Q heads (nh):** 16
- **KV heads (nkv):** 4
- **GQA group_size:** 4 (16 / 4)
- **Head dimension (dh):** 128
- **Total layers:** 20
- **partial_rotary_factor:** < 1.0 (uses partial RoPE)
- **use_qk_norm:** True

---

## 2. Root Cause Analysis

### 2.1 Primary Hypothesis: GQA Padding Issue in Decode Sharding

**Finding:** The `_forward_decode_paged()` method creates a height-sharded memory config for K/V states before writing to the paged cache:

```python
# Lines 2863-2874 in attention.py
tile_size = 32
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size  # = 32 for nkv=4

core_grid = ttnn.CoreGrid(y=1, x=batch_size)
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),  # (32, 128)
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
value_states = ttnn.to_memory_config(value_states, shard_cfg)
```

**Problem:** The shard height is padded to 32, but the actual KV data only occupies 4 heads. The remaining 28 "heads" in the shard are garbage/zeros. If this padding is not correctly handled when:
1. Writing to the paged cache via `paged_update_cache`, or
2. Reading from the paged cache via `paged_sdpa_decode`

...the kernel may read from the padded garbage data instead of the actual KV heads.

**Why Layer 10 specifically?**
- KV cache writes are cumulative across layers. By Layer 10, there have been 10 previous writes to the cache.
- If the padding garbage from early layers corrupts the cache structure, it manifests more severely at middle layers where:
  - The cumulative cache state is larger
  - The kernel's block-boundary arithmetic may hit edge cases

### 2.2 Secondary Hypothesis: Block Boundary Issues (Issue #30362)

Per the research cache (`guides/paged_sdpa_decode_for_gqa/ch5_known_issues/issue_30362_pcc_failures.md`):

> Sporadic PCC failures at certain `cur_pos` values in the range 0-16K when using paged SDPA decode. Failures cluster near block boundaries (multiples of `block_size`).

**Relevance:**
- Ling uses `block_size=64` (from `PagedAttentionConfig`)
- At decode step 1, `cur_pos` could be at a block boundary depending on prefill length
- Layer 10 may happen to align with a problematic `cur_pos` value

### 2.3 Tertiary Hypothesis: Tensor Layout/Permute Mismatch

The decode path performs multiple permutations:

```python
# Line 2851-2854: After RoPE, permute from B H S D -> S B H D
query_states = ttnn.permute(query_states, (2, 0, 1, 3))
key_states = ttnn.permute(key_states, (2, 0, 1, 3))
value_states = ttnn.permute(value_states, (2, 0, 1, 3))
```

Then after SDPA:

```python
# Line 2898: Permute back
attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, head_dim]
```

**Problem:** The paged SDPA decode kernel expects Q in shape `[1, b, nh, dh]` per the API reference. But after the permute, the shape is `[S, B, H, D]` = `[1, 1, 16, 128]`. This matches `[1, b, nh, dh]` only if:
- `S=1` (decode)
- `B=1` (batch_size)

If the kernel interprets axis 1 as batch and axis 2 as num_heads, this could work for batch_size=1, but the semantic mismatch could cause issues with layer indexing in the cache.

---

## 3. Implementation Plan

### Phase 1: Diagnostic Instrumentation (1-2 hours)

#### Step 1.1: Add Per-Layer PCC Logging to `_forward_decode_paged`

Create a debug version of the method that captures:
1. Input hidden_states PCC vs torch reference
2. Q/K/V states PCC after projection
3. Q/K states PCC after RoPE
4. K/V states PCC after sharding (before cache write)
5. Cache content PCC after write
6. SDPA output PCC

```python
# Add at the beginning of _forward_decode_paged
if os.environ.get("LING_DECODE_DEBUG"):
    layer_idx = self._fallback_torch_layer.layer_idx
    print(f"[Layer {layer_idx}] Decode debug start")
    # Store reference tensors for comparison
```

#### Step 1.2: Verify KV Cache Contents Per Layer

Add a test that:
1. Runs prefill normally
2. Runs decode step 1
3. Reads back K/V cache contents for layers 0, 10, 19
4. Compares against torch reference

```python
# In test_ling_mini_2_0_attention_pcc_stages.py
def test_per_layer_cache_contents_after_decode(device):
    # Setup model and cache
    # Run prefill
    # Run decode step 1
    for layer_idx in [0, 10, 19]:
        k_cache, v_cache = read_cache_blocks_to_torch(paged_cache, layer_idx, seq_length, device)
        k_ref, v_ref = get_torch_kv_for_layer(model, layer_idx)
        pcc_k = compute_pcc(k_ref, k_cache)
        pcc_v = compute_pcc(v_ref, v_cache)
        print(f"[Layer {layer_idx}] K cache PCC: {pcc_k:.4f}, V cache PCC: {pcc_v:.4f}")
```

### Phase 2: Fix Sharding Configuration (1-2 hours)

#### Step 2.1: Remove Unnecessary Padding in Shard Height

**Current code:**
```python
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size
```

**Fix:** The paged_update_cache kernel should handle KV heads directly without requiring tile-aligned sharding. Try:

```python
# Option A: Use actual KV head count without padding
shard_h = self.num_kv_heads  # 4 for Ling

# Option B: If sharding is required for performance, ensure padding is zeros
# and that paged_update_cache only writes the first num_kv_heads entries
```

#### Step 2.2: Verify paged_update_cache Input Shape

Per the API reference (`ch2_ttnn_api/tensor_shape_reference.md`):
> `paged_update_cache` input: `[b x nkv x 1 x dh]`

But the code permutes to `[S, B, H, D]` = `[1, 1, 4, 128]` which is semantically different from `[b, nkv, 1, dh]` = `[1, 4, 1, 128]`.

**Fix:** Add explicit reshape before cache write:
```python
# After permute, current shape is [1, b, nkv, dh]
# Reshape to match API: [b, nkv, 1, dh]
key_states = ttnn.permute(key_states, (1, 2, 0, 3))  # [b, nkv, 1, dh]
value_states = ttnn.permute(value_states, (1, 2, 0, 3))
```

### Phase 3: Verify GQA Group Size Preservation (1 hour)

#### Step 3.1: Check Effective Group Size

Per `ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md`:
> Correct padding: `nkv_padded = nh_padded / original_group_size`

Add assertion:
```python
pnh = ((self.num_heads + 31) // 32) * 32  # = 32 for nh=16
nkv_padded_expected = pnh // (self.num_heads // self.num_kv_heads)  # = 32 // 4 = 8

# Verify cache is allocated with correct nkv
assert paged_cache.num_kv_heads == self.num_kv_heads  # Should be 4, not 8
```

**Note:** There may be a discrepancy here. The cache stores `nkv=4` heads, but if the kernel expects `nkv_padded=8` for GQA math to work correctly, we may need to pad the KV cache allocation.

### Phase 4: Test and Validate (2-3 hours)

#### Step 4.1: Unit Test for Decode-Only Path

```python
def test_decode_only_layer_10_accuracy(device):
    """Test Layer 10 decode accuracy in isolation."""
    # Create model with only layer 10
    # Initialize cache with known values
    # Run single decode step
    # Compare output PCC
    # Target: PCC > 0.99
```

#### Step 4.2: Multi-Step Decode Regression Test

```python
def test_multi_step_decode_pcc_stability(device):
    """Verify PCC does not degrade over decode steps."""
    for step in range(30):
        output = run_decode_step(model, step)
        pcc = compute_pcc(output, reference[step])
        assert pcc > 0.98, f"Step {step}: PCC {pcc:.4f} < 0.98"
```

#### Step 4.3: Block Boundary Sweep

```python
def test_block_boundary_positions(device):
    """Test cur_pos values around block boundaries."""
    block_size = 64
    for k in range(1, 10):
        for delta in range(-2, 3):
            cur_pos = k * block_size + delta
            output = run_decode_at_position(model, cur_pos)
            pcc = compute_pcc(output, reference)
            assert pcc > 0.98, f"cur_pos={cur_pos}: PCC {pcc:.4f}"
```

---

## 4. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Layer 10 min PCC | 0.5286 | > 0.98 |
| Layer 10 avg PCC | 0.7288 | > 0.99 |
| Layer 19 min PCC | 0.8318 | > 0.98 |
| Multi-step decay | 0.99 -> 0.43 | Stable > 0.98 |
| Generated text | Garbled | Coherent |

---

## 5. Risk Assessment

### High Risk
- **GQA padding change may require kernel modification**: If the TTNN kernel expects padded KV heads, we cannot fix this in Python alone. Would need to file an issue or work around with explicit KV expansion.

### Medium Risk
- **Performance regression**: Removing sharding or adding reshapes may slow down decode. Need to benchmark before/after.

### Low Risk
- **Breaking prefill path**: Changes should be isolated to decode. Prefill already works correctly.

---

## 6. Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `TTNNBailingMoEAttention._forward_decode_paged()` - Main fix target
   - `TTNNPagedAttentionKVCache.paged_update_on_device()` - Verify input handling

2. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0_attention_pcc_stages.py`
   - Add per-layer cache content verification tests
   - Add block boundary sweep tests

---

## 7. Next Steps

1. **Immediate**: Run diagnostic test to verify KV cache contents per layer
2. **If cache contents are wrong**: Fix sharding/permute issue in `_forward_decode_paged`
3. **If cache contents are correct**: Issue is in `paged_sdpa_decode` kernel - file bug report
4. **Validate**: Run full multi-step decode test to confirm fix

---

## 8. References

- `guides/paged_sdpa_decode_for_gqa/` - Full API documentation
- `guides/paged_sdpa_decode_for_gqa/ch5_known_issues/issue_30362_pcc_failures.md` - Known block boundary bug
- `guides/paged_sdpa_decode_for_gqa/ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md` - GQA padding requirements
- `guides/paged_sdpa_decode_for_gqa/ch6_debugging/root_cause_isolation.md` - Debug workflow
