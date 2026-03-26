# PLAN: Minimize Ling-mini-2.0 Code Changes

**Date:** 2026-03-26
**Author:** Architect Agent
**Goal:** Reduce the current 993-insertion/425-deletion diff to the minimal set of changes required for correct text generation.

---

## Executive Summary

The current diff contains 5 essential bug fixes wrapped in extensive refactoring and optimization changes. The minimal diff should be approximately **350-400 lines changed** (down from 993+425), keeping only the essential correctness fixes.

---

## Classification of All Changes

### A. ESSENTIAL BUG FIXES (MUST KEEP)

These changes fix actual bugs that cause garbled output. Without any one of them, the model produces incorrect text.

#### A1. Q/K Weight Permutation: HF-to-Meta Layout
**Files:** `attention.py` (lines 2220-2289, 2434-2441, 2446-2448, 2486-2492)
**What:** Added `_reverse_permute_weight()` and `_reverse_permute_1d()` functions. Applied in `from_torch()` to Q/K weights and biases, and to QK norm weights.
**Why essential:** The `rotary_embedding_llama` kernel rotates adjacent pairs `(2i, 2i+1)` (Meta/interleaved format), but HuggingFace weights produce split-half pairs `(i, i+head_dim/2)`. Without this permutation, RoPE is applied to the wrong element pairs, producing garbled attention.
**Verdict:** KEEP all of `_reverse_permute_weight`, `_reverse_permute_1d`, and their usage in `from_torch()`.

#### A2. `update()` Method Fix: `=` Instead of `+=` for seq_lengths
**Files:** `attention.py` (lines 265-277)
**What:** Changed `_seq_lengths[layer_idx] += seq_len` to `= seq_len`, and added CPU-side KV cache concatenation (`_cpu_key_cache`/`_cpu_value_cache`).
**Why essential:** The original `update()` just returned the incoming K/V without concatenating with history. PyTorch fallback layers that call `update()` expected DynamicCache semantics (concatenate and return full history). Without the CPU cache, PyTorch layers see only the current token's K/V, not the full sequence. The `= seq_len` fix is needed because the concatenated tensor's seq dim IS the total length, not an increment.
**Verdict:** KEEP the CPU cache fields, the concatenation logic, and the `= seq_len` fix.

#### A3. Removal of Double-Counting `_seq_lengths += seq_length` in Decode Paths
**Files:** `attention.py` (Bailing decode path ~line 2898, Glm4 decode path ~line 1940)
**What:** Removed `past_key_values._seq_lengths[layer_idx] += seq_length` and `_seen_tokens += seq_length` from `_forward_decode_paged`.
**Why essential:** `paged_update_on_device()` (line 212) already does `_seq_lengths[layer_idx] += seq_len`. The forward method was incrementing a second time, causing `get_seq_length()` to return 2x the actual position. This made `cache_position` and `cur_pos_tt` wrong, corrupting all subsequent decode steps.
**Verdict:** KEEP these removals.

#### A4. `past_key_value` (Singular) Parameter Name Mapping
**Files:** `attention.py` (forward method, ~line 2973)
**What:** `if past_key_value is not None and past_key_values is None: past_key_values = past_key_value`
**Why essential:** HuggingFace BailingMoeV2 passes `past_key_value` (singular), not `past_key_values` (plural). Without this mapping, the KV cache is silently ignored and every decode step has no history.
**Note:** This fix already existed in the original code (line 2937-2938). The new version just cleaned up the surrounding code. **No change needed here** -- the original already had this.
**Verdict:** This is NOT a new change. The original already handled it. KEEP AS-IS (original code is fine).

#### A5. RoPE Format Consistency: BailingRotarySetup for Both Prefill and Decode
**Files:** `attention.py` (lines 2547-2558, 2684-2700, 2820-2872), `rope.py` (lines 339-708)
**What:** Introduced `BailingRotarySetup` class that pre-computes cos/sin in Meta format with identity padding for partial rotary. Both prefill and decode now use `rotary_embedding_llama` with Meta-format cos/sin.
**Why essential:** The original code used HuggingFace-format position embeddings from the model's `rotary_emb` for prefill (via `TTNNRotaryPositionEmbedding` which calls `ttnn.experimental.rotary_embedding`), but decode needs `rotary_embedding_llama` which expects Meta format. When K values are stored in the KV cache during prefill using HF-format RoPE and then Q values during decode use Meta-format RoPE, the rotary phases are incompatible. This is the root cause of the "RoPE format mismatch" bug.
**However:** The question is whether we can use the SIMPLER approach of making decode also use HF format, instead of making prefill use Meta format. The answer is NO -- `rotary_embedding_llama` for decode with HEIGHT_SHARDED tensors only supports Meta format. The prefill path could theoretically use either, but the K values stored in KV cache must be rotated with the same convention as Q during decode. Therefore, both must use Meta format.
**Verdict:** KEEP `BailingRotarySetup` and its usage. This is the largest essential change.

#### A6. Decode Path: `nlp_create_qkv_heads_decode` / `nlp_concat_heads_decode`
**Files:** `attention.py` (lines 2765-2940)
**What:** Complete rewrite of decode path to use `nlp_create_qkv_heads_decode` for head splitting and `nlp_concat_heads_decode` for head concatenation, with HEIGHT_SHARDED memory configs throughout.
**Why essential (partially):** The original decode path used `permute(0,2,1,3)` to go from `[B,H,S,D]` to `[S,B,H,D]` format. The new path uses `nlp_create_qkv_heads_decode` which directly produces `[1,B,H,D]` from a fused QKV tensor, then everything stays HEIGHT_SHARDED. This is **tightly coupled** with the RoPE fix (A5) because `rotary_embedding_llama` in decode mode (`is_decode_mode=True`) requires HEIGHT_SHARDED inputs.
**Key question:** Could the original permute-based decode path work with just the RoPE fix? The answer is: the original path could potentially work IF we handle the HEIGHT_SHARDING requirements manually (the original did `ttnn.to_memory_config(key_states, shard_cfg)` for KV). However, the original decode path did NOT shard Q for RoPE, and `rotary_embedding_llama` decode mode requires Q to be HEIGHT_SHARDED too. So the `nlp_create_qkv_heads_decode` approach is the cleaner way to get everything HEIGHT_SHARDED.
**Verdict:** KEEP the nlp_create/concat_heads_decode approach. It is simpler than manually adding HEIGHT_SHARDING to the original permute-based path.

---

### B. OPTIMIZATION CHANGES (TEST BEFORE DECIDING)

These changes improve numerical accuracy but the model may still generate correct text without them.

#### B1. `compute_kernel_config` with HiFi2/fp32 Accumulation in Linear Layers
**Files:** `linear.py` (59 lines of changes across 5 linear classes)
**What:** Added `compute_kernel_config` with `HiFi2`, `fp32_dest_acc_en=True`, `packer_l1_acc=True` to all linear layer classes, and passed it to `ttnn.linear()`.
**Impact:** Changes matmul computation from default (likely LoFi) to HiFi2 with FP32 accumulation. This significantly improves numerical precision.
**Risk of reverting:** Without FP32 accumulation, error compounds across 24 layers of attention + MoE. The model might still produce readable text but with higher probability of degradation over longer sequences.
**Verdict:** TEST WITHOUT to see if text is still correct. If yes, REVERT (can be added later as optimization). If no, KEEP.

#### B2. `compute_kernel_config` with HiFi2/fp32 Accumulation in RMSNorm
**Files:** `normalization.py` (17 lines)
**What:** Same pattern as B1 but for RMSNorm operations.
**Risk of reverting:** Lower risk than linear since norm is a simpler operation.
**Verdict:** TEST WITHOUT. Likely revertable.

#### B3. `async all_gather` Changed to `sync all_gather`
**Files:** `attention.py` (lines 2321-2341)
**What:** `_maybe_all_gather` changed from `ttnn.experimental.all_gather_async()` (with semaphores, barrier, synchronize_device) to simple `ttnn.all_gather()`.
**Impact:** Simplification. The sync version is simpler but potentially slower. However, this is NOT a correctness issue.
**Note:** The original also removed the `_is_distributed` property check (which checked for ccl_manager). The new code always calls `ttnn.all_gather()`.
**Verdict:** This simplification is SAFE TO KEEP (less code, works correctly). But it should be tested to ensure T3K doesn't hang.

---

### C. PURE REFACTORING (REVERT ALL)

These changes have zero impact on correctness and only reorganize code.

#### C1. New File: `tensor_utils.py` (191 lines)
**What:** Extracted `ensure_ttnn_bfloat16()` and `to_replicated_topology()` and `ensure_replicated_position_embeddings()` into a new module.
**Verdict:** REVERT. Inline these functions back into attention.py. Eliminates new file entirely.

#### C2. `ensure_ttnn_bfloat16()` Helper Function
**What:** Replaced inline `if hasattr(x, "to_ttnn"): x = x.to_ttnn; if x.dtype != bfloat16: typecast` patterns.
**Verdict:** REVERT to inline patterns. The helper is nice but adds to diff size.

#### C3. `to_replicated_topology()` Extraction
**What:** Moved `_to_replicated()` body to tensor_utils.py.
**Verdict:** REVERT. Keep `_to_replicated()` inline in each class that needs it (Bailing, Glm4).

#### C4. `_project_qkv_t3k()` Helper Method
**What:** Extracted Q/K/V projection + all-gather from `_forward_prefill` and `_forward_decode_paged` into shared method.
**Verdict:** REVERT. Inline the projection code back into prefill/decode. Reduces diff by avoiding structural changes to both methods.

#### C5. Comments Added to `qwen_attention.py` (5 lines)
**What:** Added explanatory comments about data replication vs tensor parallelism.
**Verdict:** REVERT. Comments-only changes, no functional impact.

#### C6. Extended Test Prompt in `test_ling_mini_2_0.py` (2 lines)
**What:** Changed test prompt from short to very long.
**Verdict:** REVERT. Short prompt is sufficient for correctness testing.

#### C7. Comments Added to Glm4MoeLiteAttention._to_replicated
**What:** Added 5-line comment about data replication.
**Verdict:** REVERT.

#### C8. Removed `_is_distributed` Property
**What:** Removed the `_is_distributed` property and `_use_separate_qkv` flag.
**Verdict:** KEEP the removal of `_use_separate_qkv` and `query_key_value` since Bailing is T3K-only now. But this is a design choice, not essential.

#### C9. `from_torch()` Removed `distributed` Parameter
**What:** Simplified from_torch to always use T3K mode.
**Verdict:** Can be partially reverted for minimal diff. But keeping it is also fine since Bailing only runs on T3K.

#### C10. T3K-Only Validation in `forward()`
**What:** Added `if num_devices <= 1: raise RuntimeError(...)` check.
**Verdict:** REVERT. Validation is nice but not essential. The original code just had different paths.

#### C11. Removed DynamicCache Support from `_forward_prefill`
**What:** Removed the `else:` branch in prefill that handled DynamicCache.
**Verdict:** This IS needed because the DynamicCache path called `past_key_values.update()` which was broken (see A2). But now that A2 is fixed, the DynamicCache path could theoretically work again. However, Bailing uses paged cache exclusively, so removing dead code is OK. KEEP for simplicity.

#### C12. Docstring Changes
**What:** Various docstring improvements across the class.
**Verdict:** REVERT where possible to minimize diff.

#### C13. RoPE Padding Change (32 -> 64 alignment)
**Files:** `rope.py` (lines 195-198)
**What:** Changed `rotary_dim % 32` to `rotary_dim % 64` for padding alignment.
**Impact:** This is in `TTNNRotaryPositionEmbedding` which is only used for prefill in the ORIGINAL code. Since the new code uses `BailingRotarySetup` for both prefill and decode, this change only matters for non-Bailing models.
**Verdict:** REVERT. Not relevant for Bailing. If it breaks other models, it should be a separate PR.

#### C14. `apply_partial_rope_decode` Standalone Function
**Files:** `rope.py` (lines 34-92)
**What:** Added standalone function for decode-mode RoPE. Not actually used by the Bailing attention code (which calls `rotary_embedding_llama` directly).
**Verdict:** REVERT. Dead code for this use case.

---

## Minimal Change Set

### Changes to KEEP

| # | Change | File | Estimated Lines |
|---|--------|------|----------------|
| A1 | `_reverse_permute_weight` and `_reverse_permute_1d` functions | attention.py | +70 |
| A1b | Apply permutation in `from_torch()` (Q/K weights, biases, QK norm weights) | attention.py | +15 |
| A2 | CPU-side KV cache in `update()` method | attention.py | +20 |
| A3 | Remove double-counting in Bailing decode | attention.py | -4 |
| A3b | Remove double-counting in Glm4 decode | attention.py | -4 |
| A5 | `BailingRotarySetup` class | rope.py | +200 |
| A5b | `_compute_cos_sin_cache` and `_get_rotation_transformation_mat` helpers | rope.py | +80 |
| A5c | Initialize `_rotary_setup` in `move_weights_to_device_impl` | attention.py | +10 |
| A5d | Use `_rotary_setup` in prefill path for Meta-format RoPE | attention.py | +15 |
| A5e | Use `_rotary_setup` in decode path for Meta-format RoPE | attention.py | +30 |
| A6 | Decode path using `nlp_create_qkv_heads_decode` / `nlp_concat_heads_decode` | attention.py | +60 |
| B3 | Sync all_gather (simpler, less code) | attention.py | -5 |
| **Total** | | | **~490 lines** |

### Changes to REVERT

| # | Change | File | Lines Saved |
|---|--------|------|-------------|
| C1 | Delete `tensor_utils.py` | tensor_utils.py | -191 lines (whole file) |
| C2 | Inline `ensure_ttnn_bfloat16` back | attention.py | -20 |
| C3 | Keep `_to_replicated` inline | attention.py | -10 |
| C4 | Inline `_project_qkv_t3k` back | attention.py | -30 |
| C5 | Revert qwen_attention.py comments | qwen_attention.py | -5 |
| C6 | Revert test prompt change | test_ling_mini_2_0.py | -2 |
| C7 | Revert Glm4 comment addition | attention.py | -5 |
| C10 | Revert T3K validation | attention.py | -8 |
| C12 | Revert docstring changes | attention.py | ~-30 |
| C13 | Revert 32->64 padding | rope.py | -2 |
| C14 | Revert `apply_partial_rope_decode` | rope.py | -60 |

### Changes to TEST BEFORE DECIDING

| # | Change | File | Test Procedure |
|---|--------|------|----------------|
| B1 | HiFi2/fp32 in linear.py | linear.py | Run test_ling_mini_2_0.py WITHOUT this change. If text is coherent, revert. |
| B2 | HiFi2/fp32 in normalization.py | normalization.py | Same test as B1. Test together. |

---

## Implementation Plan for the Implementer

### Phase 1: Create Minimal Branch (from clean HEAD)

1. **Create new branch** from the commit BEFORE the big change
2. **Cherry-pick or manually apply** only the essential changes listed above

### Phase 2: Apply Essential Bug Fixes (in order)

**Step 1: Fix `update()` method in TTNNPagedAttentionKVCache**
- Add `_cpu_key_cache` and `_cpu_value_cache` fields to `__init__`
- Rewrite `update()` to concatenate with history and use `= seq_len`
- This is a small, isolated change

**Step 2: Remove double-counting of seq_lengths in decode paths**
- Remove `past_key_values._seq_lengths[layer_idx] += seq_length` from `TTNNBailingMoEAttention._forward_decode_paged`
- Remove same from `TTNNGlm4MoeLiteAttention._forward_decode_paged`
- These are 4-line removals

**Step 3: Add weight permutation functions and apply them**
- Add `_reverse_permute_weight()` and `_reverse_permute_1d()` as module-level functions
- In `TTNNBailingMoEAttention.from_torch()`, apply permutation to Q/K weights after splitting from fused QKV
- Apply `_reverse_permute_1d()` to QK norm weights if `use_qk_norm` is True
- Apply `_reverse_permute_1d()` to Q/K biases if present

**Step 4: Add BailingRotarySetup to rope.py**
- Add `_get_rotation_transformation_mat()` helper
- Add `_compute_cos_sin_cache()` helper
- Add `BailingRotarySetup` class with `get_cos_sin_for_prefill()`, `get_cos_sin_for_decode()`, `get_trans_mat()`, and `get_trans_mat_decode_sharded()`

**Step 5: Rewrite prefill path to use BailingRotarySetup**
- In `_forward_prefill()`, replace the HF-format position_embeddings and `_ensure_replicated_tensor` with `self._rotary_setup.get_cos_sin_for_prefill()` and `rotary_embedding_llama`
- Initialize `self._rotary_setup` in `move_weights_to_device_impl()`

**Step 6: Rewrite decode path to use nlp_create_qkv_heads_decode**
- Replace the permute-based head splitting with `nlp_create_qkv_heads_decode`
- Use `BailingRotarySetup.get_cos_sin_for_decode()` for Meta-format RoPE
- Use HEIGHT_SHARDED configs for RoPE inputs
- Use `nlp_concat_heads_decode` for head concatenation
- Remove the old `_to_replicated()` calls for Q/K/V (the fused tensor is replicated once instead)

### Phase 3: Test

1. Run `test_ling_mini_2_0.py` -- must produce coherent text
2. Run `test_bailing_attention_accuracy.py` -- must pass PCC > 0.99
3. If both pass, the minimal diff is validated

### Phase 4: Test Optimizations (Optional)

4. If Phase 3 passes, ALSO test without B1/B2 (HiFi2/fp32 changes):
   - Revert linear.py and normalization.py changes
   - Re-run both tests
   - If still passing, leave them out (submit as separate optimization PR)
   - If failing, add them back

---

## Key Answers to Questions

### Q1: Can the original decode path (permute-based) work if we ONLY fix the RoPE format?
**No.** The original decode path uses `permute(query_states, (2, 0, 1, 3))` to go from `[B,H,S,D]` to `[S,B,H,D]`, then passes to paged kernels. But `rotary_embedding_llama` in decode mode requires HEIGHT_SHARDED inputs. The original path only shards K/V (for paged_update_cache) but NOT Q or cos/sin. We would need to add manual HEIGHT_SHARDING for Q, cos, sin, and trans_mat to the original path, which would be MORE code than the nlp_create_qkv_heads_decode approach.

### Q2: Are the compute_kernel_config changes (HiFi2/fp32) needed for correct text?
**Unknown -- needs testing.** They definitely improve accuracy, but the model might produce readable text without them. The default compute kernel uses LoFi which has lower precision. For a 24-layer MoE model, accumulated error could potentially cause issues.

### Q3: Can we keep the original `_ensure_replicated_tensor` inline?
**No.** The original `_ensure_replicated_tensor` handles converting HF-format position embeddings to replicated TTNN tensors. But now we use `BailingRotarySetup` which pre-computes position embeddings in the correct format at initialization. The inline function is no longer needed because the position embeddings are generated internally, not passed in from the model.

### Q4: Is the CPU-side KV cache (`_cpu_key_cache`) truly needed?
**Yes.** Without it, `update()` returns only the current token's K/V states. PyTorch fallback layers (used for non-TTNN attention layers) call `update()` expecting DynamicCache semantics where the returned K/V contains the full sequence history. The paged KV cache on device handles TTNN layers, but PyTorch layers need the CPU concatenation.

---

## Estimated Final Diff Size

- **attention.py:** ~400 lines changed (down from 883)
- **rope.py:** ~300 lines added (down from 452, removing dead code and comments)
- **linear.py:** 0 lines (if B1 is revertable) or 59 lines (if needed)
- **normalization.py:** 0 lines (if B2 is revertable) or 17 lines (if needed)
- **qwen_attention.py:** 0 lines (reverted)
- **test_ling_mini_2_0.py:** 0 lines (reverted)
- **tensor_utils.py:** 0 lines (not created)

**Total: ~700 lines changed (down from ~1400) in the best case, ~770 lines if HiFi2 changes are needed.**
