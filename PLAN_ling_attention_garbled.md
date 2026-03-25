# PLAN: Ling Attention Garbled Text Investigation

## Problem Description

The Ling-mini-2.0 (BailingMoeV2) model generates garbled text output (e.g., "a lot's a lot's...") when using TTNN-accelerated attention with paged KV cache. Component-level PCC tests pass (>0.999) but full model inference produces incoherent output.

**Key observation:** The attention-only test (`test_ling_attention_only.py`) PASSES and produces correct output like "Paris", but the full model test fails.

## Ling GQA Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2048 |
| Q Heads (nh) | 16 |
| KV Heads (nkv) | 4 |
| GQA Ratio | 4:1 |
| Head Dimension | 128 |
| Partial Rotary Factor | 0.5 (64 dims get RoPE) |

## Root Cause Analysis: Comparison with Working Implementations

### Working Implementations Found

1. **tt_transformers/tt/attention.py** (lines 482-642) - Reference working GQA
2. **TTNNQwen3FullAttention** (qwen_attention.py lines 697-821) - Similar to Ling, working

### Critical Differences

| Aspect | Ling (Broken) | tt_transformers (Working) |
|--------|---------------|---------------------------|
| Permutes after nlp_create_qkv_heads_decode | **3 permutes** | **0 permutes** |
| KV memory config | HEIGHT_SHARDED explicit | Let kernel handle |
| cur_pos for SDPA | Was +1 (documented bug) | Same as update |
| RoPE | Custom `_apply_partial_rope` on [B,H,S,D] | `rotary_embedding_llama` on [1,B,H,D] |

## Hypotheses (Ranked by Confidence)

### 1. Excessive Permute Operations (HIGH CONFIDENCE)

**Evidence:**
- Ling uses THREE permute operations:
  1. `ttnn.permute(query_states, (1, 2, 0, 3))` - after nlp_create_qkv_heads_decode
  2. QK norm and RoPE processing
  3. `ttnn.permute(query_states, (2, 0, 1, 3))` - for paged kernels

- tt_transformers does NOT permute Q after `nlp_create_qkv_heads_decode`; Q stays in `[1, B, H, D]` format and goes directly to paged SDPA

- Extra permutes may introduce numerical precision loss with bfloat16

### 2. HEIGHT_SHARDED Configuration Mismatch (MEDIUM CONFIDENCE)

**Evidence:**
- Ling explicitly creates HEIGHT_SHARDED memory config for KV before `paged_update_cache` (lines 2891-2913)
- tt_transformers does not explicitly shard KV tensors
- Shard height computation: `((self.num_kv_heads + 31) // 32) * 32` = 32 for 4 KV heads

### 3. GQA Padding Mismatch (MEDIUM CONFIDENCE)

**Evidence from paged_sdpa_decode_for_gqa guide:**
- Ling: nh=16, nkv=4, group_size=4
- If padded independently: nh_padded=32, nkv_padded=32 -> effective_group_size=1 (MHA collapse!)
- Correct padding: nh_padded=32, nkv_padded=8 -> effective_group_size=4

### 4. Partial RoPE Handling (LOW-MEDIUM CONFIDENCE)

- Ling uses `partial_rotary_factor=0.5` (only 64 of 128 dims get RoPE)
- tt_transformers uses `rotary_embedding_llama` with `is_decode_mode=True`
- Mismatch in non-rotated dimension handling could cause errors

## Recommended Investigation Steps

### Step 1: Validate GQA Padding
Add assertions to verify `nh_padded / nkv_padded == 4` after all padding operations.

### Step 2: Simplify Permute Operations
Match tt_transformers pattern: keep Q in `[1, B, H, D]` format after `nlp_create_qkv_heads_decode`. Only permute K/V to `[S, B, H, D]` for paged_update_cache.

### Step 3: Remove HEIGHT_SHARDED Explicit Configuration
Let paged_update_cache handle memory layout internally (match tt_transformers).

### Step 4: Test Component Isolation
Run decode-only test with fixed KV cache contents. Compare tensor values at each step against PyTorch reference.

### Step 5: PCC at Each Decode Step
Add per-step PCC logging between TTNN and PyTorch reference. Identify when/where PCC drops below 0.99.

## Critical Files

| File | Purpose |
|------|---------|
| `models/experimental/tt_symbiote/modules/attention.py` | TTNNBailingMoEAttention._forward_decode_paged (lines 2734-2943) |
| `models/tt_transformers/tt/attention.py` | Reference working GQA decode (lines 482-680) |
| `models/experimental/tt_symbiote/modules/qwen_attention.py` | Similar working GQA (lines 697-821) |

## Related Plans

- `PLAN_ling_incorrect_text.md` - Documents cur_pos +1 bug (may be fixed)
- `PLAN_ling_permute_fix.md` - Documents sharding/permute issues with Option A quick fix

## Success Criteria

1. **PCC > 0.99** between TTNN and PyTorch at each decode step
2. **Coherent text generation** matching PyTorch baseline quality
3. **Component tests still pass** after fixes
