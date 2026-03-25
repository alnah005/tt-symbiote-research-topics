# PLAN: Ling T3K Multi-Device Attention Bug

## Problem Description

The Ling-mini-2.0 (BailingMoeV2) model generates garbled text output (e.g., "Theendienaukeeaukee...") when using TTNN-accelerated attention with paged KV cache on **T3K (8-device mesh)**.

## CRITICAL FINDING: Multi-Device Specific Bug

**Single device (1x1 mesh):**
- PCC stages tests: ALL PASS with PCC > 0.999
- QKV, QK-Norm, RoPE, SDPA, Dense - all stages have excellent PCC
- Min PCC: 0.999705, Max: 0.999940, Mean: 0.999858 over 32 decode steps

**Multi-device (T3K 1x8 mesh):**
- Produces garbled text "Theendienaukeeaukee..."
- Despite high per-stage PCC, end-to-end output is wrong

**This indicates the bug is in multi-device tensor handling**, not in the core attention math.

## Ling GQA Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 2048 |
| Q Heads (nh) | 16 |
| KV Heads (nkv) | 4 |
| GQA Ratio | 4:1 |
| Head Dimension | 128 |
| Partial Rotary Factor | 0.5 (64 dims get RoPE) |

## Multi-Device Specific Operations

The T3K path uses different modules than single-device:
- `TTNNLinearIColShardedWRowSharded` for Q projection
- `TTNNLinearIReplicatedWColSharded` for K/V projection
- `TTNNDistributedRotaryPositionEmbedding` (uses `rotary_embedding_llama`)
- All-gather and replicated tensor conversions

## Investigation Areas

1. **Q/K/V Tensor Gathering** - How tensors are gathered across devices
2. **Paged KV Cache Multi-Device** - How cache updates work across mesh
3. **Distributed RoPE** - Uses `rotary_embedding_llama` vs regular `rotary_embedding`
4. **All-gather/Replicate Conversions** - Data movement between devices

## Recommended Next Steps

1. Create T3K-specific PCC test that runs on multi-device
2. Add debugging to show tensor shapes/values at each step on T3K
3. Compare T3K tensor values against single-device reference
4. Check if all-gather produces correct results

## Files to Investigate

| File | Component |
|------|-----------|
| `modules/attention.py` | `_to_replicated()`, all-gather logic |
| `modules/rope.py` | `TTNNDistributedRotaryPositionEmbedding` |
| `modules/linear.py` | Sharded linear implementations |
