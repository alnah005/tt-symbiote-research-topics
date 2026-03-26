# Plan: Eliminate ReshapeView Ops in BailingAttention

**Date:** 2026-03-26
**Context:** ReshapeView is #1 device kernel bottleneck at 26.9% of device kernel time (328 ops, 11,081us total, 33.8us avg across all 8 T3K devices).

---

## 1. Complete Inventory of All 328 ReshapeView Ops

### 1.1 How the 328 Total Breaks Down

The profiled test runs: 1 warmup prefill + 3 warmup decode + 1 profiled prefill + 8 profiled decode = **13 forward passes** (but Tracy captures warmup too, so all 13 are counted). Wait -- the test has 14 total: 1 warmup prefill + 3 warmup decode + 1 profiled prefill + 8 profiled decode = 13 passes.

**Correction after careful count:** 1 warmup prefill + 3 warmup decode + 1 profiled prefill + 8 profiled decode = 13 passes.

Per device:
- Prefill passes: 2 (1 warmup + 1 profiled)
- Decode passes: 11 (3 warmup + 8 profiled)

### 1.2 View vs Device Kernel Classification

**Critical insight:** Most `ttnn.reshape` calls do NOT generate device kernels. The TTNN reshape implementation (reshape.cpp lines 374-384) checks `this_is_view`: when the last dimension stays the same, memory config matches, and padding is compatible, it performs a host-side metadata-only view -- zero device cost.

Only reshapes that change the last dimension in TILE_LAYOUT generate actual ReshapeView device kernels (they go through `reshape_tiled()` which launches reader/writer kernels to rearrange tiles).

### 1.3 Device-Kernel Reshapes (the real 328 ops)

| # | Location | Shape Change | Path | Per-Pass Count |
|---|----------|-------------|------|----------------|
| 1 | `attention.py:2685` | `[1,1,2048]` -> `[1,1,16,128]` | Decode: Q head split | 1 |
| 2 | `attention.py:2686` | `[1,1,512]` -> `[1,1,4,128]` | Decode: K head split | 1 |
| 3 | `attention.py:2687` | `[1,1,512]` -> `[1,1,4,128]` | Decode: V head split | 1 |
| 4 | `attention.py:2572` | `[B,S,2048]` -> `[B,S,16,128]` | Prefill: Q head split | 1 |
| 5 | `attention.py:2573` | `[B,S,512]` -> `[B,S,4,128]` | Prefill: K head split | 1 |
| 6 | `attention.py:2574` | `[B,S,512]` -> `[B,S,4,128]` | Prefill: V head split | 1 |
| 7 | `attention.py:2626` | `[B,S,16,128]` -> `[B,S,2048]` | Prefill: output head merge | 1 |

**Decode path:** 3 device-kernel reshapes per pass (Q/K/V head splits only)
**Prefill path:** 4 device-kernel reshapes per pass (Q/K/V head splits + output merge)

### 1.4 Verification Math

```
Decode device-kernel reshapes: 11 passes x 3 = 33 per device
Prefill device-kernel reshapes: 2 passes x 4 = 8 per device
Total per device: 33 + 8 = 41
Total across 8 devices: 41 x 8 = 328  <<< EXACT MATCH
```

### 1.5 View-Only Reshapes (NOT in the 328 count -- zero device cost)

These fire as host-side metadata operations and are NOT bottlenecks:

| Location | Shape Change | Why It's a View |
|----------|-------------|-----------------|
| `linear.py:247` (qkv_proj input) | `[B,1,3072]` -> `[B,1,1,3072]` | Last dim unchanged |
| `linear.py:287` (qkv_proj output) | `[B,1,1,3072]` -> `[B,1,3072]` | Last dim unchanged |
| `linear.py:383` (dense input) | `[B,1,embed]` -> `[B,1,1,embed]` | Last dim unchanged |
| `linear.py:387` (dense output) | `[B,1,1,out]` -> `[B,1,out]` | Last dim unchanged |
| `attention.py:2488` (QK norm Q flatten) | `[1,1,16,128]` -> `[16,128]` | Last two dims unchanged |
| `attention.py:2489` (QK norm K flatten) | `[1,1,4,128]` -> `[4,128]` | Last two dims unchanged |
| `attention.py:2508` (QK norm Q unflatten) | `[16,128]` -> `[1,1,16,128]` | Last two dims unchanged |
| `attention.py:2509` (QK norm K unflatten) | `[4,128]` -> `[1,1,4,128]` | Last two dims unchanged |
| `attention.py:2838` (output reshape) | `[1,1,X]` -> `[1,1,X]` | Shape unchanged, no-op |

These 8-9 per decode + 12-14 per prefill add host dispatch overhead (~1-2us each) but no device kernel time.

---

## 2. Eliminable Reshapes

### 2.1 Decode Path: Q/K/V Head Splits (3 per decode, 33.8us avg each)

**What they do:** Split flat projection output `[1,1,H*D]` into multi-head format `[1,B,H,D]` for RoPE and paged attention.

**Why they exist:** The fused QKV projection outputs a flat `[B,1,Q+K+V]` tensor. After slicing into Q/K/V, each needs to be reshaped from `[B,1,H*D]` to `[1,B,H,D]` -- the last dimension changes from H*D to D, requiring a device kernel to rearrange tiles.

#### Option A: Use `nlp_create_qkv_heads_decode` (Eliminates all 3)

**What:** Instead of slice + 3x reshape, use the fused `ttnn.experimental.nlp_create_qkv_heads_decode` op that splits AND reshapes in a single kernel.

**Code change:**
```python
# BEFORE (attention.py lines 2659-2687):
query_states = ttnn.slice(qkv_states, [0, 0, 0], [batch_size, 1, q_size])
key_states = ttnn.slice(qkv_states, [0, 0, q_size], [batch_size, 1, q_size + kv_size])
value_states = ttnn.slice(qkv_states, [0, 0, q_size + kv_size], [batch_size, 1, q_size + 2 * kv_size])
ttnn.deallocate(qkv_states)
query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))

# AFTER:
query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv_states,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
ttnn.deallocate(qkv_states)
```

**Risk:** MEDIUM.
- `nlp_create_qkv_heads_decode` expects 4D input `[1,1,B,H*D_total]` with specific padding (B padded to 32). The fused QKV output is `[B,1,Q+K+V]` which needs a view to `[1,1,B,Q+K+V]` first.
- Phase 1 lesson: This was tried before and REGRESSED due to host dispatch overhead from the `_to_replicated` round-trip that was needed at the time. The fused QKV + all_reduce path now produces replicated output, so the `_to_replicated` round-trip is no longer needed.
- Need to verify that `nlp_create_qkv_heads_decode` output has correct topology metadata for downstream RoPE and paged attention.

**Expected savings per decode:**
- Eliminates: 3 ReshapeView device kernels (3 x 33.8us = ~101us device time) + 3 slice ops
- Adds: 1 nlp_create_qkv_heads_decode op (~30-50us device time)
- Net savings: ~50-70us device time per decode per device
- Also saves 5 host dispatch round-trips (3 slices + 2 extra ops vs 1 fused op)

#### Option B: Reshape to 4D with same last dim, then transpose (Eliminates 3 reshapes, adds 3 transposes)

**What:** Instead of `[B,1,H*D]` -> `[1,B,H,D]`, do `[B,1,H*D]` -> `[B,1,H,D]` (view, since we can pick shapes where last dim stays at D... but H*D -> D still changes last dim). This does NOT work -- any split of the last dimension requires a device kernel.

**Verdict:** Not viable.

#### Option C: Change linear output format to pre-split heads (Eliminates reshapes at source)

**What:** Modify `TTNNLinearIColShardedWAllReduced` to output in `[1,B,H,D]` format directly by using a different weight layout and output reshape.

**Risk:** HIGH. Would require restructuring the weight matrix and matmul output handling. The matmul naturally produces `[B,1,out_features]` -- splitting into heads requires rearranging tiles regardless of where you do it.

**Verdict:** Not viable -- the fundamental issue is that matmul outputs flat features and heads require tile rearrangement.

### 2.2 Prefill Path: Q/K/V Head Splits + Output Merge (4 per prefill)

**What they do:** Same head splitting for prefill, plus merging heads back after SDPA.

#### Option A: Use `nlp_create_qkv_heads` for prefill (Eliminates 3 head splits)

```python
# BEFORE (attention.py lines 2572-2574 + permutes 2576-2578):
query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
key_states = ttnn.reshape(key_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (batch_size, seq_length, self.num_kv_heads, self.head_dim))
query_states = ttnn.permute(query_states, (0, 2, 1, 3))
key_states = ttnn.permute(key_states, (0, 2, 1, 3))
value_states = ttnn.permute(value_states, (0, 2, 1, 3))

# AFTER: Concat Q/K/V -> use nlp_create_qkv_heads (handles reshape + transpose)
qkv_concat = ttnn.concat([query_states, key_states, value_states], dim=-1)
query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads(
    qkv_concat,
    num_heads=self.num_heads,
    num_kv_heads=self.num_kv_heads,
    transpose_key=False,
)
```

**Risk:** LOW-MEDIUM. Prefill is not the bottleneck path. Saves 3 reshapes + 3 permutes, replaces with 1 concat + 1 fused op.

**Expected savings per prefill:** ~135us device time (4 x 33.8us) but only 2 prefill passes so total = ~270us x 8 devices = 2,160us. Negligible compared to decode savings.

#### Option B: Use `nlp_concat_heads` for prefill output merge (Eliminates 1 reshape)

The output merge `[B,S,H,D]` -> `[B,S,H*D]` can potentially use `ttnn.experimental.nlp_concat_heads`.

**Risk:** LOW. Standard pattern used in tt-transformers.

### 2.3 View-Only Reshapes: Linear 3D-to-4D Padding (0 device cost, but host overhead)

The linear module reshapes `[B,S,F]` -> `[B,1,S,F]` before matmul and back after. These are views (no device kernel) but add ~1-2us host dispatch each. With 2 linear calls per decode x 2 reshapes = 4 host dispatches.

**Optimization:** Make the linear modules accept 3D input directly by removing the while loop that pads to 4D. The TTNN matmul should handle 3D tensors.

**Risk:** LOW. But savings are tiny (~4-8us host time per decode).

**Verdict:** Low priority. Do this as cleanup after higher-impact changes.

---

## 3. Priority Ordering

### Priority 1: Decode Q/K/V head splits via `nlp_create_qkv_heads_decode` (HIGH IMPACT)

- **Eliminates:** 3 ReshapeView + 3 Slice = 6 device ops per decode
- **Adds:** 1 nlp_create_qkv_heads_decode
- **Net savings:** ~50-70us device time per decode per device
- **Total across profiled run:** 11 decode x 50us x 8 devices = ~4,400us
- **Fraction of ReshapeView budget eliminated:** 33/41 per device in decode = 80% of decode reshapes
- **Implementation:** ~15 lines of code change in `_forward_decode_paged`
- **Risk:** MEDIUM -- need to verify tensor topology metadata compatibility

### Priority 2: Prefill Q/K/V head splits via `nlp_create_qkv_heads` (LOW IMPACT)

- **Eliminates:** 3 ReshapeView + 3 Permute = 6 device ops per prefill
- **Adds:** 1 concat + 1 nlp_create_qkv_heads
- **Net savings:** ~100us device time per prefill per device
- **Total:** Small (only 2 prefill passes in this test)
- **Implementation:** ~10 lines in `_forward_prefill`
- **Risk:** LOW -- prefill is not latency-critical

### Priority 3: Prefill output merge via `nlp_concat_heads` (NEGLIGIBLE IMPACT)

- **Eliminates:** 1 ReshapeView per prefill
- **Adds:** 1 nlp_concat_heads
- **Net savings:** ~0-10us (the fused op may cost the same)
- **Verdict:** Not worth the code change unless already touching prefill path

### Priority 4: Remove linear 3D-to-4D view reshapes (NEGLIGIBLE IMPACT)

- Zero device kernel time savings (already views)
- Saves ~4-8us host dispatch per decode
- **Verdict:** Cleanup only, do when touching linear.py for other reasons

---

## 4. Reshapes That CANNOT Be Eliminated

### 4.1 The Head Split Problem is Fundamental

The Q/K/V head split (flat features -> multi-head format) requires rearranging tiles because the last dimension changes. A matmul inherently outputs `[B,S,H*D]` and attention kernels need `[1,B,H,D]`. This tile rearrangement must happen somewhere -- the question is whether it happens in 3 separate ReshapeView kernels or 1 fused `nlp_create_qkv_heads_decode` kernel.

**We cannot eliminate the tile rearrangement work, only reduce the number of kernel launches.**

### 4.2 Why the Output Merge Reshape (Decode) is Already Free

The decode output path uses `nlp_concat_heads_decode` (line 2826-2829) which already produces `[1,1,B,H*D]` format. The subsequent reshape at line 2838 (`[B,1,-1]`) is a view (last dim unchanged), so it costs zero device time. This is already optimal.

### 4.3 View Reshapes in QK Norm

The 4 QK norm reshapes (flatten to 2D for layernorm, unflatten after) are all views because the last dimension (head_dim=128) never changes. These have zero device kernel cost and cannot be further optimized.

---

## 5. Summary

| Category | Count (per device, all passes) | Device Time | Eliminable? | How? |
|----------|-------------------------------|-------------|-------------|------|
| Decode Q/K/V head splits | 33 (11 passes x 3) | ~1,117us | YES | `nlp_create_qkv_heads_decode` |
| Prefill Q/K/V head splits | 6 (2 passes x 3) | ~203us | YES | `nlp_create_qkv_heads` |
| Prefill output merge | 2 (2 passes x 1) | ~68us | YES | `nlp_concat_heads` |
| **Total device-kernel reshapes** | **41** | **~1,388us** | | |
| View-only reshapes (host) | ~130 | 0us device | N/A | Already free |

**Recommended action:** Implement Priority 1 only (decode `nlp_create_qkv_heads_decode`). This addresses 80% of decode-path ReshapeView ops with a single, well-understood code change. The remaining 20% are in the prefill path which is not the decode-latency bottleneck.

**Expected outcome:** Reduce per-decode ReshapeView device time from ~101us (3 x 33.8us) to ~30-50us (1 fused op), saving ~50-70us per decode per device. Also eliminates 3 Slice ops and 5 host dispatch round-trips.

---

## 6. Implementation Notes for Priority 1

### Prerequisites
1. The fused QKV output from `TTNNLinearIColShardedWAllReduced` is `[B,1,3072]` (3D, replicated on all devices after all_reduce).
2. `nlp_create_qkv_heads_decode` expects input shape `[1,1,B,Q+K+V]` where B is padded to 32.

### Steps
1. View-reshape QKV output: `[1,1,3072]` -> `[1,1,1,3072]` (free view, last dim unchanged)
2. If B < 32, may need padding (check if the op handles this internally)
3. Call `nlp_create_qkv_heads_decode(qkv_4d, num_heads=16, num_kv_heads=4)`
4. Verify output shapes: Q=`[1,B_pad,16,128]`, K=`[1,B_pad,4,128]`, V=`[1,B_pad,4,128]`
5. Verify output tensor topology is compatible with downstream RoPE (`rotary_embedding_llama` with `is_decode_mode=True`) and `paged_update_on_device` / `paged_sdpa_decode`
6. Remove the 3 slice ops and 3 reshape ops

### Fallback
If `nlp_create_qkv_heads_decode` has topology/metadata issues (as happened in Phase 1 with `_to_replicated`), the fallback is to keep the current 3-reshape approach but investigate if a single `ttnn.reshape` on the un-sliced QKV tensor can split all heads at once, reducing 3 kernels to 1.
