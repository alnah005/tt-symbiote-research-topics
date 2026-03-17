# Research Topics

This file tracks research topics that the Architect needs to investigate for making informed decisions.

---

## Format

```
## [Topic Name]
**Date:** YYYY-MM-DD
**Status:** Pending | In Progress | Completed
**Why Needed:** [Reason this research is necessary]
**Questions:**
- Question 1
- Question 2

**Findings:**
[Results of research go here]

---
```

## Topics

---

## MoE Optimization Techniques for TTNN
**Date:** 2026-03-16
**Status:** Pending
**Why Needed:** Need to understand best practices for optimizing Mixture of Experts models on Tenstorrent hardware, specifically comparing batched matmul vs sparse_matmul approaches.
**Questions:**
- What are the performance characteristics of sparse_matmul vs batched matmul for MoE?
- How should sparsity tensors be constructed for optimal performance?
- What program configs are recommended for different batch/sequence sizes?

**Findings:**
[Pending research]

---

## T3K Mesh Device Optimizations
**Date:** 2026-03-16
**Status:** Pending
**Why Needed:** TTNNQwen3MoE runs on T3K (1x8 mesh) and needs device-specific optimizations for expert parallelism.
**Questions:**
- What are the optimal num_links settings for all_to_all operations on T3K?
- How should memory configs (L1 vs DRAM) be chosen for decode vs prefill?
- What are the bandwidth characteristics between T3K devices?

**Findings:**
[Pending research]

---

## Expert Parallelism Strategies
**Date:** 2026-03-16
**Status:** Pending
**Why Needed:** Qwen3.5-35B has 256 experts with top-8 routing. Need optimal dispatch/combine strategies.
**Questions:**
- How does all_to_all_dispatch/combine compare to alternative expert routing schemes?
- What is the optimal expert-to-device assignment for 256 experts on 8 devices?
- How should routing weights be processed to minimize overhead?

**Findings:**
[Pending research]

---

## Weight Quantization for MoE Experts
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** DeepSeek-V3 uses bfloat4_b/bfloat8_b weight quantization for experts, but Qwen uses full bfloat16. Need to evaluate quantization trade-offs.
**Questions:**
- What accuracy loss is expected from bfloat4_b vs bfloat8_b vs bfloat16 for expert weights?
- How does weight quantization affect compute throughput on Wormhole?
- Which projections (gate/up/down) are most sensitive to quantization?

**Findings:**
[Pending research]

---

## Compute Kernel Configuration for MoE
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** DeepSeek-V3 uses COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc, but Qwen MoE doesn't specify compute kernel configs. Need to optimize.
**Questions:**
- What is the performance difference between LoFi, HiFi2, and HiFi4 for MoE expert matmuls?
- How does packer_l1_acc affect throughput for expert computations?
- What is the accuracy trade-off for using math_approx_mode?

**Findings:**
[Pending research]

---

## Expert Weight Memory Layout Optimization
**Date:** 2026-03-17
**Status:** Pending
**Why Needed:** Current implementation stores expert weights in DRAM with standard interleaved config. DRAM-sharded layouts may improve memory bandwidth.
**Questions:**
- What is the performance gain from DRAM-sharded weight storage?
- How should expert weights be laid out for optimal prefetch patterns?
- What are the tile size constraints for expert weight sharding?

**Findings:**
[Pending research]

---

## Paged SDPA Decode for GQA (Group Query Attention)
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Ling model generates incorrect text during decode. Need to understand paged_sdpa_decode kernel expectations for GQA with 4 KV heads and 16 Q heads.
**Questions:**
1. What does paged_sdpa_decode kernel expect for GQA (4 KV heads to 16 Q heads)?
2. Is there a mismatch in how cur_pos is interpreted?
3. Are there any known issues with TTNN paged attention?

**Findings:**

### 1. Expected Tensor Shapes for paged_sdpa_decode

From FlashDecode technical report and TTNN test utilities:

**Query tensor:**
- Expected shape: `[1, bsz, num_q_heads, head_dim]` (notation: 1BQD)
- The kernel puts heads in the Y dimension for parallelization
- For decode, seq_len=1 in dim 0

**KV Cache tensors:**
- Shape: `[max_num_blocks, num_kv_heads, block_size, head_dim]` for paged attention
- Or `[bsz, num_kv_heads, kv_len, head_dim]` for non-paged

**GQA Handling:**
- The kernel handles GQA natively
- `n_qh_per_kvh = n_q_heads // n_kv_heads` (e.g., 16/4 = 4)
- Parallelization: `bsz * n_kv_heads` across cores, `n_qh_per_kvh` within a core
- Query is internally reshaped: `query.view(1, bsz*n_kv_heads, n_qh_per_kvh, head_dim)`

**cur_pos_tensor:**
- Shape: `[batch_size]` - 1D tensor of int32
- Contains the current decoding position for each batch item
- Used for causal masking (attends only to positions 0..cur_pos for each batch)

### 2. cur_pos Interpretation

The `cur_pos_tensor` is interpreted as:
- The **current position** where the new token is being decoded
- Cache reads: `k_cache[:, :, :cur_pos, :]` (all positions up to cur_pos)
- This matches what paged_update_cache expects for `update_idxs_tensor`

**CRITICAL:** The cur_pos passed to `paged_sdpa_decode` should be the position **after** updating the cache. If you write to position N, then cur_pos should be N so that attention attends to positions 0..N.

### 3. Potential Issues Identified

**Issue A: Query Shape After Permute**
In tt_symbiote, query is permuted from `[B, H, S, D]` to `[S, B, H, D]`:
- For decode (S=1): results in `[1, B, H, D]` which matches expected `[1, bsz, num_q_heads, head_dim]`
- This appears CORRECT

**Issue B: KV Cache Layout vs Standard**
tt_symbiote cache shape: `(max_num_blocks, num_kv_heads, block_size, head_dim)`
Standard expected: `(bsz * max_num_blocks_per_seq, num_kv_heads, block_size, head_dim)`
- This appears CORRECT (matches test utilities)

**Issue C: Position Tracking vs RoPE Position**
The PLAN document already identified this: cache position uses `get_seq_length(layer_idx)` while RoPE uses external `position_ids`. If these diverge:
- KV written to wrong cache slot
- Attention reads wrong context

**Issue D: q_chunk_size=0 in decode config**
tt_symbiote uses `q_chunk_size=0, k_chunk_size=0` for decode config. The FlashDecode documentation suggests these should be computed based on sequence length. Zero chunk sizes may trigger default behavior which could be incorrect.

**Recommendation:** Match the chunk size calculation from the working model (`models/tt_transformers/tt/attention.py`) which uses `get_attn_sdpa_decode_program_config()`.

### 4. No Known TTNN Paged Attention Bugs

Test file `test_sdpa_decode.py` shows paged attention tests passing for:
- GQA configurations (nh=32, nkv=8 for llama 3.1)
- Various block sizes (16, 32, 64, 128)
- Bug fixes for issue #37927 (block_size=16, q_chunk_size==head_size edge cases)

The kernel itself is well-tested. Issues are likely in how tt_symbiote calls it.

### 5. Root Cause Hypothesis

Based on the existing PLAN document and this analysis, the most likely issues are:

1. **Missing reduce-scatter in dense projection** (documented in PLAN Section 10)
2. **MoE router precision** causing expert selection drift (documented in PLAN Section 11)
3. **Position tracking mismatch** between RoPE and cache position

The paged_sdpa_decode kernel itself appears to be called correctly for GQA.
