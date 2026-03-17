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
**Status:** Pending
**Why Needed:** Ling model generates incorrect text during decode. Need to understand paged_sdpa_decode kernel expectations for GQA with 4 KV heads and 16 Q heads.
**Questions:**
1. What does paged_sdpa_decode kernel expect for GQA (4 KV heads to 16 Q heads)?
2. Is there a mismatch in how cur_pos is interpreted?
3. Are there any known issues with TTNN paged attention?

Findings: [Pending research]
