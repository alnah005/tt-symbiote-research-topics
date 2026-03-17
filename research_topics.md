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

## RoPE Position Embedding Mesh Handling Bug
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Ling model generates repetitive garbage text, indicating position encoding issues.
**Questions:**
- How should replicated position embeddings (cos/sin) be extracted from ConcatMeshToTensor?
- What is the correct shape check for distributed cos/sin tensors?

**Findings:**
**Root Cause Identified:** In `_prepare_cos_sin_for_rope()` (attention.py:2182-2184), the shape check `cos_t.shape[0] == num_devices` is incorrect.

**The Bug:**
- cos/sin after unsqueeze have shape `[batch, 1, seq, dim]`
- After `ConcatMeshToTensor(dim=0)`, shape becomes `[num_devices * batch, 1, seq, dim]`
- The check `shape[0] == num_devices` only works when `batch=1`
- For other batch sizes, replicated tensors are NOT properly sliced

**The Fix:**
```python
# Replace incorrect check:
if cos_t.shape[0] == num_devices:  # WRONG
    cos_t = cos_t[:1]

# With correct slice:
original_batch = cos_t.shape[0] // num_devices
cos_t = cos_t[:original_batch]
```

**Impact:** This bug corrupts position embeddings in distributed mode, causing the model to lose positional information and generate repetitive output patterns.

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
