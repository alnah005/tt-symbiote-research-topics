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
**Status:** Completed
**Why Needed:** Need to understand best practices for optimizing Mixture of Experts models on Tenstorrent hardware, specifically comparing batched matmul vs sparse_matmul approaches.
**Questions:**
- What are the performance characteristics of sparse_matmul vs batched matmul for MoE?
- How should sparsity tensors be constructed for optimal performance?
- What program configs are recommended for different batch/sequence sizes?

**Findings:**
`guides/moe_optimization_techniques_for_ttnn/`

---

## T3K Mesh Device Optimizations
**Date:** 2026-03-16
**Status:** Completed
**Why Needed:** TTNNQwen3MoE runs on T3K (1x8 mesh) and needs device-specific optimizations for expert parallelism.
**Questions:**
- What are the optimal num_links settings for all_to_all operations on T3K?
- How should memory configs (L1 vs DRAM) be chosen for decode vs prefill?
- What are the bandwidth characteristics between T3K devices?

**Findings:**
`guides/t3k_mesh_device_optimizations/`

---

## Expert Parallelism Strategies
**Date:** 2026-03-16
**Status:** Completed
**Why Needed:** Qwen3.5-35B has 256 experts with top-8 routing. Need optimal dispatch/combine strategies.
**Questions:**
- How does all_to_all_dispatch/combine compare to alternative expert routing schemes?
- What is the optimal expert-to-device assignment for 256 experts on 8 devices?
- How should routing weights be processed to minimize overhead?

**Findings:**
`guides/expert_parallelism_strategies/`

---

## Weight Quantization for MoE Experts
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** DeepSeek-V3 uses bfloat4_b/bfloat8_b weight quantization for experts, but Qwen uses full bfloat16. Need to evaluate quantization trade-offs.
**Questions:**
- What accuracy loss is expected from bfloat4_b vs bfloat8_b vs bfloat16 for expert weights?
- How does weight quantization affect compute throughput on Wormhole?
- Which projections (gate/up/down) are most sensitive to quantization?

**Findings:**
`guides/weight_quantization_for_moe_experts/`

---

## Compute Kernel Configuration for MoE
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** DeepSeek-V3 uses COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc, but Qwen MoE doesn't specify compute kernel configs. Need to optimize.
**Questions:**
- What is the performance difference between LoFi, HiFi2, and HiFi4 for MoE expert matmuls?
- How does packer_l1_acc affect throughput for expert computations?
- What is the accuracy trade-off for using math_approx_mode?

**Findings:**
`guides/compute_kernel_configuration_for_moe/`

---

## Expert Weight Memory Layout Optimization
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Current implementation stores expert weights in DRAM with standard interleaved config. DRAM-sharded layouts may improve memory bandwidth.
**Questions:**
- What is the performance gain from DRAM-sharded weight storage?
- How should expert weights be laid out for optimal prefetch patterns?
- What are the tile size constraints for expert weight sharding?

**Findings:**
`guides/expert_weight_memory_layout_optimization/`

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
`guides/paged_sdpa_decode_for_gqa/`

---

## Tracy Profiling and MoE Forward Pass Analysis
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Need op-level breakdown of MoE forward pass to identify bottlenecks and understand where the 16ms gap occurs.
**Questions:**
1. Have you captured a Tracy trace or op-level breakdown of the MoE forward pass?
2. What operations occur between expert dispatch and combine?
3. Does the 16ms gap scale with sequence length?

**Findings:**
`guides/tracy_profiling_and_moe_forward_pass_analysis/`

---

## SiLU Activation Latency Measurement
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Need to understand SiLU activation contribution to overall MoE latency.
**Questions:**
1. What is the current measured latency of the SiLU activation in MoE expert computation?
2. How does SiLU latency compare to the matmul operations?
3. Would fusing SiLU with matmul improve performance?

**Findings:**
`guides/silu_activation_latency_measurement/`

---

## TTNN Device-Level Profiling with Tracy
**Date:** 2026-03-17
**Status:** Completed
**Why Needed:** Need to understand how to use the Tracy profiler to capture device-level op timing for TTNN operations, interpret per-RISC kernel durations, and diagnose whether ops are compute-bound or bandwidth-bound.
**Questions:**
1. How is the Tracy profiler invoked for a TTNN pytest (env vars, CLI flags)?
2. What does each field in the ops_perf_results CSV mean (DEVICE KERNEL DURATION, BRISC/NCRISC/TRISC durations, PM IDEAL, FPU UTIL)?
3. How do you determine if an op is compute-bound vs bandwidth-bound from the profiler output?
4. What are common causes of low FPU utilization and how can they be addressed?
5. How does host dispatch overhead compare to device kernel time for small vs large ops?

**Guide:** `guides/ttnn_device_level_profiling_with_tracy/`

**Findings:**
`guides/ttnn_device_level_profiling_with_tracy/`

---

## TTNN Ops Trace
**Date:** 2026-03-23
**Status:** Completed
**Guide:** guides/ttnn_ops_trace/
**Why Needed:** Need to understand trace capture, command queues, pipelines, applicable async ops, when to use trace when not to use trace. How to know if something needs tracing and how to estimate the improvment that could come from trace.
**Questions:**
1. What is trace capture?
2. What are command queues?
3. What are async ops?
4. How is everything related?
5. How to estimate improvement?

**Guide:** `guides/ttnn_ops_trace/`

**Findings:**
`guides/ttnn_ops_trace/`

---

## Distributed RMS Norm Garbled Output
**Date:** 2026-03-23
**Status:** Completed
**Why Needed:** Model output becomes garbled after adding Distributed RMS norm. Need to investigate potential issues with distributed normalization implementation.
**Questions:**
1. What changes were made to add Distributed RMS norm?
2. How is the reduction handled across distributed devices?
3. Are tensor shapes and memory layouts correct for distributed operation?
4. How is the normalization scale/weight applied in the distributed case?
5. What patterns do other distributed normalization ops follow?

**Findings:**
Root causes identified in `TTNNDistributedRMSNorm`:

1. **Weight shaping is incorrect** - Uses `dims=(None, 2)` with `ShardTensor2dMesh` which doesn't properly distribute weights. Reference implementations use `ShardTensorToMesh` with weight shape `(num_devices, 1, -1, 32)` sharded on dim 0.

2. **Missing cluster_axis parameter** - The `all_gather_async` call is missing `cluster_axis=1`, which could cause gathering on the wrong axis and corrupt statistics.

3. **Missing stats reshape** - Reference implementation in `ccl.py` reshapes stats after `rms_norm_pre_all_gather` to `(1, 1, inp.shape[-2], 32)`. This step is missing in `TTNNDistributedRMSNorm`.

4. **Missing compute_kernel_config** - Reference implementations pass compute kernel config for numerical precision.

See `PLAN_distributed_rmsnorm_fix.md` for detailed fix plan.
