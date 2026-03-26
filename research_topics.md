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

## TT Transformers Trace Capture
**Date:** 2026-03-23
**Status:** Completed
**Why Needed:** Need to understand how to add trace capture by default and how adding trace capture with tracy support can be used. In addition, how does model warm-up affect trace capture and tracy support.
**Questions:**
1. How is trace captured in tt-transformers?
2. How is tracy used when run with tt-transformers?
3. How to differentiate warm-up calls from actual calls in tracy?
4. How to differentiate trace captured ops from normal ops in tracy?

**Guide:** `guides/tt_transformers_op_trace/`

**Findings:**
`guides/tt_transformers_op_trace/`


## TT Transformers Key Optimizations
**Date:** 2026-03-23
**Status:** Completed
**Why Needed:** Need to understand the key optimizations done in tt-transformers for LLMs. This includes optimizations for attention, matmul, and other critical kernels. Understanding these optimizations will help in identifying potential areas for further improvement and ensuring that we are leveraging the full capabilities of the hardware.

**Questions:**
1. What are the key optimizations implemented in tt-transformers for attention mechanisms?
2. What matmul optimizations are present in tt-transformers for LLMs?
3. Are there any specific optimizations for memory access patterns in tt-transformers?
4. How do these optimizations impact the overall performance of LLMs running on Tenstorrent hardware?
5. What other optimizations are present in tt-transformers that are critical for LLM performance?

**Findings:**
`guides/tt_transformers_key_optimizations/`

## TT Symbiote
**Date:** 2026-03-23
**Status:** Completed
**Why Needed:** Need to understand what TT Symbiote is, how it works, and how it can be used to optimize LLM performance on Tenstorrent hardware. This includes understanding the architecture of TT Symbiote, the types of optimizations it provides, and how it integrates with existing frameworks like tt-transformers. Additionally, I want to know about any specific use cases or scenarios where TT Symbiote has shown significant performance improvements.

**Questions:**
1. What is TT Symbiote and what are its main features?
2. How does TT Symbiote optimize LLM performance on Tenstorrent hardware?
3. What is the architecture of TT Symbiote and how does it integrate with tt-transformers?
4. Are there any specific use cases or scenarios where TT Symbiote has demonstrated significant performance improvements?
5. How can I get started with using TT Symbiote for optimizing LLM performance?

**Findings:**
`guides/tt_symbiote/`


## TT Transformers Into TT Symbiote
**Date:** 2026-03-23
**Status:** Completed
**Why Needed:** Need to understand how to integrate tt-transformers with TT Symbiote to leverage the optimizations from tt-transformers in tt-symbiote for LLMs running on Tenstorrent hardware. This includes understanding the steps required for integration, any potential challenges or considerations, and the expected performance benefits from using TT Symbiote with tt-transformers. Additionally, I want to know which features would need to be rewritten from scratch in tt-symbiote and which features can be reused from tt-transformers.

**Questions:**
1. What are the steps required to integrate tt-transformers with TT Symbiote?
2. Are there any potential challenges or considerations to be aware of during the integration process?
3. What are the expected performance benefits from using TT Symbiote with tt-transformers for LLMs on Tenstorrent hardware?
4. Which features from tt-transformers would need to be rewritten from scratch in tt-symbiote, and which features can be reused?
5. Are there any specific examples or case studies of successful integration of tt-transformers with TT Symbiote that I can reference?

**Findings:**
`guides/tt_transformers_into_tt_symbiote/`

## BailingAttention Decode Path Profiling Analysis
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Need op-level profiling breakdown of BailingAttention standalone decode to identify bottlenecks and prioritize optimizations.
**Questions:**
1. Which ops dominate decode time?
2. How much time is host overhead vs device compute?
3. Where are the `_to_replicated` host round-trips showing up?
4. How many `all_gather` and `to_memory_config` ops per decode iteration and their total cost?
5. What fraction of time is spent in actual compute (matmul, SDPA) vs data movement?

**Findings:**
`ANALYSIS_bailing_attention_profiling.md`
Key results: Steady-state decode = 9.7ms wall clock. Only 24.5% is sub-module device compute (matmuls, RMSNorm). 70.1% is inline attention ops (all_gather x4, to_memory_config x9, _to_replicated host round-trip, RoPE, paged SDPA). ~35% is host/wrapper overhead. Top optimizations: eliminate _to_replicated (-1-2ms), reduce all_gather 4->1 (-1.5-2ms), trace capture (-2-3ms). Target: 3-5ms per decode with all optimizations.

## Tracy Profiler DRAM Buffer Configuration
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Device profiler DRAM buffer overflowed at ~12,000 markers when profiling 32 decode iterations, dropping markers and producing incomplete data. Need to understand buffer sizing and how to increase it.
**Questions:**
1. What controls the profiler DRAM buffer size?
2. What env vars can increase the buffer?
3. How to get the full ops_perf_results CSV with op names from device profiling?
4. What is the right combination of buffer size increase + iteration reduction?

**Findings:**
`PLAN_profile_bailing_attention_v2.md`
Key results: Buffer size is controlled by `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` env var (default 1000, set to 2000 to double buffer). Buffer overflow caused by 32 decode iterations x ~20 ops/iter x 5 RISCs x 2 markers. Fix: reduce `num_decode_tokens` to 8 AND set `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=2000`. The per-op CSV with op names is `cpp_device_perf_report.csv`, generated when `TT_METAL_PROFILER_CPP_POST_PROCESS=1` is set. Columns include OP NAME, DEVICE KERNEL DURATION, per-RISC durations (BRISC/NCRISC/TRISC), and OP TO OP LATENCY.

## BailingAttention Decode Path Optimization Plan
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Decode path is 72% data-movement-bound. Top bottlenecks: AllGather 28.9%, ReshapeView 20.5%, ReduceScatter 14.1%, Matmul 13.2%. Need concrete optimization plan to reduce from ~9.5ms to 3-5ms target.
**Questions:**
1. How to reduce 4 all_gathers to fewer?
2. How to eliminate unnecessary reshapes (328 per decode)?
3. Can reduce_scatter be combined with all_gather (all_reduce)?
4. What is the optimal QKV projection sharding strategy?
5. How does tt-transformers (LLaMA-70B) handle this?

**Findings:**
`PLAN_optimize_bailing_attention.md`
Key results: 3-phase plan targeting ~6ms savings total.
- Phase 1 (low-hanging fruit): Use nlp_create_qkv_heads_decode, eliminate linear 4D reshapes, pre-cache cos/sin sharded, test 4D QK norm. Saves 0.7-1.0ms.
- Phase 2 (fused QKV): Create TTNNLinearIColShardedWAllReduced, fuse Q+K+V into single matmul + all_reduce. Eliminates 3 all_gathers + 3 matmuls + 1 reduce_scatter, replaces with 1 matmul + 1 all_reduce. Saves 2-3ms.
- Phase 3 (trace capture): Enable TTNN trace for decode path. Saves 2-3ms.
Reference: LLaMA-70B uses fused QKV matmul + nlp_create_qkv_heads_decode + column-parallel sharding. BailingMoE needs all_reduce (not reduce_scatter) due to 4 KV heads < 8 devices.

## Fused QKV + AllReduce Detailed Implementation Plan
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Phase 1 (nlp_create_qkv_heads_decode) REGRESSED -- host dispatch overhead dominates, not device op count. Must reduce 5 CCL ops to 1 by fusing Q/K/V projections into a single column-parallel matmul + all_reduce.
**Questions:**
1. Which CCL pattern to use? (all_reduce vs reduce_scatter+all_gather vs hybrid)
2. How to handle GQA (4 KV heads < 8 devices) with reduce_scatter?
3. Does ttnn.all_reduce work on T3K (1x8 mesh)?
4. What weight preparation changes are needed for fused QKV?
5. Can all_reduce_create_qkv_heads (Galaxy op) be used on T3K?
6. How does bias handling change with all_reduce vs reduce_scatter?

**Findings:**
`PLAN_fused_qkv_allreduce.md`
Key results:
- Use `ttnn.all_reduce` (not reduce_scatter) because KV heads (4) < devices (8) doesn't divide for reduce_scatter
- all_reduce replicates full 3072-dim output on all devices; tiny tensors in decode so bandwidth waste is negligible
- `all_reduce_create_qkv_heads` is Galaxy-only (TG 8x4), not suitable for T3K
- New linear class `TTNNLinearIColShardedWAllReduced`: matmul + all_reduce, bias must be replicated (not sharded)
- Prefill path unchanged (keeps separate Q/K/V projections)
- RISK: T3K test skips 8-device all_reduce due to "hang in all gather"; fallback is composite RS+AG (2 ops, still saves 3)
- Estimated savings: 2.0-2.7ms per decode (4 fewer CCL host dispatches + 2 fewer matmuls)
- Memory cost: +12.6MB/layer (403MB total for 32 layers, 3.4% of T3K DRAM)

## TTNNMoE / TTNNBailingMoE Profiling Plan
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Need to profile TTNNMoE (TTNNBailingMoE) module used in test_ling_mini_2_0.py to understand per-op performance breakdown, identify bottlenecks in the MoE forward pass (all_gather, routing, sparse_matmul experts, all_to_all dispatch/combine, reduce_scatter, shared experts).
**Questions:**
1. What is the TTNNMoE/TTNNBailingMoE architecture and its op composition?
2. How to profile the full MoE forward pass with device-level Tracy profiling?
3. What are the expected bottlenecks (routing, expert compute, CCL ops, host overhead)?
4. How to set up environment and run profiling for test_ling_mini_2_0.py?

**Findings:**
`PLAN_profile_ttnn_moe.md`
Key results: TTNNBailingMoE inherits from TTNNMoE. Forward pass: all_gather_async -> float32 gate matmul -> TTNNMoERouterDecode (3-pass topk routing with 30+ TTNN ops) -> TTNNExperts (all_to_all_dispatch, 3x sparse_matmul, silu, all_to_all_combine, weight application) -> reduce_scatter_minimal_async -> shared_experts (3 matmuls + silu). Two profiling methods: (1) DispatchManager host-level timing (built into test), (2) Device-level Tracy profiling with TT_METAL_DEVICE_PROFILER=1 + TT_METAL_PROFILER_CPP_POST_PROCESS=1. Buffer overflow likely with 128 decode tokens; reduce to 8-16 and set TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000. Expected bottlenecks: routing overhead (many small ops), CCL ops (all_gather + reduce_scatter), sparse_matmul efficiency.
