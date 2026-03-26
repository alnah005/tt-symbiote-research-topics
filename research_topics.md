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

---

## TTNNMoE Performance Optimization on T3K
**Date:** 2026-03-26
**Status:** Completed
**Guide:** `guides/ttnn_moe_performance_optimization_on_t3k/`
**Why Needed:** Running MoE on T3K is currently the most time-consuming operation, making it a critical bottleneck to address for overall model throughput.
**Questions:**
- `TTNNMoE.forward` runs all-gather (Linear topology, num_links=1) before routing and reduce-scatter (Ring topology, chunks_per_sync=10, num_workers_per_link=2) after experts — what are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?
- `TTNNExperts.forward` pads tokens to SPARSITY_BLOCK_SIZE=32 then runs `all_to_all_dispatch` → `moe_expert_token_remap` → 3× `sparse_matmul` → `all_to_all_combine` — which of these steps dominates latency at batch=1 decode?
- The `sparse_matmul` program config uses `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1` — are these optimal for the hidden/intermediate sizes in GLM-4-MoE and Bailing, or should they be tuned per model?
- Expert matmuls use `HiFi2` math fidelity while the gate routing linear uses `HiFi4` — is HiFi2 sufficient for expert computation, and would LoFi improve throughput without accuracy loss?
- After `all_to_all_combine`, expert outputs are weighted by broadcasting `topk_experts_weights` to `(hidden_size, 1, 1, 1)` via `ttnn.repeat` then permuting — is this weight application a meaningful overhead, and is there a cheaper alternative (e.g. elementwise after reshape instead of broadcast+permute)?
- `TTNNGlm4MoeMoE` (the older Glm4 path) still runs experts on CPU via `Glm4MoeNaiveMoeHybrid` (the `ttnn = False` flag disables TTNN experts) — how does its latency compare to `TTNNMoE`/`TTNNExperts`, and is there any remaining code path that silently falls back to CPU during inference?
- The router in `TTNNMoERouterDecode` uses a 3-pass BF16 centering trick for precision — what is the latency cost of this routing logic versus a simpler single-pass topk, and is the precision benefit measurable in output quality?
- What is the best way to profile the full `TTNNMoE` forward at op-level granularity on T3K to identify the single biggest bottleneck (Tracy, ttnn op timers, or another tool)?

**Findings:**
`guides/ttnn_moe_performance_optimization_on_t3k/`

---

## TTNNBailingMoEAttention Performance Optimization
**Date:** 2026-03-26
**Status:** Pending
**Why Needed:** `TTNNBailingMoEAttention` is used for every attention layer in the Bailing MoE model and its performance directly impacts overall model throughput. We need to understand where time is spent in the attention forward pass and what the best optimization opportunities are.
**Questions:**
- What are the dominant latency contributors in `TTNNBailingMoEAttention.forward` at batch=1 decode on T3K, and how do collective communication ops compare to compute ops?
- How do the Q, K, and V projection strategies (sharded Q vs replicated K/V) affect decode latency, and is there a more efficient sharding scheme for T3K's 1×8 mesh?
- What is the cost of host-device round-trips in the decode path, and are there fully on-device alternatives for the operations that currently require them?
- How does the memory layout transition sequence in the decode path (moving tensors between DRAM, L1, and various sharded layouts across different ops) affect throughput, and which transitions are avoidable?
- What is the performance impact of QK normalization in decode, and how does it compare to the cost of the projection and attention ops?
- What math fidelity and compute kernel settings are optimal for SDPA in `TTNNBailingMoEAttention` — is the current HiFi4 setting necessary for correctness, and what accuracy/throughput tradeoff does HiFi2 offer?
- For each identified bottleneck in the decode path, what are the best alternative implementations or algorithmic approaches (e.g. fusing ops, reordering ops, different collective communication strategies, or alternative kernel configurations), and what speedup can realistically be expected from each?
- Are there attention implementations in other models in the tt-symbiote or tt-transformers codebase that handle similar GQA configurations more efficiently, and what specific techniques do they use that could be applied to `TTNNBailingMoEAttention`?

**Findings:**
[Pending]

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

## BailingMoE (TTNNBailingMoE / TTNNMoE) Decode Path Optimization
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** The MoE module in Ling-mini-2.0 has multiple performance bottlenecks: CPU fallback in the router sort, 3-pass topk with excessive typecasts, expensive weight application pattern (permute/unsqueeze/repeat), and sequential shared expert execution. Need a detailed optimization plan.
**Questions:**
1. What is the full op sequence and cost breakdown for TTNNMoE.forward() decode path?
2. How can the CPU fallback sort in TTNNMoERouterDecode be eliminated?
3. Can the 3-pass topk centering approach be simplified or eliminated for Ling-mini-2.0 (64 experts, n_group=1)?
4. How can the weight application pattern (permute/unsqueeze/repeat/broadcast multiply/sum) be optimized?
5. Can shared expert computation overlap with routed expert computation?
6. What layout conversions (ROW_MAJOR <-> TILE_LAYOUT) can be eliminated?

**Findings:**
`PLAN_optimize_bailing_moe.md`
Key results: 4-phase plan targeting significant latency reduction.
- Phase 1 (Router): Eliminate CPU sort fallback, simplify 3-pass topk to 1-pass (n_group <= topk_group), eliminate unnecessary f32 typecasts. Saves ~2-4ms.
- Phase 2 (Expert pipeline): Optimize weight application with ttnn.embedding_with_broadcast or in-place weighted accumulation, reduce ROW_MAJOR<->TILE conversions. Saves ~1-2ms.
- Phase 3 (Shared experts): Overlap shared expert MLP with routed expert dispatch/compute using async ops. Saves ~0.5-1ms.
- Phase 4 (Trace capture): Enable TTNN trace for full MoE decode. Saves ~2-3ms.

## TT-Symbiote Wrapper Overhead Analysis (Ling-mini-2.0 Decode)
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Need to quantify exactly where the TorchTTNNTensor wrapper overhead and other host-side overhead occurs in the full Ling-mini-2.0 decode path, to prioritize optimizations.
**Questions:**
1. How many wrapper calls (wrap_to_torch_ttnn_tensor, _set_device_wrap, _unwrap_to_torch) per decode token?
2. What is the total time in wrapper overhead vs actual TTNN compute dispatch?
3. Which modules/layers contribute the most wrapper overhead?
4. What are the root causes (compose_transforms, device sync, output wrapping)?
5. What specific optimizations can reduce each overhead source?

**Findings:**
`REPORT_wrapper_overhead.md`
Key results: Single decode token = 942ms wall clock. 50.5% is overhead (wrapper + Python), 49.5% is TTNN op dispatch. Breakdown:
- Input wrapping (compose_transforms): 158.9ms (16.9%), 594 calls
- Output unwrapping (_unwrap_to_torch + device sync): 76.6ms (8.1%), 84 calls at 42 sync points
- Output wrapping (wrap_to_torch_ttnn_tensor): 60.6ms (6.4%), 297 calls
- Print statements + Python control flow: 165.1ms (17.5%), 363 prints
- Distributed config + aten ops: 13ms (1.4%)
Top optimizations: (1) Eliminate _unwrap_to_torch by doing residual adds in TTNN on-device (-80ms), (2) Remove print statements (-150ms), (3) Fast-path compose_transforms for already-wrapped tensors (-140ms), (4) Flatten module dispatch nesting (-60ms). Combined: ~450ms savings (47%).

## Wrapper Overhead Optimization Plan (5 optimizations)
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** Need detailed implementation plan for 5 wrapper overhead optimizations targeting ~440ms savings (47% of decode latency).
**Questions:**
1. How to create TTNNBailingMoEDecoderLayer to eliminate residual add device syncs?
2. How to gate prints without removing them?
3. How to fast-path compose_transforms for already-wrapped tensors?
4. How to flatten module dispatch nesting in attention/decoder layer?
5. How to optimize tree_map for common argument patterns?

**Findings:**
`PLAN_wrapper_optimizations.md`
Key results: 5-optimization plan in implementation order: P2 (gated prints, ~150ms), P5 (fast tree_map, ~10ms), P3 (fast compose_transforms, ~140ms), P1 (TTNNBailingMoEDecoderLayer with on-device residual adds, ~80ms), P4 (flatten dispatch nesting, ~60ms). Total: ~440ms (47%). P1 creates new TTNNBailingMoEDecoderLayer that replaces BailingMoeV2DecoderLayer, creates child TTNNModules in from_torch, does ttnn.add for residuals. Layer 0 special case (dense MLP, not MoE) handled. P2 uses existing NormalRun.verbose flag. P3 adds isinstance+device fast-path. P4 calls child.forward() directly bypassing module_run. P5 adds fast paths for single tensor, small tuples, shallow dicts.

## Ling-mini-2.0 T3K Hang Root Cause
**Date:** 2026-03-26
**Status:** Completed
**Why Needed:** test_ling_mini_2_0.py hangs on T3K during inference. Need to identify root cause and fix.
**Questions:**
1. Where exactly does the test hang?
2. What operation causes the hang?
3. Is this a known issue with TTNN CCL operations?
4. What is the fix?

**Findings:**
Root cause: `ttnn.all_reduce` with 8 devices on T3K hangs. This is a **known issue** documented in the official T3K CCL test suite at `tests/nightly/t3000/ccl/test_all_reduce.py` line 22: `# (8, 1), # skipped as 8 devices result in hang in all gather`.

The hang occurs in `TTNNLinearIColShardedWAllReduced.forward()` (linear.py:218) which calls `ttnn.all_reduce(tt_output, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1)`. This class is used by `TTNNBailingMoEAttention._forward_prefill()` and `._forward_decode_paged()` for the fused QKV projection (`self.qkv_proj`).

The test hangs during the first `model.generate()` call (warmup at line 125) when the prefill forward pass reaches the first attention layer's QKV projection.

Fix: Replace `ttnn.all_reduce` with composite `reduce_scatter` + `all_gather` (2 ops instead of 1, but doesn't hang), or use the async variants (`reduce_scatter_minimal_async` + `all_gather_async`) with proper semaphore management via `ccl_manager`, matching the pattern used by `TTNNMoE.forward()` which works on T3K.
