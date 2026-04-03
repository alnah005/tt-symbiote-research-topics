# Research Topics

This file tracks research topics that the Architect needs to investigate for making informed decisions.

---

## Format

```
## [Topic Name]
**Date:** YYYY-MM-DD
**Status:** Pending | In Progress | Completed
**Guide:** `path/to/guide/`
**Why Needed:** [Reason this research is necessary]
**Questions:**
- Question 1
- Question 2

---
```

## Topics

---

## MoE Optimization Techniques for TTNN
**Date:** 2026-03-16
**Status:** Completed
**Guide:** `guides/moe_optimization_techniques_for_ttnn/`
**Why Needed:** Need to understand best practices for optimizing Mixture of Experts models on Tenstorrent hardware, specifically comparing batched matmul vs sparse_matmul approaches.
**Questions:**
- What are the performance characteristics of sparse_matmul vs batched matmul for MoE?
- How should sparsity tensors be constructed for optimal performance?
- What program configs are recommended for different batch/sequence sizes?

---

## T3K Mesh Device Optimizations
**Date:** 2026-03-16
**Status:** Completed
**Guide:** `guides/t3k_mesh_device_optimizations/`
**Why Needed:** TTNNQwen3MoE runs on T3K (1x8 mesh) and needs device-specific optimizations for expert parallelism.
**Questions:**
- What are the optimal num_links settings for all_to_all operations on T3K?
- How should memory configs (L1 vs DRAM) be chosen for decode vs prefill?
- What are the bandwidth characteristics between T3K devices?

---

## Expert Parallelism Strategies
**Date:** 2026-03-16
**Status:** Completed
**Guide:** `guides/expert_parallelism_strategies/`
**Why Needed:** Qwen3.5-35B has 256 experts with top-8 routing. Need optimal dispatch/combine strategies.
**Questions:**
- How does all_to_all_dispatch/combine compare to alternative expert routing schemes?
- What is the optimal expert-to-device assignment for 256 experts on 8 devices?
- How should routing weights be processed to minimize overhead?

---

## Weight Quantization for MoE Experts
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/weight_quantization_for_moe_experts/`
**Why Needed:** DeepSeek-V3 uses bfloat4_b/bfloat8_b weight quantization for experts, but Qwen uses full bfloat16. Need to evaluate quantization trade-offs.
**Questions:**
- What accuracy loss is expected from bfloat4_b vs bfloat8_b vs bfloat16 for expert weights?
- How does weight quantization affect compute throughput on Wormhole?
- Which projections (gate/up/down) are most sensitive to quantization?

---

## Compute Kernel Configuration for MoE
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/compute_kernel_configuration_for_moe/`
**Why Needed:** DeepSeek-V3 uses COMPUTE_KERNEL_CONFIG_LOFI with packer_l1_acc, but Qwen MoE doesn't specify compute kernel configs. Need to optimize.
**Questions:**
- What is the performance difference between LoFi, HiFi2, and HiFi4 for MoE expert matmuls?
- How does packer_l1_acc affect throughput for expert computations?
- What is the accuracy trade-off for using math_approx_mode?

---

## Expert Weight Memory Layout Optimization
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/expert_weight_memory_layout_optimization/`
**Why Needed:** Current implementation stores expert weights in DRAM with standard interleaved config. DRAM-sharded layouts may improve memory bandwidth.
**Questions:**
- What is the performance gain from DRAM-sharded weight storage?
- How should expert weights be laid out for optimal prefetch patterns?
- What are the tile size constraints for expert weight sharding?

---

## Paged SDPA Decode for GQA
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/paged_sdpa_decode_for_gqa/`
**Why Needed:** Ling model generates incorrect text during decode. Need to understand paged_sdpa_decode kernel expectations for GQA with 4 KV heads and 16 Q heads.
**Questions:**
- What does the paged_sdpa_decode kernel expect for GQA (4 KV heads to 16 Q heads)?
- Is there a mismatch in how cur_pos is interpreted?
- Are there any known issues with TTNN paged attention?

---

## Tracy Profiling and MoE Forward Pass Analysis
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/tracy_profiling_and_moe_forward_pass_analysis/`
**Why Needed:** Need op-level breakdown of MoE forward pass to identify bottlenecks and understand where the 16ms gap occurs.
**Questions:**
- Have you captured a Tracy trace or op-level breakdown of the MoE forward pass?
- What operations occur between expert dispatch and combine?
- Does the 16ms gap scale with sequence length?

---

## SiLU Activation Latency Measurement
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/silu_activation_latency_measurement/`
**Why Needed:** Need to understand SiLU activation contribution to overall MoE latency.
**Questions:**
- What is the current measured latency of the SiLU activation in MoE expert computation?
- How does SiLU latency compare to the matmul operations?
- Would fusing SiLU with matmul improve performance?

---

## TTNN Device-Level Profiling with Tracy
**Date:** 2026-03-17
**Status:** Completed
**Guide:** `guides/ttnn_device_level_profiling_with_tracy/`
**Why Needed:** Need to understand how to use the Tracy profiler to capture device-level op timing for TTNN operations, interpret per-RISC kernel durations, and diagnose whether ops are compute-bound or bandwidth-bound.
**Questions:**
- How is the Tracy profiler invoked for a TTNN pytest (env vars, CLI flags)?
- What does each field in the ops_perf_results CSV mean (DEVICE KERNEL DURATION, BRISC/NCRISC/TRISC durations, PM IDEAL, FPU UTIL)?
- How do you determine if an op is compute-bound vs bandwidth-bound from the profiler output?
- What are common causes of low FPU utilization and how can they be addressed?
- How does host dispatch overhead compare to device kernel time for small vs large ops?

---

## TTNN Ops Trace
**Date:** 2026-03-23
**Status:** Completed
**Guide:** `guides/ttnn_ops_trace/`
**Why Needed:** Need to understand trace capture, command queues, pipelines, applicable async ops, when to use trace and when not to use trace, and how to estimate the improvement that could come from tracing.
**Questions:**
- What is trace capture?
- What are command queues?
- What are async ops?
- How is everything related?
- How do you estimate the improvement from adding trace?

---

## TT Transformers Trace Capture
**Date:** 2026-03-23
**Status:** Completed
**Guide:** `guides/tt_transformers_op_trace/`
**Why Needed:** Need to understand how to add trace capture by default and how adding trace capture with Tracy support can be used. In addition, how does model warm-up affect trace capture and Tracy support.
**Questions:**
- How is trace captured in tt-transformers?
- How is Tracy used when run with tt-transformers?
- How to differentiate warm-up calls from actual calls in Tracy?
- How to differentiate trace-captured ops from normal ops in Tracy?

---

## TT Transformers Key Optimizations
**Date:** 2026-03-23
**Status:** Completed
**Guide:** `guides/tt_transformers_key_optimizations/`
**Why Needed:** Need to understand the key optimizations done in tt-transformers for LLMs, including attention, matmul, and other critical kernels. Understanding these optimizations will help identify areas for further improvement and ensure the full capabilities of the hardware are leveraged.
**Questions:**
- What are the key optimizations implemented in tt-transformers for attention mechanisms?
- What matmul optimizations are present in tt-transformers for LLMs?
- Are there any specific optimizations for memory access patterns in tt-transformers?
- How do these optimizations impact the overall performance of LLMs running on Tenstorrent hardware?
- What other optimizations are present in tt-transformers that are critical for LLM performance?

---

## TT Symbiote
**Date:** 2026-03-23
**Status:** Completed
**Guide:** `guides/tt_symbiote/`
**Why Needed:** Need to understand what TT Symbiote is, how it works, and how it can be used to optimize LLM performance on Tenstorrent hardware. This includes understanding the architecture, the types of optimizations it provides, and how it integrates with existing frameworks like tt-transformers.
**Questions:**
- What is TT Symbiote and what are its main features?
- How does TT Symbiote optimize LLM performance on Tenstorrent hardware?
- What is the architecture of TT Symbiote and how does it integrate with tt-transformers?
- Are there any specific use cases or scenarios where TT Symbiote has demonstrated significant performance improvements?
- How can I get started with using TT Symbiote for optimizing LLM performance?

---

## TT Transformers Into TT Symbiote
**Date:** 2026-03-23
**Status:** Completed
**Guide:** `guides/tt_transformers_into_tt_symbiote/`
**Why Needed:** Need to understand how to integrate tt-transformers with TT Symbiote to leverage tt-transformers optimizations in tt-symbiote for LLMs on Tenstorrent hardware. This includes the steps required for integration, potential challenges, expected performance benefits, and which features can be reused vs rewritten from scratch.
**Questions:**
- What are the steps required to integrate tt-transformers with TT Symbiote?
- Are there any potential challenges or considerations to be aware of during the integration process?
- What are the expected performance benefits from using TT Symbiote with tt-transformers for LLMs on Tenstorrent hardware?
- Which features from tt-transformers would need to be rewritten from scratch in tt-symbiote, and which features can be reused?
- Are there any specific examples or case studies of successful integration of tt-transformers with TT Symbiote?

---

## TTNNBailingMoEAttention Performance Optimization on T3K
**Date:** 2026-03-26
**Status:** Completed
**Guide:** `guides/ttnn_bailing_moe_attention_performance_optimization_on_t3k/`
**Why Needed:** `TTNNBailingMoEAttention` is the attention layer for the Ling (BailingMoeV2) model on T3K and contains several performance-sensitive paths — fused QKV projection, paged SDPA decode, HEIGHT_SHARDED RoPE, and a host-roundtrip tensor replication step — whose combined latency contribution is not yet understood.
**Questions:**
- `TTNNBailingMoEAttention` uses a fused QKV projection (`TTNNLinearIColShardedWAllReduced`: 1 matmul + 1 all_reduce) replacing 3 separate matmuls + 5 CCL ops — what are the actual latency savings on T3K's 1×8 mesh, and is `num_links=1` in `_maybe_all_gather` optimal for the hidden size?
- `_to_replicated` round-trips the all-gathered QKV tensor through host CPU to satisfy paged-attention kernel topology requirements — what is the host-transfer overhead at decode batch=1, and is there a device-side alternative?
- The decode path places Q, K, V into HEIGHT_SHARDED L1 before RoPE then re-shards K/V again before `paged_update_on_device` — how many memory-config transitions occur per decode step, and which dominates overhead?
- `paged_sdpa_decode` is invoked with `q_chunk_size=0, k_chunk_size=0` — what does chunk size 0 mean for the paged SDPA kernel, and are these correct for the Ling model's GQA configuration (16 Q heads, 4 KV heads, head_dim=128)?
- The SDPA compute kernel uses `HiFi4` with `fp32_dest_acc_en=True` and `packer_l1_acc=True` — is HiFi4 with fp32 accumulation necessary for attention correctness, or would HiFi2 improve throughput without measurable accuracy loss?
- When `use_qk_norm=True`, Q and K are moved to L1, reshaped, normalized via `TTNNRMSNorm`, then reshaped back — what is the latency of this QK norm path relative to the fused QKV matmul, and is the L1 move avoidable?
- `partial_rotary_factor < 1.0` forces `TTNNRotaryPositionEmbedding` (non-distributed) — what is the performance cost of non-distributed RoPE on T3K, and is there a way to use the distributed kernel with partial rotary without padding cos/sin to full head_dim?
- What is the best way to profile the full `TTNNBailingMoEAttention` forward at op-level granularity on T3K to identify the single biggest decode bottleneck?

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
- After `all_to_all_combine`, expert outputs are weighted by broadcasting `topk_experts_weights` then permuting — is this weight application a meaningful overhead, and is there a cheaper alternative?
- `TTNNGlm4MoeMoE` still runs experts on CPU via `Glm4MoeNaiveMoeHybrid` — how does its latency compare to `TTNNMoE`/`TTNNExperts`, and is there any remaining code path that silently falls back to CPU during inference?
- The router in `TTNNMoERouterDecode` uses a 3-pass BF16 centering trick for precision — what is the latency cost versus a simpler single-pass topk, and is the precision benefit measurable in output quality?
- What is the best way to profile the full `TTNNMoE` forward at op-level granularity on T3K to identify the single biggest bottleneck?

---

## Deploying a TT Symbiote Model on tt-inference-server
**Date:** 2026-03-27
**Status:** Completed
**Guide:** `guides/deploying_a_tt_symbiote_model_on_tt_inference_server/`
**Why Needed:** Need to understand what is required to serve a TT Symbiote model through the tt-inference-server infrastructure, including the interface contracts, serving stack, and any integration work needed to make a tt-symbiote model a first-class citizen in that deployment pipeline.
**Questions:**
- What is the high-level architecture of tt-inference-server — what serving framework does it use (vLLM, custom, other), and what is the entry point for adding a new model backend?
- What interface or protocol must a model implementation satisfy to be loaded by tt-inference-server (e.g. a specific Python class, method signatures, config schema)?
- How are model weights discovered and loaded — does tt-inference-server expect HuggingFace checkpoints, a custom format, or does it delegate to the model implementation?
- How does tt-inference-server handle tokenization — is it bundled, delegated to HuggingFace, or must the model provide its own tokenizer path?
- What changes, if any, are needed inside the TT Symbiote model code (e.g. forward signature, KV cache management, batch/sequence length constraints) to match what tt-inference-server expects?
- How is hardware initialization handled — does tt-inference-server own device setup, or must the model bring its own mesh/device context?
- What configuration files or environment variables control model selection, device mapping, and serving parameters (port, max batch, max sequence length)?
- Are there existing examples of non-HuggingFace or custom TTNN models already integrated into tt-inference-server that can serve as a reference implementation?

---

## Async CCL Semaphore Behavior Under Trace Replay
**Date:** 2026-03-27
**Status:** Completed
**Guide:** `guides/async_ccl_semaphore_behavior_under_trace_replay/`
**Why Needed:** `TT_CCL.get_and_cycle_*` methods cycle through 2 double-buffered semaphore handles using a host-side modular counter. During trace capture, a specific semaphore handle is baked into the command buffer. On replay, the host counter continues cycling but the trace always uses the handle from capture time. Need to understand the exact interaction to enable tracing for modules that use async CCL ops (reduce_scatter_minimal_async, all_gather_async).
**Questions:**
- Are semaphore handles stored as kernel arguments (baked into trace) or as buffer addresses that can be updated before replay?
- Can `ttnn.experimental.reduce_scatter_minimal_async` and `ttnn.all_gather_async` be used inside a trace capture at all?
- Does tt-transformers have any existing patterns for tracing models that use async CCLs with cycling semaphores?
- What is the correct way to synchronize host-side semaphore cycling state with trace replay boundaries?
- Would resetting semaphore indices before each replay (to match capture-time state) be sufficient, or are there device-side semaphore states that also need resetting?


---

## ttnn.all_reduce Trace Compatibility
**Date:** 2026-03-27
**Status:** Completed
**Guide:** `guides/ttnn_all_reduce_trace_compatibility/`
**Why Needed:** `TTNNLinearIColShardedWAllReduced` uses synchronous `ttnn.all_reduce` (no cycling semaphores). Need to confirm this multi-device collective operation is trace-compatible since it will be used inside traced linear layer execution.
**Questions:**
- Is `ttnn.all_reduce` (synchronous, Ring topology) compatible with trace capture and replay?
- Does `ttnn.all_reduce` use any internal semaphore state that could conflict with trace replay?
- Are there any known limitations or requirements for using `ttnn.all_reduce` inside a traced region?


---

## Gated Delta Net and Gated Attention on T3K
**Date:** 2026-03-27
**Status:** Completed
**Guide:** `guides/gated_delta_net_and_gated_attention_on_t3k/`
**Why Needed:** Qwen-Coder-Next and Qwen3.5 models introduce Gated Delta Net (a linear-recurrent attention variant) alongside standard Gated Attention. Understanding the mathematical foundations and compute/memory characteristics of these operations is required before mapping them onto the T3K 1×8 mesh.
**Questions:**
- What is the Gated Delta Net mechanism — what are the core mathematical operations (delta rule update, gating, state matrix), and how does it differ from standard softmax attention and other linear attention variants (e.g. RetNet, Mamba, GLA)?
- What is the Gated Attention mechanism used in these models — how does the gating interact with the standard QKV projection and SDPA, and what tensor shapes does it introduce relative to vanilla multi-head attention?
- What are the data dependencies and recurrence structure in Gated Delta Net — is the state update strictly sequential per token, or can it be parallelized across the sequence dimension (e.g. via parallel scan)?
- What TTNN primitive operations would be needed to implement a single Gated Delta Net step for decode (batch=1, single token) and for prefill (full sequence), and what are the expected tensor shapes at each step?
- How does the hidden state size and gating dimensionality of Gated Delta Net compare to the KV cache size of standard attention for the same model — what are the memory footprint implications on T3K L1 and DRAM?
- For the recurrent decode step of Gated Delta Net, is the bottleneck compute-bound (state matrix multiply) or bandwidth-bound (state read/write), and how does this map to Wormhole's compute-to-bandwidth ratio?
- How should the Gated Delta Net state matrix be sharded across the 8 devices of T3K — what parallelism strategy minimizes CCL overhead while keeping per-device memory within budget?
- Are there existing TTNN kernels or tt-transformers primitives that can express the Gated Delta Net and Gated Attention forward passes, or are new custom kernels required?

---

## Windowed Attention: Foundations and T3K Mapping
**Date:** 2026-03-27
**Status:** Completed
**Guide:** `guides/windowed_attention_foundations_and_t3k_mapping/`
**Why Needed:** Some models (e.g. Qwen3.5, Mistral) use windowed (sliding window) attention to bound KV cache size and reduce attention complexity. Understanding the mathematical foundations and compute/memory characteristics is required before mapping windowed attention onto the T3K 1×8 mesh.
**Questions:**
- What is windowed (sliding window) attention — what are the core mathematical operations, what is the window size parameter, and how does it differ from full causal attention in terms of which tokens each query attends to?
- How does windowed attention interact with KV cache management during decode — does each new token evict old KV entries, and what is the resulting KV cache size relative to full attention?
- What are the data dependencies and memory access patterns in windowed attention during prefill vs decode — can the window be expressed as a masked full-attention kernel, or does it require a specialized kernel?
- What TTNN primitive operations would be needed to implement windowed attention for decode and prefill, and what tensor shapes does the window constraint introduce?
- How does windowed attention interact with paged KV cache implementations — can paged_sdpa_decode be used with a window constraint, or does the paging scheme need to be aware of the window boundary?
- For the T3K 1×8 mesh, how should the windowed KV cache be sharded across devices — is the window applied per-device or is the full window replicated, and what are the CCL implications for cross-device attention?
- Is windowed attention compute-bound or bandwidth-bound on Wormhole for typical window sizes (e.g. 4096, 8192 tokens) and batch=1 decode, and how does this compare to full attention at the same sequence position?
- Are there existing TTNN kernels or tt-transformers primitives that already support windowed or masked attention patterns, or would a new kernel/program config be required?

---

## Qwen3.5-27B Optimizations on Tenstorrent P150x4
**Date:** 2026-03-28
**Status:** Completed
**Guide:** `guides/qwen35_27b_optimizations_on_tenstorrent_p150x4/`
**Why Needed:** Need to understand the full optimization stack for deploying Qwen3.5-27B on the P150x4 (4-chip Blackhole), covering TP=4 sharding, DRAM-sharded decode, the custom fused GDN recurrence kernel, a 5.3x TTFT speedup, and L1 state management.
**Questions:**
- How is the hybrid 48-GDN + 16-attention architecture mapped onto 4 Blackhole chips with TP=4 column/row parallel sharding?
- What are the full attention layer optimizations: partial RoPE (64/256 dims), QK L2 norms, sigmoid gating, DRAM-sharded decode, and flash SDPA prefill?
- How does the GDN decode pipeline work: conv1d 4-tap shift register, DeltaNet recurrence, and output gating?
- How does the custom `gdn_full_fused_inplace` kernel fuse L2 norm, gates, and recurrence into a single dispatch with batched NOC reads?
- How was the 5.3x TTFT speedup (498 ms/tok → 94 ms/tok) achieved via batched projections, flash attention, and B=1 GDN prefill?
- What is the plan for moving GDN recurrence states from DRAM to L1 using a rolling window of 3 layers, and what is the SDPA CB conflict?
- What does the performance analysis show (14.6 tok/s decode, GDN = 85% of decode time, DRAM bandwidth bottleneck)?

---

## Qwen3.5 Implementation
**Date:** 2026-04-01
**Status:** Completed
**Guide:** `guides/qwen35_implementation/`
**Why Needed:** Need a comprehensive reference covering all modules and constraints of the Qwen3.5-35B-A3B and Qwen3.5-27B implementations on Blackhole P100A, from model architecture to measured decode performance, to inform further optimization work.
**Questions:**
- How does the Gated DeltaNet five-step recurrence work, and what is the Blackhole SrcB TF32 constraint that forces the fused kernel path?
- How is partial RoPE (rotary_dim=64) implemented in GatedAttention layers and what is the cos/sin patching fix?
- How does `DeltaNetDecoderBlock` dispatch uniformly across DeltaNet and GatedAttention layer types?
- How is MoE routing and expert dispatch structured (256 routed + 1 shared expert, top-8 host routing, bfp4 weights, 15.7 GiB DRAM)?
- What is the HF→meta weight conversion pipeline and how are MoE keys protected during conversion?
- What is the 86 ms/token latency breakdown and what are the dominant bottlenecks?
- What is the optimization roadmap (Metal Trace, Multi-CQ, per-row MoE routing)?

---

## Gemma 4 31B Architecture and TTNN Module Mapping
**Date:** 2026-04-03
**Status:** Completed
**Guide:** `guides/gemma_4_31b_architecture_and_ttnn_module_mapping/`
**Why Needed:** Need a definitive mapping of every Gemma 4 submodule to its optimal TTNN implementation, including the heterogeneous attention (sliding vs global), K=V sharing, partial rotary, and V-norm patterns.
**Questions:**
- How should the two different attention configurations (sliding: 32Q/16KV/256dim vs global: 32Q/4KV/512dim) be handled in a single TTNNModule, or should they be separate classes?
- Does `ttnn.paged_sdpa_decode` support sliding window attention natively, or must the KV cache be manually truncated?
- Can `TTNNDistributedRMSNorm` handle the `with_scale=False` variant used for V-norm?
- What is the optimal tensor-parallel sharding strategy for Gemma4-31B on T3K given the two different KV head counts (16 for sliding, 4 for global)?

---

## Gemma 4 Vision Encoder TTNN Porting Strategy
**Date:** 2026-04-03
**Status:** Completed
**Guide:** `guides/gemma_4_vision_encoder_ttnn_porting_strategy/`
**Why Needed:** Determine whether to port the Gemma4 vision encoder to TTNN or run it on CPU, and if porting, how much of the existing Gemma3 TTNN vision encoder can be reused.
**Questions:**
- How different is Gemma4VisionModel from Gemma3's SigLIP vision encoder, and can the existing `models/demos/multimodal/gemma3/tt/` modules be reused directly?
- What is the latency of the Gemma4 vision encoder on CPU vs. the expected latency on TTNN?
- Does the vision encoder's 2D factored RoPE (theta=100) pose any issues for existing TTNN RoPE implementations?

---
