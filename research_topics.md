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

## Gemma 4 HuggingFace Transformers Implementation Deep Dive
**Date:** 2026-04-05
**Status:** Completed
**Guide:** `guides/gemma_4_huggingface_transformers_implementation_deep_dive/`
**Why Needed:** Need a comprehensive understanding of every file in the HuggingFace `transformers/models/gemma4/` package and every PyTorch module in the model, covering the full multimodal architecture (text, vision, audio, video), configuration hierarchy, preprocessing pipelines, and the modular inheritance structure from Gemma3/Gemma3n/Mixtral/Llama.
**Questions:**
- What does each file in `transformers/models/gemma4/` do — `__init__.py`, `configuration_gemma4.py`, `convert_gemma4_weights.py`, `feature_extraction_gemma4.py`, `image_processing_gemma4.py`, `image_processing_pil_gemma4.py`, `modeling_gemma4.py`, `modular_gemma4.py`, `processing_gemma4.py`, `video_processing_gemma4.py` — and how do they relate to each other?
- What is the configuration hierarchy (`Gemma4Config`, `Gemma4TextConfig`, `Gemma4VisionConfig`, `Gemma4AudioConfig`) and what key parameters does each define (sliding window size, layer types, head counts, MoE settings, RoPE parameters)?
- What is the complete PyTorch module tree of `Gemma4ForConditionalGeneration` — what are all nn.Module subclasses, their parent classes (Gemma3, Gemma3n, Mixtral, Llama inheritance), and what does each module compute?
- How does the text decoder work — `Gemma4TextModel`, `Gemma4TextDecoderLayer`, `Gemma4TextAttention` (sliding vs global attention with different head dims and KV head counts), `Gemma4TextMLP`, `Gemma4TextExperts`/`Gemma4TextRouter` (MoE), `Gemma4TextScaledWordEmbedding` (per-layer input embeddings)?
- How does the vision encoder work — `Gemma4VisionModel`, `Gemma4VisionEncoder`, `Gemma4VisionEncoderLayer`, `Gemma4VisionAttention`, `Gemma4VisionMLP`, `Gemma4VisionPatchEmbedder`, `Gemma4VisionPooler`, `Gemma4VisionRotaryEmbedding` (2D factored RoPE)?
- How does the audio encoder work — `Gemma4AudioModel`, `Gemma4AudioLayer`, `Gemma4AudioAttention`, `Gemma4AudioSubSampleConvProjection`, `Gemma4AudioFeedForward`, `Gemma4AudioCausalConv1d`, `Gemma4AudioLightConv1d`, `Gemma4AudioRelPositionalEncoding`?
- How does `Gemma4MultimodalEmbedder` merge vision, audio, and text token embeddings into a unified sequence for the text decoder?
- How do the preprocessing pipelines work — `Gemma4Processor` (orchestrator), `Gemma4ImageProcessor` (torchvision-based), `Gemma4PilImageProcessor` (PIL-based), `Gemma4VideoProcessor`, `Gemma4AudioFeatureExtractor` — and what transformations does each apply?
- What is the relationship between `modular_gemma4.py` and `modeling_gemma4.py` — how does the modular file use inheritance from Gemma3/Gemma3n/Mixtral/Llama base classes while `modeling_gemma4.py` is the auto-generated flattened version?
- How does `convert_gemma4_weights.py` convert from Google's Orbax checkpoint format to HuggingFace safetensors, and what weight mapping is applied?

---

## TT-Lang Architecture and TT-Symbiote Integration Strategy
**Date:** 2026-04-09
**Status:** Completed
**Guide:** `guides/tt_lang_architecture_and_tt_symbiote_integration_strategy/`
**Why Needed:** TT-Lang is a Python DSL and MLIR-based compiler for authoring custom high-performance kernels on Tenstorrent hardware, sitting between high-level TTNN ops and low-level TT-Metalium. TT-Symbiote is a PyTorch-to-TTNN acceleration framework that transparently replaces PyTorch modules with TTNN-optimized equivalents. Understanding TT-Lang's full architecture — its programming model, compilation pipeline, simulator, and performance tools — is essential for evaluating how it can complement TT-Symbiote by providing fused custom kernels, reducing module boilerplate, and enabling hardware-aware optimizations that TTNN's op library alone cannot express. The TT-Lang source is at `/localdev/salnahari/testing_dir/tt-lang` and TT-Symbiote is at `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/`.
**Questions:**
- What is TT-Lang's programming model — how do `@ttl.operation`, `@ttl.compute`, and `@ttl.datamovement` decorators compose, what are Dataflow Buffers (DFBs) and TensorBlocks, and how does the multi-node grid execution model work?
- What is TT-Lang's compilation pipeline — how does Python DSL lower through TTL MLIR → Compute dialect → TTKernel dialect → C++ codegen → JIT compilation, and what optimization passes exist at each stage?
- How does TT-Lang's functional simulator work — what can be validated without hardware, how does the DFB state machine enforce correctness, and what are the simulator's limitations vs. on-device execution?
- What performance analysis tools does TT-Lang provide — how do TTLANG_PERF_DUMP, TTLANG_AUTO_PROFILE, signpost profiling, and Perfetto trace integration work, and what metrics do they expose?
- What are TT-Symbiote's key architectural pain points that TT-Lang could address — specifically the boilerplate in TTNNModule lifecycle methods, the manual weight management pipeline, the 100+ hand-written ATen dispatch handlers, and the lack of custom kernel integration?
- How can TT-Lang fused kernels be integrated as drop-in replacements for TTNN ops inside TT-Symbiote modules — what is the interface contract (ttnn.Tensor in/out), how does JIT compilation interact with TT-Symbiote's weight preprocessing and device placement, and what changes to TTNNModule.forward() are needed?
- Which TT-Symbiote operations are the highest-value targets for TT-Lang kernel fusion — specifically the MoE expert dispatch/combine pipeline (sparse_matmul + all_to_all), fused attention variants (QKV projection + RoPE + SDPA), and fused activation patterns (Linear + SiLU/GELU)?
- Can TT-Lang's grid and DFB abstractions simplify TT-Symbiote's multi-device distribution code — replacing manual ShardTensor2dMesh configuration, all-gather/reduce-scatter coordination in TTNNDistributedRMSNorm, and the ad-hoc topology selection logic?
- What is the developer workflow for writing a TT-Lang kernel that replaces an existing TT-Symbiote TTNN op — from simulator validation through on-device profiling to production integration — and how does this compare to the current workflow of tuning TTNN program configs?

---

## Qwen3.6-35B-A3B Architecture and Innovations
**Date:** 2026-04-21
**Status:** Completed
**Guide:** `guides/qwen36_35b_a3b_architecture_and_innovations/`
**Why Needed:** Qwen3.6-35B-A3B is the latest release in the Qwen series and a direct successor to Qwen3.5-35B-A3B. Understanding its architecture, key innovations, and differences from Qwen3.5 is essential for evaluating whether the existing Qwen3.5 TTNN implementation needs changes, what new optimizations are possible, and how the model's hybrid Gated DeltaNet + MoE architecture has evolved. The model uses the same `Qwen3_5MoeForConditionalGeneration` architecture class but introduces post-training improvements focused on agentic coding and thinking preservation.
**Questions:**
- What is the complete architecture of Qwen3.6-35B-A3B — the hybrid 10×(3×Gated DeltaNet + 1×Gated Attention) layout, 256-expert MoE with top-8+1 shared routing, and how do these components interact in a single forward pass?
- How does Gated DeltaNet work in detail — what are the mathematical operations (delta rule state update, gating mechanism, conv1d local mixing), what are the QK/V head asymmetries (16 QK heads vs 32 V heads, 128-dim), and how does this compare to standard linear attention variants (RetNet, GLA, Mamba)?
- What are the exact architectural differences between Qwen3.6-35B-A3B and Qwen3.5-35B-A3B at the config/weight level — are there any changes to layer count, hidden dimensions, expert configuration, attention parameters, or vision encoder, or is the architecture identical with only post-training differences?
- What is the "Thinking Preservation" feature introduced in Qwen3.6 — how does retaining reasoning context from historical messages work mechanically, does it require architectural changes or is it purely a prompting/inference-time technique, and what are the implications for KV cache management?
- How does the partial rotary embedding (25% of head_dim = 64 dims with RoPE, 75% without) work in the Gated Attention layers, and what is the motivation for applying RoPE to only a quarter of the dimensions compared to full rotary in standard transformers?
- What is the M-RoPE (multimodal rotary position embedding) scheme with sections [11, 11, 10] — how does it encode spatial and temporal positions for vision/video tokens, and how does it interact with the text-only RoPE during mixed-modality inference?
- What are the key benchmark improvements from Qwen3.5 to Qwen3.6, specifically in agentic coding (SWE-bench, Terminal-Bench, SkillsBench, NL2Repo), and what post-training techniques (RL, data, scaffolding) drove these improvements?
- How does the Multi-Token Prediction (MTP) training objective work in Qwen3.6 — what is the `mtp_num_hidden_layers=1` configuration, how does it relate to speculative decoding at inference time, and what accuracy/throughput tradeoffs does it introduce?
- How does Qwen3.6's MoE configuration (256 experts, 8 routed + 1 shared, intermediate dim 512) compare to other recent MoE models (DeepSeek-V3, Gemma4-26B-A4B), and what are the implications of using many small experts vs fewer large experts for hardware utilization on accelerators?
- What are the vision encoder specifications — the 27-layer ViT with patch size 16, spatial merge 2, temporal patch 2 — and how does it compare to the Qwen3.5 vision encoder and other recent multimodal model vision encoders (Gemma4, LLaVA)?

---

## TT-Lang Full pip install Support
**Date:** 2026-04-09
**Status:** Completed
**Guide:** `guides/tt_lang_full_pip_install_support/`
**Why Needed:** TT-Lang currently requires a multi-step CMake-driven build via `scripts/build-and-install.sh` that manually orchestrates LLVM/MLIR toolchain compilation, tt-metal/tt-mlir submodule builds, nanobind C++ extension compilation, and Python package installation into a custom venv. Need to understand how to make the full project (compiler tier + DSL/runtime tier) installable via standard `pip install .` so that developers can use familiar Python packaging workflows. The TT-Lang source is at `/localdev/salnahari/testing_dir/tt-lang`.
**Questions:**
- What is the current end-to-end build and installation flow — how does `scripts/build-and-install.sh` orchestrate CMake configuration, LLVM/tt-mlir/tt-metal dependency builds, nanobind extension compilation, and Python package installation, and what are the implicit environment assumptions (env vars, paths, pre-installed tools)?
- How does the existing `pyproject.toml` + `python/setup.py` CMakeBuild integration work — what does the custom `CMakeBuild` class do, what CMake variables does it set, and what prevents `pip install .` from working out of the box today?
- What are the C++ extension build dependencies — what specific LLVM/MLIR libraries, tt-mlir artifacts, and tt-metal headers/libraries must be available before nanobind modules (`_ttlang`, `_ttmlir`) can compile, and how are they currently discovered?
- How do other MLIR-based Python projects (e.g., torch-mlir, triton, IREE) handle `pip install` with heavy C++ dependencies — do they use pre-built wheels, bundled toolchains, scikit-build-core, or other approaches?
- What changes to `pyproject.toml`, `setup.py`, and `CMakeLists.txt` are needed to support `pip install .` assuming the LLVM/tt-mlir/tt-metal toolchain is pre-built and installed at a known location (e.g., via `TTLANG_TOOLCHAIN_DIR` env var)?
- Can the build be split into a two-phase approach — a toolchain wheel (`ttl-toolchain`) containing pre-built LLVM/MLIR/tt-metal shared libraries, and a main wheel (`ttl`) that depends on it and only compiles the nanobind extensions + pure Python packages?
- What are the packaging constraints for the compiled nanobind extensions — how should `.so` files be bundled in the wheel, what RPATHs or `auditwheel` fixes are needed, and how should MLIR dialect Python bindings be included?
- How should the `sim`-only (no hardware) installation mode be exposed — as a separate package, an extras_require group (`pip install ttl[sim]`), or a build-time flag?

---

## Qwen3.6-35B-A3B Weight Compatibility with Qwen3.5-35B-A3B TTNN Modules
**Date:** 2026-04-21
**Status:** Completed
**Guide:** `guides/qwen36_35b_a3b_weight_compatibility_with_qwen35_35b_a3b_ttnn_modules/`
**Why Needed:** Qwen3.6-35B-A3B uses the same `Qwen3_5MoeForConditionalGeneration` architecture class and `qwen3_5_moe` model type as Qwen3.5-35B-A3B, but the weights are separately trained. Need to confirm that all existing TTNN modules (TTNNQwen3FullAttention, TTNNQwen3LinearAttention, TTNNQwen3MoE, etc.) load and execute correctly with Qwen3.6 weights without any shape mismatches or dtype issues.
**Questions:**
- Are there any weight tensor shape differences between Qwen3.6-35B-A3B and Qwen3.5-35B-A3B that would cause loading failures in existing TTNN modules?
- Does the explicit `partial_rotary_factor: 0.25` at the top level (vs only in `rope_parameters`) change how HuggingFace resolves the rotary dimension, potentially affecting TTNNRotaryPositionEmbedding?
- Does the `bos_token_id: 248044` addition in Qwen3.6 config affect tokenizer behavior or generation loop initialization?
- Does the Multi-Token Prediction head (`mtp_num_hidden_layers: 1`) add extra weight keys that could interfere with `AutoModelForCausalLM.from_pretrained` loading?

---

## M-RoPE (Multimodal RoPE) Implementation on TTNN
**Date:** 2026-04-21
**Status:** Completed
**Guide:** `guides/m_rope_multimodal_rope_implementation_on_ttnn/`
**Why Needed:** Qwen3.6-35B-A3B uses Multimodal RoPE (M-RoPE) with interleaved sections [11, 11, 10] for vision/video inputs. The current TTNNRotaryPositionEmbedding handles standard RoPE and partial RoPE but may not support M-RoPE's per-modality position ID assignment. Understanding M-RoPE is needed for future multimodal bring-up beyond text-only inference.
**Questions:**
- How does M-RoPE differ from standard RoPE for text-only inputs — does it reduce to standard RoPE when no vision/video tokens are present, or does the interleaved section structure always apply?
- What are the M-RoPE section dimensions [11, 11, 10] — do they partition the rotary_dim=64 into three sub-groups (temporal, height, width), and how are position IDs assigned per sub-group?
- Can the existing TTNNRotaryPositionEmbedding be extended to support M-RoPE by pre-computing per-modality cos/sin tables, or does it require a fundamentally different implementation?
- What is the performance cost of M-RoPE vs standard RoPE on TTNN — does the per-section position indexing introduce additional memory accesses or kernel launches?


---

## dots.ocr on TT Hardware: Architecture, TTNN Port, and Relationship to Qwen 2.5 VL
**Date:** 2026-04-22
**Status:** Completed
**Guide:** `guides/dots_ocr_on_tt_hardware/`
**Why Needed:** `rednote-hilab/dots.ocr` is a 1.7B multimodal document parser (SOTA on OmniDocBench) with an in-progress TTNN port at `tenstorrent/tt-metal` branch `ign/dots_ocr`. The port lives in `models/demos/dots_ocr/` and reuses tt_transformers infrastructure, making it directly relevant to tt_symbiote. Understanding the architecture, the current state of the port, what is and is not on TTNN yet, and the T3K topology constraints will determine how much work remains to make it production-ready in tt_symbiote.
**Questions:**
- What is the exact architecture of dots.ocr — how does its `DotsOCRForCausalLM` (28-layer Qwen2-style decoder, hidden_size=1536, GQA 12Q/2KV) relate to Qwen 2.5 VL at the config and weight level, and what are the key differences from the full Qwen 2.5 VL model family?
- What is the current state of the `ign/dots_ocr` TTNN port — which components run on device (TTNN patch merger, text decoder via tt_transformers) vs. on host (42-layer ViT via HF PyTorch), what is the accuracy (PCC targets), and what is the measured decode throughput?
- What is the hybrid vision strategy — why is the 42-layer ViT kept on host rather than ported to TTNN, what would it take to port the `DotsVisionTransformer` fully to TTNN, and how does this compare to the Qwen 2.5 VL and Qwen3.6 vision encoder porting approaches?
- How does the T3K topology constraint arise from the GQA configuration — why does `num_key_value_heads=2` limit tensor parallelism to TP≤2 even on an 8-device mesh, and how does the submesh approach (`create_submesh` over a full `open_mesh_device`) handle this?
- What is the relationship between dots.ocr's vision encoder (42-layer ViT, patch_size=14, spatial_merge_size=2, hidden_size=1536) and the Qwen 2.5 VL vision encoder — are they architecturally identical or derived from a common base, and can the TTNN code from `qwen25_vl` (patch merger already ported) be reused directly?
- What are the remaining implementation gaps in `ign/dots_ocr` before the demo is production-ready — what do the commit messages ("removing qwen reference", "PC decode", "prefill at 0.98", "partial mesh support") indicate about what has and has not been stabilized?

---

## Multi-Token Prediction (MTP) on TT Hardware
**Date:** 2026-04-21
**Status:** Completed
**Guide:** `guides/multi_token_prediction_mtp_on_tt_hardware/`
**Why Needed:** Qwen3.6-35B-A3B has `mtp_num_hidden_layers: 1`, indicating a Multi-Token Prediction head that can predict multiple future tokens simultaneously. Need to understand whether MTP is active during standard autoregressive generation or is training-only, and whether implementing MTP on TT hardware could improve decode throughput via speculative decoding.
**Questions:**
- Does the MTP head participate in standard `model.generate()` calls in HuggingFace Transformers, or is it only used during training and can be safely ignored for inference?
- If MTP is inference-active, what is the computational structure — does it share the backbone hidden states and only add a lightweight prediction head, or does it require additional forward passes?
- Could MTP be used as a speculative decoding mechanism on TT hardware — predict N tokens speculatively, then verify in a single forward pass — and what would the throughput improvement be?
- What are the weight shapes and computation cost of the MTP head (`mtp_num_hidden_layers: 1`) relative to the main model, and would it fit in L1 or require DRAM placement?

---

## Removing synchronize_device from _maybe_all_gather in Hybrid Attention Modules
**Date:** 2026-04-22
**Status:** Pending
**Guide:** `guides/removing_synchronize_device_from_maybe_all_gather/`
**Why Needed:** Both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` call `ttnn.synchronize_device()` inside `_maybe_all_gather`, which is a host-blocking synchronization point. This prevents the full attention stack (and any LayerStack containing it) from being captured under Metal Trace, because trace capture requires a fully async device command stream with no host readback. Removing or replacing this synchronize_device call is a prerequisite for enabling end-to-end trace capture across the entire hybrid DeltaNet + full-attention decoder stack.
**Questions:**
- Why does `_maybe_all_gather` call `ttnn.synchronize_device()` — is it working around a race condition in the all_gather result, ensuring a prior async op completes, or something else?
- Is `ttnn.synchronize_device()` strictly necessary here, or can the synchronization be achieved via a lighter-weight mechanism (e.g., a device-side semaphore, event, or by relying on TTNN's implicit command queue ordering)?
- What is the async CCL pattern used in tt-transformers for all_gather operations inside traced regions, and can it be adopted for `_maybe_all_gather`?
- Are there other modules in tt-symbiote that call `ttnn.synchronize_device()` inside their forward pass that would also block full-stack trace capture?
- What is the latency cost of `ttnn.synchronize_device()` in `_maybe_all_gather` at decode batch=1 on T3K, and what throughput improvement would be gained by removing it?
- After removing `synchronize_device`, what validation is needed to confirm there are no race conditions in the resulting async all_gather → downstream compute pipeline?

---

## Partial Rotary Embedding Numerical Correctness for Non-Tile-Aligned rotary_dim in TTNN
**Date:** 2026-04-22
**Status:** Pending
**Guide:** `guides/partial_rotary_non_tile_aligned_numerics/`
**Why Needed:** `TTNNRotaryPositionEmbedding` in `rope.py` pads cos/sin with zeros when `rotary_dim % 32 != 0` (non-tile-aligned). Testing with `rotary_dim=48, head_dim=128` produces PCC ~0.71 vs PyTorch reference even in warm-up (no trace), indicating a numerical bug independent of tracing. Need to understand whether the zero-padding scheme is mathematically correct for `ttnn.experimental.rotary_embedding`, or whether the only safe option is to enforce tile-alignment.
**Questions:**
- What does `ttnn.experimental.rotary_embedding` expect for cos/sin shapes — does it require `rotary_dim` to exactly match the non-padded dimension, or does it accept zero-padded cos/sin and apply rotation only to the first `rotary_dim` elements?
- When `cos/sin` is zero-padded from dim 48 → 64 and `ttnn.experimental.rotary_embedding` is called with `head_dim=128`, does the rotate_half pairing correctly skip the padded zeros, or does it pair real elements (indices 16–47) with zero positions (indices 48–63), corrupting the output?
- Is there a trace-safe alternative to `ttnn.pad` with fill_value for this case — e.g., concat with a pre-allocated zeros buffer — and does it fix the numerical issue as well as the write-during-trace issue?
- Should `TTNNRotaryPositionEmbedding` enforce `rotary_dim % 32 == 0` as a precondition (raising an error for non-tile-aligned configs) rather than attempting to pad?
- What model configurations in the current symbiote codebase actually use non-tile-aligned rotary_dim values — is this path exercised in production, or is it dead code for current supported models?

---

## Pure TTNN DeltaNet Decode Step Without Host Readback
**Date:** 2026-04-22
**Status:** Pending
**Guide:** `guides/pure_ttnn_deltanet_decode_step/`
**Why Needed:** `TTNNQwen3LinearAttention` (the DeltaNet decode block) currently calls a PyTorch recurrence kernel on the host CPU during the state update step. This host readback breaks the device command stream and prevents Metal Trace from capturing the linear attention layer. Replacing this with a pure TTNN implementation that stays on-device is the second prerequisite (alongside removing `synchronize_device`) for full-stack trace capture across the entire Qwen3.6-35B-A3B decoder.
**Questions:**
- What exactly does the PyTorch recurrence kernel in `TTNNQwen3LinearAttention` compute — what is the mathematical recurrence (delta rule: `S_t = S_{t-1} * (1 - beta_t * k_t^T) + v_t * k_t^T`), and what tensors are read from / written to the device during host execution?
- Does TTNN have primitives that can express the DeltaNet recurrence without host readback — specifically outer product accumulation, element-wise state gating, and in-place state update for a rank-2 state matrix?
- What is the existing `gdn_full_fused_inplace` kernel (referenced in the Qwen3.5-27B Blackhole implementation) — can it be reused or adapted for the T3K Wormhole architecture?
- What are the tensor shape and memory layout requirements for the DeltaNet state matrix on T3K (hidden_size=2048, head_dim=128, 16 QK heads) — does it fit in L1 per device, or must it be DRAM-resident?
- What is the host-CPU roundtrip latency for the current PyTorch recurrence kernel at decode batch=1, and what throughput gain is expected from a pure on-device implementation?
- Are there existing TTNN or tt-metal kernels for scan/recurrence operations (e.g., parallel prefix scan, selective scan from Mamba) that could be adapted for the DeltaNet state update?
- What accuracy (PCC) is acceptable for a TTNN recurrence kernel relative to the reference PyTorch implementation, and how sensitive is overall model output quality to small errors in the state matrix update?


---

## Trace-safe pre-replication of position embeddings in TTNNQwen3FullAttention
**Date:** 2026-04-23
**Status:** Pending
**Guide:** `guides/trace_safe_cossin_prereplication/`
**Why Needed:** The `_ensure_replicated` helper added to fix the rotary_embedding crash (cos/sin sharded across TP devices) calls `ttnn.from_torch`, which allocates new device buffers — a host operation incompatible with Metal Trace capture. For end-to-end traced decode, cos/sin must be pre-allocated as replicated buffers during warm-up and updated via `ttnn.copy` inside the traced region, matching the existing `_decode_cur_pos` pre-allocation pattern. This is a prerequisite for full end-to-end trace capture across the entire Qwen3.6-35B-A3B hybrid decoder stack.
**Questions:**
- What is the `_decode_cur_pos` pre-allocation pattern in `move_weights_to_device_impl`, and how can it be adapted for cos/sin position embeddings (which change every decode step)?
- Does `TracedRun._alloc_kwarg_tensor` already pre-allocate cos/sin buffers, and if so, are they replicated or non-distributed (single-device)?
- Can `ttnn.copy` from a replicated source to a pre-allocated replicated destination be used inside a trace to update cos/sin between decode steps without breaking trace replay?
- What layout and memory config (DRAM vs L1, ROW_MAJOR vs TILE) is required for pre-allocated cos/sin buffers to be compatible with `ttnn.unsqueeze` and `ttnn.experimental.rotary_embedding` inside a trace?
- After pre-replication, does the guard in `TTNNRotaryPositionEmbedding.forward` (checking `rotary_dim % 64 != 0`) still correctly detect wrongly-sharded inputs during warm-up?

---

## TT-DiT Framework Architecture and TT-Symbiote Porting Strategy
**Date:** 2026-04-23
**Status:** Completed
**Guide:** `guides/tt_dit_framework_architecture_and_tt_symbiote_porting_strategy/`
**Why Needed:** TT-DiT (`models/tt_dit/`) is Tenstorrent's optimized framework for running Diffusion Transformer models (SD3.5, Flux1, Motif, Mochi, Wan2.2, Qwen-Image) on Wormhole hardware. It implements a complete stack — custom layers (Linear, RMSNorm, DistributedLayerNorm), attention with ring joint SDPA, 3-axis parallelism (CFG/SP/TP), CCL management, VAE decoders, text encoders (CLIP, T5, UMT5), and end-to-end pipelines with tracing support. TT-Symbiote (`models/experimental/tt_symbiote/`) is a PyTorch-to-TTNN acceleration framework with its own TTNNModule base class, dispatcher, weight preprocessing pipeline, and module library (attention, linear, normalization, MoE, etc.). Understanding TT-DiT's full architecture and evaluating a porting strategy to TT-Symbiote is essential for bringing DiT-based generative models into TT-Symbiote's unified model serving infrastructure. The TT-DiT source is at `/localdev/salnahari/testing_dir/tt-metal/models/tt_dit/` and TT-Symbiote is at `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/`.
**Questions:**
- What is the complete TT-DiT architecture — how do the `Module`/`Parameter` base classes (`layers/module.py`), the `CCLManager` (`parallel/manager.py`), and `DiTParallelConfig` (`parallel/config.py`) compose to form the model lifecycle (weight loading, device placement, forward execution), and how does this compare to TT-Symbiote's `TTNNModule` base class with its `preprocess_weights_impl`/`move_weights_to_device_impl`/`forward` pattern?
- How does TT-DiT's 3-axis parallelism (CFG parallel, sequence parallel, tensor parallel via `DiTParallelConfig`) work end-to-end — how are submeshes created, how do `ColParallelLinear`/`RowParallelLinear` shard weights, how does `DistributedLayerNorm` use `dit_layernorm_pre_allgather`/`dit_layernorm_post_allgather` across devices, and how does this map to TT-Symbiote's `DistributedConfig`/`DistributedTensorConfig` and `ShardTensor2dMesh` patterns?
- How does TT-DiT's joint attention mechanism work — what is the relationship between `Attention` (Q/K/V projection with merged QKV, RMSNorm on Q/K, RoPE, spatial+prompt joint SDPA) and `TransformerBlock` (adaptive norm with time embedding shift/scale/gate, attention, FFN), and which of these components have direct equivalents in TT-Symbiote's `TTNNAttention`/`TTNNGroupQueryAttention` vs. requiring new modules?
- What custom TTNN operations does TT-DiT depend on that TT-Symbiote does not currently use — specifically `ttnn.experimental.minimal_matmul`/`minimal_matmul_split`, `ttnn.transformer.joint_scaled_dot_product_attention`/`ring_joint_scaled_dot_product_attention`, `ttnn.experimental.dit_layernorm_pre_allgather`/`dit_layernorm_post_allgather`, `ttnn.experimental.wan_fused_rmsnorm_pre_allgather`/`wan_fused_rmsnorm_post_allgather`, and `ttnn.alt_complex_rotate90` for RoPE — and what would it take to integrate each into TT-Symbiote's dispatcher?
- How do TT-DiT's end-to-end pipelines (`pipelines/motif/`, `pipelines/mochi/`, `pipelines/wan/`, etc.) orchestrate the full inference flow — text encoding, scheduler loop, transformer denoising steps, VAE decoding, and tracing — and how would this pipeline orchestration map to TT-Symbiote's model registration and serving infrastructure?
- What is TT-DiT's weight loading and preprocessing pipeline — how does `Module.load_torch_state_dict` → `_prepare_torch_state` work for each layer type (QKV merging and head padding in `Attention`, chunked output reordering in `TransformerBlock`, row-major workarounds in `LayerNorm`, group norm mask creation), and how much of this can be handled by TT-Symbiote's existing `from_torch` → `preprocess_weights_impl` pattern vs. requiring custom preprocessing?
- What are the key architectural differences between TT-DiT's diffusion workload and TT-Symbiote's current LLM-focused workload — specifically the iterative denoising loop (20-50 forward passes per generation vs. autoregressive token-by-token), the absence of KV cache, the joint spatial+prompt attention pattern, the adaptive normalization with per-timestep modulation, and CFG parallel — and how do these differences affect the porting strategy?
- Which TT-DiT components can be directly reused as-is in TT-Symbiote (e.g. importing TT-DiT modules) vs. which need to be reimplemented as TTNNModule subclasses — considering TT-DiT's `Module` class is not a subclass of `TTNNModule` and has a different weight lifecycle, device management, and forward signature?
- How does TT-DiT handle tracing (`utils/tracing.py`) and performance profiling — does it use `ttnn.begin_trace_capture`/`end_trace_capture`/`execute_trace` patterns similar to tt-transformers, and how would traced DiT execution integrate with TT-Symbiote's `trace_enabled` infrastructure?
- What is the recommended porting priority — which of the 6 supported models (SD3.5, Flux1, Motif, Mochi, Wan2.2, Qwen-Image) would be the best first candidate for TT-Symbiote integration based on architectural complexity, performance maturity, and reuse of existing TT-Symbiote modules (e.g. T5/CLIP encoders already partially supported)?
## Tenstorrent Hardware Hang Causes, Debugging Tools, and Chip Reset Reduction
**Date:** 2026-05-02
**Status:** Completed
**Guide:** `guides/tenstorrent_hardware_hang_causes_debugging_tools_and_chip_reset_reduction/`
**Why Needed:** Running workloads on Tenstorrent hardware (via tt-metal/TTNN) can hang — the program stops making progress and the chip becomes unresponsive, often requiring a `tt-smi` reset or full system reboot. Understanding every root cause category (kernel deadlocks, NOC congestion, CCL/multi-chip synchronization failures, L1 memory corruption, firmware bugs, host-device synchronization issues, etc.) is critical for both preventing hangs during development and debugging them when they occur. A comprehensive catalog of existing Tenstorrent debugging tools and workflows, plus an analysis of gaps where new tools or features could reduce hang frequency and debug time, would significantly improve developer productivity.
**Questions:**
- What are all the known categories of root causes that can make a program hang on Tenstorrent hardware — kernel-level deadlocks (BRISC/NCRISC/TRISC synchronization, circular buffer producer/consumer stalls), NOC transaction deadlocks, L1 memory corruption or overflow, dispatch command queue stalls, firmware watchdog failures, and any others?
- What are the multi-chip and CCL-specific hang causes — all_gather/reduce_scatter deadlocks, semaphore protocol violations, topology misconfiguration, Ethernet link failures, and cross-chip synchronization issues on T3K and Galaxy systems?
- What are the host-device interaction hang causes — command queue full/stall, `synchronize_device` deadlocks, trace replay failures, async op ordering violations, and mismatched device/host state after errors?
- What are the memory-related hang causes — L1 bank collision stalls, DRAM bandwidth saturation causing NOC backpressure, circular buffer overflow, tile size mismatch causing DMA hangs, and out-of-memory conditions that silently corrupt state rather than erroring?
- What existing tools and utilities does Tenstorrent provide for detecting, diagnosing, and recovering from hangs — `tt-smi` (reset modes, status monitoring), `watcher` (firmware-level hang detection and register dumps), Tracy profiler (identifying the last completed op before hang), dispatch debug modes, and any kernel-level assertions or watchdog timers?
- What are the current best practices and developer workflows for debugging a hang — how to reproduce, how to narrow down the offending op or kernel, how to read watcher dumps, how to use binary search with op-level checkpoints, and how to distinguish hardware faults from software bugs?
- What future tools, features, or infrastructure improvements could reduce hang frequency or make hangs easier to debug — automatic hang detection with root cause classification, device-side heartbeat monitoring, automatic state snapshots before reset, deterministic replay of the command stream leading to a hang, and better error propagation from firmware to the Python layer?
- How can the need for chip resets via `tt-smi` or system reboots be reduced — graceful error recovery mechanisms, partial device reset (single core/tensix vs full chip), firmware-level watchdog with automatic recovery, and techniques for making workloads more resilient to transient hardware errors?
- What are the differences in hang behavior and debugging across Tenstorrent chip generations (Grayskull, Wormhole, Blackhole) and system configurations (single chip, T3K, Galaxy) — are certain hang categories specific to multi-chip configurations or specific architectures?

---

## nlp_create_qkv_heads Compatibility with Gemma4 Fused QKV Layout
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** The QKV reshape+permute chain in `_project_qkv()` (`gemma4_attention.py` lines 406–422) is the single largest measured cost in isolated attention modules — Reshape = 44.73% of `attention_sliding` device time and 43.57% of `attention_global` device time. `ttnn.experimental.nlp_create_qkv_heads` (used in tt-transformers) fuses slice+reshape+permute into a single kernel. If it supports Gemma4's heterogeneous GQA configurations (32Q/16KV/256dim sliding, 32Q/4KV/512dim global), it could collapse ~9 shape ops per prefill forward (3 slices + 3 reshapes + 3 permutes) into a single dispatch. For decode: 6 reshapes reduced to ~2–3. Critical risk: Gemma4 global layers use `v_weight = k_weight.clone()` (`gemma4_attention.py` line ~153), giving a 3-way unequal head split (32Q + 4K + 4V) that must be handled correctly.
**Questions:**
- Does `ttnn.experimental.nlp_create_qkv_heads` accept Q_heads=32, K_heads=4/16, V_heads=4/16 (GQA with unequal splits) and head_dim=512/256?
- Does it correctly handle the K=V shared projection case in global attention?
- What input tensor layout requirements must be met for the fused op?
- What is the measured device-time reduction in `attention_sliding` and `attention_global` after replacing the reshape+permute chain?

**Findings (partial, Phase 0a synthetic-shape PCC gate, session 2026-05-04):**
- **Q1 (head split / head_dim 256 & 512):** YES. `ttnn.experimental.nlp_create_qkv_heads` (prefill, interleaved DRAM, bf16, TILE) accepts Gemma4 sliding (nQ=32, nKV=16, head_dim=256) and global (nQ=32, nKV=4, head_dim=512) on T3K with `transpose_k_heads=False`, `memory_config=DRAM_MEMORY_CONFIG`. PCC vs the current `_project_qkv` slice→reshape→permute reference = **1.000000** for Q, K, and V. Empirically refutes the prior `head_dim ≤ 128` claim from `PLAN_gemma4_decode_perf_final.md:129`. Source confirms no validator upper bound: `nlp_create_qkv_heads.cpp:31` infers head_dim from `padded_shape[3]/(num_q_heads + 2*num_kv_heads)`; `nlp_create_qkv_heads_program_factory.cpp:62` loops `q_out_w_tiles = head_dim/TILE_WIDTH` (=8 for 256, =16 for 512) without an upper-bound check.
- **Q1 (decode):** `ttnn.experimental.nlp_create_qkv_heads_decode` also passes at head_dim=256 and 512 with batch=1 (sliding and global) using HEIGHT_SHARDED `(y=1,x=1)` core grid, shard `(round_up(num_kv_heads,32)=32, head_dim)`, then `sharded_to_interleaved → L1` for read-back. PCC = **1.000000** for Q, K, V across both head configs. Decode op constraints honored: `num_q_heads=32` (at the boundary, `≤ 32`), bf16 (not bf8_b), `input_shape[3] % TILE_WIDTH == 0`.
- **Q2 (K=V sharing):** verified safe by construction. `gemma4_attention.py:153` materializes `v_weight = k_weight.clone()` at weight-build time, so the fused projection's V band is bit-identical to its K band. The op is unaware of K=V sharing — it sees a regular GQA tensor. The Phase 0a global cases asserted `(V_band == K_band).all()` on the host input and PCC=1.0 holds on the device output. Different post-op `v_norm` vs `k_norm` then breaks the symmetry, matching legacy semantics.
- **Q3 (input layout):** `[B, 1, S, fused]` rank-4 TILE for prefill (rank-3 → rank-4 reshape required since `qkv_proj` returns rank-3); `[1, 1, B, fused]` rank-4 TILE for decode. dtype bf16 or fp32 (not bf8_b for decode). DRAM_INTERLEAVED or WIDTH_SHARDED L1 input accepted. Output for prefill is INTERLEAVED, for decode is HEIGHT_SHARDED `(round_up(num_kv_heads,32), head_dim)` per device. The decode HEIGHT_SHARDED layout exactly matches the existing reshard at `gemma4_attention.py:768–780`, so that block becomes redundant once decode is wired up (deferred to Phase 3 cleanup).
- **Q4 (perf delta in `attention_sliding` / `attention_global`):** Measured on T3K via `test_gemma4_profile_attention_{sliding,global}.py` env-on/off. **Sliding prefill: total `attention_sliding` device time −33.9 % (45,134 → 29,830 us); Reshape −67 % absolute (22,810 → 7,479 us). Global prefill: total `attention_global` device time −26.1 % (724,837 → 535,224 us); Reshape −68.3 % absolute (310,727 → 98,590 us).** Decode delta is small (sliding decode segment −4.8 %, global decode segment −2.4 %): the canonical `sharded_to_interleaved → norm → re-shard` round-trip pattern (`attention_1d.py:619–628`) eats most of the fused-decode op's saving on the legacy 6-reshape decode chain. Combined sliding+global prefill alone exceeds the original ≥25 % combined target.

- **Production integration root-cause finding (2026-05-04):** Initial integration produced **garbled E2E generation despite per-op PCC=1.0** because the production `_project_qkv` env-on path performed `ttnn.deallocate(qkv_states)` immediately after `ttnn.reshape` of the rank-3→rank-4 view but BEFORE `nlp_create_qkv_heads(_decode)` consumed the view. `ttnn.reshape` on a TILE_LAYOUT, DRAM_INTERLEAVED `[B, S, fused]` tensor that adds a unit dim of 1 (rank-3 → rank-4) is metadata-only (no data copy) — so deallocating the source invalidated the view and the fused op read partially-overwritten memory. Symptom was deterministic (PCC=0.5777 / 0.2516 / 0.2500 for Q/K/V on prefill_global) and config-dependent (sliding `head_dim=256` got lucky on heap state; global `head_dim=512` consistently failed). Fix: move `ttnn.deallocate(qkv_states)` to AFTER the fused op call. Confirmed at PCC=1.000000 for all four (sliding/global × prefill/decode) configurations. **Lesson:** unit/op-level PCC tests that don't reproduce the production deallocate-ordering can pass while the integration silently corrupts state — a bottom-up real-weight `_project_qkv`-level PCC gate caught it where Phase 0a synthetic-shape op-level PCC, profile-test "passes" (no PCC asserts), and E2E "passes" (only `len > 0`) all missed it.

**Reproduction (Phase 0a smoke + Phase 1 bottom-up):**
- `MESH_DEVICE=T3K pytest --timeout=0 models/experimental/tt_symbiote/tests/test_gemma4.py::test_gemma4_nlp_qkv_smoke -s -v` (op-level synthetic PCC, 4 cases)
- `MESH_DEVICE=T3K pytest --timeout=0 models/experimental/tt_symbiote/tests/test_gemma4_bottom_up.py::test_gemma4_change_nlp_qkv -s -v` (real-weight `_project_qkv` legacy-vs-fused PCC, 4 cases — exposes deallocate-ordering bugs that op-level smoke misses)

---

## Partial RoPE Integration into TTNNGemma4Attention for Global Attention Layers
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** `test_gemma4_profile_partial_rope.py` confirms PARTIAL_DECODE at 2.91 ms/iter vs FULL_CHUNK_DECODE at 4.72 ms/iter — a 1.6x speedup for global attention decode. The production code in `TTNNGemma4Attention` still uses the full-chunk split RoPE path (`rope.py` lines 140–146: 4 `rotary_embedding_llama` calls + 8 Slice + 2 Concat). Integration would eliminate the Slice+Concat overhead (15.54% of rope module time) and reduce RoPE kernel calls from 4 to 1 per global attention forward. Scope: 10 global layers of 60 total.
**Questions:**
- Is the partial_rope path fully trace-compatible (no dynamic allocations, no host readbacks)?
- What is the exact measured device-time saving per global decoder layer when switching to partial_rope in the production attention path?
- With 10 global layers, what is the total per-decode-step latency saving end-to-end?
- Does partial_rope maintain required numerical accuracy (PCC target) for Q and K rotations?

---

## TTNNGemma4DistributedRMSNorm AllGather Topology and Link Count Optimization
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** AllGather is 48.97% of `distributed_rmsnorm` device time (8,593 us). The Gemma4 subclass (`gemma4_normalization.py` line ~64) uses `ttnn.Topology.Linear` while the base class (`normalization.py`) uses `ttnn.Topology.Ring`. This fires 4 times per decoder layer × 60 layers = 240 times per full forward pass. With a tiny variance-stats payload (one partial scalar per device), fixed per-hop latency dominates; Ring (parallel hops) is likely faster than Linear (7 sequential hops). Whether the Linear choice is intentional or a legacy holdover is unknown.
**Questions:**
- What is the measured AllGather latency for the variance-stats payload with Linear vs Ring topology on T3K?
- Can `num_links=2` be used for this AllGather without correctness issues?
- Is the `ttnn.Topology.Linear` in the Gemma4 subclass intentional, and what is the justification vs the base class's Ring default?
- What is the total impact across 240 AllGather calls per full decode pass?

---

## Fused GeGLU (gate × gelu(up)) TTNN Kernel for Gemma4 MLP
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** After `fused_gate_up_proj` produces `[B, S, 43008]`, the MLP code issues 4 ops: 2x Slice (`gemma4_mlp.py` lines 92–93) + gelu (line 97) + multiply (line 100). Combined: Slice 4.28% + Unary 2.12% + BinaryNg 3.20% = 9.6% of MLP device time (~5,100 us per isolated MLP forward). A fused GeGLU kernel would eliminate 2 intermediate Slice materializations and merge 3 kernel launches into 1. Applies to all 60 decoder layers.
**Questions:**
- Is there an existing TTNN op or program config that fuses slice+gelu+multiply for the GeGLU sequence (in `ttnn.transformer`, `ttnn.experimental`, or a custom activation module)?
- What is the performance difference between the 4-op chain and a fused implementation for `[B, S, 43008]` input on Wormhole B0?
- Would the fused kernel accept col-sharded MLP input (after all_reduce) directly?

---

## Trace-Safe ttnn.all_reduce to Eliminate Decomposed ReduceScatter+AllGather Overhead
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** `TTNNLinearIColShardedWAllReduced.forward()` (`linear.py` lines 166–184) decomposes `ttnn.all_reduce` into `reduce_scatter + all_gather` for Metal Trace compatibility (code comment at lines 168–169). The final AllGather accounts for 14.27% of linear device time and ~20% of MLP device time — it has zero compute value. Every QKV projection and every gate+up projection across 60 decoder layers pays this overhead: 120 total AllGather ops per full decode pass.
**Questions:**
- Has `ttnn.all_reduce` been updated with pre-allocated intermediates to make it trace-safe?
- What is the measured latency of `ttnn.all_reduce` vs decomposed `reduce_scatter+all_gather` for output sizes 21504 (QKV) and 43008 (gate+up) on T3K?
- Can an intermediate buffer be passed explicitly to `ttnn.all_reduce` as a pre-allocated static buffer to enable trace capture without the two-phase decomposition?
- What is the total per-decode-step savings if the AllGather is eliminated from 120 ops?

---

## DRAM-Sharded Weight Placement for Bandwidth-Bound Batch=1 Decode Matmul in Gemma4
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** Matmul is 59.28% of `linear` device time and 49% of `mlp` device time at batch=1 decode, DRAM-bandwidth-bound. The weight matrix per device must be streamed from DRAM every token. Current placement: `DRAM_MEMORY_CONFIG` (dram_interleaved). In sliding decoder layers (50 of 60), Matmul is now the #1 device-time op after reshape reduction — making this the highest-priority hardware-level optimization. Current program_config uses no custom sharding.
**Questions:**
- What program_config maximizes weight streaming throughput for `[1, 1, 672]` × `[672, 21504]` at batch=1 on Wormhole B0?
- Would DRAM_SHARDED vs dram_interleaved reduce the ~26,000 us Matmul device time?
- Would bfloat8_b weights for QKV and gate+up projections cause unacceptable accuracy degradation (PCC vs bfloat16)?
- Does FPU utilization analysis confirm the bandwidth-bound hypothesis (expected: FPU < 10%)?

---

## AllGather/ReduceScatter num_links and Topology Tuning on T3K for Gemma4 Payloads
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** All collectives in `linear.py` use `num_links=1, topology=Ring`. T3K has 2 Ethernet links per device. At decoder-layer level, AllGather (14.79%) + ReduceScatter (13.06%) = 27.85% of `decoder_layer_v2` device time. Payload sizes: 21504 elements (QKV all_reduce), 43008 elements (gate+up all_reduce), 672 elements (4x RMSNorm all_gather).
**Questions:**
- What is the measured latency for reduce_scatter and all_gather with `num_links=1` vs `num_links=2` for the Gemma4 payload sizes on T3K Ring topology?
- Is `num_links=2` compatible with Metal Trace?
- What is the minimum payload size at which `num_links=2` outperforms `num_links=1` on T3K?
- For the tiny RMSNorm AllGather (variance stats), does Linear or Ring perform better, and does increasing num_links help at this payload size?

---

## Tracy Device Profiler Intermittent CSV Failure on T3K
**Date:** 2026-05-03
**Status:** Pending
**Guide:** TBD
**Why Needed:** `decoder_layer_sliding` and `partial_rope` profiling fail intermittently with "cpp_device_perf_report.csv not found" despite tests running successfully. This creates a profiling reliability gap — two key Gemma4 modules cannot be profiled consistently. The workaround (`decoder_layer_v2` as a substitute for sliding) works but is fragile.
**Questions:**
- What is the exact sequence of events causing the CSV to be absent after a successful test run? Is it a Tracy reader race, early `ReadDeviceProfiler()` flush, or chip state issue?
- Is there a deterministic reproduction path?
- Would `ttnn.synchronize_device()` before the Tracy CSV export phase prevent the race?
- Is there a minimum drain time after `tt-smi -r` that eliminates the failure reliably?

---

## Empirical head_dim Envelope of nlp_create_qkv_heads(_decode) on Wormhole
**Date:** 2026-05-04
**Status:** Pending
**Guide:** TBD
**Why Needed:** `ttnn.experimental.nlp_create_qkv_heads` and `nlp_create_qkv_heads_decode` are the canonical fused replacements for the slice+reshape+permute QKV-head chain in transformer attention. Upstream tests (`tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_create_qkv_heads.py`) exercise head_dim ≤ 128 only. A prior Gemma4 plan (`PLAN_gemma4_decode_perf_final.md:129`) claimed `head_dim ≤ 128` was a hard kernel limit, but inspection of the C++ source (`nlp_create_qkv_heads.cpp:31` infers head_dim from input shape; `program_factory.cpp:62` loops tiles without an upper-bound check) shows no validator actually enforces this. Phase 0a empirical validation on T3K confirmed head_dim 256 and 512 work at PCC=1.0 for both prefill and decode variants on Gemma4 head configs (32Q/16KV and 32Q/4KV). A complete empirical envelope across head_dim, dtype, and layout is needed to unblock any future model with non-standard head_dim, prevent rediscovering this for each model port, and document the exact constraints.
**Questions:**
- What is the maximum head_dim that produces correct output on Wormhole B0 / T3K for `nlp_create_qkv_heads` (interleaved prefill, sharded prefill) and `nlp_create_qkv_heads_decode` (HEIGHT_SHARDED output)?
- Test matrix: head_dim ∈ {64, 96, 128, 192, 256, 384, 512, 1024} × dtype ∈ {bf16, bf8_b, fp32} × layout ∈ {DRAM_INTERLEAVED, WIDTH_SHARDED L1, sharded prefill program-factory variant}. Which combinations PCC vs torch reference, which fail, and how do they fail (validator reject vs silent miscompute vs hang)?
- Which of `nlp_create_qkv_heads_segformer`, `nlp_create_qkv_heads_falcon7b`, `create_qkv_heads_from_separate_tensors` have different head_dim envelopes, and is there a decision tree for picking the right variant for a new model?
- For decode, does the `num_q_heads ≤ 32` cap interact with head_dim (e.g. is correctness sensitive at the 32-head boundary for large head_dim)?

---

## Resharding Cost vs. Fused-Op Win for Decode QKV-Head Creation
**Date:** 2026-05-04
**Status:** Pending
**Guide:** TBD
**Why Needed:** `nlp_create_qkv_heads_decode` requires WIDTH_SHARDED L1 or INTERLEAVED input and emits HEIGHT_SHARDED output. Per-head RMSNorm doesn't accept HEIGHT_SHARDED inputs (documented at `models/common/modules/attention/attention_1d.py:618`), so the fused-decode path implies a `sharded_to_interleaved → norm → re-shard` round trip around the per-head norm — adding two `to_memory_config` ops to the path. Whether the fused decode op is a net win after accounting for this depends on (num_q_heads, num_kv_heads, head_dim, batch). For Gemma4 the legacy decode path is already hand-optimized to 6 reshapes (`gemma4_attention.py:402–417`), and the fused-decode op may not net-beat the existing 6-reshape path even though it net-beats the 9-op prefill path. A reusable decision rule across GQA configs would prevent re-deriving this trade-off per model.
**Questions:**
- For each of the four (num_q_heads, num_kv_heads, head_dim, batch) regimes spanning current production models — Llama-3 (32/8/128, B≤32), Mistral (32/8/128), Qwen3 (variable), Gemma4 (32/16/256 sliding; 32/4/512 global, B=1) — is `to_memory_config(WIDTH_SHARDED L1) + nlp_create_qkv_heads_decode + sharded_to_interleaved + norm + re-shard` a net win over the legacy slice + 6-reshape path?
- What is the breakeven point on batch and head_dim where the reshard overhead is dominated by the fused-op savings?
- Does the answer change if the upstream `qkv_proj` is modified to emit WIDTH_SHARDED L1 natively (eliminating one reshard)?
- Is there a way to bypass the HEIGHT_SHARDED output mem-config requirement (e.g. via `output_tensors` argument) and have the fused decode op emit interleaved directly, eliminating the post-op reshard?

---

## Trace-Capture Compatibility of nlp_create_qkv_heads(_decode) and Other Fused TM Ops
**Date:** 2026-05-04
**Status:** Pending
**Guide:** TBD
**Why Needed:** Symbiote's decode path runs under Metal Trace (warmup phase 2 in `test_gemma4.py:237–251`). Operations that perform host readbacks, non-deterministic allocations, or shape-dependent control flow at runtime cannot be trace-captured safely. `ttnn.all_reduce` famously had to be decomposed into `reduce_scatter + all_gather` for trace safety (see `linear.py:166–184` comment). We need a reusable checklist of trace-capture properties for fused TM ops (`nlp_create_qkv_heads`, `nlp_create_qkv_heads_decode`, `create_qkv_heads_from_separate_tensors`, `nlp_concat_heads`, `all_reduce_create_qkv_heads`) so that adopters don't rediscover trace-incompatibilities deep into integration. The checklist should also generalize to other fused-kernel ops (RoPE, GeGLU, RMSNorm).
**Questions:**
- Does `nlp_create_qkv_heads` allocate any intermediate device buffers whose addresses are non-deterministic across trace replay?
- Does `nlp_create_qkv_heads_decode` with `batch_offset` or `slice_size` parameters interact correctly with trace replay (these may introduce shape-dependent control flow)?
- What is the canonical pre-flight check for "is this op trace-safe" — is there a single API or runtime mode that surfaces incompatibilities without requiring full integration?
- Does `all_reduce_create_qkv_heads` (which fuses CCL with head creation) require the same `reduce_scatter+all_gather` decomposition workaround under trace, or is it natively trace-safe?
- For each currently-trace-unsafe op in `models/experimental/tt_symbiote/`, what's the upstream issue tracking trace-safety, and which Symbiote modules are affected?

---

## TTNN ttnn.reshape View-vs-Copy Semantics and Source Deallocate Ordering Hazard
**Date:** 2026-05-04
**Status:** Pending
**Guide:** TBD
**Why Needed:** A production integration of `ttnn.experimental.nlp_create_qkv_heads(_decode)` in `gemma4_attention.py::_project_qkv` produced silently garbled end-to-end output (1024-token decode generation became multilingual gibberish) while every per-op PCC test passed. Root cause: the code did `qkv_4d = ttnn.reshape(qkv_states, ...)` followed by `ttnn.deallocate(qkv_states)` BEFORE the fused op consumed `qkv_4d`. The reshape (rank-3 `[B, S, fused]` → rank-4 `[B, 1, S, fused]`, TILE_LAYOUT, DRAM_INTERLEAVED) was metadata-only (a view, not a data copy) — so deallocating the source freed the underlying buffer that `qkv_4d` aliased, and the next op read partially-overwritten memory. Failure was deterministic (PCC=0.5777 / 0.2516 / 0.2500) and config-dependent: smaller `head_dim`/`Wt` configs got lucky on heap state and PASSED, while `head_dim=512, num_q_heads=32` consistently failed. The fix (move deallocate to after the consuming op) is trivial, but the class of bug is generic — every model author who writes `ttnn.reshape` followed by `ttnn.deallocate` of the source risks this depending on shape/layout/alloc order. A clear public contract (when does `ttnn.reshape` view vs copy?) and a runtime/static detection mechanism would prevent future occurrences across all tt-metal model integrations.
**Questions:**
- For each combination of (input rank → output rank, layout TILE/ROW_MAJOR, memory_config DRAM/L1/SHARDED, dtype, total-element-preserving vs non-preserving), is `ttnn.reshape` guaranteed to be a metadata-only view, guaranteed to copy, or does the choice depend on runtime state? Where is this contract documented?
- Specifically for TILE_LAYOUT, DRAM_INTERLEAVED, BFLOAT16, with the new dim being a unit-dim insertion (rank-3 → rank-4 with shape `[B, 1, S, K]`): is this always a view? Does it ever fall back to a copy?
- Does TTNN provide an explicit `ttnn.reshape_view(...)` and `ttnn.reshape_copy(...)` that make the semantics unambiguous, OR an `aliases(t1, t2) -> bool` that callers can use to decide whether a deallocate is safe?
- Are there static-analysis or runtime-debug modes (e.g. an "alias-tracking" build) that can flag `deallocate(source); op(view_of_source)` patterns automatically? `TT_METAL_ASSERT_VIEW_ALIASING` or similar?
- Is there a tt-metal utility test pattern (similar to AddressSanitizer's use-after-free) that can catch deallocate-after-view across a model's full forward pass without requiring per-op manual PCC gates?
- Reverse direction: when an op consumes a view and the user wants to free both view and source, is there a single `ttnn.deallocate_view_chain(view)` that handles it correctly? Currently developers must guess whether to deallocate source, view, or both.
- How does this interact with Metal Trace? If a trace captures the buggy `deallocate(source); op(view)` pattern, does the trace replay deterministically reproduce the use-after-free, or does heap state evolution make the bug intermittent across replays?

---

## Layout Convention Mismatch Between nlp_create_qkv_heads (Prefill) and _decode Variants
**Date:** 2026-05-04
**Status:** Pending
**Guide:** TBD
**Why Needed:** The prefill variant `ttnn.experimental.nlp_create_qkv_heads` expects input `[B, 1, S, fused_dim]` (batch in dim-0, sequence in dim-2). The decode variant `nlp_create_qkv_heads_decode` expects input `[1, 1, B, fused_dim]` (batch in dim-2, with B padded to 32). This means production models that share a `qkv_proj` between prefill and decode must do a different `ttnn.reshape` per path, and the upstream `qkv_proj` cannot emit a single layout that's natively compatible with both ops. Every model author re-derives the layout shim and the rank-3 → rank-4 reshape; the cost is small but compounds with bugs and inconsistency. A documented preferred upstream layout — or a thin TTNN wrapper that abstracts the difference — would prevent recurring bugs across model ports.
**Questions:**
- Is there a documented reason for the convention difference (e.g. kernel program-factory layout requirement vs historical accident)?
- Is there an existing TTNN utility that wraps both ops behind a single Python interface (`create_qkv_heads(input, num_heads, ..., is_decode)`) that hides the layout shim?
- What is the preferred upstream `qkv_proj` output layout that minimizes shim cost on both paths — rank-3 `[B, S, fused]`, rank-4 prefill-style `[B, 1, S, fused]`, rank-4 decode-style `[1, 1, B, fused]`, or width-sharded L1?
- Does the rank-3 → rank-4 `ttnn.reshape` always degrade to a metadata-only `ReshapeViewDeviceOperation` (free) or can it fall back to a real reshape kernel under certain memory configs?
- Would a single fused projection op that emits `[B, num_heads, S, head_dim]` directly (combining matmul + create_qkv_heads) be feasible, eliminating both the shim and the all-reduce → reshape path?
