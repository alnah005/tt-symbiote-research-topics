# Chapter 7 -- MoE Architecture and Cross-Model Comparison

## Overview

This chapter provides a deep dive into the Mixture of Experts configuration used in Qwen3.6-35B-A3B and compares it with two contemporary MoE models: DeepSeek-V3 and Gemma4-26B-A4B. The central question is what the "many-small-experts" design philosophy (256 experts each with an intermediate size of 512) means for compute efficiency, memory bandwidth, routing overhead, and expert-parallel deployment on Tenstorrent hardware.

Because Qwen3.6 is architecturally identical to Qwen3.5 (see Chapter 3), the MoE configuration examined here applies equally to both. No changes to the MoE routing or expert forward pass are required for the TTNN implementation to support Qwen3.6 weights.

---

## Learning Objectives

After completing this chapter, readers will be able to:

1. **State the exact MoE configuration** of Qwen3.6: 256 routed experts, 1 shared expert, top-8 routing, `moe_intermediate_size=512`, `hidden_size=2048`.

2. **Derive the per-expert parameter count** from the SwiGLU architecture ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$) and aggregate it to per-layer and total expert parameter counts.

3. **Compute active-parameter FLOPs per token per MoE layer** using the 9-expert (8 routed + 1 shared) execution path.

4. **Compare Qwen3.6's MoE design** against DeepSeek-V3 (same expert count, much larger experts and hidden size) and Gemma4-26B-A4B (fewer, larger experts with top-1 routing), identifying the tradeoffs of each approach.

5. **Analyze the hardware utilization implications** of the many-small-experts design for Tenstorrent T3K devices, including DRAM bandwidth pressure, expert parallelism partitioning, and the role of bfp4 quantization.

---

## Chapter Contents

| File | Description |
|------|-------------|
| [`qwen36_moe_architecture.md`](./qwen36_moe_architecture.md) | Complete MoE configuration for Qwen3.6: expert architecture (SwiGLU FFN), router design, parameter counts, FLOP analysis, load balancing, and TTNN deployment implications |
| [`cross_model_moe_comparison.md`](./cross_model_moe_comparison.md) | Side-by-side comparison with DeepSeek-V3 and Gemma4-26B-A4B; many-small vs fewer-large expert analysis; hardware utilization implications for Tenstorrent |

---

## Cross-References to Existing Guides

This chapter should be read alongside the following existing guides, which cover MoE optimization on Tenstorrent hardware from the implementation perspective:

- `guides/moe_optimization_techniques_for_ttnn/` -- Techniques for mapping MoE forward passes onto TTNN operations, including expert batching, token dispatch, and weight layout.
- `guides/expert_parallelism_strategies/` -- Expert parallelism sharding strategies for multi-device deployments, including all-to-all communication patterns and load balance considerations.
- `guides/ttnn_moe_performance_optimization_on_t3k/` -- T3K-specific performance optimization for MoE layers, covering DRAM bandwidth, compute utilization, and quantization strategies for expert weights.

The present chapter focuses on the architectural properties and cross-model context. The guides above provide the TTNN implementation details that complement this analysis.

---

## Relationship to Other Chapters

- **Chapter 1 (Architecture Overview)** introduces the MoE configuration as part of the full hyperparameter table. This chapter deepens that introduction with a full parameter and FLOP analysis.
- **Chapter 3 (Qwen3.5 vs Qwen3.6 Differences)** establishes that the MoE configuration is identical between Qwen3.5 and Qwen3.6. Everything in this chapter applies to both models.
- **Chapter 2 (Gated DeltaNet Deep Dive)** covers the attention component that precedes each MoE FFN block in the forward pass data flow.

---

**Previous:** [Chapter 6 -- Thinking Preservation](../ch6_thinking_preservation/index.md)
**Next:** [Chapter 8 -- Vision Encoder and Multimodal Integration](../ch8_vision_encoder/index.md)
