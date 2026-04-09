# Chapter 7 -- High-Value Fusion Targets

This chapter identifies the TT-Symbiote operations that stand to benefit most from TT-Lang kernel fusion and sketches concrete kernel designs for each.

## Selection Criteria

A fusion target is high-value when it meets one or more of the following criteria:

| Criterion | Why It Matters | Screening Metric |
|-----------|---------------|-----------------|
| **Multiple sequential TTNN calls** | Each call is a separate kernel launch with its own DRAM read/write cycle. Fusing N calls into one eliminates N-1 intermediate round-trips. | Count of `ttnn.*` calls in the module's `forward()` method. |
| **High memory traffic** | Large intermediate tensors are written to DRAM only to be read back immediately by the next op. The compute-to-memory ratio is low. | Sum of intermediate tensor sizes (bytes) between first and last op in the sequence. |
| **Complex dispatch logic** | Control flow (loops over experts, conditional padding, layout conversions) adds host-side latency and prevents TTNN trace capture. | Presence of Python `for` loops, `if` branches, or `ttnn.to_layout` calls in hot paths. |
| **Existing partial fusion** | TT-Symbiote already fuses some ops (e.g., `TTNNLinearSilu`), proving the pattern is viable. TT-Lang can extend these to arbitrary compositions. | Classes like `TTNNLinearActivation`, `TTNNLinearSilu`, `TTNNLinearGelu` in `modules/linear.py`. |

Applying these criteria to the TT-Symbiote module catalog (see [Chapter 5](../ch5_symbiote_architecture/index.md)) yields three high-value fusion families:

## Contents

| File | Fusion Family | Primary Bottleneck |
|------|--------------|-------------------|
| [`moe_expert_pipeline.md`](./moe_expert_pipeline.md) | MoE Expert Pipeline | 3 separate `sparse_matmul` calls + `silu` + `mul` with DRAM intermediates; Python expert-dispatch loop on fallback path |
| [`fused_attention.md`](./fused_attention.md) | Attention (QKV + RoPE + SDPA + Output) | Separate QKV projections, standalone RoPE pass, and output projection each materialize full tensors to DRAM |
| [`fused_activations.md`](./fused_activations.md) | Fused Activations (Linear + Activation) | Standalone activation ops read/write entire tensors; SwiGLU requires two linear ops + activation + elementwise multiply |

## Key Takeaways

1. **The MoE expert pipeline is the single highest-value fusion target.** In `TTNNExperts.forward()`, the sequence `sparse_matmul(w1)` then `silu` then `mul` then `sparse_matmul(w2)` generates three full-sized intermediate tensors in DRAM. A fused TT-Lang kernel can keep all intermediates in L1, cutting DRAM traffic by roughly 3x for the expert compute phase.

2. **Attention fusion is the broadest opportunity.** Every decoder layer runs the full attention pipeline. Fusing QKV projection into a single matmul (already done in `TTNNFusedQKVSelfAttention`) is the first step; fusing RoPE application and the softmax-value multiply into the SDPA kernel eliminates two additional DRAM round-trips per layer.

3. **Activation fusion is the lowest-effort, highest-frequency win.** `TTNNLinearSilu` and `TTNNLinearGelu` already demonstrate the pattern of calling `ttnn.silu`/`ttnn.gelu` after `ttnn.linear` in separate kernel launches. TT-Lang can fold the activation into the matmul's post-processing tile, eliminating one DRAM write+read per activation. For SwiGLU (used in LLaMA/Qwen/GLM-4 MLP), this extends to fusing `gate_proj` then `silu` then `mul(up_proj)` into a single kernel.

4. **All designs follow the DFB (DataFlow Buffer) pattern** from [Chapter 1](../ch1_programming_model/index.md): data movement threads stream tiles through circular buffers while compute threads process them, keeping the compute pipeline fed without waiting on DRAM.

5. **Integration uses the contract defined in [Chapter 6](../ch6_integration_strategy/index.md):** each fused kernel is wrapped in a `TTNNModule` subclass with `preprocess_weights_impl`, `move_weights_to_device_impl`, and a `forward()` that dispatches to the TT-Lang compiled kernel.
