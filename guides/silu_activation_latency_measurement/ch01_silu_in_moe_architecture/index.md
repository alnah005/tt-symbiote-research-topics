# Chapter 1: SiLU in MoE Architecture

This chapter establishes the computational context for SiLU activation latency measurement. It covers how SiLU appears inside the Feed-Forward Network (FFN) block of Mixture-of-Experts (MoE) models, why it is applied in the specific SwiGLU variant used by modern LLMs, and why it may carry non-trivial cost on SFPU-based hardware such as Tenstorrent Wormhole.

---

## Prerequisites

Before reading this chapter, you should be comfortable with the following:

- [ ] PyTorch `nn.Linear`, `F.silu`, and basic tensor operations
- [ ] Conceptual understanding of transformer FFN blocks (two linear layers with an activation in between)
- [ ] Familiarity with `ttnn.matmul` and `ttnn.silu` at the API level (knowing they exist; internals are not required yet)
- [ ] Basic understanding of MoE routing: a router selects K experts per token, and each expert is an independent FFN
- [ ] Awareness of 4D tensor conventions in TTNN: `[batch, seq, rows, cols]`; when a dimension is unused, it is set to `1`, giving shapes like `[1, 1, M, K]`

If you are unfamiliar with MoE routing or TTNN tensor conventions, review the project-level prerequisites document before continuing.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe the structure of a dense FFN and identify where it differs in an MoE FFN (expert weight slicing, per-expert dispatch).
2. Draw the SwiGLU compute graph showing `gate_proj`, `up_proj`, `down_proj`, and the SiLU gate applied before element-wise multiplication.
3. State the mathematical definition of SiLU and SwiGLU and distinguish them from plain gated linear units.
4. Identify which production models (Llama 3, Mixtral 8x7B, Qwen2-MoE) use SwiGLU and what that means for per-token per-expert SiLU invocations.
5. Articulate why SiLU may be more expensive than ReLU on SFPU-based architectures and frame the cost hypothesis that motivates the rest of this guide.

---

## Chapter Contents

| File | Description |
|---|---|
| [`ffn_compute_graph.md`](ffn_compute_graph.md) | Structure of dense and MoE FFN blocks; annotated SwiGLU compute graph; why SiLU is per-token per-expert |
| [`swiglu_variant.md`](swiglu_variant.md) | Mathematical definition of SwiGLU and SiLU; standalone vs fused matmul+activation; model survey |
| [`compute_role_and_cost_hypothesis.md`](compute_role_and_cost_hypothesis.md) | Why activations may be non-trivial on SFPU hardware; SiLU vs ReLU instruction cost; cost hypothesis framing |

---

## Glossary

| Term | Definition |
|---|---|
| **SiLU** | Sigmoid Linear Unit. Also called Swish-1. A smooth, non-monotonic activation function. See [`swiglu_variant.md`](swiglu_variant.md) for the full definition. |
| **SwiGLU** | Swish-Gated Linear Unit. A gated variant of the FFN in which one linear projection is passed through SiLU and used as a multiplicative gate on a second linear projection. See [`swiglu_variant.md`](swiglu_variant.md) for the full formula and derivation. |
| **gate projection** | The linear layer whose output is passed through SiLU to form the gate signal. Parameter matrix `W_gate` of shape `[d_model, d_ffn]`. |
| **up projection** | The linear layer whose output is multiplied element-wise with the gate. Parameter matrix `W_up` of shape `[d_model, d_ffn]`. Sometimes called the "value" projection in gated FFN literature. |
| **down projection** | The linear layer that projects the gated intermediate representation back to model dimension. Parameter matrix `W_down` of shape `[d_ffn, d_model]`. |
| **FFN block** | Feed-Forward Network block. In a transformer layer, the component that applies position-wise nonlinear transformations after the attention sublayer. |
| **MoE FFN** | A Mixture-of-Experts variant of the FFN block. The weight matrices are partitioned into N expert sub-matrices. The router selects K experts per token; only those K experts compute a forward pass for each token. |
| **expert dispatch** | The process of routing tokens to their assigned experts and collecting results. In hardware terms this involves tensor slicing (selecting expert weight rows/columns) and scatter/gather operations. |
| **SFPU** | Special Function Processing Unit. A compute unit on Tenstorrent Wormhole Tensix cores that handles non-linear operations (sigmoid, exp, reciprocal, etc.). SFPU instructions are serialized per tile, making non-linear activations more expensive relative to matrix multiply throughput. |
| **Tensix core** | The basic compute tile on Tenstorrent hardware. Each Tensix core contains a matrix multiply engine (FPU/MathFidelity) and an SFPU for element-wise and non-linear operations. |
| **ttnn.silu** | The TTNN API for applying SiLU element-wise to a tensor on-device. Dispatches SFPU instructions. |
| **ttnn.matmul** | The TTNN API for matrix multiplication on-device. Uses the Tensix matrix multiply engine, not the SFPU. |

---

## Next Steps

Proceed to [`ffn_compute_graph.md`](ffn_compute_graph.md) to understand where SiLU sits inside the FFN compute graph and why MoE models invoke it more frequently than dense models.
