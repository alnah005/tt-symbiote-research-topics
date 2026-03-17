# SwiGLU Variant: Math, Models, and TTNN Implementation Paths

This file defines SiLU and SwiGLU mathematically, distinguishes them from related activations, and surveys which production MoE models use SwiGLU. It also distinguishes the standalone `ttnn.silu` path from fused matmul+activation configurations.

---

## SiLU: Mathematical Definition

SiLU (Sigmoid Linear Unit), also called Swish-1, is defined as:

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

Properties relevant to this guide:

- **Smooth and non-monotonic.** Unlike ReLU, SiLU has a small negative region (minimum near x ≈ -1.28, value ≈ -0.278). This requires the hardware to evaluate both the linear term and the sigmoid.
- **Requires `exp` evaluation.** `sigmoid(x) = 1 / (1 + exp(-x))` requires an exponential, which is a multi-step SFPU operation on Tenstorrent hardware.
- **Unbounded above, bounded below.** SiLU(x) → x as x → +∞ and SiLU(x) → 0 as x → -∞, giving a soft gating effect.

Comparison with related activations:

| Activation | Formula | Requires exp? | Monotonic? | Differentiable at 0? |
|---|---|---|---|---|
| ReLU | `max(0, x)` | No | Yes | No (subgradient) |
| GELU (approx) | `0.5x(1 + tanh(...))` | Yes (via tanh) | No | Yes |
| SiLU / Swish-1 | `x * sigmoid(x)` | Yes | No | Yes |
| SiLU^β (Swish-β) | `x * sigmoid(βx)` | Yes | No (β>0) | Yes |

SiLU is the β=1 special case of Swish-β. In practice "SiLU" and "Swish" are used interchangeably in transformer literature and in TTNN.

---

## SwiGLU: Mathematical Definition

SwiGLU (Swish-Gated Linear Unit) was introduced by Noam Shazeer (2020) as a drop-in replacement for the FFN activation in transformers. It is a special case of the GLU (Gated Linear Unit) family where the gating activation is SiLU.

The general GLU family is:

```
GLU_f(x, W, V) = f(xW) ⊙ (xV)
```

where `f` is an activation function, `W` and `V` are linear projections, and `⊙` is element-wise multiplication.

For SwiGLU, `f = SiLU`:

```
SwiGLU(x, W_gate, W_up) = SiLU(x @ W_gate) ⊙ (x @ W_up)
```

The full FFN using SwiGLU is:

```
FFN_SwiGLU(x) = SwiGLU(x, W_gate, W_up) @ W_down
             = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

> **Note on dimension scaling:** To keep parameter count comparable to a two-layer FFN with hidden dimension `d_ffn`, SwiGLU FFNs typically set `d_ffn_swiglu = (2/3) * d_ffn`, rounded up to a multiple of 64 or 256. This compensates for the third projection (`W_gate`) adding parameters.

For example:
- Dense FFN: W_up `[d_model, 4*d_model]` and W_down `[d_ffn, d_model]` = `[4*d_model, d_model]` — `2 * d_model * 4*d_model` parameters
- SwiGLU FFN: W_gate and W_up each `[d_model, (8/3)*d_model]`, W_down `[(8/3)*d_model, d_model]` — `3 * d_model * (8/3)*d_model` = `8*d_model^2` parameters (same)

---

## SiLU vs SwiGLU: Clarifying Terminology

When this guide says "SiLU latency," it means the latency of `ttnn.silu` (the element-wise op), not the full SwiGLU FFN block. SwiGLU is the architectural pattern that determines when and how often `ttnn.silu` is invoked — see the `index.md` glossary for term definitions.

---

## Standalone vs Fused Matmul+Activation

TTNN provides two paths for applying SiLU in a SwiGLU FFN:

### Path 1: Standalone (two separate ops)

```python
gate_hidden = ttnn.matmul(x, W_gate)          # [1, 1, T, d_ffn]
gate_hidden = ttnn.silu(gate_hidden)           # [1, 1, T, d_ffn]  ← standalone SiLU
up_hidden   = ttnn.matmul(x, W_up)            # [1, 1, T, d_ffn]
intermediate = ttnn.mul(gate_hidden, up_hidden) # [1, 1, T, d_ffn]
output       = ttnn.matmul(intermediate, W_down) # [1, 1, T, d_model]
```

This is the baseline path. The SiLU runs as a separate SFPU kernel after the matmul completes and the result has been written to L1 SRAM.

### Path 2: Fused (activation folded into matmul)

```python
gate_hidden = ttnn.matmul(
    x, W_gate,
    activation="silu"   # fuses SiLU into the matmul output stage
)
```

In the fused path, TTNN schedules the SiLU SFPU instructions immediately after the partial sum accumulation within the matmul kernel, before writing tiles back to L1. This avoids a separate kernel dispatch and a round-trip through L1 for the intermediate `gate_hidden` tensor.

> **Tip:** Fused activation is not always faster in wall-clock terms. For large matmuls, the bottleneck is the matrix engine, and the SiLU SFPU work overlaps with data movement. For small matmuls (e.g., per-expert dispatch with few tokens), the SFPU work may become the bottleneck. Measuring both paths is a key goal of this guide.

> **Warning:** Not all `ttnn.matmul` configurations support fused activation. Fused activation requires that the output compute format, tile shape, and program config are compatible with SFPU post-processing. Verify the program config explicitly when benchmarking.

---

## Model Survey: Which Models Use SwiGLU

The following production models use SwiGLU (i.e., SiLU-gated FFN) and are relevant to MoE inference on Tenstorrent hardware:

| Model | Type | N experts | K (top-K) | d_model | d_ffn_expert | Notes |
|---|---|---|---|---|---|---|
| Llama 3 8B | Dense | — | — | 4096 | 14336 | SwiGLU in every FFN layer |
| Llama 3 70B | Dense | — | — | 8192 | 28672 | SwiGLU; larger intermediate |
| Mixtral 8x7B | MoE | 8 | 2 | 4096 | 14336 | SiLU called 2x per layer |
| Mixtral 8x22B | MoE | 8 | 2 | 6144 | 16384 | SiLU called 2x per layer |
| Qwen2-MoE (57B-A14B) | MoE | 64 | 8 | 3584 | 2048 (shared) | Fine-grained MoE; SiLU called up to 8x per layer |
| DeepSeek-V2 | MoE | 160 | 6 | 5120 | 1536 | Fine-grained; shared experts + routed experts |

> **Note on Llama 3:** Dense models are relevant as a baseline. The SiLU tensor shape in a dense Llama 3 8B forward pass (`[1, 1, T, 14336]`) matches the per-expert shape in Mixtral 8x7B when all tokens route to a single expert. This makes dense Llama 3 benchmarks reusable as expert-scale calibration points.

### SwiGLU vs Plain SiLU vs Other Activations

The distinction matters for code paths:

- **SwiGLU** (Llama 3, Mixtral, Qwen2-MoE, DeepSeek-V2): SiLU is applied to `gate_proj` output, then multiplied with `up_proj` output. The `ttnn.silu` call input is a `[T, d_ffn]` tensor from a matmul.
- **Plain SiLU** (uncommon at FFN level): SiLU applied to a single projection without gating. Not used in the models above.
- **GELU** (GPT-2, BERT, early GPT-J): Different SFPU cost profile; not the focus of this guide.
- **ReLU** (original transformer, T5): Single SFPU clamp instruction; used as a cost baseline in Chapter 3.

*See Chapter 3, `baseline_measurements.md` for details on how ReLU and GELU measurements are used as calibration baselines.*

---

## Next Steps

Proceed to [`compute_role_and_cost_hypothesis.md`](compute_role_and_cost_hypothesis.md) for an analysis of why SiLU may carry non-trivial latency on SFPU-based hardware and the cost hypothesis that motivates the measurement methodology in later chapters.
