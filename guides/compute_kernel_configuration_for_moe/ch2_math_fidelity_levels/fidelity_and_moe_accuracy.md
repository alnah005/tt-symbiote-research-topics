# Math Fidelity and MoE Accuracy: Why Different Projections Need Different Levels

## Overview

The three linear projections inside a MoE FFN expert — gate (w1), up (w3), and down (w2) — have different numerical characteristics. Understanding why gate and up projections tolerate LoFi while down projections require HiFi2 requires examining: what the output of each projection is used for, how many K tiles accumulate into each output element, and whether there is a nonlinearity between the projection and the residual stream that can absorb rounding error.

---

## Why Gate Projections Tolerate LoFi

The gate projection computes `gate_proj(x) = x W1^T`, producing a matrix of logit-like values that immediately feeds into the SiLU gate activation:

```
gate_output = silu(gate_proj(x))
```

SiLU is defined as `silu(z) = z * sigmoid(z)`. It is a soft saturation function: it compresses large-magnitude values and applies a smooth gating effect near zero. Small mantissa errors in the gate logits — on the order of LoFi rounding — are largely absorbed by this compression. The saturation region of SiLU is forgiving of small input perturbations because the output sensitivity (`d(silu)/dz`) is bounded and approaches zero for large `|z|`.

The practical effect is that LoFi rounding errors in the gate projection output produce negligible change in `gate_output` after the SiLU is applied. Empirically, for standard MoE configurations (e.g., DeepSeek-V3 expert shapes), gate_proj with LoFi vs HiFi2 typically yields PCC > 0.999 for the post-SiLU output.

---

## Why Up Projections Tolerate LoFi

The up projection computes `up_proj(x) = x W3^T`. Its output is multiplied element-wise with the SiLU gate output:

```
ffn_output = silu(gate_proj(x)) * up_proj(x)
```

The element-wise product mixes the up_proj error with the SiLU gate output. The SiLU output is already bounded (the sigmoid component constrains values to a limited range), which in turn limits how much the up_proj rounding error can amplify in the product. The composite error from both projections remains bounded.

Empirically: up_proj with LoFi yields the same PCC range as gate_proj — typically PCC > 0.999 vs a float32 reference, measured on the element-wise product output.

---

## Why Down Projections Need HiFi2

The down projection computes `down_proj(ffn_output) = ffn_output W2^T`, accumulating the full `d_ff` dimension into each output element. For DeepSeek-V3 experts, `d_ff = 2048`, which means `K_t = 2048 / 32 = 64` K-tiles accumulate into each output element.

The primary factor making the down projection accuracy-sensitive is direct residual stream injection:

**Direct residual stream injection.** The down projection output is added directly to the residual stream — it is not followed by any nonlinearity before the next layer's normalization and attention. There is no SiLU, ReLU, or other compressive function to absorb accumulated rounding error. Any per-tile LoFi rounding error flows into every subsequent layer without a reset.

Gate and up projections benefit from SiLU absorption as established above; down projections have no equivalent absorber.

Empirically: down_proj at LoFi yields PCC ~0.99 vs a float32 reference for DeepSeek-V3 expert shapes. Switching to HiFi2 restores PCC > 0.999.

> **Warning:** A PCC of ~0.99 on a single expert's down projection may seem acceptable in isolation, but across 256 experts per forward pass (as in DeepSeek-V3), the accumulated residual stream drift can meaningfully degrade end-to-end model quality. Always validate at the model output level, not just per-projection.

---

## K-Loop Depth and Nonlinearity Absorption

As a general heuristic, PCC degradation from LoFi scales roughly with `sqrt(K_t)` — a consequence of the central limit theorem applied to per-tile rounding errors that accumulate like approximately independent noise. However, this heuristic applies only when K-loop depth is actually the dominant sensitivity driver. For MoE projections, downstream nonlinearity is the more important factor.

Concrete comparison for DeepSeek-V3 shapes:

| Matmul | K dimension | K_t (K / 32) | LoFi sensitivity |
|---|---|---|---|
| Gate/Up projections | d_model = 7168 | 224 | Lower: SiLU absorbs accumulated error |
| Down projection | d_ff = 2048 | 64 | Higher: residual stream, no nonlinearity |

Gate/up projections have a deeper K-loop (K_t=224 vs K_t=64) — under the sqrt(K_t) heuristic they accumulate more rounding noise (sqrt(224) ≈ 15 units vs sqrt(64) = 8). Despite this, gate/up projections are less fidelity-sensitive, because the SiLU nonlinearity absorbs and compresses the accumulated error before it reaches the residual stream. The down projection has fewer K-tiles but no such absorber: its output is injected directly into the residual stream, making it the more sensitive operation regardless of K-loop depth.

> **Tip:** When evaluating fidelity sensitivity, the presence or absence of a downstream nonlinearity dominates K-loop depth. A projection with K_t=64 feeding directly into the residual stream is more fidelity-sensitive than one with K_t=224 feeding a SiLU gate.

---

## How to Measure PCC in Python

PCC (Pearson Correlation Coefficient) measures the linear correlation between the reference output and the TTNN output. A PCC of 1.0 indicates perfect correlation; values below ~0.999 indicate meaningful numerical divergence for production inference.

```python
import torch

# ref is float32 output from PyTorch reference
# ttnn_out is BF16 output moved to CPU via ttnn.to_torch()
def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()

score = pcc(ref_output, ttnn_output)
print(f"PCC: {score:.6f}")
```

PCC captures proportionality errors (scaling or offset between reference and output) as well as element-wise deviation, making it a more sensitive diagnostic than per-element max absolute error for catching subtle fidelity regressions.

---

## Note on PCC Test Reliability

See the PCC reliability note in `fidelity_selection_workflow.md` before finalizing any measurement.

---

## Next Steps

Read `fidelity_selection_workflow.md` for the step-by-step process to apply these principles to a new MoE model: how to establish a float32 reference baseline, step down through fidelity levels, and identify the lowest safe fidelity for each projection type.
