# Fidelity Selection Workflow

## Overview

Choosing `math_fidelity` for a new MoE model is a structured step-down process. The goal is to find the lowest fidelity level that keeps PCC above your deployment quality threshold for each projection type. Starting from the top (HiFi4) and stepping down guarantees you have a validated baseline before introducing any precision reduction.

---

## The Workflow

### Step 1: Establish a float32 reference

Run the expert matmul in PyTorch using `torch.matmul` with float32 weights and activations. This is your PCC ground truth — the output that all fidelity configurations will be measured against.

```python
ref_output = torch.matmul(activation.float(), weight.float())
```

Use realistic inputs: activations from a real or representative prompt (not all-zeros or random uniform), and weights loaded from a checkpoint or initialized with the model's standard scheme. The choice of inputs affects the PCC estimate, so representative data is important.

### Step 2: Run with HiFi4

Run the same inputs through TTNN with `math_fidelity=ttnn.MathFidelity.HiFi4`. Compute PCC vs the float32 reference.

**Expected result:** PCC > 0.9999.

**If PCC is not > 0.9999 at HiFi4:** stop and investigate before proceeding. A HiFi4 failure indicates a non-fidelity issue — dtype mismatch, incorrect layout, weight initialization problem, or a bug in how the op is being called. Fix the underlying issue before stepping down to lower fidelity.

### Step 3: Step down to HiFi2

Run the same inputs with `math_fidelity=ttnn.MathFidelity.HiFi2`. Measure PCC vs the float32 reference.

**If PCC >= 0.999:** HiFi2 is acceptable for this projection. Proceed to Step 4.

**If PCC < 0.999:** Use HiFi4 for this projection. Do not continue stepping down.

### Step 4: Step down to LoFi

Run with `math_fidelity=ttnn.MathFidelity.LoFi`. Measure PCC.

**If PCC >= 0.999:** Use LoFi for this projection — you have reached the maximum throughput configuration while remaining within the accuracy threshold.

**If PCC < 0.999:** Step back up to HiFi2. LoFi is not acceptable for this projection.

### Step 5: Accept the lowest passing fidelity

The lowest fidelity level that keeps PCC above your deployment threshold is the correct choice for this projection. Record the result per projection type (gate, up, down) since they typically differ.

---

## Python Template: Parameterized PCC Sweep

```python
import ttnn
import torch

def measure_pcc_for_fidelity(device, a, b, fidelity):
    """Measure PCC of ttnn.matmul(a, b) with given math_fidelity vs float32 reference."""
    ref = torch.matmul(a.float(), b.float())

    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_tt = ttnn.matmul(a_tt, b_tt, compute_kernel_config=config)
    out = ttnn.to_torch(out_tt).float()

    out_flat = out.flatten()
    ref_flat = ref.flatten()
    return torch.corrcoef(torch.stack([out_flat, ref_flat]))[0, 1].item()


fidelities = [
    ("HiFi4", ttnn.MathFidelity.HiFi4),
    ("HiFi2", ttnn.MathFidelity.HiFi2),
    ("LoFi",  ttnn.MathFidelity.LoFi),
]
for name, fid in fidelities:
    score = measure_pcc_for_fidelity(device, activation, weight, fid)
    print(f"{name}: PCC = {score:.6f}")
```

Run this sweep separately for each projection type using the corresponding activation and weight tensors. Record the results to build your per-projection fidelity configuration.

> **Warning:** The template above sets `fp32_dest_acc_en=False`. For down projections, you will typically also want to test with `fp32_dest_acc_en=True`, since residual stream accumulation is sensitive to accumulator precision as well as multiply precision. Chapter 3 covers `fp32_dest_acc_en` in depth; run the fidelity sweep first, then confirm the combined config.

---

## Warning About Small Batch Sizes

PCC on a single decode token (M=1, shape `[1, d_model]`) has high variance due to the small number of elements contributing to the correlation estimate. A single outlier element can shift the measured PCC by several thousandths.

Always test with M >= 128 for reliable PCC estimates. For decode-mode inference where M=1 is the target deployment shape, you must still run your PCC validation sweep at M >= 128 and trust those results. The numerical properties of the matmul kernel do not change between M=1 and M=128; only the statistical reliability of the PCC estimate changes.

---

## Decision Threshold Guidance

| PCC range | Interpretation |
|---|---|
| > 0.9995 | Safe for production deployment |
| 0.999–0.9995 | Acceptable for most use cases; run an end-to-end downstream task check before shipping |
| < 0.999 | Do not use this fidelity level for this projection in production |

These thresholds are general guidelines. The appropriate PCC threshold depends on the model architecture, task sensitivity, and your quality bar. A model doing summarization may tolerate more residual stream drift than a model doing code generation or arithmetic reasoning.

> **Tip:** Before finalizing any fidelity choice in production, run an end-to-end generation quality check — perplexity on a held-out set, or a downstream task metric (BLEU, HumanEval pass@k, GSM8K accuracy). PCC > 0.999 at the projection level is a necessary condition, not a sufficient one, for production quality.

---

## Summary: Per-Projection Fidelity Map for DeepSeek-V3 Expert Shapes

| Projection | K tiles (K_t) | Recommended Fidelity | Typical PCC at that fidelity |
|---|---|---|---|
| Gate (w1) | 224 (d_model=7168) | LoFi | > 0.999 |
| Up (w3) | 224 (d_model=7168) | LoFi | > 0.999 |
| Down (w2) | 64 (d_ff=2048) | HiFi2 | > 0.999 |

Run the workflow above to confirm these recommendations hold for your model's weight initialization, input distribution, and quality requirements before deploying.

> **Note:** For the complete production-ready implementation including `packer_l1_acc` and `math_approx_mode` assignments, see Chapter 5, `index.md`.

---

**Next:** [Chapter 3 — `packer_l1_acc` — Throughput Effect](../ch3_packer_l1_acc/index.md)
