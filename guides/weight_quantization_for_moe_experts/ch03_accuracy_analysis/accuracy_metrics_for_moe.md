# Accuracy Metrics for MoE Expert Quantization

## PCC: Primary Correctness Metric

Pearson Correlation Coefficient (PCC) measures linear similarity between two tensors. In TTNN
it is the standard metric for verifying that a quantized op output is numerically close to the
bfloat16 CPU reference. PCC is scale-invariant: a uniform gain error (e.g., a missing scale
factor) does not affect PCC, which makes it a reliable signal for quantization noise independent
of any constant offset.

### Definition

For two flat vectors `a` and `b`:

```
PCC(a, b) = cov(a, b) / (std(a) * std(b))
```

PCC ranges from -1 (perfectly anti-correlated) to +1 (perfectly correlated). Values above 0.99
indicate negligible quantization noise at the layer level.

### Computing PCC with PyTorch

`torch.corrcoef` returns a 2×2 matrix for a stack of two vectors. The off-diagonal element
`[0, 1]` (or `[1, 0]`) is the PCC.

```python
import torch

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return PCC between two tensors of any shape."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    # torch.corrcoef returns 2x2 matrix; [0,1] is the cross-correlation
    corr_matrix = torch.corrcoef(torch.stack([a_flat, b_flat]))
    return corr_matrix[0, 1].item()

# Example
ref  = torch.randn(4096, 1024)
quant_out = ref + 0.01 * torch.randn_like(ref)   # simulate small quantization noise
pcc = compute_pcc(ref, quant_out)
print(f"PCC: {pcc:.6f}")   # Expected: > 0.999 for this noise level
```

### PCC Thresholds in Practice

| Dtype | Projection | Minimum Acceptable PCC | Notes |
|---|---|---|---|
| bfloat16 | all | > 0.999 | Reference; used to validate bf16 ops |
| bfloat8_b | gate/up | 0.98–0.99 | Acceptable for production |
| bfloat8_b | down | 0.975–0.985 | Acceptable for production |
| bfloat4_b | gate/up | 0.96–0.98 | Validate per model family |
| bfloat4_b | down | 0.94–0.97 | High risk; prefer bfloat8_b |

## Perplexity Delta: Secondary Metric

Perplexity delta (ΔPPL) measures the end-to-end language modelling degradation caused by
quantization. It propagates the cumulative effect of per-layer quantization errors across all
MoE layers.

Expected deltas relative to bfloat16 baseline:

- **bfloat8_b** (all projections): < 1 PPL on WikiText-2 / C4.
- **bfloat4_b** (gate/up only, down in bfloat8_b): 0.5–2 PPL; model-dependent.
- **bfloat4_b** (all projections): 1–3 PPL; only acceptable for heavily quantization-aware
  trained models (e.g., DeepSeek-V3).

### Task Sensitivity

Code generation (MBPP, HumanEval) and multi-step reasoning (MATH, GSM8K) degrade faster than
open-ended generation because they require low-entropy token distributions at critical positions.
A perplexity delta of 1–2 PPL can correspond to a 3–6 point drop on HumanEval pass@1.

## Why PCC Alone Is Insufficient

A PCC of 0.97 at a single layer compounds across depth. For a 60-layer transformer with 28 MoE
layers, if each MoE layer output has PCC 0.97 relative to its bfloat16 counterpart, the
effective degradation to the final hidden state is not simply 1 - 0.97 = 3%. Residual stream
additions accumulate quantization error additively, not multiplicatively, and attention layers
can amplify or suppress the error unpredictably.

**Validation protocol**: always measure both per-layer PCC AND end-to-end perplexity delta.
Per-layer PCC tells you where the problem is; perplexity delta tells you whether it matters.

---

**Next:** [`projection_sensitivity.md`](./projection_sensitivity.md)
