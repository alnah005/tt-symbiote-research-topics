# bfloat16 vs. bfloat8_b vs. bfloat4_b Accuracy Comparison

## Dtype Encoding Summary

| Dtype | Sign | Mantissa Bits | Exponent | Total Bits | Notes |
|---|---|---|---|---|---|
| bfloat16 | 1 | 7 (full) | 8 (per element) | 16 | IEEE-like; standard reference |
| bfloat8_b | 1 | 7 | per block | 8 | Block FP; shared exponent per tile |
| bfloat4_b | 1 | 3 | per block | 4 | Block FP; 3-bit mantissa, high compression |

Key distinction: in bfloat8_b and bfloat4_b the exponent is shared across a block of values
(typically 32 elements in a tile row), not stored per element. This is block floating-point
(BFP), not the same as FP8/FP4 formats used in some GPU frameworks.

Dequantization always outputs **bfloat16** regardless of the input dtype. Compute accumulation
precision is a separate concern controlled by MathFidelity; see `projection_sensitivity.md`
§ Recommended Fidelity Settings.

## PCC Ranges by Dtype and Projection

For the complete PCC threshold table (per dtype, per projection, with acceptable ranges),
see `accuracy_metrics_for_moe.md` § PCC Thresholds in Practice.

## Cumulative Error in Deep Models

For the theory of cumulative residual-stream error, see `accuracy_metrics_for_moe.md`
§ Why PCC Alone Is Insufficient.

Rule of thumb: if per-layer PCC drops below 0.97 for the down projection, expect end-to-end
perplexity delta to exceed 2 PPL.

## Weight Distribution Effects

Block floating-point quantization works best when the weights within a block have a narrow
dynamic range (low kurtosis, few outliers). The shared exponent is set to cover the largest
magnitude value in the block; small values lose relative precision.

- **Expert weights after load-balancing training**: tend toward more uniform magnitudes, which
  is favorable for BFP. The load-balancing auxiliary loss encourages each expert to be activated
  with equal frequency, indirectly regularizing weight magnitudes.
- **Down-projection outliers**: the w2 weight matrix in large MoE models often has heavier
  tails than w1/w3. This makes it more susceptible to bfloat4_b quantization error because the
  shared block exponent must accommodate the outlier, wasting all 3 mantissa bits on small values.
- **High-kurtosis distributions**: outliers cause the block exponent to be set high, leaving
  only 3 mantissa bits to represent values many times smaller than the block maximum. This is
  the primary mechanism of bfloat4_b accuracy degradation.

## Code: Measure PCC Difference Between Dtypes

```python
import torch
import ttnn

def pcc_for_dtype(weight_cpu: torch.Tensor,
                  input_cpu: torch.Tensor,
                  quant_dtype: ttnn.DataType,
                  device) -> float:
    """
    Compute PCC between bfloat16 CPU reference matmul and a TTNN quantized matmul.
    weight_cpu: [out_features, in_features] bfloat16 tensor
    input_cpu:  [batch, in_features] bfloat16 tensor
    """
    # CPU bfloat16 reference
    ref = input_cpu.float() @ weight_cpu.float().T   # [batch, out_features]

    # Quantized TTNN run
    w_tt = ttnn.from_torch(weight_cpu, dtype=quant_dtype, device=device,
                           layout=ttnn.TILE_LAYOUT)
    x_tt = ttnn.from_torch(input_cpu, dtype=ttnn.bfloat16, device=device,
                           layout=ttnn.TILE_LAYOUT)
    out_tt = ttnn.linear(x_tt, w_tt)
    out = ttnn.to_torch(out_tt).float()

    # compute_pcc defined in accuracy_metrics_for_moe.md
    return compute_pcc(ref, out)


def compare_dtypes(weight_cpu, input_cpu, device):
    results = {}
    for name, dtype in [("bfloat8_b", ttnn.bfloat8_b),
                        ("bfloat4_b", ttnn.bfloat4_b)]:
        pcc = pcc_for_dtype(weight_cpu, input_cpu, dtype, device)
        results[name] = pcc
        print(f"{name}: PCC = {pcc:.6f}")
    return results
```

## Practical Decision Rule

- Use **bfloat8_b** for all projections as the default safe choice.
- Use **bfloat4_b** for gate/up projections only after validating PCC >= 0.96 on a
  representative calibration set.
- Never use **bfloat4_b** for the down projection without a full perplexity evaluation
  showing ΔPPL < 1.5 on WikiText-2.

---

**Next:** [`qwen_vs_deepseek_accuracy_comparison.md`](./qwen_vs_deepseek_accuracy_comparison.md)
