# Projection Sensitivity in SwiGLU MoE Experts

## SwiGLU FFN Structure

Each MoE expert is a three-projection SwiGLU FFN:

```
gate_out = SiLU(x @ w1.T)          # w1: gate projection
up_out   = x @ w3.T                 # w3: up projection
inter    = gate_out * up_out        # element-wise product (SwiGLU)
out      = inter @ w2.T             # w2: down projection -> residual stream
```

The expert output `out` is accumulated directly into the residual stream:
`hidden_state += router_weight * out`

## Sensitivity Ordering

```
down (w2)  >  gate (w1)  >  up (w3)
```

The down projection is most accuracy-sensitive. Gate and up projections tolerate lower fidelity
because the SiLU nonlinearity acts as an error filter before the damage reaches the residual.

## Why Down Projection Is Most Sensitive

### No Downstream Nonlinearity

The down projection output flows directly into the residual stream addition with no intervening
nonlinearity. Quantization error in `w2` becomes additive noise on the residual stream at full
magnitude.

### Error Propagation Path

```
w2 quantization error
  -> error in inter @ w2.T
  -> added to residual stream
  -> fed into next layer's LayerNorm + Attention
  -> no compression or clipping occurs
```

Because LayerNorm normalizes the residual stream, it can amplify small deviations if the
mean-shift from quantization noise is large.

### Residual Stream Injection Is Cumulative

With 28 MoE layers, the down-projection quantization error accumulates additively across layers.
Each layer contributes a fresh error term to the same residual stream vector.

## Why Gate/Up Projections Tolerate Lower Fidelity

### SiLU as Error Filter

The SiLU activation applied to the gate projection output compresses extreme values:

```
SiLU(x) = x * sigmoid(x)
```

For large positive `x`, SiLU saturates toward `x` (near-linear). For large negative `x`, SiLU
suppresses the output toward 0. Quantization errors that produce large-magnitude deviations in
`gate_out` are partially absorbed by this saturation behavior.

### Element-Wise Product Dilution

The intermediate `gate_out * up_out` means that error in either `gate_out` or `up_out` is
multiplied by the other factor. If the gate output is near zero for a given activation (a
"closed" gate), then the error contribution from up-projection quantization is also suppressed.

## Recommended Fidelity Settings

| Projection | Dtype | MathFidelity | Rationale |
|---|---|---|---|
| down (w2) | bfloat8_b | HiFi2 | Highest sensitivity; 2 accumulation passes |
| gate (w1) | bfloat4_b or bfloat8_b | LoFi | SiLU absorbs error; 1 accumulation pass |
| up (w3) | bfloat4_b or bfloat8_b | LoFi | Gate product dilutes error; 1 accumulation pass |

MathFidelity controls the number of accumulation passes in the matrix engine, not the dtype:
- **LoFi**: 1 pass (fastest, least precise accumulation)
- **HiFi2**: 2 passes (recommended for down projection)
- **HiFi4**: 4 passes (reference quality; rarely needed for inference)

These settings are independent: dtype controls weight storage precision; MathFidelity controls
the compute accumulation precision. Dequantization always outputs bfloat16 regardless of input dtype.

## Empirical Evidence from DeepSeek-V3

The sensitivity ordering (down > gate > up) is empirically confirmed by DeepSeek-V3, which uses
w2 = bfloat8_b + HiFi2 and w1/w3 = bfloat4_b + LoFi, validated at full-model PCC ~0.97 relative
to bfloat16 reference. For per-model detail and the architectural rationale behind this choice,
see `qwen_vs_deepseek_accuracy_comparison.md` § DeepSeek-V3 Quantization Strategy.

## Code: Sensitivity Probe

```python
import torch
import ttnn

def measure_projection_pcc(x, w, quant_dtype, math_fidelity, device):
    """Measure PCC of a single projection under a given dtype and fidelity."""
    # Reference: bfloat16 matmul on CPU
    ref_out = x.float() @ w.float().T

    # Quantize weight and run on device
    w_tt = ttnn.from_torch(w, dtype=quant_dtype, device=device)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    prog_cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=1,
    )
    compute_cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=math_fidelity)
    out_tt = ttnn.matmul(x_tt, w_tt, program_config=prog_cfg,
                         compute_kernel_config=compute_cfg)
    out = ttnn.to_torch(out_tt).float()

    # compute_pcc defined in accuracy_metrics_for_moe.md
    return compute_pcc(ref_out, out)
```
