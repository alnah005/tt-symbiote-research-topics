# Down Projection Strategy

## The Down Projection's Role in the Residual Stream

The down projection (w2) maps the SwiGLU intermediate back to the model's hidden
dimension and adds the result directly to the residual stream:

```
inter    = gate_out * up_out           # from gate and up projections
w2_out   = inter @ w2.T               # down projection: [batch, d_ff] -> [batch, d_model]
hidden  += router_weight * w2_out     # accumulate into residual stream
```

This is the critical structural difference from the gate and up projections. The output
of w2 is not filtered by any nonlinearity before it reaches the residual stream. Whatever
quantization noise is introduced by w2 is injected directly — at full magnitude — into
the token representation vector that feeds into the next layer's LayerNorm and attention.

## Why bfloat4_b Is Not Adequate for the Down Projection

### No Downstream Nonlinearity to Clip Errors

The gate projection's quantization noise is partially absorbed by SiLU before it
contributes to `inter`. The down projection has no such absorber. The data flow after
w2 is:

```
w2 quantization error
  -> error in inter @ w2.T
  -> added directly to residual stream (no clipping)
  -> enters next layer's LayerNorm
  -> LayerNorm normalizes the stream; mean-shift from quantization noise can be amplified
  -> enters next layer's self-attention Q/K/V projections
```

At `bfloat4_b` precision, the down projection PCC falls to approximately 0.941–0.943
against the bfloat16 reference, as documented in Chapter 4,
`bandwidth_vs_accuracy_tradeoff.md`. This places the down projection at or just above
the 0.94 acceptable floor, which causes measurable perplexity degradation for most
production deployments.

### Residual Stream Errors Accumulate Across Layers

Qwen 235B-A22B has MoE layers distributed throughout a 94-layer transformer. Each MoE
layer's down-projection error is a fresh additive term injected into the same residual
stream. There is no mechanism that resets or compresses the accumulated error between
layers. With per-layer PCC of 0.941 for the down projection, the compounding across
all MoE layers produces an end-to-end deviation that exceeds typical quality budgets.

The full sensitivity analysis establishing this ordering — down (w2) > gate (w1) > up
(w3) — is in Chapter 3, `projection_sensitivity.md`.

### LayerNorm Amplification

Because LayerNorm re-normalises the residual stream to zero mean and unit variance, a
systematic bias introduced by quantization noise in w2 is not passively ignored. If the
noise has a non-zero mean component — which block floating-point quantization can
introduce for weight matrices with non-symmetric distributions — LayerNorm will shift
the normalisation statistics, potentially amplifying the downstream effect.

## Recommended Configuration

### Dtype

```
ttnn.bfloat8_b
```

`bfloat8_b` is block floating-point with 8 bits per element and a shared exponent per
32×32 tile. It achieves 2× memory reduction compared to `bfloat16` and approximately
2× DRAM bandwidth reduction in decode mode, while maintaining PCC ~0.977 for the down
projection — well above the 0.975 validation threshold.

### Compute Kernel Config

```python
import ttnn

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,    # False is correct and authoritative for HiFi2
    packer_l1_acc=True,
)
```

> **Warning:** Do not set `fp32_dest_acc_en=True` for HiFi2. The authoritative
> configuration for the down projection uses `fp32_dest_acc_en=False`. This matches
> the validated DeepSeek-V3 configuration from which these recommendations are derived.

### Why HiFi2 Is Needed for the Down Projection

HiFi2 uses two accumulation passes in the Wormhole matrix engine, compared to LoFi's
single pass. The two-pass accumulation reduces rounding error in the partial-sum
accumulation path. For the down projection, which already carries `bfloat8_b` weight
quantization noise, adding single-pass (LoFi) accumulation noise compounds the error
further. HiFi2 keeps the accumulation-induced component of the total error small
relative to the weight quantization component.

The empirical consequence: `bfloat8_b` down projection at LoFi achieves PCC ~0.963,
which is below the 0.975 threshold. Upgrading to HiFi2 raises this to ~0.977. The
two-pass accumulation is therefore not an optional refinement — it is required for the
down projection to clear the validation threshold when using `bfloat8_b` weights.

See Chapter 4, `bandwidth_vs_accuracy_tradeoff.md` for the full PCC table comparing
LoFi and HiFi2 at bfloat8_b for the down projection.

## Validation Criterion

**Criterion:** PCC of `w2_out` against the bfloat16 reference must be **≥ 0.975**.

This threshold is tighter than the gate/up threshold (≥ 0.96) because errors in w2_out
propagate without compression into subsequent layers.

### Fallback Options If PCC Falls Below 0.975

| Observed PCC | Action |
|---|---|
| 0.970 – 0.975 | Switch to `bfloat8_b` + HiFi4 (`MathFidelity.HiFi4`, four accumulation passes) |
| 0.960 – 0.970 | Consider `bfloat16` for the down projection only; keep gate/up at `bfloat4_b` |
| Below 0.960 | Check weight conversion correctness; verify tile alignment and memory config |

> **Tip:** For Qwen 235B-A22B specifically, the model was trained in bfloat16 without
> quantization-aware training. If down projection PCC is marginal after quantization,
> the conservative option — bfloat8_b + HiFi2 — almost always clears the 0.975 threshold.
> Only consider HiFi4 or bfloat16 fallback if the calibration perplexity delta exceeds
> your budget after validating with HiFi2.

## Code Pattern

```python
import ttnn

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,    # authoritative: False for HiFi2
    packer_l1_acc=True,
)

def load_down_weights(w2_torch, device):
    """Convert down projection weight to bfloat8_b and place on device.

    Args:
        w2_torch: Down projection weight, shape [d_model, d_ff], torch.bfloat16.
                  (Transposed form: [d_model, d_ff] so matmul produces [batch, d_model].)
        device: Target TTNN device or mesh device.

    Returns:
        w2_tt: bfloat8_b TTNN tensor in DRAM.
    """
    w2_tt = ttnn.as_tensor(
        w2_torch,
        dtype=ttnn.bfloat8_b,         # 8-bit block float; 2× memory reduction vs bfloat16
        layout=ttnn.TILE_LAYOUT,       # required for bfloat8_b
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return w2_tt


def expert_down_projection(inter_tt, w2_tt):
    """Apply down projection to the SwiGLU intermediate.

    Args:
        inter_tt: SwiGLU intermediate, shape [batch, seq, d_ff], bfloat16.
        w2_tt: Down weight, bfloat8_b, shape [d_model, d_ff] (transposed for matmul).

    Returns:
        w2_out: Down projection output, shape [batch, seq, d_model], bfloat16.
                This tensor is added to the residual stream by the caller.
    """
    # HiFi2 two-pass accumulation mitigates rounding error on top of bfloat8_b noise
    w2_out = ttnn.linear(
        inter_tt,
        w2_tt,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
    )
    return w2_out
```

## Summary

| Property | Value |
|---|---|
| Dtype | `ttnn.bfloat8_b` |
| MathFidelity | `HiFi2` |
| `fp32_dest_acc_en` | `False` |
| `packer_l1_acc` | `True` |
| Validation threshold | PCC(w2_out) ≥ 0.975 |
| Memory vs. bfloat16 | 0.5× (2× reduction) |
| Decode BW vs. bfloat16 | 0.5× |
| Fallback if PCC < 0.975 | bfloat8_b + HiFi4, or bfloat16 for down projection only |

## Next Steps

Continue to `mixed_precision_memory_layout.md` to see how gate, up, and down projection
weight tensors with different dtypes are organised in a single MoE module, including the
total DRAM footprint calculation for Qwen 235B-A22B on a T3K system.
