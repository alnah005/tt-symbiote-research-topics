# Gate and Up Projection Strategy

## The SwiGLU Data Flow

Each MoE expert implements a SwiGLU feed-forward network (FFN) with three projections.
The gate (w1) and up (w3) projections compute:

```
gate_out = SiLU(x @ w1.T)      # gate path: linear then nonlinearity
up_out   = x @ w3.T            # up path: linear, no nonlinearity
inter    = gate_out * up_out   # element-wise product (SwiGLU combine)
out      = inter @ w2.T        # down projection -> residual stream
```

The key structural fact is that neither `gate_out` nor `up_out` enters the residual
stream directly. Both flow into the element-wise product `inter`, and only `inter` is
passed to the down projection (w2). This structural isolation from the residual stream
is what allows gate and up to tolerate `bfloat4_b` quantization.

## Why bfloat4_b Is Tolerable for Gate and Up

### Reason 1: SiLU Compresses Quantization Noise on the Gate Path

The gate projection output passes through the SiLU (Sigmoid Linear Unit) activation
before it reaches the multiplication:

```
SiLU(x) = x * sigmoid(x)
```

SiLU has a saturation regime for large negative inputs: as `x → -∞`,
`SiLU(x) → 0`. For moderately positive inputs, SiLU is near-linear. The result is
that the activated gate output is bounded and, crucially, extreme-magnitude deviations
caused by quantization error in `w1` are partially suppressed by this saturation
behavior.

When `bfloat4_b` weight quantization shifts a pre-activation value by a few
quantization steps, the post-SiLU deviation is smaller than it would be after a linear
operation. The nonlinearity acts as an error filter between the quantization site
(inside the matmul with `w1`) and the point where the result is consumed (the
element-wise multiplication with `up_out`).

### Reason 2: Element-Wise Product Dilutes Uncorrelated Errors

The SwiGLU combine step multiplies `gate_out` by `up_out`:

```
inter = gate_out * up_out
```

Both factors contain quantization noise when `bfloat4_b` weights are used for both
w1 and w3. However, the errors in `gate_out` and `up_out` are uncorrelated — they
arise from different weight matrices with independent quantization grids. For
uncorrelated noise sources, the error terms in the product are:

```
(gate_out + e_gate) * (up_out + e_up)
  = gate_out * up_out
  + gate_out * e_up
  + up_out * e_gate
  + e_gate * e_up       # second-order term; negligible for small errors
```

The cross-terms `gate_out * e_up` and `up_out * e_gate` do not add constructively when
the errors are uncorrelated. In practice, gates that are closed (values near zero in
`gate_out`) suppress the error contribution from the up-projection quantization noise,
and vice versa. This partial cancellation improves the effective signal-to-noise ratio
of the combined intermediate representation.

For a full empirical characterisation of the resulting PCC levels, see Chapter 4,
`bandwidth_vs_accuracy_tradeoff.md` — the gate projection at bfloat4_b + LoFi achieves
PCC ~0.971 and the up projection achieves PCC ~0.972 against their bfloat16 references.
The deeper mechanistic sensitivity analysis is in Chapter 3, `projection_sensitivity.md`.

## Recommended Configuration

### Dtype

```
ttnn.bfloat4_b
```

Store both `w1_experts` (gate) and `w3_experts` (up) in `bfloat4_b`. This achieves a
4× memory reduction compared to `bfloat16` and, in decode mode, a proportional reduction
in DRAM bandwidth consumption per matmul.

### Compute Kernel Config

```python
import ttnn

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,    # False is correct and authoritative for LoFi
    packer_l1_acc=True,
)
```

> **Warning:** Do not set `fp32_dest_acc_en=True` for the LoFi config. The authoritative
> configuration for gate and up projections uses `fp32_dest_acc_en=False`. Setting it to
> True would not match the validated configuration and would impose unnecessary compute
> overhead.

LoFi uses a single accumulation pass in the Wormhole matrix engine, which is the highest
throughput option. For gate and up projections, where the SiLU nonlinearity and
element-wise product already dominate the noise budget, the single-pass accumulation of
LoFi does not materially worsen output quality relative to HiFi2.

## Validation Criterion

Before deploying this configuration, validate the element-wise product output:

**Criterion:** PCC of `gate_out * up_out` (the `inter` tensor) against the bfloat16
reference must be **≥ 0.96**.

This threshold is set at the combined product level because that is the signal that
enters the down projection. A PCC ≥ 0.96 at the product level is consistent with the
full-layer PCC ≥ 0.97 target (see Chapter 7, `per_layer_pcc_validation.md` for the
full threshold table).

If the element-wise product PCC falls below 0.96, diagnose in this order:

1. Check weight conversion PCC: `w1` bfloat4_b weight vs. original bfloat16 weight
   should have PCC ≥ 0.97.
2. Verify that `COMPUTE_KERNEL_CONFIG_LOFI` is passed correctly to `ttnn.linear`.
3. Check for shape padding issues: weights must be tile-aligned (multiples of 32 in both
   dimensions) before dtype conversion.

## Code Pattern

```python
import ttnn
import torch

# -- Build compute kernel config for gate and up projections --
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,   # authoritative: False for LoFi
    packer_l1_acc=True,
)

def load_gate_up_weights(w1_torch, w3_torch, device):
    """Convert gate and up projection weights to bfloat4_b and place on device.

    Args:
        w1_torch: Gate projection weight, shape [d_ff, d_model], torch.bfloat16.
        w3_torch: Up projection weight, shape [d_ff, d_model], torch.bfloat16.
        device: Target TTNN device or mesh device.

    Returns:
        (w1_tt, w3_tt): bfloat4_b TTNN tensors in DRAM.
    """
    w1_tt = ttnn.as_tensor(
        w1_torch,
        dtype=ttnn.bfloat4_b,        # 4-bit block float; 4× memory reduction vs bfloat16
        layout=ttnn.TILE_LAYOUT,      # required for bfloat4_b
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w3_tt = ttnn.as_tensor(
        w3_torch,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return w1_tt, w3_tt


def swiglu_gate_up(x_tt, w1_tt, w3_tt):
    """Compute gate and up projections and return the SwiGLU intermediate.

    Args:
        x_tt: Activation tensor, shape [batch, seq, d_model], bfloat16.
        w1_tt: Gate weight, bfloat4_b, shape [d_ff, d_model] (transposed).
        w3_tt: Up weight, bfloat4_b, shape [d_ff, d_model] (transposed).

    Returns:
        inter_tt: Element-wise product gate_out * up_out, bfloat16.
    """
    # Gate path: linear then SiLU activation
    gate_pre = ttnn.linear(
        x_tt,
        w1_tt,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,  # LoFi for gate
    )
    gate_out = ttnn.silu(gate_pre)   # SiLU compresses quantization noise

    # Up path: linear only, no activation
    up_out = ttnn.linear(
        x_tt,
        w3_tt,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,  # LoFi for up
    )

    # SwiGLU combine: element-wise product
    # Uncorrelated noise in gate_out and up_out partially cancels here
    inter_tt = ttnn.mul(gate_out, up_out)
    return inter_tt
```

> **Tip:** In decode mode (batch=1, seq=1), the gate and up matmuls are memory-bound.
> Using `bfloat4_b` reduces the DRAM read volume by 4× compared to `bfloat16`, which
> directly translates to latency reduction. This is where the largest practical speedup
> from gate/up quantization is observed. See Chapter 4, `decode_memory_bandwidth.md`
> for the arithmetic intensity analysis.

## Summary

| Property | Value |
|---|---|
| Dtype | `ttnn.bfloat4_b` |
| MathFidelity | `LoFi` |
| `fp32_dest_acc_en` | `False` |
| `packer_l1_acc` | `True` |
| Validation threshold | PCC(gate_out × up_out) ≥ 0.96 |
| Memory vs. bfloat16 | 0.25× (4× reduction) |
| Decode BW vs. bfloat16 | 0.25× per projection |

## Next Steps

Continue to `down_projection_strategy.md` to understand why the down projection requires
a different dtype and a higher-fidelity compute config, based on its direct contribution
to the residual stream.
