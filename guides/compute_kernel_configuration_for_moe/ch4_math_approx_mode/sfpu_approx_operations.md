# SFPU Approximation Operations

## What the SFPU Is

Each Tensix core on Wormhole B0 contains two distinct compute units:

- **FPU (Floating Point Unit)**: handles matrix multiply and dot-product; operates on tiles using multiply-accumulate (MAC) chains. `math_fidelity` (LoFi / HiFi2 / HiFi4) controls how many accumulation passes the FPU performs.
- **SFPU (Special Function Processing Unit)**: a scalar unit that handles non-linear and transcendental functions element-by-element within a tile. `math_approx_mode` controls the SFPU computation strategy.

The FPU and SFPU are on separate datapaths. `math_approx_mode` touches only the SFPU; it has no interaction with the FPU's MAC accumulation.

Wormhole B0 has 80 Tensix cores arranged in an 8x10 grid. Every core has its own SFPU, so SFPU throughput scales with core count just as FPU throughput does.

## Operations Routed Through the SFPU

When `math_approx_mode=True`, the following ops use piecewise polynomial lookup instead of iterative Newton-Raphson refinement:

| Op | Typical Use Site | Approx Error |
|---|---|---|
| `exp` | softmax numerator and denominator | ~0.1–0.2% relative |
| `reciprocal` | layer norm, softmax denominator | ~0.1–0.2% relative |
| `sqrt` | norm computations | ~0.1–0.2% relative |
| `sigmoid` | gating, some attention variants | ~0.1–0.2% relative |
| `gelu` | FFN activation (BERT-style) | ~0.2–0.3% relative (compound) |
| `silu` | SwiGLU gate activation in MoE FFN | ~0.1–0.2% relative |

The approximation error quoted above is per single evaluation on inputs with typical magnitudes. It does not account for accumulation across sequence dimensions.

## How the Approximation Works

**Approximate path (`math_approx_mode=True`):**
Inputs are range-reduced and looked up in a piecewise polynomial table stored in SFPU registers. A short polynomial (degree 2–3 depending on op) is evaluated with fused multiply-add. This is 1–2 clock cycles per element.

**Exact path (`math_approx_mode=False`):**
The SFPU runs iterative Newton-Raphson refinement steps after the initial approximation, converging to a result within 1 ULP of the IEEE754 result. This adds 2–4 extra cycles per element.

```
# Pseudocode for SFPU exp evaluation
# approx_mode=True:
  y = polynomial_lookup(x_reduced)          # 1 polynomial eval

# approx_mode=False:
  y = polynomial_lookup(x_reduced)          # 1 polynomial eval
  y = y * (1 + correction(x_reduced, y))   # Newton-Raphson step 1
  y = y * (1 + correction(x_reduced, y))   # Newton-Raphson step 2
```

## Operations NOT Affected by math_approx_mode

Pure matmul and linear layers use the FPU path exclusively:

```python
# math_approx_mode has zero effect on this operation's output or speed
out = ttnn.matmul(
    input, weight,
    compute_kernel_config=WormholeComputeKernelConfig(
        math_fidelity=MathFidelity.LoFi,
        math_approx_mode=True,   # does nothing for matmul
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
)
```

Setting `math_approx_mode=True` on a pure matmul kernel does not change outputs, does not change throughput, and does not raise an error. The flag is simply ignored by the FPU datapath.

## Why COMPUTE_KERNEL_CONFIG_LOFI Sets math_approx_mode=False

The LoFi config is used for gate (w1) and up (w3) projections in MoE. These are pure matmuls — no fused transcendental activation at the SFPU level. Because the SFPU is not exercised, `math_approx_mode` is irrelevant. Setting it to `False` is conservative and makes intent explicit: this kernel is not using SFPU approximations.

See `index.md` § Key Config Reference for the canonical `COMPUTE_KERNEL_CONFIG_LOFI` definition.

## Why COMPUTE_KERNEL_CONFIG_HIFI2 Sets math_approx_mode=True

The HiFi2 config is shared across the broader model, including attention softmax and layer norm, both of which use SFPU ops (`exp`, `reciprocal`). The throughput gain from polynomial approximation is measurable in these fused kernels, and the error is within acceptable bounds for bfloat16 accumulation. Setting `math_approx_mode=True` is intentional for this config.

See `index.md` § Key Config Reference for the canonical `COMPUTE_KERNEL_CONFIG_HIFI2` definition.
