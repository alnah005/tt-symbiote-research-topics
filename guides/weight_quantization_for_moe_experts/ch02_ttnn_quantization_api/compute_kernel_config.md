# Compute Kernel Configuration

`WormholeComputeKernelConfig` is the TTNN object that controls the arithmetic precision of matmul operations on Wormhole (Tenstorrent's Wormhole architecture) Tensix cores. When using quantized weight dtypes, the kernel config determines how much precision is recovered during the dequantize-and-multiply step inside the matmul kernel.

This page explains each field, the available `MathFidelity` levels, and how to construct configs for the two precision modes most relevant to expert weight matmuls: LoFi (low fidelity) and HiFi2 (high fidelity level 2).

## The Four Key Fields

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

cfg = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,  # FPU multiply-accumulate precision
    math_approx_mode=False,             # SFPU approximation toggle
    fp32_dest_acc_en=True,              # accumulate to fp32 before packing
    packer_l1_acc=True,                 # accumulate partials in L1
)
```

### `math_fidelity`

Controls the precision of the FPU (Floating Point Unit) multiply-accumulate (MAC) operations. This field applies **only to the FPU datapath** — it does not affect SFPU (Special Function Processing Unit) operations such as softmax, GELU, or layernorm.

| MathFidelity | Description | Accumulation Passes | Relative Speed |
|---|---|---|---|
| `LoFi` | Single-pass accumulation; fewest passes through the MAC pipeline | 1 | Fastest |
| `HiFi2` | Two-pass accumulation; good balance of speed and accuracy | 2 | Medium |
| `HiFi3` | Three-pass accumulation; higher precision than HiFi2 | 3 | Slower |
| `HiFi4` | Four-pass accumulation; full hardware precision | 4 | Slowest |

The fidelity level trades arithmetic accuracy for throughput by controlling how many accumulation passes the FPU performs per MAC. For quantized weight inference, the precision bottleneck is usually the weight format (bfloat4_b has 3 mantissa bits), so using HiFi4 math fidelity on a bfloat4_b weight provides no measurable accuracy benefit over HiFi2 while paying a throughput penalty.

> **Warning:** `math_fidelity` affects FPU MACs only. If your kernel dispatches SFPU operations (activation functions, normalization), their precision is governed by `math_approx_mode`, not by `math_fidelity`.

### `math_approx_mode`

A boolean that enables faster approximate implementations of SFPU transcendental functions (exponential, reciprocal, square root, etc.). When `True`, these functions use polynomial approximations that are faster but less accurate than the full iterative implementations.

For matmul-only kernels (no activation function fused into the same kernel), `math_approx_mode` has no effect on output accuracy because the SFPU is not exercised. It is safe to leave it `False` unless you are profiling a fused matmul-activation kernel and need maximum SFPU throughput.

### `fp32_dest_acc_en`

When `True`, the matmul accumulates partial sums into a 32-bit floating-point (fp32) destination register before packing the result back to bfloat16. When `False`, accumulation uses bfloat16 throughout.

This field matters most when using quantized weight dtypes. The dequantized weight values are bfloat16, but accumulating many bfloat16 products without fp32 intermediate storage can produce visible rounding errors in the dot products. Enabling `fp32_dest_acc_en=True` mitigates this by preserving more precision during accumulation.

> **Note:** For the interaction between `fp32_dest_acc_en` and output buffer sizing, see `dtype_in_linear_and_matmul.md`.

### `packer_l1_acc`

When `True`, the packer accumulates partial results in L1 (level-1 SRAM) before writing the final output to DRAM. This can improve effective precision for very long reduction dimensions by avoiding repeated bfloat16 round-trips through DRAM.

For typical expert weight matrix sizes, the benefit of `packer_l1_acc` is modest. It is generally safe to set it to `True` when `fp32_dest_acc_en=True` and leave it `False` for LoFi configs where throughput is the primary objective.

## Standard Configurations

### LoFi Config

Use LoFi when maximum throughput is the goal and the weight format already limits precision (such as bfloat4_b).

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

lofi_config = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,   # single-pass accumulation, fastest MAC
    math_approx_mode=True,              # enable approximate SFPU if used
    fp32_dest_acc_en=False,             # bfloat16 accumulation is sufficient at LoFi
    packer_l1_acc=False,                # no L1 accumulation needed
)
```

Rationale: bfloat4_b weights have 3 mantissa bits (1 sign + 3 mantissa = 4 total bits per element). Accumulating with full fp32 precision does not recover precision that was lost during quantization. The throughput gain from LoFi math fidelity and bfloat16 accumulation is the primary benefit when using bfloat4_b.

### HiFi2 Config

Use HiFi2 when you need better accuracy than LoFi and the weight format retains more information (such as bfloat8_b).

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

hifi2_config = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,  # intermediate FPU precision
    math_approx_mode=False,             # full SFPU precision
    fp32_dest_acc_en=True,              # accumulate to fp32 for better dot product accuracy
    packer_l1_acc=True,                 # accumulate partials in L1
)
```

Rationale: bfloat8_b weights have 7 mantissa bits (1 sign + 7 mantissa = 8 total bits per element), significantly more than bfloat4_b. The additional precision in the weight format is worth preserving through higher-fidelity accumulation. `fp32_dest_acc_en=True` is the critical setting here — without it, accumulation rounding can degrade Pearson Correlation Coefficient (PCC) scores noticeably for longer reduction dimensions.

### HiFi4 Config (Reference)

HiFi4 is the full hardware precision mode, equivalent to running unconstrained bfloat16 matmuls. It is commonly used for attention SDPA (Scaled Dot-Product Attention) computations where high numerical fidelity is critical.

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

hifi4_config = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi4,  # full hardware precision
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

HiFi4 is not typical for expert weight matmuls with quantized dtypes. The throughput cost is high and the accuracy benefit over HiFi2 is negligible when the weight tensor is already quantized to bfloat8_b. It is documented here for completeness and for use in reference implementations.

## Passing Config to `ttnn.linear`

The compute kernel config is passed as a keyword argument. The weight dtype and the kernel config are independent; you can combine any dtype with any config.

```python
import ttnn

# lofi_config and hifi2_config are defined in the Standard Configurations section above.

# Forward pass
def expert_forward(x, weight_bfloat4b, weight_bfloat8b):
    # weight_bfloat4b was converted with dtype=ttnn.bfloat4_b
    out_lofi = ttnn.linear(
        x,
        weight_bfloat4b,
        compute_kernel_config=lofi_config,  # LoFi for bfloat4_b weight
    )

    # weight_bfloat8b was converted with dtype=ttnn.bfloat8_b
    out_hifi2 = ttnn.linear(
        x,
        weight_bfloat8b,
        compute_kernel_config=hifi2_config,  # HiFi2 for bfloat8_b weight
    )
    return out_lofi, out_hifi2
```

> **Tip:** Construct `WormholeComputeKernelConfig` objects once during model initialization and store them as model attributes. Creating them inside the forward pass adds Python object allocation overhead on every call.

## Config Selection Summary

| Config | `math_fidelity` | `fp32_dest_acc_en` | Typical use |
|---|---|---|---|
| LoFi | `MathFidelity.LoFi` | `False` | Maximum throughput; pairs well with bfloat4_b weights |
| HiFi2 | `MathFidelity.HiFi2` | `True` | Balanced accuracy and speed; pairs well with bfloat8_b weights |
| HiFi4 | `MathFidelity.HiFi4` | `True` | Full precision reference; attention SDPA |

Which projection types (gate, up, down) should use LoFi vs HiFi2 is covered in Chapter 5, after accuracy characterization of each projection is established.

---

**Next:** [`dtype_in_linear_and_matmul.md`](./dtype_in_linear_and_matmul.md)
