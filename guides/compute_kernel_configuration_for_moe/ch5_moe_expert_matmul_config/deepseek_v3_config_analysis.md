# DeepSeek-V3 Config Analysis

## Overview

DeepSeek-V3 is the primary reference implementation for `WormholeComputeKernelConfig` assignment in MoE expert FFN layers. Its `models/demos/deepseek_v3/tt/experts.py` defines two configs — `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` — and assigns them to specific projections based on their role in the SwiGLU block. This file walks through those definitions, explains each field choice, and quantifies the `packer_l1_acc` savings using DeepSeek-V3's concrete dimensions.

---

## Config Definitions

> **Note:** The canonical constructor reference is in Chapter 1, `wormhole_compute_kernel_config_api.md`; the block is reproduced here for inline analysis.

Both configs are defined at module level in `models/demos/deepseek_v3/tt/experts.py`:

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

COMPUTE_KERNEL_CONFIG_LOFI = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI2 = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

---

## Projection-Level Assignment

| Weight tensor | Projection role | Config assigned |
|---|---|---|
| `w1_experts` | gate_proj | `COMPUTE_KERNEL_CONFIG_LOFI` |
| `w3_experts` | up_proj | `COMPUTE_KERNEL_CONFIG_LOFI` |
| `w2_experts` | down_proj | `COMPUTE_KERNEL_CONFIG_HIFI2` |

The SwiGLU computation order is: `gate = silu(w1 @ x)`, `up = w3 @ x`, `hidden = gate * up`, `out = w2 @ hidden`. The critical paths are:

- **gate/up → LOFI**: Both projections feed into the SiLU nonlinearity and an element-wise multiply. The SiLU function (and the multiplication) smooths out small rounding errors introduced by LoFi's single accumulation pass. Numerical drift at this stage does not compound directly into the model's residual stream.
- **down → HiFi2**: The down projection output is added directly to the residual stream and accumulated across all MoE layers. Rounding errors here compound with depth. HiFi2's two-pass accumulation provides sufficient fidelity to keep per-layer PCC above 0.999.

---

## Field-by-Field Rationale

### `math_fidelity`

LoFi executes one accumulation pass per output tile; HiFi2 executes two. Each additional pass doubles the FPU time for that matmul. Using LoFi for gate/up gives a ~2× FPU throughput improvement over HiFi2 at no measurable accuracy cost for those projections. The down projection uses HiFi2 because it directly contributes to the residual.

### `math_approx_mode=False` (LOFI) vs. `True` (HiFi2)

`math_approx_mode` only affects SFPU ops (`exp`, `silu`, `sigmoid`, `sqrt`, `reciprocal`). Expert matmuls are FPU-path ops; `math_approx_mode` has no effect on them. The `False` setting on LOFI is a conservative default. The `True` setting on HiFi2 reflects that any SFPU ops using this config (e.g., softmax in attention, which shares the same config object) are acceptable with approximation.

### `fp32_dest_acc_en=False`

Leaving fp32 destination accumulation disabled keeps the output tensor in bfloat16, halving the L1 accumulation buffer size compared to fp32. For decode-mode expert matmuls where `packer_l1_acc=True` is enabled, the accumulation buffer must fit in L1; disabling fp32 dest acc reduces that pressure. The bfloat16 precision of the accumulator is sufficient given the input dtypes (bfloat8_b weights, bfloat16 activations).

### `packer_l1_acc=True`

This is the highest-leverage field for decode-mode matmuls. See the quantification section below.

---

## DeepSeek-V3 Dimensions

| Parameter | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` per expert | 2048 |
| K-dimension for down projection | 2048 (expert hidden dim) |
| K tiles (`K_t = d_ff / 32`) for down | 64 |
| Decode batch size (`b`) | 1 |

---

## `packer_l1_acc` Savings Quantification

For K_t=64 (the DeepSeek-V3 down projection), enabling `packer_l1_acc=True` eliminates 63 of 64 DRAM reads per output tile — a **98.4% bandwidth reduction**. For the full derivation, see Chapter 3, `throughput_impact.md`.

---

## Summary

DeepSeek-V3 applies two configs, not one. The assignment is structural: gate/up → LoFi (tolerant of rounding, feeds nonlinearity), down → HiFi2 (accumulates into residual, requires higher fidelity). Both configs share `packer_l1_acc=True`, which eliminates 98.4% of accumulation-related DRAM reads for the K_t=64 down projection at decode. This pattern is the reference for applying configs to other MoE models including Qwen.

---

**Next:** [`qwen_moe_current_state.md`](./qwen_moe_current_state.md)
