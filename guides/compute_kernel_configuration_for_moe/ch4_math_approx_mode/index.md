# Chapter 4: math_approx_mode

## Overview

`math_approx_mode` is a boolean flag in `WormholeComputeKernelConfig` that controls whether the SFPU (Special Function Processing Unit) uses polynomial lookup approximations for transcendental functions instead of iterative refinement. It has no effect on matmul or dot-product operations, which run on the FPU path.

This chapter covers which TTNN ops are affected, the accuracy tradeoffs, and the recommended settings for MoE expert layer kernels.

## Prerequisites

- Chapter 1: `WormholeComputeKernelConfig` fields and the LoFi/HiFi2/HiFi4 fidelity hierarchy
- Familiarity with the DeepSeek-V3 / Qwen MoE FFN structure: gate (w1) → silu → element-wise multiply with up (w3) → down (w2)

## Learning Objectives

By the end of this chapter you should be able to:

1. Identify which TTNN ops route through the SFPU and are therefore subject to `math_approx_mode`.
2. Explain why `math_approx_mode` has zero effect on pure matmul throughput or accuracy.
3. Choose the correct `math_approx_mode` setting for each projection in a SwiGLU MoE FFN block.
4. Know the sequence-length threshold above which approximate softmax exp accumulation becomes a measurable accuracy risk.

## Affected vs. Unaffected Operations

Six SFPU ops are subject to `math_approx_mode`: `exp`, `reciprocal`, `sqrt`, `sigmoid`, `gelu`, and `silu`; each carries ~0.1–0.3% per-evaluation relative error when approximation is enabled. FPU ops (`matmul`, `linear`, `dot_product`) are unaffected regardless of this flag's value.

For the full table with approximate error ranges per op, see `sfpu_approx_operations.md` § Operations Routed Through the SFPU.

## Key Config Reference

```python
from ttnn import WormholeComputeKernelConfig, MathFidelity

# DeepSeek-V3 gate/up projection kernel config
COMPUTE_KERNEL_CONFIG_LOFI = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,
    math_approx_mode=False,   # no transcendental ops; flag is irrelevant, set conservatively
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# DeepSeek-V3 down projection / attention kernel config
COMPUTE_KERNEL_CONFIG_HIFI2 = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=True,    # acceptable for softmax/layer-norm in broader model
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

## Chapter Structure

- `sfpu_approx_operations.md` — SFPU architecture and which ops use approximation
- `approx_mode_accuracy_risks.md` — When approximation causes problems vs. when it is safe
- `approx_mode_for_moe.md` — Recommended settings for each MoE FFN projection
