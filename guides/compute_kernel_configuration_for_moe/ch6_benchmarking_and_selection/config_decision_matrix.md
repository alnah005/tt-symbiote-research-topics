# Config Decision Matrix

## Overview

This file codifies the decision rules for selecting a `WormholeComputeKernelConfig` for each MoE projection type and deployment regime. The rules are derived from the parameter analysis in Chapters 1–4 and validated against the DeepSeek-V3 production configs in Chapter 5. Use this matrix as the starting point before benchmarking; benchmark to confirm or override.

---

## Canonical Production Configs (Reference)

Two configs cover the majority of MoE expert matmul use cases on Wormhole B0. Their parameter values are fixed — do not alter individual fields without a benchmarked justification.

| Config constant | `math_fidelity` | `math_approx_mode` | `fp32_dest_acc_en` | `packer_l1_acc` |
|---|---|---|---|---|
| `COMPUTE_KERNEL_CONFIG_LOFI`  | `LoFi`  | `False` | `False` | `True` |
| `COMPUTE_KERNEL_CONFIG_HIFI2` | `HiFi2` | `True`  | `False` | `True` |

Both configs set `fp32_dest_acc_en=False`. This is the established canonical value: the bfloat16 accumulation in the destination register is sufficient given the other fidelity controls in each config. Enabling `fp32_dest_acc_en=True` is only warranted under the strict PCC regime described later in this file.

For the canonical LOFI and HIFI2 constructor definitions, see Chapter 1, `wormhole_compute_kernel_config_api.md` and Chapter 5, `deepseek_v3_config_analysis.md`.

---

## Decision Rules by Projection Type

### Gate Projection (w1)

**Rule: always use `COMPUTE_KERNEL_CONFIG_LOFI`.**

| Field | Value | Reason |
|---|---|---|
| `math_fidelity` | `LoFi` | Gate output feeds SiLU nonlinearity; rounding errors in logits are absorbed by the soft saturation of SiLU and do not accumulate downstream |
| `math_approx_mode` | `False` | No transcendental ops in the gate matmul itself; flag is irrelevant, set conservatively to False |
| `fp32_dest_acc_en` | `False` | BF16 destination accumulation is sufficient; adding fp32 dest costs L1 and FPU register pressure without measurable PCC benefit for gate logits |
| `packer_l1_acc` | `True` | Eliminates DRAM round-trips during K-loop accumulation; highest-ROI field for bandwidth-bound decode |

This rule applies at any sequence length and any batch size. The gate projection's tolerance for rounding is structural — it does not depend on M, K_t, or d_model.

### Up Projection (w3, SwiGLU/SiLU gating)

**Rule: always use `COMPUTE_KERNEL_CONFIG_LOFI`** — identical reasoning to gate projection.

The up projection output is multiplied element-wise with `silu(gate_output)`. Rounding errors in the up projection output are scaled by the gating signal and do not propagate into an accumulation across K. The numerical tolerance is the same as gate.

> **Tip:** If your model uses a different activation structure where the up projection directly enters an accumulation without a nonlinear gate (e.g., a plain FFN without gating), reconsider LoFi. The SiLU/SwiGLU structure is what makes LoFi safe for gate and up projections.

### Down Projection (w2) — Decode Regime (M <= 32)

**Rule: use `COMPUTE_KERNEL_CONFIG_HIFI2`.**

| Field | Value | Reason |
|---|---|---|
| `math_fidelity` | `HiFi2` | Down projection accumulates `d_ff` tiles directly into the residual stream; rounding errors from K_t iterations compound and shift hidden state values enough to degrade end-to-end PCC below threshold at LoFi |
| `math_approx_mode` | `True` | The HiFi2 config is shared across multiple op types in the broader model; for pure matmul, this flag has no effect (Chapter 4) |
| `fp32_dest_acc_en` | `False` | Canonical value; `packer_l1_acc=True` with BF16 accumulation provides sufficient precision at HiFi2 |
| `packer_l1_acc` | `True` | Decode-mode down projection is heavily bandwidth-bound; L1 accumulation eliminates K_t-1 redundant DRAM reads per output tile |

The decode regime (M=1–32) is the primary performance-critical path. The down projection K dimension is d_ff (e.g., 2048 for DeepSeek and Qwen 235B-A22B), giving K_t = 64 tiles. As shown in Chapter 3's bandwidth reduction formula, this yields a 98.4% reduction in redundant DRAM reads when b=1 — the dominant optimization.

### Down Projection (w2) — Prefill Regime (seq >= 512)

**Rule: start with `COMPUTE_KERNEL_CONFIG_HIFI2`; benchmark `COMPUTE_KERNEL_CONFIG_LOFI` and accept it if PCC remains above your model's threshold.**

In prefill mode with large sequence lengths, the matmul is more compute-bound. Two competing effects interact:

1. The accuracy argument for HiFi2 over LoFi is the same as in decode — down projection feeds the residual stream and accumulated rounding matters.
2. The throughput gap between HiFi2 and LoFi narrows as the matmul becomes compute-bound rather than bandwidth-bound.

The practical recommendation is: run the sweep from `benchmarking_methodology.md` with your prefill batch and sequence length. If LoFi PCC >= 0.999 (token-level, vs. float32 reference) and the latency improvement over HiFi2 is >= 5%, use LoFi. Otherwise, stay with HiFi2.

> **Warning:** Do not assume that the prefill PCC result transfers to decode. Prefill with large M has different statistical properties — the K-loop accumulation is the same but the M dimension averages out some rounding variance. Measure both regimes independently.

### Any Projection — Strict PCC Requirement (> 0.9995)

**Rule: use HiFi4 with `fp32_dest_acc_en=True`, and benchmark to confirm.**

```python
COMPUTE_KERNEL_CONFIG_HIFI4_STRICT = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,   # fp32 dest required at this precision tier
    packer_l1_acc=True,      # still beneficial for bandwidth; verify L1 budget
)
```

> **Warning:** Before enabling `fp32_dest_acc_en=True` with `packer_l1_acc=True`, verify the L1 budget. The fp32 accumulation buffer is 2x the size of the BF16 buffer. For large `per_core_N`, this combination can exceed the 1.5 MB per-core L1 on Wormhole B0 and cause an allocation error at dispatch. Run a dry dispatch (no profiling) at your maximum batch size before committing this config to production. See `production_config_checklist.md` for the verification procedure.

This regime applies to models with strict numerical quality gates — for example, fine-tuned generation models where end-to-end BLEU or perplexity is sensitive to low-level rounding, or validation runs where the TTNN output must match a reference implementation within a tight tolerance.

---

## Summary Decision Table

| Projection | Regime | Recommended config | Notes |
|---|---|---|---|
| gate (w1) | any | `COMPUTE_KERNEL_CONFIG_LOFI` | Structural tolerance from SiLU nonlinearity |
| up (w3) | any | `COMPUTE_KERNEL_CONFIG_LOFI` | Same as gate; element-wise gating absorbs rounding |
| down (w2) | decode (M <= 32) | `COMPUTE_KERNEL_CONFIG_HIFI2` | K-loop accumulation into residual; bandwidth-bound |
| down (w2) | prefill (seq >= 512) | `COMPUTE_KERNEL_CONFIG_HIFI2` (default); benchmark LoFi | Compute-bound; LoFi acceptable if PCC >= 0.999 |
| any | PCC > 0.9995 required | HiFi4, `fp32_dest_acc_en=True`, `packer_l1_acc=True` | Verify L1 budget before deploying |

---

## How d_ff / d_model Ratio Affects the Decision

The d_ff / d_model ratio determines the depth of the K dimension in the down projection relative to the width of the gate and up projections. A higher ratio means more K-loop iterations accumulate into the residual stream, amplifying the effect of per-iteration rounding errors.

**Concrete examples:**

| Model | d_model | d_ff per expert | K_t (down) | K_t (gate/up) | d_ff / d_model |
|---|---|---|---|---|---|
| DeepSeek-V3 | 7168 | 2048 | 64 | 224 | 0.29 |
| Qwen 235B-A22B | 7168 | 2048 | 64 | 224 | 0.29 |
| Hypothetical large-expert MoE | 4096 | 8192 | 256 | 128 | 2.0 |

For the hypothetical model with d_ff/d_model = 2.0, the down projection K_t = 256 means 256 accumulation iterations. At LoFi fidelity, rounding error accumulates over 256 steps. At HiFi2, each step uses more mantissa bits and the per-step error is smaller. The larger the K_t in the down projection, the more important it becomes to use HiFi2 or higher.

**Rule of thumb:** If K_t (down) > 128, benchmark HiFi4 for the down projection even if HiFi2 meets your PCC threshold at K_t = 64. At K_t = 256, HiFi2 may produce PCC = 0.998 where HiFi4 produces 0.9997.

> **Tip:** The gate and up projection K dimension is d_model (not d_ff). For models with large d_model (e.g., 7168), K_t(gate/up) is also large (224 tiles). However, the gate/up rounding tolerance comes from the structural role of those projections — not from the K dimension being small. LoFi is appropriate regardless of d_model size for gate and up.

---

## How Top-K Routing Depth Interacts

In MoE routing, top-K selects K experts per token and sums their outputs:

```
layer_output = sum_{k=1}^{K} g_k * expert_k(x)
```

where `g_k` is the routing weight for expert k. Higher top-K means more expert outputs are summed into the layer output per token per layer.

**Interaction with rounding bias:**

If each expert's down projection introduces a small systematic rounding bias (say, a consistent sign-preferred rounding direction under LoFi), that bias is multiplied by the routing weight and then summed across K expert outputs. With top-1 routing, the bias from a single expert is the entire contribution. With top-8 routing, the biases from 8 experts are summed — and if the bias is correlated across experts (which it tends to be for weights drawn from the same distribution), the summed bias can be up to 8x larger.

**Practical implication for config selection:**

| top-K | Down projection fidelity recommendation |
|---|---|
| top-1 | `COMPUTE_KERNEL_CONFIG_HIFI2` is sufficient for most models |
| top-2 | `COMPUTE_KERNEL_CONFIG_HIFI2`; verify layer-level PCC |
| top-4 to top-8 | `COMPUTE_KERNEL_CONFIG_HIFI2`; measure layer-level PCC explicitly; consider HiFi4 if PCC margin is thin |
| top-8, strict PCC | Benchmark HiFi4 with `fp32_dest_acc_en=True` |

Qwen 235B-A22B uses top-8 routing with 128 experts (8 experts active per token per layer). Each token's layer output is the sum of 8 expert contributions. This makes layer-level PCC measurement (not just per-projection PCC) especially important during config validation. See `benchmarking_methodology.md` for the layer-level PCC recipe.

> **Warning:** Top-K routing amplifies systematic rounding bias, not random rounding noise. If you benchmark with random synthetic inputs and observe PCC > 0.9995, that does not guarantee the same PCC on real model weights with real activation distributions. Always validate with actual model weights and representative inputs from the target task.

---

## When to Deviate from the Two-Config Pattern

The two-config pattern (`COMPUTE_KERNEL_CONFIG_LOFI` for gate/up, `COMPUTE_KERNEL_CONFIG_HIFI2` for down) is the correct starting point for any SwiGLU/SiLU MoE model on Wormhole B0. Deviations are warranted in the following specific cases:

| Situation | Deviation | Reason |
|---|---|---|
| PCC > 0.9995 required (any projection) | HiFi4, `fp32_dest_acc_en=True` | BF16 accumulation insufficient at this tier |
| d_ff / d_model > 1.5, down projection | Benchmark HiFi4 | More K-loop iterations amplify rounding |
| top-8 routing, end-to-end PCC degradation observed | HiFi4 for down | Summed bias across 8 experts exceeds threshold |
| Prefill seq >= 512, down projection, latency critical | Benchmark LoFi | Compute-bound; LoFi may be acceptable |
| Shared experts (dense FFN, not gated) | Benchmark separately | Different numerical role; see `production_config_checklist.md` |

Do not deviate from `packer_l1_acc=True` without an L1 allocation error as justification. The bandwidth benefit is unconditionally positive for bandwidth-bound matmuls and negligible (not negative) for compute-bound matmuls.

---

## Next Steps

Proceed to [production_config_checklist.md](production_config_checklist.md) for the pre-deployment verification steps before merging a kernel config change into a production model.
