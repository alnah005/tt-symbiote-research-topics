# Chapter 5: MoE Expert Matmul Configuration

## Overview

Chapters 1–4 established the four fields of `WormholeComputeKernelConfig` in isolation: the LoFi/HiFi2/HiFi4 fidelity hierarchy (Chapter 1 and 2), the throughput effect of `packer_l1_acc` (Chapter 3), and the limited scope of `math_approx_mode` for SFPU-only ops (Chapter 4). This chapter synthesizes those concepts into a concrete, two-config pattern used in production MoE models on Wormhole hardware.

The central observation is that a SwiGLU MoE FFN block contains two structurally distinct matmul classes — gate/up projections that feed into an activation function, and a down projection that accumulates directly into the residual stream — and each class has different numerical requirements. A single kernel config applied uniformly leaves performance or accuracy on the table.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. State the recommended `WormholeComputeKernelConfig` for gate/up projections and justify each field value.
2. State the recommended `WormholeComputeKernelConfig` for the down projection and explain why it differs from gate/up.
3. Explain why `packer_l1_acc=True` is the highest-leverage single-field change for decode-mode expert matmuls.
4. Identify the current gap in the Qwen MoE implementation (no explicit `compute_kernel_config`) and describe its performance cost.
5. Apply the two-config pattern to a new MoE model by mapping its projection types to the appropriate config.

---

## Recommended Config Reference

The table below is the primary reference for this chapter. All other files in this chapter derive from or expand on these entries.

| Projection | `math_fidelity` | `math_approx_mode` | `fp32_dest_acc_en` | `packer_l1_acc` | Rationale |
|---|---|---|---|---|---|
| gate (w1) | `LoFi` | `False` | `False` | `True` | Output feeds SiLU; rounding absorbed by nonlinearity; L1 accumulation eliminates DRAM round-trips |
| up (w3) | `LoFi` | `False` | `False` | `True` | Output multiplied element-wise with gated activation; same tolerance as gate |
| down (w2) | `HiFi2` | `True` | `False` | `True` | Accumulates into residual stream; requires higher fidelity to avoid drift across layers; `packer_l1_acc` still applies |

In code:

See Chapter 1, `wormhole_compute_kernel_config_api.md` for the canonical `WormholeComputeKernelConfig` constructor reference.

---

## Prerequisites

- **Chapter 1** — `WormholeComputeKernelConfig` field definitions; the LoFi/HiFi2/HiFi4 hierarchy; what a math fidelity level controls at the FPU level.
- **Chapter 2** — Math fidelity levels in detail: LoFi=1 accumulation pass; HiFi2=2 passes; HiFi4=4 passes; per-pass latency; when LoFi rounding is acceptable.
- **Chapter 3** — `packer_l1_acc`: packer pipeline, partial-sum accumulation in L1 vs. DRAM, bandwidth reduction formula, L1 overflow constraints.
- **Chapter 4** — `math_approx_mode`: SFPU-only scope; zero effect on matmul FPU path; recommended setting for transcendental ops adjacent to expert FFN.

---

## Files in This Chapter

| File | Contents |
|---|---|
| [deepseek_v3_config_analysis.md](deepseek_v3_config_analysis.md) | DeepSeek-V3 config definitions, projection-level assignment, numerical rationale, `packer_l1_acc` bandwidth savings quantified for DeepSeek dimensions |
| [qwen_moe_current_state.md](qwen_moe_current_state.md) | Current Qwen MoE implementation gap — no explicit `compute_kernel_config` — and the performance cost of falling back to device defaults |
| [applying_configs_to_qwen.md](applying_configs_to_qwen.md) | Step-by-step application: where to add configs in the Qwen expert forward pass, validation via PCC, latency measurement, edge cases |

Read in order. `qwen_moe_current_state.md` assumes familiarity with the DeepSeek reference from `deepseek_v3_config_analysis.md`. `applying_configs_to_qwen.md` assumes both.

---

## Next Steps

Begin with [deepseek_v3_config_analysis.md](deepseek_v3_config_analysis.md) to see the two-config pattern in a verified production implementation, then read [qwen_moe_current_state.md](qwen_moe_current_state.md) to understand the current gap, and finish with [applying_configs_to_qwen.md](applying_configs_to_qwen.md) for the concrete application steps.
