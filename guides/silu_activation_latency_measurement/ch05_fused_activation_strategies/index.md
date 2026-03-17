# Chapter 5: Fused Activation Strategies in TTNN

This chapter covers the mechanisms available in TTNN to fuse SiLU with the preceding matmul kernel, explains when fusion is architecturally valid, and documents the SwiGLU-specific pattern that applies to production MoE models such as Llama 3, Mixtral, and Qwen-MoE.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Use the `activation` parameter on `ttnn.matmul` to fuse a SiLU post-op into the matmul kernel, eliminating a separate SFPU kernel dispatch.
2. Identify when to set `fused_activation` at the program config level versus the top-level API parameter, and why sharded tensors require the program config path.
3. Describe what kernel-level fusion means for L1 traffic: no separate op launch, no extra L1 round-trip for the activation tensor.
4. Apply Pattern A (fused SiLU into gate_proj, separate `ttnn.mul`) as the standard SwiGLU fusion pattern.
5. Select the correct `activation_dtype` for a given inference precision target and validate accuracy against a BF16 baseline.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Chapter 2 | SFPU execution model: why SiLU is a sequential SFPU pass over FPU output tiles; the L1-to-DRAM round-trip that fusion eliminates |
| Chapter 4 | Latency ratio results: the `num_tokens < 16` regime where SiLU cost justifies fusion; the decode vs. prefill threshold |

---

## Overview of Fusion Mechanisms in TTNN

TTNN exposes two levels at which a post-matmul activation can be folded into the matmul kernel:

**Top-level parameter.** `ttnn.matmul` and `ttnn.linear` accept an `activation` keyword argument that takes a string name (`"silu"`, `"relu"`, `"gelu"`). When this argument is provided, TTNN compiles the activation SFPU pass into the same kernel binary as the FPU matmul, so the op appears as a single `MatmulMultiCoreReuse*` entry in the Tracy profiler CSV with no separate `silu` kernel entry.

**Program config parameter.** `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreReuseMultiCast1DProgramConfig` each expose a `fused_activation` field. This path must be used when the input or output tensor is sharded, because the program config also controls the shard layout and core grid assignment. Passing `activation="silu"` at the top level while also providing a sharded program config that omits `fused_activation` will not fuse — the activation parameter at the top level is ignored when a program config is present.

Both paths achieve the same kernel-level effect: the SFPU activation pass is folded into the same tile-level dispatch loop as the FPU matrix multiply-accumulate, and the activation output is written directly to the destination L1 region without an intermediate tensor allocation.

---

## Summary of Available Fusion Mechanisms

| Mechanism | API Location | When to Use |
|---|---|---|
| `activation="silu"` | `ttnn.matmul(... activation="silu")` | Row-major or interleaved DRAM output; no custom program config |
| `fused_activation` field | `MatmulMultiCoreReuseMultiCastProgramConfig(fused_activation=...)` | Sharded output; custom core grid; any case where a program config is already being set |
| `activation_dtype` field | `MatmulMultiCoreReuseMultiCastProgramConfig(activation_dtype=...)` | Controls precision of the fused activation output; use `ttnn.bfloat8_b` to halve L1 footprint |

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` (this file) | Chapter overview, learning objectives, prerequisites, fusion mechanism summary, file map |
| [`ttnn_fused_activation_api.md`](./ttnn_fused_activation_api.md) | `ttnn.matmul` `activation` parameter; program config `fused_activation` field; kernel-level fusion semantics; `activation_dtype`; code examples |
| [`swiglu_fusion_pattern.md`](./swiglu_fusion_pattern.md) | SwiGLU fusion challenge; Pattern A (recommended), Pattern B (baseline), Pattern C (custom kernel); memory config requirements |
| [`activation_dtype_and_precision.md`](./activation_dtype_and_precision.md) | BFP8_B vs. BF16 accuracy tradeoff; L1 footprint reduction; when to use each dtype; accuracy validation approach |

---

## Next Steps

Begin with [`ttnn_fused_activation_api.md`](ttnn_fused_activation_api.md) to understand the API surface and kernel-level semantics before examining the SwiGLU-specific fusion pattern. After completing this chapter, Chapter 6 (`configuration_recommendations.md`) synthesizes the fusion recommendations into a decision table for production MoE configurations on Wormhole B0 and T3K.
