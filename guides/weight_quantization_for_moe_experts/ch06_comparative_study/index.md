# Chapter 6: Comparative Study — DeepSeek-V3 vs. Qwen Quantization Approach

## Overview

This chapter places the abstract principles of Chapters 1–5 in contact with two concrete
production systems: the DeepSeek-V3 TTNN implementation, which uses aggressive mixed-precision
quantization, and the Qwen 235B-A22B TTNN implementation, which currently uses bfloat16 for
all expert weights. The comparison is not a competition — it is an illustration of how a
model's training history, accuracy budget, and deployment regime jointly determine the
viable quantization strategy.

The central question answered here is: **can the DeepSeek-V3 quantization design be applied
to Qwen, and if so, how?** The answer is nuanced. The two models share the same architectural
dimensions (`d_model=7168`, `d_ff=2048`, `num_experts=128`, `top_k=8`) but differ in one
critical property: DeepSeek-V3 was trained with quantization-aware objectives that pre-adapt
weights to low-precision storage, whereas Qwen was trained in standard bfloat16. That
difference shapes every recommendation in this chapter.

## Learning Objectives

After completing this chapter, you will be able to:

1. Describe the DeepSeek-V3 quantization design — which projection uses which dtype and
   compute kernel config — and explain why the model achieves PCC ~0.97 against a bfloat16
   reference despite using bfloat4_b on gate and up projections.
2. Calculate the per-expert and system-level memory savings from mixed-precision quantization
   relative to the Qwen bfloat16 baseline, using the authoritative byte-per-tile figures.
3. Connect the "16ms gap" in Qwen MoE decode latency to bfloat16 DRAM bandwidth cost and
   quantify the expected improvement from switching to bfloat8_b.
4. Apply the three-criterion decision framework (accuracy budget, deployment regime,
   training history) to choose an initial quantization strategy for any bfloat16-trained
   MoE model.
5. State the recommended starting point for Qwen quantization and the fallback condition
   for reverting any projection to bfloat16.

## Prerequisites

- Chapter 1, `bfloat4_b_format.md` and `bfloat8_b_format.md`: block floating-point format
  definitions, tile memory footprints, and throughput multipliers.
- Chapter 2, `compute_kernel_config.md` and `weight_conversion.md`: `WormholeComputeKernelConfig`
  fields, LoFi and HiFi2 config construction, and `ttnn.as_tensor` dtype conversion.
- Chapter 3, `projection_sensitivity.md`: empirical sensitivity ordering and the DeepSeek-V3
  PCC validation evidence.
- Chapter 4, `decode_memory_bandwidth.md` and `bandwidth_vs_accuracy_tradeoff.md`: arithmetic
  intensity analysis confirming that decode is memory-bound and that bandwidth reduction is the
  primary speedup mechanism.
- Chapter 5, `index.md` through `qwen_adaptation_guide.md`: the per-projection mixed-precision
  configuration table and weight conversion procedure.

## Side-by-Side Summary

The table below compresses the core comparison. Detailed analysis for each column appears in
the files listed in the next section.

| Property | DeepSeek-V3 (production) | Qwen 235B-A22B (current) | Recommended Qwen start |
|---|---|---|---|
| Gate (w1) dtype | `bfloat4_b` | `bfloat16` | `bfloat8_b` |
| Up (w3) dtype | `bfloat4_b` | `bfloat16` | `bfloat8_b` |
| Down (w2) dtype | `bfloat8_b` | `bfloat16` | `bfloat8_b` |
| Gate/Up kernel | LoFi | — | HiFi2 |
| Down kernel | HiFi2 | — | HiFi2 |
| `fp32_dest_acc_en` | False (all) | — | False (all) |
| `packer_l1_acc` | True (all) | — | True (all) |
| Memory / expert | ~28.0 MB | ~84.0 MB | ~42.0 MB |
| Memory reduction | 3× vs BF16 | baseline | ~2× vs BF16 |
| MoE layer PCC | ~0.97 | >0.999 | ~0.99 (expected) |
| Training history | FP8/BF4 QAT | bfloat16 standard | bfloat16 standard |
| Decode BW relief | 3× reduction | none | 2× reduction |

> **Tip:** The recommended Qwen starting point is `bfloat8_b` for all three projections
> with HiFi2, not the full DeepSeek-V3 bfloat4_b gate/up strategy. This is a deliberate
> conservative first step: it achieves approximately 2× memory reduction and 2× decode
> bandwidth relief with low accuracy risk on a model not trained with quantization awareness.
> Migrate to bfloat4_b gate/up only after validating perplexity on a representative benchmark.

> **Warning:** The DeepSeek-V3 mixed-precision design (bfloat4_b gate/up + bfloat8_b down)
> achieves PCC ~0.97 because the model was trained to tolerate that precision level. Applying
> the same dtype assignments to Qwen without QAT may yield lower PCC. Always validate
> per-layer PCC before deploying a quantized Qwen checkpoint.

## Compute Kernel Configs (Authoritative)

For the authoritative LOFI and HIFI2 constructor definitions with all field values, see `deepseek_v3_quantization_design.md` in this chapter.

## Files in This Chapter

| File | Contents |
|---|---|
| `deepseek_v3_quantization_design.md` | DeepSeek-V3 dtype and kernel config per projection; training context (FP8/bfloat4 QAT); PCC outcome ~0.97; memory and throughput gains |
| `qwen_bfloat16_baseline.md` | Qwen 235B-A22B bfloat16 baseline; memory cost analysis; throughput cost; the 16ms gap motivation |
| `recommendations_and_decision_framework.md` | Three decision criteria; recommended starting point; fallback conditions |

## Next Steps

Begin with `deepseek_v3_quantization_design.md` to understand the production quantization
design whose PCC outcomes and memory savings are used as the quantitative upper bound on
what is achievable for Qwen.
