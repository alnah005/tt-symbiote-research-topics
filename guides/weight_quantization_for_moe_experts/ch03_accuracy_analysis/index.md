# Chapter 3: Accuracy Analysis for MoE Expert Quantization

## Overview

This chapter characterizes the accuracy loss incurred when quantizing MoE expert weights at
each supported precision level (bfloat16, bfloat8_b, bfloat4_b). It identifies which metrics
are most informative, which projections are most sensitive to quantization noise, and what
practical tolerance limits apply to production deployments.

## Learning Objectives

After completing this chapter you will be able to:

- Compute and interpret PCC (Pearson Correlation Coefficient) as the primary correctness
  signal for TTNN quantized layers.
- Understand why the down projection (w2) is the most accuracy-sensitive weight in a SwiGLU
  FFN and why gate/up projections tolerate lower fidelity.
- Distinguish the accuracy profiles of bfloat8_b and bfloat4_b and apply the correct
  mixed-precision strategy (down=HiFi2, gate/up=LoFi).
- Explain model-specific sensitivity differences between DeepSeek-V3 and Qwen MoE families.

## Prerequisites

- Chapter 1: Block Floating-Point Fundamentals (bfloat8_b and bfloat4_b encoding)
- Chapter 2: MoE Expert Weight Layout and TTNN Op Mapping

## File Map

| File | Contents |
|---|---|
| [`accuracy_metrics_for_moe.md`](./accuracy_metrics_for_moe.md) | PCC definition, thresholds, perplexity delta, code |
| [`projection_sensitivity.md`](./projection_sensitivity.md) | Sensitivity ordering across gate/up/down projections |
| [`bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`](./bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md) | Per-dtype PCC ranges, comparison table, code |
| [`qwen_vs_deepseek_accuracy_comparison.md`](./qwen_vs_deepseek_accuracy_comparison.md) | Empirical comparison across model families |

## Summary Table: Quantization Level vs. Expected Accuracy

For full per-dtype, per-projection PCC ranges, perplexity deltas, and recommended use guidance,
see `accuracy_metrics_for_moe.md` § PCC Thresholds in Practice. In brief: bfloat8_b is the safe
default for all projections; bfloat4_b gate/up requires per-model validation; bfloat4_b down
projection is high-risk and should be avoided unless full perplexity evaluation confirms ΔPPL
within budget.

## Setup Checklist

Before running the accuracy experiments in this chapter:

- [ ] PyTorch >= 2.1 available with `torch.corrcoef` support.
- [ ] `tt-metal` environment activated; `import ttnn` succeeds.
- [ ] Wormhole B0 device opened via `ttnn.open_device(device_id=0)`.
- [ ] Reference bfloat16 tensors pre-computed on CPU for comparison.
- [ ] WikiText-2 or C4 tokenized split available for perplexity evaluation.
- [ ] Expert weight checkpoint accessible (Qwen or DeepSeek shards).

## Key Constants

```python
PCC_BFLOAT16_BASELINE   = 0.999   # minimum acceptable for bf16 reference ops
PCC_BFLOAT8B_MIN        = 0.970   # minimum acceptable for bfloat8_b experts
PCC_BFLOAT4B_GATE_UP_MIN = 0.960  # minimum acceptable for bfloat4_b gate/up
PCC_BFLOAT4B_DOWN_MIN   = 0.940   # risky; prefer bfloat8_b for down projection
```
