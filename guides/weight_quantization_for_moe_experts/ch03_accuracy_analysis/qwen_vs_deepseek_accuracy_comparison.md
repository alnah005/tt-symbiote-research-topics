# Qwen vs. DeepSeek Accuracy Comparison Under Quantization

## Model Family Overview

| Property | DeepSeek-V3 | Qwen3.5-35B (MoE) |
|---|---|---|
| Total parameters | 671B | ~35B |
| Active parameters | ~37B | ~14B |
| Expert count | 256 experts, top-8 routing | 64 experts, top-4 routing |
| Hidden dim | 7168 | 4096 |
| Intermediate dim (per expert) | 2048 | varies |
| Training precision | FP8 mixed-precision QAT | bfloat16 |
| TTNN down projection dtype | bfloat8_b + HiFi2 | bfloat16 (baseline) |
| TTNN gate/up projection dtype | bfloat4_b + LoFi | bfloat16 (baseline) |

## DeepSeek-V3 Quantization Strategy

DeepSeek-V3 was trained with FP8 mixed-precision quantization-aware training (QAT). Its weights
were optimized during training to be stored in low-precision formats, so the weight distributions
are better suited for block floating-point. Specifically:

- Outlier magnitudes in expert weights are suppressed by the FP8 QAT training objective.
- Gate and up projections (w1, w3): **bfloat4_b** dtype + **LoFi** compute.
- Down projection (w2): **bfloat8_b** dtype + **HiFi2** compute.

This mixed-precision strategy was validated at approximately PCC 0.97 for the full MoE layer
output relative to a bfloat16 CPU reference.

### Why DeepSeek-V3 Down Projection Is Particularly Sensitive

DeepSeek-V3's expert FFN down projection has K_t = 64 (d_ff / 32 = 2048 / 32 = 64 tiles — the K-loop depth for the down projection matmul). The down
projection maps from intermediate dimension to hidden dimension of 7168 with no downstream
nonlinearity. Because of the large hidden dimension and the absence of any SiLU absorption,
quantization error in w2 is injected into the residual stream at full scale and then interacts
with the large attention blocks. This makes bfloat4_b for w2 unacceptable even for a
QAT-trained model.

## Qwen3.5-35B Architecture Differences

- Trained in bfloat16; no quantization-aware training stage.
- Expert weights may have larger magnitude outliers than DeepSeek-V3 because the training
  objective did not regularize for low-precision storage.
- Smaller hidden dimension (4096 vs. 7168) reduces the magnitude of down-projection errors
  relative to the residual stream, but does not eliminate the sensitivity.

## Empirical Accuracy Under Quantization

### bfloat8_b (All Projections)

Both model families tolerate bfloat8_b for all projections with minimal accuracy loss:

| Model | Projection | Observed PCC | ΔPPL (WikiText-2) |
|---|---|---|---|
| DeepSeek-V3 | gate/up (w1/w3) | ~0.98–0.99 | < 0.5 |
| DeepSeek-V3 | down (w2) | ~0.975–0.985 | < 1.0 |
| Qwen3.5-35B | gate/up (w1/w3) | ~0.97–0.98 | < 1.0 |
| Qwen3.5-35B | down (w2) | ~0.97–0.98 | < 1.0 |

### bfloat4_b Gate/Up + bfloat8_b Down (Mixed Precision)

| Model | Gate/Up PCC | Down PCC | ΔPPL | Verdict |
|---|---|---|---|---|
| DeepSeek-V3 | ~0.97 | ~0.975 | ~0.5–1.0 | Validated; production-safe |
| Qwen3.5-35B | ~0.95–0.97 | ~0.97 | ~1.0–2.0 | Validate per checkpoint |

Qwen3.5-35B is more sensitive to bfloat4_b on gate/up projections because its weights were
not QAT-trained. Higher outlier frequency in expert weight blocks raises the block exponent,
reducing relative precision for small-magnitude weights.

## Recommended Evaluation Procedure

Run the following progression and compare each against the bfloat16 baseline:

1. **All bfloat8_b** (gate/up/down): establish safe lower bound.
2. **bfloat4_b gate/up + bfloat8_b down**: primary target for memory reduction.
3. **All bfloat4_b**: only if step 2 passes; expect significant accuracy risk for Qwen.

```python
# Pseudocode for staged evaluation
configs = [
    ("all_bf8b",    {"gate": bf8b, "up": bf8b, "down": bf8b}),
    ("mixed_bf4_8", {"gate": bf4b, "up": bf4b, "down": bf8b}),
    ("all_bf4b",    {"gate": bf4b, "up": bf4b, "down": bf4b}),
]
for name, cfg in configs:
    ppl = eval_perplexity(model, cfg, dataset="wikitext-2")
    print(f"{name}: PPL = {ppl:.2f}, ΔPPL = {ppl - baseline_ppl:.2f}")
```

## Key Takeaway

DeepSeek-V3's QAT training makes it inherently more robust to low-precision inference. Qwen
models trained in bfloat16 require per-projection validation before deploying bfloat4_b, and
the down projection should remain at bfloat8_b in all cases until full perplexity evaluation
confirms the delta is within the 1.5 PPL budget.
