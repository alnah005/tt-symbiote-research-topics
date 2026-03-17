# Bandwidth vs Accuracy Tradeoff

## Per-Projection Analysis

Each projection type has a different sensitivity to quantization error. This section
documents expected PCC (Pearson Correlation Coefficient) against output quality, paired
with DRAM bandwidth consumption relative to bfloat16.

### Gate Projection

Gate output feeds into the SwiGLU gating branch. Errors here modulate the magnitude of
up-projection outputs but do not bypass the nonlinearity.

| Dtype | MathFidelity | PCC (typical) | BW vs BF16 |
|-------|-------------|---------------|------------|
| bfloat16 | HiFi2 | 1.000 | 1× |
| bfloat8_b | HiFi2 | 0.999 | 0.5× |
| bfloat8_b | LoFi | 0.997 | 0.5× |
| bfloat4_b | HiFi2 | 0.975 | 0.25× |
| bfloat4_b | LoFi | 0.971 | 0.25× |

Gate is tolerant of bfloat4_b + LoFi due to SwiGLU noise filtering; see Chapter 3, `projection_sensitivity.md` for the full mechanistic analysis.

### Up Projection

Up output is element-wise multiplied with the gated branch. Similar noise tolerance.

| Dtype | MathFidelity | PCC (typical) | BW vs BF16 |
|-------|-------------|---------------|------------|
| bfloat16 | HiFi2 | 1.000 | 1× |
| bfloat8_b | HiFi2 | 0.999 | 0.5× |
| bfloat8_b | LoFi | 0.998 | 0.5× |
| bfloat4_b | HiFi2 | 0.976 | 0.25× |
| bfloat4_b | LoFi | 0.972 | 0.25× |

Up projection is co-optimal with gate: bfloat4_b + LoFi sits on the Pareto frontier.

### Down Projection

Down projection maps `[batch, d_ff]` back to `[batch, d_model]` and adds to the
residual stream. Errors propagate directly to subsequent attention layers.

| Dtype | MathFidelity | PCC (typical) | BW vs BF16 |
|-------|-------------|---------------|------------|
| bfloat16 | HiFi2 | 1.000 | 1× |
| bfloat8_b | HiFi2 | 0.977 | 0.5× |
| bfloat8_b | LoFi | 0.963 | 0.5× |
| bfloat4_b | HiFi2 | 0.943 | 0.25× |
| bfloat4_b | LoFi | 0.941 | 0.25× |

bfloat4_b brings down projection PCC to ~0.941–0.943, at or barely above the 0.94 acceptable floor. For most production use cases this results in measurable perplexity degradation, placing it off the Pareto frontier.

### Dense MLP Layers (Non-Expert FFNs)

Non-expert dense FFNs in Qwen MoE lack the SwiGLU noise-filtering path common to MoE
experts in some configurations. Conservative quantization is appropriate.

| Dtype | MathFidelity | PCC (typical) | BW vs BF16 |
|-------|-------------|---------------|------------|
| bfloat16 | HiFi2 | 1.000 | 1× |
| bfloat8_b | HiFi2 | 0.998 | 0.5× |
| bfloat8_b | LoFi | 0.991 | 0.5× |
| bfloat4_b | HiFi2 | 0.942 | 0.25× |

Recommended: bfloat8_b + HiFi2 as the default for dense MLP layers.

## Efficiency Frontier

The Pareto frontier contains configurations where no alternative achieves both higher
PCC and lower bandwidth simultaneously:

```
PCC
1.00 |  BF16/HiFi2 (1.0×BW)
     |
0.999|  bfloat8_b/HiFi2 (0.5×BW)  ←── Pareto
     |
0.997|  bfloat8_b/LoFi  (0.5×BW)
     |
0.975|  bfloat4_b/HiFi2 (0.25×BW) ←── Pareto (gate/up only)
     |
0.971|  bfloat4_b/LoFi  (0.25×BW) ←── Pareto (gate/up only)
     |
0.941|  bfloat4_b/LoFi (down)      ←── NOT on frontier
     |
0.94 |............................ PCC floor for acceptable perplexity
     +---------------------------------------------------> BW reduction
         1×        0.5×        0.25×
```

## Pareto-Optimal Configurations

For the recommended dtype and MathFidelity per projection type, see Chapter 3, `projection_sensitivity.md`.

## Why Full bfloat4_b Is Not on the Pareto Frontier

Applying bfloat4_b to all three projections (gate + up + down) yields:
- Bandwidth: 0.25× of BF16 for all three projections (maximum reduction)
- Down projection PCC: ~0.941 (at or just above the 0.94 threshold)

A model with down projection at bfloat4_b/LoFi is dominated by:
- bfloat8_b/HiFi2 for down + bfloat4_b/LoFi for gate+up: achieves PCC>0.977 on down
  while retaining 0.25×BW for the higher-volume gate/up projections.

The marginal bandwidth saving from also quantizing down to 4-bit (from 0.5×BW to
0.25×BW on a single projection) does not justify the PCC drop for most deployments.
Gate and up projections account for 2/3 of expert weight volume, so bfloat4_b on those
two alone delivers most of the bandwidth benefit.

## Summary

- Gate and up projections: bfloat4_b + LoFi is Pareto-optimal (PCC ~0.97, 0.25×BW).
- Down projection: bfloat8_b + HiFi2 is Pareto-optimal (PCC ~0.977, 0.5×BW).
- Dense MLP: bfloat8_b + HiFi2 is the conservative default (PCC ~0.998, 0.5×BW).
- Full bfloat4_b across all projections is not on the Pareto frontier: down PCC reaches
  only ~0.941–0.943, at or just above the 0.94 floor, causing unacceptable perplexity without further compensation.
