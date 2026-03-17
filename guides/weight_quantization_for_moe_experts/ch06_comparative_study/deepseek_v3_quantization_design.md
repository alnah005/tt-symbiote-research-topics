# DeepSeek-V3 Quantization Design

## Model Dimensions

DeepSeek-V3 and Qwen 235B-A22B share the same MoE FFN shape:

| Parameter | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` | 2048 |
| `num_experts` | 128 |
| `top_k` | 8 |

This alignment is what makes the comparison tractable: any per-expert memory figure or
arithmetic intensity calculation derived for DeepSeek-V3 applies directly to Qwen under the
same dtype. Differences in quantization outcome are therefore attributable to training
history, not architecture.

## Per-Projection Dtype and Kernel Config

DeepSeek-V3's TTNN implementation applies a differentiated strategy across the three
projections in each SwiGLU expert FFN:

| Projection | Weight name | Dtype | MathFidelity | `fp32_dest_acc_en` | `packer_l1_acc` |
|---|---|---|---|---|---|
| Gate | w1 | `ttnn.bfloat4_b` | LoFi | False | True |
| Up | w3 | `ttnn.bfloat4_b` | LoFi | False | True |
| Down | w2 | `ttnn.bfloat8_b` | HiFi2 | False | True |
| Dense MLP (non-expert) | — | `ttnn.bfloat8_b` | HiFi2 | False | True |

The LoFi config used for gate and up projections:

```python
WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

The HiFi2 config used for down projections and dense MLP layers:

```python
WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

> **Warning:** Both configs use `fp32_dest_acc_en=False`. This is the authoritative value
> for MoE expert projections on Wormhole B0. Do not set `fp32_dest_acc_en=True` for HiFi2
> in the expert FFN path — the configuration documented here is what the production
> DeepSeek-V3 implementation uses.

The rationale for this assignment is covered in Chapter 5. Briefly: the SiLU activation on
the gate projection output, and the element-wise product between gate and up outputs, both
provide natural error compression before the result reaches the down projection. The down
projection's output enters the residual stream directly, where uncorrected quantization error
accumulates across layers — hence bfloat8_b + HiFi2 there.

## Training Context: FP8 and bfloat4-Aware QAT

DeepSeek-V3 was trained using FP8 mixed-precision training, with explicit quantization
simulation for weights at bfloat4_b precision during forward passes. This process is
commonly called quantization-aware training (QAT). Its effects on the final weight
distribution are concrete and measurable:

**Reduced outlier magnitudes.** Standard bfloat16 training allows occasional large weight
values to develop in attention projections and expert FFNs. These outliers compress the
effective dynamic range available to low-bit formats: the shared exponent in a bfloat4_b
tile is forced up to accommodate the outlier, degrading resolution for the remaining
elements. QAT regularises the weight distribution during training, suppressing outliers and
keeping per-tile dynamic range narrow.

**Lower per-layer quantization error.** Because QAT exposes the model to quantization noise
at each forward pass during training, the optimizer learns to route information through paths
that are robust to that noise. The resulting weights are not just smaller in magnitude — they
are arranged so that the quantization rounding error in one layer partially cancels with that
in adjacent layers.

**Implication for Qwen.** Qwen 235B-A22B was trained in standard bfloat16 with no
quantization simulation. Its weight distributions have not been shaped to tolerate bfloat4_b
rounding. This does not make Qwen un-quantizable, but it does mean that applying the
DeepSeek-V3 strategy (bfloat4_b gate/up) to Qwen may yield larger per-layer PCC drops than
the ~0.97 observed for DeepSeek-V3. Empirical validation is required before deploying the
aggressive tier.

## Performance Outcome

The DeepSeek-V3 TTNN implementation produces MoE forward pass outputs with PCC approximately
0.97 against a bfloat16 reference across representative token batches and layer depths. The
accepted PCC thresholds by projection are:

| Projection | Dtype | PCC threshold |
|---|---|---|
| Gate (w1) | bfloat4_b | ≥ 0.96 |
| Up (w3) | bfloat4_b | ≥ 0.96 |
| Down (w2) | bfloat8_b | ≥ 0.975 |
| Full MoE layer | mixed | ≥ 0.97 |

For comparison, a pure bfloat16 implementation is expected to produce PCC > 0.999 against a
CPU floating-point reference. The ~0.97 full-layer figure for DeepSeek-V3 reflects the
combined effect of bfloat4_b on gate and up. End-to-end perplexity on standard benchmarks
remains within an acceptable range for production deployment, confirming that the PCC drop
does not translate to visible quality regression at the generation level — an outcome
enabled by the model's QAT-shaped weight distributions.

> **Tip:** PCC of ~0.97 at the MoE layer level does not mean each token's output is 3%
> wrong. PCC measures linear correlation of the full output tensor. At 0.97, outputs are
> highly correlated; the perplexity impact is typically sub-1 PPL for QAT models.

## Memory and Throughput Gains

### Per-Expert Memory Footprint

Tile sizes on Wormhole B0:

| Dtype | Bytes per 32×32 tile |
|---|---|
| bfloat16 | 2048 bytes |
| bfloat8_b | 1024 bytes |
| bfloat4_b | 512 bytes |

Per-expert memory under each strategy:

**bfloat16 baseline** (all three projections):
```
3 × d_model × d_ff × 2 bytes
= 3 × 7168 × 2048 × 2
= 88,080,384 bytes
= 84.0 MB per expert
```

**DeepSeek-V3 mixed** (bfloat4_b gate + bfloat4_b up + bfloat8_b down):
```
gate:  7168 × 2048 × 0.5 = 7,340,032 bytes
up:    7168 × 2048 × 0.5 = 7,340,032 bytes
down:  7168 × 2048 × 1   = 14,680,064 bytes
total: 29,360,128 bytes = 28.0 MB per expert
```

This is a **3× reduction** in per-expert weight memory relative to bfloat16.

### System-Level Impact on T3K

A T3K system has 8 chips (Wormhole B0, n300 configuration). With 128 experts distributed
across 8 chips, each chip holds approximately 16 experts.

| Metric | bfloat16 | DeepSeek-V3 mixed | Reduction |
|---|---|---|---|
| Per-expert weight | 84.0 MB | 28.0 MB | 3× |
| Per-chip (16 experts) | 1,344 MB | 448 MB | 3× |
| Total system (128 experts) | 10,752 MB | 3,584 MB | 3× |

The freed DRAM capacity can accommodate larger KV caches, longer context windows, or higher
batch sizes — all of which are throughput multipliers during decode.

### Decode Throughput

During decode (batch ≤ 32, seq = 1), expert FFN matmuls are memory-bound on Wormhole B0
(see Chapter 4, `decode_memory_bandwidth.md` for the arithmetic intensity derivation).
The DRAM bandwidth on an n300 is approximately 576 GB/s. Decode latency per expert
projection scales directly with bytes read from DRAM:

| Dtype | Bytes read (gate proj, d_model=7168, d_ff=2048) | Latency (at 576 GB/s) |
|---|---|---|
| bfloat16 | 29,360,128 bytes | ~51.0 µs |
| bfloat8_b | 14,680,064 bytes | ~25.5 µs |
| bfloat4_b | 7,340,032 bytes | ~12.7 µs |

Gate and up together at bfloat4_b read approximately the same DRAM volume as a single
projection at bfloat8_b, which is a meaningful decode latency advantage.

## Key Insight: QAT as a Prerequisite for Aggressive Quantization

The lesson from DeepSeek-V3 is not "use bfloat4_b for gate and up." The lesson is:

> **Aggressive weight quantization (bfloat4_b) is reliable only when the model has been
> trained to tolerate it.** QAT is the mechanism that makes the trained weights compatible
> with the quantization grid. Without QAT, the same dtype assignment may produce PCC values
> below the accepted thresholds for some layers, and the accuracy impact is unpredictable
> without per-layer validation.

For practitioners working with bfloat16-trained models, the practical implication is a
two-stage approach: start with bfloat8_b (which bfloat16-trained models generally tolerate
without QAT), validate accuracy thoroughly, and then consider bfloat4_b for gate and up only
if the accuracy budget permits and the throughput improvement justifies the validation cost.

## Next Steps

Proceed to `qwen_bfloat16_baseline.md` to understand the current state of the Qwen
235B-A22B TTNN implementation: why it uses bfloat16, what that costs in memory and
decode throughput, and how the "16ms gap" connects to this choice.
