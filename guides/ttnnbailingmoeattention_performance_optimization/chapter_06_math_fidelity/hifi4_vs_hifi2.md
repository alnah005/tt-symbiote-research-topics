# HiFi4 vs HiFi2: Math Fidelity on Wormhole

## What Math Fidelity Controls

The HiFi4/HiFi2/LoFi levels and the `fp32_dest_acc_en` / `packer_l1_acc` parameters are defined in `index.md`'s Parameter Reference section. In brief: `math_fidelity` controls mantissa bits per BFP8 product (4 / 2 / 1 for HiFi4 / HiFi2 / LoFi per Tenstorrent Wormhole hardware documentation); `fp32_dest_acc_en=True` uses a 32-bit FP accumulator (4 dst tiles available) while `False` uses a 16-bit BFP accumulator (8 dst tiles available, higher throughput).

The combined effect of all four parameters on the four candidate configurations is:

## How `fp32_dest_acc_en` Interacts with Fidelity

**Combined effects:**

| Config | Product precision | Accumulator format | dst tiles available | L1 write traffic |
|---|---|---|---|---|
| HiFi4 + fp32=True + packer_l1=True | 4-bit mantissa | FP32 | 4 | Reduced (L1 acc) |
| HiFi4 + fp32=False + packer_l1=False | 4-bit mantissa | BFP16 | 8 | Direct to output buffer |
| HiFi2 + fp32=True + packer_l1=True | 2-bit mantissa | FP32 | 4 | Reduced (L1 acc) |
| HiFi2 + fp32=False + packer_l1=False | 2-bit mantissa | BFP16 | 8 | Direct to output buffer |

`TTNNBailingMoEAttention` currently uses the first row. `TTNNQwen3FullAttention` uses the second row with HiFi4 fidelity. Neither has adopted HiFi2.

---

## HiFi2 Fidelity: Expected Error Characteristics

Switching from HiFi4 to HiFi2 introduces additional rounding error in each BFP8 product. The error propagates through the attention computation as follows:

1. **QK^T matmul**: each element of `QK^T` is the dot product of Q_i and K_j over `head_dim=128`. Each dot product sums 128 products. At HiFi2, each product's mantissa is rounded to 2 bits, introducing an error of up to ±4 ulps per product. Over 128 products, accumulated error is bounded (with high probability) by approximately ±`sqrt(128)` × ulp ≈ 11 ulps in the HiFi2 case vs ±2 ulps for HiFi4.

2. **Softmax**: the input to softmax is `QK^T / sqrt(D)`. The error in QK^T propagates additively through the softmax exponential. The post-softmax probabilities are more sensitive to errors in large QK^T values than small ones; large negative values (unlikely tokens) are clamped to near-zero regardless, while the top attention logits dominate the output.

3. **Attention-weighted V sum**: errors in the attention weights (post-softmax) multiply the V values. If the top-k attention weights are accurate, the output vector is close to the HiFi4 reference even if low-weight V contributions are slightly off.

In practice, for causal language models, the output quality degradation from HiFi2 in attention is typically small and task-dependent. Models with very large attention logit variance (e.g., models relying on sharp attention distributions) are more sensitive than models with diffuse distributions.

---

## HiFi2 for Bailing MoE: Feasibility Assessment

Three factors favor HiFi2 for Bailing MoE attention:

1. **QK norm is already applied**: `_apply_qk_norm` (defined at line 2454 of `attention.py`, called at line 2659) normalizes Q and K to unit scale before the attention score computation. This bounds the magnitude of QK^T entries, reducing the dynamic range that fidelity reduction must handle. Models with large unbounded logits are more vulnerable to fidelity reduction; QK norm substantially mitigates this.

2. **GQA with 4 KV heads**: the grouped query attention structure means each Q head attends to KV pairs shared with 4 Q heads (H/Hkv = 16/4 = 4, using the H=16 and Hkv=4 parameters established in Chapter 1). The effective attention distribution per KV group tends to be less peaked than in standard multi-head attention, providing some robustness to small logit errors.

3. **Dense projection follows attention**: the output of `paged_sdpa_decode` is projected through `dense` (a linear layer) before being passed up to the MoE layer. The projection averages across the head dimension, smoothing out per-head errors.

One factor that argues for caution: Bailing MoE is a precision-sensitive model class. The MoE routing decisions depend on the final hidden state representation, which accumulates errors across all operations in the layer — not just attention. Quantifying the end-to-end effect of HiFi2 in attention requires generation quality benchmarking, not just tensor-level comparison.

**Recommendation**: HiFi2 is a candidate optimization for the SDPA kernel. The presence of QK norm and the GQA structure make it more likely to be safe than in a norm-free attention architecture. The accuracy impact requires measurement before adoption; the methodology is described in `accuracy_throughput_tradeoff.md`.

---

## The `fp32_dest_acc_en=False` Change: Independent of Fidelity

Switching from `fp32_dest_acc_en=True` to `False` is independent of the math fidelity choice. With HiFi4 (the current fidelity), switching to `fp32_dest_acc_en=False` gives 8 dst tiles instead of 4 at the cost of using a BFP16 accumulator. For the attention computation at `head_dim=128`:

- The bfloat16 accumulator uses 1 sign + 8-bit exponent + 7-bit mantissa. Each partial sum of 128 BFP8 products into a bfloat16 accumulator loses precision compared to FP32, but the magnitude of attention scores after QK norm is bounded, reducing the risk of exponent overflow.
- The doubled dst capacity allows the SDPA kernel to process twice as many KV tiles per inner-loop iteration, reducing the overhead of loop control and data staging.

This change matches what Qwen3 has already validated. The comment in `qwen_attention.py` attributes it to a port of DeepSeek V3 settings. Since Bailing MoE and Qwen3 share the same T3K target hardware, the `fp32_dest_acc_en=False` change is a well-precedented optimization that is unlikely to cause correctness regressions at HiFi4 fidelity.

---

**Next:** [Accuracy vs Throughput Tradeoff Analysis](accuracy_throughput_tradeoff.md)
