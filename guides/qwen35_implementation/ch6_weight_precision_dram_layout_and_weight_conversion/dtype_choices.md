# dtype Choices: Per-Category Precision and DRAM Layout

Qwen3.5 uses four distinct numeric formats across its weight categories. Each choice reflects a specific tradeoff between DRAM capacity, numerical accuracy, and compute throughput on Blackhole P100A.

---

## Format Reference

| Format | TTNN symbol | Bits/element | Notes |
|--------|-------------|-------------|-------|
| bfloat4 block | `ttnn.bfloat4_b` | 4 | Block floating point, 4-bit mantissa |
| bfloat8 block | `ttnn.bfloat8_b` | 8 | Block floating point, 8-bit mantissa |
| bfloat16 | `ttnn.bfloat16` | 16 | Standard BF16 |
| float32 | `ttnn.float32` | 32 | IEEE single precision |

---

## Category-by-Category Choices

### bfp4 — Routed Expert Weights (A3B Only)

The 256 routed experts per MoE layer each hold a fused gate+up projection and a down projection:

```
experts.gate_up_proj  shape: [256, 2*intermediate, hidden]   = [256, 1024, 2048]
experts.down_proj     shape: [256, hidden, intermediate]     = [256, 2048, 512]
```

Both are stored at `ttnn.bfloat4_b` (bfp4).

**Rationale:** At bfp8, 256 experts across 40 layers would require approximately ~30.0 GiB for expert weights alone, exceeding the 28 GB Blackhole DRAM ceiling with nothing left for non-expert weights, the KV cache, or activations. Halving to bfp4 brings expert weights to ~15.0 GiB, which is the dominant but manageable fraction of the 15.7 GB total DRAM budget.

The quality impact is bounded because individual expert quantization noise is averaged across the 8 selected experts per token. Routing weights (softmax probabilities) further attenuate any single expert's contribution. In practice, the A3B model exceeds the AmpereOne CPU baseline at 9.05 tok/s (llama.cpp Q4\_K) even at bfp4, reaching 11.7 tok/s.

### bfp8 — DeltaNet Projections and MLP Weights

DeltaNet linear attention projections use `ttnn.bfloat8_b`:

```
linear_attn.in_proj_all   (fused in_proj_qkv + in_proj_z + in_proj_b + in_proj_a)
linear_attn.out_proj
```

For the 27B dense model, MLP weights also use bfp8:

```
feed_forward.w1   (gate_proj)
feed_forward.w2   (down_proj)
feed_forward.w3   (up_proj)
```

For the A3B model, shared expert weights use bfp8 because the shared expert is always active and has proportionally higher impact on output quality than any single routed expert:

```
feed_forward.shared_expert.w1
feed_forward.shared_expert.w2
feed_forward.shared_expert.w3
```

**Rationale for DeltaNet projections:** The DeltaNet layer takes in 5120 (27B) or 2048 (A3B) hidden dimensions and projects to large intermediate dimensions. For 27B, the fused `in_proj_all` has output dimension $conv\_dim + V\_{dim} + H\_v + H\_v = (2 \times K\_{dim} + V\_{dim}) + V\_{dim} + H\_v + H\_v = 10240 + 6144 + 48 + 48 = 16480$. At bfp8 across 48 DeltaNet layers, projections account for 5.4 GB; at bf16 this would double to ~10.8 GB, which is prohibitive.

**Rationale for dense MLP (27B):** The 27B MLP intermediate size is 17408 dimensions. All three weight matrices together across 64 layers cost 17.1 GB at bfp8, which is already 61% of Blackhole DRAM. At bf16 it would be 34.2 GB, exceeding the device capacity outright.

**DRAM breakdown — 27B (Dense):**

| Component | Dtype | Size |
|-----------|-------|------|
| DeltaNet projections (48 layers) | bfp8 | 5.4 GB |
| MLP w1+w2+w3 (64 layers) | bfp8 | 17.1 GB |
| Attention QKV+WO+gate (16 layers) | bf16 | 2.2 GB |
| Other (norms, embeddings) | bf16 | ~0.3 GB |
| **Total** | | **~25 GB / 28 GB** |

### bf16 — Attention Weights, Router, and Norms

All attention-side weight matrices use `ttnn.bfloat16`:

```
attention.wq           q_proj.weight  (after gate split)
attention.wq_gate      q_proj_gate.weight
attention.wk           k_proj.weight
attention.wv           v_proj.weight
attention.wo           o_proj.weight
attention.q_norm.weight
attention.k_norm.weight
```

Router and shared-expert gate weights use bf16:

```
feed_forward.gate.weight               [hidden, num_experts]
feed_forward.shared_expert_gate.weight [hidden, 1]
```

RMSNorm scale weights throughout the model use bf16:

```
attention_norm.weight
ffn_norm.weight
norm.weight   (final RMSNorm before LM head)
```

The KV cache tensors (10 layers for A3B, 16 layers for 27B) are also allocated in bf16.

**Rationale:** Attention weight tensors are small relative to the MLP and expert weights. For A3B, attention QKV + WO + gate across 10 layers totals only 0.5 GB even at bf16. The gate mechanism in GatedAttention applies an element-wise $\sigma(\mathbf{x} W_{\text{gate}})$ sigmoid to the softmax attention output; this gate path is numerically sensitive because a small error in $W_{\text{gate}}$ propagates multiplicatively through every attention layer's output. Keeping gate weights in bf16 preserves the precision of this correction signal.

Router weights produce 256 logits used for `top-k` selection. Precision in routing directly determines which experts are activated; a quantization error that changes a logit ranking changes the entire expert set for a token. The 256-float router output is also tiny (1 KB per sync), so there is no DRAM incentive to quantize below bf16.

**DRAM breakdown — A3B (MoE):**

| Component | Dtype | Size |
|-----------|-------|------|
| Expert weights (256 × gate+up+down, 40 layers) | bfp4 | ~15.0 GiB |
| Shared expert weights (40 layers) | bfp8 | 0.8 GB |
| DeltaNet projections (30 layers) | bfp8 | 1.2 GB |
| Attention QKV+WO+gate (10 layers) | bf16 | 0.5 GB |
| Router + shared gate weights | bf16 | 0.1 GB |
| KV cache (10 layers) | bf16 | 0.3 GB |
| **Total** | | **~15.7 GB / 28 GB** |

### float32 — Recurrent State Tensor

The DeltaNet recurrent state is maintained as `ttnn.float32` on device DRAM:

```python
self._dev_state  # shape: [batch_size, H, head_k_dim, head_v_dim]
                 #        e.g. [1, 32, 128, 128] for A3B
```

**Rationale:** The DeltaNet recurrence accumulates across every token step:

$$\mathbf{S}_t = \mathbf{S}_{t-1} \cdot e^{g_t} + \mathbf{k}_t \otimes \boldsymbol{\delta}_t$$

where $\boldsymbol{\delta}_t = (\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t) \cdot \beta_t$. The state $\mathbf{S}_t \in \mathbb{R}^{H \times K \times V}$ participates in both the retrieval step $\mathbf{S}_{t-1}\mathbf{k}_t$ and the outer-product update $\mathbf{k}_t \delta_t^\top$. Any quantization error in $\mathbf{S}_{t-1}$ is amplified by the decay factor and then added back into the next state. Running this recurrence in bf16 across 30+ DeltaNet layers compounding errors at each step produces garbage output. The README states this observation directly: "bf16 compounds over 30+ layers producing garbage output."

The fp32 state does not carry a large DRAM cost: at A3B dimensions $[1, 32, 128, 128]$ the state is only $1 \times 32 \times 128 \times 128 \times 4 = 2$ MB per DeltaNet layer.

---

## Compute Kernel Configuration

Projection matmuls (DeltaNet in/out projections, attention QKV, MLP) use:

```python
ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

`HiFi2` provides two-pass FP16 accumulation in the matrix engine, giving better numerical fidelity than `LoFi` without the full cost of `HiFi4`. `fp32_dest_acc_en=False` keeps the destination accumulator in bf16, which is sufficient for bfp8 and bf16 input weights. `packer_l1_acc=True` enables L1-side accumulation buffering, improving throughput for the large projection matmuls at the cost of slightly more L1 usage.

---

**Next:** [`hf_to_meta_conversion.md`](./hf_to_meta_conversion.md)
