# Guide: Gated Delta Net and Gated Attention on T3K

This guide answers the eight research questions for Qwen-Coder-Next / Qwen3.5-35B-A3B's hybrid attention architecture on the T3K 1×8 Wormhole mesh.

## Model Context

**Qwen3.5-35B-A3B** uses a hybrid layer stack:
- 30 **Gated Delta Net** layers — linear-recurrent attention with a fixed-size state matrix $S \in \mathbb{R}^{d_k \times d_v}$ updated via the delta rule with a scalar decay gate
- 10 **Gated Attention** layers — standard SDPA with Q-sigmoid-gating and Q/K RMSNorm (not Gated Linear Attention)

Hidden dimension H = 2048. Gated Delta Net: H_v = 32 value heads, H_k = 16 key heads, d_k = d_v = 128. Gated Attention: n_q_h = 16, n_kv_h = 2, d_h = 256.

---

## Chapters

| Chapter | Directory | Questions Answered |
|---|---|---|
| [1 — Attention Variants and Linear Attention Foundations](./ch1_attention_variants_and_linear_attention_foundations/index.md) | `ch1_attention_variants_and_linear_attention_foundations/` | Q1 (foundations) |
| [2 — Gated Delta Net: Mathematical Formulation and Recurrence](./ch2_gated_delta_net_math_and_recurrence/index.md) | `ch2_gated_delta_net_math_and_recurrence/` | Q1, Q3 |
| [3 — Gated Attention: Mechanism and Tensor Shapes](./ch3_gated_attention_mechanism/index.md) | `ch3_gated_attention_mechanism/` | Q2 |
| [4 — TTNN Primitive Mapping: Decode and Prefill Forward Passes](./ch4_ttnn_primitive_mapping/index.md) | `ch4_ttnn_primitive_mapping/` | Q4, Q8 |
| [5 — Roofline Analysis on a Single Wormhole Chip](./ch5_roofline_analysis/index.md) | `ch5_roofline_analysis/` | Q6 |
| [6 — T3K Sharding Strategy for Gated Delta Net State](./ch6_t3k_sharding/index.md) | `ch6_t3k_sharding/` | Q5, Q7 |
| [7 — Existing Implementations, Kernel Gaps, and Development Roadmap](./ch7_kernel_gaps_and_roadmap/index.md) | `ch7_kernel_gaps_and_roadmap/` | Q8 |

---

## Key Findings by Question

**Q1 — What is Gated Delta Net?**
Gated Delta Net combines GLA-style scalar decay ($g_t = \exp(\alpha_t) \in (0,1)$) with DeltaNet-style error-correcting writes. Master recurrence:

$$S_t = g_t \cdot S_{t-1} + \tilde{k}_t \left(\beta_t \cdot \left(v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t\right)\right)^\top$$

See Ch1 (taxonomy) and Ch2 (derivation).

**Q2 — What is Gated Attention?**
Qwen3.5-specific SDPA: Q is projected with a sigmoid gate ($Q_{\text{gated}} = \sigma(\text{gate}) \odot Q$), then Q and K are normalized per-head (RMSNorm), then RoPE applied to 64 of 256 head dimensions, then standard SDPA with 2 KV heads expanded 8× to serve 16 Q heads. See Ch3.

**Q3 — Can Gated Delta Net be parallelized?**
The recurrence is strictly sequential at the token level. Chunkwise parallelism (WY-decomposition, C=64 chunks) enables within-chunk parallelism for prefill. Associative scan allows theoretical inter-chunk parallelism but is not used in the current implementation. See Ch2.

**Q4 — What TTNN operations are needed?**
Decode: `ttnn.linear` (projections), `ttnn.mul` (decay, gating), `ttnn.matmul` (state retrieval, outer-product write, output query), `ttnn.rms_norm`, `ttnn.sigmoid`, `ttnn.experimental.all_gather_async`. Prefill additionally requires chunkwise QK/AV matmuls. See Ch4.

**Q5 — Memory footprint comparison?**
Gated Delta Net state: 1 MiB/layer (fixed, B=1), 30 MiB total. Gated Attention KV cache: 512 MiB/layer at T=262,144, 5.12 GiB total for 10 layers — the dominant memory consumer at long context. See Ch6.

**Q6 — Compute-bound or bandwidth-bound?**
Heavily bandwidth-bound in both modes. Decode: 1.74 FLOP/byte (262× below the 455 FLOP/byte ridge). Prefill (T=8192, C=64): 32.0 FLOP/byte (14.2× below ridge). DeltaNet decode is 253× faster per layer than Gated Attention at T=262,144. See Ch5.

**Q7 — T3K sharding strategy?**
Head-parallel blocked sharding: each of 8 devices holds 4 value heads. Per-device state: 128 KiB/layer. CCL: one 4 KiB all-gather per layer (0.16 µs — 2.2% of state I/O time). Gated Attention KV heads replicated (n_kv_h=2 is non-divisible by 8). See Ch6.

**Q8 — Which kernels exist, which are gaps?**
Available in TTNN: all projections, SDPA (both modes), paged KV cache, Q/K RMSNorm, Q gating, RoPE, all CCL. Four required custom-kernel gaps: `recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`. Priority 1 (highest impact): fused recurrent decode kernel with L1-resident state. See Ch7.
