# Chapter 2 — Gated DeltaNet Deep Dive

## Overview

This chapter provides the complete mathematical and architectural treatment of the Gated DeltaNet
mechanism as used in Qwen3.6-35B-A3B. It covers the delta rule state update, gating, conv1d local
mixing, QK/V head asymmetries, and a systematic comparison to other linear attention variants.

Chapter 1 established that Qwen3.6-35B-A3B is architecturally identical to Qwen3.5-35B-A3B: 40
layers arranged as 10 repetitions of (3 Gated DeltaNet + 1 Gated Attention), each followed by a
MoE FFN block. This chapter zooms into the Gated DeltaNet component and derives everything from
first principles.

---

## Learning Objectives

After completing this chapter you will be able to:

1. Write the complete Gated DeltaNet recurrence from memory, identifying the roles of the decay
   gate $g_t$, the update rate $\beta_t$, the delta correction, the rank-1 outer-product write,
   and the output retrieval step.
2. Derive the decay gate $\alpha_t$ from the `A_log`, `dt_bias`, and `in_proj_a` parameters and
   explain why $g_t \in (0, 1)$ is guaranteed for any finite input.
3. Trace every projection in one Gated DeltaNet layer with full input/output shapes.
4. Explain the QK/V head asymmetry (16 QK heads vs 32 V heads) and the GQA-style
   `repeat_interleave` expansion.
5. Describe what the causal conv1d does, why it is applied before the Q/K/V split, and how it
   is implemented as a 4-slot shift register during decode.
6. Place Gated DeltaNet in the linear attention taxonomy relative to RetNet, GLA, Mamba2, and
   standard DeltaNet, and explain why it was selected for Qwen3.6.

---

## Notation

The notation below applies throughout this chapter. It is consistent with Chapter 1.

| Symbol | Value | Description |
|--------|-------|-------------|
| H | 2048 | Model hidden dimension |
| $d_k$ | 128 | DeltaNet key/query head dimension |
| $d_v$ | 128 | DeltaNet value head dimension |
| $H_k$ | 16 | DeltaNet key/query head count |
| $H_v$ | 32 | DeltaNet value head count |
| B | batch size | Variable |
| T | sequence length | Variable |
| $g_t$ | scalar $\in (0, 1)$ | Decay gate at step t |
| $\beta_t$ | scalar $\in (0, 1)$ | Update rate (delta step size) at step t |
| $S_t$ | $\mathbb{R}^{d_k \times d_v}$ | Recurrent state per head at step t |
| $\tilde{k}_t$ | $\mathbb{R}^{d_k}$ | L2-normalized key vector |
| $\tilde{q}_t$ | $\mathbb{R}^{d_k}$ | L2-normalized query vector; scaled by $1/\sqrt{d_k}$ in the output step: $o_t = S_t^\top(\tilde{q}_t/\sqrt{d_k})$ |
| $v_t$ | $\mathbb{R}^{d_v}$ | Value vector |
| $o_t$ | $\mathbb{R}^{d_v}$ | Output vector per head: $S_t^\top(\tilde{q}_t/\sqrt{d_k})$ |

---

## Files in This Chapter

| File | Contents |
|------|----------|
| [`delta_rule_formulation.md`](./delta_rule_formulation.md) | Complete recurrence derivation; term-by-term interpretation; decay gate derivation; state matrix dimensions and memory; L2 normalization; gated RMSNorm output; mamba_ssm_dtype |
| [`head_asymmetry_and_projections.md`](./head_asymmetry_and_projections.md) | QK/V head asymmetry and GQA expansion; full projection inventory with shapes; conv1d local mixing and decode shift register |
| [`comparison_to_linear_attention_variants.md`](./comparison_to_linear_attention_variants.md) | General gated linear attention form; RetNet, GLA, Mamba2, DeltaNet formulations; Gated DeltaNet as synthesis; summary comparison table; rationale for selection |

---

## Reading Order

Read the files in the order listed above:

1. **[`delta_rule_formulation.md`](./delta_rule_formulation.md)** — establishes the mathematical
   recurrence and all of its parameters. All other files reference the notation defined here.

2. **[`head_asymmetry_and_projections.md`](./head_asymmetry_and_projections.md)** — grounds the
   mathematics in the concrete projection shapes and hardware layout. Requires the recurrence
   variables from the first file.

3. **[`comparison_to_linear_attention_variants.md`](./comparison_to_linear_attention_variants.md)** —
   contextualizes Gated DeltaNet among related methods. Can be read independently if you only
   need the taxonomy.

---

## Cross-References to Existing Guides

This chapter focuses on the formulation as it applies to Qwen3.6. Two existing guides cover
complementary implementation details:

- **[`guides/gated_delta_net_and_gated_attention_on_t3k/`](../../../gated_delta_net_and_gated_attention_on_t3k/index.md)** —
  T3K-specific implementation, covering sharding strategies, roofline analysis, TTNN primitive
  mapping, and kernel gaps. Chapter 2 of that guide
  ([`ch2_gated_delta_net_math_and_recurrence/`](../../../gated_delta_net_and_gated_attention_on_t3k/ch2_gated_delta_net_math_and_recurrence/index.md))
  provides the same recurrence with additional coverage of the chunkwise WY-decomposition for
  parallel prefill and a state-vs-KV-cache memory comparison.

- **[`guides/qwen35_implementation/`](../../../qwen35_implementation/index.md)** — Blackhole
  P100A implementation. Chapter 2 of that guide
  ([`ch2_gated_deltanet_linear_attention_on_blackhole/`](../../../qwen35_implementation/ch2_gated_deltanet_linear_attention_on_blackhole/index.md))
  covers the fused `ttnn.experimental.gated_delta_net` kernel, the `host_recurrence` fallback
  path (required because Blackhole does not support fp32 compute buffers in bfloat16 pipelines),
  and the circular ring-buffer implementation of the conv1d in Metal Trace mode.

Because Qwen3.6-35B-A3B is architecturally identical to Qwen3.5-35B-A3B (see Chapter 1), all
implementation details in those guides apply to Qwen3.6 without modification.

---

## Key Takeaways (Preview)

- The Gated DeltaNet recurrence combines a scalar decay gate (from GLA) with a targeted
  error-correcting write (from the delta rule). Neither mechanism alone is sufficient for both
  selective forgetting and precise memory correction.
- The state $S \in \mathbb{R}^{d_k \times d_v} = \mathbb{R}^{128 \times 128}$ per head is kept
  in fp32 for numerical stability; the full per-layer state at B=1 is approximately 2 MB,
  independent of sequence length T.
- The 16/32 head asymmetry (QK vs V) is resolved by a GQA-style `repeat_interleave(2)` that
  doubles Q and K to 32 heads before the recurrence, reducing projection cost without shrinking
  the state matrix.
- Conv1d (kernel size 4) applied before the Q/K/V split gives each token a 4-token local
  receptive field, complementing the global (but bounded-capacity) recurrent state.

---

Begin reading: [`delta_rule_formulation.md`](./delta_rule_formulation.md)
