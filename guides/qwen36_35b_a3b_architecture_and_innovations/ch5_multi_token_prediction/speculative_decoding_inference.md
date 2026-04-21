# Speculative Decoding at Inference

## Overview

This section covers the mechanics of MTP-based speculative decoding: how the MTP head acts as a draft model, the accept/reject loop, throughput tradeoffs, and TTNN implications.

---

## What Is Speculative Decoding?

Standard autoregressive decoding generates one token per forward pass. Speculative decoding breaks this serial bottleneck by using a cheap draft model to propose candidate tokens and a verification pass through the full model to accept or reject them. The full model is the authority: if a draft token matches what the full model would have chosen, both are accepted; if not, the draft is discarded and only the verified token is kept. The output distribution is identical to running the full model alone.

MTP provides a natural built-in draft model: the MTP transformer layer is already trained to predict the token at $t+2$ given the hidden states at position $t$. At inference time this can be re-framed as predicting the token at position $t+1$ given hidden states that are one step ahead of where the current forward pass has computed.

---

## Speculative Decoding Steps Using the MTP Head

The following describes a single speculative decoding step for generating token at position $t+1$:

**Step 1 — Main model forward pass for position $t$.**
The main 40-layer decoder processes the current sequence up to token $t$ and produces final hidden states $H_t \in \mathbb{R}^{d_\text{model}}$ for position $t$. The standard LM head produces logits over the vocabulary, and token $x_{t+1}$ is sampled (or greedily selected).

**Step 2 — MTP head produces a draft for position $t+2$.**
Simultaneously (or immediately after), the MTP transformer layer is applied to $H_t$:

$$H_t^\text{mtp} = \text{TransformerLayer}_\text{mtp}(H_t)$$

$$\text{logits}_{t+2}^\text{draft} = H_t^\text{mtp} \cdot E^T$$

The draft token $\hat{x}_{t+2}$ is sampled from this distribution.

**Step 3 — Next forward pass processes both tokens together.**
The next decoder pass processes the sequence extended by both $x_{t+1}$ (the verified token from step 1) and $\hat{x}_{t+2}$ (the draft token from step 2). The main decoder produces hidden states and logits at both positions.

**Step 4a — Draft accepted.**
The acceptance decision follows the standard speculative decoding criterion (Leviathan et al., 2023). Let $p$ denote the main model's distribution and $q$ denote the MTP draft distribution at position $t+2$. The draft token $\hat{x}_{t+2}$ is accepted with probability:

$$\min\!\left(1,\; \frac{p(\hat{x}_{t+2})}{q(\hat{x}_{t+2})}\right)$$

If accepted, both $x_{t+1}$ and $x_{t+2}$ are committed to the output. The model has generated two tokens in the time it would normally take to generate one (plus the overhead of one MTP layer evaluation). Under greedy decoding this reduces to a simple argmax equality check; under stochastic sampling the ratio test is required to preserve the lossless guarantee.

**Step 4b — Draft rejected.**
If the draft is rejected, it is discarded and a corrected token is sampled from the adjusted distribution $\text{norm}(\max(0, p - q))$ at position $t+2$. Only $x_{t+1}$ is committed. No incorrect token is ever emitted; the output distribution remains identical to that of the main model alone.

This process repeats for each subsequent position, with the MTP head continuously issuing one draft token ahead.

---

## Acceptance Rate

The fraction of draft tokens that the full model accepts is called the **acceptance rate** $\alpha$. For a well-trained MTP head:

$$\alpha \approx 50\%\text{–}80\%$$

Acceptance rate depends on:

- **Task type.** Factual, formulaic, or code generation tasks with lower entropy next tokens tend toward the higher end. Open-ended creative generation with high entropy tends toward the lower end.
- **Training quality of the MTP head.** A head trained with appropriate loss weighting $\lambda$ converges to a better predictor of two-step-ahead tokens.
- **Sampling temperature.** Greedy decoding (temperature = 0) gives the highest acceptance rates because both the draft and the verification pass use the same deterministic argmax. Stochastic sampling reduces acceptance rates because the draft and the verified sample may diverge even when the distributions are similar.

---

## Throughput Tradeoff

Let:

- $C_\text{main}$ = compute cost of one full 40-layer decoder forward pass
- $C_\text{mtp}$ = compute cost of one MTP transformer layer pass
- $\alpha$ = acceptance rate

Without speculative decoding, generating $N$ tokens requires $N$ full decoder passes:

$$\text{Cost}_\text{baseline} = N \cdot C_\text{main}$$

With MTP speculative decoding, each step costs one main pass plus one MTP pass, but on average produces $1 + \alpha$ tokens:

$$\text{Cost}_\text{speculative} = \frac{N}{1 + \alpha} \cdot (C_\text{main} + C_\text{mtp})$$

The speedup ratio is:

$$\text{Speedup} = \frac{C_\text{main}}{(C_\text{main} + C_\text{mtp}) / (1 + \alpha)} = \frac{(1 + \alpha) \cdot C_\text{main}}{C_\text{main} + C_\text{mtp}}$$

Since the MTP module is one dense layer while the main decoder is 40 mostly-MoE layers, $C_\text{mtp} \ll C_\text{main}$. Crucially, however, the main model's MoE layers are **sparse** — each layer activates only 9 expert FFN paths (8 routed + 1 shared) out of 256+. The per-token FLOPs of one MoE layer are therefore far less than those of a full dense layer of the same $d_\text{model}$. The MTP dense layer's 4× FFN costs more FLOPs per token than a single sparse MoE FFN, so the effective ratio is:

$$\frac{C_\text{mtp}}{C_\text{main}} \approx \frac{1 \text{ dense layer}}{40 \text{ sparse-MoE layers}} \approx 0.035$$

(rather than the naïve 1/40 = 0.025, because the MTP layer is denser per token than each MoE layer). With $\alpha = 0.65$:

$$\text{Speedup} \approx \frac{1.65 \cdot C_\text{main}}{1.035 \cdot C_\text{main}} \approx 1.59\times$$

The practical speedup observed in deployed systems is typically 1.4–1.8x depending on the hardware memory bandwidth profile and batch size. Memory-bandwidth-bound regimes (small batch, high memory bandwidth pressure) benefit most because the extra MTP layer adds minimal wall-clock time while each accepted draft saves a full memory traversal of all model weights.

---

## Accuracy Guarantee

Speculative decoding with an accept/reject loop is **lossless** with respect to the full model's output distribution. This is a mathematical guarantee, not an empirical approximation:

- Accepted tokens are exactly the tokens the full model would have produced.
- Rejected drafts are replaced by the token the full model determines at that position.
- The output sequence is identical in distribution to a sequence generated by the main decoder alone.

This means enabling MTP speculative decoding cannot degrade model quality. It is a pure throughput optimization.

---

## TTNN Implications

### Case 1: MTP Disabled at Inference (Default)

When MTP speculative decoding is not enabled:

- The MTP layer weights are present in the checkpoint but never loaded into device memory (or loaded but never called).
- The main decoder's forward pass is unmodified.
- No changes to existing TTNN graph compilation, weight loading, or KV cache management are required.
- This is the zero-cost baseline: MTP has no inference-time effect whatsoever.

### Case 2: MTP Speculative Decoding Enabled

Supporting MTP-based speculative decoding in TTNN requires the following additional engineering:

**MTP layer weight loading.**
The single MTP transformer layer must be loaded onto device memory. Its attention weights, FFN weights, and normalization parameters are a small but non-zero addition to the device memory footprint.

**MTP forward pass implementation.**
A TTNN subgraph must be compiled for the MTP transformer layer. This mirrors the implementation of a single main decoder layer but without MoE routing (the MTP layer is dense). Key operations:

- RMSNorm
- Grouped-query attention (with or without a separate KV cache for the MTP layer, depending on design)
- Dense FFN
- Residual addition

**Verify-and-accept logic.**
After each main decoder pass, a comparator must check whether the draft token matches the main model's argmax at the draft position. This is a lightweight operation but must be implemented as part of the decoding loop. The control flow is (greedy-decoding variant; stochastic sampling requires the full probability-ratio acceptance criterion from Steps 4a/4b above):

```
draft_logits = mtp_forward(H_final)
x_next       = main_lm_head(H_final)         # committed
x_draft      = argmax(draft_logits)
# --- next forward pass ---
verify_logit = main_lm_head(H_final_at_t+2)
if argmax(verify_logit) == x_draft:          # greedy accept check
    commit(x_draft)   # both x_next and x_draft accepted
else:
    commit(argmax(verify_logit))   # only verified token accepted; draft discarded
```

**Variable-length token sequences per step.**
When a draft is accepted, the KV cache is extended by two positions in one step instead of one. When a draft is rejected, only one position is appended. The KV cache management logic must handle this variable-length extension cleanly. On TTNN this means the paged attention or static KV cache update logic must account for steps that write either one or two new cache entries.

**Batching considerations.**
In a batched setting, different sequences in the same batch may have different acceptance outcomes for the same step. This creates a variable-length problem across the batch dimension, which complicates padding and mask logic. A common solution is to process speculative decoding in small per-sequence units or to accept that some batched steps will waste compute on rejected drafts.

### Summary of TTNN Work Items

| Item | Required for MTP disabled | Required for MTP speculative decoding |
|------|--------------------------|--------------------------------------|
| Load MTP layer weights | No | Yes |
| Compile MTP subgraph | No | Yes |
| Verify-and-accept loop | No | Yes |
| Variable KV cache extension | No | Yes |
| Changes to main decoder | No | No |
| Changes to sampling / output | No | Minor (accept/reject bookkeeping) |

The cleanest implementation strategy is to make MTP speculative decoding a feature flag. When the flag is off, the code path is identical to a standard single-head decoder. When the flag is on, the additional subgraph and control flow are activated. This avoids burdening the default inference path with any speculative-decoding overhead.

---

**Next:** [Chapter 6 — Thinking Preservation](../ch6_thinking_preservation/index.md)
