# Acceptance Rate Estimation

## What the Acceptance Rate Measures

The acceptance rate α is the token-level probability that the draft model's sampled token matches what the target model would have sampled. Formally, for draft distribution q and target distribution p:

```
α = E_{x ~ q}[ min(1, p(x) / q(x)) ]
```

In practice, α is measured empirically: run the full loop on held-out prompts, count how often the draft token is accepted.

## Estimated Values by Domain

The following acceptance rate ranges are illustrative estimates for MTP-style draft models, based on the known domain-dependence of token predictability. They are **not directly cited** from a specific published measurement of MTP speculative decoding; the DeepSeek-V3 technical report (the architectural ancestor of Qwen3.6's MTP design) describes MTP primarily as an auxiliary training objective and does not report spec-decoding acceptance rates.

| Domain | Estimated α range |
|---|---|
| Code generation | 0.75 – 0.85 |
| Mathematical reasoning | 0.65 – 0.80 |
| General instruction following | 0.55 – 0.75 |

These ranges reflect the observation that high-constraint outputs (code syntax, math notation) tend to produce tighter draft–target distribution alignment than open-ended text. Treat them as order-of-magnitude priors pending empirical measurement on your actual workload.

## Domain Dependence

Acceptance rates vary significantly by task type because acceptance depends on how predictable the next token is:

- **Coding**: high repetition, syntactically constrained output (brackets, keywords, identifiers). Draft and target distributions align closely → high α.
- **Math**: structured step sequences, LaTeX notation, operator patterns → moderately high α.
- **Conversational / creative**: high entropy, many equally valid next tokens → lower α, draft and target diverge more.

This means throughput improvements from MTP spec decoding will be domain-dependent. A deployment serving primarily code completion will see better results than one serving open-ended dialogue.

## Qwen3.6-35B-A3B: Expected Range

The Qwen3.6 MTP head has not been evaluated for acceptance rate in publicly available benchmarks. Based on the joint training approach and the domain-dependence of token predictability, the expected range is:

```
α ∈ [0.5, 0.8]  depending on domain
```

The lower bound (0.5) applies to high-entropy conversational tasks; the upper bound (0.8) applies to code and math. These are estimates pending empirical measurement.

## How to Measure Empirically

```python
def measure_acceptance_rate(model, prompts, n_steps=50):
    accepts = 0
    total   = 0

    for prompt in prompts:
        context = tokenize(prompt)

        for _ in range(n_steps):
            # Primary pass
            out = model(context, output_hidden_states=True)
            primary_logits = out.logits[:, -1, :]
            x_t1 = sample(primary_logits)

            # Draft
            h      = out.hidden_states[-1][:, -1:, :]
            emb_t1 = model.model.embed_tokens(x_t1.unsqueeze(1))
            draft_logits = model.future_prediction[0](h, emb_t1).squeeze(1)
            x_hat  = sample(draft_logits)

            # Verify
            verify_out = model(
                input_ids=torch.cat([x_t1.unsqueeze(1), x_hat.unsqueeze(1)], dim=1),
                past_key_values=out.past_key_values,
            )
            p_x_hat = verify_out.logits[:, 0, :].softmax(-1)[0, x_hat]
            q_x_hat = draft_logits.softmax(-1)[0, x_hat]

            accept_prob = min(1.0, (p_x_hat / q_x_hat).item())
            accepted    = (torch.rand(1).item() < accept_prob)

            accepts += int(accepted)
            total   += 1

            # Advance context (simplified: always take x_t1; conditionally take x_hat)
            context = advance(context, x_t1, x_hat if accepted else None)

    return accepts / total
```

Run this over 50–100 prompts per domain (code, math, chat) to get domain-stratified estimates before committing to any deployment configuration.

## Sensitivity of Throughput to α

Recall from the throughput analysis: at batch size 1, speedup = (1+α)/2 — always below 1. At larger batches the picture improves. The table below shows how throughput scales with α at batch sizes where verification cost is reduced (assuming `C_verify ≈ 0.7 × C_decode` at batch 32, reflecting partial compute saturation).

| α | Speedup, batch 1 | Speedup, batch 32 (estimated) |
|---|---|---|
| 0.5 | 0.75 | ~1.07 |
| 0.7 | 0.85 | ~1.18 |
| 0.8 | 0.90 | ~1.24 |
| 0.85 | 0.925 | ~1.27 |

The batch 32 estimates are illustrative. Actual values require profiling on TT hardware to measure the true cost ratio of a 2-token prefill vs. a 1-token decode at that batch size.

## Key Finding

> The Qwen3.6 MTP head acceptance rate is unknown and must be measured empirically. Based on the domain-dependence of token predictability, expect α ∈ [0.5, 0.8] as an order-of-magnitude prior, with code and math at the high end. Even at α = 0.85, the throughput benefit at batch size 1 is negative on bandwidth-bound TT hardware. Measuring α is still worthwhile to characterize the regime at which larger batch deployments become profitable.

---
**Next:** [Chapter 5 — Custom TTNN Generation Loop](../../ch05_ttnn_implementation/index.md) *(forthcoming)*
