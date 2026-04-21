# Speculative Decoding Primer

## The Core Idea

Speculative decoding separates token generation into two roles:

- **Draft model**: fast, cheap; generates K candidate tokens speculatively.
- **Verifier (target model)**: the full model; evaluates all K draft tokens in one parallel forward pass.

Because the verifier runs a *prefill-style* pass over K tokens rather than K independent decode passes, it can accept multiple tokens per cycle without changing the output distribution. The accepted tokens are identical in distribution to what the target model would have produced autoregressively.

## The Algorithm

```
Input: prompt tokens x_1..x_t, draft model q, target model p, draft length K

Loop:
  1. Draft:  sample x̂_{t+1}, x̂_{t+2}, ..., x̂_{t+K} from q autoregressively
  2. Verify: run p(x_1..x_t, x̂_{t+1}..x̂_{t+K}) in one forward pass
             → get target probs p(· | x_1..x_{t+i}) for i=1..K
  3. Accept/Reject (left to right):
       for i = 1..K:
         accept x̂_{t+i} with prob min(1, p(x̂_{t+i}) / q(x̂_{t+i}))
         if rejected: resample x_{t+i} from adjusted distribution; break
  4. If all K accepted: sample one bonus token from p(· | x_1..x_{t+K})
  5. Advance context by (accepted_count + 1) tokens
```

The accept/reject rule guarantees the output distribution equals the target model's distribution exactly (no approximation).

## Expected Tokens Per Cycle

Let α = token-level acceptance rate (probability a single draft token is accepted). Assuming independence across draft positions:

```
P(exactly j tokens accepted from K drafts) = α^j * (1-α)  for j = 0..K-1
P(all K accepted)                           = α^K

E[draft tokens accepted only] = sum_{j=0}^{K-1} j * α^j * (1-α)  +  K * α^K
                              = α(1 - α^K) / (1 - α)

E[tokens per cycle] = 1 + E[draft tokens accepted only]
                    = (1 - α^{K+1}) / (1 - α)   [for α < 1]
```

For K=1 (single draft token):

```
E[tokens per cycle] = 1 + α
```

This is the formula relevant to `mtp_num_hidden_layers = 1`.

## Speedup Formula

Let `C_draft` = cost of drafting K tokens, `C_verify` = cost of the verification pass.

```
Speedup = E[tokens per cycle] / (C_draft + C_verify)
        vs.
Baseline = 1 token / C_decode   (one regular decode step)

Relative speedup = E[tokens per cycle] * C_decode / (C_draft + C_verify)
```

## Why Spec Decoding Targets Bandwidth-Bound Hardware

On memory-bandwidth-bound hardware, the dominant cost of a decode step is loading model weights from DRAM — not compute. A single-token decode and a K-token prefill-style pass load the *same* weights once.

This means:

- `C_verify(K tokens) ≈ C_decode(1 token)` in the bandwidth-bound regime.
- A conventional external draft model adds a full second model's worth of bandwidth overhead.
- **Embedded draft mechanisms** (like MTP) are attractive because they produce draft tokens with near-zero *incremental* cost during the primary decode pass.

The catch, which the throughput analysis makes precise, is that the verification pass still costs one full backbone traversal. With only K=1 draft token (N=1), the gain of `α` additional tokens per cycle must pay for that entire extra pass.

## Key Finding

> Speculative decoding is most beneficial when: (a) `C_draft` is small relative to `C_verify`, and (b) `E[tokens per cycle]` is large enough that the verification cost is amortized. With an embedded draft mechanism like MTP, condition (a) is satisfied. Whether condition (b) is satisfied depends on K (number of draft tokens) and α. For K=1, E[tokens] = 1+α, and the verification pass costs nearly as much as a regular decode — leaving minimal room for speedup.

---
**Next:** [mtp_as_draft_model.md](mtp_as_draft_model.md)
