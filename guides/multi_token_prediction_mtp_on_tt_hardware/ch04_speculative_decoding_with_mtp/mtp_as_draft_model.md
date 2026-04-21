# MTP as the Draft Model

## How MTP Maps Onto the Draft-Model Role

Standard speculative decoding uses a separate, smaller model as the drafter. MTP replaces that with a lightweight head attached to the backbone's final hidden state.

For Qwen3.6-35B-A3B:

- `mtp_num_hidden_layers = 1` → one MTP head block, one draft position.
- The head takes `(hidden_state, shifted_embedding)` and produces logits for position **t+2** given a context ending at **t**.
- Head parameters: ~160M, 304.6 MiB BF16, accessed via `model.future_prediction[0]`.
- Draft cost: the hidden state is already computed during the primary decode pass; the MTP head is a shallow 1-layer transformer applied to a single vector.

This maps directly onto the draft-model role with K=1 and near-zero marginal draft cost.

## Constraint: Custom Generation Loop Required

Chapter 3 confirmed:

- The MTP head is gated by `labels is not None AND self.training is True` in the HuggingFace implementation.
- `model.generate()` never invokes `future_prediction`.
- `GenerationMixin` has no hook point for injecting MTP into the loop.

Everything below requires a **from-scratch generation loop** that calls the backbone and MTP head explicitly.

## Algorithm: MTP Speculative Decode for Qwen3.6 (N=1)

The following pseudocode describes one speculative decode cycle starting from context `x_1..x_t`.

```python
# --- Step 1: Primary backbone forward pass ---
output = model(
    input_ids=context_ids,          # shape [batch, t]
    use_cache=True,
    output_hidden_states=True,
)
# Primary logits for position t+1
primary_logits = output.logits[:, -1, :]           # [batch, vocab]
past_kv        = output.past_key_values

# --- Step 2: Sample primary token ---
x_t1 = sample(primary_logits)                      # [batch]
# x_t1 is sampled directly from the target backbone (p); no accept/reject needed for this token

# --- Step 3: Draft token via MTP head ---
# hidden_state from the final backbone layer at the last position
h = output.hidden_states[-1][:, -1:, :]            # [batch, 1, H]

# shifted_embedding: embedding of x_t1 (the just-sampled token)
emb_t1 = model.model.embed_tokens(x_t1.unsqueeze(1))  # [batch, 1, H]

draft_logits = model.future_prediction[0](h, emb_t1)   # [batch, 1, vocab]
draft_logits = draft_logits.squeeze(1)                 # [batch, vocab]

# --- Step 4: Sample draft token ---
x_hat_t2 = sample(draft_logits)                    # [batch]
# Record q(x_hat_t2) = draft_logits probability for accept/reject

# --- Step 5: Verification pass ---
# Feed [x_t1, x_hat_t2] to backbone as a 2-token prefill
verify_ids = torch.cat([x_t1.unsqueeze(1),
                        x_hat_t2.unsqueeze(1)], dim=1)  # [batch, 2]

verify_out = model(
    input_ids=verify_ids,
    past_key_values=past_kv,    # reuse KV cache from step 1
    use_cache=True,
    output_hidden_states=False,
)
# verify_out.logits[:, 0, :] = p(· | x_1..x_t, x_t1)   → for accepting x_hat_t2
# verify_out.logits[:, 1, :] = p(· | x_1..x_t, x_t1, x_hat_t2) → bonus token if accepted

p_verify_t2 = verify_out.logits[:, 0, :]           # [batch, vocab]

# --- Step 6: Accept/reject x_hat_t2 ---
p_accepted  = p_verify_t2.softmax(-1)[range(batch), x_hat_t2]
q_drafted   = draft_logits.softmax(-1)[range(batch), x_hat_t2]
accept_prob = torch.clamp(p_accepted / q_drafted, max=1.0)
accept_mask = torch.rand(batch) < accept_prob

# --- Step 7: Advance context ---
# Accepted path: append [x_t1, x_hat_t2] (advance by 2); cycle cost = 2 × C_decode
# Rejected path: append [x_t1] only (advance by 1); x_hat_t2 is discarded
#   Do NOT resample position t+2 in this cycle — resampling would always produce 2 tokens
#   per cycle regardless of α, collapsing E[tokens/cycle] to 2 and speedup to 1.0.
#   Without resampling: E[tokens/cycle] = (1-α)×1 + α×2 = 1+α → speedup = (1+α)/2.
```

## Comparison to an External Draft Model

| Property | External small model | MTP head (Qwen3.6) |
|---|---|---|
| Draft cost | Full inference on a second model | Single shallow head pass on existing hidden state |
| Memory footprint | Second model loaded separately | 304.6 MiB, already part of checkpoint |
| Draft tokens per cycle (K) | Arbitrary (K=3–7 typical) | 1 (N=1 for Qwen3.6) |
| Distribution alignment | Separate training; may diverge | Trained jointly with backbone |
| Implementation complexity | Two models, two KV caches | One model, one KV cache, explicit head call |

The MTP head is a significantly cheaper drafter than a separate model. The limitation is K=1 with the current Qwen3.6 config. Future checkpoints with `mtp_num_hidden_layers > 1` could support iterative chaining (K=2, K=3) using successive head blocks.

## Key Finding

> The MTP head fits the draft-model role cleanly: it is cheap to run, jointly trained, and produces draft logits as a side effect of the primary decode pass. For Qwen3.6-35B-A3B, K=1 (one draft position). The custom loop above is the correct implementation; whether K=1 yields a throughput improvement on TT hardware is determined by the bandwidth cost model in the next file.

---
**Next:** [throughput_analysis_on_tt_hardware.md](throughput_analysis_on_tt_hardware.md)
