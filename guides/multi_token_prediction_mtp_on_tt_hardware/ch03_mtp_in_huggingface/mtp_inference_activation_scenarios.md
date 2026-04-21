# MTP Inference Activation Scenarios

## Section 1: Summary Table

| Scenario | MTP head active? | Code change required | Correctness risk |
|---|---|---|---|
| Standard `model.generate()` (greedy/sampling) | **No** — MTP gate requires both `labels is not None` AND `self.training is True`; generate() satisfies neither | None | None |
| Manual speculative decoding using MTP logits | **Yes** (explicit call) | Yes — custom generation loop | Medium — requires correct accept/reject logic |
| HuggingFace `AssistantModel`-based spec decoding | **No** — MTP head not recognized as assistant model | Yes — custom adapter class | Low — well-defined interface |
| TT-Symbiote current inference path | **No** | None | None |
| TT-Symbiote MTP speculative decoding | **Yes** (designed) | Yes — Chapter 5 details | Medium |

---

## Section 2: Scenario A — Standard Generation (No MTP)

The MTP head is not called during `model.generate()` — the mechanism is covered in [`huggingface_mtp_forward_pass.md`](./huggingface_mtp_forward_pass.md). The generation loop produces only the standard backbone logit for position `t+1` at each step.

Inference throughput is identical to Qwen3.5: under Scenario B loading the MTP head weights are present in memory but contribute zero FLOPs; under Scenario A loading they are not in device memory at all.

---

## Section 3: Scenario B — Manual Speculative Decoding

To use MTP draft logits for speculative decoding, the generation loop must be fully replaced with a custom implementation. The required steps at each draft cycle are:

1. Call `model()` with `output_hidden_states=True` to obtain the backbone's final hidden states for the current prefix.
2. Extract the last-layer final-token hidden state: `hidden_state = output.hidden_states[-1][:, -1:, :]` (shape `[batch, 1, H]`). Pass this to the MTP head (call signature per Chapter 1 architecture): `model.future_prediction[0](hidden_state, shifted_embedding)` to produce draft logits for the next-next position.
3. Sample `N` draft tokens from the draft logits.
4. Run a full verification forward pass over the `N` draft tokens appended to the current context.
5. Apply the accept/reject logic to determine how many draft tokens to accept before resampling.

None of this is supported by any HuggingFace generation utility. `GenerationMixin` has no hook point for injecting a sub-module draft step into the generation loop. The entire loop must be written from scratch.

Chapter 4 details the speculative decoding algorithm. Chapter 5 provides the TTNN implementation plan for TT hardware.

---

## Section 4: Scenario C — HuggingFace `AssistantModel` Interface

HuggingFace's `AssistantModel`-based speculative decoding (introduced in Transformers ≥ 4.41) expects an external draft model with a `.generate()` interface — a fully independent `PreTrainedModel` instance.

The MTP head does not satisfy this interface. It is a sub-module (`nn.Module`) of the main model, not a standalone `PreTrainedModel`. It has no `config`, no tokenizer binding, and no `.generate()` method. Wrapping it as a standalone `AssistantModel` would require a custom adapter class that:

- Holds a reference to the parent model's backbone
- Implements `.generate()` by calling the backbone forward and then the MTP head forward
- Manages KV cache state that is consistent with the parent model's decode state

This is a non-trivial engineering effort. No such adapter is available in the HuggingFace ecosystem today.

---

## Section 5: Implication for TT-Symbiote Bring-Up

**Current inference path (standard autoregressive decode):**
The MTP head is irrelevant. See [`mtp_weight_loading_behavior.md`](./mtp_weight_loading_behavior.md) for the key-filtering procedure. No code changes to the generation loop are needed; the model behaves identically to Qwen3.5 at inference time.

**Future MTP speculative decoding path:**
The MTP head weights must be preserved and loaded into dedicated TTNN tensors. The generation loop must be redesigned to implement the draft-verify cycle described in Scenario B above. Chapter 5 covers the TTNN implementation plan for this path.

---
**Next:** [Chapter 4 — Speculative Decoding with MTP on TT Hardware](../ch04_speculative_decoding_with_mtp/index.md)
