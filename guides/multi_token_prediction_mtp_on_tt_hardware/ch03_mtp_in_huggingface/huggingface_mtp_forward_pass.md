# HuggingFace MTP Forward Pass

## Section 1: `Qwen3_5MoeForConditionalGeneration.forward()` — MTP Head Invocation

`Qwen3_5MoeForConditionalGeneration.forward()` accepts a `labels` argument for computing the next-token-prediction loss. The MTP head's `forward()` is invoked only when two conditions are simultaneously true:

1. `labels is not None` — a target sequence is provided
2. `self.training is True` — the model is in training mode

**During training:** `labels` are provided to `forward()`. Both the standard autoregressive loss (`L_AR`) and the auxiliary MTP loss (`L_aux`) are computed. The combined loss is returned:

```math
L_total = L_AR + λ · L_aux
```

where `λ` is the MTP loss weighting coefficient configured at training time.

**During eval mode (`model.eval()`):** All standard inference code paths pass `labels=None` to `forward()`. The MTP head `forward()` is never called. The 11 MTP weight tensors (`model.future_prediction.0.*`) sit unused in GPU or host memory for the duration of inference.

---

## Section 2: `model.generate()` and the MTP Head

`GenerationMixin.generate()` builds a generation loop that calls `model.forward()` iteratively (via `model.__call__`) at each decode step.

In each generation step, `model.forward()` is called with `input_ids` and `attention_mask`. The `labels` argument is never passed by the generation loop — not by any built-in generation strategy.

Specific flags that do **not** cause `labels` to be passed:

- `return_dict_in_generate=True` — affects output packaging only
- `output_scores=True` — causes token scores to be returned; does not touch MTP
- `output_hidden_states=True` — causes backbone hidden states to be included in the `GenerateOutput` dict, but does **not** trigger the MTP head; the MTP head requires both `labels` and training mode, neither of which is satisfied

**Result:** `model.generate()` never invokes the MTP head forward pass, regardless of the generation strategy in use — greedy decoding, multinomial sampling, or beam search.

---

## Section 3: Training vs. Eval Mode Gate

In PyTorch, calling `model.eval()` sets `self.training = False` on the model and all sub-modules recursively, including `model.future_prediction[0]`. This gate is a deliberate design choice: the MTP head is an auxiliary training objective that improves backbone representations during training but is not part of the inference computation graph.

Calling `model.train()` before generation would technically satisfy the `self.training` gate, but this is not standard usage and carries significant side effects: dropout layers activate (RMSNorm/LayerNorm are unaffected by train/eval mode), and generation quality degrades due to stochastic dropout in residual connections. This is not a viable inference path.

---

> **Key Finding:** The MTP head is a training-only auxiliary module. `model.generate()` calls `forward()` in eval mode without `labels`, so the MTP head forward pass is never executed during standard autoregressive generation. The MTP weights are present in the model but are inactive at inference time.

---
**Next:** [`mtp_weight_loading_behavior.md`](./mtp_weight_loading_behavior.md)
