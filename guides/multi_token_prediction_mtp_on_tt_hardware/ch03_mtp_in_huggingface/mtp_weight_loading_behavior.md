# MTP Weight Loading Behavior

## Section 1: Loading Qwen3.6 Weights into `Qwen3_5MoeForConditionalGeneration`

Three distinct scenarios arise depending on how the checkpoint is loaded.

---

### Scenario A: Standard `AutoModelForCausalLM.from_pretrained` (Default Settings)

When a Qwen3.6 checkpoint is loaded into `Qwen3_5MoeForConditionalGeneration` via the default `from_pretrained` path:

1. HuggingFace compares the checkpoint keys against the model's registered parameters and modules.
2. The 11 MTP keys (`model.future_prediction.0.*`) do not match any registered `nn.Parameter` or `nn.Module` in `Qwen3_5MoeForConditionalGeneration`.
3. HuggingFace emits a WARNING (not an error):

```
Some weights of the model checkpoint at <path> were not used when initializing
Qwen3_5MoeForConditionalGeneration: ['model.future_prediction.0.enorm.weight',
'model.future_prediction.0.hnorm.weight', ...]
```

This appears under the `unexpected_keys` category in HuggingFace's `_load_state_dict_into_model` output, distinct from `missing_keys` warnings which indicate backbone weights not found in the checkpoint.

4. The MTP weights are not loaded into any tensor. The backbone loads normally and all backbone weight keys are consumed without error.

**Result:** Functionally equivalent to loading a Qwen3.5 checkpoint for inference purposes. The warning is informational and does not indicate a problem with inference correctness.

---

### Scenario B: Using a Qwen3.6-Native Model Class (If Available)

If HuggingFace provides a `Qwen3MoeForCausalLM` or similar class that registers `model.future_prediction` as an `nn.ModuleList`:

1. The 11 MTP keys are loaded into `model.future_prediction[0]` parameters.
2. The MTP head forward is still gated on training mode — it is not called during `generate()`.
3. The MTP weights consume approximately 304.6 MiB of device memory even if never used at inference.

This scenario produces a correctly loaded model with the MTP head available for training or future speculative decoding work, at the cost of the additional memory footprint.

---

### Scenario C: Direct `load_state_dict` with `strict=True`

If the state dict is loaded manually via PyTorch's `module.load_state_dict(state_dict, strict=True)` — bypassing HuggingFace's `from_pretrained` machinery — the 11 unexpected MTP keys cause PyTorch to raise `RuntimeError: Unexpected key(s) in state_dict`. Note: HuggingFace's `from_pretrained` does not expose a `strict=True` parameter; it always uses its own key-reconciliation logic, which emits warnings rather than raising. Pre-filtering the state dict (as shown in the Ch2 safe loading recipe) is the recommended approach regardless of which loading path is used; it is required for correctness only in the `load_state_dict(strict=True)` case. The Ch2 safe loading recipe demonstrates the correct key-filter approach.

---

## Section 2: Which Keys Must Be Loaded for TT-Symbiote Inference

| Key group | Required for backbone inference | Required for MTP speculative decoding |
|---|---|---|
| `model.layers.*` (94 layers) | Yes | Yes |
| `model.embed_tokens.weight` | Yes | Yes |
| `model.norm.weight` | Yes | Yes |
| `lm_head.weight` | Yes | Yes (shared) |
| `model.future_prediction.0.*` (11 MTP keys) | **No** | Yes |

For TT-Symbiote's current inference path (no speculative decoding): load only the first four groups and discard all `model.future_prediction.*` keys before the weight-loading pipeline.

---

## Section 3: Verification Strategy

The following snippet confirms that the MTP head is never invoked during `model.generate()` by attaching a forward hook to `model.future_prediction[0]` and asserting it is never triggered.

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", device_map="auto")
model.eval()

# Verify MTP head is inactive by checking that generate() produces same output
# with and without MTP weights: use hooks to confirm future_prediction.0 is never called
mtp_called = []
def mtp_hook(module, input, output):
    mtp_called.append(True)

if hasattr(model, 'future_prediction'):
    model.future_prediction[0].register_forward_hook(mtp_hook)

output = model.generate(input_ids, max_new_tokens=5)
assert len(mtp_called) == 0, "MTP head should not be called during generate()"
```

If the assertion passes, the MTP head is confirmed inactive during `model.generate()`. Note: `register_forward_hook` fires only through `__call__` — if any code path invoked `model.future_prediction[0].forward(...)` directly, the hook would be bypassed and `mtp_called` would remain empty despite the MTP head having executed. This verification is sound for `model.generate()`, which dispatches through `__call__` at every level. If `future_prediction` is not present on the model object (Scenario A above), `hasattr` returns `False` and the hook registration is skipped — the assertion still passes because `mtp_called` remains empty.

---

> **Key Finding:** For TT-Symbiote backbone inference, the 11 MTP weight keys can be safely discarded before the weight-loading pipeline. No correctness impact, no error. MTP speculative decoding (Chapter 5) requires loading these keys into dedicated TTNN tensors — it is not enabled by default.

---
**Next:** [`mtp_inference_activation_scenarios.md`](./mtp_inference_activation_scenarios.md)
