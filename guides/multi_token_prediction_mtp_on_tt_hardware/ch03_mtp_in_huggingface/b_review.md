## B Feedback — Pass 1

1. **File:** `huggingface_mtp_forward_pass.md`, Section 3 (~line 40)
   **Error:** Section 3 states the MTP head "is guarded by the `labels is not None` check, which in practice is only satisfied during training." This reduces the two-part gate to a single condition. Section 1 of the same file correctly states both `labels is not None` AND `self.training is True` must hold simultaneously. A reader of Section 3 in isolation could conclude that passing `labels` in eval mode would activate the MTP head — it would not, because `self.training` is also required.
   **Fix:** Change the sentence to: "The MTP head is guarded by two simultaneous conditions: `labels is not None` and `self.training is True`. In practice, both are only satisfied during training."

2. **File:** `mtp_inference_activation_scenarios.md`, Section 1 table (~line 6)
   **Error:** The first row's "MTP head active?" cell reads "**No** — gated by `self.training = False`". This states only one of the two gate conditions and frames it in a way that implies `self.training = False` is the sole control. A reader building a custom forward call could pass `labels` in eval mode, see that MTP is still inactive, and form an incorrect mental model of which condition is actually controlling the behavior.
   **Fix:** Change the cell to: "**No** — both gate conditions unmet (`labels=None` and `self.training=False`)".

3. **File:** `mtp_weight_loading_behavior.md`, Section 2 table (~line 50)
   **Error:** The table lists `model.layers.*` as "40 layers." The ground truth for this chapter series specifies Qwen3.6-35B-A3B, whose architecture has 94 decoder layers (not 40). 40 is the layer count for the smaller Qwen3.5-MoE variant. A TT-Symbiote engineer filtering or mapping weight keys using this number as a reference will be off.
   **Fix:** Update the row description to `model.layers.*` (94 layers) to match Qwen3.6-35B-A3B's actual backbone depth, or remove the parenthetical count if it cannot be confirmed from the source model config.

4. **File:** `mtp_weight_loading_behavior.md`, Scenario A (~lines 17-19)
   **Error:** The warning text shown — "Some weights of the model checkpoint at <path> were not used when initializing" — is the correct HuggingFace message for checkpoint keys absent from the model, and is accurate. However, the ground truth specifies the warning is about "unexpected keys." These two HuggingFace warnings are distinct: "unexpected keys" refers to keys in the `state_dict` that have no matching parameter in the model, which is exactly the MTP scenario. The displayed message text is correct, but the introductory prose on line 15 says "HuggingFace emits a WARNING (not an error)" without clarifying this is specifically an `unexpected_keys` warning (as opposed to a `missing_keys` warning). An implementer checking logs for the wrong warning class will miss the signal.
   **Fix:** Add one sentence after the warning block: "This appears under the `unexpected_keys` category in HuggingFace's `_load_state_dict_into_model` output, distinct from `missing_keys` warnings which indicate backbone weights not found in the checkpoint."

5. **File:** `mtp_inference_activation_scenarios.md`, Section 3, step 2 (~line 28)
   **Error:** The manual speculative decoding step calls `model.future_prediction[0].forward(hidden_states, shifted_embedding)` and describes the output as "draft logits for the next-next position." With `mtp_num_hidden_layers = 1`, the single MTP head predicts position `t+2` given the prefix through position `t`. This is correct. However, step 1 says to call `forward()` with `output_hidden_states=True` "to obtain the backbone's final hidden states for the current prefix." The ground truth confirms `output_hidden_states=True` returns backbone hidden states without triggering the MTP head — but it does not return the hidden state at the position needed by the MTP head (the final hidden state after the last generated token). The hidden states tensor returned under `output_hidden_states=True` is a tuple of per-layer states, not a single final hidden state. Passing the raw `output.hidden_states` directly to the MTP head without extracting the last-layer, last-token slice will silently produce incorrect draft logits — the shape will be wrong and no error will be raised.
   **Fix:** Change step 2 to: "Extract the last-layer final-token hidden state: `h = output.hidden_states[-1][:, -1, :]`. Pass this to the MTP head: `model.future_prediction[0].forward(h, shifted_embedding)` to produce draft logits for position `t+2`."

## B Feedback Application Log — Pass 1

- Fix 1: Updated Section 3 of `huggingface_mtp_forward_pass.md` to state both gate conditions (labels is not None AND self.training is True)
- Fix 2: Updated summary table in `mtp_inference_activation_scenarios.md` to name both gate conditions
- Fix 3: Changed "40 layers" to "94 layers" in decision table in `mtp_weight_loading_behavior.md`
- Fix 4: Added clarification that MTP keys appear in `unexpected_keys` (not `missing_keys`) list in `mtp_weight_loading_behavior.md` Scenario A
- Fix 5: Updated step 2 in manual speculative decoding recipe in `mtp_inference_activation_scenarios.md` to extract correct hidden state slice `output.hidden_states[-1][:, -1:, :]`

## B Feedback — Pass 2

**No feedback — chapter approved.**

## B Feedback — Pass 3

1. **`mtp_inference_activation_scenarios.md`, Section 1 Summary Table** — "gated by `self.training = False` AND `labels=None`" inverts the gate logic. The MTP head fires when BOTH positive conditions are true (`labels is not None` AND `self.training is True`); either one being false is sufficient to suppress it. The table description conflates the suppression conditions and misleads readers reasoning about edge cases (e.g., eval mode with labels passed).

2. **`mtp_weight_loading_behavior.md`, Scenario C** — Describes `from_pretrained` called with `strict=True` raising `RuntimeError`. HuggingFace's `from_pretrained` does not expose a `strict=True` parameter; it always uses its own key-reconciliation logic (emitting warnings, not exceptions). The `strict=True` raising behavior belongs to PyTorch's native `module.load_state_dict(state_dict, strict=True)` — a different API.

3. **`huggingface_mtp_forward_pass.md`, Section 3** — "layer norm statistics shift to batch-dependent behavior" is false for RMSNorm/LayerNorm, which normalize over the feature dimension and are unaffected by `model.train()`. Only BatchNorm has batch-dependent statistics. Qwen uses RMSNorm throughout.

## B Feedback Application Log — Pass 3

- Fix 1: Changed Summary Table first row to "MTP gate requires both `labels is not None` AND `self.training is True`; generate() satisfies neither".
- Fix 2: Updated Scenario C to describe `module.load_state_dict(state_dict, strict=True)` (PyTorch native API), not `from_pretrained`. Added note that `from_pretrained` always uses warning-based key reconciliation.
- Fix 3: Changed to "dropout layers activate (RMSNorm/LayerNorm are unaffected by train/eval mode), and generation quality degrades due to stochastic dropout in residual connections."

## B Feedback — Pass 4

1. **`huggingface_mtp_forward_pass.md`, Section 1 heading** — `Qwen3_5MoeForConditionalGeneration` flagged as seq2seq naming convention. **Not applied** — this IS the correct HuggingFace class name for the Qwen3.5 MoE model; it uses this naming convention for historical reasons.

2. **`huggingface_mtp_forward_pass.md`, Section 3** — RMSNorm/LayerNorm framing said to be misleading without mentioning BatchNorm. **Not applied** — the parenthetical "(RMSNorm/LayerNorm are unaffected by train/eval mode)" is factually correct and model-specific; mentioning BatchNorm would add irrelevant information for a Qwen model.

3. **`mtp_weight_loading_behavior.md`, Scenario C** — "The solution in both cases is to pre-filter the state dict before loading" implies `from_pretrained` also fails without pre-filtering, which is false.

4. **`mtp_inference_activation_scenarios.md`, Section 3, Step 2** — `model.future_prediction[0].forward(hidden_state, shifted_embedding)` calls `.forward()` directly, bypassing all registered PyTorch hooks (including the verification hook defined in the same document).

5. **`mtp_inference_activation_scenarios.md`, Section 3, Step 2** — Two-argument call signature `(hidden_state, shifted_embedding)` presented as established fact with no citation; actual signature not defined in this chapter or prerequisites.

## B Feedback Application Log — Pass 4

- Fix 1: Not applied — class name is correct.
- Fix 2: Not applied — parenthetical is correct.
- Fix 3: Fixed "in both cases" → clarified pre-filtering is required only for `load_state_dict(strict=True)`; for `from_pretrained` it is recommended but not required for correctness.
- Fix 4: Changed `.forward(hidden_state, shifted_embedding)` to `(hidden_state, shifted_embedding)` (standard `__call__` invocation that respects registered hooks).
- Fix 5: Added "call signature per Chapter 1 architecture" qualifier to the MTP head call.

## B Feedback — Pass 5

1. **`mtp_weight_loading_behavior.md`, Section 3** — `register_forward_hook` fires only through `__call__`; direct `.forward()` invocations bypass hooks entirely. The hook-based negative proof is not sound — if any code path called `.forward()` directly, the hook would be bypassed and `mtp_called` would remain empty, producing a false negative.

2. **`mtp_inference_activation_scenarios.md`, Section 3, Step 1** — Step 1 instructs "Call `model.forward()` with `output_hidden_states=True`", calling `.forward()` directly and bypassing all registered hooks. This directly contradicts the hook-based verification strategy in `mtp_weight_loading_behavior.md` Section 3.

## B Feedback Application Log — Pass 5

- Fix 1: Added caveat to Section 3 of `mtp_weight_loading_behavior.md` that `register_forward_hook` only catches `__call__`-path invocations; noted that this verification is sound for `model.generate()` which dispatches through `__call__` at every level.
- Fix 2: Changed `model.forward()` → `model()` in Step 1 of Section 3 in `mtp_inference_activation_scenarios.md` to use standard `__call__` semantics.
