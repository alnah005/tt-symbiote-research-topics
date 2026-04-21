# Step-by-Step Migration Guide

## Overview

This guide transitions a TT-Symbiote deployment from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B. The backbone TTNN modules (`TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, `TTNNQwen3MoE`, and related helpers) require no changes. The seven steps below cover checkpoint loading, config handling, weight preprocessing, generation loop safety, and validation.

---

## Step 1 — Update the Checkpoint Path

Update the model identifier passed to `AutoModelForCausalLM.from_pretrained`:

```python
# Before:
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-35B-A3B", ...)
# After:
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", ...)
```

No other changes are needed to weight loading initialization. All backbone weight keys are present with identical names and shapes in both checkpoints (see `../ch2_weight_shapes/`).

---

## Step 2 — Add `partial_rotary_factor` Defensive Fallback

Any TT-Symbiote code that reads `partial_rotary_factor` from the model config should use the following defensive pattern to maintain backward compatibility with both Qwen3.5 and Qwen3.6:

```python
# Defensive fallback — works for both Qwen3.5 and Qwen3.6:
partial_rotary_factor = (
    getattr(config, "partial_rotary_factor", None)
    or (getattr(config, "rope_parameters", None) or {}).get("partial_rotary_factor", 1.0)
)
rotary_dim = int(config.head_dim * partial_rotary_factor)
# Result: rotary_dim = int(128 * 0.25) = 32 for both Qwen3.5 and Qwen3.6
```

If existing code already uses this pattern or reads the value via `AutoConfig` indirection, no change is needed. If it reads `config.partial_rotary_factor` directly as a top-level attribute only, add the fallback — the attribute exists in Qwen3.6 but not Qwen3.5, and omitting the fallback will cause `AttributeError` when loading Qwen3.5 checkpoints (see `../ch3_partial_rotary_factor/`).

---

## Step 3 — Filter MTP Weight Keys Before TTNN Preprocessing

Insert the following filter between safetensors loading and TT-Symbiote's weight preprocessing pipeline. The `model.future_prediction[0].*` keys added by Qwen3.6 are not consumed by any TTNN module and must be removed before preprocessing to avoid unrecognized-key errors (see `../ch5_mtp_head/loading_recipe.md`):

```python
def filter_mtp_keys(state_dict: dict) -> dict:
    return {k: v for k, v in state_dict.items()
            if not k.startswith("model.future_prediction")}

# Usage:
raw_state_dict = load_safetensors("path/to/qwen3.6-35b-a3b/")
filtered = filter_mtp_keys(raw_state_dict)
tt_symbiote_model.load_weights(filtered)
```

This filter removes approximately 304.6 MiB of training-only weights. The MTP head is never invoked during inference (`model.generate()` does not reach it due to the `labels is not None AND self.training is True` gate).

---

## Step 4 — Suppress `bos_token_id` Auto-Prepend

After loading the model, immediately clear `bos_token_id` from the generation config to prevent out-of-range embedding lookups. The value `248044` in Qwen3.6's config exceeds `vocab_size = 151,936` and will produce silent garbage values if used as an embedding index on the TTNN device (see `../ch4_bos_token_id/`):

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B", ...)
model.generation_config.bos_token_id = None  # suppress out-of-range auto-prepend
```

Additionally, audit TT-Symbiote's generation loop init for any direct read of `config.bos_token_id`:

```bash
grep -r "bos_token_id" tt_symbiote/models/qwen/
```

If found, remove or gate the read so it does not construct a token index from `config.bos_token_id`. The generation loop should always receive pre-formed `input_ids` from the tokenizer rather than constructing them from config fields.

---

## Step 5 — Add Embedding Bounds Check in CI

Add a pre-flight assertion before the first TTNN forward pass in the test suite and in any CI entry point:

```python
assert input_ids.max().item() < model.config.vocab_size, (
    f"input_ids contain out-of-range token ID {input_ids.max().item()}; "
    f"vocab_size={model.config.vocab_size}"
)
```

This catches any out-of-range token ID on CPU before it reaches the TTNN device, where the failure is silent. The assertion adds negligible overhead and protects against the `bos_token_id` failure path as well as any future token ID range regressions.

---

## Step 6 — Run PCC Validation Suite

Run the existing PCC test suite (`test_pcc.py` / `test_a3b_pcc.py`) against the Qwen3.6 checkpoint:

```bash
pytest tests/models/qwen/test_pcc.py --checkpoint Qwen/Qwen3.6-35B-A3B
pytest tests/models/qwen/test_a3b_pcc.py --checkpoint Qwen/Qwen3.6-35B-A3B
```

Expected PCC thresholds are identical to Qwen3.5 (≥ 0.99 per layer). The architecture is unchanged between the two checkpoints — only weight values differ. Any PCC regression below 0.99 indicates a loading or preprocessing error, not an architectural incompatibility.

---

## Step 7 — End-to-End Generation Smoke Test

Run `demo_a3b.py` with a known prompt and compare output token sequence and throughput against the Qwen3.5 baseline:

```bash
python demo_a3b.py --checkpoint Qwen/Qwen3.6-35B-A3B --prompt "Explain mixture-of-experts routing."
```

Expected results:

- **Throughput:** similar to the Qwen3.5 baseline (86 ms/token target); deviations indicate a kernel or scheduling regression unrelated to this migration.
- **Token sequences:** different from Qwen3.5 output. This is expected and not a regression — Qwen3.5 and Qwen3.6 have different trained weights and will produce different outputs for the same prompt. Do not compare output text between the two checkpoints; compare only throughput and absence of errors.

---

## Summary Checklist

- [ ] Step 1: Checkpoint path updated to `Qwen/Qwen3.6-35B-A3B`
- [ ] Step 2: `partial_rotary_factor` defensive fallback added or verified
- [ ] Step 3: MTP key filter inserted in weight preprocessing pipeline
- [ ] Step 4: `model.generation_config.bos_token_id = None` set; generation loop audited
- [ ] Step 5: Embedding bounds check assertion added to CI
- [ ] Step 6: PCC test suite passes with Qwen3.6 weights
- [ ] Step 7: End-to-end smoke test passes
