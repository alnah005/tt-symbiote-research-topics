# Extra Weight Keys in Qwen3.6

## The 11 MTP Head Weight Keys

Qwen3.6-35B-A3B introduces a Multi-Token Prediction (MTP) head under the key prefix `model.future_prediction.0.*`. All 11 keys are absent from the Qwen3.5-35B-A3B checkpoint. The MTP head reuses the full `hidden_size` and `head_dim` of the backbone, and its FFN is **dense** (not MoE) with `intermediate_size = 14336`.

> **Caveat:** The key prefix `model.future_prediction.0.*` matches the pattern observed in DeepSeek-V3 and related implementations that use a single MTP step (the trailing `0` is the step index). The actual checkpoint should be verified against `state_dict.keys()` before writing TTNN weight-loading code, as the exact prefix or number of MTP steps may differ from what is documented here.

| Key name (relative to `model.future_prediction.0.`) | Full key | Shape | Dimension derivation |
|---|---|---|---|
| `enorm.weight` | `model.future_prediction.0.enorm.weight` | `[7168]` | `hidden_size` = 7168 (new MTP-specific norm) |
| `hnorm.weight` | `model.future_prediction.0.hnorm.weight` | `[7168]` | `hidden_size` = 7168 (new MTP-specific norm) |
| `self_attn.q_proj.weight` | `model.future_prediction.0.self_attn.q_proj.weight` | `[8192, 7168]` | same as backbone `self_attn.q_proj` |
| `self_attn.k_proj.weight` | `model.future_prediction.0.self_attn.k_proj.weight` | `[1024, 7168]` | same as backbone `self_attn.k_proj` |
| `self_attn.v_proj.weight` | `model.future_prediction.0.self_attn.v_proj.weight` | `[1024, 7168]` | same as backbone `self_attn.v_proj` |
| `self_attn.o_proj.weight` | `model.future_prediction.0.self_attn.o_proj.weight` | `[7168, 8192]` | same as backbone `self_attn.o_proj` |
| `mlp.gate_proj.weight` | `model.future_prediction.0.mlp.gate_proj.weight` | `[14336, 7168]` | same as backbone dense `mlp.gate_proj` |
| `mlp.up_proj.weight` | `model.future_prediction.0.mlp.up_proj.weight` | `[14336, 7168]` | same as backbone dense `mlp.up_proj` |
| `mlp.down_proj.weight` | `model.future_prediction.0.mlp.down_proj.weight` | `[7168, 14336]` | same as backbone dense `mlp.down_proj` |
| `input_layernorm.weight` | `model.future_prediction.0.input_layernorm.weight` | `[7168]` | same as backbone `input_layernorm` |
| `post_attention_layernorm.weight` | `model.future_prediction.0.post_attention_layernorm.weight` | `[7168]` | same as backbone `post_attention_layernorm` |

The MTP head FFN uses `intermediate_size = 14336` (same as the dense backbone FFN layers), **not** `moe_intermediate_size = 2048`. There are no routed experts in the MTP head.

---

## MTP Head Parameter Count

```math
params =
  (q_proj)      2 × 8192 × 7168
+ (k_proj)      2 × 1024 × 7168
+ (o_proj)      (included in first term via symmetry — see below)
+ (FFN)         3 × 14336 × 7168
+ (layer norms) 4 × 7168

= 2 × (8192 × 7168)          [q_proj + o_proj, both [8192, 7168] or [7168, 8192]]
+ 2 × (1024 × 7168)           [k_proj + v_proj]
+ 3 × (14336 × 7168)          [gate_proj + up_proj + down_proj]
+ 4 × 7168                    [enorm, hnorm, input_layernorm, post_attention_layernorm]

= 117,440,512
+  14,680,064
+ 308,281,344
+      28,672

= 440,430,592  ≈ 440M parameters
```

This is approximately 1.3% of the total Qwen3.6-35B-A3B parameter count and does not affect the backbone's inference-time compute graph when MTP is disabled.

---

## How `from_pretrained` Handles Unexpected Keys

When loading a Qwen3.6 checkpoint into a `Qwen3_5MoeForConditionalGeneration` model instance using `AutoModelForCausalLM.from_pretrained(...)`, HuggingFace Transformers encounters the 11 MTP keys and has no registered `nn.Parameter` or `nn.Module` to map them to.

Default behavior (effective `strict=False` for top-level loading):

- HuggingFace's `from_pretrained` does **not** raise an error for unexpected keys. The missing-key and unexpected-key lists are collected and emitted as log messages at `WARNING` level.
- The log output will include a line such as:

```
Some weights of the model checkpoint at <path> were not used when initializing
Qwen3_5MoeForConditionalGeneration: ['model.future_prediction.0.enorm.weight', ...]
```

- None of the 11 MTP weights are loaded into any tensor in the instantiated model. They are simply ignored.
- The backbone weights all load normally; the final model is functionally identical to a model loaded from a Qwen3.5 checkpoint (subject to the actual weight values, which differ because Qwen3.6 was trained independently).

If `from_pretrained` is called with an explicit `state_dict` argument and `strict=True`, the unexpected keys will raise a `RuntimeError`. The safe pattern is to pre-filter the state dict (see Section 6 below).

---

## Impact on TT-Symbiote Weight Loading

TT-Symbiote weight loading operates on a key-filtered state dict rather than routing through HuggingFace's `from_pretrained` machinery. The relevant behavior is:

1. **Key pattern matching:** TT-Symbiote weight preprocessing iterates over checkpoint keys and maps each key to a TTNN device tensor via a lookup table or regex filter. The prefix `model.future_prediction.*` does not match any key pattern registered for `Qwen3_5MoeForConditionalGeneration` backbone modules (attention, MoE FFN, dense FFN, embeddings, lm_head, layer norms).

2. **Pass-through without loading:** Keys that do not match the filter are passed over. No device memory is allocated for them, no host-to-device transfer occurs, and no TTNN tensor is created. The net effect is identical to the keys not existing in the checkpoint.

3. **No action required:** Unless MTP inference is explicitly added to TT-Symbiote (a separate engineering effort), the 11 MTP keys require no handling change. Existing weight-loading code will silently ignore them.

4. **Verification recommendation:** After loading, assert that the set of loaded TTNN weight keys equals the expected backbone key set (see the pseudocode in `shared_weight_shapes.md`). This catches both unexpected inclusions (MTP keys being loaded erroneously) and unexpected omissions (backbone keys failing to load).

---

## `bos_token_id = 248044` — Out-of-Range Embedding Index

Chapter 1 noted that Qwen3.6's `config.json` sets `bos_token_id = 248044`. This value lies **outside** the embedding table range `[0, 151935]` (since `vocab_size = 151936`).

Implications for weight loading and inference:

- `model.embed_tokens.weight` has shape `[151936, 7168]`. Valid row indices are 0 through 151935. An embedding lookup for token ID 248044 is a bounds violation that will raise an `IndexError` in PyTorch or produce undefined behavior in a TTNN kernel that does not perform bounds checking.
- `lm_head.weight` has shape `[151936, 7168]`. The output logit space covers only tokens 0–151935; token 248044 has no logit.
- The tokenizer vocabulary (in `tokenizer.json` or `vocab.json`) governs which token IDs are actually produced by `tokenizer.encode(...)`. The token with ID 248044, if any, is not in the embedding table and cannot be a generation output from this model. It is not safe to assume what this token ID represents without inspecting the vocabulary file.
- **Safe recipe:** Always pass pre-tokenized `input_ids` produced by the matching tokenizer to the model. Validate that no element of `input_ids` exceeds `vocab_size - 1 = 151935` before submitting to the device. In particular, if any code path inserts a BOS token by referencing `config.bos_token_id`, replace that reference with the actual tokenizer's BOS token ID (which the tokenizer resolves correctly from its own vocabulary).

```python
# Safe BOS token validation example
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
# Use the tokenizer's own bos_token_id, not config.bos_token_id
safe_bos_id = tokenizer.bos_token_id
assert safe_bos_id is not None and safe_bos_id < 151936, (
    f"Tokenizer BOS token ID {safe_bos_id} is out of range for vocab_size=151936"
)
```

---

## Safe Loading Recipe

The following key filter predicate excludes all MTP head keys from the state dict before TTNN weight preprocessing. Apply it immediately after loading the raw checkpoint and before any shape transformations or device transfers.

```python
import re
from safetensors import safe_open
from pathlib import Path

MTP_KEY_PATTERN = re.compile(r"^model\.future_prediction\.")


def load_backbone_state_dict(checkpoint_dir: str) -> dict[str, object]:
    """
    Load all safetensors shards and return only backbone weight tensors.
    MTP head keys (model.future_prediction.*) are excluded.
    """
    state_dict = {}
    excluded_keys = []

    for shard_path in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                if MTP_KEY_PATTERN.match(key):
                    excluded_keys.append(key)
                else:
                    state_dict[key] = f.get_tensor(key)

    if excluded_keys:
        print(
            f"INFO: Excluded {len(excluded_keys)} MTP head key(s) from state dict "
            f"(model.future_prediction.* prefix). These are Qwen3.6-only and are "
            f"not loaded into any TTNN module."
        )

    return state_dict


# Usage
state_dict = load_backbone_state_dict("/path/to/qwen36_checkpoint")
# Pass state_dict to TTNN weight preprocessing pipeline
```

This recipe is additive — it works correctly with Qwen3.5 checkpoints too (where no MTP keys are present, so the filter excludes nothing). There is no need to branch on which checkpoint version is being loaded.

---

**Next:** [Chapter 3 — `partial_rotary_factor` Promotion and RoPE Resolution](../ch3_partial_rotary_factor/index.md)
