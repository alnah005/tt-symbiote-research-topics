# MTP Head Architecture

## What MTP Is

Multi-Token Prediction (MTP) is an auxiliary training objective that teaches the model to predict tokens at positions t+2, t+3, ... beyond the standard next-token prediction at t+1. Instead of only learning to predict the immediately following token, the model is trained to simultaneously predict multiple future tokens, which can improve representation quality and downstream task performance.

`mtp_num_hidden_layers: 1` in the Qwen3.6 config means one additional transformer block is trained to make these extended predictions. This is the sole structural config difference between Qwen3.5-35B-A3B and Qwen3.6-35B-A3B (see `../ch1_config_diff/new_and_modified_fields.md`).

## Module Structure in Qwen3_5MoeForConditionalGeneration

The MTP head is a submodule `model.future_prediction` on the top-level `Qwen3_5MoeForConditionalGeneration` object. It is a list of `nn.Module` blocks, one per `mtp_num_hidden_layers`. With `mtp_num_hidden_layers: 1`, the list has a single entry:

- `model.future_prediction[0]` — the first (and only) MTP transformer block

`model.future_prediction[0]` contains standard transformer block components:

- Attention projection weights (Q, K, V, O)
- A dense FFN (gate, up, down projections)
- Layer norms (pre-block RMS norm, post-attention norm)

The FFN inside the MTP block is **dense**, not MoE. This distinguishes it from the backbone's MoE FFN layers, which use expert routing. The MTP block shares the backbone's hidden dimension H=4096 and attention configuration (`num_key_value_heads = 8`, `head_dim = 128`).

## When forward() Is Called

The MTP block's `forward()` is called only inside `Qwen3_5MoeForConditionalGeneration.forward()` when two conditions are simultaneously true:

1. `labels is not None` — ground-truth token labels are provided (training scenario)
2. `self.training is True` — the model is in training mode

This gate is a plain Python `if` conditional. It is never part of the default inference code path.

Calling `model.eval()` sets `self.training = False` on the entire module tree. This permanently silences the MTP gate for that model object — subsequent `forward()` calls will never enter the MTP branch, regardless of what `labels` is set to.

## `model.generate()` and MTP

`GenerationMixin.generate()` calls `model.forward()` with `labels=None`. Generation does not supply ground-truth labels; the model autoregressively samples from its own output distribution. As a result:

- The first MTP gate condition (`labels is not None`) is `False` throughout generation
- The MTP block is never invoked
- MTP is **inference-inactive** in standard HuggingFace usage

This holds regardless of whether `model.eval()` has been called explicitly — `labels=None` alone is sufficient to suppress the MTP forward pass.

## Weight Keys Introduced by the MTP Head

The following keys appear in the Qwen3.6-35B-A3B checkpoint under the `model.future_prediction[0].*` prefix and are absent from any Qwen3.5-35B-A3B checkpoint:

| Key pattern | Description |
|---|---|
| `model.future_prediction[0].norm.weight` | Pre-block RMS norm |
| `model.future_prediction[0].self_attn.q_proj.weight` | Q projection |
| `model.future_prediction[0].self_attn.k_proj.weight` | K projection |
| `model.future_prediction[0].self_attn.v_proj.weight` | V projection |
| `model.future_prediction[0].self_attn.o_proj.weight` | Output projection |
| `model.future_prediction[0].mlp.gate_proj.weight` | FFN gate |
| `model.future_prediction[0].mlp.up_proj.weight` | FFN up |
| `model.future_prediction[0].mlp.down_proj.weight` | FFN down |
| `model.future_prediction[0].post_attention_layernorm.weight` | Post-attention norm |

These 9 key patterns account for the full ~160M parameter, ~304.6 MiB BF16 MTP head contribution to the checkpoint size.

## `lm_head` Sharing

The MTP block produces auxiliary logits by projecting its hidden states through the same `lm_head` (language model head) used by the backbone. The `lm_head` weight is **not duplicated** in the checkpoint.

The `lm_head` is tied to the embedding table: `lm_head.weight` is the same tensor as `model.embed_tokens.weight`, with `vocab_size = 151,936` (identical between Qwen3.5 and Qwen3.6 — see `../ch1_config_diff/new_and_modified_fields.md`). This weight is loaded once as part of the backbone. The MTP keys listed in the table above do not include any `lm_head` entry. At runtime, the MTP block references the already-loaded backbone `lm_head` directly.

For TT-Symbiote purposes: the `lm_head` / `model.embed_tokens.weight` tie is unchanged from Qwen3.5. No special handling is needed.

## Training vs. Qwen3.5 Comparison

Qwen3.5-35B-A3B has no `mtp_num_hidden_layers` field in its config and no `model.future_prediction` submodule. The full set of MTP weight keys (all 9 patterns above) is absent from any Qwen3.5 checkpoint.

The MTP keys are entirely **additive** in Qwen3.6. They do not replace, rename, or modify any existing Qwen3.5 weight keys. All backbone weight shapes are identical between the two models (established in `../ch2_weight_shapes/`). The MTP head is a net-new addition sitting alongside the unchanged backbone, not a modification of it.
