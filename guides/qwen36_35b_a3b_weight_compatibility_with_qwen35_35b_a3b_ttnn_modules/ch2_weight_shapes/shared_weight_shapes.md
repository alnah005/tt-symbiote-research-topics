# Shared Backbone Weight Shapes

## Purpose

Asserting that two models are "architecturally identical" based on config values is necessary but not sufficient for writing weight-loading code. The actual guarantee needed is: for every weight key consumed by a TTNN module, the tensor shape in the Qwen3.6 checkpoint is identical to the shape in the Qwen3.5 checkpoint. This file enumerates every logical weight group in the shared backbone, states each shape, and shows which hyperparameter governs it. The identity confirmation table in Section 7 then makes the guarantee explicit.

> **Caveat:** The exact value of `decoder_sparse_step` (which determines which layer indices are MoE layers versus dense layers) should be verified directly from the actual `config.json` of each checkpoint before writing layer-dispatch code. This file documents the shape patterns for both layer types; the hyperparameters that govern each shape are identical across both checkpoints regardless of which layers instantiate which type.

---

## Attention Projection Weights

> **Note:** The q_proj, k_proj, v_proj, o_proj, q_norm, and k_norm weights shown here correspond to the **full-attention layers (indices 30–39)** only, handled by `TTNNQwen3FullAttention`. The linear attention layers (indices 0–29), handled by `TTNNQwen3LinearAttention`, use a different attention mechanism. Both share the same `hidden_size` but differ in projection structure.

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `self_attn.q_proj.weight` | `[8192, 7168]` | `num_attention_heads × head_dim` = 64 × 128 = 8192 rows; `hidden_size` = 7168 cols |
| `self_attn.k_proj.weight` | `[1024, 7168]` | `num_key_value_heads × head_dim` = 8 × 128 = 1024 rows; `hidden_size` = 7168 cols |
| `self_attn.v_proj.weight` | `[1024, 7168]` | `num_key_value_heads × head_dim` = 8 × 128 = 1024 rows; `hidden_size` = 7168 cols |
| `self_attn.o_proj.weight` | `[7168, 8192]` | `hidden_size` = 7168 rows; `num_attention_heads × head_dim` = 8192 cols |
| `self_attn.q_norm.weight` | `[128]` | Per-head RMSNorm; length = `head_dim` = 128 |
| `self_attn.k_norm.weight` | `[128]` | Per-head RMSNorm; length = `head_dim` = 128 |

`q_norm` and `k_norm` are applied per attention head after projecting, before RoPE. They are scalar weight vectors of length `head_dim`, not full projection matrices.

---

## Layer Norm Weights

Each of the 40 transformer layers carries two RMSNorm weight vectors. Key paths follow `model.layers.<N>.<name>`.

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `input_layernorm.weight` | `[7168]` | `hidden_size` = 7168 |
| `post_attention_layernorm.weight` | `[7168]` | `hidden_size` = 7168 |

---

## MoE FFN Weights

For layers designated as MoE layers, the FFN is replaced by a mixture-of-experts block. Key paths follow `model.layers.<N>.mlp.<name>`.

### Routed experts

There are 128 routed experts per MoE layer (`num_experts = 128`). Each expert has three weight matrices. Key path pattern: `model.layers.<N>.mlp.experts.<E>.<name>` for expert index E in [0, 127].

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `experts.<E>.gate_proj.weight` | `[2048, 7168]` | `moe_intermediate_size` = 2048 rows; `hidden_size` = 7168 cols |
| `experts.<E>.up_proj.weight` | `[2048, 7168]` | `moe_intermediate_size` = 2048 rows; `hidden_size` = 7168 cols |
| `experts.<E>.down_proj.weight` | `[7168, 2048]` | `hidden_size` = 7168 rows; `moe_intermediate_size` = 2048 cols |

### MoE router

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `mlp.gate.weight` | `[128, 7168]` | `num_experts` = 128 rows; `hidden_size` = 7168 cols |

### Shared expert

Each MoE layer also contains one always-active shared expert. Key paths follow `model.layers.<N>.mlp.shared_expert.<name>`.

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `shared_expert.gate_proj.weight` | `[2048, 7168]` | `shared_expert_intermediate_size` = 2048 rows; `hidden_size` = 7168 cols |
| `shared_expert.up_proj.weight` | `[2048, 7168]` | `shared_expert_intermediate_size` = 2048 rows; `hidden_size` = 7168 cols |
| `shared_expert.down_proj.weight` | `[7168, 2048]` | `hidden_size` = 7168 rows; `shared_expert_intermediate_size` = 2048 cols |

### Shared expert gate scalar

If present in the checkpoint (verify against actual `state_dict.keys()`):

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `mlp.shared_expert_gate.weight` | `[1, 7168]` | Scalar gate broadcast across `hidden_size` = 7168 |

---

## Dense FFN Weights

For layers designated as dense layers, the FFN follows a standard gated-linear-unit layout. Key paths follow `model.layers.<N>.mlp.<name>`.

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `mlp.gate_proj.weight` | `[14336, 7168]` | `intermediate_size` = 14336 rows; `hidden_size` = 7168 cols |
| `mlp.up_proj.weight` | `[14336, 7168]` | `intermediate_size` = 14336 rows; `hidden_size` = 7168 cols |
| `mlp.down_proj.weight` | `[7168, 14336]` | `hidden_size` = 7168 rows; `intermediate_size` = 14336 cols |

Note: `intermediate_size = 14336` is approximately twice `hidden_size = 7168`, a conventional 2× ratio for dense FFN layers in this architecture family.

---

## Embeddings and Output Head

These weights are not per-layer; they are model-level tensors.

| Weight key | Shape | Dimension derivation |
|---|---|---|
| `model.embed_tokens.weight` | `[151936, 7168]` | `vocab_size` = 151936 rows; `hidden_size` = 7168 cols |
| `model.norm.weight` | `[7168]` | Final RMSNorm; length = `hidden_size` = 7168 |
| `lm_head.weight` | `[151936, 7168]` | `vocab_size` = 151936 rows; `hidden_size` = 7168 cols; separate tensor because `tie_word_embeddings = false` |

Because `tie_word_embeddings = false`, `lm_head.weight` and `model.embed_tokens.weight` are independent parameters that happen to share the same shape. TTNN weight loading must load both separately; it cannot alias one to the other.

---

## Shape Identity Confirmation

The following table makes the guarantee explicit. All shapes are governed solely by hyperparameters that are identical across both checkpoints (verified in Chapter 1).

| Component | Qwen3.5 shape | Qwen3.6 shape | Governed by |
|---|---|---|---|
| `self_attn.q_proj.weight` | `[8192, 7168]` | `[8192, 7168]` | `num_attention_heads`, `head_dim`, `hidden_size` |
| `self_attn.k_proj.weight` | `[1024, 7168]` | `[1024, 7168]` | `num_key_value_heads`, `head_dim`, `hidden_size` |
| `self_attn.v_proj.weight` | `[1024, 7168]` | `[1024, 7168]` | `num_key_value_heads`, `head_dim`, `hidden_size` |
| `self_attn.o_proj.weight` | `[7168, 8192]` | `[7168, 8192]` | `hidden_size`, `num_attention_heads`, `head_dim` |
| `self_attn.q_norm.weight` | `[128]` | `[128]` | `head_dim` |
| `self_attn.k_norm.weight` | `[128]` | `[128]` | `head_dim` |
| `input_layernorm.weight` | `[7168]` | `[7168]` | `hidden_size` |
| `post_attention_layernorm.weight` | `[7168]` | `[7168]` | `hidden_size` |
| `experts.<E>.gate_proj.weight` | `[2048, 7168]` | `[2048, 7168]` | `moe_intermediate_size`, `hidden_size` |
| `experts.<E>.up_proj.weight` | `[2048, 7168]` | `[2048, 7168]` | `moe_intermediate_size`, `hidden_size` |
| `experts.<E>.down_proj.weight` | `[7168, 2048]` | `[7168, 2048]` | `hidden_size`, `moe_intermediate_size` |
| `mlp.gate.weight` | `[128, 7168]` | `[128, 7168]` | `num_experts`, `hidden_size` |
| `shared_expert.gate_proj.weight` | `[2048, 7168]` | `[2048, 7168]` | `shared_expert_intermediate_size`, `hidden_size` |
| `shared_expert.up_proj.weight` | `[2048, 7168]` | `[2048, 7168]` | `shared_expert_intermediate_size`, `hidden_size` |
| `shared_expert.down_proj.weight` | `[7168, 2048]` | `[7168, 2048]` | `hidden_size`, `shared_expert_intermediate_size` |
| `mlp.gate_proj.weight` (dense) | `[14336, 7168]` | `[14336, 7168]` | `intermediate_size`, `hidden_size` |
| `mlp.up_proj.weight` (dense) | `[14336, 7168]` | `[14336, 7168]` | `intermediate_size`, `hidden_size` |
| `mlp.down_proj.weight` (dense) | `[7168, 14336]` | `[7168, 14336]` | `hidden_size`, `intermediate_size` |
| `model.embed_tokens.weight` | `[151936, 7168]` | `[151936, 7168]` | `vocab_size`, `hidden_size` |
| `model.norm.weight` | `[7168]` | `[7168]` | `hidden_size` |
| `lm_head.weight` | `[151936, 7168]` | `[151936, 7168]` | `vocab_size`, `hidden_size` |

---

## Verification Pseudocode

The following Python snippet loads the shape dictionaries from two checkpoints and asserts that every backbone key present in the Qwen3.5 checkpoint has an identical shape in the Qwen3.6 checkpoint.

```python
from safetensors import safe_open
from pathlib import Path

def collect_shapes(checkpoint_dir: str) -> dict[str, tuple[int, ...]]:
    """Return {key: shape} for all tensors across all safetensors shards."""
    shapes = {}
    for shard_path in sorted(Path(checkpoint_dir).glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def compare_backbone_shapes(qwen35_dir: str, qwen36_dir: str) -> None:
    shapes_35 = collect_shapes(qwen35_dir)
    shapes_36 = collect_shapes(qwen36_dir)

    # MTP keys are only in Qwen3.6; exclude them from the comparison
    mtp_prefix = "model.future_prediction."
    backbone_keys_35 = {k for k in shapes_35 if not k.startswith(mtp_prefix)}
    backbone_keys_36 = {k for k in shapes_36 if not k.startswith(mtp_prefix)}

    missing_in_36 = backbone_keys_35 - backbone_keys_36
    extra_in_36 = backbone_keys_36 - backbone_keys_35

    assert not missing_in_36, f"Keys in Qwen3.5 missing from Qwen3.6 backbone: {missing_in_36}"
    assert not extra_in_36, f"Unexpected extra backbone keys in Qwen3.6: {extra_in_36}"

    mismatches = {}
    for key in backbone_keys_35:
        if shapes_35[key] != shapes_36[key]:
            mismatches[key] = {"qwen35": shapes_35[key], "qwen36": shapes_36[key]}

    if mismatches:
        for key, diff in mismatches.items():
            print(f"SHAPE MISMATCH: {key}  3.5={diff['qwen35']}  3.6={diff['qwen36']}")
        raise AssertionError(f"{len(mismatches)} shape mismatch(es) found.")
    else:
        print(f"OK: all {len(backbone_keys_35)} backbone weight shapes are identical.")
```

Run this script against the actual checkpoint directories before modifying any TTNN weight-loading code. A clean run (no assertion errors, `OK:` printed) is the definitive confirmation that Chapter 2's claim holds for the specific checkpoint revisions being deployed.

---

**Next:** [`extra_weight_keys.md`](./extra_weight_keys.md)
