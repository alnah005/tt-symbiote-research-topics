# Config Diff: Qwen3.5 vs Qwen3.6

## Overview

This document provides a field-by-field comparison of the `config.json` files for Qwen3.5-35B-A3B and Qwen3.6-35B-A3B. The comparison is organized into four categories:

1. **Identical fields** — the vast majority; these confirm architectural equivalence.
2. **Added in Qwen3.6** — fields present in Qwen3.6 but absent in Qwen3.5.
3. **Removed in Qwen3.6** — fields present in Qwen3.5 but absent in Qwen3.6.
4. **Changed fields** — fields present in both but with different values.

The conclusion is stated up front: **there are zero architectural differences**. Every addition, removal, and change is either a convenience default being made explicit, a redundancy being cleaned up, or a metadata string being updated.

---

## 1. Identical Fields

The following fields are present in both configs with identical values. They collectively define the entire neural architecture.

### Model Class and Type

| Field | Qwen3.5 | Qwen3.6 |
|-------|---------|---------|
| `architectures[0]` | `"Qwen3_5MoeForConditionalGeneration"` | `"Qwen3_5MoeForConditionalGeneration"` |
| `model_type` | `"qwen3_5_moe"` | `"qwen3_5_moe"` |

Both versions use the same HuggingFace model class. This is the field that determines which Python class is instantiated when you call `AutoModelForCausalLM.from_pretrained()`. Identical class means identical forward pass graph.

### Text Encoder: Core Dimensions

| Field | Value (both) | Notes |
|-------|-------------|-------|
| `num_hidden_layers` | `40` | Total transformer block depth |
| `hidden_size` | `2048` | Token embedding / residual stream dimension |
| `intermediate_size` | `768` | Dense MLP intermediate width (for `mlp_only_layers`; all layers are MoE so this field is currently unused) |
| `num_attention_heads` | `16` | Multi-head attention heads |
| `num_key_value_heads` | `2` | GQA key/value head count |
| `head_dim` | `256` | Per-head dimension (explicit config field; set to 256 directly, not derived from other fields) |

### Hybrid Attention Layer Types

The `layer_types` list specifies whether each of the 40 layers uses DeltaNet linear attention (`"deltanet"`) or full softmax attention (`"full_attention"`). This list is identical in both versions.

- Full attention layers occur at positions that satisfy $(i + 1) \bmod \text{full\_attention\_interval} = 0$, where `full_attention_interval = 4`.
- The pattern produces full attention at layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 (0-indexed).
- All remaining 30 layers use DeltaNet.

Both configs encode the same 40-element list and the same `full_attention_interval = 4`.

### DeltaNet Configuration

| Field | Value (both) |
|-------|-------------|
| `deltanet_use_beta` | `true` |
| `deltanet_use_short_conv` | `true` |
| `deltanet_conv_size` | `4` |
| `linear_num_key_heads` | `16` |
| `linear_num_value_heads` | `32` |
| `linear_key_head_dim` | `128` |
| `linear_value_head_dim` | `128` |
| `deltanet_qk_norm` | `"l2"` |
| `deltanet_use_output_gate` | `true` |

These fields fully specify the DeltaNet linear attention kernel. Identical values confirm identical operator shapes.

**DeltaNet state buffer — implementer note.** Q and K are each projected to 16 heads of dimension 128, producing output shape `[B, T, 16, 128]`. The key tensor is then expanded via `repeat_interleave(2)` along the head dimension, yielding `[B, T, 32, 128]`; key head `i` (for i in 0..15) is thereby paired with value heads `2i` and `2i+1`. The state update runs over all 32 (K_expanded, V) pairs, each maintaining an independent state matrix in R^{128×128}. Consequently the per-layer state buffer has shape `[B, 32, 128, 128]`.

**DeltaNet output retrieval — implementer note.** The query tensor is also projected to 16 heads of dimension 128, giving `[B, T, 16, 128]`. It is then expanded via the same `repeat_interleave(2)` to `[B, T, 32, 128]`, so query head `i` (for i in 0..15) addresses state matrices `2i` and `2i+1`. For each of the 32 expanded heads `j`, the output at time step `t` is computed as `o_t^j = S_t^j · (q̃_t^j / sqrt(128))`, where `S_t^j` is the 128×128 state matrix and `q̃_t^j` is the length-128 query vector; this produces a length-128 output vector per head, with overall shape `[B, T, 32, 128]`. The 32 head outputs are then concatenated along the head dimension to yield `[B, T, 4096]` (= 32 × 128). Finally, `out_proj` maps `[B, T, 4096]` → `[B, T, 2048]`, restoring the residual stream dimension.

### MoE (Mixture of Experts) Configuration

| Field | Value (both) | Notes |
|-------|-------------|-------|
| `moe_intermediate_size` | `1536` | Per-expert FFN intermediate width |
| `num_experts` | `128` | Total routed experts per MoE layer |
| `num_experts_per_tok` | `8` | Top-K routing at inference |
| `norm_topk_prob` | `true` | Normalize routing probabilities after top-K |
| `router_aux_loss_coef` | `0.001` | Auxiliary load-balancing loss coefficient |
| `shared_expert_intermediate_size` | `768` | Shared (always-active) expert width |
| `mlp_only_layers` (Qwen3.5) | `[]` | Empty list — no layers use plain MLP |

Note: `mlp_only_layers` is discussed under "Removed in Qwen3.6" below. In Qwen3.5 it was an empty list, so its removal has no functional effect.

### Attention Mechanism

| Field | Value (both) | Notes |
|-------|-------------|-------|
| `attention_dropout` | `0.0` | No attention dropout at inference |
| `sliding_window` | `null` | No sliding window attention |
| `use_sliding_window` | `false` | Confirms no sliding window |
| `max_window_layers` | `40` | All layers can use full window |
| `rope_theta` | `1000000.0` | RoPE base frequency |
| `max_position_embeddings` | `32768` | Maximum context length |

### RoPE Parameters (Nested)

Both configs include a `rope_parameters` nested object under `text_config` specifying partial rotary:

| Field | Value (both) |
|-------|-------------|
| `rope_type` | `"default"` |
| `partial_rotary_factor` (in `rope_parameters`) | `0.5` |

This means RoPE is applied to only the first 50% of each head's dimension. This is the same in both versions. (See Chapter 4 for a deep dive on partial RoPE.)

### Tokenizer and Vocabulary

| Field | Value (both) |
|-------|-------------|
| `vocab_size` | `248320` |
| `tie_word_embeddings` | `false` |

### Vision Encoder Configuration

The vision encoder fields (image resolution, patch size, vision hidden size, vision layers, vision attention heads, etc.) are fully identical between Qwen3.5 and Qwen3.6. Both encode the same ViT-based visual encoder. A complete listing appears in Chapter 6 (Vision Encoder).

### Miscellaneous Identical Fields

| Field | Value (both) |
|-------|-------------|
| `hidden_act` | `"silu"` |
| `rms_norm_eps` | `1e-6` |
| `initializer_range` | `0.02` |
| `use_cache` | `true` |
| `torch_dtype` | `"bfloat16"` |

---

## 2. Fields Added in Qwen3.6

These fields appear in Qwen3.6's `config.json` but are absent from Qwen3.5's. In every case the field either (a) makes a previously implicit default explicit, or (b) exposes a training-time parameter that is meaningless at inference.

### `bos_token_id`: 248044

**What it does:** Specifies the beginning-of-sequence token ID. This is a tokenizer convenience field stored in the model config for completeness. With `vocab_size = 248320`, the value 248044 is a valid token ID (248044 < 248320).

**Why it was absent in Qwen3.5:** The value is also present in `tokenizer_config.json`. Qwen3.5 omitted the redundant copy in `config.json`; Qwen3.6 adds it explicitly.

**Architectural impact:** None. The BOS token ID does not affect any weight tensor or forward pass computation.

### `output_router_logits`: false

**What it does:** When `true`, the model's forward pass returns the raw MoE router logit tensors in addition to the hidden states. This is used during training to compute the auxiliary load-balancing loss.

**Why it was absent in Qwen3.5:** The default is already `false` in the HuggingFace implementation. Qwen3.6 makes the inference-time setting explicit.

**Architectural impact:** None at inference. Setting this to `false` means router logits are computed internally but discarded — identical to the implicit behavior in Qwen3.5.

### `pad_token_id`: null

**What it does:** Specifies the padding token ID for batched inference. `null` means no padding token is defined, which is the standard configuration for decoder-only models that use left-padding or dynamic batching.

**Why it was absent in Qwen3.5:** Implicit null is the default.

**Architectural impact:** None. Padding token ID affects batch collation in the data loader, not the model forward pass.

### `partial_rotary_factor` at top-level `text_config`

**What it does:** Exposes the partial rotary factor (`0.5`) as a top-level field in `text_config`, in addition to its presence inside the nested `rope_parameters` object.

**Why it was absent in Qwen3.5:** In Qwen3.5, `partial_rotary_factor` only appeared inside `rope_parameters`. Qwen3.6 duplicates it at the top level, presumably for compatibility with code that reads it from either location.

**Architectural impact:** None. The value is `0.5` in both locations in Qwen3.6, and `0.5` inside `rope_parameters` in Qwen3.5. No computation changes.

---

## 3. Fields Removed in Qwen3.6

### `mlp_only_layers`: [] (empty list in Qwen3.5, absent in Qwen3.6)

**What it does:** Specifies which layer indices should use a plain MLP instead of an MoE layer. An empty list means every layer uses MoE (no plain MLP layers).

**Why it was removed:** An empty list is functionally equivalent to the field being absent. Qwen3.6 simply drops the no-op field to clean up the config.

**Architectural impact:** None. Both configs specify zero plain-MLP layers.

---

## 4. Changed Fields

### Model Name Strings

| Field | Qwen3.5 | Qwen3.6 |
|-------|---------|---------|
| `_name_or_path` | `"Qwen/Qwen3.5-35B-A3B"` | `"Qwen/Qwen3.6-35B-A3B"` |

This is a metadata string identifying the HuggingFace Hub repository. It does not affect model loading, weight shapes, or any computation.

### `transformers_version`

| Field | Qwen3.5 | Qwen3.6 |
|-------|---------|---------|
| `transformers_version` | `"4.57.0.dev0"` | `"4.57.1"` |

This records the version of the HuggingFace `transformers` library used to save the config. Qwen3.5 was released using a development build; Qwen3.6 was released after the stable `4.57.1` release. This field has no effect on model behavior.

---

## Complete Diff Summary

```
Fields identical:     ~45 fields covering all of: architectures, model_type,
                      num_hidden_layers, hidden_size, intermediate_size,
                      num_attention_heads, num_key_value_heads, head_dim,
                      layer_types (all 40), full_attention_interval,
                      all linear_*/deltanet_* fields, all moe/expert fields,
                      attention_dropout, sliding_window, use_sliding_window,
                      rope_theta, max_position_embeddings, vocab_size,
                      hidden_act, rms_norm_eps, torch_dtype,
                      rope_parameters.partial_rotary_factor,
                      all vision encoder fields

Fields added in 3.6:  4 fields (bos_token_id, output_router_logits,
                      pad_token_id, partial_rotary_factor at top level)
                      → all are defaults made explicit; zero architectural impact

Fields removed in 3.6: 1 field (mlp_only_layers = [])
                       → empty list equivalent to absence; zero impact

Fields changed:       2 fields (_name_or_path string, transformers_version)
                      → metadata only; zero architectural impact
```

---

## Conclusion: Architectural Equivalence

The config diff establishes unambiguously that Qwen3.5-35B-A3B and Qwen3.6-35B-A3B are architecturally identical. They share:

- The same HuggingFace model class (`Qwen3_5MoeForConditionalGeneration`), meaning the same Python forward pass graph.
- The same `model_type` string (`qwen3_5_moe`), meaning the same registered config and modeling code.
- Identical values for every field that determines weight tensor shapes, dtypes, and op types.

The four added fields and one removed field carry no architectural meaning. The two changed fields are metadata strings.

## Implication for TTNN

Because the architecture is identical, the TTNN implementation consequence is direct:

> **Any TTNN device graph that correctly executes Qwen3.5-35B-A3B will also correctly execute Qwen3.6-35B-A3B with zero model code changes.**

The only required change is pointing the weight loading code at the Qwen3.6 checkpoint directory. All tensor shapes, all kernel dispatch decisions, all memory layout assignments, and all op fusion decisions remain valid. The `output_router_logits = false` field in Qwen3.6 is already the implicit behavior in any inference-optimized TTNN backend that does not expose router logits, so no special handling is needed.

---

**Next:** [`post_training_differences.md`](./post_training_differences.md)
