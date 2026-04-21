# Qwen3.6-35B-A3B MTP Configuration

This file grounds the abstract MTP architecture described in `mtp_head_architecture.md` in the specific configuration of Qwen3.6-35B-A3B. It covers the relevant model config fields, the expected weight-key naming convention in the checkpoint, the relationship between the MTP head's hyperparameters and the backbone's hyperparameters, and the lineage context comparing Qwen3.6 to Qwen3.5.

---

## Relevant Configuration Fields

Qwen3.6-35B-A3B is a Mixture-of-Experts (MoE) model. Its `config.json` contains both backbone hyperparameters and MTP-specific fields. The fields relevant to understanding the MTP head are:

| Field | Value | Role |
|-------|-------|------|
| `mtp_num_hidden_layers` | `1` | Number of transformer decoder blocks in the MTP head; determines draft depth $N = 1$ |
| `hidden_size` | `7168` | Hidden dimension $H$; used by both backbone and MTP head attention/FFN |
| `intermediate_size` | `2048` | Dense FFN intermediate dimension (used in backbone non-MoE sublayers and, as described below, in the MTP head) |
| `moe_intermediate_size` | `2048` | Per-expert intermediate dimension in the MoE FFN layers |
| `num_hidden_layers` | `94` | Number of backbone transformer layers (not MTP head layers) |
| `num_attention_heads` | `64` | Number of query attention heads in the backbone; MTP head matches |
| `num_key_value_heads` | `8` | Number of key/value heads (GQA); MTP head matches |
| `head_dim` | `112` | Per-head dimension ($H / \text{num\_attention\_heads} = 7168 / 64 = 112$) |
| `vocab_size` | `151936` | Vocabulary size $V$; dimension of `lm_head` output |
| `max_position_embeddings` | `131072` | Maximum sequence length $S_{\text{max}}$ |
| `num_experts` | `128` | Number of MoE experts in each MoE FFN layer of the backbone |
| `num_experts_per_tok` | `8` | Number of experts activated per token in the backbone MoE layers |

The MTP head does **not** use a MoE FFN. The MTP head block uses a dense FFN with intermediate dimension equal to `intermediate_size` (2048). This is a critical distinction: the backbone's transformer layers (with the exception of a small number of dense layers at fixed positions) use sparse MoE FFNs with 128 experts, whereas the single MTP head block uses a standard dense FFN. The MTP head is therefore substantially lighter than a typical backbone layer in terms of FFN parameter count (see Chapter 2, `mtp_weight_inventory.md` for the full accounting).

The MTP head's attention sublayer uses the same grouped query attention (GQA) configuration as the backbone: 64 query heads and 8 key/value heads with head dimension 112. Rotary position embeddings (RoPE) are applied in the MTP head's attention sublayer with the same rope parameters as the backbone.

---

## MTP Head Weight Keys in the Checkpoint

Qwen3.6-35B-A3B's checkpoint stores MTP head weights under a dedicated prefix in the `state_dict`. Based on the model's HuggingFace implementation (class `Qwen3MoeForCausalLM` with the MTP sub-module), the expected naming convention is:

```
model.future_prediction.0.enorm.weight
model.future_prediction.0.hnorm.weight
model.future_prediction.0.self_attn.q_proj.weight
model.future_prediction.0.self_attn.k_proj.weight
model.future_prediction.0.self_attn.v_proj.weight
model.future_prediction.0.self_attn.o_proj.weight
model.future_prediction.0.mlp.gate_proj.weight
model.future_prediction.0.mlp.up_proj.weight
model.future_prediction.0.mlp.down_proj.weight
model.future_prediction.0.input_layernorm.weight
model.future_prediction.0.post_attention_layernorm.weight
```

The index `0` in `future_prediction.0` is the block index within the MTP head stack. For `mtp_num_hidden_layers: 1` there is only index `0`. For a hypothetical `mtp_num_hidden_layers: 2` model there would also be `future_prediction.1.*` keys.

Note that `enorm` and `hnorm` are the dedicated layer norm weights applied to the shifted token embedding and backbone hidden state respectively, prior to their combination (as described in `mtp_head_architecture.md`). These are distinct from the backbone's layer norms and from the MTP head block's internal `input_layernorm` and `post_attention_layernorm`.

The `lm_head` weight is stored as `lm_head.weight` and is shared between the backbone and the MTP head. There is no separate `model.future_prediction.0.lm_head` key; the same tensor is referenced by both the backbone's output projection and the MTP head's output projection.

**Important caveat:** The exact key names above reflect the pattern observed in DeepSeek-V3 and related open-source MTP implementations. The Qwen3.6 HuggingFace implementation should be confirmed against the actual checkpoint's `state_dict.keys()` output before writing TTNN weight-loading code. Chapter 3, `mtp_weight_loading_behavior.md`, covers the verification procedure in detail.

---

## MTP Head Hyperparameters vs. Backbone Hyperparameters

The MTP head block uses the same attention and normalization hyperparameters as the backbone (hidden size, attention heads, head dimension, RMSNorm, SiLU activation, RoPE — see the first table above). The key difference is in the FFN:

| Hyperparameter | Backbone (MoE layers) | Backbone (dense layers) | MTP Head |
|---------------|----------------------|------------------------|----------|
| FFN type | Sparse MoE | Dense | Dense |
| FFN intermediate size | 2048 per expert × 128 experts | 2048 | 2048 |
| Effective intermediate width per token | 2048 per expert × 8 experts = 16384 effective | 2048 | 2048 |

The consequence is that the MTP head block is computationally equivalent to a single dense backbone layer (not to a MoE backbone layer). Its FFN has $3 \times 7168 \times 2048$ parameters (gate, up, down projections) plus the dedicated `enorm`/`hnorm` normalization weights. This is substantially smaller than a MoE backbone layer, which has $128 \times 3 \times 2048 \times 7168$ parameters across all experts but $8 \times 3 \times 2048 \times 7168$ active parameters per token at the MoE routing level.

---

## Lineage Comparison: Qwen3.6 vs. Qwen3.5

Understanding when MTP was introduced into the Qwen lineage is useful for reasoning about which models require MTP porting work and which do not.

**Qwen3.5-35B-A3B** (the immediate predecessor in the 35B MoE lineage): The Qwen3.5 model series does not include `mtp_num_hidden_layers` in its `config.json`. The HuggingFace model class for Qwen3.5 (`Qwen3MoeForCausalLM` in the `transformers` library) does not define an MTP head sub-module. Qwen3.5 checkpoints contain no `model.future_prediction.*` weight keys.

**Qwen3.6-35B-A3B**: Adds `mtp_num_hidden_layers: 1` and the corresponding MTP head weights to the checkpoint. The model class gains the `future_prediction` sub-module. This is the first model in the Qwen 35B MoE lineage to include an MTP head.

The implication is that MTP in the Qwen lineage was introduced at the Qwen3.6 generation. Bring-up engineers who have already completed a Qwen3.5-35B-A3B integration in `tt-transformers` will encounter the MTP head as a new component requiring explicit handling — either loading and using its weights for speculative decoding, or deliberately discarding them during weight loading.

**Qwen3 dense models** (e.g., Qwen3-8B, Qwen3-32B): These also include `mtp_num_hidden_layers: 1` in their configurations, suggesting MTP was applied consistently across the Qwen3 generation. The MTP head architecture for dense Qwen3 models is structurally identical to the MoE case, with the difference that the backbone layers are dense rather than sparse, and the MTP head FFN intermediate size matches the backbone's dense FFN intermediate size.

**Summary table:**

| Model | `mtp_num_hidden_layers` | MTP head in checkpoint | MTP class in HF |
|-------|------------------------|----------------------|-----------------|
| Qwen3.5-35B-A3B | Absent (field not present) | No | No |
| Qwen3.6-35B-A3B | 1 | Yes | Yes |
| Qwen3-8B (dense) | 1 | Yes | Yes |
| Qwen3-32B (dense) | 1 | Yes | Yes |
| DeepSeek-V3-671B | 1 | Yes | Yes |

The absence of `mtp_num_hidden_layers` in Qwen3.5 is not a default-zero situation — the field is simply not present, and the model class lacks the corresponding sub-module entirely. Attempting to load a Qwen3.5 checkpoint with code written for Qwen3.6's MTP head will produce missing-key warnings for all `model.future_prediction.*` keys, which is the expected behavior if the Qwen3.5 checkpoint is loaded into the Qwen3.6 model class. The reverse — loading a Qwen3.6 checkpoint into a Qwen3.5 model class — would produce unexpected-key warnings for all MTP weights, and those weights would be silently discarded.

---

## References

- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- [Qwen35] Qwen Team, "Qwen3.5 Technical Report", Alibaba Cloud, 2024.
- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- [HF-Qwen3] HuggingFace Transformers, `Qwen3MoeForCausalLM` model implementation, https://github.com/huggingface/transformers, accessed 2025.

---

**Next:** [Chapter 2 — MTP Head Weight Shapes and Memory Footprint](../ch02_mtp_weights_and_memory/index.md)
