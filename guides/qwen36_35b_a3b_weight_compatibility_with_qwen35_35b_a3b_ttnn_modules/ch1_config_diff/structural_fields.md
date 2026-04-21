# Structural Fields: Identical Config Parameters

## Purpose of This File

This file documents every `config.json` field that is **identical** between
`Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B`. It then explains why
identical values for these specific fields guarantee that all weight tensor
shapes consumed by the existing TTNN module suite are preserved, and that no
TTNN matmul program config needs to change. It also covers a small set of
fields that differ numerically between the two versions but do not affect any
tensor dimension.

## Architecture Identity Fields

Both checkpoints use the same HuggingFace architecture class and model type
string. These fields drive which Python class `AutoModelForCausalLM` and
`AutoConfig` instantiate.

| Field | Value (both versions) | Notes |
|---|---|---|
| `architectures` | `["Qwen3_5MoeForConditionalGeneration"]` | Single-element list; determines the PyTorch module class |
| `model_type` | `"qwen3_5_moe"` | Used by the HuggingFace model registry to resolve `AutoConfig` and `AutoModel` |
| `torch_dtype` | `"bfloat16"` | Default inference dtype; does not affect weight shapes |
| `transformers_version` | _(may differ by patch)_ | Metadata only; no impact on architecture or weights |

Because `architectures` and `model_type` are identical, `AutoModelForCausalLM.from_pretrained`
instantiates the same class (`Qwen3_5MoeForConditionalGeneration`) for both
checkpoints. TT-Symbiote's module replacement logic — which identifies the host
model class by its `model_type` or class name before substituting TTNN modules
— is therefore unaffected.

## Structural Hyperparameters Governing Weight Shapes

The fields below directly set the dimensions used in weight allocation. Because
they are identical in both configs, every weight tensor produced by
`Qwen3_5MoeForConditionalGeneration.__init__` has the same shape for both
checkpoints.

| Field | Value (both versions) | Governed weight dimensions |
|---|---|---|
| `hidden_size` | `7168` | `q_proj`, `k_proj`, `v_proj`, `o_proj` input/output dims; MoE router input dim; embedding `[vocab_size, 7168]`; all RMSNorm weight vectors |
| `num_hidden_layers` | `94` | Total decoder layers; determines size of `model.layers` list |
| `num_attention_heads` | `64` | Query head count; `q_proj` output rows = `num_attention_heads * head_dim` |
| `num_key_value_heads` | `4` | GQA KV head count; `k_proj` and `v_proj` output rows = `num_key_value_heads * head_dim` |
| `head_dim` | `128` | Per-head dimension; $\text{q\_proj shape} = [64 \times 128, 7168] = [8192, 7168]$; $\text{k/v\_proj shape} = [4 \times 128, 7168] = [512, 7168]$ |
| `intermediate_size` | `14336` | Dense MLP intermediate dim (used only in layers that lack MoE; in this hybrid architecture, confirmed by `decoder_sparse_step`) |
| `moe_intermediate_size` | `2048` | Per-expert hidden dimension; each routed expert `gate_proj` and `up_proj` shape = `[2048, 7168]`; `down_proj` shape = `[7168, 2048]` |
| `num_experts_per_tok` | `8` | Router top-K; does not set any weight dimension directly but governs routing logic |
| `num_experts` | `256` | Number of routed experts per MoE layer; determines the count of expert weight tensors per layer |
| `shared_expert_intermediate_size` | `2048` | Shared-expert hidden dim; shared expert `gate_proj`/`up_proj` shape = `[2048, 7168]`; `down_proj` shape = `[7168, 2048]` |
| `vocab_size` | `151936` | Token embedding table shape = `[151936, 7168]`; LM head shape = `[151936, 7168]` (tied or untied) |

Because every hyperparameter entering these expressions is identical in both configs, the shapes are provably identical without inspecting the safetensors files.

## RoPE and Rotary Embedding Fields

The fields governing the rotary position embedding computation are identical
between the two versions. This means the cosine and sine tables produced by
`TTNNRotaryPositionEmbedding` have the same shape and content when constructed
from either config.

| Field | Value (both versions) | Notes |
|---|---|---|
| `rope_theta` | `1000000.0` | Base frequency for the default RoPE schedule |
| `rope_scaling.rope_type` | `"yarn"` | Extended-context YaRN scaling |
| `rope_scaling.factor` | `4.0` | YaRN scale factor |
| `rope_scaling.original_max_position_embeddings` | `32768` | Pre-extension context length |
| `rope_scaling.beta_fast` | `32` | YaRN fast-decay frequency boundary |
| `rope_scaling.beta_slow` | `1` | YaRN slow-decay frequency boundary |
| `rope_scaling.mscale` | `0.1` | YaRN magnitude scaling coefficient |
| `rope_scaling.mscale_all_dim` | `0.1` | YaRN per-dimension magnitude scaling |
| `rope_parameters.partial_rotary_factor` | `0.25` | Fraction of head dimensions that receive rotation |
| `rope_parameters.rope_type` | `"yarn"` | Mirrors `rope_scaling.rope_type` |

The `partial_rotary_factor` value inside `rope_parameters` is `0.25` in both
configs. The Qwen3.6 config additionally places this value at the **top level**
of `config.json`. This promotion is discussed in
[`new_and_modified_fields.md`](./new_and_modified_fields.md) and analysed in
full in [Chapter 3](../ch3_partial_rotary_factor/index.md). The value itself
does not change; the numeric consequence (`d_rot = 32`) is derived in
[`new_and_modified_fields.md`](./new_and_modified_fields.md).

## Why Identical Structural Fields Mean No TTNN Changes Are Needed

The TTNN module suite for Qwen3.5-35B-A3B allocates all device-side weight
buffers with shapes derived exclusively from the hyperparameters listed above.
Because every one of those hyperparameters is identical in the Qwen3.6 config:

- **No matmul program config changes are needed.** Tile counts, subblock sizes, and loop orders are functions of tensor shapes; identical shapes mean identical program configs.
- **No weight preprocessing changes are needed.** Dtype casts, shard layouts, and tensor permutations are keyed on weight key names and shapes, both of which are unchanged.
- **No TTNN tensor allocation size changes are needed.** Every `ttnn.from_torch` allocation is passed a shape derived from the hyperparameters above; identical hyperparameters mean identical allocation sizes.
- **KV cache allocation is unaffected.** The paged KV cache is dimensioned by `num_key_value_heads`, `head_dim`, and `num_hidden_layers`, all of which are unchanged.

## Fields Confirmed Identical — Listed for Completeness

A small number of additional fields are listed here for completeness. As the
table below confirms, these fields carry identical values in both configs and
have no bearing on any weight matrix dimension.

| Field | Qwen3.5 value | Qwen3.6 value | Impact |
|---|---|---|---|
| `initializer_range` | `0.02` | `0.02` | Training-time weight initialization stddev; not read at inference |
| `rms_norm_eps` | `1e-6` | `1e-6` | Controls numerical floor in RMSNorm; no shape impact; possible minor numerical difference in mixed-precision inference |
| `hidden_act` | `"silu"` | `"silu"` | Activation function for FFN gate; SiLU is already compiled into the TTNN MoE kernel; no change required |
| `tie_word_embeddings` | `false` | `false` | Whether `lm_head.weight` shares storage with `embed_tokens.weight`; both false means LM head is a separate weight tensor of shape `[151936, 7168]` |
| `attention_bias` | `false` | `false` | No bias vectors in Q/K/V/O projections; confirmed absent from both checkpoints |
| `max_position_embeddings` | `131072` | `131072` | Maximum sequence length; used only for position ID bounds checking and KV cache sizing, not for weight shapes |

The values in this table are consistent between the two versions. They are
documented here to confirm there are no hidden numerical changes that could
alter TTNN kernel selection or numerical behaviour at inference time.

---

**Next:** [`new_and_modified_fields.md`](./new_and_modified_fields.md)
