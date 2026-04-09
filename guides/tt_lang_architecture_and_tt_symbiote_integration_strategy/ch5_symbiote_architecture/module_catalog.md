# Module Catalog

**Source:** `models/experimental/tt_symbiote/modules/`

This catalog inventories every `TTNNModule` subclass in TT-Symbiote. Understanding the full breadth of modules reveals where boilerplate concentrates and where TT-Lang could have the most impact.

## Activation Modules

**Source:** `modules/activation.py`

| Class | TTNN Op | Notes |
|-------|---------|-------|
| `TTNNSilu` | `ttnn.silu` | Stateless; auto-converts to TILE_LAYOUT |
| `TTNNReLU` | `ttnn.relu` | Stateless; preserves input memory_config |
| `TTNNGelu` | `ttnn.gelu` | Stateless; auto-converts to TILE_LAYOUT |

**Pattern:** All three are structurally identical --- no weights, no preprocessing, just a layout check and a single TTNN call. The `_fallback_torch_layer` is set in `__init__` to the corresponding `torch.nn` activation.

**Pain point:** Three classes that differ only in the TTNN function name. A declarative approach could express all three as a single parameterized template.

## Linear Modules

**Source:** `modules/linear.py`

| Class | Precision | Sharding | CCL | Trace |
|-------|-----------|----------|-----|-------|
| `TTNNLinear` | bfloat16 | None | No | Enabled |
| `TTNNLinearInputShardedWeightSharded` | bfloat16 | Input + Weight | No | Enabled |
| `TTNNLinearIColShardedWRowSharded` | bfloat16 | I:col, W:row | `reduce_scatter` | Enabled |
| `TTNNLinearIColShardedWAllReduced` | bfloat16 | I:col, W:all | `reduce_scatter` + `all_gather` | Enabled |
| `TTNNLinearInputReplicatedWeightSharded` | bfloat16 | I:replicated, W:sharded (dim=-1) | No | Enabled |
| `TTNNLinearIReplicatedWColSharded` | bfloat16 | I:replicated, W:col | No | Enabled |
| `TTNNLinearLLama` | bfloat8_b | None | No | Disabled |
| `TTNNLinearLLamaIColShardedWRowSharded` | bfloat8_b | I:col, W:row | `reduce_scatter` | Disabled |
| `TTNNLinearLLamaBFloat16` | bfloat16 | None | No | Disabled |
| `TTNNLinearActivation` | (varies) | (varies) | No | (varies) |
| `TTNNLinearGelu` | (varies) | (varies) | No | (varies) |
| `TTNNLinearSilu` | (varies) | (varies) | No | (varies) |
| `TTNNViTIntermediate` | (varies) | (varies) | No | (varies) |
| `SmartTTNNLinear` | bfloat16 | None | No | Enabled |
| `SmartTTNNLinearLLama` | bfloat8_b | None | No | Disabled |
| `SmartTTNNLinearLLamaBFloat16` | bfloat16 | None | No | Disabled |

**Source (additional):** `modules/linear_intelligent.py`

The `SmartTTNNLinear` family extends `TTNNLinear` with **automatic prefill/decode dispatch**. Based on the input sequence length (threshold: 32 tokens), `forward()` routes to either a decode path or a prefill path with cached `MatmulMultiCoreReuseMultiCastProgramConfig` program configs. This eliminates manual mode selection by the caller.

- `SmartTTNNLinearLLama` overrides `preprocess_weights_impl()` to use `bfloat8_b` precision and applies `@deallocate_weights_after` for memory efficiency.
- `SmartTTNNLinearLLamaBFloat16` keeps the default `bfloat16` precision but also applies `@deallocate_weights_after`.

**Pattern:** The linear family is the most proliferated module category. The base `TTNNLinear` implements the full lifecycle; variants override specific methods:
- **Sharding variants** override `preprocess_weights_impl()` and `move_weights_to_device_impl()` to add mesh mappers.
- **Precision variants** change `dtype=ttnn.bfloat16` to `dtype=ttnn.bfloat8_b`.
- **CCL variants** add `reduce_scatter` and/or `all_gather` calls in `forward()`.
- **Trace variants** use `@trace_disabled` and `@deallocate_weights_after`.
- **Fused variants** (`TTNNLinearActivation`, `TTNNLinearGelu`, `TTNNLinearSilu`) compose a linear with an activation function.

**Pain point:** 8 linear variants exist because sharding, precision, CCL strategy, and trace compatibility are cross-cutting concerns that the class hierarchy cannot factor cleanly. A TT-Lang approach could express these as orthogonal parameters rather than separate classes.

## Normalization Modules

**Source:** `modules/normalization.py`

| Class | TTNN Op | Notes |
|-------|---------|-------|
| `TTNNLayerNorm` | `ttnn.layer_norm` | Standard LayerNorm |
| `TTNNRMSNorm` | `ttnn.rms_norm` | RMSNorm for DeepSeek/LLaMA-style models |
| `TTNNDistributedRMSNorm` | `ttnn.rms_norm` | RMSNorm with `all_gather` for distributed input |

**Pattern:** Each follows the same lifecycle --- `from_torch()` extracts weights, `preprocess_weights_impl()` converts to TTNN, `move_weights_to_device_impl()` transfers to device, `forward()` calls the TTNN op.

`TTNNDistributedRMSNorm` adds `all_gather` before normalization when operating on sharded inputs, plus a `@run_on_devices(DeviceArch.T3K)` constraint.

## Attention Modules

**Source:** `modules/attention.py`

| Class | Description |
|-------|-------------|
| `TTNNPagedAttentionKVCache` | Paged KV cache using `ttnn.experimental.paged_update_cache` and `ttnn.transformer.paged_scaled_dot_product_attention` |
| `TTNNSDPAAttention` | Scaled dot-product attention wrapper |
| `TTNNBailingMoEAttention` | Full attention for Bailing/Ling models with Q/K/V projections, RoPE, and paged attention |

**Pattern:** Attention modules are the most complex in the hierarchy. `TTNNBailingMoEAttention` composes `TTNNLinear` (for Q/K/V/O projections), `TTNNRotaryPositionEmbedding` (for RoPE), and `TTNNPagedAttentionKVCache` (for KV management). It also handles GQA head repetition and distributed variants.

`TTNNPagedAttentionKVCache` manages block tables, page allocation, and cache updates --- substantial stateful logic that goes well beyond simple weight preprocessing.

## Qwen-Specific Attention

**Source:** `modules/qwen_attention.py`

| Class | Description |
|-------|-------------|
| `TTNNQwenPagedAttentionKVCache` | KV cache with layer-index mapping for Qwen3.5 hybrid attention (maps absolute layer indices to cache slots) |
| `TTNNQwen3FullAttention` | Full GQA attention with Q gating and Q/K normalization |
| `TTNNQwen3LinearAttention` | Linear (DeltaNet) attention with TTNN projections |

**Pattern:** Qwen modules inherit from the base attention classes and override behavior for model-specific needs (softmax vs. sigmoid routing, hybrid full/linear attention patterns). The layer-index mapping in `TTNNQwenPagedAttentionKVCache` is a good example of model-specific logic that does not generalize.

## RoPE Modules

**Source:** `modules/rope.py`

| Class | Description |
|-------|-------------|
| `TorchRotaryPositionEmbedding` | PyTorch reference implementation (fallback) |
| `TTNNRotaryPositionEmbedding` | TTNN-accelerated RoPE using `ttnn.experimental.rotary_embedding` |
| `TTNNDistributedRotaryPositionEmbedding` | Distributed RoPE for multi-device |

**Pattern:** RoPE modules are stateless (no learnable weights). The TTNN version calls `ttnn.experimental.rotary_embedding` which fuses the rotate-half + cos/sin multiply pattern into a single device kernel. The distributed variant handles sharded Q/K tensors.

## MoE Modules

**Source:** `modules/moe.py`

| Class | Description |
|-------|-------------|
| `TTNNMoERouterDecode` | Router: gate linear + topk + normalization |
| `TTNNExperts` | Sparse matmul expert execution with fused gate/up projections |
| `TTNNMoE` | Full MoE block: router + experts + shared expert |
| `TTNNGlm4MoeTopkRouter` | GLM-4 specific router with sigmoid activation |
| `TTNNGlm4MoeMLP` | GLM-4 dense MLP layer |
| `Glm4MoeRouteTokenToExperts` | Token-to-expert routing logic |
| `TTNNBailingMoeV2MLP` | Bailing dense MLP (gate + up + down) |
| `TTNNBailingMoE` | Full Bailing MoE with shared expert and expert gating |

**Pattern:** MoE modules are the most architecturally complex. They involve:
- Sparse matmul with custom `MatmulMultiCoreReuseMultiCast1DProgramConfig`
- TopK routing with padded tensors (minimum width 64)
- Token permutation and unpermutation via index/scatter operations
- Shared expert parallelism with optional gating

The `_make_sparse_matmul_program_config` helper constructs TTNN program configs by querying the device's compute grid size --- a prime example of hardware-aware code that TT-Lang could abstract.

## Qwen-Specific MoE

**Source:** `modules/qwen_moe.py`

| Class | Description |
|-------|-------------|
| `TTNNQwenMoERouterDecode` | Qwen router: softmax instead of sigmoid |
| `TTNNQwenExperts` | Qwen experts: fused w1/w3 sparse matmul |
| `TTNNQwen3MoE` | Full Qwen MoE with `shared_expert` (singular) and optional gate |

**Pattern:** These inherit from the GLM base classes in `moe.py` and override only the differing behavior (softmax vs. sigmoid, fused gate/up projections). The `TTNNQwenMoERouterDecode.forward()` docstring notes: "The only difference from parent is line 22 uses `ttnn.softmax` instead of `ttnn.sigmoid`."

**Pain point:** Creating a full subclass for a one-line change highlights the rigidity of the current approach. A parameterized module or TT-Lang kernel spec could handle this variation without a new class.

## Embedding Modules

**Source:** `modules/embedding.py`

| Class | Description |
|-------|-------------|
| `TTNNEmbedding` | Basic embedding lookup via `ttnn.embedding` |
| `TTNNBailingPaddedEmbedding` | Padded embedding with power-of-2 sequence padding, hidden-dim sharding across mesh |

**Pattern:** `TTNNEmbedding` is straightforward. `TTNNBailingPaddedEmbedding` adds padding logic to align sequences to power-of-2 lengths for TTNN efficiency.

## Conv Modules

**Source:** `modules/conv.py`

| Class | Description |
|-------|-------------|
| `NHWCConvPytorch` | PyTorch Conv2d wrapper handling NHWC layout (fallback) |
| `NHWCMaxpoolPytorch` | PyTorch MaxPool2d wrapper handling NHWC layout (fallback) |
| `NHWCUpsamplePytorch` | PyTorch Upsample wrapper handling NHWC layout (fallback) |
| `TTNNConv2d` | TTNN-accelerated Conv2d using `TtConv2d` builder |
| `TTNNMaxPool2d` | TTNN-accelerated MaxPool2d using `TtMaxPool2d` builder |

**Pattern:** Conv modules use `Conv2dConfiguration` and `MaxPool2dConfiguration` from `models.tt_cnn.tt.builder`, delegating to pre-built TTNN conv implementations. The NHWC wrappers exist because TTNN operates in NHWC format while PyTorch uses NCHW.

## Tensor Operation Modules

**Source:** `modules/tensor.py`

| Class | Description |
|-------|-------------|
| `TTNNPermute` | `ttnn.permute` wrapper |
| `TTNNReshape` | `ttnn.reshape` wrapper |
| `TTNNAdd` | `ttnn.add` wrapper |
| `TTNNMultiply` | `ttnn.multiply` wrapper |
| `TTNNConcat` | `ttnn.concat` wrapper |

**Pattern:** These are thin wrappers around single TTNN ops, used when a model needs an explicit module (e.g., for tracing) rather than relying on dispatch interception. Each has a corresponding `Torch*` fallback class.

## Decoder Layer Modules

**Source:** `modules/decoder_layer.py`

| Class | Description |
|-------|-------------|
| `TTNNBailingMoEDecoderLayer` | Full decoder layer: layernorm + attention + residual add + layernorm + MLP/MoE + residual add |

**Pattern:** The decoder layer composes normalization, attention, and MoE modules, and critically performs **residual additions on-device** using `ttnn.add`. The docstring notes this "eliminates 2 host round-trips per layer" compared to letting the dispatch system handle the additions.

This is a key architectural insight: sometimes the module path is preferred over the dispatch path specifically to avoid host-device synchronization overhead.

## Summary: Module Count by Category

| Category | Module Count | Boilerplate Methods per Module |
|----------|-------------|-------------------------------|
| Activation | 3 | 1 (forward only) |
| Linear | 16 | 3--5 (full lifecycle) |
| Normalization | 3 | 3--4 (full lifecycle) |
| Attention | 3 + 3 Qwen | 3--5 (full lifecycle + KV cache) |
| RoPE | 3 | 1--2 (stateless or minimal) |
| MoE | 8 + 3 Qwen | 3--5 (full lifecycle + routing) |
| Embedding | 2 | 3--4 (full lifecycle) |
| Conv | 2 TTNN + 3 fallback | 3--4 (full lifecycle) |
| Tensor ops | 5 | 1 (forward only) |
| Decoder layer | 1 | 2 (from_torch + forward) |

**Total: ~54 classes** (including PyTorch fallback wrappers).

## Cross-Cutting Pain Points

1. **Combinatorial explosion of variants**: Linear modules alone have 12 variants for sharding x precision x trace x dispatch-mode combinations. Adding a new precision format or sharding strategy requires new subclasses.

2. **Copy-paste inheritance**: Many subclasses differ by one or two lines (e.g., `bfloat16` vs. `bfloat8_b`, `softmax` vs. `sigmoid`). The class hierarchy does not factor these orthogonal concerns.

3. **No weight schema**: There is no declarative description of what weights a module needs, their shapes, or their TTNN layout requirements. This information is embedded in imperative Python code across `preprocess_weights_impl()` and `move_weights_to_device_impl()`.

4. **Device-aware computation hidden in `forward()`**: Grid sizes, program configs, memory configs, and CCL topology are computed inside `forward()` methods. These hardware-specific details could be separated from the algorithmic logic.

5. **Fallback management is manual**: Every `TTNNModule` must maintain a `_fallback_torch_layer`. There is no mechanism to auto-generate fallbacks or to gradually migrate fallback paths to device execution.

These pain points collectively make the case for TT-Lang integration: a declarative language for specifying module behavior, weight layouts, sharding strategies, and device constraints --- with automatic boilerplate generation and compile-time validation.

---

**Next:** [Chapter 6 --- Integration Strategy](../ch6_integration_strategy/index.md)
