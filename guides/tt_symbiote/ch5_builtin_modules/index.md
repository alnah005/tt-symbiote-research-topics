# Chapter 5 — Built-In TTNN Modules

This chapter catalogues every production-ready `TTNNModule` subclass that ships with TT Symbiote. Each class is a drop-in TTNN-accelerated replacement for a corresponding PyTorch layer. The catalogue is grouped by functional type. Detailed documentation for each group follows in the sub-pages listed in the navigation footer.

---

## 5.1 Module Catalogue

### Linear Layers

Source: `models/experimental/tt_symbiote/modules/linear.py` and `modules/linear_intelligent.py`

| Class | Parent | Weight dtype | Notes |
|---|---|---|---|
| `TTNNLinear` | `TTNNModule` | `bfloat16` | Baseline TTNN linear; `@trace_enabled` |
| `TTNNLinearLLama` | `TTNNLinear` | `bfloat8_b` | LLaMA-optimised; `@trace_disabled`, `@deallocate_weights_after` |
| `TTNNLinearLLamaBFloat16` | `TTNNLinear` | `bfloat16` | LLaMA variant keeping bfloat16; `@trace_disabled`, `@deallocate_weights_after` |
| `TTNNLinearInputShardedWeightSharded` | `TTNNLinear` | `bfloat16` | Base class for input-sharded / weight-sharded family |
| `TTNNLinearIColShardedWRowSharded` | `TTNNLinearInputShardedWeightSharded` | `bfloat16` | Input col-sharded, weight row-sharded; uses `reduce_scatter_minimal_async`; T3K only |
| `TTNNLinearInputReplicatedWeightSharded` | `TTNNLinear` | `bfloat16` | Replicated input, col-sharded weight |
| `TTNNLinearIReplicatedWColSharded` | `TTNNLinearInputReplicatedWeightSharded` | `bfloat16` | Convenience subclass; T3K only |
| `TTNNLinearLLamaIColShardedWRowSharded` | `TTNNLinearIColShardedWRowSharded` | `bfloat8_b` | Combines LLaMA bfloat8 + col/row sharding; T3K only |
| `TTNNLinearActivation` | `TTNNModule` | inherits inner `linear_class` | Fused linear + arbitrary TTNN activation |
| `TTNNLinearGelu` | — (factory) | inherits inner `linear_class` | Factory: wraps `TTNNLinearActivation` with `ttnn.gelu` |
| `TTNNLinearSilu` | — (factory) | inherits inner `linear_class` | Factory: wraps `TTNNLinearActivation` with `ttnn.silu` |
| `TTNNViTIntermediate` | `TTNNLinearGelu` | `bfloat16` | ViT intermediate projection with GELU |
| `SmartTTNNLinear` | `TTNNLinear` | `bfloat16` | Dispatches to prefill or decode path at runtime |
| `SmartTTNNLinearLLama` | `SmartTTNNLinear` | `bfloat8_b` | Smart dispatch + bfloat8 + `@deallocate_weights_after` |
| `SmartTTNNLinearLLamaBFloat16` | `SmartTTNNLinear` | `bfloat16` | Smart dispatch + bfloat16 + `@deallocate_weights_after` |

### Normalization Layers

Source: `models/experimental/tt_symbiote/modules/normalization.py`

| Class | Parent | TTNN op | Notes |
|---|---|---|---|
| `TTNNLayerNorm` | `TTNNModule` | `ttnn.layer_norm` | Weight + bias in `bfloat16 / TILE_LAYOUT` |
| `TTNNRMSNorm` | `TTNNModule` | `ttnn.rms_norm` | Weight broadcast-expanded to 32 rows |
| `TTNNDistributedRMSNorm` | `TTNNModule` | `ttnn.rms_norm_pre_all_gather` + `ttnn.rms_norm_post_all_gather` | Two-phase distributed norm over T3K mesh; `@trace_enabled` |

### Activation Layers

Source: `models/experimental/tt_symbiote/modules/activation.py`

| Class | Parent | TTNN op |
|---|---|---|
| `TTNNSilu` | `TTNNModule` | `ttnn.silu` |
| `TTNNReLU` | `TTNNModule` | `ttnn.relu` |
| `TTNNGelu` | `TTNNModule` | `ttnn.gelu` |

### Tensor-Operation Modules

Source: `models/experimental/tt_symbiote/modules/tensor.py`

| Class | Parent | TTNN op |
|---|---|---|
| `TTNNPermute` | `TTNNModule` | `ttnn.permute` |
| `TTNNReshape` | `TTNNModule` | `ttnn.reshape` |
| `TTNNAdd` | `TTNNModule` | `ttnn.add` |

### Convolutional, Pooling, and Embedding Modules

Source: `models/experimental/tt_symbiote/modules/conv.py`

| Class | Parent | Description |
|---|---|---|
| `TTNNConv2dNHWC` | `TTNNModule` | NHWC conv2d using `TtConv2d`; `@trace_enabled` |
| `TTNNConv2dBNNHWC` | `TTNNConv2dNHWC` | Conv2d with BatchNorm folded into weights |
| `TTNNConv2dBNActivationNHWC` | `TTNNConv2dBNNHWC` | Conv + BN + ReLU fused; `ttnn.UnaryOpType.RELU` |
| `TTNNConv2dNHWCInputMultipleOf16` | `TTNNConv2dNHWC` | Auto-pads `in_channels` to multiple of 16 |
| `TTNNMaxPool2dNHWC` | `TTNNModule` | NHWC max-pooling using `TtMaxPool2d` |
| `TTNNUpsampleNHWC` | `TTNNModule` | Nearest / bilinear upsample via `ttnn.upsample` |
| `TTNNPatchEmbedding` | `TTNNModule` | ViT patch embedding via fold + linear |
| `TTNNViTEmbeddings` | `TTNNModule` | Full ViT embedding: patch + CLS token + position |
| `TTNNBottleneck` | `TTNNModule` | ResNet Bottleneck block composed of TTNN conv sub-modules |

### Attention Modules

Source: `models/experimental/tt_symbiote/modules/attention.py`, `qwen_attention.py`

| Class | Parent | Description |
|---|---|---|
| `TTNNSDPAAttention` | `TTNNModule` | Scaled-dot-product attention via `ttnn.transformer.scaled_dot_product_attention` |
| `TTNNFusedQKVSelfAttention` | `TTNNModule` | Fused QKV projection + SDPA self-attention |
| `TTNNSelfAttention` | `TTNNModule` | Fused QKV projection (via `TTNNFusedQKVSelfAttention`) + SDPA self-attention |
| `TTNNViTSelfAttention` | `TTNNSelfAttention` | ViT specialisation of `TTNNSelfAttention` |
| `TTNNWhisperAttention` | `TTNNModule` | Whisper encoder/decoder cross-attention |
| `LlamaAttention` | `TTNNModule` | LLaMA multi-head attention with RoPE + paged KV cache |
| `TTNNGR00TSelfAttention` | `TTNNModule` | GR00T robot-model self-attention variant |
| `TTNNGlm4MoeLiteAttention` | `TTNNModule` | GLM-4 MoE lightweight attention |
| `TTNNQwen3NextGatedAttention` | `TTNNModule` | Qwen3 gated attention |
| `TTNNBailingMoEAttention` | `TTNNModule` | Bailing MoE model attention |
| `TTNNQwen3FullAttention` | `TTNNModule` | Qwen3 full (non-linear) attention |
| `TTNNQwen3LinearAttention` | `TTNNModule` | Qwen3 linear attention variant |

### RoPE Modules

Source: `models/experimental/tt_symbiote/modules/rope.py`

| Class | Parent | Description |
|---|---|---|
| `TTNNRotaryPositionEmbedding` | `TTNNModule` | Single-device RoPE application |
| `TTNNDistributedRotaryPositionEmbedding` | `TTNNModule` | Multi-device distributed RoPE |

### MoE (Mixture-of-Experts) Modules

Source: `models/experimental/tt_symbiote/modules/moe.py`, `qwen_moe.py`

| Class | Parent | Description |
|---|---|---|
| `TTNNGlm4MoeExpertLayers` | `TTNNModule` | GLM-4 MoE expert linear layers |
| `TTNNGlm4MoeNaiveMoe` | `TTNNModule` | GLM-4 naive (non-fused) MoE |
| `TTNNGlm4MoeTopkRouter` | `TTNNLinearIColShardedWRowSharded` | GLM-4 top-k expert router |
| `TTNNGlm4MoeMLP` | `TTNNModule` | GLM-4 MoE MLP block |
| `TTNNGlm4MoeRouteTokenToExperts` | `TTNNModule` | GLM-4 token-to-expert routing |
| `TTNNGlm4MoeMoE` | `TTNNModule` | Full GLM-4 MoE block |
| `TTNNMoERouterDecode` | `TTNNModule` | Decode-phase MoE router |
| `TTNNExperts` | `TTNNModule` | Expert weight storage and dispatch |
| `TTNNMoE` | `TTNNModule` | Generic MoE block |
| `TTNNBailingMoE` | `TTNNMoE` | Bailing model MoE specialisation |
| `TTNNQwenMoERouterDecode` | `TTNNMoERouterDecode` | Qwen MoE router (decode) |
| `TTNNQwenExperts` | `TTNNExperts` | Qwen expert layers |
| `TTNNQwen3MoE` | `TTNNMoE` | Qwen3 full MoE block |

---

## 5.2 Sub-Pages

| Page | Contents |
|---|---|
| [linear_layers.md](./linear_layers.md) | All linear, fused-activation, and smart-dispatch linear modules |
| [normalization_and_activation.md](./normalization_and_activation.md) | Norm and point-wise activation modules |
| [attention_and_conv.md](./attention_and_conv.md) | Attention, RoPE, conv, pooling, embedding, tensor-op, and MoE catalogue |

---

**Navigation:** [Plan](../../plan.md) | Chapter 5 Index | [Linear Layers](./linear_layers.md) | [Normalization and Activation](./normalization_and_activation.md) | [Attention and Conv](./attention_and_conv.md)
