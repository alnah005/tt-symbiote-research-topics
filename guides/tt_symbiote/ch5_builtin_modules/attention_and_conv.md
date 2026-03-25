# Chapter 5.3 — Attention, Conv, and Supporting Modules

This page is an existence catalogue. Detailed per-class documentation for attention modules is covered in the model-specific chapters of the guide series. The purpose here is to record every `TTNNModule` subclass that lives in the supporting module files so the full surface area of Chapter 5 is complete.

---

## Attention Modules

Source: `models/experimental/tt_symbiote/modules/attention.py`

| Class | Parent | One-line description |
|---|---|---|
| `TTNNSDPAAttention` | `TTNNModule` | General-purpose scaled-dot-product attention using `ttnn.transformer.scaled_dot_product_attention` |
| `TTNNFusedQKVSelfAttention` | `TTNNModule` | Self-attention with a single fused QKV projection matrix |
| `TTNNSelfAttention` | `TTNNModule` | Self-attention using an internal fused QKV projection (`TTNNFusedQKVSelfAttention`) + SDPA |
| `TTNNViTSelfAttention` | `TTNNSelfAttention` | ViT-specialised self-attention; inherits `TTNNSelfAttention` |
| `TTNNWhisperAttention` | `TTNNModule` | Whisper encoder/decoder cross-attention (see Ch4 of the model guide) |
| `LlamaAttention` | `TTNNModule` | LLaMA multi-head attention with RoPE and paged KV cache (see Ch3 of the model guide) |
| `TTNNGR00TSelfAttention` | `TTNNModule` | Self-attention variant for the GR00T robot model |
| `TTNNGlm4MoeLiteAttention` | `TTNNModule` | Lightweight attention block used in GLM-4 MoE layers |
| `TTNNQwen3NextGatedAttention` | `TTNNModule` | Gated attention for Qwen3 Next architecture |
| `TTNNBailingMoEAttention` | `TTNNModule` | Attention block for the Bailing MoE model |

Source: `models/experimental/tt_symbiote/modules/qwen_attention.py`

| Class | Parent | One-line description |
|---|---|---|
| `TTNNQwen3FullAttention` | `TTNNModule` | Qwen3 standard (non-linear) multi-head attention |
| `TTNNQwen3LinearAttention` | `TTNNModule` | Qwen3 linear-complexity attention variant |

### Supporting Attention Infrastructure

| Class | Parent | Description |
|---|---|---|
| `TTNNPagedAttentionKVCache` | `Cache` (HuggingFace) | Paged key-value cache backed by TTNN tensors; manages page tables for variable-length sequences |
| `TTNNQwenPagedAttentionKVCache` | `TTNNPagedAttentionKVCache` | Qwen-specific paged KV cache |

---

## RoPE Modules

Source: `models/experimental/tt_symbiote/modules/rope.py`

| Class | Parent | One-line description |
|---|---|---|
| `TTNNRotaryPositionEmbedding` | `TTNNModule` | Applies rotary position embeddings to Q and K tensors on a single device |
| `TTNNDistributedRotaryPositionEmbedding` | `TTNNModule` | Distributed RoPE application across a multi-device mesh |

---

## Conv, Pooling, and Embedding Modules

Source: `models/experimental/tt_symbiote/modules/conv.py`

All conv and pooling modules operate in NHWC layout natively. The fallback PyTorch layers (`NHWCConvPytorch`, `NHWCMaxpoolPytorch`, `NHWCUpsamplePytorch`) handle the NCHW-to-NHWC permutation for CPU execution.

| Class | Parent | One-line description |
|---|---|---|
| `TTNNConv2dNHWC` | `TTNNModule` | TTNN Conv2d in NHWC format; caches compiled `TtConv2d` instances by input shape; `@trace_enabled` |
| `TTNNConv2dBNNHWC` | `TTNNConv2dNHWC` | Conv2d with BatchNorm weights folded into conv weights at preprocessing time |
| `TTNNConv2dBNActivationNHWC` | `TTNNConv2dBNNHWC` | Conv + BN + ReLU fused; passes `ttnn.UnaryOpType.RELU` into the conv config (ReLU only) |
| `TTNNConv2dNHWCInputMultipleOf16` | `TTNNConv2dNHWC` | Auto-pads `in_channels` to the nearest multiple of 16; falls back to `TTNNConv2dNHWC.from_torch` if already valid |
| `TTNNMaxPool2dNHWC` | `TTNNModule` | NHWC max-pooling via `TtMaxPool2d`; integer kernel/stride/padding/dilation only |
| `TTNNUpsampleNHWC` | `TTNNModule` | Nearest or bilinear upsample via `ttnn.upsample` in NHWC format |
| `TTNNPatchEmbedding` | `TTNNModule` | ViT patch embedding: fold + `ttnn.linear`; weights in `bfloat8_b` |
| `TTNNViTEmbeddings` | `TTNNModule` | Full ViT input embedding: patch projection + CLS token concatenation + positional embedding addition |
| `TTNNBottleneck` | `TTNNModule` | ResNet Bottleneck block composed of two `TTNNConv2dBNActivationNHWC` layers and one `TTNNConv2dBNNHWC` layer |

---

## Tensor-Operation Modules

Source: `models/experimental/tt_symbiote/modules/tensor.py`

These modules wrap primitive tensor operations as `TTNNModule` subclasses so they participate in the fallback, weight lifecycle, and run-mode machinery.

| Class | Parent | TTNN op | Fallback PyTorch op |
|---|---|---|---|
| `TTNNPermute` | `TTNNModule` | `ttnn.permute(input, perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)` | `tensor.permute(perm)` |
| `TTNNReshape` | `TTNNModule` | `ttnn.reshape(input, shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)` | `tensor.reshape(shape)` |
| `TTNNAdd` | `TTNNModule` | `ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)` | `a + b` |

`TTNNAdd` calls `ensure_tile_layout` on both inputs before the add operation.

---

## MoE (Mixture-of-Experts) Modules

Source: `models/experimental/tt_symbiote/modules/moe.py`

| Class | Parent | One-line description |
|---|---|---|
| `TTNNGlm4MoeExpertLayers` | `TTNNModule` | Expert linear layers for GLM-4 MoE blocks |
| `TTNNGlm4MoeNaiveMoe` | `TTNNModule` | Non-fused (naive) MoE implementation for GLM-4 |
| `TTNNGlm4MoeTopkRouter` | `TTNNLinearIColShardedWRowSharded` | Top-k expert router for GLM-4; inherits col/row sharding |
| `TTNNGlm4MoeMLP` | `TTNNModule` | MLP block inside a GLM-4 MoE layer |
| `TTNNGlm4MoeRouteTokenToExperts` | `TTNNModule` | Token-to-expert dispatch logic for GLM-4 |
| `TTNNGlm4MoeMoE` | `TTNNModule` | Complete GLM-4 MoE block |
| `TTNNMoERouterDecode` | `TTNNModule` | Decode-phase MoE router (used at generation time) |
| `TTNNExperts` | `TTNNModule` | Expert weight storage and matmul dispatch |
| `TTNNMoE` | `TTNNModule` | Generic MoE block; base for model-specific MoE subclasses |
| `TTNNBailingMoE` | `TTNNMoE` | Bailing model MoE specialisation |

Source: `models/experimental/tt_symbiote/modules/qwen_moe.py`

| Class | Parent | One-line description |
|---|---|---|
| `TTNNQwenMoERouterDecode` | `TTNNMoERouterDecode` | Qwen-specific decode MoE router |
| `TTNNQwenExperts` | `TTNNExperts` | Qwen expert layers |
| `TTNNQwen3MoE` | `TTNNMoE` | Qwen3 full MoE block |

---

**Next:** [Chapter 6 — Authoring a New TTNN Module](../ch6_authoring_modules/index.md)
