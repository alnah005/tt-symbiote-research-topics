# Chapter 4 — Patch Embedding and Adaptive Pooling in TTNN

## Learning Objectives

After completing this chapter, you will be able to:

- Explain how Gemma 4's flatten-then-linear patch embedding differs from Gemma 3's Conv2d approach, and why this simplifies the TTNN port
- Implement the 2D learned position embedding lookup using `ttnn.embedding` and element-wise addition
- Describe how the adaptive 2D average pooler assigns patches to grid cells and reduces the token count
- Evaluate three TTNN implementation strategies for the pooler (built-in avg_pool2d, reshape+mean, custom op)
- Identify the variable-input-shape implications for TTNN program caching and tracing

## Prerequisites

- Completion of [Chapter 1 — Gemma 4 Vision Encoder Architecture Overview](../ch01_gemma4_vision_architecture/index.md) (variable-resolution processing, patch grid concept)
- Completion of [Chapter 2 — SigLIP vs. Gemma 4 Comparison](../ch02_siglip_vs_gemma4_comparison/index.md), specifically [`module_mapping.md`](../ch02_siglip_vs_gemma4_comparison/module_mapping.md) (patch embedding and projector differences)
- Familiarity with TTNN ops: `ttnn.linear`, `ttnn.embedding`, `ttnn.reshape`, `ttnn.mul`, `ttnn.add`

## Chapter Contents

| File | Topic |
|------|-------|
| [`patch_embedding_port.md`](./patch_embedding_port.md) | Porting the patch embedder: flatten+linear projection, 2D position embeddings, and variable-shape implications |
| [`adaptive_pooling_port.md`](./adaptive_pooling_port.md) | Porting the adaptive pooler: 2D grid-based average pooling, standardization, RMSNorm+linear projection |

## Overview

The patch embedder and the adaptive pooler are the two modules in Gemma 4's vision encoder that diverge most from the Gemma 3 SigLIP architecture. They sit at opposite ends of the encoder stack — the embedder transforms raw pixels into the initial hidden-state sequence, and the pooler compresses the encoded sequence into the final token budget — but both share a common challenge: they must handle variable spatial dimensions that change per image.

### Why These Are Critical Path Items

These modules are critical path for the TTNN port for three reasons:

1. **No direct Gemma 3 equivalent to copy.** The Gemma 3 SigLIP encoder uses a Conv2d patch embedding with fixed 896x896 input and a fixed-count average pooling layer. Neither of these can be reused for Gemma 4 without a major rewrite. In contrast, the 27 encoder layers (attention + MLP) share the same hidden dimensions and can be adapted from existing code (see [Chapter 2, `module_mapping.md`](../ch02_siglip_vs_gemma4_comparison/module_mapping.md)).

2. **Variable spatial dimensions affect the entire pipeline.** The patch embedder determines the sequence length that flows through all 27 encoder layers, and the pooler determines the final token count injected into the language model. Any TTNN implementation must either handle dynamic shapes or commit to a fixed set of supported resolutions.

3. **They gate end-to-end integration.** Until the embedder produces correctly positioned hidden states and the pooler produces the correct number of output tokens, the full vision encoder cannot be assembled and validated against the CPU reference.

The encoder layers between these two bookend modules (attention, MLP, RMSNorm) are largely reusable from Gemma 3 and are covered in [Chapter 6](../ch06_reuse_strategy/index.md). The 2D RoPE applied within those encoder layers is covered in [Chapter 3](../ch03_2d_factored_rope/index.md). This chapter focuses exclusively on the entry and exit points of the vision encoder.

### Reading Order

Start with [`patch_embedding_port.md`](./patch_embedding_port.md), which covers the simpler of the two modules and establishes the variable-shape conventions that the pooler also depends on. Then proceed to [`adaptive_pooling_port.md`](./adaptive_pooling_port.md), which addresses the more complex adaptive pooling logic and the downstream RMSNorm + linear projection.
