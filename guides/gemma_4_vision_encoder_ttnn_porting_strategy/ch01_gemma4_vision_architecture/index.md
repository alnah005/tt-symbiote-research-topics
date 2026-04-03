# Chapter 1 — Gemma 4 Vision Encoder Architecture Overview

## Learning Objectives

After completing this chapter, you will be able to:

- Describe the complete module hierarchy of the Gemma 4 vision encoder, from raw pixels to language model soft tokens
- Identify every sub-module, its role, and the tensor shapes flowing between them
- Explain the key configuration parameters of the 31B model's vision encoder and how they determine the architecture
- Understand how Gemma 4 handles variable-resolution images with aspect-ratio preservation and configurable token budgets
- Recognize the TTNN porting implications of variable input shapes and the 2D RoPE scheme

## Prerequisites

- Familiarity with Vision Transformer (ViT) architectures and patch-based image encoding
- Working knowledge of HuggingFace Transformers model structure (`PreTrainedModel`, `nn.Module` hierarchies)
- Basic understanding of Rotary Position Embeddings (RoPE)
- Exposure to the Gemma 3 SigLIP vision encoder is helpful but not required

## Chapter Contents

| File | Topic |
|------|-------|
| [`module_hierarchy.md`](./module_hierarchy.md) | Full module tree, sub-component roles, and end-to-end data flow |
| [`config_parameters.md`](./config_parameters.md) | Complete Gemma4VisionConfig parameter table for the 31B model |
| [`variable_resolution_processing.md`](./variable_resolution_processing.md) | Aspect-ratio preservation, divisibility constraints, and token budgets |

## Overview

The Gemma 4 vision encoder represents a significant architectural departure from the Gemma 3 approach. Where Gemma 3 relied on a frozen SigLIP encoder with fixed 224x224 or 896x896 square inputs, Gemma 4 introduces a custom vision encoder built from scratch with three defining characteristics:

1. **Variable-resolution input** — images retain their native aspect ratio rather than being squashed into a fixed square. Both height and width must be divisible by 48 (see [`variable_resolution_processing.md`](./variable_resolution_processing.md) for derivation).

2. **2D Rotary Position Embeddings** — instead of 1D positional encodings over a flattened patch sequence, Gemma 4 applies multidimensional RoPE using explicit (x, y) grid coordinates. This gives the encoder genuine spatial awareness.

3. **Configurable token budgets** — a single image can be encoded to 70, 140, 280, 560, or 1120 soft tokens, letting the user trade visual fidelity for inference speed.

The encoder produces approximately 570M parameters and outputs hidden states of dimension 1152, which are then projected to the language model's hidden dimension (5376 for the 31B variant) through an RMSNorm + linear projection layer.
