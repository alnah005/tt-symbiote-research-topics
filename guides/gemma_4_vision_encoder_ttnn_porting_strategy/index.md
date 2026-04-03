# Gemma 4 Vision Encoder TTNN Porting Strategy

This guide presents a complete porting strategy for the Gemma 4 vision encoder (~570M params, hidden_size=1152, 27 transformer layers) from HuggingFace reference code to TTNN, covering architecture analysis, operator mapping, Gemma 3 code reuse opportunities, and a latency-driven implementation roadmap. It is written for ML systems engineers and kernel developers on the TT-NN / tt-symbiote stack who need to bring up the vision encoder on Tenstorrent hardware.

## Prerequisites

**Required:**
- Familiarity with vision transformer (ViT) architectures (patch embedding, multi-head self-attention, MLP blocks)
- Working knowledge of TTNN tensor operations, memory configs, and program configs
- Experience with `TTNNModule` authoring in tt-symbiote (module replacement, `forward` signatures)
- Basic understanding of rotary position embeddings (RoPE) in the context of attention

**Not required:**
- Knowledge of SigLIP or Gemma 4 specific vision encoder innovations (2D factored RoPE, adaptive pooling)
- Prior experience with variable-resolution image processing in vision encoders
- Understanding of how the Gemma 3 SigLIP TTNN modules differ from the Gemma 4 vision encoder
- Familiarity with CPU-to-TTNN latency profiling methodology

## How to Use This Guide

| Goal | Recommended Path | Entry Point |
|------|-----------------|-------------|
| Understand the Gemma 4 vision encoder architecture from scratch | Ch01 → Ch02 → Ch03 → Ch04 | [Ch 01](./ch01_gemma4_vision_architecture/index.md) |
| Decide what to reuse from existing Gemma 3 TTNN modules | Ch02 → Ch06 → Ch07 | [Ch 02](./ch02_siglip_vs_gemma4_comparison/index.md) |
| Implement 2D factored RoPE for vision on TTNN | Ch01 → Ch03 | [Ch 03](./ch03_2d_factored_rope/index.md) |
| Port the patch embedding and adaptive pooling layers | Ch01 → Ch04 → Ch05 | [Ch 04](./ch04_patch_embedding_and_pooling/index.md) |
| Profile and prioritize operators by latency impact | Ch05 → Ch07 | [Ch 05](./ch05_cpu_vs_ttnn_latency/index.md) |
| Plan the full implementation with risk assessment | Ch01 → Ch05 → Ch06 → Ch07 | [Ch 07](./ch07_implementation_roadmap/index.md) |
| Maximize code reuse and minimize new kernel work | Ch02 → Ch06 | [Ch 06](./ch06_reuse_strategy/index.md) |

## Chapter Index

| Chapter | Title | Description | Key Concepts |
|---------|-------|-------------|--------------|
| [Ch 01](./ch01_gemma4_vision_architecture/index.md) | Gemma 4 Vision Encoder Architecture Overview | Complete architecture reference for the ~570M param vision encoder: 27 layers, hidden_size=1152, variable resolution, and novel components | 27 transformer layers, hidden_size=1152, num_heads=16, head_dim=72, variable resolution input |
| [Ch 02](./ch02_siglip_vs_gemma4_comparison/index.md) | SigLIP vs. Gemma 4 Comparison | Side-by-side comparison of Gemma 3 SigLIP and Gemma 4 vision encoder architectures to identify structural deltas | Learned vs. factored positional encoding, pooling differences, layer count and dimension changes |
| [Ch 03](./ch03_2d_factored_rope/index.md) | 2D Factored RoPE for Vision | Deep-dive into the 2D factored rotary position embedding with theta=100, replacing learned position embeddings | 2D factored RoPE, theta=100, row/column frequency decomposition, variable-resolution support |
| [Ch 04](./ch04_patch_embedding_and_pooling/index.md) | Patch Embedding and Adaptive Pooling | Patch embedding convolution and adaptive average pooling with pooling_kernel_size=3 for token count reduction | Conv2d patch embedding, adaptive avg pooling, pooling_kernel_size=3, token sequence compression |
| [Ch 05](./ch05_cpu_vs_ttnn_latency/index.md) | CPU vs. TTNN Latency Analysis | Operator-level latency profiling comparing CPU reference to projected TTNN performance for prioritization | Per-operator latency breakdown, attention vs. MLP cost, data transfer overhead, bottleneck identification |
| [Ch 06](./ch06_reuse_strategy/index.md) | Reuse Strategy for Gemma 3 Modules | Assessment of which Gemma 3 SigLIP TTNN modules can be reused, adapted, or must be written from scratch | ~15% direct reuse, ~50% reusable with modifications, ~35% new; MLP and norms carry over, attention needs RoPE delta, pooling is new |
| [Ch 07](./ch07_implementation_roadmap/index.md) | Implementation Roadmap and Risk Assessment | Phased implementation plan with effort estimates, dependency ordering, and risk mitigation strategies | Phase ordering, critical path through RoPE and pooling, testing milestones, risk matrix |

## Quick Reference

| Concept / Component | What It Does | Where to Learn More |
|---------------------|-------------|---------------------|
| Vision encoder (~570M params) | 27-layer ViT that encodes variable-resolution images into token embeddings for the Gemma 4 LLM decoder | [Ch 01](./ch01_gemma4_vision_architecture/index.md) |
| hidden_size=1152, 16 heads, head_dim=72 | Core transformer dimensions for every encoder layer | [Ch 01](./ch01_gemma4_vision_architecture/index.md) |
| 2D factored RoPE (theta=100) | Decomposes rotary embeddings into row and column spatial frequencies for 2D patch grids; replaces learned position embeddings | [Ch 03](./ch03_2d_factored_rope/index.md) |
| Adaptive average pooling (kernel=3) | Reduces the token sequence length after the encoder by pooling over spatial neighborhoods | [Ch 04](./ch04_patch_embedding_and_pooling/index.md) |
| Variable resolution input | Supports different image resolutions without fixed position embedding tables | [Ch 01](./ch01_gemma4_vision_architecture/index.md), [Ch 03](./ch03_2d_factored_rope/index.md) |
| SigLIP (Gemma 3) vs. Gemma 4 delta | Structural differences that determine what can and cannot be reused from the existing Gemma 3 port | [Ch 02](./ch02_siglip_vs_gemma4_comparison/index.md) |
| Gemma 3 code reuse (~15% direct, ~50% with mods, ~35% new) | MLP and norms reuse directly; attention, block, config need modifications; RoPE, pooling, position embeddings are new | [Ch 06](./ch06_reuse_strategy/index.md) |
| Operator latency profiling | CPU baseline vs. projected TTNN latency per operator to guide porting priority | [Ch 05](./ch05_cpu_vs_ttnn_latency/index.md) |
| Phased implementation plan | Ordered roadmap from patch embedding through full encoder integration with risk assessment | [Ch 07](./ch07_implementation_roadmap/index.md) |

## Source Code Location

This guide is self-contained and does not depend on external source code. The reference HuggingFace implementation of the Gemma 4 vision encoder lives in the [transformers](https://github.com/huggingface/transformers) repository under `src/transformers/models/gemma4/`. TTNN op implementations (including `conv2d`, `rms_norm`, `scaled_dot_product_attention`, and `avg_pool2d`) live in the [tt-metal](https://github.com/tenstorrent/tt-metal) repository. Model-level `TTNNModule` implementations, including the existing Gemma 3 SigLIP vision encoder port, are authored in the [tt-symbiote](https://github.com/tenstorrent/tt-symbiote) repository.
