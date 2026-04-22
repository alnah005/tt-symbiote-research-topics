# Chapter 3 — Full TTNN Vision Stack

## Overview

This chapter traces the evolution of the dots.ocr vision encoder implementation from the interim hybrid approach — where the 42 HuggingFace ViT layers ran on the CPU host and only `PatchMergerTT` executed on device — to the current full TTNN mode, where every component from pixel patch extraction through spatial token merging runs entirely on TT hardware. It covers the role and implementation of each TTNN module, the reuse of `PatchMergerTT` from the `qwen25_vl` port, and the scatter fusion mechanism that inserts vision tokens into the text embedding sequence.

Prerequisites: Chapter 1 (`vision_encoder_specs.md`) for the `post_norm=True` flag, `patch_size=14`, `spatial_merge_size=2`, and the 42-layer depth; Chapter 2 (`model_args_and_transformer.md`) for `DotsModelArgs` and the two-stack design philosophy.

All file paths are relative to `models/demos/dots_ocr/` unless explicitly prefixed otherwise.

## Reading Order

| File | Contents |
|------|----------|
| [`vision_components_ttnn.md`](./vision_components_ttnn.md) | `PatchEmbedTT`, `VisionBlockTT` post-norm ordering, `VisionAttentionTT`, `VisionMLPTT` SwiGLU, `VisionRMSNorm`, `DotsVisionModelArgs`, `VisionTransformerTT` orchestration |
| [`patch_merger_and_fusion.md`](./patch_merger_and_fusion.md) | `PatchMergerTT` reuse from `qwen25_vl`, spatial merge math and worked example, `tt/fusion.py` scatter mechanism |

Read in order. This index establishes the component flow and the rationale for full TTNN mode before the detail files cover each component.

## Component Flow

The following diagram shows the full data path from raw pixel data to fused embeddings ready for the text decoder.

```
Input image (H × W pixels, RGB)
        │
        ▼
  PatchEmbedTT          tt/vision_patch_embed.py
  ─────────────         14×14 conv expressed as TTNN matmul
  [B, 1, S_patch, 1536] S_patch = (H/14) × (W/14)
        │
        ▼
  VisionBlockTT × 42    tt/vision_block.py (loop in tt/vision.py)
  ─────────────         Post-norm: RMSNorm after attention, RMSNorm after MLP
  [B, 1, S_patch, 1536] shape preserved through all 42 layers
        │
        ▼
  Post-trunk RMSNorm    tt/vision_rmsnorm.py (applied once after layer 42)
  [B, 1, S_patch, 1536]
        │
        ▼
  PatchMergerTT         tt/patch_merger.py (reused from models/demos/qwen25_vl/)
  ─────────────         2×2 spatial merge: 4 tokens → 1 token
  [B, 1, S_img, 1536]   S_img = S_patch / 4 = H×W / 784
        │
        ▼
  scatter fusion        tt/fusion.py
  ─────────────         Index-scatter: vision tokens → image_token_id positions
  [B, S_total, 1536]    Fused embedding tensor for DotsTransformer
        │
        ▼
  DotsTransformer (text decoder)
```

Each box corresponds to a TTNN class or op. The shape annotations use the conventions from `vision_config`: `hidden_size=1536`, `S_patch` for the pre-merge token count, `S_img` for the post-merge token count.

## Why Full TTNN Mode

### Hybrid Approach, Cost, and Retention

The hybrid mode (`use_full_ttnn=False`) ran all 42 `VisionBlockTT` layers as HuggingFace PyTorch ops on the CPU host, with only `PatchMergerTT` and the text decoder on device. The cost was PCIe bandwidth at the vision-encoder output boundary: for a high-resolution document image (e.g., 896×1344), the output tensor `[B, 1, S_patch, 1536]` with `S_patch=6144` must transfer from host DRAM to device memory over PCIe on every forward pass.

Full TTNN mode (`use_full_ttnn=True`, the default) eliminates this transfer — the entire forward pass from `PatchEmbedTT` through `PatchMergerTT` executes on-device. `FULL_TTNN_VISION_PLAN.md` estimated the vision encoder at ~1.2B parameters and the port at 2–3× the effort of the hybrid path; the latency improvement justifies it.

Hybrid mode is retained as a fallback for CPU-only environments. The mode-selection logic applies two rules: (a) when `mesh_device is not None`, `use_full_ttnn` is forced to `True` regardless of the flag passed to `VisionEncoder`; (b) when `mesh_device is None` and `use_full_ttnn=False`, hybrid mode is active and the HuggingFace `vision_tower` runs on host. Any T3K deployment will always have `mesh_device is not None`, so hybrid mode is never active in production.

### Requirement: Real Checkpoint Weights

`VisionEncoder` requires a `state_dict` to be provided at initialization. Dummy weights are not supported for the vision stack. This is consistent with `DotsModelArgs`'s `dummy_weights=False` override described in Chapter 2: the vision encoder's post-norm RMSNorm and patch embedding projections are numerically sensitive enough that random weights produce incoherent activations that make PCC validation meaningless.

---

**Next:** [`vision_components_ttnn.md`](./vision_components_ttnn.md)
