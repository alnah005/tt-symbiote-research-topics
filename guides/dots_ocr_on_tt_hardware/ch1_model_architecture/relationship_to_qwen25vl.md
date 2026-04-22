# Relationship to Qwen 2.5 VL

This file characterizes the architectural relationship between dots.ocr and Qwen 2.5 VL. The key claim is that dots.ocr is a **derived architecture** — it inherits the design pattern and tooling of the Qwen2-VL family but was trained from scratch with an incompatible weight configuration.

## Architectural lineage

dots.ocr descends from the Qwen2-VL line in the following concrete ways:

**1. Text decoder pattern.** `DotsOCRForCausalLM` inherits from `Qwen2ForCausalLM`. The decoder block structure — pre-norm RMSNorm, grouped query attention with attention bias, SwiGLU MLP, RoPE positional encoding — follows the Qwen2 specification exactly. The class hierarchy and module naming conventions in the HuggingFace implementation match those of Qwen2.

**2. Shared vocabulary.** `vocab_size: 151936` is identical to Qwen2 and Qwen2.5. The tokenizer (tiktoken BPE) is the same. This means text token embeddings are in the same token ID space; a text sequence tokenized with the Qwen2 tokenizer produces the same token IDs whether processed by Qwen2 or dots.ocr.

**3. Image preprocessing utilities.** dots.ocr uses `qwen_vl_utils` for image preprocessing, the same utility library as Qwen2-VL. Patch extraction, normalization constants, and dynamic resolution handling are inherited directly.

**4. Vision config schema.** The fields in dots.ocr's `vision_config` — `spatial_merge_size`, `temporal_patch_size`, `post_norm`, `attn_implementation` — use the same field names and semantics as the Qwen2-VL vision config schema. The patch embedding output dimension equals `hidden_size` in both models. The PatchMerger module and its 2×2 spatial merge operation are structurally identical.

**5. Special token placement.** The `image_token_id` (151665) and `video_token_id` (151656) fall in the Qwen2 special token range (above 151,643), using the same convention of sentinel tokens at high token IDs.

## Shared identifiers with notes

| Identifier | dots.ocr | Qwen2.5-VL-7B | Notes |
|---|---|---|---|
| `vocab_size` | 151936 | 151936 | Identical — same tokenizer |
| `patch_size` | 14 | 14 | Identical — same patch grid |
| `spatial_merge_size` | 2 | 2 | Identical — 2×2 merge in PatchMerger |
| `rope_theta` | 1000000 | 1000000 | Same RoPE base frequency |
| `hidden_act` | silu | silu | Same activation |
| `rms_norm_eps` (text) | 1e-06 | 1e-06 | Same text decoder norm epsilon |
| `image_token_id` | 151665 | 151655 | Different by 10 positions |
| `video_token_id` | 151656 | 151646 | Different by 10 positions |

The 10-position offset in `image_token_id` and `video_token_id` is notable. Both models share the same vocabulary, but dots.ocr places its image sentinel token at a different position in the special token range. Code that hardcodes the Qwen2.5-VL `image_token_id` of 151655 will not correctly identify image token positions in dots.ocr sequences.

## Key divergences

### 1. Incompatible weight tensor shapes

Every weight tensor in the text decoder has a different shape in dots.ocr versus Qwen2.5-VL-7B:

| Tensor | dots.ocr shape | Qwen2.5-VL-7B shape |
|---|---|---|
| Input embedding | $151936 \times 1536$ | $151936 \times 3584$ |
| Q projection | $1536 \times 1536$ | $3584 \times 3584$ |
| K projection | $1536 \times 256$ | $3584 \times 512$ |
| MLP gate | $8960 \times 1536$ | $18944 \times 3584$ |
| RMSNorm scale | $1536$ | $3584$ |

No weight from Qwen2.5-VL-7B can be loaded into dots.ocr. The models are weight-incompatible in both rank and dimension. This rules out dots.ocr being a quantized, pruned, or distilled variant of Qwen2.5-VL-7B at the weight level.

### 2. Vision encoder depth inversion

Qwen2.5-VL-7B has a 32-layer vision encoder paired with a 28-layer text decoder. dots.ocr has a 42-layer vision encoder paired with the same 28-layer text decoder. The vision encoder in dots.ocr is deeper than the one it is paired with, and deeper than Qwen2.5-VL-7B's vision encoder, despite the text decoder being 4.3x smaller.

This reflects a design choice specific to document parsing: extracting high-fidelity features from dense document images is more demanding than the generation task, so the vision branch receives proportionally more capacity.

### 3. Temporal patch design

dots.ocr uses `temporal_patch_size: 1` (static-image-only design), versus Qwen2.5-VL-7B's `temporal_patch_size: 2` (video support); see [`vision_encoder_specs.md`](./vision_encoder_specs.md) for the full analysis.

### 4. Context length

`max_position_embeddings: 131072` in dots.ocr versus `32768` in Qwen2.5-VL-7B. The RoPE embeddings are precomputed at training time up to the maximum sequence length. The RoPE tables in the two models are therefore of different sizes and cover different frequency ranges.

### 5. Attention head configuration

12Q/2KV (6:1 ratio) in dots.ocr versus 28Q/4KV (7:1 ratio) in Qwen2.5-VL-7B. Both use aggressive GQA, but the absolute dimensions are different at every level.

### 6. Vision encoder hidden size

Because `vision_config.hidden_size` equals the text decoder's `hidden_size` (both 1536), no cross-modal projection is needed — see [`vision_encoder_specs.md`](./vision_encoder_specs.md) for context.

## Why dots.ocr is derived, not a fine-tune

"Fine-tuning" means starting from a pretrained checkpoint and continuing training with the same weight shapes on a new dataset or objective. Because all weight tensor shapes differ between dots.ocr and Qwen2.5-VL-7B, no fine-tuning process could produce dots.ocr from a Qwen2.5-VL-7B checkpoint. The model must have been initialized with random weights (or a separate pretraining procedure) and trained from scratch.

The relationship is instead at the **architecture design level**:

- The Qwen2 decoder pattern was adopted as the text backbone design.
- The Qwen2-VL vision encoder pattern (ViT with PatchMerger, `post_norm=True`) was adopted as the vision backbone design.
- The vocabulary, tokenizer, and preprocessing utilities were adopted unchanged.
- All hyperparameters (hidden sizes, layer counts, head counts) were chosen fresh for the 1.7B document-focused model.

This is analogous to how many models "derive from LLaMA" by adopting its architecture while training from scratch on different data with different hyperparameters. dots.ocr derives from Qwen2-VL in the same sense.

## Implications for the TTNN port

Understanding the lineage has two practical consequences for the TTNN port:

**Reuse of Qwen2-VL TTNN modules.** Any TTNN kernels or module wrappers already written for Qwen2-VL attention, MLP, RMSNorm, or PatchMerger can be reused for dots.ocr with updated dimension arguments. The control flow and module graph structure are the same; only the tensor shapes change.

**No weight loading from Qwen checkpoints.** The TTNN port must load dots.ocr's own checkpoint weights. There is no shortcut of loading a Qwen2.5-VL-7B checkpoint and patching it. The weight loading logic must target `rednote-hilab/dots.ocr` directly.

---

**Next:** [Chapter 2 — TTNN Port Architecture](../ch2_ttnn_port_architecture/index.md)
