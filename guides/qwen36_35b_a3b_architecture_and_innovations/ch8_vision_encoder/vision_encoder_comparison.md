# Vision Encoder Comparison

## Qwen3.5 vs Qwen3.6

The vision encoder is **identical** between Qwen3.5 and Qwen3.6. All six configuration fields — `num_hidden_layers`, `hidden_size`, `patch_size`, `num_attention_heads`, `spatial_merge_size`, and `temporal_patch_size` — are unchanged between Qwen3.5 and Qwen3.6. The weight values differ (different post-training runs), but tensor shapes are identical. A TTNN implementation of the Qwen3.5 vision encoder requires **zero architectural changes** to run the Qwen3.6 vision encoder. Only the checkpoint weights need to be swapped.

## Comparison with Gemma4

Gemma4's vision encoder is a shallower, narrower ViT with a different pooling strategy:

| Parameter | Qwen3.6 | Gemma4 |
|-----------|---------|--------|
| `num_hidden_layers` | 27 | 16 |
| `hidden_size` | 1152 | 768 |
| `patch_size` | 16 | 16 |
| Pooling | 2×2 spatial merge | 3×3 pooling kernel |

Qwen3.6's vision encoder is 69% deeper (27 vs 16 layers) and 50% wider (1152 vs 768 hidden size). Both use patch size 16, but their post-ViT token reduction strategies differ:

- **Qwen3.6** averages each $2 \times 2$ group of patch tokens, reducing token count by 4× (see [`vision_encoder_specs.md`](./vision_encoder_specs.md) for worked examples).
- **Gemma4** uses a $3 \times 3$ pooling kernel, reducing token count by 9×; for the same 448×448 image, 784 patches become approximately 87 vision tokens.

Qwen3.6 therefore injects more vision tokens per image into the text sequence, providing finer-grained spatial detail at the cost of a longer context. Gemma4's more aggressive pooling saves context length but discards more spatial resolution.

## Comparison with LLaVA-Style Vision Encoders

LLaVA-style models typically use a frozen, pre-trained CLIP ViT as the vision backbone. The most common variant is ViT-L/14:

| Parameter | Qwen3.6 | LLaVA (ViT-L/14) |
|-----------|---------|------------------|
| `num_hidden_layers` | 27 | 24 |
| `hidden_size` | 1152 | 1024 |
| `patch_size` | 16 | 14 |
| Training | Custom, end-to-end | Pre-trained CLIP, frozen or fine-tuned |

Qwen3.6's encoder is custom-trained (not borrowed from CLIP) and is slightly larger: 3 more layers and 128 wider per layer. The patch size difference (16 vs 14) means Qwen3.6 produces fewer patches per image at the same resolution, though this is partially offset by Qwen3.6's less aggressive spatial merge (4× reduction vs LLaVA's typical 4× or no reduction depending on variant).

The custom training of Qwen3.6's vision encoder allows it to be jointly optimized with the language decoder, avoiding the domain mismatch that can arise when a CLIP-pretrained encoder is paired with a language model trained on a very different objective.

## TTNN Deployment Considerations

### Prefill-Only Encoding

The vision encoder runs exclusively during **prefill** — the phase in which the model processes the prompt, including any embedded images or video frames. During autoregressive decode, no new image data arrives, so the vision encoder is idle. This means:

- Vision encoding is a **one-time cost** paid at the start of the conversation. The resulting vision embeddings are computed once and cached as part of the KV cache or attention context.
- The 27-layer ViT does not contribute to per-token decode latency at all.
- TTNN can treat the vision encoder as a separate, standalone prefill-time graph that executes before the main LM prefill graph.

### Text-Only Deployment

For applications that do not use images or video, the vision encoder can be **entirely omitted**. The decoder-only language model (the MoE transformer with 35B total / 3B active parameters) is fully self-contained. Omitting the vision encoder saves approximately **300M parameters** of DRAM, which is meaningful on memory-constrained hardware even though it is small relative to the 35B total model size.

### Spatial Merge and Projection Are Simple Ops

After the 27 ViT layers, the remaining vision-specific operations are:

1. **Spatial 2×2 average pool.** Each $2 \times 2$ block of adjacent patch tokens is averaged. This is a straightforward reshape + reduce-mean on the spatial dimensions, with no learned parameters.
2. **Linear projection (1152 → 2048).** A single learned weight matrix maps each pooled token from vision hidden size to decoder hidden size.

Both operations map directly to standard TTNN primitives (reduce, matmul) with no custom kernels required. The projection weight is a `[1152, 2048]` matrix, contributing approximately 2.36M parameters (1152 × 2048 = 2,359,296).

---

**End of guide.** Return to [Guide Index](../index.md)
