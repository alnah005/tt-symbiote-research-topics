# PatchMerger and Fusion

## Overview

This file covers the two operations that connect the vision encoder output to the text decoder input: `PatchMergerTT`, which reduces the vision token count by 4× through spatial merging, and the scatter fusion in `tt/fusion.py`, which inserts the merged vision tokens into their designated positions in the text embedding sequence. Both mechanisms are shared with the Qwen 2.5 VL architecture and, in the case of `PatchMergerTT`, the TTNN implementation is reused directly from the `qwen25_vl` demo.

---

## `PatchMergerTT` (`tt/patch_merger.py`)

### Reuse from `qwen25_vl`

`PatchMergerTT` is reused from `models/demos/qwen25_vl/tt/patch_merger.py` with adaptation for the dots.ocr config. The same TTNN ops, the same weight matrix layout (`TILE_LAYOUT` / `BFLOAT16`), and the same spatial merge logic are used. The adaptation is limited to updating the dimension arguments to match dots.ocr's `hidden_size=1536` and `spatial_merge_size=2`.

This reuse is justified by the architectural identity noted in Chapter 1 (`relationship_to_qwen25vl.md`): the PatchMerger module and its 2×2 spatial merge operation are structurally identical between the two models. The TTNN implementation transfers without structural changes; only the tensor shapes differ, and at `hidden_size=1536` both the input and output projections fit the same TTNN op signatures.

The `test_patch_merger_pcc.py` test confirms that `PatchMergerTT` as adapted for dots.ocr produces outputs that agree with `reference/patch_merger.py` above the PCC threshold.

### Architecture

```
Input:  [B, 1, S_patch, 1536]
  │
  ├── LayerNorm (if ln_q.bias in checkpoint, HF default)
  │   or RMSNorm (fallback)
  │       [B, 1, S_patch, 1536]
  │
  ├── 2×2 spatial group + concat
  │       [B, 1, S_img, 6144]   S_img = S_patch / 4
  │
  ├── Linear(6144 → 6144)       feed_forward.0.weight
  │       [B, 1, S_img, 6144]
  │
  ├── GELU
  │       [B, 1, S_img, 6144]
  │
  └── Linear(6144 → 1536)       feed_forward.2.weight
          [B, 1, S_img, 1536]
```

> **Note (norm selection):** LayerNorm is used when `ln_q.bias` is present in the checkpoint (the HF default for dots.ocr); RMSNorm is the fallback. The `tt/patch_merger.py` comment records: "Using RMSNorm here tanked PCC vs HF. Prefer LayerNorm when checkpoint has bias."

---

## Spatial Merge Math

### Token Count Formula

The token count formula is derived in [Chapter 1 — `vision_encoder_specs.md`](../ch1_model_architecture/vision_encoder_specs.md).

$$N = H \times W / 784$$

For dots.ocr with `spatial_merge_size=2` and `patch_size=14`: each 14×14 patch covers 196 pixels, 4 patches merge into one token covering 784 pixels.

> **Note:** The serialized patch sequence from `PatchEmbedTT` is stored in raster-scan order (row-major: left-to-right, top-to-bottom). For a $2 \times 2$ spatial merge to be correct, the implementation must reconstruct 2D grid coordinates before grouping. Merging every 4 consecutive tokens in the flat sequence is only equivalent to 2×2 spatial merge when the image width in patches is a multiple of 2 — which is guaranteed by the `qwen_vl_utils` dynamic resolution preprocessing.

### Worked Example

For a document image resized to $896 \times 1344$ pixels (a common resolution for A4-page OCR):

**Step 1: Patch grid**

$$H_p = \frac{896}{14} = 64 \text{ patches}, \quad W_p = \frac{1344}{14} = 96 \text{ patches}$$

$$S_{patch} = 64 \times 96 = 6{,}144 \text{ patch tokens}$$

**Step 2: After 2×2 spatial merge**

$$S_{img} = \frac{64}{2} \times \frac{96}{2} = 32 \times 48 = 1{,}536 \text{ vision tokens}$$

Equivalently, using the direct formula:

$$N = \frac{896 \times 1{,}344}{784} = \frac{1{,}204{,}224}{784} = 1{,}536$$

These 1,536 vision tokens are the ones that must exactly match the number of `image_token_id` placeholder positions in the tokenized input sequence. Any mismatch between the image dimensions, the tokenizer's vision placeholder count, and the `PatchMergerTT` output length causes a runtime error in the scatter fusion step.

---

## Scatter Fusion (`tt/fusion.py` and `reference/fusion.py`)

### Mechanism: `merge_vision_tokens()`

The scatter fusion operation inserts vision feature tokens into the combined (vision + text) embedding sequence at the exact positions where the tokenizer placed `image_token_id=151665` placeholder tokens.

The input to the fusion step is:

- A text embedding sequence of shape `[B, S_total, 1536]`, where `S_total` includes both real text token positions and `image_token_id` placeholder positions.
- A vision token tensor of shape `[B, S_img, 1536]` from `PatchMergerTT`.

The fusion operation identifies all positions $i$ in the sequence where `token_id[i] == 151665` and overwrites the embedding at position $i$ with the corresponding vision token. After fusion, the output sequence has shape `[B, S_total, 1536]` with every placeholder replaced by a vision feature vector.

### TTNN Index-Scatter

In `tt/fusion.py`, this is implemented as an index-scatter operation on TTNN tensors:

```python
# positions: 1-D index tensor of image_token_id positions in [0, S_total)
# vision_tokens: [B, S_img, 1536] from PatchMergerTT
# embeddings: [B, S_total, 1536] text embedding sequence (on device)

fused = ttnn.scatter(embeddings, positions, vision_tokens, dim=1)
```

The positions tensor is computed on the host from the token ID sequence (a CPU operation over small integer tensors) and then transferred to the device before the scatter op.

> **Warning:** The number of `image_token_id=151665` entries in the token sequence must exactly equal `S_img` — the number of tokens output by `PatchMergerTT`. A mismatch means either the image was preprocessed at a different resolution than the tokenizer expected, or the `grid_thw` passed to `PatchEmbedTT` was inconsistent with the actual pixel data. The error manifests at fusion time as a shape assertion failure, not at vision encoding time. Running `test_fusion.py` before `test_e2e_pcc.py` isolates this class of error.

### Reference Implementation (`reference/fusion.py`)

`reference/fusion.py` implements the same scatter fusion in pure PyTorch:

```python
# Equivalent reference logic
fused = embeddings.clone()
fused[token_ids == 151665] = vision_tokens.reshape(-1, hidden_size)
```

This is the correctness oracle for `test_fusion.py`. The TTNN scatter in `tt/fusion.py` and the PyTorch masked assignment in `reference/fusion.py` must produce identical outputs (within floating-point tolerance). The PCC between the two is expected to exceed 0.99; the scatter operation itself is exact for non-quantized BFLOAT16 tensors, so any PCC degradation indicates incorrect position indexing rather than numerical accumulation.

### Comparison with Qwen 2.5 VL Fusion

The fusion approach in dots.ocr is the same conceptual pattern as Qwen 2.5 VL: image token ID placeholders in the token sequence are replaced by vision features from the encoder. The implementation differences are:

| Aspect | dots.ocr | Qwen 2.5 VL |
|--------|----------|-------------|
| `image_token_id` | 151665 | 151655 |
| Vision token shape entering fusion | `[B, S_img, 1536]` | `[B, S_img, 1280]` (Qwen2.5-VL-7B) |

The `image_token_id` difference is critical (see `relationship_to_qwen25vl.md` in Chapter 1): hardcoding 151655 from the Qwen2.5-VL codebase when processing dots.ocr inputs causes the scatter to target the wrong positions, inserting vision tokens at text positions and leaving the actual image placeholders as embedding-table vectors.

Because the fusion logic is structurally identical, any future tt_symbiote dispatch layer that handles image token replacement can use the same pattern for both dots.ocr and Qwen 2.5 VL, parameterized only by `image_token_id` and the vision token dimension.

---

**Next:** [Chapter 4 — T3K Topology and GQA Constraint](../ch4_t3k_topology_and_gqa_constraint/index.md)
