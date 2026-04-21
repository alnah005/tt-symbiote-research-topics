# Position ID Construction for M-RoPE

## 1. Overview

The 3D position ID tensor of shape `[3, batch, seq_len]` is the key input that distinguishes M-RoPE from standard 1D RoPE. Each of the three rows carries a different spatial coordinate for every token:

- Row 0 — **temporal** coordinate $t$: identifies the frame (for video) or image index
- Row 1 — **height** coordinate $h$: identifies the patch row within a frame
- Row 2 — **width** coordinate $w$: identifies the patch column within a frame

For text tokens, all three coordinates are identical. For vision tokens, they differ. This file explains how the tensor is built for three input types: text-only, text + single image, and text + video.

---

## 2. Text-Only Construction

For a purely text sequence of length $S$:

```python
position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).repeat(3, 1)
# Result: shape [3, 1, S] (batch=1), all three rows identical
# Row 0 (temporal):  [0, 1, 2, ..., S-1]
# Row 1 (height):    [0, 1, 2, ..., S-1]
# Row 2 (width):     [0, 1, 2, ..., S-1]
```

This is the degenerate case: all three rows are equal, so the three gather operations in `Qwen2_5_VLRotaryEmbedding.forward()` produce identical column slices from the frequency table. The assembled cos/sin is numerically equal to standard 1D RoPE. Chapter 3 proves this formally.

---

## 3. Text + Single Image Construction

Consider a sequence with:
- `n_text_pre` prefix text tokens
- `H × W` image patch tokens (H patch rows, W patch columns)
- `n_text_post` suffix text tokens

Total sequence length: `n_text_pre + H * W + n_text_post`.

```python
# Image grid: H rows x W columns of patches
# (H and W are in patch units, not pixels)

# --- Prefix text tokens ---
# All three coordinates equal the sequential token position.
text_pre_pos = torch.arange(n_text_pre)

# --- Image patch tokens ---
# Temporal: constant (image index 0 for the first image in the sequence)
t_image = torch.full((H * W,), n_text_pre)           # all patches: same frame value

# Height: row index of each patch (h in [0, H)), broadcast across W columns
h_grid  = torch.arange(H).repeat_interleave(W)        # [H*W]: 0,0,...,1,1,...,H-1,...
h_image = h_grid + n_text_pre                          # offset by prefix text length

# Width: column index of each patch (w in [0, W)), repeated H times
w_grid  = torch.arange(W).repeat(H)                   # [H*W]: 0,1,...,W-1,0,1,...
w_image = w_grid + n_text_pre                          # offset by prefix text length

# --- Suffix text tokens ---
# Continue from max(t, h, w) of image tokens + 1.
# In practice: starts from n_text_pre + max(H, W).
post_start    = n_text_pre + max(H, W)
text_post_pos = torch.arange(n_text_post) + post_start

# --- Assemble each row ---
row_t = torch.cat([text_pre_pos, t_image,  text_post_pos])
row_h = torch.cat([text_pre_pos, h_image,  text_post_pos])
row_w = torch.cat([text_pre_pos, w_image,  text_post_pos])

position_ids = torch.stack([row_t, row_h, row_w], dim=0).unsqueeze(1)
# Shape: [3, 1, n_text_pre + H*W + n_text_post]
```

**Key points:**

- Text tokens always have identical values across all three rows.
- Image patch positions carry three distinct coordinates: temporal is a constant per image, while height and width encode the spatial grid location of each patch.
- Suffix text positions continue from `n_text_pre + max(H, W)` so that post-image text position IDs do not collide with image position IDs in any coordinate.

---

## 4. Text + Video Construction

For video input with multiple frames, the temporal coordinate increments per frame while the height and width coordinates repeat the same spatial grid for every frame.

For frame $f$ out of $F$ total frames, where each frame has `H × W` spatial patches:

```python
# frame_offset: the starting temporal value for this video
# spatial_offset: the starting height/width value for this video's patches

t_frame_f = torch.full((H * W,), frame_offset + f)          # frame index f for all patches
h_frame_f = torch.arange(H).repeat_interleave(W) + spatial_offset   # patch row
w_frame_f = torch.arange(W).repeat(H) + spatial_offset              # patch column
```

Concatenating across all $F$ frames:

```python
t_video = torch.cat([torch.full((H * W,), frame_offset + f) for f in range(F)])
h_video = torch.arange(H).repeat_interleave(W).repeat(F) + spatial_offset
w_video = torch.arange(W).repeat(H).repeat(F) + spatial_offset
# Each has shape [F * H * W]
```

The temporal coordinate uniquely identifies each frame. The height and width coordinates repeat the same `H × W` grid for every frame, giving the model spatial position within a frame independent of which frame it is.

---

## 6. Forward References

- The position ID construction above follows the HuggingFace reference described in [`hf_reference_implementation.md`](./hf_reference_implementation.md).
- Chapter 6 (`../ch6_integration_and_testing/integration_steps.md`) implements this construction in tt-symbiote. The existing `TTNNRotaryPositionEmbedding` text-only path does **not** need modification for text-only Qwen3.6 inference (see Chapter 6, [`integration_steps.md`](../ch6_integration_and_testing/integration_steps.md)).

---

**Next:** [Chapter 3 — Text-Only Behavior: Does M-RoPE Reduce to Standard RoPE?](../ch3_text_only_reduction/index.md)
