# M-RoPE: Motivation and Design

This file explains why standard 1D RoPE is structurally insufficient for
multimodal sequences, derives M-RoPE's three-coordinate solution, and defines
the `mrope_section` partition and the 3D position ID tensor.

## The Core Problem: 1D Position Indexing for Multimodal Tokens

Standard 1D RoPE assigns each token a single integer position `t` and encodes it in the query/key vector by rotating dimension pairs at frequencies `θ_0, θ_1, …`. The attention inner product depends only on the relative position `t_q − t_k` — for purely sequential text this is sufficient, but multimodal tokens have richer geometric structure that a single scalar cannot capture.

### Where 1D Indexing Breaks Down

Vision-language models embed image and video tokens into the same sequence as
text tokens. Consider a short video clip encoded as a grid of patch tokens:
frame `f`, row `r`, column `c`. Under 1D RoPE, each such token receives a flat
integer index `t_flat` that encodes only its position in the flattened sequence,
not its true spatial or temporal coordinates.

This creates three compounding problems:

1. **Temporal confusion.** Two patches at the same spatial location `(r, c)`
   but different frames `f` and `f'` receive different flat indices `t_flat`
   and `t_flat'`. The angular difference `t_flat' - t_flat` equals the number
   of patches between them in the flat layout, which depends on the grid size
   and frame ordering — not on the actual temporal distance `f' - f`. The model
   cannot directly learn "these two patches are at the same position but one
   frame apart."

2. **Spatial axis confusion.** A patch at `(r+1, c)` (one row down) and a
   patch at `(r, c+1)` (one column right) are both exactly one 1D-index step
   away from `(r, c)`. Standard RoPE encodes them as equally close to the
   anchor, even though they lie in orthogonal spatial directions.

3. **Text-image boundary confusion.** After a block of `N_image` image patches,
   the text token that follows receives index `t_text_start + N_image`. But the
   semantically meaningful distance between that text token and any given image
   patch is not `t_text_start + N_image - t_patch`; it has no clear geometric
   meaning in the image domain.

In summary, a single flat position integer cannot simultaneously carry temporal,
spatial-height, and spatial-width structure. Whatever embedding is chosen for
the flat index, at least one of the three geometric relationships is distorted.

## M-RoPE's Solution: A Position Triplet Per Token

### The Three-Coordinate Design

M-RoPE (introduced in Qwen2-VL; see Wang et al. 2024, "Qwen2-VL: Enhancing
Vision-Language Model's Perception of the World at Any Resolution") assigns
each token a **triplet** of integer position coordinates:

```
(t, h, w)
```

- `t` — **temporal** coordinate: the frame index for video tokens, the image
  index for image tokens, or the sequential text position for text tokens.
- `h` — **height** coordinate: the patch row within the spatial grid, or the
  sequential text position for text tokens.
- `w` — **width** coordinate: the patch column within the spatial grid, or the
  sequential text position for text tokens.

Each coordinate is encoded independently in a **dedicated sub-group of rotary
dimensions**. The attention inner product between two tokens then decomposes
into three independent rotary phase terms — one per coordinate — allowing the
model to separately attend to temporal proximity, vertical spatial proximity,
and horizontal spatial proximity.

### Encoding Each Coordinate in a Dedicated Section

The `rotary_dim` rotation pairs are partitioned into three contiguous sections
governed by `mrope_section = [s_t, s_h, s_w]`:

```math
s_t + s_h + s_w = \frac{\texttt{rotary\_dim}}{2}
```

Each section is a contiguous range of pairs in the cos/sin table:

```
Temporal section:  pairs [0,        s_t)         ← indexed by coordinate t
Height section:    pairs [s_t,      s_t+s_h)     ← indexed by coordinate h
Width section:     pairs [s_t+s_h,  rotary_dim/2) ← indexed by coordinate w
```

The apply step gathers three section slices from the cos/sin table — one lookup per coordinate `t`, `h`, `w` — assembles them into a half-length vector of width `rotary_dim//2`, then duplicates to produce the full `rotary_dim`-wide cos/sin vector. The rotate-half operation then applies exactly as in standard RoPE. See [`section_dimension_assignment.md`](./section_dimension_assignment.md) for the full derivation, dimension map, and Python reference implementation.

### Qwen3.6 Section Partition

For Qwen3.6-35B-A3B:

```
rotary_dim / 2 = 32 pairs
mrope_section  = [11, 11, 10]   (from config.rope_scaling.mrope_section)
s_t + s_h + s_w = 11 + 11 + 10 = 32  ✓ consistent with rotary_dim=64
```

The three sections therefore cover:

| Section | Pair range | Real dimension range | Coordinate |
|---|---|---|---|
| Temporal | `[0, 11)` | `[0,11) ∪ [32,43)` | `t` |
| Height | `[11, 22)` | `[11,22) ∪ [43,54)` | `h` |
| Width | `[22, 32)` | `[22,32) ∪ [54,64)` | `w` |

The section widths are nearly equal (11, 11, 10) — a design choice reflecting
that temporal, height, and width are treated as approximately equally important
positional axes with similar expected coordinate ranges.

## The 3D Position ID Tensor

### Shape and Semantics

Instead of a 1D tensor `[batch, seq_len]` carrying a single integer per token,
M-RoPE requires a 3D tensor:

```
position_ids: [3, batch, seq_len]   dtype: int32 or int64
```

Axis 0 indexes the coordinate:
- `position_ids[0]` — temporal coordinates `t` for every token in the batch
- `position_ids[1]` — height coordinates `h` for every token in the batch
- `position_ids[2]` — width coordinates `w` for every token in the batch

For a batch of size 1 and sequence length `S`, the tensor has shape `[3, 1, S]`
and contains three integer sequences of length `S`.

### Position ID Values by Modality

**Text tokens:** All three coordinates are set to the sequential text position
`p ∈ {0, 1, …}`. For a text-only sequence of length `S`:

```python
position_ids = torch.arange(S).unsqueeze(0).expand(3, 1, S)
# position_ids[0, 0, :] = [0, 1, 2, ..., S-1]  (temporal)
# position_ids[1, 0, :] = [0, 1, 2, ..., S-1]  (height)
# position_ids[2, 0, :] = [0, 1, 2, ..., S-1]  (width)
```

**Image tokens:** An image patch at grid position `(i_h, i_w)` in a grid of
`(num_patches_h, num_patches_w)` receives:

```
t = image_index  (constant across all patches in the same image)
h = i_h + offset (where offset accounts for preceding text or images)
w = i_w + offset
```

**Video tokens:** A video patch at frame `f`, spatial position `(i_h, i_w)`:

```
t = f + temporal_offset  (frame index, increments per frame)
h = i_h + offset
w = i_w + offset
```

The concrete construction of position IDs for mixed text+image sequences is
covered in full in Chapter 2
([`../ch2_qwen36_mrope_config/position_id_construction.md`](../ch2_qwen36_mrope_config/position_id_construction.md)).

## M-RoPE Degenerating to Standard RoPE for Text

When all three position ID rows are identical — which is always the case for
text-only inputs — the M-RoPE assembly reduces to:

```
cos_assembled[t_text] = [cos_table[t][0:s_t]  || cos_table[t][s_t:s_t+s_h] || cos_table[t][s_t+s_h:rotary_dim//2]]
                       = cos_table[t][0:rotary_dim//2]   (all from the same row t)
```

This is identical to indexing the full cos/sin row at position `t` — exactly
what standard partial RoPE does. The section partition has no effect on the
output values when all coordinates are equal, because the same table row is
assembled regardless of where the section boundaries fall.

This property is the foundation of the text-only reduction analysis in Chapter 3
([`../ch3_text_only_reduction/mathematical_equivalence_proof.md`](../ch3_text_only_reduction/mathematical_equivalence_proof.md)).

> **Key Finding:** M-RoPE with equal position IDs across all three axes (`t = h = w`)
> is mathematically equivalent to standard 1D partial RoPE with the same
> `rope_theta` and `rotary_dim`. The `mrope_section` partition is irrelevant
> when all coordinates are the same integer — the assembled cos/sin vector is
> simply a contiguous row of the frequency table, which is the standard RoPE
> result.

> **[SILENT FAILURE]** If an M-RoPE implementation mistakenly uses `h=0` or
> `w=0` for text tokens (instead of copying the sequential text position into
> all three rows), the height and width sections will always index row 0 of the
> cos/sin table (the zero-position embedding). All text tokens will carry the
> same height/width rotary phase — equivalent to having no positional encoding
> in those dimensions. The model will not raise an error, but its ability to
> distinguish positions within text sequences will be severely degraded for
> the height and width rotary dimensions. This bug is particularly hard to
> detect from loss alone, since text attention can partially compensate via
> the temporal section.

## Comparison: Standard RoPE vs. M-RoPE

| Property | Standard RoPE | M-RoPE |
|---|---|---|
| Coordinates per token | 1 scalar `t` | Triplet `(t, h, w)` |
| Position ID tensor shape | `[batch, seq_len]` | `[3, batch, seq_len]` |
| Cos/sin table shape | `[max_seq_len, rotary_dim]` | Same: `[max_seq_len, rotary_dim]` |
| Table indexing | Single row lookup: `table[t]` | Three partial row lookups assembled from `t`, `h`, `w` |
| Modalities supported | Text only (1D sequence) | Text, image, video with independent spatial/temporal axes |
| Rotation arithmetic | Standard rotate-half | Identical rotate-half, applied to assembled cos/sin |
| Text-only behavior | Native | Degenerate case: identical to standard RoPE |
| Config field | `rope_theta`, `rotary_dim` | Same, plus `mrope_section = [s_t, s_h, s_w]` |

---

**Next:** [`section_dimension_assignment.md`](./section_dimension_assignment.md)
