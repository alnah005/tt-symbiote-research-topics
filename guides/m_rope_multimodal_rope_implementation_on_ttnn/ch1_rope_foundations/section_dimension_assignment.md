# Section-to-Dimension Assignment in M-RoPE

This file derives exactly which frequency pairs and real vector dimensions
belong to each M-RoPE section, works through the complete Qwen3.6 dimension
map, explains why sections are contiguous in the cos/sin table but do not
correspond to contiguous slices of the weight tensor, and specifies the shape
of the assembled effective cos/sin tensor.

## Deriving Section Boundaries from `mrope_section`

### The Partition

Given `mrope_section = [s_t, s_h, s_w]` with `s_t + s_h + s_w = rotary_dim / 2`,
the three sections partition the `rotary_dim / 2` rotation pairs of the cos/sin
table into three contiguous ranges:

```
Temporal section:  pair indices [0,             s_t)
Height section:    pair indices [s_t,           s_t + s_h)
Width section:     pair indices [s_t + s_h,     s_t + s_h + s_w)  = [s_t+s_h, rotary_dim/2)
```

Each pair index `i` maps to two real dimensions in the cos/sin table: `i` and
`i + rotary_dim/2` (because of the rotate-half doubling from
[`standard_rope_recap.md`](./standard_rope_recap.md)). In the assembled
cos/sin vector — the one actually multiplied against `x` in the rotate-half
formula — the mapping from pair to real dimensions is:

```
pair i  →  cos/sin column i  and  cos/sin column (i + rotary_dim/2)
```

The full contiguous real-dimension ranges for each section in the assembled
vector are therefore:

```
Temporal:  columns [0,        s_t)         and  [rotary_dim/2,                  rotary_dim/2 + s_t)
Height:    columns [s_t,      s_t+s_h)     and  [rotary_dim/2 + s_t,            rotary_dim/2 + s_t + s_h)
Width:     columns [s_t+s_h,  rotary_dim/2) and  [rotary_dim/2 + s_t + s_h,     rotary_dim)
```

However, in the HuggingFace reference implementation, the assembled cos/sin
tensor is constructed differently: the three section slices are assembled from
the *non-doubled* `freqs` table (shape `[seq_len, rotary_dim/2]`) rather than
from the doubled `emb` table. The doubled cos/sin vector is then reconstructed
by repeating the assembled half-length vector. This is equivalent but requires
careful attention to which convention is in use. Section 3 below shows the
concrete HuggingFace assembly procedure.

## Concrete Example: Qwen3.6-35B-A3B

### Parameter Values

```
rope_theta            = 1,000,000
head_dim              = 128
partial_rotary_factor = 0.5
rotary_dim            = 64
rotary_dim / 2        = 32 pairs
mrope_section         = [11, 11, 10]     (s_t=11, s_h=11, s_w=10)
```

Sanity check: `11 + 11 + 10 = 32 = rotary_dim / 2`. ✓

### Pair-Level Section Boundaries

| Section | Pair range (half-open) | Number of pairs |
|---|---|---|
| Temporal | `[0, 11)` | 11 |
| Height | `[11, 22)` | 11 |
| Width | `[22, 32)` | 10 |

### Real Dimension Ranges in the Assembled Cos/Sin Vector

```
assembled_cos = [cos(t·θ_{0..10}),        # temporal, 11 values  → columns 0-10
                 cos(h·θ_{11..21}),        # height,   11 values  → columns 11-21
                 cos(w·θ_{22..31}),        # width,    10 values  → columns 22-31
                 cos(t·θ_{0..10}),         # temporal repeated    → columns 32-42
                 cos(h·θ_{11..21}),        # height repeated      → columns 43-53
                 cos(w·θ_{22..31})]        # width repeated       → columns 54-63
```

### Full Dimension Map: Columns 0–63

The table below maps every column in the assembled cos/sin vector to its section
and pair index, for the Qwen3.6 configuration.

| Column range | Half-offset mirror | Section | Pairs covered | Position coord |
|---|---|---|---|---|
| `[0, 11)` cols 0–10 | `[32, 43)` cols 32–42 | Temporal | pairs 0–10 | `t` |
| `[11, 22)` cols 11–21 | `[43, 54)` cols 43–53 | Height | pairs 11–21 | `h` |
| `[22, 32)` cols 22–31 | `[54, 64)` cols 54–63 | Width | pairs 22–31 | `w` |

The rotate-half operation then uses:
- Column `c` and column `c + 32` together for the rotation of output dimension `c`
- For `c ∈ [0, 11)`: both column `c` and column `c+32` are temporal → uses `t`
- For `c ∈ [11, 22)`: column `c` (height) and column `c+32` (height) → uses `h`
- For `c ∈ [22, 32)`: column `c` (width) and column `c+32` (width) → uses `w`

Every rotation pair is entirely within one section — no pair straddles a section
boundary. This is guaranteed by the construction: `s_t`, `s_h`, `s_w` are
defined in terms of pairs, so section boundaries always fall on pair boundaries.

## Why Sections Are Contiguous in the Table But Not in the Weight Tensor

### Contiguous in the Cos/Sin Table

The cos/sin table is indexed by a position scalar → it is a lookup from
"position coordinate" to "rotation values". The section boundaries define which
rows of the table (which position coordinate values) are used for which columns.
Within the assembled cos/sin vector for one token, the three sections occupy
pair-based column ranges that appear in both the first half and the second half
of the `rotary_dim`-wide vector:

See the full dimension map table above for concrete column ranges.

In the HuggingFace assembly convention, the gather step operates on the
half-length frequency table (slicing widths `s_t`, `s_h`, `s_w`), and the
full doubled vector is reconstructed by repeating the assembled half.

### Not Contiguous in the Weight Tensor

The Q, K, and V weight matrices `[hidden_size, head_dim]` have no knowledge of
M-RoPE section structure. The 128 (or 64, for partial RoPE) columns of a query
or key head vector are organized by the network's learned representations, not
by the rotary coordinate axes. The M-RoPE section partition only governs how the
*positional encoding* is assembled; the weight tensor layout is unchanged from
standard RoPE.

> **Key Finding:** M-RoPE requires no changes to Q, K, V weight tensors or to
> any linear projection. All changes are confined to the cos/sin assembly step:
> three separate position coordinate values are used to index three contiguous
> column ranges of the same frequency table, and the results are concatenated to
> form the assembled rotation vector.

This means a TTNN M-RoPE implementation can reuse the existing frequency table
(constructed identically to standard partial RoPE) and the existing rotate-half
kernel. The only new operations required are: (1) three indexed lookups using
different rows of the position ID tensor, and (2) concatenation of the three
resulting section vectors.

## Shape of the Effective Cos/Sin Tensor

### Per-Token (Scalar Position Case)

For a single token at coordinates `(t, h, w)`, the assembled cos or sin vector
has shape `[rotary_dim]`. For the Qwen3.6 example: shape `[64]`.

### Per-Sequence, Per-Batch

At inference over a sequence of length `S` in a batch of size `B`, the
assembled cos/sin tensors have shape `[B, S, rotary_dim]`.

The construction for the cos tensor is:

```python
# position_ids: [3, B, S]  (int32 or int64)
# cos_table:    [max_seq_len, rotary_dim//2]   (same table as standard partial RoPE)

s_t, s_h, s_w = mrope_section          # [11, 11, 10] for Qwen3.6

# Three separate row-gather operations from the same frequency table:
cos_t = cos_table[position_ids[0]][:, :, :s_t]           # [B, S, 11]  temporal
cos_h = cos_table[position_ids[1]][:, :, s_t:s_t+s_h]    # [B, S, 11]  height
cos_w = cos_table[position_ids[2]][:, :, s_t+s_h:]       # [B, S, 10]  width

# Assemble the first rotary_dim/2 columns, then duplicate for rotate-half:
cos_half = torch.cat([cos_t, cos_h, cos_w], dim=-1)      # [B, S, 32]
cos_full = torch.cat([cos_half, cos_half], dim=-1)        # [B, S, 64]

# Identically for sin_table → sin_full: [B, S, 64]
```

The `cos_full` tensor of shape `[B, S, 64]` is then applied to the first 64
dimensions of each Q and K head vector via the standard rotate-half operation,
and the remaining 64 dimensions of the 128-dimensional head are concatenated
unchanged — exactly as in standard partial RoPE.

> **[SILENT FAILURE]** If the three gather steps accidentally use the same
> position coordinate (e.g., all three use `position_ids[0]`) rather than
> `position_ids[0]`, `position_ids[1]`, `position_ids[2]` respectively, the
> result is standard 1D RoPE with all sections using the temporal coordinate.
> For text-only inputs this produces the correct output (since all three rows
> are equal), but for image or video inputs the height and width spatial
> structure is completely discarded. This class of bug is invisible in
> text-only benchmarks and only surfaces during vision evaluation.

## Section Width Asymmetry: Why `s_w = 10` While `s_t = s_h = 11`

The Qwen3.6 configuration uses `[11, 11, 10]` rather than a perfectly even
split like `[11, 11, 11]` (which would require `rotary_dim/2 = 33` — an odd
number). Since `rotary_dim = 64` gives exactly 32 pairs, the three sections
cannot be exactly equal. The choice to give the width section one fewer pair
than temporal and height is a pragmatic implementation decision. The asymmetry
is small (10 vs. 11 pairs = 20 vs. 22 real dimensions) and has negligible
effect on the model's positional encoding capacity.

> **Key Finding:** The `mrope_section` values `[11, 11, 10]` are not
> hyperparameters that can be freely changed after training. They are baked into
> the frequency table indexing used during both pre-training and fine-tuning. A
> TTNN implementation must read these values from `config.rope_scaling.mrope_section`
> and use them exactly as specified — they are part of the model checkpoint's
> implicit positional encoding contract.

## Cross-Reference: How This Section Map Is Used in TTNN

[Chapter 4 — TTNN Implementation](../ch4_ttnn_implementation/extension_approach.md) takes the section
map derived here and translates it into concrete TTNN operations:

- The three gather steps correspond to three `ttnn.embedding` calls, each using
  a different row of the `[3, batch, seq_len]` position ID tensor as the index.
- The concatenation step corresponds to one `ttnn.concat` call along the
  dimension axis.
- The rotate-half multiply-add is unchanged from the existing
  `TTNNRotaryPositionEmbedding` implementation.

[Chapter 2 — HuggingFace Reference](../ch2_qwen36_mrope_config/hf_reference_implementation.md) shows
the corresponding HuggingFace operations in `apply_multimodal_rotary_pos_emb`,
which serves as the numerical reference for validating the TTNN implementation.

---

**Next:** [Chapter 2 --- M-RoPE in Qwen3.6-35B-A3B: Configuration and Reference Implementation](../ch2_qwen36_mrope_config/index.md)
