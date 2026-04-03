# Chapter 4 Change Log --- Feedback Round 1

## Summary

Applied 5 feedback items from Agent B's review. All changes correct factual
errors in the mathematical formulations, code examples, and architectural
descriptions. No structural or navigation changes were needed.

---

## Issue 1: inv_freq denominator corrected (head_dim -> dim)

**Error:** The inv_freq formula used `head_dim=512` as the denominator in the
exponent. The correct denominator is `dim = int(head_dim * partial_rotary_factor)
= int(512 * 0.25) = 128`.

**Corrected formula:**
`inv_freq[i] = 1 / (1,000,000^{2i / 128})` for i = 0..63

**Files changed:**
- `global_proportional_rope.md`: Updated mathematical formulation section and
  both Strategy A/B code examples.
- `rope_precomputation.md`: Updated Strategy A and Strategy B code examples,
  added comments clarifying the denominator.

---

## Issue 2: Zero-padding attribution corrected

**Error:** The chapter described a `_compute_proportional_rope_parameters`
function that zero-pads inv_freq to full width, and attributed this to
HuggingFace's reference implementation. No such zero-padding function exists
in HuggingFace. The actual HF behavior: inv_freq has 64 elements, cos/sin
tables are width 128, and `apply_rotary_pos_emb` handles partial rotation via
split-apply-concat.

**Fix:** Reframed Strategy A (full-width tables) as a valid TTNN optimization
approach, NOT as existing HuggingFace behavior. Reframed Strategy B (narrow
128-wide tables with split-apply-concat) as the HuggingFace reference
behavior. Updated the bringup recommendation to start with Strategy B.

**Files changed:**
- `index.md`: Rewrote the "Note on cos/sin table shape" section. Updated
  reading order description.
- `global_proportional_rope.md`: Replaced "Inverse Frequencies With Zero
  Padding" section with correct "Inverse Frequencies" section describing the
  reference split-apply-concat pattern. Relabeled Strategy A as "TTNN
  Optimization" and Strategy B as "HuggingFace Reference". Updated
  recommendation section.
- `rope_precomputation.md`: Updated table set description, relabeled Strategy
  A/B headers, updated summary table.

---

## Issue 3: Table shapes corrected in index.md

**Error:** The quick reference table listed global p-RoPE inv_freq length=256
and cos/sin shape=[max_seq_len, 512].

**Corrected values:** inv_freq length=64, cos/sin shape=[max_seq_len, 128].

**Files changed:**
- `index.md`: Updated the quick reference table row for global p-RoPE.

---

## Issue 4: Wavelength calculation corrected

**Error:** With the incorrect d=512, the highest-frequency pair (i=63) was
calculated as having wavelength ~189 tokens. With the correct d=128, the
wavelength is ~5.1M tokens.

**Corrected calculation:** wavelength of highest-freq pair (i=63) =
2*pi * (10^6)^{126/128} ~= 5.1M tokens. ALL pairs have very long wavelengths,
which is the intended behavior for long-context extrapolation.

**Files changed:**
- `global_proportional_rope.md`: Rewrote the wavelength example with correct
  d=128 and updated the interpretive text.

---

## Issue 5: unsqueeze_dim corrected in sliding_rope.md

**Error:** Reference code showed `unsqueeze_dim=2` with RoPE applied BEFORE
transpose. HuggingFace defaults to `unsqueeze_dim=1` with RoPE applied AFTER
transpose (tensor layout is `[B, H, S, D]` at point of RoPE application).

**Files changed:**
- `sliding_rope.md`: Reordered the forward pass diagram to show transpose
  before RoPE. Updated reference code to show `unsqueeze_dim=1` and
  transpose before `apply_rotary_pos_emb`. Updated explanatory text.
- `global_proportional_rope.md`: Reordered the forward pass diagram to show
  transpose before RoPE, consistent with the sliding layer fix.

---

# Chapter 4 Change Log --- Feedback Round 2

## Summary

Fixed 2 residual tensor layout typos where shape comments still used the
pre-transpose `[B, S, H, D]` layout instead of the correct post-transpose
`[B, H, S, D]` layout. These were missed during the Round 1 unsqueeze_dim
correction (Issue 5).

---

## Issue 6: Shape comment in sliding_rope.md line 112

**Error:** The inline shape comment described the full-rotation tensor as
`[B, S, H, 256]`. RoPE is applied after the transpose to `[B, H, S, D]`
layout (as correctly stated on lines 105-106 of the same file), so the shape
should be `[B, H, S, 256]`.

**Files changed:**
- `sliding_rope.md`: Changed `[B, S, H, 256]` to `[B, H, S, 256]` in the
  "All 256 Dimensions Are Rotated" section.

---

## Issue 7: Shape comments in global_proportional_rope.md Strategy B code block

**Error:** Six shape comments on lines 232-243 used `[B, S, H, D]` layout
(e.g. `[B, S, 32, 128]`). Since RoPE is applied after transpose, the correct
layout is `[B, H, S, D]` (e.g. `[B, H, S, 128]`).

**Files changed:**
- `global_proportional_rope.md`: Updated all six shape comments in the
  Strategy B forward-pass code block from `[B, S, ...]` to `[B, H, ...]`.

---

# Chapter 4 Compression Analysis --- Agent C

## Summary

The four files in ch4_dual_rope are well-structured and technically dense. The
primary redundancy pattern is repeated restatement of the same facts across
files: the dual-RoPE parameters (theta, partial_rotary_factor, head_dim,
table shapes) and the Strategy A vs Strategy B tradeoffs are restated in full
in every file that touches them. There is also some intra-file redundancy
where explanatory prose restates what a code block or table already shows.
Overall the content is load-bearing and the redundancy is modest --- no
crucial compression is needed.

---

## File 1: `index.md` (84 lines)

### Crucial updates: no

### Load-Bearing Evidence

Lines 26-41 (the Quick Reference table) provide the only consolidated
side-by-side parameter comparison across both RoPE variants. This table is
the single most efficient summary in the chapter. Removing or compressing
it would force readers to reconstruct the comparison from two separate files.

### MINOR Suggestions

1. **Lines 11-18 (prose before table):** The paragraphs describing sliding
   and global RoPE parameters duplicate the Quick Reference table on lines
   28-41 almost verbatim (theta values, head_dim, rotary dims, partial
   factor). The prose could be reduced to one sentence per variant with a
   forward reference: "See the Quick Reference table below for all
   parameters." Estimated savings: ~8 lines.

2. **Lines 43-55 (Note on cos/sin table shape):** This note restates the
   Strategy A/B distinction that is covered comprehensively in
   `global_proportional_rope.md`. It could be shortened to 2-3 sentences
   with a pointer to the subpage. Estimated savings: ~6 lines.

---

## File 2: `sliding_rope.md` (223 lines)

### Crucial updates: no

### Load-Bearing Evidence

Lines 129-141 (cos/sin table precomputation pseudocode) provide the
concrete implementation reference for sliding-layer table construction.
This code block is the authoritative source that `rope_precomputation.md`
refers back to (and duplicates). Removing it would break the self-contained
nature of this file.

### MINOR Suggestions

1. **Lines 129-141 vs rope_precomputation.md lines 26-37:** The sliding
   table precomputation code is duplicated nearly identically between these
   two files. `sliding_rope.md` could remove its copy and refer readers to
   `rope_precomputation.md`, or vice versa. Keeping one authoritative copy
   avoids future drift. Estimated savings: ~12 lines from one file.

2. **Lines 146-173 (Device Placement):** The replication rationale (256 MB
   per device at 256K, Option 1 vs Option 2) is repeated in
   `rope_precomputation.md` lines 134-153. One file should be the
   authority; the other should cross-reference. Estimated savings: ~15
   lines from one file.

3. **Lines 103-106:** The sentence "Note that RoPE is applied **after** the
   transpose, so the tensor layout is `[B, H, S, D]` at the point of
   application" restates what the diagram on lines 67-84 already shows.
   Minor, but it is repeated emphasis. Estimated savings: ~2 lines.

---

## File 3: `global_proportional_rope.md` (334 lines)

### Crucial updates: no

### Load-Bearing Evidence

Lines 44-62 (the split-apply-concat pattern explanation) are the
authoritative description of how partial rotation works in the HuggingFace
reference. This is referenced by both `index.md` and
`rope_precomputation.md` and cannot be shortened without losing clarity on
a non-obvious mechanism.

### MINOR Suggestions

1. **Lines 107-119 (Contrast With Sliding Layers table):** This table
   duplicates a subset of the Quick Reference table in `index.md` lines
   28-41. Since readers arrive here from `index.md`, this table could be
   replaced with a cross-reference. Estimated savings: ~12 lines.

2. **Lines 262-298 (Compatibility and Performance Impact sections):** The
   explanation that RoPE is a per-element operation requiring no
   cross-device communication is stated three times: once in the
   "Compatibility" section (line 269-278), once in "Performance Impact"
   (lines 287-292), and once at the end of "Performance Impact" (lines
   294-298 restating "practical impact: minimal"). These could be merged
   into a single paragraph. Estimated savings: ~10 lines.

3. **Lines 1-8 (opening paragraph):** Restates theta, partial_rotary_factor,
   128/512 split, and 256K extrapolation --- all covered in `index.md`.
   Could be shortened to one sentence since this is a subpage. Estimated
   savings: ~4 lines.

---

## File 4: `rope_precomputation.md` (257 lines)

### Crucial updates: no

### Load-Bearing Evidence

Lines 79-117 (Memory Footprint section with the two detailed tables) are
the only place in the chapter that quantifies DRAM cost at multiple context
lengths for both strategies. This data directly informs the Strategy A vs B
decision and cannot be compressed without losing actionable information.

### MINOR Suggestions

1. **Lines 26-76 (Sliding and Global table computation code blocks):** Both
   code blocks duplicate what already appears in `sliding_rope.md` lines
   129-141 and `global_proportional_rope.md` lines 186-196 / 218-224.
   This file could reference the source files and show only a summary or
   the differences. Estimated savings: ~30 lines.

2. **Lines 243-252 (Summary of Decisions table):** Several rows restate
   conclusions already made in the preceding sections of the same file
   (e.g., "Tables are static" repeats line 151; "Absolute positions"
   repeats lines 196-211). The table is useful as a quick reference, but
   the Rationale column is redundant with the prose. The Rationale column
   could be shortened to brief phrases. Estimated savings: ~3 lines.

---

## Cross-File Redundancy Summary

| Redundancy Pattern | Files Involved | Estimated Saveable Lines |
|--------------------|---------------|--------------------------|
| Sliding table precomputation code | sliding_rope.md, rope_precomputation.md | ~12 |
| Device placement / replication rationale | sliding_rope.md, rope_precomputation.md | ~15 |
| Strategy A vs B tradeoff prose | global_proportional_rope.md, rope_precomputation.md, index.md | ~10 |
| Sliding vs Global parameter comparison | index.md, global_proportional_rope.md | ~12 |
| RoPE-is-per-element-no-cross-device | sliding_rope.md, global_proportional_rope.md | ~10 |
| **Total** | | **~59 lines** |

---

## VERDICT

**No crucial compression needed.** The chapter has modest cross-file
redundancy (~59 lines across 898 total lines, ~6.6%) that is typical for
a multi-file guide where each subpage is designed to be partially
self-contained. The redundancy is annoying but not harmful --- it does not
obscure meaning or create maintenance hazards beyond minor drift risk for
duplicated code blocks. The MINOR suggestions above could tighten the
chapter without losing any information.
