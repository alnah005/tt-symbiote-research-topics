# Correctness Validation

## Overview

Four test cases cover the correctness surface of the M-RoPE implementation, ordered from simplest to most complex. All tests run on CPU in float32 first; Tests 2–4 are then re-run on the TTNN device against the CPU reference.

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| 1 | Text-only batch — M-RoPE degeneracy | `torch.equal` (exact identity) |
| 2 | Single image patch in a 3-token sequence | PCC > 0.9999 vs HF reference |
| 3 | Full image grid (32×32 = 1024 patches) | PCC > 0.9999 vs HF reference + per-section structural checks |
| 4 | Multi-frame video (4 frames × 16×16 patches) | PCC > 0.9999 vs HF reference + temporal axis checks |

> **Key Finding:** Test 1 uses `torch.equal` (not `assert_close`) because the mathematical proof in Ch3 (`../ch3_text_only_reduction/mathematical_equivalence_proof.md`) guarantees bitwise numerical identity for equal position IDs — not merely floating-point closeness. Any divergence in Test 1 indicates a logic error in the section gather, not a precision issue.

---

## Test Case 1 — Text-Only Batch (M-RoPE Degeneracy)

**Goal:** Confirm that the M-RoPE path with identical position IDs across all three axes produces NUMERICALLY IDENTICAL output to the existing standard partial RoPE path.

When all three axes carry the same sequential position IDs, the cos/sin gather for each axis returns the same rows of the table. The three section slices concatenate to the same full `rotary_dim` cos/sin vector that the standard 1D path would produce. No approximation is involved — the operations are algebraically identical.

```python
def test_text_only_mrope_identity(rope_module_standard, rope_module_mrope):
    """M-RoPE with equal position IDs must equal standard RoPE output exactly."""
    batch, seq_len, num_heads, head_dim = 1, 64, 64, 128
    rotary_dim = 64

    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)
    position_ids_1d = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
    position_ids_3d = position_ids_1d.unsqueeze(0).expand(3, -1, -1)  # [3, 1, seq_len]

    q_std, k_std = rope_module_standard.forward(q, k, position_ids_1d)
    q_mrp, k_mrp = rope_module_mrope.forward(q, k, position_ids_3d)

    # MUST be numerically identical (not just close) — Ch3 proved this algebraically
    assert torch.equal(q_std, q_mrp), "M-RoPE text-only diverges from standard RoPE"
    assert torch.equal(k_std, k_mrp), "M-RoPE text-only diverges from standard RoPE"
    print("Test 1 PASS: text-only M-RoPE is numerically identical to standard RoPE")
```

### What to check if Test 1 fails

| Symptom | Likely cause |
|---------|-------------|
| Values differ for all token positions | Section gather indexing is wrong (off-by-one in column slicing) |
| Values differ only for the last few rotary dimensions | `mrope_section` sum does not equal `rotary_dim // 2`; last section is truncated or padded |
| Values are close but not equal (`torch.allclose` passes, `torch.equal` fails) | The standard path uses a different dtype or op order; force both to float32 before comparison |

---

## Test Case 2 — Single Image Token

**Goal:** Confirm that a minimal batch with one text prefix token + one image patch + one text suffix token applies M-RoPE correctly to each token type.

```python
def test_single_image_patch(rope_module_mrope, hf_mrope_fn):
    """Single image patch: temporal/height/width rotations applied correctly."""
    # Build 3D position IDs for: [text_0, img_patch(t=0, h=1, w=1), text_2]
    # text_0:    t=0, h=0, w=0 (equal → identical to standard RoPE at pos 0)
    # img_patch: t=0, h=1, w=1 (temporal=frame 0, height row=1, width col=1)
    # text_2:    t=2, h=2, w=2 (equal → identical to standard RoPE at pos 2)
    position_ids_3d = torch.tensor([
        [[0, 0, 2]],   # temporal: text=0, image_frame=0, text=2
        [[0, 1, 2]],   # height:   text=0, image_row=1,   text=2
        [[0, 1, 2]],   # width:    text=0, image_col=1,   text=2
    ], dtype=torch.int32)  # [3, 1, 3]

    q = torch.randn(1, 64, 3, 128)
    k = torch.randn(1, 64, 3, 128)
    cos, sin = compute_mrope_cos_sin(position_ids_3d, mrope_section=[11, 11, 10])

    # Compare against HF apply_multimodal_rotary_pos_emb reference
    q_ttnn, k_ttnn = rope_module_mrope.forward(q, k, position_ids_3d)
    q_hf, k_hf = hf_mrope_fn(q, k, cos, sin, mrope_section=[11, 11, 10])
    assert compute_pcc(q_ttnn, q_hf) > 0.9999
    assert compute_pcc(k_ttnn, k_hf) > 0.9999
    print("Test 2 PASS: single image patch rotations correct")
```

### Structural checks for the image patch token (position index 1)

For `position_ids_3d[:, 0, 1] = [0, 1, 1]` (temporal=0, height=1, width=1):

- The temporal section (cos/sin pairs 0–10) should contain `cos_table[0, 0:11]` — same as a text token at position 0 for temporal pairs.
- The height section (cos/sin pairs 11–21) should contain `cos_table[1, 11:22]` — position 1 in the height dimension.
- The width section (cos/sin pairs 22–31) should contain `cos_table[1, 22:32]` — position 1 in the width dimension.

Add explicit spot-check assertions for the assembled cos/sin at token index 1 in addition to the end-to-end PCC check.

---

## Test Case 3 — Full Image Grid (32×32 = 1024 Patches)

**Goal:** Verify correct section-dimension assignment for a full image grid embedded in a text sequence.

```python
def test_full_image_grid(rope_module_mrope, hf_mrope_fn):
    """Full 32x32 image grid: verify per-axis cos/sin assignment and HF match."""
    num_h, num_w = 32, 32
    n_text_pre, n_text_post = 16, 16
    position_ids_3d = build_position_ids_3d(
        n_text_pre=n_text_pre,
        image_grid=(num_h, num_w),
        n_text_post=n_text_post,
        frame_idx=0,
    )  # [3, 1, 1056]

    seq_len = n_text_pre + num_h * num_w + n_text_post
    q = torch.randn(1, 64, seq_len, 128)
    k = torch.randn(1, 64, seq_len, 128)
    cos, sin = compute_mrope_cos_sin(position_ids_3d, mrope_section=[11, 11, 10])

    q_ttnn, k_ttnn = rope_module_mrope.forward(q, k, position_ids_3d)
    q_hf, k_hf = hf_mrope_fn(q, k, cos, sin, mrope_section=[11, 11, 10])

    assert compute_pcc(q_ttnn, q_hf) > 0.9999
    assert compute_pcc(k_ttnn, k_hf) > 0.9999
    print("Test 3 PASS: full 32x32 image grid PCC above threshold")
```

### Per-axis structural invariants to assert

These checks verify that the gather is assigning the right axis positions to the right rotary dimension slots — independent of the HF comparison:

**Temporal section invariant (pairs 0–10):** All 1024 image patches have `temporal_position = 0` (single frame, `frame_idx=0`). Therefore, all patches must have identical temporal cos/sin values.

```python
# Extract assembled cos tensor: [1, seq_len, rotary_dim]
cos_assembled = gather_cos_sections(cos_table, position_ids_3d, mrope_section=[11, 11, 10])

patch_start = n_text_pre
patch_end = n_text_pre + num_h * num_w
# Temporal section occupies the first 2*s_t=22 columns of the assembled cos
temporal_cos = cos_assembled[0, patch_start:patch_end, 0:22]  # [1024, 22]
# All rows should be identical (all patches at temporal position 0)
assert torch.all(temporal_cos == temporal_cos[0:1, :]), \
    "Temporal cos values differ across image patches — temporal gather is wrong"
```

**Height section invariant (pairs 11–21):** Patches in the same row have the same height position ID, so they must have identical height cos/sin values. Patches in different rows must have different height cos/sin values (since different rows index different table rows).

```python
height_cos = cos_assembled[0, patch_start:patch_end, 22:44]  # [1024, 22]
height_cos_2d = height_cos.reshape(num_h, num_w, 22)  # [32, 32, 22]
# Within each row (dim=1), all values should be equal
for row_i in range(num_h):
    assert torch.all(height_cos_2d[row_i] == height_cos_2d[row_i, 0:1, :]), \
        f"Row {row_i}: height cos values differ across columns — height gather wrong"
# Across rows, values should differ (position IDs are distinct)
assert not torch.all(height_cos_2d[0] == height_cos_2d[1]), \
    "Rows 0 and 1 have identical height cos — height position IDs not being used"
```

**Width section invariant (pairs 22–31):** Patches in the same column have the same width position ID.

```python
width_cos = cos_assembled[0, patch_start:patch_end, 44:64]  # [1024, 20]
width_cos_2d = width_cos.reshape(num_h, num_w, 20)  # [32, 32, 20]
for col_j in range(num_w):
    assert torch.all(width_cos_2d[:, col_j, :] == width_cos_2d[0:1, col_j, :]), \
        f"Col {col_j}: width cos values differ across rows — width gather wrong"
```

---

## Test Case 4 — Video Input (Multi-Frame)

**Goal:** Verify that the temporal position index increments correctly across frames while height/width positions repeat per-frame.

```python
def test_video_multiframe(rope_module_mrope, hf_mrope_fn):
    """4-frame video: temporal axis increments; height/width repeat per frame."""
    n_frames = 4
    num_h, num_w = 16, 16
    n_text_pre, n_text_post = 8, 8
    position_ids_3d = build_position_ids_3d_video(
        n_text_pre=n_text_pre,
        image_grid=(num_h, num_w),
        n_frames=n_frames,
        n_text_post=n_text_post,
    )  # [3, 1, seq_len]

    n_patches_per_frame = num_h * num_w
    seq_len = n_text_pre + n_frames * n_patches_per_frame + n_text_post
    q = torch.randn(1, 64, seq_len, 128)
    k = torch.randn(1, 64, seq_len, 128)
    cos, sin = compute_mrope_cos_sin(position_ids_3d, mrope_section=[11, 11, 10])

    q_ttnn, k_ttnn = rope_module_mrope.forward(q, k, position_ids_3d)
    q_hf, k_hf = hf_mrope_fn(q, k, cos, sin, mrope_section=[11, 11, 10])

    assert compute_pcc(q_ttnn, q_hf) > 0.9999
    assert compute_pcc(k_ttnn, k_hf) > 0.9999
    print("Test 4 PASS: multi-frame video PCC above threshold")
```

### Per-axis structural invariants for video

**Temporal invariant:** Within each frame `f`, all patches have `temporal_position = f`. Across frames, temporal position differs. Therefore:
- Within-frame temporal cos/sin values must be identical for all patches of that frame.
- Across-frame temporal cos/sin values must differ (since frames 0, 1, 2, 3 index different rows of the cos table).

```python
cos_assembled = gather_cos_sections(cos_table, position_ids_3d, mrope_section=[11, 11, 10])
patch_start = n_text_pre
for f in range(n_frames):
    f_start = patch_start + f * n_patches_per_frame
    f_end = f_start + n_patches_per_frame
    temporal_cos_f = cos_assembled[0, f_start:f_end, 0:22]  # [256, 22]
    assert torch.all(temporal_cos_f == temporal_cos_f[0:1, :]), \
        f"Frame {f}: within-frame temporal cos values differ"

# Temporal values should differ across frames (different row indices)
temporal_cos_frame0 = cos_assembled[0, patch_start, 0:11]
temporal_cos_frame1 = cos_assembled[0, patch_start + n_patches_per_frame, 0:11]
assert not torch.equal(temporal_cos_frame0, temporal_cos_frame1), \
    "Frames 0 and 1 have identical temporal cos — frame index not advancing"
```

**Height/width cross-frame invariant:** The height and width position IDs repeat identically across frames (same grid layout for each frame). Therefore:

```python
# Corresponding patch positions across frames must have equal height/width cos
for patch_idx in range(n_patches_per_frame):
    f0_cos = cos_assembled[0, patch_start + patch_idx, 22:64]
    f1_cos = cos_assembled[0, patch_start + n_patches_per_frame + patch_idx, 22:64]
    assert torch.equal(f0_cos, f1_cos), \
        f"Patch {patch_idx}: height/width cos differs between frame 0 and frame 1"
```

---

## PCC Threshold Rationale

The threshold of 0.9999 (not 0.99) is chosen for two reasons:

1. **Test 1 provides the baseline:** The mathematical proof in Ch3 guarantees bitwise exact equivalence for text-only tokens. Any implementation that produces the correct cos/sin gather for text-only inputs and extends it correctly to vision tokens will have very high PCC. A PCC of 0.9999 is a generous lower bound; a correct implementation should produce PCC > 0.99999 on float32.

2. **Cross-layer error accumulation:** RoPE is applied at every attention layer. For Qwen3.6-35B-A3B with 64 layers, a per-layer cos/sin error that yields PCC = 0.9999 at the RoPE level can reduce end-to-end generation quality. The threshold is tight precisely because it is the only quality gate before full model integration.

| Test | Criterion | Rationale |
|------|-----------|-----------|
| 1 (text-only) | `torch.equal` | Ch3 algebraic proof guarantees exact identity |
| 2 (single patch) | PCC > 0.9999 | Vision token routing; single-token sensitivity |
| 3 (full grid) | PCC > 0.9999 | Full grid exercises all 32 rotary pairs |
| 4 (video) | PCC > 0.9999 | Temporal axis correctness across frames |

---

## References

- `../ch3_text_only_reduction/mathematical_equivalence_proof.md`
- `../ch2_qwen36_mrope_config/position_id_construction.md`
- `../ch2_qwen36_mrope_config/hf_reference_implementation.md`
- `../ch4_ttnn_implementation/gather_operation_on_ttnn.md`
- `../ch5_performance_analysis/operation_cost_breakdown.md`
- `integration_steps.md` (Steps 5–6, CPU and device validation)
