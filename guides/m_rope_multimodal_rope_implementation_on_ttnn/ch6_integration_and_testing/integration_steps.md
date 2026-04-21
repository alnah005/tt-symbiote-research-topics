# Integration Steps

## Overview

These steps assume the text-only TTNN path is already working. All changes are additive and do not modify the existing text-only code path. Steps 1–4 build the implementation; Steps 5–6 validate it before and after moving to the TTNN device.

> **Key Finding:** The cos/sin frequency table computation is UNCHANGED from the existing implementation. M-RoPE reuses the same precomputed `[max_seq_len, rotary_dim]` tables — the only change is how positions are looked up (three separate gather operations instead of one) and how the resulting cos/sin slices are assembled (concatenation along the rotary dimension).

---

## Step 1 — Extract `mrope_section` from the Qwen3.6 Config

The `mrope_section` list partitions `rotary_dim // 2 = 32` cos/sin pairs into three axes: temporal, height, width. For Qwen3.6-35B-A3B the values are read directly from `config.rope_scaling.mrope_section`.

```python
def get_mrope_section(config) -> list[int]:
    """Extract mrope_section from config.rope_scaling or compute fallback."""
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    mrope_section = rope_scaling.get("mrope_section", None)
    if mrope_section is not None:
        return mrope_section
    # Fallback: derive from rotary_dim (balanced split)
    rotary_dim = int(config.head_dim * getattr(config, "partial_rotary_factor", 1.0))
    n_pairs = rotary_dim // 2
    # Even split fallback (not used for Qwen3.6, which has [11, 11, 10])
    return [n_pairs // 3, n_pairs // 3, n_pairs - 2 * (n_pairs // 3)]

# For Qwen3.6-35B-A3B:
# mrope_section = [11, 11, 10]  (from config.rope_scaling.mrope_section)
```

The sum of `mrope_section` must equal `rotary_dim // 2`. For Qwen3.6: `11 + 11 + 10 = 32 = 64 // 2`. Assert this invariant at construction time.

> **[SILENT FAILURE]** If `mrope_section` is read from the wrong config field or a future model changes the field name, the gather operations will silently index the wrong cos/sin columns. Always assert `sum(mrope_section) == rotary_dim // 2` at construction time and log the values for inspection.

### Section-to-column mapping (Qwen3.6)

| Axis | `mrope_section` value | cos/sin column pairs | cos columns | sin columns |
|------|-----------------------|----------------------|-------------|-------------|
| Temporal | 11 | pairs 0–10 | `[0:11]` | `[0:11]` |
| Height | 11 | pairs 11–21 | `[11:22]` | `[11:22]` |
| Width | 10 | pairs 22–31 | `[22:32]` | `[22:32]` |

The full `rotary_dim = 64` cos/sin vectors are assembled by concatenating the three axis slices: `[temporal_slice | height_slice | width_slice]`, shape `[batch, seq_len, rotary_dim]`.

---

## Step 2 — Extend `TTNNRotaryPositionEmbedding` (Option A from Ch4)

Modify the constructor to accept `mrope_section: list[int]` and a `use_mrope: bool` flag. The cos/sin table computation is UNCHANGED (same frequencies, same shape). Only the forward method changes.

```python
class TTNNRotaryPositionEmbedding:
    def __init__(self, device, config, use_mrope: bool = False):
        self.use_mrope = use_mrope
        self.mrope_section = get_mrope_section(config) if use_mrope else None
        if use_mrope:
            assert sum(self.mrope_section) == config.rotary_dim // 2, (
                f"mrope_section sum {sum(self.mrope_section)} != rotary_dim//2 "
                f"{config.rotary_dim // 2}"
            )
        # Precompute cos/sin table — UNCHANGED from existing implementation
        self.cos_table = precompute_cos_table(config)   # [max_seq_len, rotary_dim]
        self.sin_table = precompute_sin_table(config)   # [max_seq_len, rotary_dim]

    def forward(self, q, k, position_ids):
        if self.use_mrope:
            return self._mrope_forward(q, k, position_ids)
        else:
            return self._standard_forward(q, k, position_ids)  # existing path, unchanged

    def _mrope_forward(self, q, k, position_ids_3d):
        # position_ids_3d: [3, batch, seq_len]
        s_t, s_h, s_w = self.mrope_section
        # Three embedding lookups (one per axis)
        # Temporal: rows from position_ids_3d[0], columns [0:2*s_t] of cos/sin tables
        # Height:   rows from position_ids_3d[1], columns [2*s_t:2*(s_t+s_h)]
        # Width:    rows from position_ids_3d[2], columns [2*(s_t+s_h):rotary_dim]
        cos = self._gather_sections(self.cos_table, position_ids_3d, s_t, s_h, s_w)
        sin = self._gather_sections(self.sin_table, position_ids_3d, s_t, s_h, s_w)
        # rotate-half: same as standard path
        q_rot = rotate_half(q, cos, sin)
        k_rot = rotate_half(k, cos, sin)
        return q_rot, k_rot
```

The `_standard_forward` path is IDENTICAL to the current implementation with no added branches.

### `_gather_sections` implementation

This is the core of the M-RoPE forward. It performs three `ttnn.embedding` lookups and concatenates the results.

```python
def _gather_sections(self, table, position_ids_3d, s_t, s_h, s_w):
    """
    Gather cos or sin values for all three M-RoPE axes and concatenate.

    Args:
        table:           [max_seq_len, rotary_dim] cos or sin table on device
        position_ids_3d: [3, batch, seq_len] position IDs (int32) on device
        s_t, s_h, s_w:  mrope_section values (number of cos/sin pairs per axis)

    Returns:
        [batch, seq_len, rotary_dim] assembled cos/sin tensor
    """
    batch = position_ids_3d.shape[1]
    seq_len = position_ids_3d.shape[2]

    # Slice the full table into per-axis column ranges.
    # mrope_section values are pair counts; each section of s_i pairs spans 2*s_i
    # actual columns in the rotary_dim=64 table (matching HF apply_multimodal_rotary_pos_emb
    # which multiplies mrope_section*2 before splitting).
    # ttnn.slice or pre-slice on host at init time to avoid repeated device slicing.
    table_t = table[:, 0:2*s_t]              # [max_seq_len, 2*s_t]   e.g. columns [0:22]
    table_h = table[:, 2*s_t:2*(s_t+s_h)]   # [max_seq_len, 2*s_h]   e.g. columns [22:44]
    table_w = table[:, 2*(s_t+s_h):]        # [max_seq_len, 2*s_w]   e.g. columns [44:64]

    # Flatten position IDs for ttnn.embedding (expects 1D or 2D index tensor)
    ids_t = position_ids_3d[0].reshape(-1)   # [batch * seq_len]
    ids_h = position_ids_3d[1].reshape(-1)   # [batch * seq_len]
    ids_w = position_ids_3d[2].reshape(-1)   # [batch * seq_len]

    # Gather: ttnn.embedding treats position IDs as row indices into the weight table
    emb_t = ttnn.embedding(ids_t, table_t)   # [batch*seq_len, 2*s_t]
    emb_h = ttnn.embedding(ids_h, table_h)   # [batch*seq_len, 2*s_h]
    emb_w = ttnn.embedding(ids_w, table_w)   # [batch*seq_len, 2*s_w]

    # Concatenate along embedding dimension to reconstruct full rotary_dim (64 columns)
    emb = ttnn.concat([emb_t, emb_h, emb_w], dim=-1)  # [batch*seq_len, rotary_dim=64]

    # Reshape to [batch, seq_len, rotary_dim]
    return emb.reshape(batch, seq_len, -1)
```

> **[SILENT FAILURE]** Pre-slicing the cos/sin table on the host at `__init__` time (rather than re-slicing on device at every forward call) avoids 6 unnecessary device slice ops per step. However, if the table is modified after construction (e.g., re-uploaded during weight loading), the cached slices become stale. Invalidate and re-slice whenever the base table is replaced.

---

## Step 3 — Modify the Attention Module Forward

When vision tokens are present in the batch, pass a `[3, batch, seq_len]` position ID tensor to the RoPE module. When only text tokens are present, pass a standard 1D position tensor (existing behavior).

```python
# In TTNNQwen3FullAttention.forward():
if has_vision_tokens:
    q, k = rope_module.forward(q, k, position_ids_3d)   # [3, B, S]
else:
    q, k = rope_module.forward(q, k, position_ids_1d)   # existing path [B, S]
```

The gate `has_vision_tokens` is derived from whether the input batch contains any image/video patch tokens (determined by the tokenizer's token type IDs or an explicit flag from the generation loop).

### Detection strategy for `has_vision_tokens`

The cleanest approach is to pass an explicit boolean flag from the generation loop rather than re-examining token type IDs inside the attention module:

```python
# In the generation loop / model forward:
has_vision = (image_embeds is not None) or (pixel_values is not None)

# Pass through the model as a non-traced constant (Python bool, not a tensor)
model_output = model.forward(
    input_ids=input_ids,
    position_ids=position_ids_3d if has_vision else position_ids_1d,
    has_vision_tokens=has_vision,
)
```

The Python bool `has_vision` is resolved at trace compilation time, not at runtime, which means the Metal Trace captures either the M-RoPE branch or the standard branch — not both. If both branches need to be supported in the same trace, the dispatch must happen outside the trace (see `tracing_and_program_cache_considerations.md`).

---

## Step 4 — Implement Position ID Construction for Qwen VL Inputs

For a sequence with `n_text_pre` pre-image text tokens + `n_patches = num_h × num_w` image patches + `n_text_post` post-image text tokens:

```python
def build_position_ids_3d(
    n_text_pre: int,
    image_grid: tuple[int, int],  # (num_patches_h, num_patches_w)
    n_text_post: int,
    frame_idx: int = 0,
) -> torch.Tensor:
    """Build [3, 1, seq_len] position IDs for text + single image."""
    num_h, num_w = image_grid
    n_patches = num_h * num_w
    seq_len = n_text_pre + n_patches + n_text_post

    # Temporal, height, width: all initialized to sequential text positions
    t = torch.arange(seq_len, dtype=torch.int32)
    h = torch.arange(seq_len, dtype=torch.int32)
    w = torch.arange(seq_len, dtype=torch.int32)

    # Override image patch positions
    patch_start = n_text_pre
    for idx in range(n_patches):
        row = idx // num_w
        col = idx % num_w
        t[patch_start + idx] = frame_idx
        h[patch_start + idx] = row + n_text_pre   # offset by pre-image text length
        w[patch_start + idx] = col + n_text_pre

    # Post-image text: sequential positions starting after max(t, h, w) of patches
    post_start = patch_start + n_patches
    post_offset = max(t[patch_start:post_start].max().item(),
                      h[patch_start:post_start].max().item(),
                      w[patch_start:post_start].max().item()) + 1
    for i in range(n_text_post):
        t[post_start + i] = post_offset + i
        h[post_start + i] = post_offset + i
        w[post_start + i] = post_offset + i

    return torch.stack([t, h, w], dim=0).unsqueeze(1)  # [3, 1, seq_len]
```

### Multi-frame extension

For video inputs with `n_frames` frames, loop over frames and set `frame_idx = f` for frame `f`. The height and width grid repeats identically for each frame; only the temporal axis advances.

```python
def build_position_ids_3d_video(
    n_text_pre: int,
    image_grid: tuple[int, int],
    n_frames: int,
    n_text_post: int,
) -> torch.Tensor:
    """Build [3, 1, seq_len] position IDs for text + multi-frame video."""
    num_h, num_w = image_grid
    n_patches_per_frame = num_h * num_w
    n_patches_total = n_patches_per_frame * n_frames
    seq_len = n_text_pre + n_patches_total + n_text_post

    t = torch.arange(seq_len, dtype=torch.int32)
    h = torch.arange(seq_len, dtype=torch.int32)
    w = torch.arange(seq_len, dtype=torch.int32)

    patch_start = n_text_pre
    for f in range(n_frames):
        for idx in range(n_patches_per_frame):
            row = idx // num_w
            col = idx % num_w
            abs_idx = patch_start + f * n_patches_per_frame + idx
            t[abs_idx] = f                     # temporal = frame index
            h[abs_idx] = row + n_text_pre      # height offset constant across frames
            w[abs_idx] = col + n_text_pre      # width offset constant across frames

    post_start = patch_start + n_patches_total
    post_offset = n_frames  # temporal ran 0..n_frames-1; next text starts at n_frames
    for i in range(n_text_post):
        t[post_start + i] = post_offset + i
        h[post_start + i] = post_offset + i
        w[post_start + i] = post_offset + i

    return torch.stack([t, h, w], dim=0).unsqueeze(1)  # [3, 1, seq_len]
```

---

## Step 5 — CPU Reference Validation (Before TTNN Device)

Use `torch.testing.assert_close` to confirm the new `_mrope_forward` implementation matches HuggingFace's `apply_multimodal_rotary_pos_emb()` output for a mixed text+image batch. Validate on CPU before moving to TTNN. Target: numerical identity (not just PCC) since both implementations run in float32 on CPU.

```python
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
)
import torch

def validate_cpu_mrope(rope_module_mrope, config):
    """Assert CPU implementation matches HF reference before touching the device."""
    batch, seq_len, num_heads, head_dim = 1, 128, 64, 128
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Build a mixed text+image position ID tensor
    position_ids_3d = build_position_ids_3d(
        n_text_pre=32, image_grid=(8, 8), n_text_post=32
    )  # [3, 1, 128]

    # Compute HF reference cos/sin from position_ids_3d
    # (use the same inv_freq table as the TTNN implementation)
    cos_hf, sin_hf = compute_mrope_cos_sin_hf(position_ids_3d, config)

    q_hf, k_hf = apply_multimodal_rotary_pos_emb(
        q, k, cos_hf, sin_hf, mrope_section=config.rope_scaling["mrope_section"]
    )

    q_ttnn, k_ttnn = rope_module_mrope._mrope_forward(q, k, position_ids_3d)

    torch.testing.assert_close(q_ttnn, q_hf, rtol=0, atol=0,
                               msg="CPU M-RoPE q diverges from HF reference")
    torch.testing.assert_close(k_ttnn, k_hf, rtol=0, atol=0,
                               msg="CPU M-RoPE k diverges from HF reference")
    print("Step 5 PASS: CPU M-RoPE matches HF reference exactly")
```

Do not proceed to Step 6 until this test passes.

---

## Step 6 — TTNN Device Validation

Move to the TTNN device after CPU validation passes. Validate using PCC > 0.9999 against the CPU reference. This threshold is tight because RoPE errors in individual attention layers accumulate across all layers of the model.

```python
def validate_device_mrope(rope_module_ttnn_device, cpu_q_ref, cpu_k_ref,
                           position_ids_3d_device):
    """Validate TTNN device M-RoPE output against CPU reference using PCC."""
    q_dev, k_dev = rope_module_ttnn_device._mrope_forward(
        q_device, k_device, position_ids_3d_device
    )

    # Move results to CPU for comparison
    q_dev_cpu = ttnn.to_torch(q_dev)
    k_dev_cpu = ttnn.to_torch(k_dev)

    pcc_q = compute_pcc(q_dev_cpu, cpu_q_ref)
    pcc_k = compute_pcc(k_dev_cpu, cpu_k_ref)

    assert pcc_q > 0.9999, f"Device M-RoPE q PCC {pcc_q:.6f} below threshold 0.9999"
    assert pcc_k > 0.9999, f"Device M-RoPE k PCC {pcc_k:.6f} below threshold 0.9999"
    print(f"Step 6 PASS: device PCC q={pcc_q:.6f}, k={pcc_k:.6f}")
```

### PCC threshold rationale

The threshold is 0.9999, not 0.99, for two reasons:

1. The mathematical proof in Ch3 guarantees exact equivalence for text-only tokens — if the text-only section diverges, PCC can drop far below 0.9999.
2. Each attention layer applies RoPE to queries and keys. A small per-layer error in the cos/sin values (e.g., from bfloat16 rounding) multiplies across all 64 layers of Qwen3.6-35B-A3B. A per-layer PCC of 0.9995 compounds to ~0.97 at layer 64 — below acceptable quality for generation tasks.

---

## References

- `../ch2_qwen36_mrope_config/qwen36_rope_config.md`
- `../ch2_qwen36_mrope_config/position_id_construction.md`
- `../ch2_qwen36_mrope_config/hf_reference_implementation.md`
- `../ch3_text_only_reduction/mathematical_equivalence_proof.md`
- `../ch4_ttnn_implementation/existing_ttnn_rope_gap_analysis.md`
- `../ch4_ttnn_implementation/extension_approach.md`
- `../ch4_ttnn_implementation/gather_operation_on_ttnn.md`
- `../ch4_ttnn_implementation/pre_computed_cos_sin_strategy.md`
- `../ch5_performance_analysis/operation_cost_breakdown.md`
