# Verification Checklist for Strategy C

This file specifies the five test cases that must pass before Strategy C can be considered production-ready. Each test case identifies the configuration, the expected PCC threshold, a PyTorch reference formula, and the specific property being validated.

---

> **Key Finding:** A passing verification suite for Strategy C requires: (1) the existing tile-aligned case remains correct, (2) the previously broken non-tile-aligned case (`rotary_dim=48`) achieves PCC > 0.9999, (3) full-head RoPE degenerates correctly, (4) a traced decode step produces no `TT_FATAL` and matches non-traced PCC > 0.9999, and (5) the `rotary_dim=32` edge case is handled correctly by the identity fill.

---

## PyTorch Reference Formula

All five test cases compare against the same PyTorch reference implementation. Use the following function as the golden reference:

```python
import torch

def apply_rotary_partial_reference(x, rotary_dim):
    """
    Correct partial RoPE reference for any rotary_dim <= head_dim.

    Args:
        x:          [..., head_dim] — input head tensor (float32 for accuracy)
        rotary_dim: int — number of dimensions to rotate (must be even)

    Returns:
        [..., head_dim] — output with rotation applied to x[..., :rotary_dim]
        and x[..., rotary_dim:] passed through unchanged
    """
    assert rotary_dim % 2 == 0
    half = rotary_dim // 2

    x_rot  = x[..., :rotary_dim]   # [..., rotary_dim]  region to rotate
    x_pass = x[..., rotary_dim:]   # [..., head_dim - rotary_dim]  passthrough

    x1 = x_rot[..., :half]   # [..., half]  pairs left element
    x2 = x_rot[..., half:]   # [..., half]  pairs right element

    # Compute cos/sin for the rotary frequencies at this position
    # (In practice these are precomputed; here we show the formula explicitly)
    # For test purposes, use the same frequency generation as the implementation:
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    # Note: / half == * 2.0 / rotary_dim — the standard formula 1/base^(2i/rotary_dim).
    # Expand to match the sequence dimension of x (assumes x has a seq dim)
    # ...

    # Rotate-half applied within the rotary_dim slice only
    rotated = torch.cat([
        x1 * cos - x2 * sin,   # output[..., 0:half]
        x1 * sin + x2 * cos,   # output[..., half:rotary_dim]
    ], dim=-1)  # [..., rotary_dim]

    # Passthrough concatenated at the end — unchanged
    return torch.cat([rotated, x_pass], dim=-1)  # [..., head_dim]
```

In each test case below, `cos` and `sin` are the real rotation values at the test sequence position, matching the frequency table used by the implementation.

---

## Test Case 1 — Tile-Aligned Partial RoPE Baseline

**Configuration:** `rotary_dim=64, head_dim=128`  
**partial\_rotary\_factor:** 0.5  
**Expected PCC:** > 0.9999

**Purpose:** Verify that Strategy C does not regress the existing working case. All currently supported Qwen3-family models use this configuration. If this test fails after applying Strategy C, there is a bug in the new `__init__` construction.

**Test setup:**

```python
rotary_dim = 64
head_dim   = 128
max_seq_len = 2048
batch, num_heads, seq_len = 1, 32, 1  # decode step

# Construct TTNNRotaryPositionEmbedding with Strategy C
rope = TTNNRotaryPositionEmbedding(
    rotary_dim=rotary_dim, head_dim=head_dim,
    max_seq_len=max_seq_len, device=device,
)

# Random input in float32 for reference, cast to bfloat16 for device
x_ref = torch.randn(batch, num_heads, seq_len, head_dim)
x_dev = ttnn.from_torch(x_ref.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

# TTNN output
out_dev = rope.forward(x_dev, start_pos=0, seq_len=seq_len)
out_ref_ttnn = ttnn.to_torch(out_dev).to(torch.float32)

# PyTorch reference
out_ref_torch = apply_rotary_partial_reference(x_ref, rotary_dim=rotary_dim, start_pos=0)

pcc = torch.corrcoef(torch.stack([
    out_ref_ttnn.flatten(), out_ref_torch.flatten()
]))[0, 1].item()
assert pcc > 0.9999, f"Test 1 failed: PCC={pcc:.6f}"
```

**Note:** This test should pass both before and after applying Strategy C. A regression here indicates a construction error.

---

## Test Case 2 — Non-Tile-Aligned Partial RoPE (The Bug Configuration)

**Configuration:** `rotary_dim=48, head_dim=128`  
**partial\_rotary\_factor:** 0.375  
**Expected PCC before Strategy C:** ~0.71 (or `TT_FATAL` in Path A)  
**Expected PCC after Strategy C:** > 0.9999

**Purpose:** This is the primary regression test for the fix. It exercises the path that was previously broken. A PCC of ~0.71 before the fix and > 0.9999 after the fix confirms that Strategy C resolves the bug.

**Test setup:**

```python
rotary_dim = 48
head_dim   = 128

# Before fix: expect TT_FATAL (Path A) or PCC ~0.71 (Path B)
# After fix: expect PCC > 0.9999

rope = TTNNRotaryPositionEmbedding(
    rotary_dim=rotary_dim, head_dim=head_dim,
    max_seq_len=max_seq_len, device=device,
)
# ... same test structure as Test 1, with rotary_dim=48
pcc = compute_pcc(rope, x_ref, x_dev, rotary_dim=rotary_dim)
assert pcc > 0.9999, f"Test 2 failed: PCC={pcc:.6f} (expected > 0.9999 after Strategy C fix)"
```

**What to observe:** Run this test against the unmodified `TTNNRotaryPositionEmbedding` to confirm the baseline PCC ~0.71. Then apply Strategy C and re-run to confirm the fix. Both measurements validate the root cause analysis.

> **[SILENT FAILURE]** If Test 2 produces PCC ~0.71 rather than `TT_FATAL` on the unfixed implementation, autoformat is padding cos/sin further from 64 to 128 before the shape check. This is Path B. The numerical corruption is silent — the forward pass completes without error. Only this explicit PCC check reveals the bug.

---

## Test Case 3 — Full-Head RoPE (No Partial)

**Configuration:** `rotary_dim=128, head_dim=128`  
**partial\_rotary\_factor:** 1.0  
**Expected PCC:** > 0.9999

**Purpose:** When `rotary_dim == head_dim`, partial RoPE degenerates to standard full-head RoPE. Strategy C should degenerate correctly: `rotary_half = head_half = 64`, so there are no identity-fill regions (Region 2 and Region 4 have zero width), and the cos/sin table is simply the standard full-head table duplicated into `[cos_first, cos_first]`. This is equivalent to passing the full-frequency cos/sin table directly.

**Note on class routing:** In tt-symbiote, `partial_rotary_factor == 1.0` typically routes to `TTNNDistributedRotaryPositionEmbedding` rather than `TTNNRotaryPositionEmbedding`. This test case verifies that Strategy C degenerates correctly as a mathematical sanity check. If `TTNNRotaryPositionEmbedding` is not instantiated for `partial_rotary_factor == 1.0` in practice, this test can be run by directly constructing `TTNNRotaryPositionEmbedding` with `rotary_dim=128`.

```python
rotary_dim = 128
head_dim   = 128

# With Strategy C and rotary_dim == head_dim:
# rotary_half = 64, head_half = 64
# Region 2 width = head_half - rotary_half = 0  (no identity fill)
# cos_first = torch.cat([cos_real, cos_identity_first], dim=-1)
#           = torch.cat([cos_real, <empty>], dim=-1) = cos_real
# cos_full  = torch.cat([cos_first, cos_first], dim=-1)  — [max_seq_len, 128]
# This equals the standard full-head cos table doubled, which is the correct
# input for ttnn.experimental.rotary_embedding in the full-head case.

rope = TTNNRotaryPositionEmbedding(
    rotary_dim=128, head_dim=128,
    max_seq_len=max_seq_len, device=device,
)
pcc = compute_pcc(rope, x_ref, x_dev, rotary_dim=128)
assert pcc > 0.9999, f"Test 3 failed: PCC={pcc:.6f}"
```

---

## Test Case 4 — Trace Compatibility

**Configuration:** `rotary_dim=48, head_dim=128` (the previously broken configuration)  
**Expected behavior:** No `TT_FATAL` during trace capture or replay; PCC between traced and non-traced output > 0.9999  
**Purpose:** Confirm that Strategy C's `__init__`-only allocation policy makes the forward pass safe inside a Metal Trace bracket.

```python
rotary_dim = 48
head_dim   = 128

rope = TTNNRotaryPositionEmbedding(
    rotary_dim=rotary_dim, head_dim=head_dim,
    max_seq_len=max_seq_len, device=device,
)

# --- Non-traced warm-up (establishes baseline output) ---
out_warmup = ttnn.to_torch(rope.forward(x_dev, start_pos=0, seq_len=1))

# --- Trace capture ---
# No TT_FATAL should fire here; all allocations occurred in __init__
trace_id = ttnn.begin_trace_capture(device, trace_buffer_size=...)
out_trace_first = rope.forward(x_dev, start_pos=0, seq_len=1)
ttnn.end_trace_capture(device, trace_id)

# --- Trace replay ---
ttnn.execute_trace(device, trace_id)
out_trace_replay = ttnn.to_torch(out_trace_first)

# PCC between non-traced and traced output
pcc_trace = compute_pcc_tensors(out_warmup.float(), out_trace_replay.float())
assert pcc_trace > 0.9999, (
    f"Test 4 failed: traced vs. non-traced PCC={pcc_trace:.6f}. "
    "If PCC < 1.0, check for buffer allocation inside forward — "
    "trace replay requires all buffers to be pre-allocated."
)
```

**What to observe:** Before Strategy C, `ttnn.pad` inside `forward` allocates a new device buffer during trace capture. During replay, this allocation is skipped (trace only replays compute, not allocations), causing the forward pass to operate on the wrong buffer. After Strategy C, no allocation occurs in `forward`, so traced and non-traced outputs match.

---

## Test Case 5 — Edge Case: rotary\_dim=32, head\_dim=128

**Configuration:** `rotary_dim=32, head_dim=128`  
**partial\_rotary\_factor:** 0.25  
**Expected PCC:** > 0.9999

**Purpose:** `rotary_dim=32` satisfies `% 32 == 0` but NOT `% 64 == 0`. This edge case verifies that Strategy C correctly handles the case where `rotary_dim` is a single tile wide (32 elements) but the two-tile constraint is on `head_dim` (128), not `rotary_dim`. The identity-fill correctly covers positions `[16, 64)` and `[80, 128)`.

```python
rotary_dim = 32   # satisfies % 32 == 0, does NOT satisfy % 64 == 0
head_dim   = 128  # satisfies % 64 == 0 (the required constraint)

# Strategy C construction for this case:
# rotary_half = 16, head_half = 64
# Region 1: positions [0, 16)   — real rotation values
# Region 2: positions [16, 64)  — identity (cos=1.0, sin=0.0)
# Region 3: positions [64, 80)  — duplicate of Region 1 (never read by kernel)
# Region 4: positions [80, 128) — identity (cos=1.0, sin=0.0)

rope = TTNNRotaryPositionEmbedding(
    rotary_dim=32, head_dim=128,
    max_seq_len=max_seq_len, device=device,
)
pcc = compute_pcc(rope, x_ref, x_dev, rotary_dim=32)
assert pcc > 0.9999, f"Test 5 failed: PCC={pcc:.6f}"
```

**Note:** This test case would FAIL with Strategy A (slice-apply-concat) because `rotary_dim=32` satisfies `% 32 == 0` but not `% 64 == 0`, and the `ttnn.experimental.rotary_embedding` two-tile constraint would fire on the sliced input of width 32. Strategy C passes this test because `head_dim=128` satisfies `% 64 == 0` and the input is never sliced.

---

## Helper Functions for Test Suite

```python
import torch

def compute_pcc(rope_instance, x_ref_float32, x_dev_bfloat16, rotary_dim, start_pos=0):
    """
    Run one forward pass through the TTNN rope and compare PCC against
    the PyTorch partial RoPE reference.

    Returns the Pearson correlation coefficient as a float.
    """
    out_dev    = rope_instance.forward(x_dev_bfloat16, start_pos=start_pos, seq_len=1)
    out_ttnn   = ttnn.to_torch(out_dev).to(torch.float32).flatten()
    out_ref    = apply_rotary_partial_reference(
        x_ref_float32, rotary_dim=rotary_dim, start_pos=start_pos
    ).flatten()
    pcc = torch.corrcoef(torch.stack([out_ttnn, out_ref]))[0, 1].item()
    return pcc


def compute_pcc_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PCC between two tensors of the same shape."""
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
```

---

## Summary of Test Cases

| Test | rotary\_dim | head\_dim | Was broken? | Expected PCC (after fix) | Property validated |
|---|---|---|---|---|---|
| 1 — Tile-aligned baseline | 64 | 128 | No | > 0.9999 | No regression |
| 2 — Non-tile-aligned (bug config) | 48 | 128 | Yes (~0.71) | > 0.9999 | Strategy C correctness |
| 3 — Full-head degenerate case | 128 | 128 | N/A | > 0.9999 | Degenerate case |
| 4 — Trace compatibility | 48 | 128 | Yes (TT_FATAL or wrong PCC) | > 0.9999 traced vs. non-traced | No allocation in forward |
| 5 — Edge case (32, 64-boundary) | 32 | 128 | Not reached (dead code) | > 0.9999 | Identity-fill for small rotary_dim |
