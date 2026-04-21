# Pre-Computed Cos/Sin Strategy

## Section 1: Does M-RoPE Require Per-Section Cos/Sin Tables?

> **Key Finding:** No. A single cos/sin table of shape `[max_seq_len, rotary_dim/2]` is sufficient for M-RoPE. All sections use the same frequency definitions — only the position coordinate used to index each section differs.

M-RoPE does not introduce new frequencies, new dimension assignments, or any change to the frequency table structure. The temporal, height, and width sections are defined by which *columns* of the table they occupy, and the three-gather construction selects rows using three different position coordinates. No additional tables are needed.

---

## Section 2: Why a Single Table Works

The frequency value `θ_i` for dimension pair `i` is determined solely by `i` and the global parameters `rope_theta` and `rotary_dim`:

```math
θ_i = 1 / rope_theta^(2i / rotary_dim)
```

This formula is independent of which section dimension `i` falls in. The section assignment controls only *which rows* of the table are selected when computing cos/sin for a given token — not the frequency values stored in those rows.

Concretely, for Qwen3.6 with `mrope_section = [11, 11, 10]`:

- **Temporal section:** uses columns `[:11]` of each gathered row — frequency pairs `θ_0, ..., θ_{10}`
- **Height section:** uses columns `[11:22]` of each gathered row — frequency pairs `θ_{11}, ..., θ_{21}`
- **Width section:** uses columns `[22:32]` of each gathered row — frequency pairs `θ_{22}, ..., θ_{31}`

All 32 of these entries exist in the same `[max_seq_len, 32]` table. The temporal axis gathers row `position_ids[0, b, t]`; the height axis gathers row `position_ids[1, b, t]`; the width axis gathers row `position_ids[2, b, t]`. Different rows are selected per axis because the position coordinates differ — but the same table is used for all three.

---

## Section 3: Memory Implication

Standard partial RoPE and M-RoPE require identical storage:

- **Standard partial RoPE:** one cos table `[max_seq_len, 32]` + one sin table `[max_seq_len, 32]`
- **M-RoPE:** same tables, same shape, same dtype

At `max_seq_len = 32768` with BF16 (2 bytes):

```math
2 × 32768 × 32 × 2 = 4,194,304 bytes ≈ 4 MiB
```

This fits in DRAM with negligible footprint. No additional table storage is required for M-RoPE support.

---

## Section 4: Video Token Caveat

The single-table assumption holds as long as all position coordinates — temporal, height, and width — are within `[0, max_seq_len)`. For text and image inputs this is always satisfied: height and width patch indices are bounded by the grid size (typically well under 1024), and sequential text positions are bounded by `max_seq_len` by construction.

Video inputs introduce a potential exception on the temporal axis. The temporal position of a video token is the frame index, not the sequence position. A long video can have frame indices far exceeding the text `max_seq_len`:

- 2-hour video at 30 fps = 216,000 frames
- Typical text `max_seq_len` for Qwen3.6-35B-A3B = 32,768

If the maximum expected temporal position exceeds `max_seq_len`, the cos/sin table must be extended to cover the full temporal range before inference. For Qwen3.6-35B-A3B VL inference, verify the maximum expected temporal position for the target video workload. If it exceeds the current `max_seq_len`, extend `max_seq_len` for table construction — or build separate tables for the temporal and spatial axes with different `max_len` values (at the cost of the single-table simplicity).

Height and width positions are bounded by the maximum number of patches along each spatial dimension and are typically much smaller than `max_seq_len`.

---
**Next:** [`gather_operation_on_ttnn.md`](./gather_operation_on_ttnn.md)
