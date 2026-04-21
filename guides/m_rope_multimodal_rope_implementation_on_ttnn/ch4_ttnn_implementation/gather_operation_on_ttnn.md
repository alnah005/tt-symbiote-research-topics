# Gather Operation on TTNN

## Section 1: The Key New Operation

The central new TTNN operation required for M-RoPE is an indexed lookup of the cos/sin table using a 2D integer position ID tensor `[batch, seq_len]`, where each entry can be any integer in `[0, max_seq_len)`. This is a random-access gather, not a contiguous slice.

**Current (standard RoPE):**
```python
cos_slice = cos_table[cur_pos : cur_pos + seq_len]  # contiguous range, O(1) address
```
One integer defines the start of a contiguous row range. All dimension pairs use the same position coordinate.

**M-RoPE:**
```python
cos_rows = cos_table[position_ids_axis, :]  # per-token row gather, arbitrary integer indices
```
Each token has its own position integer per axis. The access pattern is non-contiguous for image and video inputs and requires an embedding-style indexed lookup.

---

## Section 2: `ttnn.embedding` for Gather

> **Key Finding:** `ttnn.embedding` is semantically and computationally identical to the three-gather construction used in HuggingFace's `apply_multimodal_rotary_pos_emb()`. It looks up rows of a weight matrix by integer index — exactly what M-RoPE requires for cos/sin table access.

`ttnn.embedding` treats the cos/sin table as an embedding weight matrix and the position IDs as index tensors:

- **Embedding weight** (table): `[max_seq_len, rotary_dim/2]` — each "vocabulary item" is a row of frequency values
- **Index tensor** (position IDs per axis): `[batch, seq_len]` — one axis slice from the `[3, batch, seq_len]` M-RoPE position_ids tensor; each `ttnn.embedding` call receives one such slice
- **Output:** `[batch, seq_len, rotary_dim/2]` — the gathered rows

Usage sketch:

```python
# cos_table: ttnn.Tensor shape [max_seq_len, rotary_dim_half]  (on device, DRAM)
# position_ids_temporal: ttnn.Tensor shape [batch, seq_len]   (integer indices)

# Gather temporal rows (full row, then slice to section width)
cos_all = ttnn.embedding(position_ids_temporal, cos_table)  # [batch, seq_len, rotary_dim_half]
cos_t = cos_all[:, :, :s_t]   # [batch, seq_len, s_t=11]

# Similarly for height and width axes
cos_h_all = ttnn.embedding(position_ids_height, cos_table)  # [batch, seq_len, rotary_dim_half]
cos_h = cos_h_all[:, :, s_t:s_t+s_h]  # [batch, seq_len, s_h=11]

cos_w_all = ttnn.embedding(position_ids_width, cos_table)   # [batch, seq_len, rotary_dim_half]
cos_w = cos_w_all[:, :, s_t+s_h:]     # [batch, seq_len, s_w=10]
```

The full construction (6 embedding calls + column slices + 4 concat ops) produces the assembled `[batch, seq_len, rotary_dim]` cos and sin tensors that the rotate-half kernel consumes.

---

## Section 3: Decode vs. Prefill Considerations

**Decode (seq_len=1):** `ttnn.embedding` receives `[batch, 1]` index tensors and returns `[batch, 1, rotary_dim_half]`. Each call degenerates to a single row lookup per batch item. The 6-call overhead is negligible in absolute terms and is dominated by the attention and FFN compute. No special decode optimization is required.

**Prefill — text-only (sequential IDs):** Position IDs for text are sequential: `position_ids[axis, b, t] = t` for all axes. The embedding access pattern is sequential across rows — cache-friendly and similar in behavior to the existing contiguous slice. Performance should be comparable to the standard path.

**Prefill — image/video (grid IDs):** Height and width position IDs follow a 2D grid pattern. For a patch grid of shape `H × W`, tokens in the same row share the same height index and tokens in the same column share the same width index. This produces a non-sequential access pattern across the `seq_len` dimension of the position ID tensor. DRAM bandwidth efficiency is reduced relative to sequential access, but the total data volume is bounded by `6 × seq_len × rotary_dim_half × 2 bytes` (BF16) — manageable for typical vision sequence lengths.

---

## Section 4: Host-Side Gather (Not Recommended)

An alternative is to compute cos/sin on the CPU, assemble the full `[batch, seq_len, rotary_dim]` cos and sin tensors on host, and transfer them to device before the RoPE application.

For prefill this is acceptable: the transfer happens once per sequence, and the PCIe latency (typically ~1 ms for a few MiB) is small relative to prefill compute time.

For decode this is not acceptable: a host-side gather + PCIe transfer adds latency to every decode step, which compounds across the full output sequence. The `ttnn.embedding` path keeps all gather operations on-device and avoids this penalty.

---

## Section 5: Test Cases Required

Chapter 6 will specify the full validation suite. The minimum required tests are:

1. **Text-only equivalence:** Construct sequential position IDs `position_ids[axis, b, t] = t` for all three axes and verify that the three-gather path produces cos/sin output numerically identical (within BF16 rounding) to the standard 1D contiguous-slice path. This validates the text-only reduction proven analytically in Chapter 3.

2. **Image position ID construction:** Given a known image grid (e.g., `H=4, W=6` patches with text prefix of length `P`), verify that the `[3, batch, seq_len]` position ID tensor is constructed correctly — text tokens have sequential identical IDs across all axes; image tokens have `axis=0` (temporal) constant, `axis=1` (height) cycling through `[0, H)`, and `axis=2` (width) cycling through `[0, W)`.

3. **Numerical comparison vs. HuggingFace:** Run `apply_multimodal_rotary_pos_emb()` from the HuggingFace Qwen3.6-VL implementation with a fixed input Q/K pair and known `position_ids_3d`. Compare the TTNN output against the HuggingFace output to within BF16 tolerance. This is the ground-truth correctness check.

---
**Next:** [Chapter 5 — Performance Cost Analysis](../ch5_performance_analysis/index.md)
