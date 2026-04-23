# Kernel Rotate-Half Pairing in `rotary_embedding.cpp`

This file examines the compute kernel and reader dataflow kernel that implement the rotate-half operation. The goal is to show precisely how tile pairing is determined and to establish that the split point is always derived from the full `head_dim`, not from `rotary_dim`. This has a direct consequence for partial RoPE: the kernel provides no native mechanism to restrict rotation to a subset of positions.

---

## 1. Tile-Count Derivation in the Compute Kernel

The compute kernel lives at:

```
ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding.cpp
```

Near the top of the kernel, the total tile width and the rotate-half split point are derived:

```cpp
uint32_t Wt      = input.padded_shape()[-1] / TILE_WIDTH;  // tiles per row
uint32_t half_Wt = Wt / 2;                                  // rotate-half boundary
```

For `head_dim=128` and `TILE_WIDTH=32`:

$$W_t = \frac{128}{32} = 4 \qquad \text{half\_Wt} = \frac{4}{2} = 2$$

The first 2 tiles (indices 0 and 1, covering elements $[0, 64)$) are the "left half". The last 2 tiles (indices 2 and 3, covering elements $[64, 128)$) are the "right half".

---

## 2. How the Kernel Pairs Tiles

The rotate-half operation maps:

$$x'_i = x_i \cos\theta_i - x_{i + \text{head\_dim}/2} \sin\theta_i \quad \text{for } i \in [0, \text{head\_dim}/2)$$
$$x'_{i + \text{head\_dim}/2} = x_i \sin\theta_i + x_{i + \text{head\_dim}/2} \cos\theta_i \quad \text{for } i \in [0, \text{head\_dim}/2)$$

At the tile level, the kernel iterates `j = 0` to `j < Wt` and pairs tile `j` with tile `j + half_Wt` (when `j < half_Wt`) or tile `j - half_Wt` (when `j >= half_Wt`):

```cpp
for (uint32_t j = 0; j < Wt; ++j) {
    uint32_t cos_sin_tile_id = j;
    uint32_t paired_tile_id  = (j < half_Wt) ? (j + half_Wt) : (j - half_Wt);

    // multiply input tile j by cos tile j
    // multiply input tile paired_tile_id by sin tile j (with sign)
    // accumulate result into output tile j
}
```

> **Key Finding:** The kernel's rotate-half pairing ALWAYS splits the full `head_dim` into two equal halves using `half_Wt = Wt / 2`, where `Wt` is derived from `input.padded_shape()[-1]`. There is NO mechanism to limit rotation to a `rotary_dim` subset. The `rotary_dim` compile-time parameter (when present) gates which tiles are processed versus passed through, but the pairing of "left" and "right" halves is always over the full `head_dim`.

> **Note:** The Python golden function pairs on `rotary_dim/2`; the TTNN kernel always pairs on `head_dim/2`. These are different conventions at different layers. When `rotary_dim < head_dim`, the kernel's pairing boundary does not align with the formula from Ch1's `partial_rope_math.md`.

---

## 3. The Reader Kernel and `rotated_input_curr_id`

The reader dataflow kernel lives at:

```
ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp
```

It is responsible for fetching the correct input tiles for both the "normal" (left-half) position and the "rotated" (right-half) position. The key line is:

```cpp
uint32_t rotated_input_curr_id = start_id + half_Wt;
```

This advances the read pointer by exactly `half_Wt` tiles into the input tensor, landing on the start of the right half. The reader fetches tiles in two passes:

```cpp
// Pass 1: fetch left-half input tiles (tile indices 0 .. half_Wt-1)
for (uint32_t j = 0; j < half_Wt; ++j) {
    noc_async_read_tile(input_curr_id + j, input_cb);
}

// Pass 2: fetch right-half input tiles (tile indices half_Wt .. Wt-1)
for (uint32_t j = 0; j < half_Wt; ++j) {
    noc_async_read_tile(rotated_input_curr_id + j, rotated_input_cb);
}
```

Both `input_curr_id` and `rotated_input_curr_id` are offsets into the same input tensor buffer; the only difference is the `+ half_Wt` displacement. The cos and sin tiles are fetched independently using the same tile index `j` (0 to `half_Wt - 1`), which is why the cos/sin tensors must span the full `head_dim` width — they are indexed in lockstep with the full tile range.

---

## 4. Implication for Partial RoPE

Because `Wt` and `half_Wt` are derived from `input.padded_shape()[-1]` rather than from any `rotary_dim` argument, the kernel inherently applies rotate-half to the **entire `head_dim`**. For a model like Qwen3 where `rotary_dim=48` and `head_dim=128`:

- The kernel would attempt to rotate elements $[0, 64)$ with elements $[64, 128)$.
- Mathematically, the correct partial RoPE should rotate elements $[0, 24)$ with elements $[24, 48)$ and leave elements $[48, 128)$ unchanged.
- Under naive padding (zeros at positions `[rotary_dim:]`), the pairing is entirely wrong for positions beyond `rotary_dim/2`.

Under Strategy C (identity values at `[rotary_dim:]` — see Chapter 4), the same kernel produces correct output.

> **[SILENT FAILURE]:** If a caller manages to satisfy the shape constraint (by supplying cos/sin with `shape[-1]==128`) but fills positions $[48, 128)$ of cos/sin with arbitrary values (e.g., zeros or the wrong cache entries), the kernel will silently apply incorrect rotations to those positions. No runtime error is raised; the output is numerically wrong.

The only way to achieve correct partial RoPE through this op is to construct cos/sin such that:

- Positions $[0, \text{rotary\_dim})$: hold the actual cos/sin values for the rotation.
- Positions $[\text{rotary\_dim}, \text{head\_dim})$: hold identity-compatible values (cos=1, sin=0) so that the rotate-half formula reduces to copying the input unchanged.

This is **Strategy C (identity-filled cos/sin)**, which is analyzed in Chapter 4.

---

**Next:** [`what_the_golden_function_reveals.md`](./what_the_golden_function_reveals.md)
