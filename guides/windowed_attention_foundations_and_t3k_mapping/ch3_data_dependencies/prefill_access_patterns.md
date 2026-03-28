# Prefill Memory Access Patterns

During prefill the model processes an input sequence of length T in a single
forward pass. All T query vectors, key vectors, and value vectors are computed
simultaneously from the input embeddings, and the attention score matrix is
formed as a `[T, T]` product Q K^T / sqrt(d). The window constraint imposes a
band-diagonal structure on this matrix that changes both the computational
graph and the memory access pattern relative to full causal attention.

## The Band-Diagonal Mask on the Score Matrix

### Structure of the Mask

For full causal attention the allowed score positions form a lower-triangular
matrix: query at row t may attend to key at column s for any s in [0, t]. The
number of valid entries in row t is t + 1, and the total number of valid
entries across all rows is T(T+1)/2.

For windowed attention with window size w the valid positions form a band
along the main diagonal. Row t is allowed to attend to columns
[max(0, t-w+1), t], a span of min(t+1, w) entries. The mask is:

$$M_{\text{win}}[t, s] =
\begin{cases}
0       & \text{if } \max(0,\, t - w + 1) \leq s \leq t \\
-\infty & \text{otherwise}
\end{cases}$$

The total number of valid entries across all rows is:

$$\sum_{t=0}^{T-1} \min(t+1,\, w)
= \frac{w(w-1)}{2} + w \cdot (T - w + 1)
\approx T \cdot w \quad \text{for } T \gg w$$

For T = 32 768 and w = 4 096 the band-diagonal mask has approximately 126M
valid entries, compared with 536M for the full lower triangle — a 4× reduction
in the number of scores that must be computed and materialised.

### Mask Diagram (T = 8, w = 3)

The diagram uses `1` for allowed (score computed) and `.` for masked
(score set to -inf before softmax):

```text
Key position →  0 1 2 3 4 5 6 7
               ─────────────────
Query pos 0  │  1 . . . . . . .
Query pos 1  │  1 1 . . . . . .
Query pos 2  │  1 1 1 . . . . .
Query pos 3  │  . 1 1 1 . . . .
Query pos 4  │  . . 1 1 1 . . .
Query pos 5  │  . . . 1 1 1 . .
Query pos 6  │  . . . . 1 1 1 .
Query pos 7  │  . . . . . 1 1 1
```

Rows 0–2 are the fill phase where the window has not yet reached full width.
Rows 3–7 are the steady-state band where every row has exactly w = 3 valid
entries. The band is strictly lower-triangular (causal) and of fixed width w.

## Memory Access Pattern for K and V During Prefill

### Each Query Row Reads a Contiguous Stripe

The score for query row t depends only on key rows in [max(0, t-w+1), t].
After masking, rows outside this range contribute zero weight to the output
regardless of their values, so no memory access to those rows is strictly
necessary. The access pattern for K during prefill is therefore:

```text
Query row t accesses K rows: [max(0, t-w+1),  t]
                              ↑                ↑
                         oldest key        newest key
                         in window         (same position as query)
```

This is a **sliding stripe** of w consecutive rows in the K matrix. As t
advances by one, the stripe shifts one row forward: the oldest key row falls
out of scope and the new key row (for position t) enters. The V matrix is
accessed with the identical stripe for the weighted sum.

### Contiguity in DRAM

During prefill, K and V are typically materialised as dense tensors of shape
`[B, H, T, d]` — not yet written into the circular buffer (the circular buffer
is the decode-phase structure; Chapter 2 explains that it begins to be
populated as tokens are processed). For each head and batch item the K tensor
is a contiguous block of T×d elements in DRAM. A stripe read of w consecutive
rows starting at row `r` is therefore a single contiguous DRAM read of
`w * d` elements — highly cache-friendly and well-suited to DRAM burst
transfers.

The stride across consecutive query rows is exactly d elements (one key
vector), so successive stripe reads overlap by (w-1) rows. In a tiled
streaming implementation each new query tile shares all but one K row with
the preceding tile, enabling the reuse of `(w-1)/w` of the K data from the
previous iteration's L1 buffer.

## Masked Full-Attention Kernel vs Tiled Streaming Kernel

### Masked Full-Attention Approach

The band-diagonal mask can be applied inside an otherwise standard full-
attention kernel. The kernel materialises the full `[T, T]` score matrix,
applies M_win element-wise (setting out-of-band positions to -inf), runs
softmax row-wise, and performs the weighted sum with V. This approach is
correct and straightforward to implement: it requires no structural change to
the kernel, only the substitution of M_win for M_full.

The cost is that the full `[T, T]` score matrix is materialised in SRAM even
though `1 - w/T` of its entries are masked to -inf. For large T this is
wasteful:

- T = 32 768: score matrix is 32768^2 * 2 bytes = 2 GiB per head — far
  exceeding any on-chip SRAM capacity and requiring tiling across DRAM.
- T = 4 096 and w = 512: score matrix is 4096^2 * 2 bytes = 32 MiB, still
  much larger than Wormhole's L1 SRAM per core (1.5 MiB).

Even with tiling, a masked full-attention kernel reads K rows outside the
valid band into the on-chip tile before discarding them, wasting DRAM bandwidth
proportional to `1 - w/T`.

### Tiled Streaming Kernel

A tiled streaming kernel exploits the band structure to avoid loading masked-
out K rows entirely. It processes query tiles of height q_tile and key tiles
of height k_tile, but only loads a key tile if it overlaps with the valid
band for the query tile being processed. For a query tile covering rows
[t0, t0 + q_tile - 1], the valid key columns span
[max(0, t0 - w + 1), t0 + q_tile - 1]. Any key tile that falls entirely
outside this range is skipped.

The fraction of key tiles that must be loaded is approximately:

$$\frac{w + q\_tile - 1}{T} \approx \frac{w}{T} \quad \text{for } q\_tile \ll T$$

This is the bandwidth saving factor: the kernel reads roughly `w/T` of the K
and V data compared with a full lower-triangular kernel.

The tiled streaming approach requires the kernel to track which key tiles
intersect the current query tile's band and to handle the fill-phase rows
(t < w-1) where the band is narrower than w. Both are straightforward
bookkeeping additions. The resulting kernel is structurally similar to
FlashAttention's tiled implementation, with the causal mask replaced by the
band constraint.

## Arithmetic Intensity Analysis for Prefill

Arithmetic intensity (AI) is defined as FLOPs per byte of DRAM traffic. A
higher AI means the kernel is more compute-bound and less sensitive to DRAM
bandwidth.

### FLOPs for Windowed Prefill

For a single head and batch item, the dominant operations are:

| Operation       | FLOPs                        | Notes                            |
|-----------------|------------------------------|----------------------------------|
| QK^T scores     | 2 · T · w · d                | Each of T query rows dots with w key rows |
| Softmax         | ~5 · T · w                   | exp, sum, divide; subdominant for large d |
| Score · V       | 2 · T · w · d                | Weighted sum over w value rows per query  |
| **Total**       | **~4 · T · w · d**           | Ignoring softmax overhead        |

For T = 32 768, w = 4 096, d = 128:

$$\text{FLOPs} \approx 4 \times 32768 \times 4096 \times 128 = 68.7 \text{ GFLOPs per head}$$

### DRAM Traffic for Windowed Prefill (Tiled Streaming)

The bytes read from DRAM per attention head per batch item are:

| Tensor       | Shape read        | Bytes (bfloat16)         |
|--------------|-------------------|--------------------------|
| Q            | [T, d]            | T · d · 2                |
| K (band)     | [T · w / T, d] = [w, d] per query stripe pass | T · w · d · 2 (total unique K reads = T · d · 2 if fully reused, else up to T · w · d · 2 if not) |
| V (band)     | same as K         | same as K                |
| Output O     | [T, d]            | T · d · 2 (write)        |

In the ideal case where the sliding-stripe reuse is fully exploited (each K
row is loaded exactly once), the total K traffic is T · d · 2 bytes — the
same as loading K once regardless of w. However, this requires the K tile to
remain in L1 while all query rows that overlap it are processed, which demands
L1 capacity of at least w · d · 2 bytes. On Wormhole with d = 128 and
bfloat16, that is w · 256 bytes; for w = 4 096 this is 1 MiB, which is at the
limit of available L1 SRAM per core.

When full K reuse is not achievable (L1 too small), each unique K row is read
multiple times. In the worst case (no L1 reuse, every query row rereads its
full w-wide stripe) the K traffic is T · w · d · 2 bytes. Practical kernels
fall between these extremes depending on the tile sizes chosen.

### Arithmetic Intensity as a Function of w and T

Assuming the Flash-Attention style tiled streaming kernel with ideal K/V reuse
(K and V each loaded once from DRAM as w·d·2 bytes, Q loaded as T·d·2 bytes,
and output O written as T·d·2 bytes):

$$\text{FLOPs} = 4 \cdot T \cdot w \cdot d$$

$$\text{Bytes} = \underbrace{2 \cdot w \cdot d \cdot 2}_{\text{K + V, loaded once}}
             + \underbrace{T \cdot d \cdot 2}_{\text{Q}}
             + \underbrace{T \cdot d \cdot 2}_{\text{O write}}
             = 2 \cdot d \cdot (2w + T) + 2 \cdot T \cdot d
             = 2 \cdot d \cdot (2w + 2T)
             = 4 \cdot d \cdot (w + T)$$

$$\text{AI}_{\text{ideal reuse}}
= \frac{4 \cdot T \cdot w \cdot d}{4 \cdot d \cdot (w + T)}
= \frac{T \cdot w}{w + T}$$

When T ≫ w (late prefill, large sequence):

$$\text{AI}_{\text{ideal reuse}} \approx \frac{T \cdot w}{T} = w$$

When T = w (early prefill):

$$\text{AI}_{\text{ideal reuse}} = \frac{w^2}{2w} = \frac{w}{2}$$

This shows that AI scales linearly with w. For w = 4 096 in the T ≫ w regime:

$$\text{AI}_{\text{ideal}} \approx 4096 \text{ FLOPs/byte}$$

Wormhole's peak compute-to-bandwidth ratio is approximately:
- Peak BF16 matmul throughput: ~32 TFLOPS per chip
- Peak DRAM bandwidth: ~288 GB/s per chip
- Roofline crossover: 32e12 / 288e9 ≈ 111 FLOPs/byte

At AI ≈ 4096 FLOPs/byte, prefill with large w is firmly compute-bound, not
bandwidth-bound. The kernel's performance ceiling is set by matrix multiply
throughput, not by DRAM bandwidth.

In the worst case (no L1 reuse), AI collapses to ~1 FLOP/byte — the same result as decode AI; see [`decode_access_patterns.md`](./decode_access_patterns.md) for the derivation. At ~1 FLOP/byte the kernel is severely bandwidth-bound, illustrating why achieving K/V reuse in on-chip SRAM is critical: the difference between ideal and no-reuse arithmetic intensity is a factor of approximately w, which for w = 4096 is over 4000×.

### Summary Table

| Scenario              | AI (FLOPs/byte)        | Bound         | Notes                                        |
|-----------------------|------------------------|---------------|----------------------------------------------|
| Ideal K/V reuse       | ≈ w ≈ 4096 (T ≫ w)     | Compute-bound | K and V each loaded once, full L1 tile reuse |
| Partial reuse (tile)  | Between ~1 and ~w      | Mixed         | Depends on L1 tile size and overlap pattern  |
| No reuse              | ~1                     | Memory-bound  | Every query re-reads its K/V stripe from DRAM|

The tiled streaming kernel's design goal is to maximise K/V reuse within the
available L1 SRAM, pushing AI as close to the ideal case as possible.

---

**Next:** [`decode_access_patterns.md`](./decode_access_patterns.md)
