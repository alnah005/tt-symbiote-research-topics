# Top-k Selection Efficiency

Top-k selection is the second major operation in the Mixture-of-Experts (MoE) router, following the linear projection. For Qwen3.5-35B with $E = 256$ experts and $k = 8$ selected per token, the choice of selection algorithm has a measurable impact on router latency. This file analyzes algorithmic complexity, hardware-level execution on Wormhole B0, and the tile-parallel approach suited to the Tenstorrent architecture.

Throughout this file, the following constants apply:

| Symbol | Value |
|---|---|
| $E$ | 256 (number of experts) |
| $k$ | 8 (top-k selection) |
| $B$ | token batch size |
| Tile size | 32 elements |

---

## Section 1: Algorithmic Complexity

### Full Sort

A general-purpose sort (merge sort, quicksort) over $E = 256$ elements requires:

$$O(E \log_2 E) = 256 \times \log_2 256 = 256 \times 8 = 2{,}048 \text{ comparisons}$$

Full sort produces a complete ordering of all 256 experts. For top-$k$ selection, this is unnecessary; the bottom $E - k = 248$ positions are computed but never used.

### Partial Selection

A heap-based partial selection algorithm requires only:

$$O(E + k \log_2 k) = 256 + 8 \times \log_2 8 = 256 + 8 \times 3 = 256 + 24 = 280 \text{ comparisons}$$

The algorithm proceeds in three phases:

1. **Initialize min-heap:** Build a min-heap of size $k = 8$ from the first $k$ elements in $O(k) = O(8)$ operations.
2. **Scan remaining elements:** For each of the remaining $E - k = 248$ elements, compare against the heap minimum. If the current element is larger, replace the heap minimum and restore the heap property in $O(\log k) = O(3)$ operations. In the worst case: $248 \times 3 = 744$ operations; in practice, many elements are smaller than the current minimum and require only the comparison with no heap update.
3. **Extract top-k:** Pop all $k$ elements from the heap in $O(k \log k) = O(24)$ operations, yielding the sorted top-$k$.

**Total work budget:**

| Algorithm | Comparisons for $E=256$, $k=8$ |
|---|---|
| Full sort | $2{,}048$ |
| Partial selection (heap) | $\approx 280$ |
| Ratio | $2{,}048 / 280 \approx 7.3\times$ |

The partial selection algorithm performs approximately **7.3× less work** than a full sort for these parameters.

### Alternative: QuickSelect

The QuickSelect algorithm (based on the partition step of quicksort) finds the top-$k$ elements in $O(E)$ expected time without sorting them. For $E = 256$, expected comparisons $\approx 512$ (2× pass). It does not produce a sorted top-$k$ list; a final sort of the $k$ selected elements costs $O(k \log k) = O(24)$. Total: $\approx 536$ comparisons — better than full sort but worse than the heap-based approach for small $k$.

For $k = 8$, the min-heap approach is preferred due to its deterministic performance and small constant factor.

---

## Section 2: Batched Top-k

The router logit tensor after sigmoid activation has shape $[B, 256]$. Each row corresponds to an independent token; there are no cross-row dependencies. This structure enables full parallelism across the batch dimension.

**Parallelism properties:**

- $B$ rows can be processed simultaneously, limited only by available cores and L1 capacity.
- No synchronization is required between rows.
- The output tensors — indices $[B, 8]$ and scores $[B, 8]$ — are written independently per row.

**Vectorization:** Within each row, the heap comparison operations are sequential by nature (each comparison depends on the current heap state). However, an approximation used in practice is to partition the 256 elements into $E/k = 32$ groups of 8 and find the per-group maximum in parallel (SIMD), then run the heap selection over 32 candidates rather than 256. For $k = 8$ and natural group structure of size 8, this reduces the scan to 32 elements.

On Wormhole B0, each Tensix core has 1.5 MB of private L1 memory. A single row's logit data is $256 \times 2 = 512$ bytes in BF16, which is trivially small. Multiple rows can be buffered in L1 simultaneously: $1.5 \times 10^6 / 512 \approx 3{,}000$ rows fit in one core's L1, far exceeding any realistic batch size $B$.

---

## Section 3: Hardware Top-k Considerations on Tenstorrent

### Tile Granularity

On Wormhole B0, TTNN operates on tiles of size $32 \times 32$ elements. For a 1-D vector of $E = 256$ elements, the relevant tile dimension is 32:

$$256 / 32 = 8 \text{ tiles}$$

This is an exact integer: 256 logit elements fit into exactly 8 tiles of 32 elements each, with no padding required.

### Tile-Parallel Top-k

A tile-parallel approach distributes the 256 logit elements across multiple cores:

1. **Phase 1 — Local max per tile:** Each core holds one or more of the 8 tiles and computes the local top-$k$ candidates within its tile(s). For a tile of 32 elements selecting the local top-8, the heap approach requires $O(32 + 8 \times 3) = O(56)$ comparisons per tile.

2. **Phase 2 — Tree reduction:** The 8 per-tile top-8 candidate sets ($8 \times 8 = 64$ total candidates) are merged using a tree reduction. Merging two sorted lists of size 8 costs $O(16)$ comparisons; with $\log_2 8 = 3$ levels, total reduction cost is $\approx 3 \times 16 = 48$ comparisons.

3. **Phase 3 — Final extraction:** The global top-8 is extracted from the merged candidate list.

Total tile-parallel cost: $\approx 8 \times 56 + 48 = 496$ comparisons per token row, distributed across 8 cores. Wall-clock time is dominated by the Phase 1 cost per core ($\approx 56$ comparisons), which is $5\times$ lower than the sequential heap cost of $280$.

> **Tip:** TTNN may expose a hardware reduce primitive optimized for the 256-element, 8-tile case. Check the TTNN operator library for `ttnn.topk` with tile-aligned input sizes before implementing a custom kernel.

### L1 Footprint During Top-k

For a batch of $B = 32$ tokens:

- Input logit tensor (one token's row): $256 \times 2 = 512$ bytes
- Heap buffer (8 elements): $8 \times 2 = 16$ bytes
- Output indices (8 per token): $8 \times 4 = 32$ bytes (int32)
- Output scores (8 per token): $8 \times 2 = 16$ bytes

Per-token working set: $\approx 576$ bytes. For $B = 32$ tokens: $\approx 18$ KB. This is well within the 1.5 MB per-core L1 budget.

---

## Section 4: Fusing with Projection

The standard implementation sequence is:

```
matmul(x, W_r) → write [B, 256] to L1 → read [B, 256] from L1 → top-k pass → write [B, 8]
```

A fused approach eliminates the intermediate write-read cycle:

```
matmul tile-by-tile → update running top-k buffer → emit [B, 8] on completion
```

**How tile-level fusion works:**

The matmul over $W_r \in \mathbb{R}^{7168 \times 256}$ produces output in column tiles of 32 experts each. As each column tile of the output $g$ is computed, the 32 new logit values are compared against the current running min-heap of size 8. After all 8 column tiles are processed, the heap contains the global top-8.

**Memory savings from fusion:**

| Quantity | Value |
|---|---|
| Intermediate logit tensor at $B = 32$ | $32 \times 256 \times 2 = 16{,}384$ bytes = 16 KB |
| Top-k buffer (running heap) | $8 \times 2 = 16$ bytes |
| Reduction in working set | $\approx 1{,}000\times$ smaller |

At $B = 32$, the 16 KB intermediate tensor is small relative to L1 capacity, so the primary benefit is not memory pressure but **latency**: the fused kernel eliminates one kernel launch boundary and the associated write-read round-trip through L1.

**More important benefit:** Reducing the critical path from router computation to dispatch initiation. The all-to-all dispatch cannot start until the index tensor $[B, 8]$ is ready. Every cycle saved in the router directly reduces dispatch initiation latency.

Implementation caveats and TTNN support requirements for this fusion are covered in `router_kernel_fusion.md` Section 2, which is the authoritative home for kernel composition implementation guidance.

---

## Section 5: Worked Example

### Setup

- $B = 32$ tokens, $E = 256$ experts, $k = 8$
- Compare: full sort vs. partial heap selection, sequential vs. tile-parallel

### Full Sort Approach

Each of the $B = 32$ token rows requires a full sort of 256 elements:

$$\text{Total comparisons} = B \times O(E \log_2 E) = 32 \times 2{,}048 = 65{,}536$$

### Partial Heap Selection

Each row requires approximately 280 comparisons:

$$\text{Total comparisons} = B \times O(E + k \log_2 k) = 32 \times 280 = 8{,}960$$

### Savings

$$\frac{65{,}536}{8{,}960} \approx 7.3\times \text{ fewer comparisons with partial selection}$$

### Tile-Parallel Partial Selection

With 8 cores each handling one 32-element tile:

- Per-core Phase 1 comparisons: $32 + 8 \times 3 = 56$ per token row
- Phase 2 tree reduction: $\approx 48$ per token row
- Wall-clock cost dominated by Phase 1 per core: $32 \times 56 = 1{,}792$ core-operations (parallel)

### Actual Latency Context

The comparison operations above are illustrative. In practice, **the router projection matmul typically dominates total router latency**. The matmul over $W_r \in \mathbb{R}^{7168 \times 256}$ at $B = 32$ requires:

$$\text{FLOPs} = 2 \times B \times H \times E = 2 \times 32 \times 7{,}168 \times 256 = 117{,}440{,}512 \approx 117 \text{ MFLOPs}$$

At a conservative estimate of 10 TFLOPS effective throughput per device, the matmul takes $\approx 12 \mu\text{s}$. The top-k selection for $B = 32$ involves $\approx 8{,}960$ simple comparisons, which at 1 operation per cycle at 1 GHz takes $< 10 \mu\text{s}$ — comparable to the matmul. At larger $B$, the matmul grows linearly while top-k complexity also grows linearly, preserving the relative ratio.

The practical conclusion is that top-k selection is not negligible but is never the dominant term. The most impactful optimization is the fused kernel that eliminates the intermediate tensor materialization and kernel launch boundary between projection and top-k.

---

## References

- Blelloch, "Prefix Sums and Their Applications," CMU Technical Report CMU-CS-90-190, 1990. (Parallel selection algorithms.)
- Cormen et al., *Introduction to Algorithms*, 4th ed., MIT Press, 2022. (Heap-based selection, Chapter 9.)
- Tenstorrent, *TTNN Developer Guide*, 2024. (Tile granularity, core topology.)
- Chapter 5 of this guide: `router_forward_pass.md`, `router_kernel_fusion.md`.

---

**Next:** [weight_normalization.md](./weight_normalization.md)
