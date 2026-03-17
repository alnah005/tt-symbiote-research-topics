# Compression Analysis — Expert Parallelism Chapter 5: Routing Weight Optimization — Pass 1

## Summary
- Files reviewed: `index.md`, `router_forward_pass.md`, `topk_selection_efficiency.md`, `weight_normalization.md`, `router_kernel_fusion.md`
- Current line count: `index.md` ~73 lines, `router_forward_pass.md` ~193 lines, `topk_selection_efficiency.md` ~192 lines, `weight_normalization.md` ~206 lines, `router_kernel_fusion.md` ~381 lines (~1,045 total)
- Estimated post-compression: ~990 lines (remove ~55 lines of exact duplicates)

---

## CRUCIAL Suggestions

### 1. Top-k algorithmic complexity table

- **Location A:** `router_forward_pass.md` lines 119–123 — a 3-row table comparing full sort vs. partial selection, with columns "Method" and "Operations for E=256, k=8", giving values 2,048 comparisons vs. 280 comparisons and a ~7.3× speedup label.
- **Location B:** `topk_selection_efficiency.md` lines 40–44 — an identical 3-row table with the same column headers, same values (2,048 vs. 280 comparisons, ratio 2,048/280 ≈ 7.3×).
- **What to remove:** Delete the table in `router_forward_pass.md` (lines 119–123). The table in `topk_selection_efficiency.md` is the primary, detailed treatment. `router_forward_pass.md` already has a forward reference ("For a full quantitative comparison with batch size B=32, see `topk_selection_efficiency.md`") that can serve as the sole pointer.
- **Information lost:** None.

### 2. Heap algorithm 3-step description

- **Location A:** `router_forward_pass.md` lines 125–130 — a numbered 3-step description of the min-heap partial selection algorithm (initialize heap, scan remaining E−k=248 elements, extract top-k), including the totals O(k + (E−k) + k log k) = O(280).
- **Location B:** `topk_selection_efficiency.md` lines 34–46 — a more detailed numbered 3-step description of the identical algorithm, with the same phase names, the same values (k=8, E−k=248, O(k log k)=O(24)), and the same total formula O(E + k log k) = O(280).
- **What to remove:** Delete the 3-step algorithm block from `router_forward_pass.md` (lines 125–130). The version in `topk_selection_efficiency.md` is the authoritative, expanded treatment. The one-sentence summary in `router_forward_pass.md` ("The partial selection approach maintains a min-heap of size k:") can be replaced with a single-sentence reference to `topk_selection_efficiency.md`.
- **Information lost:** None.

### 3. Near-duplicate fusion warning (tile-level matmul + top-k caveat)

- **Location A:** `topk_selection_efficiency.md` lines 138–139 — warning block: "Tile-level fusion of matmul and top-k requires the kernel compiler to support partial accumulation into the top-k buffer as each output tile is produced. Verify TTNN kernel composition support before relying on this optimization. An alternative is to emit full rows as they complete (rather than tiles), which has weaker fusion but still avoids the two-pass structure."
- **Location B:** `router_kernel_fusion.md` lines 121–122 — warning block: "Tile-level fusion of matmul and top-k requires the kernel compiler to support partial output accumulation into the top-k buffer as each output tile completes. Verify TTNN kernel composition capabilities before relying on this optimization. If not available, a two-pass approach (full matmul, then fused sigmoid + topk) still eliminates one kernel boundary."
- **What to remove:** Delete the warning block from `topk_selection_efficiency.md` (lines 138–139). `topk_selection_efficiency.md` covers algorithmic complexity; the fusion warning is an implementation concern addressed in full in `router_kernel_fusion.md`. Replace with a one-line cross-reference: "See `router_kernel_fusion.md` Section 2 for implementation caveats."
- **Information lost:** None. The warning in `router_kernel_fusion.md` is marginally more specific (mentions the two-pass fallback explicitly), making it the superior copy.

---

## Load-Bearing Evidence

The following content is unique and technically precise; it must be preserved in full:

1. **`router_forward_pass.md` Section 5 — Numerical Precision:** The worked BF16 precision example (logits 1.0 vs. 0.9921875, sigmoid outputs 0.7311 vs. 0.7296, difference ~0.0015 vs. BF16 rounding unit ~0.0078) is unique to this file and not replicated elsewhere.

2. **`topk_selection_efficiency.md` Section 2 — Batched Top-k / Vectorization:** The approximation of partitioning 256 elements into E/k=32 groups of 8 for parallel SIMD pre-filtering, reducing the heap scan to 32 candidates, is unique to this file.

3. **`topk_selection_efficiency.md` Section 3 — Tile-Parallel Top-k cost model:** The concrete per-phase comparison counts (Phase 1: O(56) per tile, Phase 2 tree reduction: ~48, total wall-clock dominated by 56 per core), the 5× wall-clock improvement estimate, and the L1 footprint table (576 bytes per token, ~18 KB for B=32) are unique to this file.

4. **`topk_selection_efficiency.md` Section 5 — Actual Latency Context:** The FLOPs estimate (2 × 32 × 7168 × 256 = 117 MFLOPs), the ~12 µs matmul estimate at 10 TFLOPS, and the comparative latency argument that top-k is not negligible but never dominates, are unique to this file.

5. **`weight_normalization.md` Section 2 — Option A vs. Option B timing analysis:** The full quantitative comparison of dispatch payload overhead (512 bytes weight metadata vs. 458,752 bytes token embeddings), the L1 residency argument for the [B,k]=512-byte score tensor, and the deferred normalization rationale are unique to this file.

6. **`weight_normalization.md` Section 4 — Worked normalization example:** The concrete 8-score example ([0.82, 0.79, …, 0.58], sum=5.59, normalized weights summing to 1.000) is unique to this file.

7. **`router_kernel_fusion.md` Section 4 — Double-buffering:** The condition for benefit, memory cost table (with exact byte calculations for the two micro-batch buffers), and the double-buffered pipeline pseudocode are unique to this file.

8. **`router_kernel_fusion.md` Section 5 — INT8 quantization of W_r:** The INT8 size calculation (1.84 MB), the sigmoid perturbation bound derivation (Δσ ≤ 0.25 × 0.016 = 0.004), the per-column symmetric quantization implementation, and the calibration protocol (1,000+ tokens, <1% per-expert frequency shift threshold) are unique to this file.

9. **`index.md` — Qwen3.5-35B Router at a Glance table:** The consolidated parameter table (E=256, k=8, H=7168, N=8, W_r shape, W_r BF16 size, f_avg=1/32, CF=1.25) and the top-level formula $g = xW_r$ serve as the single reference for chapter-wide constants.

---

## MINOR Suggestions

1. **`router_forward_pass.md` Section 3, tip (line 134):** The tip about tile-parallel top-k and the 256/32=8 tile structure partially anticipates content in `topk_selection_efficiency.md` Section 3. The tip is short and serves as a navigation pointer; however, the sentence "With 80 Tensix cores available, the 8 output column tiles can be distributed across cores with good occupancy" in Section 1 of the same file (line 28) states a hardware detail also covered in `topk_selection_efficiency.md`. This is a minor overlap — neither copy is wrong, but one could be trimmed to a cross-reference.

2. **`weight_normalization.md` Section 1 — Softmax comparison paragraph (lines 50–52):** The brief comparison of sigmoid vs. softmax renormalization effect ("softmax selected scores sum to ~0.31 for a 256-way distribution") restates content covered in `router_forward_pass.md` Section 2 from a different angle. It adds modest value here as context for the normalization requirement; trimming to one sentence is optional.

3. **`router_kernel_fusion.md` Section 2 — Intermediate tensor size restatement:** Line 71 restates "$B \times 256 \times 2$ bytes. At $B = 32$: 16 KB" — the same value appears in `topk_selection_efficiency.md` Section 4 (lines 131–132). Both files use it in context-appropriate ways (one for fusion savings, one for memory analysis), so this is a minor rather than crucial redundancy.

4. **Constants tables (all files):** Every file opens with a per-file constants table. These are structurally appropriate for standalone readability but collectively redundant with the chapter-level table in `index.md`. No removal is recommended (they aid standalone file use), but a note that `index.md` is the canonical source would reduce confusion.

---

VERDICT: Crucial updates: yes

---

# Compression Analysis — Expert Parallelism Chapter 5: Routing Weight Optimization — Pass 2

## Summary
C1, C2, C3 applied correctly: **yes**

- **C1** (top-k complexity table removed from `router_forward_pass.md`): Confirmed. The 3-row comparison table (full sort 2,048 vs. partial 280 comparisons, 7.3× ratio) no longer appears in `router_forward_pass.md`. Section 3 now contains only a one-sentence summary mentioning the 7.3× figure with an explicit forward reference to `topk_selection_efficiency.md` at line 117. The authoritative table in `topk_selection_efficiency.md` Section 1 (lines 40–46) is intact.

- **C2** (min-heap 3-step description removed from `router_forward_pass.md`): Confirmed. No numbered heap algorithm phases appear in the current `router_forward_pass.md` Section 3. The section runs from lines 108–119 and contains only the high-level description plus the complexity summary sentence and pointer. The detailed 3-step description in `topk_selection_efficiency.md` Section 1 (lines 34–46) is intact.

- **C3** (TTNN kernel composition warning removed from `topk_selection_efficiency.md` Section 4): Confirmed. The warning block is absent. In its place, `topk_selection_efficiency.md` line 139 reads: "Implementation caveats and TTNN support requirements for this fusion are covered in `router_kernel_fusion.md` Section 2, which is the authoritative home for kernel composition implementation guidance." The authoritative warning in `router_kernel_fusion.md` Section 2 (lines 121–122) is intact.

No new duplication was introduced by these changes.

## CRUCIAL Suggestions
None.

## Load-Bearing Evidence

All critical unique content confirmed present after C1+C2+C3:

1. **`router_forward_pass.md` Section 5, lines 159–165** — BF16 precision worked example (logits 1.0 vs. 0.9921875, sigmoid outputs 0.7311 vs. 0.7296, margin 0.0015 vs. rounding unit 0.0078) is intact and unique to this file.

2. **`topk_selection_efficiency.md` Section 1, lines 34–52** — Complete min-heap 3-step algorithm with phase-by-phase comparison counts (O(k), O((E−k)×log k), O(k log k)), the full table (2,048 vs. 280, 7.3×), and the QuickSelect alternative analysis are all intact. This is the now-sole authoritative location for this content.

3. **`topk_selection_efficiency.md` Section 2, lines 56–68** — SIMD group-partition vectorization (partition 256 into E/k=32 groups of 8, parallel per-group max, heap scan over 32 candidates) is intact.

4. **`topk_selection_efficiency.md` Section 3, lines 72–105** — Tile-parallel top-k cost model (Phase 1: 56 comparisons/tile, Phase 2 tree reduction: 48, 5× wall-clock improvement, L1 footprint table: 576 bytes/token, ~18 KB at B=32) is intact.

5. **`topk_selection_efficiency.md` Section 4, lines 109–139** — Projection+top-k fusion memory savings table (16 KB intermediate vs. 16-byte heap buffer, ~1,000× working-set reduction) and latency benefit explanation are intact. Pointer to `router_kernel_fusion.md` Section 2 replaces the removed warning.

6. **`topk_selection_efficiency.md` Section 5, lines 143–183** — Actual latency context (117 MFLOPs, ~12 µs at 10 TFLOPS, top-k ~8,960 comparisons at B=32) is intact.

7. **`weight_normalization.md` Section 2, lines 58–96** — Option A vs. Option B dispatch payload analysis (512 bytes weight metadata vs. 458,752 bytes embeddings, 512-byte score tensor residency) is intact.

8. **`weight_normalization.md` Section 4, lines 181–195** — Worked normalization example ([0.82…0.58], sum=5.59, normalized weights summing to 1.000) is intact.

9. **`router_kernel_fusion.md` Section 2, lines 54–122** — Fused projection+top-k kernel pseudocode with savings table (16 KB eliminated, two kernel boundaries removed) and the authoritative TTNN composition warning are intact.

10. **`router_kernel_fusion.md` Section 4, lines 198–267** — Double-buffering memory cost table and pipeline pseudocode are intact.

11. **`router_kernel_fusion.md` Section 5, lines 271–370** — INT8 quantization of W_r: size (1.84 MB), sigmoid perturbation bound (Δσ ≤ 0.004), quantization implementation, and calibration protocol are intact.

12. **`index.md` lines 36–48** — Qwen3.5-35B Router at a Glance parameter table (E=256, k=8, H=7168, N=8, CF=1.25) is intact.

## MINOR Suggestions

1. **`router_forward_pass.md` Section 1, line 28** — The sentence "With 80 Tensix cores available, the 8 output column tiles can be distributed across cores with good occupancy" states a hardware mapping detail (80 Tensix cores, 8 column tiles) that is also covered in `topk_selection_efficiency.md` Section 3. The overlap is minor; the `router_forward_pass.md` instance appears in the context of the matmul weight layout (a different concern from top-k), but the 80-core / 8-tile claim is redundant. Trimming the parenthetical to a cross-reference would save one sentence.

2. **`router_forward_pass.md` Section 3, tip, lines 119–120** — The tip describing the 256/32=8 tile structure for top-k anticipates `topk_selection_efficiency.md` Section 3 content. It functions as a navigation primer but replicates the tile-count calculation. Could be shortened to the final pointer sentence only.

3. **`weight_normalization.md` Section 1, lines 50–52** — The softmax comparison paragraph (top-8 selected scores sum to ~0.31 for a 256-way distribution) restates content from `router_forward_pass.md` Section 2. Minor contextual duplication; trimming to one sentence ("Softmax routing requires the same renormalization formula, but as a minor rather than mandatory correction") is optional.

4. **`router_kernel_fusion.md` Section 2, line 71** — The intermediate tensor size restatement ("$B \times 256 \times 2$ bytes. At $B = 32$: 16 KB") also appears in `topk_selection_efficiency.md` Section 4. Both files use it in context-appropriate ways; this is an acceptable minor redundancy.

5. **Per-file constants tables** — All five files open with a local constants table. These are appropriate for standalone readability but are collectively redundant with `index.md`'s chapter-level table. A one-line note in each file ("See `index.md` for the canonical chapter constants") would reduce confusion without eliminating the tables.

VERDICT: Crucial updates: no
