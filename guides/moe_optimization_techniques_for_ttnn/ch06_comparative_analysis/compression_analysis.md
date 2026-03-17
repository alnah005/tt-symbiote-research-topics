# Compression Analysis — MoE Optimization Chapter 6: Comparative Analysis — Pass 1

## Summary
- Files reviewed: `index.md`, `performance_comparison_matrix.md`, `memory_and_bandwidth_tradeoffs.md`, `decision_guide.md`
- Current line count: index.md ~124, performance_comparison_matrix.md ~152, memory_and_bandwidth_tradeoffs.md ~165, decision_guide.md ~200 — approximately 641 lines total
- Estimated post-compression: ~620 lines (minor trimming only; no exact duplicate blocks found)

---

## CRUCIAL Suggestions

None.

No exact duplicate blocks were found. No passage appears verbatim in two files such that removing one copy loses zero information. The two closest candidates (stale-sparsity-tensor warnings and the $32\times$ DRAM-bandwidth claims) each carry unique detail that makes them distinct — see MINOR section below.

---

## Load-Bearing Evidence

The following content is unique, technically precise, and must be preserved in full:

1. **Expert capacity formula and four-scenario table** (`performance_comparison_matrix.md`, lines 9–32): The derivation $C = \lceil k \times B \times S / E \rceil$ and the worked calculations for all four canonical scenarios (prefill large/small, decode large/small) are the quantitative backbone of the chapter. This appears nowhere else in equivalent detail.

2. **Non-monotonic latency analysis and ASCII diagram** (`performance_comparison_matrix.md`, lines 83–138): The two-component cost model (metadata read cost + compute cost) and the crossover at $\rho \approx 0.5$ with the $L_{SM}$ vs. $L_{BM}$ diagram are unique to this file. The warning that the crossover is NOT at $\rho = 1.0$ is a non-obvious correctness point.

3. **DRAM bandwidth waste derivation** (`memory_and_bandwidth_tradeoffs.md`, lines 13–48): The element-level calculation ($57{,}344 / 1{,}835{,}008 = 3.1\%$ useful data; 96.9% waste) with both decode and prefill cases contrasted is unique to this section. The corresponding note that sparse matmul's DRAM access scales with $\rho$ rather than the full tensor size is stated precisely here.

4. **L1 footprint table and per-core calculations** (`memory_and_bandwidth_tradeoffs.md`, lines 51–99): The per-core L1 budget estimate (~1.16 MB/core for prefill large approaching the 1.5 MB limit), the 448 KB sparsity tensor at prefill, and the 7 KB sparsity tensor at decode are quantitative findings that appear only in this file.

5. **T3K per-device sparsity tensor and device-skip behavior** (`memory_and_bandwidth_tradeoffs.md`, lines 102–130): The observation that devices receiving zero tokens skip their matmul entirely under sparse matmul (not possible with batched matmul), and the per-device shape $[32, 224]$ = 7 KB, is unique to this section.

6. **Sparsity tensor construction overhead** (`memory_and_bandwidth_tradeoffs.md`, lines 134–150): The comparison of 256 integer writes vs. $32 \times 32 \times 32 = 32{,}768$ MACs per tile (three orders of magnitude difference) is a unique quantitative argument for why rebuild cost is negligible.

7. **Runtime sparsity measurement code** (`decision_guide.md`, lines 62–88): The `measure_sparsity_ratio` Python function and its usage notes (including the B=32 case approaching $\rho = 1.0$) are the only executable reference implementation in the chapter.

8. **Hybrid strategy implementation code** (`decision_guide.md`, lines 96–139): The `get_matmul_strategy` function, the `MoELayer.forward` integration pattern, and the sparsity tensor lifecycle description (prefill: no tensor; first decode: construct; each subsequent decode: reconstruct) are unique and operational.

9. **Rule 3: Expert capacity threshold** (`decision_guide.md`, lines 43–54): The $C$-based thresholds ($C > 64$, $8 < C \leq 64$, $C \leq 8$) provide an alternative observable to $\rho$ for deployment contexts where $\rho$ is harder to compute. This mapping appears only here.

10. **Anti-Pattern 4: Stale tensor under dynamic batching** (`decision_guide.md`, lines 181–185): The specific failure mode when batch size changes between requests without rebuilding the sparsity tensor is unique to this anti-pattern entry.

11. **When to profile section** (`decision_guide.md`, lines 144–157): The four explicit conditions requiring profiling (non-standard $k$/$E$, borderline batch sizes, firmware updates, differing $H$/$d_{ff}$) and the profiling procedure (100 warmup + 1,000 measurement, median not mean) are unique operational guidance.

12. **Decision flowchart** (`index.md`, lines 46–78): The ASCII flowchart provides a navigable visual summary. While the decision logic it encodes overlaps with Rule 1 and Rule 2 in `decision_guide.md`, the flowchart format is a distinct, scan-friendly artifact for the chapter overview page and serves a different reader purpose than the prose rules.

---

## MINOR Suggestions

1. **Stale-sparsity-tensor warning is near-duplicated across two files.**
   - Location A: `memory_and_bandwidth_tradeoffs.md`, line 148 — Warning callout: "The sparsity tensor MUST be rebuilt at every decode step... reusing a stale sparsity tensor produces silent correctness errors (the wrong expert rows are skipped or computed). There is no valid caching of the sparsity tensor across decode steps."
   - Location B: `decision_guide.md`, lines 177–179 — Warning callout (Anti-Pattern 3): "A stale sparsity tensor produces silent correctness errors. The model output will be wrong, but no runtime error will indicate the cause. Always rebuild the sparsity tensor at every decode step. The rebuild cost is $O(B \times k)$ — negligible."
   - Unique content in A: "the wrong expert rows are skipped or computed" (mechanism); "no valid caching" phrasing.
   - Unique content in B: "no runtime error will indicate the cause" (debuggability note); "The model output will be wrong" (explicit output corruption statement).
   - Suggested trim: The warning in `memory_and_bandwidth_tradeoffs.md` (Location A) can be condensed to a single sentence cross-referencing Anti-Pattern 3 in `decision_guide.md`, since the full explanation with debuggability context lives there. Neither copy should be fully deleted — the memory file needs to flag the requirement; the decision guide provides the full treatment.

2. **$32\times$ DRAM-bandwidth claim appears in two files.**
   - Location A: `memory_and_bandwidth_tradeoffs.md`, line 47 — Tip callout: "sparse matmul reads approximately $32\times$ fewer activation bytes from DRAM than batched matmul."
   - Location B: `decision_guide.md`, line 173 — Warning callout (Anti-Pattern 2): "Batched matmul at decode $B=1$ performs approximately $32\times$ more DRAM reads for the activation tensor than necessary."
   - These are not exact duplicates (one frames it as a sparse matmul advantage; the other frames it as a batched matmul cost). Both are appropriate in their respective contexts (memory analysis vs. anti-pattern). No removal recommended; awareness of the repetition is sufficient.

3. **"Phase detection is a static property" statement appears twice.**
   - Location A: `index.md`, line 107 — "Phase detection (prefill vs. decode) is a static property of the inference loop; no per-step runtime measurement is required to apply the hybrid strategy."
   - Location B: `decision_guide.md`, line 18 — "Phase detection is a static property of the inference loop — a single `is_decode` boolean flag is sufficient. No per-step runtime measurement is required to apply this rule." (extended in lines 133–134)
   - Location B's version is more precise (mentions the `is_decode` boolean). The `index.md` sentence is a summary context appropriate for the chapter overview. No removal needed, but the `index.md` sentence could be shortened to "Phase detection requires only a static `is_decode` flag; see `decision_guide.md` Rule 1."

---

VERDICT: Crucial updates: no
