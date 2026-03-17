# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 1

1. **All-reduce bandwidth formula is materially overestimated (`wormhole_architecture.md`, Section 4.3).** The formula presented is:

   ```
   All-reduce bandwidth (ring, 8 chips) ≈ (N-1)/N × link_bandwidth
                                        ≈ 7/8 × 12.5 GB/s ≈ 10.9 GB/s per step
   ```

   This is wrong. For a ring all-reduce (reduce-scatter + all-gather), the effective bandwidth at which a tensor of size S is fully reduced across N nodes is:

   ```
   effective_bandwidth = N / (2 × (N-1)) × link_bandwidth
   ```

   For N=8: `8 / 14 × 12.5 GB/s ≈ 7.1 GB/s`, not 10.9 GB/s. The formula `(N-1)/N × B` describes the fraction of a single *step's* bandwidth that is useful, not the end-to-end effective throughput. A reader using 10.9 GB/s to budget communication-compute overlap would underestimate all-reduce latency by roughly 1.5×, leading to incorrect conclusions about when expert parallelism overlaps cleanly with compute in Chapters 6 and 7.

2. **L1 B-buffer formula is missing the `in0_block_w` factor (`matmul_fundamentals_in_ttnn.md`, Section 2.2).** The formula given is:

   ```
   L1 for B input = 2 × per_core_N × (tile_size bytes)    (B tiles for one K step)
   ```

   This is valid only when `in0_block_w = 1`. For one K-step of width `in0_block_w`, the B buffer holds `in0_block_w × per_core_N` tiles (double-buffered: `2 × in0_block_w × per_core_N`). In the Section 5 worked example, `in0_block_w = 4`, so the actual B buffer is `2 × 4 × 64 × 1 KB = 512 KB`, not the 128 KB that the Section 2.2 formula would produce. The Section 5 budget check repeats the same error (`2 × per_core_N × 1 KB = 128 KB`). A reader applying this formula to their own config would underestimate B-buffer L1 usage by a factor of `in0_block_w`, potentially sizing a config that overflows L1 at runtime. The A-buffer formula has the same omission (`2 × per_core_M` rather than `2 × per_core_M × in0_block_w`), but the impact is smaller because per_core_M is typically small.

3. **`out_subblock_h` and `out_subblock_w` definitions do not specify the divisibility requirement on `per_core_N` with respect to the tile count, only with respect to the subblock.** This is a minor structural gap: Section 3.3 states "per_core_N must be divisible by out_subblock_w," which is correct, but nowhere does the chapter state that `out_subblock_w` itself must be a positive integer and that `per_core_N / out_subblock_w` is the number of subblock columns iterated by the packer. Without this, a downstream reader in Ch3/Ch4/Ch7 who needs to reason about packer loop count cannot do so from the definitions given here. The definition is complete enough to avoid wrong answers on subblock validation, but not complete enough for the "unambiguous" standard stated in the plan.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 2

1. **A-buffer formula is missing the `in0_block_w` factor (`matmul_fundamentals_in_ttnn.md`, Section 2.2).** The formula given is:

   ```
   L1 for A input = 2 × per_core_M × (tile_size bytes)
   ```

   This is wrong for any `in0_block_w > 1`. For one K-step of block width `in0_block_w`, a core holds `per_core_M × in0_block_w` A tiles (double-buffered: `2 × per_core_M × in0_block_w × tile_size`). The Section 2.2 inline example uses `in0_block_w=4`, so the correct A buffer is `2×1×4×2 KB = 16 KB`, not 4 KB. Section 5's worked example repeats the same value (`A buffer: 2 × 1 × 2 KB = 4 KB`). For configs with large `per_core_M` and large `in0_block_w` (e.g., `per_core_M=8`, `in0_block_w=8`), the underestimate is 8×, which would lead a reader to believe an L1 budget is safe when it overflows at runtime.

2. **BFP8 per-value mantissa described as 7 bits (`ttnn_programming_model.md`, Section 1.2).** The text states: "each value has its own 7-bit mantissa." In Tenstorrent's BFP8 (bfloat8_b) format, each value stores an 8-bit mantissa field; the shared block exponent is stored separately once per block of 16 values (not per element). Stating 7 bits per value is incorrect and conflates the BF16 mantissa width (7 bits) with BFP8's. A reader implementing BFP8 quantization or computing tile memory sizes from first principles using "7 mantissa bits + sign = 8 bits total" might happen to get the tile-size byte count right by coincidence, but would have a wrong mental model of the format, potentially causing errors when reasoning about quantization error magnitude or format compatibility.

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 3

1. **Section 2.2 inline worked example still has wrong A-buffer value (`matmul_fundamentals_in_ttnn.md`, line 104).** The formula on line 93 was correctly updated to `2 × per_core_M × in0_block_w × tile_size`, but the inline numerical example immediately below still reads:

   ```
   A buffer: 2 × 1 × 2 KB = 4 KB
   ```

   With `per_core_M = 1` and `in0_block_w = 4` (both stated on line 101), the correct value is `2 × 1 × 4 × 2 KB = 16 KB`. The `in0_block_w = 4` factor was dropped from the arithmetic. The total on line 107 (`~1156 KB`) is therefore also understated by 12 KB (should be `~1168 KB`). The conclusion (exceeds budget) is unchanged, but a reader copying this arithmetic to estimate A-buffer size for their own config with larger `per_core_M` or `in0_block_w` would underestimate L1 usage by a factor of `in0_block_w`.

2. **BFP8 mantissa width stated incorrectly as "8-bit" (`ttnn_programming_model.md`, line 46).** The Pass 2 fix changed "7-bit mantissa" to "8-bit mantissa," but Tenstorrent's `bfloat8_b` format stores each value as 1 sign bit + 7 mantissa bits = 8 bits total per value, with a shared block exponent stored once per block of 16 values. The mantissa field is 7 bits wide, not 8. The "8" in the format name and in "8-bit block floating point" refers to the total per-value storage width, not the mantissa width. The tile byte count (`32 × 32 × 1 = 1 KB`) is still correct because it follows from the 8-bits-per-value total, not the mantissa count. However, a reader who takes "8-bit mantissa" literally will compute the wrong number of representable mantissa levels (256 instead of 128), and will incorrectly model quantization noise magnitude — relevant to the chapter's own claim that BFP8 causes "typically < 0.5% degradation."

3. **Physical die described as "10×12 grid of Tensix cores" (`wormhole_architecture.md`, line 93).** The 10×12 = 120 figure describes the full NOC grid including DRAM controller tiles, ethernet tiles, and PCIe tiles — not Tensix cores specifically. The Wormhole B0 die contains 80 Tensix cores (8×10), not 120. The file partially hedges by noting some grid positions are non-Tensix, but still opens with "the physical die contains a 10×12 grid of Tensix cores … giving 120 Tensix cores." A reader who internalizes "120 Tensix cores minus some system tiles" will arrive at an incorrect model of raw compute capacity and may miscalculate peak FLOP estimates or incorrectly scale performance projections for multi-expert grid assignments discussed in Chapter 3.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 4

1. **BFP8 shared exponent scope stated as "per tile" but Tenstorrent's `bfloat8_b` uses a per-row-of-16 exponent (`ttnn_programming_model.md`, Section 1.2).** The text now reads: "values within a tile share a common block exponent per tile." In Tenstorrent's actual BFP8 (`bfloat8_b`) implementation, the shared exponent is stored once per 16-value datum (one row of a sub-block within the tile), not once per entire 1024-value tile. Describing the exponent as "per tile" overstates the sharing and understates quantization error: with a single exponent per 1024 values, values spanning a very wide dynamic range would lose far more precision than they actually do with per-16-value exponents. A reader relying on this characterization to reason about BFP8 accuracy loss (the chapter itself claims "typically < 0.5% degradation") will have a materially wrong model of how quantization error accumulates in expert FFN weights.

2. **`in0_block_w` divisibility constraint absent from Section 1.2 and the grid-mapping formulas, yet asserted in the summary table (`matmul_fundamentals_in_ttnn.md`, line 387).** The summary table states "`in0_block_w`: Must divide `K_t`." This constraint is never stated or justified in the body of the file — Section 1.3 discusses the K-loop but does not state this requirement. Section 2.2 introduces `in0_block_w` only in the context of a worked buffer formula. A reader following only the body text (as a downstream implementer would) would not know to enforce `K_t % in0_block_w == 0`, producing a mismatched inner K-loop that either silently drops K tiles or raises a runtime error. The constraint is load-bearing for correct implementation and belongs in the body where `in0_block_w` is first defined.

**No further correctness issues found.** The three fixes applied in Pass 4 (A-buffer arithmetic, BFP8 bit layout, and Tensix core count) are all correctly reflected in the current files. The all-reduce formula (`8 / (2×7) × 12.5 GB/s ≈ 7.1 GB/s`) and the Section 5 mixed-dtype L1 budget (`16 + 512 + 128 = 656 KB`) are both arithmetically correct.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 5

1. **BFP8 tile size ignores per-block exponent bytes in both files (`ttnn_programming_model.md`, Section 1.2; `wormhole_architecture.md`, Summary table).** Both files state a BFP8 tile is `32 × 32 × 1 = 1024 bytes = 1 KB`. However, the BFP8 format (now correctly described in Section 1.2) stores one shared exponent byte per 16-value block, and a 32×32 tile contains 64 such blocks. That adds 64 exponent bytes per tile, giving an actual tile storage of 1024 + 64 = **1088 bytes**, not 1024 bytes. The "1 KB" shorthand is a ~6% underestimate. This error propagates directly into every L1 budget calculation that uses a BFP8 tile count: the Section 5 worked example's B-buffer figure (`2 × 4 × 64 × 1 KB = 512 KB`) becomes `2 × 4 × 64 × 1.0625 KB ≈ 544 KB`, and the total rises from 656 KB to ~688 KB. At the margin (e.g., a reader trying to squeeze the largest possible `per_core_N` within the ~1 MB budget), the ~6% error can cause a config that appears safe on paper to overflow L1 at runtime. The claim that BFP8 "halves L1 pressure relative to BF16" is also slightly overstated (the actual ratio is ~1.88×, not 2×).

   **Fix:** In `ttnn_programming_model.md` Section 1.2, replace the tile size statement with: "Each tile stores 1024 value bytes plus 64 exponent bytes (one per 16-value block) = **1088 bytes ≈ 1.06 KB** of total storage." Update the "halves L1 pressure" claim to "reduces L1 pressure by roughly 47% relative to BF16." In `wormhole_architecture.md` summary table, update the bfloat8_b row to reflect 1088 bytes. Ensure all L1 budget examples that use BFP8 tile sizes are recalculated with 1088 bytes rather than 1024 bytes.

2. **BFP8 exponent grouping described with geometrically misleading phrasing (`ttnn_programming_model.md`, Section 1.2).** The text now reads: "shared block exponent stored once per **16-value block** (one row of a sub-tile)." A 32×32 tile has rows of 32 values, so a group of 16 contiguous values is a *half*-row of the tile, not "one row." The phrase "one row of a sub-tile" implies a 16-column sub-tile structure that is never defined or stated anywhere in the file. A reader who takes "row" at face value would compute 32 exponents per tile (one per 32-element row) rather than the 64 stated in the same sentence, creating an internal contradiction. This geometry confusion was the source of the Pass 4 issue; the exponent count fix was applied but the supporting description still uses incorrect row terminology.

   **Fix:** Replace "(one row of a sub-tile)" with "(one half-row of the 32×32 tile, i.e., 16 consecutive values in row-major order)." This makes the 64-exponents-per-tile count self-evident: 32 rows × 2 half-rows per row = 64 blocks.

3. **Inconsistent minimum grid requirement for `MatmulMultiCoreReuseMultiCastProgramConfig` across three locations in the same file (`matmul_fundamentals_in_ttnn.md`, Section 4.1 and Section 4.3).** Three statements in the same section give different minimum grid requirements:
   - Key characteristics (line 215): "Requires the output grid to be at least 2 cores **in one dimension**" — a 1×2 or 2×1 grid would satisfy this.
   - "When to use" bullet (line 243): "`M_t` × `N_t` is large enough to fill **at least a 2×2 or 2×4 core grid**" — requires both dimensions ≥ 2.
   - Decision flowchart (line 288): "Is `M_t` × `N_t` large enough to fill **a 2×N grid (N ≥ 2)**?" — requires Y ≥ 2 and X ≥ 2.

   These three formulations are mutually inconsistent. A 1×4 or 4×1 grid satisfies the key characteristics statement but fails the "when to use" and flowchart conditions. A reader choosing a grid for a workload with `M_t = 1` but large `N_t` (a typical decode-time scenario with many N columns) would get conflicting guidance: the key characteristics text says multicast is valid if at least one dimension is ≥ 2, but the flowchart sends them to `MatmulMultiCoreProgramConfig`. The correct answer (whether 1×N qualifies) is not determinable from the text alone.

   **Fix:** Decide on a single consistent threshold and apply it in all three places. If the correct minimum is a 2D grid with both dimensions ≥ 2 (which the flowchart implies), update the key characteristics bullet to read "at least a 2×2 grid" and remove the "in one dimension" qualifier. If a 1×N grid is genuinely valid for multicast along a single row, update the "when to use" bullet and flowchart accordingly.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 4 Fixes Applied

1. `ttnn_programming_model.md` — BFP8 shared exponent scope corrected: changed "per tile" to "per 16-value block". A 32×32 tile contains 64 separate shared exponents (one per row of 16 values), not one per tile. Updated all references to the exponent granularity in Section 1.2.

2. `matmul_fundamentals_in_ttnn.md` — Added `K_t % in0_block_w == 0` divisibility constraint to the body text where `in0_block_w` is first defined. The constraint is now stated and justified in the K-loop section, consistent with the summary table entry.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 5 Fixes Applied

1. `ttnn_programming_model.md` and `wormhole_architecture.md` — BFP8 tile size corrected from 1024 bytes to 1088 bytes (1024 value bytes + 64 exponent bytes). "Halves L1 pressure" updated to "~1.88× reduction." Section 5 worked example BFP8 buffer sizes updated accordingly.

2. `ttnn_programming_model.md` — BFP8 exponent granularity description corrected: "(one row of a sub-tile)" changed to "(one half-row of the 32×32 tile, i.e., 16 consecutive values in row-major order)." The count of 64 exponents per tile is unchanged and correct.

3. `matmul_fundamentals_in_ttnn.md` — Three inconsistent minimum grid statements reconciled: all now state the minimum is a 2×2 core grid (both M and N dimensions ≥ 2). The incorrect "at least 2 cores in one dimension" bullet was corrected.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 6

1. **Mixed-dtype L1 formula parenthetical is ambiguous and will produce wrong totals for mixed-dtype configs (`matmul_fundamentals_in_ttnn.md`, Section 2.2).** The formula block ends with:

   > `(for BF16; substitute 1088 bytes for BFP8 tiles instead of 2048 bytes)`

   This note does not distinguish *which* buffers should use 1088 bytes. A reader applying it literally will substitute 1088 bytes into all three terms — A, B, and C — regardless of per-tensor dtype. For the Section 5 scenario (BF16 activations, BFP8 weights, BF16 output), the per-buffer tile sizes should be:
   - A buffer: **2048 bytes** (BF16 activations)
   - B buffer: **1088 bytes** (BFP8 weights)
   - C buffer: **2048 bytes** (BF16 output, set by `dtype=ttnn.bfloat16` in the matmul call)

   If a reader substitutes 1088 bytes for all three terms, they compute:
   ```
   A: 2 × 1 × 4 × 1088 = 8,704 bytes ≈ 8.5 KB  (should be 16 KB)
   B: 2 × 4 × 64 × 1088 = 557,056 bytes = 544 KB  (correct)
   C: 1 × 64 × 1088 = 69,632 bytes = 68 KB  (should be 128 KB)
   Total: ~620 KB  (should be 688 KB)
   ```

   The ~68 KB underestimate on the total is not large enough to reverse the "fits within budget" conclusion at this particular configuration, but for tighter configs (large `per_core_N`, large `per_core_M`) the C-buffer underestimate alone can tip a budget decision. More importantly, the wrong substitution yields a structurally incorrect mental model: the reader may believe they can inflate `per_core_N` further than is actually safe because they are underestimating the BF16 C-buffer cost.

   The Section 5 example implicitly does the correct per-tensor substitution (using 2048 for A and C, 1088 for B), but the formula in Section 2.2 is never updated to reflect this, leaving a direct contradiction between the Section 2.2 parenthetical and the Section 5 arithmetic.

   **Downstream consequence:** A reader who derives their own L1 budget from the Section 2.2 formula for a mixed BFP8-weights / BF16-activations config (the most common MoE production pattern) will underestimate the C buffer by ~47% and the A buffer by ~47%, arriving at a total that is ~10–15% too low depending on the ratio of buffer sizes. At large `per_core_N` (e.g., 128 tiles), the C buffer alone is 256 KB at BF16 vs. 136 KB at the incorrectly applied BFP8 size — a 120 KB error that can cause runtime L1 overflow for configs that appear safe on paper.

   **Fix:** Replace the single parenthetical with an explicit note that identifies which buffer uses which dtype, for example:

   > `(Substitute tile_size = 2048 bytes for BF16 tensors and 1088 bytes for BFP8 tensors per buffer. In mixed-dtype configs such as BF16 activations with BFP8 weights, use 2048 bytes for the A and C buffers and 1088 bytes for the B buffer.)`

   Alternatively, split the formula into a BF16-only version and a mixed-dtype version that mirrors the Section 5 example directly.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 6 Fixes Applied

1. `matmul_fundamentals_in_ttnn.md` Section 2.2 — Mixed-dtype buffer substitution note made specific: clarified that only the B buffer uses 1088 bytes/tile for BFP8 weights; A and C buffers remain at 2048 bytes/tile when using BF16 activations/output. Section 5 worked example verified and corrected to use per-buffer dtype consistently.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 7

1. **BFP8 tile count derived from rounded BF16 figure rather than from actual L1 capacity (`wormhole_architecture.md`, Section 2.1).** The text states: "a single core can hold roughly 750 bfloat16 tiles or approximately 1410 bfloat8_b tiles."

   The exact calculations from the stated L1 capacity of 1.5 MB = 1,572,864 bytes are:
   - BF16: 1,572,864 / 2,048 = **768 tiles** exactly
   - BFP8: 1,572,864 / 1,088 ≈ **1,446 tiles**

   "Roughly 750" for BF16 is a ~2.4% underestimate of 768. The BFP8 figure of "approximately 1410" appears to have been computed as `750 × 1.88 ≈ 1410` — i.e., it was derived by multiplying the already-rounded BF16 count (750) by the 1.88× ratio, rather than by dividing the actual L1 byte count (1,572,864) by the actual BFP8 tile size (1,088 bytes). The correct figure is ~1,446, not ~1,410 — a ~2.5% underestimate.

   **Downstream consequence:** The two figures are internally consistent with each other (1410/750 ≈ 1.88) but are both systematically low relative to the 1.5 MB capacity stated in the same paragraph. A reader who uses these tile counts to manually validate the 1.88× ratio will get a consistent answer, which masks the error. However, a reader who uses "roughly 750 BF16 tiles" or "approximately 1410 BFP8 tiles" as a hard capacity limit when reasoning about how many tiles of data can reside in L1 simultaneously will underestimate capacity by ~2.4–2.5%, which can cause unnecessary grid enlargement or configuration conservatism for large-N workloads.

   **Fix:** Replace "roughly 750 bfloat16 tiles" with "roughly 768 bfloat16 tiles" (1,572,864 / 2,048 = 768, exact) and replace "approximately 1410 bfloat8_b tiles" with "approximately 1445 bfloat8_b tiles" (1,572,864 / 1,088 ≈ 1445.6). Both figures should be derived directly from dividing the 1.5 MB capacity by the respective tile size, not from multiplying one rounded figure by the compression ratio.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 7 Fixes Applied

1. `wormhole_architecture.md` Section 2.1 — BF16 and BFP8 tile counts corrected to be derived directly from L1 capacity: BF16: 1,572,864 / 2,048 = 768 tiles (was ~750); BFP8: 1,572,864 / 1,088 = ~1,445 tiles (was ~1,410). Ratio ~1.88× unchanged.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 8

1. **Stale "halves" claim in code comment contradicts corrected prose (`ttnn_programming_model.md`, line 53).**

   The Pass 5 fix correctly updated the prose description of BFP8 memory savings in Section 1.2 from "halves L1 pressure" to "reduces L1 pressure by ~1.88× relative to BF16." However, the Python code comment on the very next code block was not updated and still reads:

   ```python
   # BFP8 weight tensor — halves L1 and DRAM pressure vs. BF16
   ```

   "Halves" implies a 2× reduction. The actual tile-size ratio is 2048 / 1088 ≈ 1.882, which rounds to ~1.88×, not 2×. The prose directly above this comment (line 46) says "~1.88×" and the inline code annotation on line 56 gives the exact values ("1088 bytes per tile instead of 2048 bytes (BF16)"), so there is a direct contradiction within the same code block between the comment header ("halves") and the per-line annotation (1088 vs. 2048, which implies ~1.88×).

   **Downstream consequence:** A reader who reads the comment summary and does not recompute from the explicit byte values will carry forward "halves" (2×) as the compression factor. When they apply this to an L1 budget calculation — for example, estimating how much more `per_core_N` is achievable by switching from BF16 to BFP8 weights — they will overestimate the benefit by ~6% (computing 2× instead of 1.88×). For large `per_core_N` values close to the L1 boundary this can cause a configuration that appears safe on paper (under the 2× assumption) to overflow L1 at runtime.

   **Fix:** Replace the code comment on line 53 with:
   ```python
   # BFP8 weight tensor — reduces L1 and DRAM pressure by ~1.88× vs. BF16
   ```
   This makes the comment consistent with both the prose on line 46 and the per-line annotation on line 56.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 8 Fixes Applied

1. `ttnn_programming_model.md` line ~53 — Code comment updated from "halves L1 and DRAM pressure" to "reduces L1 and DRAM pressure by ~1.88×" to match the corrected prose description.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 9

1. **Heuristic 2 in Section 3.4 is missing a required `out_subblock_h = 1` qualifier and will produce invalid configs when `out_subblock_h > 1` (`matmul_fundamentals_in_ttnn.md`, Section 3.4).**

   The heuristic as written states:

   > "Use `out_subblock_w = per_core_N` when `per_core_N ≤ 8`. This keeps the entire output row in registers for the full K loop, minimizing L1 writes."

   This is stated as an unconditional rule gated only on `per_core_N ≤ 8`, with no reference to `out_subblock_h`. However, the register file constraint from Section 3.2 is `out_subblock_h × out_subblock_w ≤ 8`. Setting `out_subblock_w = per_core_N` is only safe when `out_subblock_h = 1`. For any `out_subblock_h ≥ 2`, applying this heuristic produces a product that exceeds the limit of 8 for any `per_core_N ≥ 5`.

   Concrete example of the failure mode: `per_core_M = 4`, `per_core_N = 8`. Section 3.4 Heuristic 1 says "prefer wider subblocks over taller ones when `per_core_N >> per_core_M`," which would steer a reader toward `out_subblock_h = 2`. Heuristic 2 then says "use `out_subblock_w = 8` since `per_core_N = 8 ≤ 8`." The result is `out_subblock_h = 2, out_subblock_w = 8`, giving a product of 16, which **violates Section 3.2's hard limit of 8** and will cause a TTNN validation error at runtime (or an incorrect kernel if validation is not exhaustive). A reader following both heuristics in sequence for this common configuration (`per_core_M = 4`, `per_core_N = 8`) is guaranteed to produce an invalid config.

   The Section 3.4 Tip immediately following Heuristic 3 correctly anchors with `out_subblock_h = 1` before setting `out_subblock_w = min(per_core_N, 8)`, which is always safe — but this qualification is not carried back into Heuristic 2 itself.

   **Downstream consequence:** Any reader who follows Heuristic 1 (choose non-unit `out_subblock_h` to improve register utilization for `per_core_M > 1`) and then follows Heuristic 2 (set `out_subblock_w = per_core_N` unconditionally) for a shape where `per_core_N > 8 / out_subblock_h` will configure an invalid subblock. For the most common large-N MoE shapes (`per_core_N = 64` or `per_core_N = 16`) the product of heuristics would be obvious nonsense (product far exceeds 8), but for `per_core_N` in the range 5–8 with `out_subblock_h = 2`, the mistake is subtle and produces a product of 10–16 — invalid, but only marginally over the limit, making it easy to miss without carefully re-checking the product. Chapters 3, 4, and 7 build on the subblock vocabulary defined here; an implementer who arrives at Chapter 3 with a broken subblock selection rule will produce incorrect kernels.

   **Fix:** Add the `out_subblock_h = 1` qualifier to Heuristic 2. Replace:

   > "Use `out_subblock_w = per_core_N` when `per_core_N ≤ 8`."

   With:

   > "Use `out_subblock_w = per_core_N` when `per_core_N ≤ 8` **and `out_subblock_h = 1`** (i.e., when `per_core_M = 1` or when the product constraint forces `out_subblock_h = 1`). For `out_subblock_h ≥ 2`, the maximum safe `out_subblock_w` is `floor(8 / out_subblock_h)`."

   This makes Heuristic 2 consistent with the Section 3.2 product constraint and with the Tip at the end of Section 3.4.

---

# Agent A Fix — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 9 Fixes Applied

1. `matmul_fundamentals_in_ttnn.md` Section 3.4 Heuristic 2 — Added `out_subblock_h = 1` qualifier and general safe bound `out_subblock_w ≤ floor(8 / out_subblock_h)`. The `out_subblock_w = per_core_N` shortcut is now explicitly qualified as only valid when `out_subblock_h = 1`.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 10

**No feedback — chapter approved.**

All prior fixes (Passes 1–9) were verified against the current file state:

- `wormhole_architecture.md`: all-reduce formula (`8 / (2×7) × 12.5 GB/s ≈ 7.1 GB/s`), BFP8 tile size (1088 bytes), BF16 tile count (768), BFP8 tile count (~1,445), 80 Tensix cores, `~1.88×` ratio in the L1 Tip — all correct and consistent.
- `ttnn_programming_model.md`: BFP8 format description (1 sign + 7 mantissa bits = 8 bits total, shared exponent per 16-value half-row, 64 exponents per tile, 1088 bytes total), `~1.88×` code comment, mixed-dtype note specificity — all correct and consistent.
- `matmul_fundamentals_in_ttnn.md`: A-buffer formula includes `in0_block_w`, inline BF16 example arithmetic (16 KB + 1024 KB + 128 KB = 1168 KB), Section 5 mixed-dtype budget (16 KB + 544 KB + 128 KB = 688 KB), `K_t % in0_block_w == 0` constraint stated in body (Section 1.3), Heuristic 2 qualified with `out_subblock_h = 1` and general bound `out_subblock_w ≤ floor(8 / out_subblock_h)`, minimum 2×2 grid stated consistently across Section 4.1 key characteristics, "When to use" bullet, and Section 4.3 flowchart — all correct and consistent.
- `index.md`: cross-references and learning objectives are consistent with the current content of all three section files.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 11

**Compression changes C1, C2, and C3 verified. One new correctness issue introduced by C3.**

---

**C1 — `wormhole_architecture.md` Summary table, "Tile sizes" row:** The cross-reference "See ttnn_programming_model.md Section 1.2" is accurate. Section 1.2 of `ttnn_programming_model.md` is titled "Dtypes: bfloat16 and bfloat8_b" and contains both tile byte sizes (2048 bytes for BF16, 1088 bytes for BFP8) and the 32×32 dimension (stated in the BF16 entry as `32 × 32 × 2 = 2048 bytes`). No information was lost and the reference resolves correctly.

**C2 — `matmul_fundamentals_in_ttnn.md` Section 2.2, compressed BFP8 parenthetical:** The one-line replacement correctly preserves both the mixed-dtype case ("in the canonical mixed config — BFP8 weights, BF16 activations/output — only the B buffer uses 1088 bytes/tile") and the pure-BFP8 case ("all three use 1088 bytes/tile for pure BFP8"). This is consistent with the Pass 6 fix and does not introduce any new ambiguity. No information was lost.

**C3 — `index.md` Key Terms, "Interleaved placement" and "Sharded placement" entries:**

**Issue found.**

**`index.md`, Key Terms section (line 59 and lines 69–70) — framing sentence contradicts pointer-only entries after C3.**

The section opens with: "The following terms are defined here and used without re-explanation in later chapters." After C3, the "Interleaved placement" and "Sharded placement" entries no longer contain definitions — they read "see `ttnn_programming_model.md` Section 2.3." These two entries are therefore not defined "here" in `index.md`; they are defined in a different file. The framing sentence is now false for two of the ten bulleted terms.

Note that the `out_subblock_h`/`out_subblock_w` entry (line 66) has the same pattern ("defined fully in `matmul_fundamentals_in_ttnn.md`"), so this was already an existing inconsistency for one entry. C3 extends it to three entries out of ten without updating the introductory claim.

**Consequence:** A reader who takes the framing sentence at face value will expect to find working definitions of "Interleaved placement" and "Sharded placement" in `index.md` and may not follow the cross-references. If they are using `index.md` as a standalone glossary lookup during later chapters (a natural use pattern for an index file), they will find only a redirect rather than a definition and may not realize the redirect is necessary. This is a low-severity usability problem, not a technical incorrectness in the underlying content, but it is an inaccuracy introduced directly by C3.

**Fix:** Update line 59 from "The following terms are defined here and used without re-explanation in later chapters" to "The following terms are introduced in this chapter and used without re-explanation in later chapters; some are defined inline and others point to their full definitions in the section files below." Alternatively, shorten to: "The following terms are used without re-explanation in later chapters:" — which is accurate regardless of whether the definition appears inline or via cross-reference.

---

# Agent B Review — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 12

**No feedback — chapter approved.**

All four files were read in full and verified against the complete fix history (Passes 1–11):

- `index.md` line 59: framing sentence reads "The following key terms are used without re-explanation in later chapters (some defined inline below, others defined in the file indicated):" — correctly resolves the Pass 11 issue; no longer falsely claims all terms are defined inline.
- `wormhole_architecture.md`: all-reduce formula (`8 / (2×7) × 12.5 GB/s ≈ 7.1 GB/s`), BF16 tile count (768, derived from 1,572,864 / 2,048), BFP8 tile count (~1,445, derived from 1,572,864 / 1,088), 80 Tensix cores (8×10), `~1.88×` ratio in the L1 Tip, BFP8 tile size (1088 bytes) in Section 2.1 — all correct and internally consistent.
- `ttnn_programming_model.md`: BFP8 format description (1 sign bit + 7 mantissa bits = 8 bits total per value; shared block exponent per 16-value block, one half-row of the 32×32 tile; 64 exponents per tile; 1088 bytes total tile storage; ~1.88× reduction vs. BF16), code comment on line 53 ("~1.88×"), mixed-dtype note in Section 2.2 parenthetical — all correct and consistent.
- `matmul_fundamentals_in_ttnn.md`: A-buffer formula includes `in0_block_w` factor; inline BF16 example arithmetic (16 KB + 1024 KB + 128 KB = 1168 KB); Section 5 mixed-dtype budget (16 KB + 544 KB + 128 KB = 688 KB); `K_t % in0_block_w == 0` constraint stated in Section 1.3 body; Heuristic 2 in Section 3.4 qualified with `out_subblock_h = 1` and safe upper bound `out_subblock_w ≤ floor(8 / out_subblock_h)`; minimum 2×2 grid stated consistently in Section 4.1 key characteristics, "When to use" bullet, and Section 4.3 flowchart — all correct and consistent.

No new correctness issues were identified.
