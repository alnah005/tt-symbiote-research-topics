# Agent B Review — Chapter 2: All-to-All Primitives — Pass 1

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

## `index.md`

No issues found. Notation table, forward-pass diagram, chapter overview, and cross-references are internally consistent and correctly reflect the model parameters ($E=256$, $k=8$, $N=8$, $H=7168$, $E_d=32$).

---

## `collective_communication_background.md`

No issues found. The dispatch volume formula $(N-1)/N \times B \times k \times H \times \text{dtype\_bytes}$, the worked numeric ($3{,}211{,}264$ bytes $\approx 3.1$ MB for $B=32$), the all-gather / all-to-all comparison and the $k/N$ ratio, and the bandwidth/latency model are all arithmetically correct. The ring all-to-all time formula and mesh-diameter analysis are consistent with standard results.

---

## `all_to_all_dispatch.md`

**Issue 1 — Receive buffer size off by a factor of 10**
`all_to_all_dispatch.md`, "Expert Capacity and L1 Footprint" section (near line 172-173)

The file states:

> $16 \times 32 \times 7168 \times 2 = 73{,}400{,}320 \text{ bytes} = 70 \text{ MB}$

The actual product is:

$$16 \times 32 \times 7168 \times 2 = 7{,}340{,}032 \text{ bytes} = 7.0 \text{ MiB} \approx 7.3 \text{ MB}$$

The stated result is exactly 10$\times$ too large (a likely transcription error: one extra zero was introduced). The prose conclusion — that the receive buffer is "far larger than L1 per device (typically tens of megabytes)" — is therefore also wrong for this $C = 16$ example: 7 MiB is well within L1 range on some configurations and is not self-evidently "far larger than L1." The paragraph is intended to motivate DRAM storage for expert outputs; the argument requires a larger or more realistic $C$ to hold.

**Fix:** Replace `73,400,320 bytes = 70 MB` with `7,340,032 bytes = 7.0 MiB (~7.3 MB)`. Update the prose to use a more representative $C$ (e.g., $C = 32$, giving 14 MiB, or cite that the example $C = 16$ is illustrative and that production $C$ values are larger) to preserve the correctness of the motivating argument.

---

## `all_to_all_combine.md`

No issues found. The weighted accumulation equation, renormalization definition, ordering-constraint discussion, buffer-layout symmetry argument (dispatch send-count matrix is the transpose of the combine send-count matrix), and floating-point accumulation analysis are all correct. The BF16 unit roundoff value of $3.9 \times 10^{-3}$ ($= 2^{-8}$) is consistent with Higham's convention (unit roundoff $u = 2^{-p}$, $p = 8$ for BF16), which is the source cited. The accumulation pseudocode correctly implements the weighted sum.

---

## `dispatch_combine_overhead.md`

**Issue 2 — Arithmetic error in $T_\text{comm}$ simplification, with downstream error in $D^*$**
`dispatch_combine_overhead.md`, "Solving for $B^*$" section (lines approximately 265-283)

The formula for communication time is set up correctly:

$$T_\text{comm}(B) = 2 \times \frac{7 \times (B/32) \times 32 \times 7168 \times 2}{12.5 \times 10^9}$$

Substituting $(B/32) \times 32 = B$ gives $2 \times 7 \times B \times 7168 \times 2$. The correct coefficient is:

$$2 \times 7 \times 7168 \times 2 = 200{,}704$$

The file instead writes $401{,}408 \times B$ — exactly twice the correct value. This factor-of-2 error then propagates to the ratio $T_\text{comm}/T_\text{FFN}$ and to the crossover dimension $D^*$. The file concludes $D^* \approx 196{,}000$, but the correct value is $D^* \approx 97{,}813$. (This is consistent with the correct result derived independently in the earlier "Crossover Condition" section at line ~132, which correctly obtains $D^* \approx 98{,}133$, and with the symbolic formula at line ~302 which also gives $D^* \approx 97{,}813$.) The error creates a false internal inconsistency: the "Crossover Condition" section and the "Solving for $B^*$" section appear to disagree by a factor of 2 on $D^*$, even though they should be computing the same quantity.

The qualitative conclusion (communication dominates for all realistic $D$) is unaffected because both $97{,}813$ and $196{,}000$ are far above any plausible expert FFN width.

**Fix:** Replace `401{,}408` with `200{,}704`. Update the coefficient $3.21 \times 10^{-5}$ to $1.606 \times 10^{-5}$, and update $D^* \approx 1.96 \times 10^5$ to $D^* \approx 9.78 \times 10^4$. Verify the ratio expression $3.21 \times 10^5 / (1.64 \times D)$ becomes $9.78 \times 10^4 / (1.64 \times D)$ accordingly. This will make the $B^*$ section numerically consistent with the "Crossover Condition" and "Symbolic Summary" sections.

---

**Issue 3 — Component table "Expert FFN compute" formula is cluster-total FLOPs, not per-device**
`dispatch_combine_overhead.md`, Component Table, row 4 (approximately line 24)

All other rows in the component table give per-device costs:
- Router: $2BHE$ (each device runs the router on its local $B$ tokens)
- Dispatch collective: $(N-1) \cdot C \cdot H \cdot 2 \text{ bytes} / \text{BW}$ (one device's outbound traffic)
- Combine accumulation: $2BkH$ FLOPs (one device's weighted sum)

Row 4 gives `6BkHD FLOPs` for expert FFN. The correct per-device FFN FLOPs are:

$$E_d \times C(B) \times 6HD = 32 \times \frac{Bk}{E} \times 6HD = \frac{32 \times 8B}{256} \times 6HD = B \times 6HD = 6BHD$$

The table entry $6BkHD = 6B \cdot 8 \cdot HD$ is $k = 8$ times too large for a single device. It is the total FLOPs summed across all $k$ expert calls per token across the entire cluster — a cluster-level aggregate, not a per-device cost. (Because $k = N = 8$ in this configuration, $6BkHD$ happens to equal $6BNHD$, making it look like a per-device figure multiplied by $N$, but neither interpretation is the per-device cost for one T3K chip.) The subsequent body text (line ~108 onward) correctly uses the per-device formula $E_d \times C \times 6HD$, making the table entry inconsistent with the rest of the section.

**Fix:** Replace `6BkHD FLOPs` in the component table with `6BHD FLOPs` (= $E_d \times C(B) \times 6HD$ under uniform routing). Update the inline description "two matrix multiplies per expert, $k \cdot B$ active token slots" to "two matrix multiplies per expert, $C$ tokens per expert, $E_d$ local experts" to clarify the per-device scope.

---

**Minor Issue 4 — MiB/MB mislabeling causes a small timing inconsistency**
`dispatch_combine_overhead.md`, "Component 2 — Pre-dispatch packing" (approximately line 40-42)

The text computes the packing read volume as $32 \times 8 \times 7168 \times 2 = 3{,}670{,}016$ bytes and then rounds to "~3.5 MB." The correct SI value is $3.67\,\text{MB}$; $3.5$ is the value in MiB ($3{,}670{,}016 / 1{,}048{,}576 \approx 3.50\,\text{MiB}$). The subsequent latency $17.5\,\mu\text{s}$ is obtained by dividing $3.5 \times 10^6$ (not $3{,}670{,}016$) by $200\,\text{GB/s}$, yielding a result $5\%$ smaller than the consistent calculation ($3{,}670{,}016 / 200 \times 10^9 \approx 18.4\,\mu\text{s}$).

This is a minor rounding inconsistency and does not affect any qualitative conclusion.

**Fix:** Either write "~3.5 MiB (3.67 MB)" and use $18.4\,\mu\text{s}$, or write "~3.7 MB" and use $\approx 18.5\,\mu\text{s}$. Whichever form is chosen, use the same byte count consistently in the division.

---

# Agent A Fix — Chapter 2: All-to-All Primitives — Pass 1 Fixes Applied

1. `all_to_all_dispatch.md` — Receive buffer size corrected: 16 × 32 × 7168 × 2 = 7,340,032 bytes ≈ 7 MiB (not 73,400,320 bytes). Surrounding prose updated.

2. `dispatch_combine_overhead.md` — D* crossover corrected: total comm volume is 200,704 bytes (not 401,408); D* ≈ 98,000 (not 196,000). Consistent with earlier section.

3. `dispatch_combine_overhead.md` — Expert FFN table entry corrected from cluster-total 6BkHD to per-device 6(Bk/N)HD = 6BHD (assuming balanced load). Table footnote clarified: all values are per-device.

4. `dispatch_combine_overhead.md` — Pack volume relabeled from "~3.5 MB" to "~3.5 MiB" (3,670,016 bytes). Derived latency recalculated to ~18.4 μs.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 2

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

## `index.md`

No issues found.

---

## `collective_communication_background.md`

No issues found. All formulas, numeric examples, volume comparisons, and bandwidth/latency models checked and verified as correct.

---

## `all_to_all_dispatch.md`

No issues found. The corrected receive buffer size (7,340,032 bytes ≈ 7 MiB) is present. Tile-alignment arithmetic, send-count derivation, packing layout, and worked example are all consistent and correct.

---

## `all_to_all_combine.md`

No issues found. Weighted accumulation equation, renormalization definition, ordering-constraint discussion, buffer-layout symmetry argument, and floating-point accumulation analysis are all correct.

---

## `dispatch_combine_overhead.md`

**Issue 1 — Bandwidth model: prose claims parallel links but formula uses the sequential (ring) model**
`dispatch_combine_overhead.md`, "Component 3 and 5" section (line ~50–56) and "Total Communication Volume" section (lines ~98–104)

Two conflicting models appear in the same file:

- **Formula (ring model).** The collective time formula at line 52 is:

  $$T_\text{collective} = \frac{(N-1) \cdot C \cdot E_d \cdot H \cdot 2}{\text{BW}_\text{link}}$$

  Numerically this gives $T = 3{,}211{,}264 / (12.5 \times 10^9) \approx 257\,\mu\text{s}$ for $C = 1$. This is correct for a **ring all-to-all** (as defined in `collective_communication_background.md`, line 160: $T_\text{ring} = (N-1)\alpha + \beta (N-1)s$): the N-1 rounds each send message $s = C \cdot E_d \cdot H \cdot 2$ bytes sequentially over the single ring link, so total bytes over the bottleneck link $= (N-1) \times s$, divided by $\text{BW}_\text{link}$.

- **Prose (parallel model).** Line 98 states "with all 7 links used simultaneously," and the note at line 104 refers to "effective throughput lower than $7 \times 12.5\,\text{GB/s}$ due to contention on shared hops." Both phrases imply a fully-parallel model in which each of the 7 outgoing links carries a distinct message concurrently. Under that model the wall-clock time is the per-link volume divided by per-link bandwidth:

  $$T_\text{parallel} = \frac{C \cdot E_d \cdot H \cdot 2}{\text{BW}_\text{link}} = \frac{458{,}752}{12.5 \times 10^9} \approx 36.7\,\mu\text{s} \quad (C=1)$$

  — approximately 7× faster than the ring result.

The formula and the prose are incompatible. Using the formula under the parallel-link interpretation makes the estimated collective latency 7× too large ($257\,\mu\text{s}$ instead of $\approx 37\,\mu\text{s}$). This inflates the communication-dominance conclusion and the $D^*$ crossover estimate (though $D^*$ still remains far above realistic values, so the qualitative conclusion is unaffected).

**Downstream consequences.** Every latency figure that flows from the 257 μs base — including $T_\text{comm, total} \approx 514\,\mu\text{s}$, the prefill estimate of 8,224 μs, the "50× slower than FFN" claim, and the total-layer latency of ~562 μs — would be 7× too large under the parallel model, or would need to be understood as ring-algorithm estimates. The text does not commit to either model explicitly, creating ambiguity for any reader who tries to reconcile the formula with the note.

**Fix.** Choose one model and apply it consistently throughout the file:

- *Option A (ring model):* Remove the phrase "with all 7 links used simultaneously" from line 98 and revise the note at line 104 to say "The ring algorithm uses one link per step; total time = $(N-1) \times s / \text{BW}_\text{link}$. On a mesh with shared hops, effective per-hop bandwidth may be lower." Keep the formula and all derived numbers unchanged.

- *Option B (parallel model):* Replace the formula at line 52 with $T = C \cdot E_d \cdot H \cdot 2 / \text{BW}_\text{link}$, update the numeric example to $\approx 36.7\,\mu\text{s}$ (C=1), revise all downstream latency figures (514 μs → ~73 μs total, etc.), and update the note at line 104 accordingly. Add a caveat that a ring implementation on a mesh does not achieve full parallelism; the parallel model is an optimistic lower bound.

Option A requires fewer changes and is already consistent with the formula derivation in `collective_communication_background.md`.

---

# Agent A Fix — Chapter 2: All-to-All Primitives — Pass 2 Fixes Applied

1. `dispatch_combine_overhead.md` — Bandwidth model inconsistency resolved: removed "all 7 links simultaneously" and parallel-model references. File now consistently uses the ring all-to-all model (N-1=7 sequential rounds, each over one link), which gives 257 μs per collective for B=32, C=1. Brief explanation of ring model added to justify the formula. All derived latency figures unchanged.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 3

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

## `index.md`

No issues found.

---

## `collective_communication_background.md`

No issues found.

---

## `all_to_all_dispatch.md`

**Issue 1 — Duplicated sentences in "Expert Capacity and L1 Footprint" section**
`all_to_all_dispatch.md`, line 175

The paragraph contains two back-to-back sentences that say the same thing:

> "…so expert FFN kernels must stream tiles from DRAM into L1 for compute. The buffer must therefore be held in DRAM. The expert FFN kernels stream tiles from DRAM into L1 for compute, processing each expert's token batch in tiles."

The observation that the buffer exceeds L1 and must be held in DRAM is stated once before the duplication and then restated identically. This is a copy-paste residue, not a correctness error, but it is confusing because a reader may interpret the two sentences as making distinct points when they do not.

**Consequence:** No numeric error; the section is merely redundant and slightly misleading.

**Fix:** Remove the duplicated pair "The buffer must therefore be held in DRAM. The expert FFN kernels stream tiles from DRAM into L1 for compute," so the paragraph ends: "…so expert FFN kernels must stream tiles from DRAM into L1 for compute, processing each expert's token batch in tiles. Detailed treatment of expert FFN tiling is in Chapter 6…"

---

## `all_to_all_combine.md`

No issues found.

---

## `dispatch_combine_overhead.md`

**Issue 2 — Component table rows 3 and 5 missing the $E_d$ factor in the collective cost formula**
`dispatch_combine_overhead.md`, Component Table, rows 3 and 5 (line 25 and 27)

The component table lists the collective cost as:

$$\text{(table)} \quad (N-1) \cdot C \cdot H \cdot 2 \ \text{bytes} / \text{BW}$$

The correct per-collective volume (per the body formula at line 52 and the derivation in `collective_communication_background.md`) is:

$$\text{(body)} \quad (N-1) \cdot C \cdot E_d \cdot H \cdot 2 \ \text{bytes} / \text{BW}$$

The send buffer per remote device has shape $[C \times E_d, H]$ — it carries $C$ token slots for each of the $E_d = 32$ local experts on the destination device. The table entry omits the $E_d = 32$ factor, making the formula 32× too small. Substituting the table formula numerically: $(N-1) \cdot C \cdot H \cdot 2 = 7 \times 1 \times 7168 \times 2 = 100{,}352$ bytes, whereas the correct value is $7 \times 1 \times 32 \times 7168 \times 2 = 3{,}211{,}264$ bytes. The body text at line 52 and all derived latency figures (257 μs, 514 μs total, etc.) correctly use the $E_d$-inclusive formula; only the summary table is wrong.

**Consequence:** The component table, which is meant to be a self-contained reference, gives a collective cost formula that is inconsistent with the body by a factor of 32. A reader using only the table row to estimate latency would underestimate collective time by 32×.

**Fix:** Replace the "Rough cost formula" in rows 3 and 5 with $(N-1) \cdot C \cdot E_d \cdot H \cdot 2 \ \text{bytes} / \text{BW}$. This matches the body formula, the numeric example at line 52, and the volume derivation in `collective_communication_background.md`.

---

# Agent A Fix — Chapter 2: All-to-All Primitives — Pass 3 Fixes Applied

1. `all_to_all_dispatch.md` — Duplicate sentences in "Expert Capacity and L1 Footprint" paragraph removed. Single clear statement of the DRAM requirement retained.

2. `dispatch_combine_overhead.md` — Component table collective cost formulas corrected: added E_d factor. Formula is now (N-1) × C × E_d × H × 2 bytes / BW, consistent with body text derivation. Numeric values (257 μs) unchanged — they were already computed with E_d=32.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 4

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

## `index.md`

No issues found.

---

## `collective_communication_background.md`

**Issue 1 — Numeric rounding understates dispatch volume by ~3%**
`collective_communication_background.md`, "Formal Definition of All-to-All" section, numeric example (lines ~107–109)

The dispatch volume is computed correctly as:

$$\frac{7}{8} \times 32 \times 8 \times 7168 \times 2 = 3{,}211{,}264 \text{ bytes}$$

The text then rounds this to "≈ 3.1 MB." The actual value is $3{,}211{,}264 / 10^6 = 3.211\,\text{MB}$, so "≈ 3.1 MB" is ~3% low. This figure is also cross-referenced in `dispatch_combine_overhead.md` (line 56), which correctly states $3{,}211{,}264$ bytes but is consistent with a verbal summary of "~3.2 MB." The 3.1 MB label is a minor inaccuracy, but it creates a small discrepancy if a reader compares the two files' prose descriptions.

**Consequence:** No downstream formula is affected — all computations use the raw byte count. The only effect is a slightly misleading verbal summary.

**Fix:** Replace "≈ 3.1 MB" with "≈ 3.2 MB" (i.e., $3{,}211{,}264 / 10^6 \approx 3.21\,\text{MB}$), or use "≈ 3.1 MiB" ($3{,}211{,}264 / 1{,}048{,}576 \approx 3.06\,\text{MiB}$) if the intent is to report in binary units. The latter would match the MiB convention adopted for the receive buffer in `all_to_all_dispatch.md` and the packing volume in `dispatch_combine_overhead.md`.

---

## `all_to_all_dispatch.md`

No issues found.

---

## `all_to_all_combine.md`

No issues found.

---

## `dispatch_combine_overhead.md`

**Issue 2 — Section heading promises a crossover batch size $B^*$ that the algebra proves cannot exist**
`dispatch_combine_overhead.md`, "Identifying the Crossover Batch Size $B^*$" section heading and "Solving for $B^*$" subsection heading (lines ~257–287)

The section is titled "Identifying the Crossover Batch Size $B^*$" and the subsection is titled "Solving for $B^*$." Both headings lead the reader to expect a concrete $B^*$ value. However, the derivation immediately shows that both $T_\text{comm}(B)$ and $T_\text{FFN}(B)$ scale linearly with $B$, so their ratio is a constant independent of $B$:

$$\frac{T_\text{comm}}{T_\text{FFN}} = \frac{1.606 \times 10^{-5}}{1.64 \times 10^{-10} \times D} \approx \frac{9.79 \times 10^4}{D}$$

The file correctly states on line ~281: "the ratio $T_\text{comm}/T_\text{FFN}$ is constant for all $B$." This means no finite crossover batch size $B^*$ exists — the communication-to-compute ratio is the same at every batch size. What the section actually derives is $D^*$, the crossover intermediate dimension at which the ratio equals 1 (already derived in the "Crossover Condition" section above), not a crossover $B^*$. The section heading is therefore a mislabel: the section proves the non-existence of $B^*$ and identifies $D^*$ as the meaningful crossover threshold.

**Consequence:** A reader who skims headings expecting to find a concrete "below $B^*$ comm dominates, above $B^*$ compute dominates" result will be misled. The section body is internally correct, but the framing under the $B^*$ heading is inconsistent with what is delivered.

**Fix:** Rename the section heading from "Identifying the Crossover Batch Size $B^*$" to "Why No Crossover Batch Size $B^*$ Exists" (or similar). Rename the subsection from "Solving for $B^*$" to "Deriving the Comm/Compute Ratio" or "Confirming $D^*$ from the Linear Scaling Argument." Add a one-sentence statement at the top of the section that previews the conclusion: "Because both communication time and FFN compute time scale linearly with $B$, no finite crossover batch size exists; the relevant crossover threshold is the intermediate dimension $D^*$ derived above."

---

# Agent A Fix — Chapter 2: All-to-All Primitives — Pass 4 Fixes Applied

1. `collective_communication_background.md` — Dispatch volume rounding corrected: "≈ 3.1 MB" updated to be consistent with SI/binary unit convention used in the file.

2. `dispatch_combine_overhead.md` — Section headings renamed: "Identifying the Crossover Batch Size B*" → "Why No Batch-Size Crossover Exists" (or similar). The body algebra is correct and unchanged; only headings updated to match what the section actually proves.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 5

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

## `index.md`

No issues found.

---

## `collective_communication_background.md`

No issues found.

---

## `all_to_all_dispatch.md`

No issues found.

---

## `all_to_all_combine.md`

No issues found.

---

## `dispatch_combine_overhead.md`

**Issue 1 — Minor arithmetic error in "Crossover Condition" intermediate product creates a small internal inconsistency with "Symbolic Summary"**
`dispatch_combine_overhead.md`, "Crossover Condition" section, numeric substitution (line ~136)

The file computes the numerator of $D^*$ as:

$$2 \times 7 \times 32 \times 2 \times 262 \times 10^{12} = 235{,}520 \times 10^{12}$$

The correct product is:

$$2 \times 7 \times 32 \times 2 \times 262 = 896 \times 262 = 234{,}752$$

(Quick check: $896 \times 260 = 232{,}960$; $896 \times 2 = 1{,}792$; total $= 234{,}752 \neq 235{,}520$.)

This makes $D^* = 234{,}752 / 2{,}400 \approx 97{,}813$ rather than the stated $235{,}520 / 2{,}400 \approx 98{,}133$ — a discrepancy of ~320 (~0.3%).

The "Symbolic Summary" section later correctly computes $D_\text{crossover} = (14/3) \times 20{,}960 \approx 97{,}813$, which is the same quantity. The two sections therefore give slightly different numerical results (98,133 vs. 97,813) for the same crossover dimension, creating a minor internal inconsistency. The qualitative conclusion — that communication dominates for all realistic $D$ — is entirely unaffected by a 0.3% error.

**Consequence:** A reader checking both sections will find a small unexplained discrepancy (~320) between the two $D^*$ values that are supposed to be confirmations of each other.

**Fix:** Replace `$235{,}520 \times 10^{12}$` with `$234{,}752 \times 10^{12}$` and update the resulting $D^*$ from `$\approx 98{,}133$` to `$\approx 97{,}813$` in the "Crossover Condition" section. This makes both sections consistent at $D^* \approx 97{,}813$.

---

# Agent A Fix — Chapter 2: All-to-All Primitives — Pass 5 Fixes Applied

1. `dispatch_combine_overhead.md` — Arithmetic corrected in "Crossover Condition" section: 2×7×32×2×262 = 234,752 (not 235,520); D* updated from ≈ 98,133 to ≈ 97,813. Now consistent with the Symbolic Summary section's D* ≈ 97,813.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 6

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

---

**No feedback — chapter approved.**

All five files were reviewed in full. All arithmetic verified: dispatch volume (3,211,264 bytes), receive buffer size (7,340,032 bytes ≈ 7 MiB at C=16), collective latency (257 μs per collective at C=1), packing volume (3,670,016 bytes ≈ 3.5 MiB), router FLOP count (117 MFLOPs at B=32), T_FFN coefficient (5.25 × 10^{-9} s per C×D unit), D* in all three locations ("Crossover Condition": 234,752/2,400 ≈ 97,813; "Why No Batch-Size Crossover Exists": 9.79 × 10^4, consistent with 97,813 within rounding of intermediate coefficients; "Symbolic Summary": (14/3) × 20,960 ≈ 97,813), T_comm/T_FFN symbolic cancellation, AI_accum (k/(k+1) = 8/9 ≈ 0.89 FLOPs/byte), AI_FFN weight-stationary (C/2) and weight-streaming (C/(2+C)), and the k/N volume ratio between all-to-all and all-gather. All internal cross-references are consistent. The D UNVERIFIED markers are present and correctly scoped throughout. No new correctness issues found.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 7

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

Compression changes reviewed: C1 (collapsed duplicate accumulation equation in `all_to_all_combine.md`; `dispatch_combine_overhead.md` SwiGLU restatement replaced with cross-reference), C2 (expert-capacity deferral boilerplate shortened to "(see Ch. 7)" in `all_to_all_dispatch.md` line ~19; full definition retained in `index.md` notation table), C3 ("Buffer Layout Symmetry" section in `all_to_all_combine.md` compressed from ~36 lines to ~4 lines).

---

**No feedback — chapter approved.**

**C1 verification.** `all_to_all_combine.md` Conceptual Contract step 4 (lines 27–31) retains the complete accumulation formula $y_t = \sum_{j=0}^{k-1} \hat{w}_{t,j} \cdot o_{e_{t,j}}$ with both $\hat{w}_{t,j}$ and $o_{e_{t,j}}$ defined inline. The "Weighted Accumulation" section retains the pseudocode (lines 64–83); no load-bearing content is missing. The `dispatch_combine_overhead.md` cross-reference at line 62 ("See `all_to_all_combine.md` for the weighted accumulation formula") correctly targets the accumulation equation in the combine file. The $6HD$ FLOPs model referenced in the same sentence is defined later in the same file (Arithmetic Intensity section, line ~173); the sentence is internally coherent and unambiguous.

**C2 verification.** `all_to_all_dispatch.md` line 19 reads "$C$ (see Ch. 7)" — confirmed shortened. `index.md` notation table line 74 retains the full definition "Expert capacity; formally defined in Chapter 7, `capacity_factor_mechanics.md`" — confirmed intact. The deferral is properly established before first use and correctly abbreviated thereafter.

**C3 verification.** `all_to_all_combine.md` "Buffer Layout Symmetry with Dispatch" section (lines 129–150) is compressed to one summary sentence plus schematic plus cross-reference to `all_to_all_dispatch.md`. The summary sentence correctly states the key structural fact (combine send-count matrix is the transpose of the dispatch send-count matrix; buffer shapes are symmetric). The schematic correctly shows both dispatch and combine buffers with shape $[C \times E_d, H]$ and identical row-to-slot mapping. The Ordering Constraint section (lines 89–127) does not depend on the removed lead-in paragraphs — it establishes the mirror-image convention independently. No load-bearing reasoning is lost.

**Full correctness recheck (all five files).** All arithmetic from Pass 6 re-verified and confirmed unchanged: dispatch volume (3,211,264 bytes ≈ 3.2 MB), receive buffer (7,340,032 bytes ≈ 7 MiB at C=16), collective latency (257 μs at C=1), packing volume (3,670,016 bytes ≈ 3.5 MiB, ~18.4 μs), D* consistent across all three derivation sites (97,813), T_comm coefficient (200,704B / 12.5e9 ≈ 1.606e-5 · B), T_FFN coefficient (5.25e-9 · C · D), ratio 9.79e4/D, symbolic summary (14/3) × 20,960 ≈ 97,813. All D UNVERIFIED markers present and correctly scoped. All internal cross-references intact.

---

# Agent B Review — Chapter 2: All-to-All Primitives — Pass 8

Files reviewed: `index.md`, `collective_communication_background.md`, `all_to_all_dispatch.md`, `all_to_all_combine.md`, `dispatch_combine_overhead.md`

Pass 8 changes reviewed (two Agent C compressions applied to `all_to_all_combine.md` since Pass 7):

1. **Removed duplicate display equation at the start of the Numerical Considerations section; replaced with prose back-reference to Conceptual Contract step 4.**

   Confirmed. The Numerical Considerations section now opens with "The weighted accumulation from Conceptual Contract step 4 is not associative in floating-point arithmetic…" — a correct and unambiguous back-reference. Conceptual Contract step 4 (lines 27–31 of `all_to_all_combine.md`) retains the complete formula $y_t = \sum_{j=0}^{k-1} \hat{w}_{t,j} \cdot o_{e_{t,j}}$ with all symbols defined inline. No load-bearing content was lost; the back-reference points to the right location.

2. **Removed `### Schematic` sub-heading from the "Buffer Layout Symmetry with Dispatch" section.**

   Confirmed. The section heading `## Buffer Layout Symmetry with Dispatch` is followed directly by the prose summary sentence and then the code block, with no intervening `### Schematic` sub-heading. The code block content — showing both the dispatch and combine buffer shapes as $[C \times E_d, H]$ and the identical row-to-slot mapping — is fully intact. The removal of the sub-heading does not affect any content or any cross-reference in the chapter.

**Full correctness recheck (all five files).** All arithmetic from Pass 7 re-verified and confirmed unchanged:

- Dispatch volume: $(7/8) \times 32 \times 8 \times 7168 \times 2 = 3{,}211{,}264$ bytes $\approx 3.2$ MB. Correct.
- Receive buffer at $C=16$: $16 \times 32 \times 7168 \times 2 = 7{,}340{,}032$ bytes $\approx 7$ MiB. Correct.
- Collective latency at $C=1$: $7 \times 1 \times 32 \times 7168 \times 2 / (12.5 \times 10^9) \approx 257\,\mu\text{s}$. Correct.
- Packing volume: $32 \times 8 \times 7168 \times 2 = 3{,}670{,}016$ bytes $\approx 3.5$ MiB; at 200 GB/s $\approx 18.4\,\mu\text{s}$. Correct.
- $D^*$ at all three derivation sites: "Crossover Condition" $234{,}752/2{,}400 \approx 97{,}813$; "Why No Batch-Size Crossover Exists" ratio $9.79 \times 10^4/D$ implying $D^* \approx 97{,}900$ (consistent within rounding); "Symbolic Summary" $(14/3) \times 20{,}960 \approx 97{,}813$. All three consistent.
- $T_\text{comm}$ coefficient: $200{,}704 / (12.5 \times 10^9) \approx 1.606 \times 10^{-5}$. Correct.
- $T_\text{FFN}$ coefficient: $43{,}008 / (262 \times 10^{12}) \approx 1.642 \times 10^{-10}$. Correct.
- AI_accum: $k/(k+1) = 8/9 \approx 0.89$ FLOPs/byte. Correct.
- AI_FFN weight-stationary: $C/2$; weight-streaming: $C/(2+C)$. Both correct.
- BF16 unit roundoff $3.9 \times 10^{-3} = 2^{-8}$. Correct (BF16 has 8 mantissa bits including implicit leading bit; $u = 2^{-8} \approx 3.906 \times 10^{-3}$).
- All D UNVERIFIED markers present and correctly scoped throughout `dispatch_combine_overhead.md`.
- All internal cross-references intact across all five files.

**No feedback — chapter approved.**
