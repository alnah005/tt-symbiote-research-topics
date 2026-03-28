# Compression Analysis: Chapter 6 — T3K Mesh Sharding and CCL Implications — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1,307 lines
- Estimated post-compression line count: ~1,020 lines
- Estimated reduction: ~22%

## CRUCIAL Suggestions

### 1. Topology description duplicated across `index.md` and `sharding_strategies.md`

`index.md` (lines 57–67) contains a full prose description of the T3K 1×8 linear
mesh topology, the chip-numbering, endpoint vs. interior chips, and the CCL
primitives that use the links. `sharding_strategies.md` (lines 7–58) then
re-covers the same topology in equal or greater detail, including the ASCII
diagram, the per-device resource table, and the asymmetry argument
(Ethernet 12.5 GB/s vs. DRAM 288 GB/s). The `index.md` section titled
"T3K Topology" (lines 55–67) is entirely pre-empted by the opening of
`sharding_strategies.md`. **Cut the "T3K Topology" section from `index.md`
entirely** and replace it with a single sentence forwarding the reader to
`sharding_strategies.md`.

### 2. CCL volume numbers computed four times across three files

The sequence-parallel K+V all-gather volume (≈ 448 MiB for the reference
parameters B=32, H_kv=8, w=4096, d=128, N=8, BF16) is computed with the full
numeric derivation in:
- `sharding_strategies.md` lines 276–285 (with formula and result)
- `sharding_strategies.md` lines 311–325 (the decision-matrix worked example block)
- `ccl_operations.md` lines 205–213 (formula + result again with same parameters)
- `ccl_operations.md` lines 252–268 (transfer-time comparison block)

The head-parallel output-gather volume (≈ 224 KiB) is computed in:
- `sharding_strategies.md` lines 170–178
- `ccl_operations.md` lines 141–148

Each derivation re-substitutes identical numbers and arrives at the same result.
**Keep the full derivation once in `sharding_strategies.md` (the decision-matrix
block is the natural home). In `ccl_operations.md` replace each repeated
derivation with a cross-reference to the result and retain only the
transfer-time implications that are unique to that file.** Estimated savings:
~40 lines.

### 3. Sequence-parallel decode step described twice in full

The sequence-parallel decode step — including the code block showing
`ttnn.all_gather(k_shard_i, dim=2)` and the observation that all eight devices
subsequently perform identical redundant computation — appears in full in:
- `sharding_strategies.md` lines 209–244 (decode-step pseudocode + critical analysis)
- `ccl_operations.md` lines 177–213 (pre-attention CCL section, which re-shows
  the same pseudocode and re-derives the same volume)

The `ccl_operations.md` pre-attention section adds nothing that is not already
in `sharding_strategies.md`. **Cut the pseudocode block from `ccl_operations.md`
lines 182–213 and replace it with a one-paragraph summary that cites the volume
result already established in `sharding_strategies.md`.** Estimated savings:
~30 lines.

### 4. Recommendation rationale restated in `index.md` after full treatment in `sharding_strategies.md`

`index.md` "Chapter Scope → Sharding Strategies" (lines 69–98) summarises both
strategies including the CCL consequence and the recommendation for head-parallel
sharding. `sharding_strategies.md` covers all of this at length and ends with a
6-point numbered "Recommendation" section. The `index.md` summary duplicates the
conclusions without adding navigational value beyond what the Reading Order list
(lines 37–51) already provides. **Trim the "Sharding Strategies" and "CCL
Operations" subsections of `index.md` (lines 69–103) to 2–3 sentences each,
keeping only the decision outcome and a pointer to the detailed file.** Estimated
savings: ~25 lines.

### 5. `per_device_window_application.md` correctness proof restated in prose and then as worked example covering the same case

The correctness argument for head-parallel sharding is stated as a formal union
proof (lines 78–93) and then immediately restated in prose (lines 95–103) under
the heading "No Boundary Issues Under Head-Parallel Sharding", which says the
same thing in different words: the window boundary is a time-dimension predicate,
all devices hold the full time axis, therefore there are no boundary cases. The
prose restatement adds no new content. **Delete lines 95–103 ("No Boundary
Issues Under Head-Parallel Sharding" subsection) and merge any unique phrasing
into the closing sentence of the proof.** Estimated savings: ~10 lines.

---

## MINOR Suggestions

### M1. `sharding_strategies.md` — verbose "Recommendation" preamble

Lines 336–337 ("The rationale is as follows:") and the six numbered points each
open with a bolded label followed by one or two sentences that restate the
decision-matrix conclusions already visible in the table on lines 293–306. Points
1, 2, 3, and 4 are all derivable from a single reading of the table. Consider
collapsing points 1–4 into a two-sentence paragraph and preserving only points 5
(tensor-parallel compatibility) and 6 (divisibility), which add context not
explicit in the table.

### M2. `ccl_operations.md` — overlap section hedging language

Lines 340–347 contain two consecutive hedging qualifications: "Whether full
overlap is achieved depends on..." and "If the CCL implementation occupies
Tensix cores for any phase of the collective, partial overlap is still possible
but not complete." These are reasonable caveats but are then immediately resolved
by the following sentence recommending treating the link bandwidth as an
independent resource. The two hedge sentences can be collapsed into one.

### M3. `ccl_operations.md` — "KV Cache Write Overlap" subsection (lines 361–368)

This subsection states that `ttnn.update_cache` is local, small, and has no
overlap opportunity because of write-before-read ordering. The same point was
already established in `sharding_strategies.md` lines 138–142 and implicitly in
the CCL summary table. The subsection can be reduced to one sentence.

### M4. `per_device_window_application.md` — repeated scalar description

The observation that `update_index = T % w` is "a scalar that is advanced
uniformly across all devices" (lines 116–118 in `per_device_window_application.md`)
repeats the same point made in `sharding_strategies.md` lines 50 and 165–169.
One cross-reference is sufficient.

### M5. `per_device_window_application.md` — "Practical Implementation" code block

The two-block pseudocode at lines 119–148 (steady state and fill phase for
head-parallel) mirrors the logic already shown in `sharding_strategies.md` lines
114–134 and the fill-phase discussion in Chapter 4. If the target audience has
read Chapters 2 and 4, this block can be collapsed to a single annotated snippet
showing only the fill-phase mask construction, which is the only element unique
to the multi-device setting.

---

## Load-Bearing Evidence

- **`index.md` lines 20–35 (Prerequisites block):** The four-item prerequisite
  list is the only place in Chapter 6 that explicitly cross-links the arithmetic
  intensity result from Chapter 3 (`AI ≈ 1 FLOP/byte`) as the basis for the CCL
  cost model. Removing or compressing it would break the stated dependency chain.

- **`sharding_strategies.md` lines 292–306 (Decision Matrix table):** The
  seven-column comparison table is the single most information-dense structure in
  the chapter. Every row encodes a distinct criterion (CCL volume, latency,
  criticality, compute replication, `w`-scaling, divisibility, GQA
  compatibility). No row is redundant with another. The table must be kept intact.

- **`sharding_strategies.md` lines 375–385 (Limitation of Head-Parallel
  Sharding — MQA / H_kv = 1 case):** This is the only location in Chapter 6
  that identifies the failure mode of the recommended strategy and prescribes a
  fallback for MQA models. It appears nowhere else in the chapter and cannot be
  cut.

- **`ccl_operations.md` lines 274–310 (Impact of Window Size `w` on CCL
  Volume):** The two comparative tables showing CCL volume vs. `w` for both
  strategies (with concrete values at w=1024, 4096, 8192, 32768) are the only
  quantitative demonstration that sequence-parallel sharding degrades linearly
  with `w`. The conclusion references specific models (Qwen2, Mistral-style
  large-context) that do not appear elsewhere in the chapter. This section must
  be retained.

- **`per_device_window_application.md` lines 243–285 (Edge Cases When `w` Is
  Not Divisible by `N`):** The non-divisibility padding analysis — including the
  formula for unequal shard sizes, the TTNN uniform-shard-size constraint, the
  worked example at w=9 N=8, and the worst-case padding overhead bound
  `(N-1)/w` — is the only treatment of this edge case in the entire chapter. It
  is not summarised or previewed anywhere else and cannot be reduced.

- **`per_device_window_application.md` lines 311–355 (Correctness Check: Fill
  Phase with Non-Uniform Valid Slots):** The concrete w=16, N=8, T=3 trace and
  the two-case proof that the boundary always falls either between devices or
  within exactly one device's shard is load-bearing: it is the formal
  justification for the per-device mask construction described in lines 219–241.
  The worked example is not restated elsewhere.

---

## VERDICT
- Crucial updates: yes

## Change Log (Pass 1)

All five CRUCIAL suggestions applied on 2026-03-28.

**C1 — `index.md` T3K Topology section (was lines 55–67):**
Replaced the full prose description of the 1×8 linear mesh (chip numbering,
endpoint vs. interior chips, CCL primitives, sharding divisibility paragraph)
with a single forward-reference sentence pointing to `sharding_strategies.md`.
Reduction: ~11 lines.

**C2 — CCL volume derivations in `sharding_strategies.md`:**
- "CCL Volume — Head-Parallel" section: removed the two `math` blocks
  (formula + numeric substitution yielding 224 KiB) and the transfer-time
  sentence. Replaced with a two-sentence summary (value + cross-reference to
  `ccl_operations.md`). Reduction: ~10 lines.
- "CCL Volume — Sequence-Parallel" section: removed the three `math` blocks
  (K formula, K+V formula, numeric substitution yielding 448 MiB) and the
  transfer-time sentence. Replaced with a two-sentence summary (value +
  cross-reference to `ccl_operations.md`). Reduction: ~14 lines.

**C3 — Sequence-parallel decode pseudocode in `ccl_operations.md`:**
Removed the `Device i — KV reconstruction (sequence-parallel)` pseudocode
block from the "Pre-Attention CCL: KV All-Gather" section. Replaced with a
cross-reference paragraph pointing to `sharding_strategies.md` for the full
decode-step pseudocode. The canonical K and K+V volume derivation formulas
immediately following the removed block were retained. Reduction: ~10 lines.

**C4 — `index.md` "Sharding Strategies" and "CCL Operations" subsections
(was lines 69–103):**
- "Sharding Strategies": removed the full prose descriptions of head-parallel
  and sequence-parallel strategies (two `**bold**` paragraphs). Replaced with
  2–3 sentences giving the outcome and a pointer to the detail file.
  Reduction: ~16 lines.
- "CCL Operations": removed the bulleted list of CCL operations per strategy
  and the two-sentence bandwidth narrative. Replaced with 2 sentences
  summarising the volume contrast and pointing to `ccl_operations.md`.
  Reduction: ~10 lines.

**C5 — `per_device_window_application.md` prose restatement of union proof:**
Deleted the "No Boundary Issues Under Head-Parallel Sharding" subsection
(was ~9 lines of prose restating that the window predicate is on the time axis
and therefore has no per-device boundary effect). The unique content — the
cross-links to Chapters 2 and 4 — was folded into the closing sentence of the
correctness proof. Reduction: ~9 lines.

**Total estimated reduction: ~70 lines (~5% of 1,307).**
No formulas, tables, diagrams, or numerical values were changed. All
navigation footers and clickable links in `index.md` were preserved.

# Compression Analysis: Chapter 6 — T3K Mesh Sharding and CCL Implications — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1,233 lines (index.md 98 + sharding_strategies.md 364 + ccl_operations.md 390 + per_device_window_application.md 381)
- Estimated post-compression line count: ~1,133 lines
- Estimated reduction: ~8%

## CRUCIAL Suggestions
None — all resolved.

**C1 verified:** `index.md` lines 55–59 — the T3K Topology subsection now reads as a single forward-reference sentence: "The full topology description — 1×8 linear mesh, per-device resources, and CCL primitive overview — is in [`sharding_strategies.md`](./sharding_strategies.md)." The prior 11-line prose block is gone.

**C2 verified:** `sharding_strategies.md` "CCL Volume — Head-Parallel" (lines 161–166) and "CCL Volume — Sequence-Parallel" (lines 253–259) each contain only a summary value and a cross-reference to `ccl_operations.md`. No inline formula re-derivations remain in those sections.

**C3 verified:** `ccl_operations.md` "Pre-Attention CCL: KV All-Gather" (lines 177–183) contains no pseudocode block. A single cross-reference paragraph points to `sharding_strategies.md` for the decode-step pseudocode, and the volume formulas immediately following are the canonical derivation, not a repeat.

**C4 verified:** `index.md` "Sharding Strategies" subsection (lines 63–67) is 3 sentences. "CCL Operations" subsection (lines 71–75) is 3 sentences. Both are within the 2–3 sentence target.

**C5 verified:** `per_device_window_application.md` contains no "No Boundary Issues Under Head-Parallel Sharding" prose subsection. The correctness proof closes at line 99 with the cross-links to Chapters 2 and 4 folded into the final sentence; the section moves directly to "Practical Implementation" without restating the proof in prose.

## MINOR Suggestions

### M1. `sharding_strategies.md` — "Recommendation" numbered list still verbose

The six-point numbered "Recommendation" rationale (lines 310–347) repeats conclusions already encoded in the decision matrix on lines 266–280. Points 1–4 restate CCL criticality, DRAM utilisation, genuine parallelism, and `w`-independence — all of which are explicit matrix rows. Collapsing points 1–4 into a two-sentence paragraph and retaining only points 5 (tensor-parallel weight compatibility) and 6 (divisibility guarantee) would save approximately 20 lines while preserving the two non-obvious rationale items.

### M2. `ccl_operations.md` — overlap section hedging sentences (lines 332–339)

Two consecutive hedging qualifications ("Whether full overlap is achieved depends on..." and "If the CCL implementation occupies Tensix cores...") precede a sentence that immediately resolves the hedge by recommending the link bandwidth be treated as an independent resource. The two hedge sentences can be merged into one without losing content.

### M3. `ccl_operations.md` — "KV Cache Write Overlap" subsection (lines 352–360)

This subsection establishes that `ttnn.update_cache` is local, latency-negligible, and has no overlap opportunity due to write-before-read ordering. The same point is made in `sharding_strategies.md` lines 138–142 and is implicit in the CCL summary table. The subsection can be reduced to a single sentence appended to the preceding overlap discussion.

## Load-Bearing Evidence

- **`index.md` lines 20–35:** The prerequisites block is the only location in Chapter 6 that explicitly ties the arithmetic-intensity result (AI ≈ 1 FLOP/byte from Chapter 3) to the CCL cost model. Quoted: "The arithmetic intensity result from `../ch3_data_dependencies/decode_access_patterns.md` establishing that windowed decode is bandwidth-bound (AI ≈ 1 FLOP/byte), which informs the CCL cost model in this chapter."

- **`sharding_strategies.md` lines 266–280:** The decision matrix is the single most information-dense structure in the chapter. Quoted header row: "| Criterion | Head-Parallel | Sequence-Parallel |" — all eleven criterion rows are distinct and load-bearing.

- **`ccl_operations.md` lines 283–297:** The `w`-scaling table for head-parallel CCL is the canonical demonstration that CCL cost is invariant with `w`. Quoted: "CCL cost is invariant under changes to w."

- **`per_device_window_application.md` lines 243–274:** The non-divisibility edge-case analysis (padding formula, TTNN uniform-shard-size constraint, worked example w=9 N=8) is unique to this file. Quoted: "TTNN's `all_gather` requires uniform shard sizes along the gathered dimension."

## VERDICT
- Crucial updates: no
