# Compression Analysis: ch02_silu_on_wormhole_hardware

## Load-Bearing Evidence

The following facts must NOT be removed during any compression pass. They are either unique to
a single file, required for downstream chapters, or provide the quantitative grounding for
experimental design.

1. **FPU destination register sharing is the root cause of sequential FPU→SFPU execution.**
   Stated in `tensix_compute_engine.md` lines 43–45 and 68–83. This is the mechanistic
   explanation for why SiLU latency is additive to matmul latency, not overlapping. It must
   be preserved in exactly one place.

2. **The SFPU LReg is 32 elements wide, forcing 32 sequential passes per 32×32 BF16 tile.**
   This is the hardware constraint (1024 elements / 32 per pass = 32 passes) that drives all
   downstream instruction-count and cycle estimates. It must appear in one authoritative location.

3. **SiLU instruction sequence: negate → exp approx → add → reciprocal → multiply (5–8 SFPU
   instructions per pass; 160–256 per tile).** Stated in `silu_sfpu_execution.md` table lines
   35–45 and the derivation lines 69–74. This is the primary technical payload of the chapter.

4. **`math_approx_mode` saves ~2 instructions/pass (~64 instructions/tile) at the cost of
   small numerical error in sigmoid tails.** Stated in `silu_sfpu_execution.md` lines 117–124.
   This is the only actionable tuning lever described in the chapter.

5. **Gate_proj matmul at 32 tokens has arithmetic intensity ~31.6 FLOP/byte, making it
   memory-bound (well below Wormhole's ~437 FLOP/byte ridge point).** Quantitative derivation
   in `cycles_vs_matmul.md` lines 86–96. Required for the roofline placement conclusion.

6. **SiLU has arithmetic intensity ~0.5 FLOP/byte and is memory-bound at all practical tensor
   sizes.** Stated in `cycles_vs_matmul.md` lines 103–116. This is the roofline conclusion that
   motivates empirical measurement over FLOP-based estimates.

7. **SiLU ≈ 15–40% of gate_proj matmul latency when both are memory-bound.** Derived in
   `cycles_vs_matmul.md` lines 142–162. This is explicitly flagged as the prior for
   experimental design in Chapter 3 and must not be silently removed.

---

## CRUCIAL Suggestions

### C1: Deduplicate the 32-pass constraint derivation (tensix_compute_engine.md and silu_sfpu_execution.md)

**Location of duplication:**
- `tensix_compute_engine.md` lines 60–62: Introduces the 32-pass constraint with the
  1024/32 = 32 derivation.
- `silu_sfpu_execution.md` lines 62–80: Re-derives the same constraint with the same
  arithmetic, adds the instruction-count multiplication, and emphasizes the serialization
  point.

**Evidence of near-verbatim overlap:**

`tensix_compute_engine.md` lines 60–62:
> The 32-element width of the LReg is a hard hardware constraint. A 32×32 tile has 1024
> elements, so the SFPU must make **32 sequential passes** to process one full tile. Each pass
> loads one row of 32 elements, runs the instruction sequence, and writes back.

`silu_sfpu_execution.md` lines 62–68:
> The SFPU LReg is 32 elements wide. A standard BF16 tile on Wormhole is 32×32 = 1024
> elements. Therefore:
> passes per tile = 1024 elements / 32 elements per pass = 32 passes

The hardware fact (LReg width = 32, tile = 1024 elements, 32 passes) is identical. The
explanation text is near-verbatim. The instruction-count extension in `silu_sfpu_execution.md`
lines 69–80 is unique and load-bearing, but the derivation preamble is redundant.

**Recommended action:** Remove the 32-pass derivation paragraph from `tensix_compute_engine.md`
(lines 60–62). Replace it with a one-sentence forward reference: "The SFPU LReg is 32 elements
wide; the full derivation of the 32-pass constraint appears in `silu_sfpu_execution.md`." The
Section in `silu_sfpu_execution.md` is more complete and should be canonical.

**Estimated line savings:** ~6 lines in `tensix_compute_engine.md`.

---

### C2: Deduplicate the SFPU instruction budget formula (silu_sfpu_execution.md and cycles_vs_matmul.md)

**Location of duplication:**

`silu_sfpu_execution.md` lines 69–74 (the 32-pass section):
> Each pass runs the full 5–8 instruction sequence on the 32 loaded elements. The total
> instruction count per tile is:
> ~32 passes × (5–8 instructions/pass) = 160–256 SFPU instructions per tile

`cycles_vs_matmul.md` lines 47–48 (the SFPU Budget section):
> 32 passes × (5–8 instructions/pass) = 160–256 SFPU instructions per tile

The formula and the result range (160–256) are verbatim identical. `cycles_vs_matmul.md` goes
on to use the result in the gate_proj tile-count multiplication (lines 50–55), which is unique.
The formula derivation itself is duplicated.

**Recommended action:** In `cycles_vs_matmul.md`, remove the inline re-derivation and replace
it with a back-reference: "From `silu_sfpu_execution.md`, each SiLU tile requires 160–256 SFPU
instructions." The numeric result is all that is needed in `cycles_vs_matmul.md`; the derivation
already appeared in `silu_sfpu_execution.md`.

**Estimated line savings:** ~3–4 lines in `cycles_vs_matmul.md`.

---

### C3: Remove the "Why This Matters for SiLU in MoE" section from tensix_compute_engine.md

**Location:** `tensix_compute_engine.md` lines 101–112.

This section restates the sequential pipeline constraint (already fully explained in the same
file at lines 68–83) and then previews the "small token count → memory-bound → SiLU fraction
becomes measurable" argument that is the explicit content of `cycles_vs_matmul.md`. It adds no
new technical content; it is a motivational bridge that repeats previously stated material.

Specifically:
- The claim "Step 2 cannot begin until step 1 is complete" is already the conclusion of the
  Sequential Pipeline Constraint section immediately above.
- The claim "at small token counts the gate_proj matmul may already be memory-bound" is the
  central finding of `cycles_vs_matmul.md` lines 86–96, which derives this quantitatively with
  actual numbers.

**Recommended action:** Delete the "Why This Matters for SiLU in MoE" section entirely (lines
101–112 in `tensix_compute_engine.md`). The reader transitions to `cycles_vs_matmul.md` via
the Next Steps footer and will encounter the quantitative argument there.

**Estimated line savings:** ~12 lines in `tensix_compute_engine.md`.

---

## MINOR Suggestions

### M1: Remove redundant "Next Steps" footers from detail files

`tensix_compute_engine.md` line 115–116, `silu_sfpu_execution.md` lines 128–129, and
`cycles_vs_matmul.md` lines 175–177 each contain a "Next Steps" callout directing the reader
to the next file or chapter. `index.md` lines 42–47 already defines the reading order
explicitly in prose ("Read the files in the order listed…") and lists all three filenames.
The per-file footers duplicate this navigation.

**Recommended action:** Remove the "Next Steps" paragraphs from the two inter-file transitions
(`tensix_compute_engine.md` and `silu_sfpu_execution.md`). Keep only the footer in
`cycles_vs_matmul.md` that points to Chapter 3, since that cross-chapter pointer is not
covered in `index.md`'s scope.

**Estimated line savings:** ~4 lines across two files.

---

### M2: Collapse the activation comparison table in silu_sfpu_execution.md with the learning objective in index.md

`index.md` learning objective 5 (line 24–25) states:
> Compare the relative cost ordering of element-wise activations: ReLU < SiLU ≈ GELU in
> SFPU cycles per tile.

`silu_sfpu_execution.md` lines 84–108 present a full table and three paragraphs elaborating
ReLU vs SiLU vs GELU with instruction counts and the `ReLU << SiLU ≈ GELU` ordering.

The learning objective in `index.md` fully telegraphs the conclusion before the reader opens
the file. A reader who has read `index.md` already knows the ordering; the detail file then
re-announces the same conclusion before explaining it. The first sentence of the Comparison
section ("ReLU is the cheapest element-wise activation available") and the concluding `ReLU <<
SiLU ≈ GELU` code block both restate what is already in `index.md`.

**Recommended action:** Do not remove the Comparison section from `silu_sfpu_execution.md` —
it contains load-bearing instruction counts in the table. However, remove the redundant
concluding `ReLU << SiLU ≈ GELU` code block (lines 101–104) since the ordering is already the
learning objective and is implied by the table immediately above it. The table itself is
non-redundant and should stay.

**Estimated line savings:** ~5 lines.

---

### M3: The Data Type Considerations table in tensix_compute_engine.md is a minor forward-looking summary that restates the BF16 path already described in the main text

`tensix_compute_engine.md` lines 89–97 present a three-row table covering BF16, BFP8, and
FP32 data types. The prose immediately following the table (lines 95–97) then states: "For
Qwen3 MoE and similar models running in BF16, the BF16 matmul output flows directly from the
FPU destination register into the SFPU." This is a restatement of the SFPU section already
established at lines 49–62, where BF16 is the described path throughout. BFP8 and FP32 rows
in the table are not referenced anywhere else in the chapter.

**Recommended action:** Retain the table (it is a useful quick reference for readers coming
from non-BF16 contexts) but delete the prose paragraph following it (lines 95–97) since it
adds no information beyond "we are using BF16," which is established by the chapter context.

**Estimated line savings:** ~3 lines.

---

## Summary

| Metric | Value |
|---|---|
| Files analyzed | 4 |
| Current line count | 473 lines (index.md: 48, tensix_compute_engine.md: 117, silu_sfpu_execution.md: 130, cycles_vs_matmul.md: 178) |
| Estimated post-compression line count | ~440 lines |
| Estimated reduction | ~33 lines (~7%) |

Breakdown of estimated savings by suggestion:

| Suggestion | Savings |
|---|---|
| C1: 32-pass derivation dedup (tensix_compute_engine.md) | ~6 lines |
| C2: SFPU budget formula dedup (cycles_vs_matmul.md) | ~4 lines |
| C3: "Why This Matters" section removal (tensix_compute_engine.md) | ~12 lines |
| M1: Redundant Next Steps footers | ~4 lines |
| M2: Redundant ReLU << SiLU ≈ GELU concluding block | ~5 lines |
| M3: BF16 prose restatement after data type table | ~3 lines |
| **Total** | **~34 lines** |

The chapter is not heavily redundant overall. The content is well-scoped and each file has a
distinct primary responsibility. The duplication is concentrated in three specific locations:
the 32-pass constraint (C1), the SFPU instruction budget formula (C2), and the motivational
bridge section in `tensix_compute_engine.md` (C3). The MINOR items are cosmetic and low-risk.

---

## VERDICT

Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- tensix_compute_engine.md: Replaced duplicate 32-pass derivation with brief statement + cross-reference to silu_sfpu_execution.md
- cycles_vs_matmul.md: Replaced duplicate 32×(5-8)/tile formula with one-line reference to silu_sfpu_execution.md
- tensix_compute_engine.md: Removed "Why This Matters for SiLU in MoE" section body; replaced with forward reference to cycles_vs_matmul.md

---

# Compression Analysis: Chapter 2 — SiLU on Wormhole Hardware — Pass 2

## Summary

| Metric | Value |
|---|---|
| Files analyzed | 4 |
| Current line count | 457 lines (index.md: 47, tensix_compute_engine.md: 106, silu_sfpu_execution.md: 129, cycles_vs_matmul.md: 175) |
| Pass 1 line count (baseline) | 473 lines |
| Lines removed in Pass 1 | ~16 lines (actual; estimate was ~22 for C1+C2+C3 combined) |
| Estimated post-Pass-2 line count | ~448 lines |
| Estimated Pass 2 reduction | ~9 lines (~2%) |

---

## Pass 1 Fix Verification

All three CRUCIAL fixes from Pass 1 have been correctly applied:

**Fix 1 — C1 (tensix_compute_engine.md): 32-pass derivation replaced.**
`tensix_compute_engine.md` line 60 now reads: "A BF16 tile's 1024 elements require 32 SFPU passes; see `silu_sfpu_execution.md` for the full derivation in the context of SiLU." The original 3-sentence paragraph restating the 1024/32 = 32 derivation has been removed. The cross-reference is in place. VERIFIED.

**Fix 2 — C2 (cycles_vs_matmul.md): SFPU instruction budget formula replaced.**
`cycles_vs_matmul.md` lines 44–46 now read: "As derived in `silu_sfpu_execution.md`, a SiLU tile requires 160–256 SFPU instructions." The verbatim `32 passes × (5–8 instructions/pass) = 160–256 SFPU instructions per tile` formula has been removed. The back-reference is in place. VERIFIED.

**Fix 3 — C3 (tensix_compute_engine.md): "Why This Matters for SiLU in MoE" section body replaced.**
`tensix_compute_engine.md` lines 99–101 now contain only the section heading and a single forward-reference sentence: "For the quantitative matmul vs. SiLU cost comparison — including roofline placement and the estimated SiLU fraction of FFN time — see `cycles_vs_matmul.md`." The original 11-line section body restating the sequential constraint and previewing memory-bound arguments has been removed. VERIFIED.

---

## CRUCIAL Suggestions

None — all Pass 1 CRUCIAL items resolved.

The three previously identified crucial duplications (C1, C2, C3) have all been addressed. No new crucial duplications were identified in Pass 2. Each file now has a clear and non-overlapping primary responsibility:
- `tensix_compute_engine.md`: hardware architecture, the three pipelines, the FPU destination register sharing constraint, and the data type table.
- `silu_sfpu_execution.md`: authoritative location for the 32-pass derivation, the LLK instruction sequence, the activation comparison table, and the `math_approx_mode` analysis.
- `cycles_vs_matmul.md`: authoritative location for tile-count arithmetic, roofline placement, and the 15–40% SiLU fraction estimate.
- `index.md`: chapter map and reading order only.

---

## MINOR Suggestions

The following unresolved MINOR items carry forward from Pass 1. None have been addressed yet.

### M1 (carried forward): Remove redundant "Next Steps" footers from tensix_compute_engine.md and silu_sfpu_execution.md

`tensix_compute_engine.md` lines 105–106 and `silu_sfpu_execution.md` lines 128–129 each contain a "Next Steps" callout directing the reader to the next file. `index.md` lines 43–47 already defines the reading order explicitly in prose. The inter-file "Next Steps" footers duplicate this navigation. The footer in `cycles_vs_matmul.md` (lines 173–175) pointing to Chapter 3 should be kept — that cross-chapter pointer is not covered by `index.md`.

**Recommended action:** Remove the "Next Steps" paragraphs from `tensix_compute_engine.md` (lines 105–106) and `silu_sfpu_execution.md` (lines 128–129). Keep the Chapter 3 pointer in `cycles_vs_matmul.md`.

**Estimated line savings:** ~4 lines across two files.

---

### M2 (carried forward): Remove redundant `ReLU << SiLU ≈ GELU` concluding code block from silu_sfpu_execution.md

`silu_sfpu_execution.md` lines 100–104 contain a standalone code block that states:

```
ReLU  <<  SiLU ≈ GELU
```

This ordering is already stated as learning objective 5 in `index.md` (line 24–25) and is directly implied by the comparison table immediately above it in `silu_sfpu_execution.md` (lines 86–90). The code block is a restatement, not new information.

**Recommended action:** Remove the `ReLU << SiLU ≈ GELU` code block (lines 100–104 in `silu_sfpu_execution.md`). The comparison table is load-bearing and must stay; the redundant concluding block is not.

**Estimated line savings:** ~5 lines.

---

### M3 (carried forward): Remove BF16 prose paragraph following the Data Type Considerations table in tensix_compute_engine.md

`tensix_compute_engine.md` lines 93–95 contain a prose paragraph: "For Qwen3 MoE and similar models running in BF16, the BF16 matmul output flows directly from the FPU destination register into the SFPU — there is no precision conversion overhead beyond what the LLK sequence itself performs." This restates the BF16 row of the table immediately above it and adds no new information beyond what is already established by the chapter context.

**Recommended action:** Delete lines 93–95 in `tensix_compute_engine.md`.

**Estimated line savings:** ~3 lines.

---

## Load-Bearing Evidence

The following items were confirmed present in their authoritative locations after Pass 1 edits. They must not be removed in any future pass.

1. **FPU destination register sharing as the root cause of sequential FPU→SFPU execution.** Present in `tensix_compute_engine.md` lines 43–45 and 78. The mechanistic explanation for why SiLU latency is additive to matmul latency, not overlapping. Confirmed intact after Pass 1 edits.

2. **SFPU LReg is 32 elements wide; 32 sequential passes required per 32×32 BF16 tile (1024/32 = 32).** Now canonical in `silu_sfpu_execution.md` lines 62–67. The cross-reference from `tensix_compute_engine.md` line 60 points here. The derivation is no longer duplicated.

3. **SiLU LLK instruction sequence: negate → exp approx → add → reciprocal → multiply; 5–8 SFPU instructions per pass; 160–256 per tile.** Present in `silu_sfpu_execution.md` lines 35–45 (table) and lines 69–74 (derivation). This is the primary technical payload of the chapter and is now the sole canonical location for both the instruction sequence and the instruction-count formula.

4. **`math_approx_mode` saves ~2 instructions/pass (~64 instructions/tile) at the cost of small numerical error in sigmoid tails.** Present in `silu_sfpu_execution.md` lines 117–124 (table). The only actionable tuning lever in the chapter. Confirmed intact.

5. **Gate_proj matmul at 32 tokens has arithmetic intensity ~31.6 FLOP/byte, placing it well below Wormhole's ~437 FLOP/byte ridge point (memory-bound).** Quantitative derivation present in `cycles_vs_matmul.md` lines 86–94. Confirmed intact.

6. **SiLU has arithmetic intensity ~0.5 FLOP/byte and is memory-bound at all practical tensor sizes.** Present in `cycles_vs_matmul.md` lines 103–114. Confirmed intact.

7. **SiLU ≈ 15–40% of gate_proj matmul latency when both are memory-bound.** Present in `cycles_vs_matmul.md` line 155. Explicitly flagged as the prior for experimental design. The supporting byte-footprint table (lines 145–148) and the 160–256 instructions reference (line 152) are also intact. Confirmed intact.

---

## VERDICT

Crucial updates: no
