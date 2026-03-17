# Compression Analysis — Chapter 3: Batched Matmul for MoE — Pass 1

## Summary
- Files reviewed: `index.md`, `formulating_batched_matmul.md`, `program_configs_batched.md`, `performance_profile_batched.md`
- Current line count: ~1013 lines (total across all files: 96 + 296 + 408 + 213)
- Estimated post-compression line count: ~830 lines (~18% reduction)

---

## CRUCIAL Suggestions

**C1 — Duplicate FLOP efficiency formula and decode example across `formulating_batched_matmul.md` and `performance_profile_batched.md`**

- `formulating_batched_matmul.md` lines 122–132: Defines FLOP efficiency as `filled slots / (C × E) = k × B × S / (C × E)`, shows decode example (B=32, k=8, S=1, C=1 → efficiency = 1.0), and explains that imbalance pushes efficiency below 1.0.
- `performance_profile_batched.md` lines 56–82: Re-defines the identical formula with the same symbolic expression, re-runs the same B=32/S=1/C=1 decode example reaching the same 1.0 conclusion, and re-explains imbalance degradation.

The second instance adds one new detail not in the first: the tile-level efficiency drop to ~1/32 at C=1 (lines 78–82 in `performance_profile_batched.md`). That detail is load-bearing and must be retained. However, the formula re-statement, the worked decode example, and the imbalance warning at lines 56–73 of `performance_profile_batched.md` are fully redundant with `formulating_batched_matmul.md`. Content lost if cut: none — a cross-reference to `formulating_batched_matmul.md` Section 2.3 suffices. Savings: ~15 lines.

**C2 — Duplicate gather/scatter cost characterization across `formulating_batched_matmul.md` and `performance_profile_batched.md`**

- `formulating_batched_matmul.md` line 74 (Warning block): Explains that the gather loop is illustrative, that a device op is required in production, and that host-side gather cost grows linearly with B×S.
- `performance_profile_batched.md` lines 17–23 (Section 1.1) and lines 136–141 (Section 4.2): Both re-characterize gather as a strided DRAM read with poor locality, note the O(k×B×S×H) cost, and quantify the 0.46 MB (decode) and 29.4 MB (prefill) buffer sizes.

The quantified buffer sizes (0.46 MB / 29.4 MB) in `performance_profile_batched.md` Section 4.2 are load-bearing — they appear nowhere in `formulating_batched_matmul.md`. However, Section 1.1 of `performance_profile_batched.md` duplicates the cost characterization already stated in the Warning at `formulating_batched_matmul.md` line 74 without adding new quantitative content. That portion (lines 17–19 of `performance_profile_batched.md`) can be compressed to a one-line reference. Savings: ~4 lines.

**C3 — Duplicate "use MatmulMultiCoreProgramConfig for decode, MatmulMultiCoreReuseMultiCastProgramConfig for prefill" decision across `program_configs_batched.md` and `index.md`**

- `index.md` lines 52–53 (Summary table): States the config selection rule (decode C=1 → `MatmulMultiCoreProgramConfig`; prefill/high-C → `MatmulMultiCoreReuseMultiCastProgramConfig`).
- `program_configs_batched.md` lines 12–34 (Section 1.1): Restates the same decision tree from Chapter 2 and re-applies it to the MoE batched case, reaching the same conclusion.

The `index.md` summary is purposefully a preview/overview; the duplication is minor in that `index.md` is meant to be a map. However, the decision-tree code block in `program_configs_batched.md` lines 18–22 is a verbatim reproduction of a tree already attributed to Chapter 2 (`matmul_fundamentals_in_ttnn.md` Section 4.3). Since Chapter 2 is a prerequisite, this re-copy adds no new content. It could be replaced with a prose sentence and cross-reference, removing the code fence. Content lost: none. Savings: ~7 lines.

**C4 — Near-duplicate decode C=1 example computed three times**

The capacity formula calculation for B=32, S=1, C=1 is worked out in full in three places:
- `formulating_batched_matmul.md` lines 110–111 (Section 2.2 example)
- `formulating_batched_matmul.md` lines 222–224 (Section 4, Step 2)
- `program_configs_batched.md` lines 65–67 (Table, row for C=1)

The Section 2.2 example (first occurrence) is definitional. The Step 2 worked example in Section 4 legitimately re-uses the result in context and serves the worked-example narrative. The table row in `program_configs_batched.md` is also legitimate as a quick-reference lookup. However, the prose sentence at `formulating_batched_matmul.md` lines 223–224 ("Each expert receives exactly 1 token slot on average. Under uniform routing, each of the 256 experts receives exactly 1 of the 256 assignments.") repeats the explanation already given in full at lines 100–101 of the same file. That sentence pair can be cut from the worked example. Content lost: none. Savings: ~2 lines.

---

## Load-Bearing Evidence

- **`index.md`**: The Pros/Cons/Recommended-Use table (lines 55–68) is the only place across all four files that consolidates the trade-off framing in a scannable comparative format; it should not be cut. The Chapter Notation table (lines 30–43) is the canonical symbol reference for the chapter and is referenced by downstream files.

- **`formulating_batched_matmul.md`**: The complete decode forward-pass worked example (Section 4, lines 198–290) is irreplaceable: it is the only place that traces all six tensor shapes (hidden_states through output_buffer) end-to-end with dtypes and tile-padding effects. The capacity-factor α explanation (lines 102–106) and the distinction between average-case formula and imbalance reality (lines 130–132) are load-bearing nuance not repeated elsewhere.

- **`program_configs_batched.md`**: The `validate_config` Python function (lines 286–332) and `estimate_l1_bytes` function (lines 338–366) are the only executable, copy-pasteable validation tools in the chapter — they must not be cut. The `per_core_M` scaling table (lines 65–73) with concrete grid_y suggestions is a useful lookup not found in other files.

- **`performance_profile_batched.md`**: The arithmetic intensity derivation (Section 3, lines 86–122) — including the ridge-point calculation (AI_ridge ≈ 682 FLOPs/byte), the large-C saturation limit HD/(H+D), and the decode AI ≈ 2 result — appears nowhere else in the chapter and is analytically foundational. The tile-level efficiency drop to 1/32 at C=1 (Section 2.2, lines 78–82) is also unique to this file and explains the primary decode bottleneck.

---

## MINOR Suggestions

**M1 — Redundant "Next Steps" navigation prose at file ends**

Each file ends with a paragraph directing the reader to the next file. Within a rendered guide with hyperlinked navigation, the prose transitions at `formulating_batched_matmul.md` line 295, `program_configs_batched.md` lines 405–407, and `performance_profile_batched.md` lines 209–212 repeat the sequential reading order already stated in `index.md` lines 80 and 93. The terminal paragraph in each file could be reduced to a single line. Savings: ~8 lines.

**M2 — Repeated "[D UNVERIFIED — verify against Qwen3 Technical Report]" inline tags**

This annotation appears approximately 28 times across the four files. In `formulating_batched_matmul.md` alone it appears 15+ times. A single prominent callout box at the top of `formulating_batched_matmul.md` (or in `index.md`'s notation table where D already carries the tag) stating that all D-dependent quantities are unverified would allow most inline repetitions to be shortened to `[D UNVERIFIED]` (already the shortened form used inconsistently) or removed where the surrounding sentence makes context obvious (e.g., in code comments immediately following a prose warning). This would not lose any information. Savings: ~10 lines across files.

**M3 — Redundant BFP8 byte-size explanation repeated in `program_configs_batched.md` and `performance_profile_batched.md`**

- `program_configs_batched.md` lines 128–131: Explains tile_bytes_B = 1088 as "32×32 × 1 byte + 64 bytes exponent overhead."
- `performance_profile_batched.md` lines 96–98: Re-explains BFP8 as "1 byte per element for the mantissa plus a shared exponent per 16 elements, giving approximately 1.0625 bytes/element."

These two explanations say the same thing in different detail levels. One of them (the shorter approximation in `performance_profile_batched.md`) can defer to the definition in `program_configs_batched.md`. Savings: ~3 lines.

**M4 — Verbose preamble sentences at top of `program_configs_batched.md`**

Lines 3–5 of `program_configs_batched.md` state that "all config vocabulary is defined in Chapter 2 and is used here without re-derivation." This is already established as a prerequisite in `index.md` lines 87–88. The sentence is not wrong, but it restates prerequisite scope that is implicit. It could be removed without loss. Savings: ~2 lines.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: Removed redundant FLOP efficiency formula re-definition, decode example (B=32/S=1/C=1), and imbalance warning from `performance_profile_batched.md` Section 2; replaced with cross-reference to `formulating_batched_matmul.md` Section 2.3. Load-bearing tile-level 1/32 granularity content retained.
- C2 applied: Compressed gather cost characterization in `performance_profile_batched.md` Section 1.1 to a single cross-reference sentence; quantified buffer sizes (0.46 MB / 29.4 MB) retained.
- C3 applied: Replaced fenced decision-tree code block in `program_configs_batched.md` Section 1.1 with a prose sentence citing Chapter 2 Section 4.3.
- C4 applied: Deleted redundant sentence pair in `formulating_batched_matmul.md` Section 4 Step 2 (C=1 uniform-routing explanation repeated from Section 2.2).

---

# Compression Analysis — Chapter 3: Batched Matmul for MoE — Pass 2

## Summary
- Files reviewed: `index.md`, `formulating_batched_matmul.md`, `program_configs_batched.md`, `performance_profile_batched.md`
- Current line count: ~980 lines (total: 95 + 293 + 399 + 193)
- Estimated post-compression: ~955 lines (~3% reduction)

## Pass 1 Item Verification

**C1 — ADDRESSED.** `performance_profile_batched.md` Section 2 now opens with a cross-reference to `formulating_batched_matmul.md` Section 2.3 ("For the FLOP efficiency formula and decode worked example, see...") and jumps directly to the load-bearing tile-level granularity content (Section 2.2). The formula re-statement and worked B=32/S=1 example are absent from `performance_profile_batched.md`.

**C2 — ADDRESSED.** `performance_profile_batched.md` Section 1.1 is now a single sentence ("Gather is a strided DRAM read — see `formulating_batched_matmul.md` §1 Warning — with quantified buffer sizes:") followed immediately by the quantified buffer figures (0.46 MB / 29.4 MB). The redundant cost-characterization prose has been removed.

**C3 — ADDRESSED.** `program_configs_batched.md` Section 1.1 now reads "Apply the Chapter 2 config selection rule (`matmul_fundamentals_in_ttnn.md` Section 4.3): use `MatmulMultiCoreProgramConfig` when C=1 (decode)..." — prose sentence, no fenced code block for the decision tree.

**C4 — ADDRESSED.** The sentence pair "Each expert receives exactly 1 token slot on average. Under uniform routing, each of the 256 experts receives exactly 1 of the 256 assignments." is absent from `formulating_batched_matmul.md` Section 4 Step 2. A grep for the phrase confirms no match.

## CRUCIAL Suggestions

None.

## Load-Bearing Evidence

- **`index.md`** (95 lines): The Pros/Cons/Recommended-Use table (lines 55–68) is the only consolidated trade-off summary in the chapter; the Chapter Notation table (lines 30–43) is the canonical symbol reference cited by downstream files. Both must be preserved.

- **`formulating_batched_matmul.md`** (293 lines): The complete decode forward-pass worked example (Section 4, ~90 lines tracing all six tensor shapes with dtypes and tile-padding effects) is irreplaceable. The capacity factor α discussion (lines 102–106) and the imbalance caveat (lines 130–132) are the only treatments of those nuances in the chapter.

- **`program_configs_batched.md`** (399 lines): The `validate_config` Python function (~40 lines) and `estimate_l1_bytes` function (~25 lines with worked example) are the only executable validation tools in the chapter. The `per_core_M` scaling table (lines 57–66) with concrete grid_y suggestions is a useful lookup found nowhere else.

- **`performance_profile_batched.md`** (193 lines): The arithmetic intensity derivation (Section 3, ~35 lines) — including the ridge-point calculation (AI_ridge ≈ 682 FLOPs/byte), large-C saturation limit HD/(H+D), and decode AI ≈ 2 result — is analytically unique and must be retained in full. The tile-level efficiency drop to 1/32 at C=1 (Section 2.2) is also unique here and explains the primary decode bottleneck.

## MINOR Suggestions

**M1 — Missing `### 2.1` subsection heading in `performance_profile_batched.md`.**
The C1 deletion left `## 2. FLOP Efficiency vs. Expert Capacity` jumping directly to `### 2.2 Why Decode is Particularly Inefficient` with no `### 2.1` subsection. This is a structural artifact. Either rename `2.2` to `2.1`, or add a one-line `### 2.1 Overview` bridging sentence. Savings: 0 lines (no redundancy), but fixes a broken section numbering that will confuse readers. This is a correctness fix, not a compression item.

**M2 — Redundant "[D UNVERIFIED — verify against Qwen3 Technical Report]" inline tags (58 total across all four files: 15 in `formulating_batched_matmul.md`, 31 in `program_configs_batched.md`, 10 in `performance_profile_batched.md`, 2 in `index.md`).**
Pass 1 Minor item M2 was not applied. A single callout block at the top of `formulating_batched_matmul.md` (or in the `index.md` notation table, where D already carries the unverified tag) stating that all D-dependent quantities are unverified would allow the majority of inline repetitions to be reduced to the short form `[D UNVERIFIED]` or omitted entirely where the surrounding sentence makes the context obvious (e.g., in code comments immediately following a prose warning). Estimated savings: ~10 lines.

**M3 — Redundant BFP8 byte-size explanation (Pass 1 Minor item M3, not yet applied).**
`program_configs_batched.md` line 122 defines `tile_bytes_B = 1088` as "32×32 × 1 byte + 64 bytes exponent overhead." `performance_profile_batched.md` line 79 re-explains BFP8 as "1 byte per element for the mantissa plus a shared exponent per 16 elements, giving approximately 1.0625 bytes/element (= 1088/1024 = 17/16)." These are two explanations of the same quantity at different detail levels. The `performance_profile_batched.md` instance could be condensed to a parenthetical "(BFP8 ≈ 1.0 byte/element; see `program_configs_batched.md` Section 3 for tile-level derivation)." Savings: ~2 lines.

**M4 — Verbose "Next Steps" navigation sections (Pass 1 Minor item M1, not yet applied).**
Each of `formulating_batched_matmul.md`, `program_configs_batched.md`, and `performance_profile_batched.md` ends with a `## Next Steps` heading plus a multi-clause navigation sentence. The reading order is already fully specified in `index.md` lines 80 and 93. Each terminal section could be reduced to a single bare link line (e.g., `Next: [program_configs_batched.md](program_configs_batched.md)`). Savings: ~6 lines across three files.

VERDICT: Crucial updates: no
