# Compression Analysis: Chapter 3 — Memory Layout Transitions and L1 Pressure — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~405 lines
- Estimated post-compression line count: ~315 lines
- Estimated reduction: ~22%

---

## CRUCIAL Suggestions

### C1 — `transition_analysis.md`: Remove the restated data-volume table in the "Three Lives of K" section (lines 145–153)

The table in the "Symmetry Summary: The Three Lives of K" section restates byte volumes that were already stated individually in the per-transition sections (Steps 8b, 12d, 16a). The prose summary following it (`Total: 98,304 bytes… By contrast V is copied once…`) also repeats the totals from the quantitative baseline table in `index.md`. The entire section can be trimmed to two sentences without loss of information.

**Suggested replacement:**

> K is the most-copied tensor: three complete copies totalling 98,304 bytes (steps 8b → 12d → 16a). V is copied once; Q is copied twice (steps 8a, 12c), and the SDPA output `attn_output` adds a third 131 KB DRAM→L1 move at step 20.

Saves approximately 12 lines.

---

### C2 — `avoidable_transitions.md`: Collapse the duplicate per-step byte table in section 3 (lines 160–168)

The table under "Why it is potentially avoidable" (section 3, Step 20) lists all six avoidable transitions with byte costs. Every entry in that table duplicates values already established in `index.md`'s quantitative-baseline table and in each step's own "Data volume" paragraph in `transition_analysis.md`. The table's only new information is the grand total (491,520 bytes / ~91%). Cut the table; keep only the two-line summary:

> Eliminating all avoidable transitions saves approximately 491,520 bytes (~480 KB) of NoC data movement per decode step per device — roughly 91% of the total 528 KB. Step 20 is the single highest-value elimination (131 KB, DRAM→L1).

Saves approximately 10 lines.

---

### C3 — `index.md`: Deduplicate the "Note on V asymmetry" paragraph (lines 49–50)

The note at the end of the inventory table ("V does not pass through steps 8b, 12a, 12b, or 12d… V travels directly from the all-gather output…") is restated almost verbatim in `transition_analysis.md` step 16b ("V received no prior L1 ops…") and again in the "Three Lives of K" closing sentence. One occurrence is sufficient. The `index.md` copy is the most redundant because the detail does not belong in the inventory table's footnote — it belongs in the per-step analysis. Remove the paragraph from `index.md`.

Saves approximately 4 lines.

---

## MINOR Suggestions

### M1 — `transition_analysis.md` steps 12a/12b: Trim hedging in the avoidability paragraph (line 52)

> "but they cannot be eliminated entirely unless the RoPE kernel is changed to accept DRAM input in decode mode"

This restates the constraint already explained in the "Why it is required" paragraph two sentences earlier. The avoidability paragraph can end at "reduce scheduling overhead" without loss of meaning.

---

### M2 — `avoidable_transitions.md` section 1: Remove over-long inline arithmetic in the pre-all-gather feasibility block (lines 33–54)

The repeated expansions of `[B, 1, 256]` → "2 heads × head_dim=128 = 256 elements" and the inline cost comparison (`32 * 256 * 2 = 16,384 bytes versus B * H * D * 2 = 131,072 bytes`) are spelled out twice within three paragraphs. The second expansion (lines 48–54) repeats numbers already computed five lines above. Trim the second expansion to a single reference: "an 8× reduction matching the tensor-parallelism factor N=8."

---

### M3 — `avoidable_transitions.md` section 2: Shorten the `kv_vol` inline comment chain (lines 109–111)

The code block comment `# = (32 * 4 * 128) // 128 // 32 = 16384 // 128 // 32 = 128 // 32 = 4` expands every intermediate division step. A single `# = Hkv = 4` is sufficient; the expansion adds no insight beyond what the surrounding prose already states.

---

### M4 — `avoidable_transitions.md` section 2: Remove restatement of the incompatibility reason (lines 117–118)

> "The incompatibility is fundamental: TILE_SIZE=32 versus Hkv=4. The RoPE kernel uses TILE_SIZE as the natural processing unit (tiles are 32×32 in TTNN on Wormhole), while the paged cache update kernel organises data by KV-head count per batch element."

This restates the explanation given three paragraphs earlier under "Why step 16a is required" in `transition_analysis.md` (line 109). One of the two occurrences should be removed; the `transition_analysis.md` version is more concise and should be kept.

---

### M5 — `index.md`: Trim the opening "Two distinct costs apply" enumeration

The three bold sub-headings (DRAM → L1 interleaved; Interleaved L1 → HEIGHT_SHARDED; HEIGHT_SHARDED → different HEIGHT_SHARDED) each restate the transition type that is then explained in the paragraph. The bold label can be dropped because the paragraph's first sentence already names the transition. This saves 3 label lines with no information loss.

---

## Load-Bearing Evidence

- **`index.md` (line 29):** `"Total: approximately 528 KB of NoC data movement per decode step, just from layout transitions."` — the 528 KB baseline figure is the anchor for every "bytes saved" claim downstream; this line must not be cut.

- **`transition_analysis.md` (line 24):** `"The comment on line 2655 reads 'Move to L1 for QK norm (reshape doesn't work on sharded tensors)', confirming that the actual blocker is the reshape inside _apply_qk_norm, not the norm kernel itself."` — this is the only place that disambiguates the reshape constraint from the norm constraint; removing it would leave the root cause analysis incomplete.

- **`avoidable_transitions.md` (line 56):** `"because Hkv=4 < N=8 (there are more devices than KV heads), col-sharding does not produce complete per-head slices on each device"` — this is the key asymmetry that explains why the pre-all-gather QK norm proposal is straightforward for Q but non-trivial for K; it must be preserved.

---

## VERDICT
- Crucial updates: yes

## Agent A Change Log — Compression Pass 1
- C1: Removed the 3-row data-volume table and its multi-sentence prose summary in the "Symmetry Summary: The Three Lives of K" section of transition_analysis.md (lines 146–152). Replaced with two sentences covering K's three copies (98,304 bytes total), V's single copy, Q's two copies, and the attn_output DRAM→L1 move at step 20.
- C2: Removed the 6-row per-step byte table in avoidable_transitions.md section 3 (Step 20) that listed all six avoidable transitions with individual byte costs. Replaced with the two-line summary stating 491,520 bytes (~480 KB, ~91% of total) saved and identifying step 20 as the single highest-value elimination (131 KB, DRAM→L1).
- C3: Removed the "Note on V asymmetry" paragraph from index.md (previously between the inventory table and the horizontal rule). The paragraph explained that V skips QK norm and RoPE and crosses DRAM→L1 only once; this is already covered in transition_analysis.md step 16b and the C1-reduced Three Lives of K summary.

---

# Compression Analysis: Chapter 3 — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~386 lines
- Estimated post-compression line count: ~368 lines
- Estimated reduction: ~5%

---

## CRUCIAL Suggestions

### C1 (re-check) — RESOLVED
The "Symmetry Summary: The Three Lives of K" table and prose block in `transition_analysis.md` were fully replaced by Agent A with the two-sentence summary specified in Pass 1. The current lines 142–148 contain exactly the suggested replacement text. No residual redundancy.

### C2 (re-check) — RESOLVED
The per-step byte table in `avoidable_transitions.md` section 3 was removed by Agent A. The current text at lines 155–161 contains only the two-line summary (`491,520 bytes… ~91%… Step 20 is the single highest-value elimination…`). No residual redundancy.

### C3 (re-check) — RESOLVED
The "Note on V asymmetry" paragraph was removed from `index.md` by Agent A. The inventory table now ends at line 47 and is followed immediately by `---`. No residual redundancy.

---

## MINOR Suggestions

### M1 (carry-forward from Pass 1) — `transition_analysis.md` line 52: Remove trailing hedging clause in steps 12a/12b avoidability paragraph

The sentence ending "but they cannot be eliminated entirely unless the RoPE kernel is changed to accept DRAM input in decode mode" restates the hard kernel precondition already stated in the "Why it is required" paragraph two sentences earlier. The avoidability paragraph's final sentence can be trimmed to end at "reduce scheduling overhead" — the reader already knows the RoPE kernel constraint from the preceding paragraph. Saves 1 line / ~20 words.

### M2 (new) — `avoidable_transitions.md` lines 117–118: Remove second statement of TILE_SIZE vs. Hkv incompatibility

The sentence "The incompatibility is fundamental: TILE_SIZE=32 versus Hkv=4. The RoPE kernel uses TILE_SIZE as the natural processing unit (tiles are 32×32 in TTNN on Wormhole), while the paged cache update kernel organises data by KV-head count per batch element." is the third place this incompatibility is stated. It was explained once in `transition_analysis.md` step 16a ("Why step 16a is required", line 109) and once in the preceding shard-spec comparison paragraphs within `avoidable_transitions.md` section 2 itself. The two sentences at lines 117–118 can be removed without information loss. Saves ~3 lines.

---

## Load-Bearing Evidence

- **`index.md` (line 29):** `"Total: approximately 528 KB of NoC data movement per decode step, just from layout transitions."` — the 528 KB baseline is the anchor for all downstream "bytes saved" claims across both `transition_analysis.md` and `avoidable_transitions.md`; this line must not be cut.

- **`transition_analysis.md` (line 24):** `"The comment on line 2655 reads 'Move to L1 for QK norm (reshape doesn't work on sharded tensors)', confirming that the actual blocker is the reshape inside _apply_qk_norm, not the norm kernel itself."` — the only place that disambiguates the reshape constraint from the norm constraint; removing it leaves the root-cause analysis incomplete.

- **`avoidable_transitions.md` (line 56):** `"because Hkv=4 < N=8 (there are more devices than KV heads), col-sharding does not produce complete per-head slices on each device"` — the key asymmetry that explains why the pre-all-gather QK norm proposal is feasible for Q but non-trivial for K; must be preserved.

---

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Chapter 3 — Pass 3

## Summary
- Total files analyzed: 3
- Estimated current line count: ~389 lines (index.md ~59, transition_analysis.md ~151, avoidable_transitions.md ~179)
- Estimated post-compression line count: ~385 lines
- Estimated reduction: ~1%

---

## CRUCIAL Suggestions

### C1 (re-check) — RESOLVED
`transition_analysis.md` lines 142–148 contain exactly the two-sentence replacement specified in Pass 1 ("K is the most-copied tensor: three complete copies totalling 98,304 bytes (steps 8b → 12d → 16a). V is copied once; Q is copied twice (steps 8a, 12c), and the SDPA output `attn_output` adds a third 131 KB DRAM→L1 move at step 20."). No table or multi-sentence prose block remains. No residual redundancy.

### C2 (re-check) — RESOLVED
`avoidable_transitions.md` section 3 (Step 20, lines 155–161) contains only the two-line summary ("Eliminating all avoidable transitions saves approximately 491,520 bytes (~480 KB)… Step 20 is the single highest-value elimination (131 KB, DRAM→L1)."). The six-row per-step byte table from Pass 1 is absent. No residual redundancy.

### C3 (re-check) — RESOLVED
`index.md` inventory table ends at line 47 and is followed immediately by `---`. The "Note on V asymmetry" paragraph removed by Agent A in Pass 1 is not present. No residual redundancy.

---

## MINOR Suggestions

### M1 (carry-forward from Pass 2, still open) — `transition_analysis.md` line 52: Trim trailing hedging clause in steps 12a/12b avoidability paragraph

The clause "but they cannot be eliminated entirely unless the RoPE kernel is changed to accept DRAM input in decode mode" repeats the hard kernel precondition already stated in the "Why it is required" paragraph immediately above it. Trimming the avoidability paragraph's final sentence to end at "reduce scheduling overhead" removes the restatement without loss of meaning. Saves approximately 1 line / 20 words.

---

## Load-Bearing Evidence

- **`index.md` (line 29):** "Total: approximately 528 KB of NoC data movement per decode step, just from layout transitions." — the 528 KB baseline anchors every "bytes saved" claim in both downstream files; must not be cut.

- **`transition_analysis.md` (line 24):** "The comment on line 2655 reads 'Move to L1 for QK norm (reshape doesn't work on sharded tensors)', confirming that the actual blocker is the reshape inside `_apply_qk_norm`, not the norm kernel itself." — the only location that disambiguates the reshape constraint from the norm constraint; removing it leaves the root-cause analysis incomplete.

- **`avoidable_transitions.md` (line 56):** "because Hkv=4 < N=8 (there are more devices than KV heads), col-sharding does not produce complete per-head slices on each device" — the key asymmetry establishing why pre-all-gather QK norm is feasible for Q but non-trivial for K; must be preserved.

---

## VERDICT
- Crucial updates: no
