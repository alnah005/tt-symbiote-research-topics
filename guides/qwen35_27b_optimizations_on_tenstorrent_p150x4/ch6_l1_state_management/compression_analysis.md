# Compression Analysis: Chapter 6 — L1 State Management and Rolling Window — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~349 lines
- Estimated post-compression line count: ~295 lines
- Estimated reduction: ~15%

## CRUCIAL Suggestions

### C1 — `sdpa_l1_conflict.md` lines 72–73: Full NOC transaction derivation restated verbatim from `l1_state_design.md`

`l1_state_design.md` line 15 already states:
> "the fused kernel issues 16 NOC tile reads and 16 NOC tile writes for the recurrence state per pair. With 384 pairs total, that is 12,288 NOC transactions per layer and roughly 590,000 state NOC transactions per full forward pass."

`sdpa_l1_conflict.md` lines 72–73 then repeat the same derivation word-for-word under "Current Status":
> "For each GDN layer the fused kernel currently issues 16 NOC tile reads and 16 NOC tile writes per pair. With 384 pairs total that is 12,288 NOC transactions per layer for state alone. HEIGHT_SHARDED reduces this to zero for state, leaving only Q/K/V/scalar reads from DRAM and the output writes to DRAM — which are inherently required regardless of state location."

The entire "expected impact" paragraph in `sdpa_l1_conflict.md` (lines 72–73, ~3 lines) is a copy of material already established in the Profiler Breakdown section. The Current Status table above it already conveys the practical outcome (DRAM baseline vs. validated L1 configurations). The paragraph adds nothing that a reader of `l1_state_design.md` has not already seen.

**Action:** Delete the "expected impact" paragraph in `sdpa_l1_conflict.md` (lines 72–73). Optionally add a one-line cross-reference: "For NOC transaction counts, see the Profiler Breakdown in `l1_state_design.md`." Net saving: ~3 lines.

---

### C2 — `height_sharded_kernel.md` lines 102–103: Shard arithmetic re-derived in prose immediately after the code that already contains it

The Python snippet (lines 89–99) already contains inline comments that show all intermediate values:
```
total_rows = batch_size * args.gdn_nv_tp * args.gdn_dk  # 32 * 12 * 128 = 49152
shard_h = total_rows // NUM_HS_CORES  # 512
```

The paragraph immediately following (lines 102–103) re-derives these same values step by step:
> "With `shard_h = 512` rows per core and `gdn_dv = 128` columns, each core holds 512 × 128 × 2 = 131,072 bytes (128 KB) of state. The 96-core grid spans total_rows / shard_h = 384 pairs across Dk = 128 rows each: each core handles 512 / 128 = 4 pairs worth of state tiles."

Every multiplication in this paragraph is already implicit from the annotated code above it. The paragraph walks the code line-by-line without introducing any new fact or context.

**Action:** Replace the two-sentence paragraph with a single result sentence: "Each core holds 4 pairs of state (128 KB); the 96-core grid covers all 384 pairs." Net saving: ~2 lines.

---

### C3 — `height_sharded_kernel.md` lines 127–135 and `sdpa_l1_conflict.md` lines 66–70: Validation status table duplicated with partial overlap

`height_sharded_kernel.md` "Validation Status" section (lines 127–135) reports:
- `test_e2e_l1_hs.py` with `N_L1_LAYERS = 2`: correct output, `assert is_l1` passes
- L1 INTERLEAVED works with up to 4 layers
- Both kernel modes compile correctly

`sdpa_l1_conflict.md` "Current Status" table (lines 66–70) is a strict superset, listing all three configurations (DRAM baseline, L1 INTERLEAVED, HEIGHT_SHARDED) with validation state and notes. The `height_sharded_kernel.md` validation section adds no row or fact not already present in the `sdpa_l1_conflict.md` table — it just describes the same data in prose and an incomplete table.

**Action:** Replace the `height_sharded_kernel.md` "Validation Status" section (~7 lines including heading and blank lines) with a single forward-reference sentence: "Validation status for all three state configurations is summarized in `sdpa_l1_conflict.md` §Current Status." Net saving: ~6 lines.

---

## MINOR Suggestions

### M1 — `l1_state_design.md` lines 29–30: Anti-fragmentation rationale duplicated

Line 29–30 contains: "This pre-allocated buffer is reused throughout inference — states are copied back into it rather than allocating new DRAM tensors, which avoids memory fragmentation." This rationale is restated at line 110: "Pre-allocated `_dram_state` buffers ensure zero DRAM allocation overhead during swaps." Only one instance is needed. Remove the parenthetical clause at line 29 ("which avoids memory fragmentation"); let line 110 carry the full explanation. Saving: ~0.5 lines of prose density.

### M2 — `sdpa_l1_conflict.md` lines 3–4: Opening sentence restates `index.md` context

The file opens: "The rolling window strategy keeps 3 GDN layers' states in L1 at a time and swaps groups at the boundary between each set of 3 GDN layers and the following attention layer." This is a direct restatement of the rolling window mechanism already described in `index.md` line 3 and covered in full in `l1_state_design.md`. Since the files are read in sequence, cut this sentence and open directly at "But the attention layer itself presents a problem." Saving: ~1 line.

### M3 — `l1_state_design.md` lines 65–66: Monkey-patching explanation is two sentences saying the same thing

"The `forward()` method... injects swap logic into the layer loop without modifying the parent `TTTransformer.forward()`. This is achieved by temporarily monkey-patching each GDN layer's `forward` method." The second sentence restates the mechanism already implied by "injects swap logic." Merge into one: "The `forward()` method injects swap logic by temporarily monkey-patching each GDN layer's `forward`, leaving the parent `TTTransformer.forward()` unmodified." Saving: ~1 line.

### M4 — `height_sharded_kernel.md` lines 117–123: C++ comment block quote adds no quantitative fact

Lines 119–123 quote a 3-line C++ comment from the writer kernel header explaining output tile layout mapping. The prose at line 125 already captures the load-bearing point ("The output tensor `[1, B, value_dim_tp]` feeds into the subsequent RMS norm and output projection, which expect DRAM-resident inputs"). The quoted comment restates this in slightly different words without adding a number or constraint not present in the prose. The block quote (lines 118–123, ~6 lines including fenced code markers) can be dropped, leaving only the prose sentence. Saving: ~5 lines.

### M5 — `l1_state_design.md` lines 100–101: Swap timing prose restates the preceding table

Lines 100–101 read: "The attention layer between each group runs after the old group's states have already been saved to DRAM by the swap that precedes the next group." The Swap Timing table directly above (lines 92–98) already shows this: GDN indices 3,4,5 trigger a "Yes" swap before they run, which implicitly means the attention layer between blocks 0 and 1 sees already-saved state. The sentence is a verbal summary of the table's visual structure and can be removed. Saving: ~1 line.

---

## Load-Bearing Evidence

- **`index.md` line 3** — "GDN layers consume 85% of total decode time, and the dominant cost within each layer is DRAM bandwidth for reading and writing the recurrence state." Cannot cut: sole motivation sentence for the chapter; removing it leaves the index as a structureless file table.

- **`l1_state_design.md` lines 9–14** (Profiler Breakdown table) — Exact ms values (469.6 / 69.2 / 15.7) are the only quantitative performance anchor in the chapter. All percentage claims and the "15 swaps" cost argument trace back to this table.

- **`l1_state_design.md` lines 73–84** (`make_wrapped_forward` factory code block and line 84 note) — The explanation of Python's closure late-binding problem is non-obvious and is the only place in the chapter that justifies the factory pattern. Cutting it would leave the implementation unmaintainable without re-discovering the problem from scratch.

- **`height_sharded_kernel.md` lines 106–113** (DRAM vs L1 INTERLEAVED vs HEIGHT_SHARDED comparison table) — The only location in the chapter where all three storage modes are compared on NOC transaction count and address generator type side by side. Directly underpins the conflict explanation in `sdpa_l1_conflict.md`.

- **`sdpa_l1_conflict.md` lines 36–62** (four Potential Solutions subsections) — Each solution documents a distinct engineering trade-off (address partitioning, CB footprint reduction, pre-allocation, hybrid approach). Not restated elsewhere; removing any subsection destroys forward-looking design options.

- **`sdpa_l1_conflict.md` line 14** — "The SDPA kernel's circular buffer region extends to approximately 1,264 KB per core." Cannot cut: this is the exact watermark number that gates all four potential solutions. The 240 KB remainder on a 1,504 KB Blackhole core (line 43) depends on it.

---

## VERDICT
- Crucial updates: yes

## Agent A Change Log — Pass 1 CRUCIAL fixes

Applied all 3 CRUCIAL suggestions:
1. sdpa_l1_conflict.md: Deleted duplicate NOC transaction derivation paragraph
2. height_sharded_kernel.md: Deleted prose re-derivation of shard arithmetic after code block
3. height_sharded_kernel.md: Replaced "Validation Status" section with single forward-reference sentence to sdpa_l1_conflict.md

---

## Pass 2

**Summary:** 0 crucial updates, 5 minor suggestions carried forward from Pass 1
**Crucial updates:** no

### Pass 1 Fix Verification

All three Pass 1 CRUCIAL fixes are confirmed in place:

1. **C1 confirmed** — `sdpa_l1_conflict.md` has no "expected impact" paragraph. The file ends at line 75 with the Current Status table and the chapter-forward link. No duplicate NOC transaction derivation present.
2. **C2 confirmed** — `height_sharded_kernel.md` `HEIGHT_SHARDED Config Construction` section (lines 84-100) ends at the closing code fence. Line 102 immediately opens the comparison table heading with no intervening prose re-derivation paragraph.
3. **C3 confirmed** — `height_sharded_kernel.md` line 125 reads: "Validation status for all three state configurations (DRAM baseline, L1 INTERLEAVED, and HEIGHT_SHARDED) is summarized in `sdpa_l1_conflict.md` §Current Status." The former multi-line "Validation Status" section with partial table has been fully replaced.

### CRUCIAL (must fix before chapter is done)

None. No new redundancy introduced by the Pass 1 edits, and no pre-existing cross-file or within-file duplication was overlooked in Pass 1.

### MINOR Suggestions (optional)

All five MINOR items from Pass 1 remain unaddressed and are still valid. Carried forward verbatim:

**M1 — `l1_state_design.md` lines 29-30: Anti-fragmentation rationale duplicated.**
Line 29 states "which avoids memory fragmentation"; line 110 states "Pre-allocated `_dram_state` buffers ensure zero DRAM allocation overhead during swaps." Only one instance is needed. Remove the parenthetical clause at line 29. Saving: ~0.5 lines.

**M2 — `sdpa_l1_conflict.md` line 3: Opening sentence restates `index.md` context.**
The file opens with a restatement of the rolling window mechanism already in `index.md` line 3 and covered fully in `l1_state_design.md`. Cut this sentence and open directly at "But the attention layer itself presents a problem." Saving: ~1 line.

**M3 — `l1_state_design.md` lines 65-66: Monkey-patching explanation restated across two sentences.**
"The `forward()` method... injects swap logic into the layer loop without modifying the parent `TTTransformer.forward()`." and "This is achieved by temporarily monkey-patching each GDN layer's `forward` method." The second sentence restates the mechanism implied by the first. Merge into one sentence. Saving: ~1 line.

**M4 — `height_sharded_kernel.md` lines 117-123: C++ comment block quote adds no quantitative fact.**
The 3-line quoted C++ comment (fenced block, lines 117-121) explains output tile layout mapping in words already captured by the prose at line 123. The block quote can be dropped, leaving only the prose sentence. Saving: ~5 lines.

**M5 — `l1_state_design.md` lines 100-101: Swap timing prose restates the preceding table.**
"The attention layer between each group runs after the old group's states have already been saved to DRAM by the swap that precedes the next group." The Swap Timing table directly above already encodes this relationship. Remove the sentence. Saving: ~1 line.

### Load-Bearing Evidence

**C1 (confirmed removed) — nothing was lost.** The deleted paragraph in `sdpa_l1_conflict.md` restated "16 NOC reads + 16 NOC writes per pair × 384 pairs = 12,288 per layer, ~590,000 total." This arithmetic is fully preserved at `l1_state_design.md` line 15. The Current Status table in `sdpa_l1_conflict.md` captures the practical outcome. Zero load-bearing information lost.

**C2 (confirmed removed) — nothing was lost.** The deleted prose paragraph re-derived `512 rows × 128 cols × 2 bytes = 128 KB per core` and `4 pairs per core` from values already present in the code block comments (`# 32 * 12 * 128 = 49152`, `# 512`, `# 96 total`). The key result "4 pairs, 128 KB" is stated in `sdpa_l1_conflict.md` line 18 ("each core holds 4 pairs worth of state: 4 × 32 KB = 128 KB per core"). Zero load-bearing information lost.

**C3 (confirmed removed) — nothing was lost.** The former "Validation Status" section in `height_sharded_kernel.md` contained a partial table that was a strict subset of the Current Status table in `sdpa_l1_conflict.md`. The forward-reference sentence at `height_sharded_kernel.md` line 125 directs the reader to the authoritative table. Zero load-bearing information lost.

### VERDICT

**Crucial updates: no**
