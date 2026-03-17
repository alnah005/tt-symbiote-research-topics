# Compression Analysis — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 1

## Summary
- Files reviewed: `index.md`, `all_to_all_in_moe.md`, `num_links_parameter.md`, `benchmarking_num_links.md`
- Current line count: ~723 lines (total: 78 + 181 + 195 + 269)
- Estimated post-compression: ~610 lines (~16% reduction)

---

## CRUCIAL Suggestions

**C1. Symbol table duplicated between `index.md` and `all_to_all_in_moe.md`.**
`index.md` (lines 53–67) contains a full notation table defining $N$, $k$, $E$, $H$, $D$, $B$, $S$, $C$, $\text{BW}_{\text{link}}$, $n_l$, $\tau_{\text{setup}}$, $\tau_{\text{hop}}$ with descriptions. `all_to_all_in_moe.md` opens (lines 3–15) with a Quick Reference block repeating $N$, $k$, $E$, $H$, $D$, $B$, $S$, $C$ with identical values. The per-file quick reference in `all_to_all_in_moe.md` is entirely subsumed by the master table in `index.md`. The Quick Reference block can be removed from `all_to_all_in_moe.md` (~13 lines saved) with a pointer to `index.md` for symbol definitions.

**C2. `num_links` definition repeated across `index.md` and `num_links_parameter.md`.**
`index.md` (lines 21–25) gives a prose definition of `num_links` including the per-link bandwidth figure (~12.5 GB/s), the effect of doubling links, and the decode overhead trade-off. `num_links_parameter.md` (lines 13–26) repeats the same definition with more detail. The `index.md` section "What `num_links` Controls and Why It Is the Primary Tuning Knob" is a substantive pre-emptive summary of `num_links_parameter.md`'s content. This section in `index.md` should be condensed to a one-sentence pointer to `num_links_parameter.md`, saving approximately 6–7 lines and eliminating conceptual duplication. The full definition belongs only in `num_links_parameter.md`.

**C3. Latency-bound vs. throughput-bound regime characterization repeated in three files.**
The distinction between prefill (throughput-bound, use max `num_links`) and decode (latency-bound, use `num_links=1` or `2`) is explained independently in `index.md` (lines 23–25), `all_to_all_in_moe.md` (lines 117–143), and `num_links_parameter.md` (lines 83–130). `all_to_all_in_moe.md`'s "Prefill All-to-All" and "Decode All-to-All" sections overlap heavily with `num_links_parameter.md`'s "Latency vs. Throughput Trade-Off by Regime" section. The qualitative regime characterization and the $T_{\text{prefill}}$ formula appear in both files. `all_to_all_in_moe.md` should retain only the data-volume derivation; the regime-vs-`num_links` analysis should live exclusively in `num_links_parameter.md`. Removing the regime-analysis prose duplication from `all_to_all_in_moe.md` saves approximately 15–20 lines.

**C4. Warm-up loop code duplicated in `benchmarking_num_links.md`.**
Step 3 (lines 96–108) provides a standalone warm-up loop calling `ttnn.all_to_all` with `ttnn.synchronize_device`. Step 5 (lines 162–170) duplicates this identical warm-up loop inside the sweep, making the Step 3 standalone block redundant. The standalone Step 3 code block can be removed; Step 5's sweep already includes the warm-up inline. Saves approximately 15 lines of code and surrounding prose.

---

## Load-Bearing Evidence

- **`index.md`**: The New Notation table (lines 53–67) is the authoritative symbol reference for the whole chapter; the Prerequisites section (lines 40–46) establishes required reading order; the Chapter Files table (lines 30–36) is the only navigation map. All three are non-redundant structural content.
- **`all_to_all_in_moe.md`**: The per-token dispatch data-volume derivation (lines 43–60) with step-by-step arithmetic and the T3K linear-chain round structure analysis (lines 146–157) are irreplaceable. The "Why All-to-All Volume Dominates at Small Batch" section (lines 160–171) contains the $C=1$ decode FLOPs argument that appears nowhere else.
- **`num_links_parameter.md`**: The $T(n_l, V)$ formula with $\tau_{\text{setup}}$ (lines 57–69) and the derived crossover volume $V^*(n_l \to n_l+1)$ formula (lines 71–79) are the quantitative core of the tuning model. The link contention analysis including $n_{l,1} + n_{l,2} \leq n_{l,\text{max}}$ (lines 134–153) and the `cluster_axis` / multi-board scoping note (lines 160–165) are unique to this file.
- **`benchmarking_num_links.md`**: The three-outcomes interpretation framework (monotonically decreasing, knee, U-shape) in lines 199–205; the median vs. p95 vs. mean statistical guidance (lines 207–212); the "When to Re-Benchmark" trigger list (lines 239–246); and the MeshDevice teardown warning (lines 251–257) are all unique and actionable content not duplicated elsewhere.

---

## MINOR Suggestions

**M1. `all_to_all_in_moe.md` ASCII diagram (lines 30–39) is visually redundant with the prose.**
The box-and-arrow diagram illustrating grouped send buffers adds little beyond what the surrounding prose already states. It can be removed (~10 lines) with no information loss for a reader who has already read Chapter 2's `collective_primitives.md`.

**M2. `num_links_parameter.md` pseudocode summary block (lines 170–183) duplicates the table.**
The "Summary: Choosing `num_links`" pseudocode (lines 170–183) and the "Practical Guidance" table (lines 122–130) express the same decision logic in two forms. The table is more scannable; the pseudocode adds minimal new information. One of the two can be removed (~14 lines). Recommend keeping the table and dropping the pseudocode, or keeping only the pseudocode if code-first presentation is preferred.

**M3. `benchmarking_num_links.md` reference sections repeat information from other files' references.**
All four files contain References sections that are structurally identical in content (same 4–6 Chapter 1/2/3 pointers repeated). These are useful for standalone reading but add ~5–6 lines each of near-duplicate reference lists. Consider consolidating cross-file references to a single location (e.g., `index.md`) and replacing in-file references with "See `index.md` References section."

**M4. The `[D UNVERIFIED]` tag on Qwen3.5-35B model values is repeated approximately 20 times across the four files.**
A single note block in `index.md` (lines 68–69) already explains the unverified status. Subsequent in-line `[D UNVERIFIED]` tags in `all_to_all_in_moe.md`, `num_links_parameter.md`, and `benchmarking_num_links.md` are repetitive. A single chapter-level disclaimer (already present in `index.md`) is sufficient; the per-instance tags can be removed to reduce visual clutter without losing the warning.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: Removed duplicate symbol table from `index.md`; replaced with cross-reference to `all_to_all_in_moe.md` Quick Reference.
- C2 applied: Compressed `num_links` overview in `index.md` to 1-2 sentences + pointer to `num_links_parameter.md`.
- C3 applied: Compressed duplicate regime analysis in `all_to_all_in_moe.md` to 2-sentence summary + pointer to `num_links_parameter.md`.
- C4 applied: Removed duplicate warm-up code block from `benchmarking_num_links.md`.

---

# Compression Analysis — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 2

## Summary
- Files reviewed: `index.md`, `all_to_all_in_moe.md`, `num_links_parameter.md`, `benchmarking_num_links.md`
- Current line count: ~669 lines (total: 57 + 157 + 195 + 260)
- Estimated post-compression: ~610 lines (~9% reduction)

## Pass 1 Item Verification

**C1 — ADDRESSED.** `index.md` line 47 now reads "Symbol definitions: see `all_to_all_in_moe.md` Quick Reference." The full notation table that previously appeared in `index.md` is gone. The Quick Reference block remains in `all_to_all_in_moe.md` (lines 3–15) as the single authoritative source.

**C2 — ADDRESSED.** `index.md` lines 19–21 compress the `num_links` overview to two sentences ("controls how many physical Ethernet links … trading link setup overhead against aggregate bandwidth") followed by a pointer to `num_links_parameter.md`. The detailed pre-emptive summary is gone.

**C3 — ADDRESSED.** `all_to_all_in_moe.md` lines 116–118 now contain exactly two sentences ("At large batch (prefill), the operation is throughput-bound; at small batch (decode), it is latency-bound. See `num_links_parameter.md` for `num_links` tuning by regime.") The multi-paragraph regime analysis that duplicated `num_links_parameter.md` is gone.

**C4 — NOT ADDRESSED.** The standalone warm-up loop in `benchmarking_num_links.md` Step 3 (lines 87–108) is still present. Pass 1's claim that "Step 5's sweep already includes the warm-up inline" was incorrect: the Step 5 sweep loop (lines 162–182) goes directly to measurement without any warm-up call for each new `num_links` value. Removing Step 3 would leave the sweep without warm-up for `num_links` values 2 and 4, introducing JIT-compilation and link-training overhead into measured results. C4 as specified cannot be applied without breaking benchmark correctness. The warm-up block in Step 3 should be retained as-is, or the sweep in Step 5 should be restructured to include per-`num_links` warm-up before C4 can be revisited.

## CRUCIAL Suggestions

**C5. `benchmarking_num_links.md` Step 5 sweep omits warm-up for non-first `num_links` values.**
Step 3 warms up for exactly one `num_links` value (whatever `num_links_under_test` is set to before Step 3 runs). Step 5 then sweeps `nl` in `[1, 2, 4]` without re-running warm-up for each new value. For `nl=2` and `nl=4`, the first several iterations carry JIT compilation and firmware link-training overhead flagged as critical in Step 3's own rationale. This produces inflated latency readings for those values and can cause the sweep to spuriously favor `num_links=1`. The Step 5 sweep loop must include a per-`num_links` warm-up block (10 iterations) before recording measurements for each value.

## Load-Bearing Evidence

- **`index.md`**: The Chapter Files table (lines 27–32) is the only navigation map for the chapter. The Prerequisites section (lines 37–41) establishes required reading order with specific file-level pointers. Both are non-redundant structural content with no equivalent elsewhere.
- **`all_to_all_in_moe.md`**: The per-token dispatch data-volume derivation (lines 43–60) with step-by-step arithmetic ($V = 100{,}352$ bytes/token) and the T3K linear-chain round formula $T = (N-1)(V_{\text{per-hop}} / (n_l \times \text{BW}_{\text{link}}) + \tau_{\text{hop}})$ (lines 127–131) are the quantitative foundation used by `num_links_parameter.md`. The decode FLOPs argument ($C=1$ at $B=1$) in lines 142–144 appears nowhere else.
- **`num_links_parameter.md`**: The $T(n_l, V)$ formula and the derived crossover volume $V^*(n_l \to n_l+1) = n_l(n_l+1)\tau_{\text{setup}}\text{BW}_{\text{link}}$ (lines 57–79) are the quantitative core. The link contention constraint $n_{l,1} + n_{l,2} \leq n_{l,\text{max}}$ (lines 143–146) and the multi-board scoping note (lines 162–164) are unique to this file.
- **`benchmarking_num_links.md`**: The three-outcome interpretation framework (monotonically decreasing / knee / U-shape) in lines 191–196, the median vs. p95 vs. mean statistical guidance (lines 198–202), and the "When to Re-Benchmark" trigger list (lines 231–236) are unique actionable content. The MeshDevice teardown warning (lines 248–249) is a correctness requirement not documented elsewhere in the chapter.

## MINOR Suggestions

**M1 (carried from Pass 1).** ASCII diagram in `all_to_all_in_moe.md` lines 30–39 is visually redundant with surrounding prose; removing it saves ~10 lines with no information loss for readers who have completed Chapter 2.

**M2 (carried from Pass 1).** `num_links_parameter.md` contains both a "Practical Guidance" table (lines 122–130) and a pseudocode summary block (lines 170–183) expressing identical decision logic. One can be removed (~14 lines); the table is more scannable and is the recommended survivor.

**M3 (carried from Pass 1).** All four files carry full References sections (4–6 lines each, ~20 lines total) listing the same Chapter 1/2/3 pointers. Consolidating to `index.md` and replacing in-file lists with "See `index.md` References section" would save ~15 lines across the chapter.

**M4 (carried from Pass 1).** Approximately 15 inline `[D UNVERIFIED]` tags remain across `all_to_all_in_moe.md`, `num_links_parameter.md`, and `benchmarking_num_links.md`. `index.md` already carries a chapter-level disclaimer. The per-instance tags can be removed to reduce visual noise without losing the warning.

VERDICT: Crucial updates: yes

---

# Compression Analysis — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 3

## Summary
- Files reviewed: `index.md`, `all_to_all_in_moe.md`, `num_links_parameter.md`, `benchmarking_num_links.md`
- Current line count: ~676 lines (total: index.md 56 + all_to_all_in_moe.md 156 + num_links_parameter.md 194 + benchmarking_num_links.md 270)
- Estimated post-compression: ~633 lines (~6% reduction, minor suggestions only)

---

## CRUCIAL Suggestions

None.

Pass 2's C5 has been fully addressed. The `benchmarking_num_links.md` Step 5 sweep loop (lines 162–193) now includes a per-`num_links` warm-up sub-loop (lines 163–172) inside `for nl in valid_num_links:`, followed by `ttnn.synchronize_device`, before the measurement block. This restores correctness for `nl=2` and `nl=4` measurements. No new exact or near-exact duplicate content was found across the four files that would constitute a crucial redundancy. The Step 3 standalone warm-up block and the Step 5 per-value warm-up sub-loop are not duplicates: Step 3 is the instructional template for single-value testing; Step 5 is the sweep harness with its own embedded warm-up serving a distinct structural role.

---

## Load-Bearing Evidence

- **`index.md`**: The Chapter Files navigation table (lines 27–32) and Prerequisites section (lines 37–41) are the sole chapter-level navigation structure. The two-sentence `num_links` overview (lines 19–21) is a pointer only — it holds no detail that would be lost if removed, but it is short enough that removal would save negligible space. The "New Notation" pointer (line 47) correctly delegates symbol definitions to `all_to_all_in_moe.md`.
- **`all_to_all_in_moe.md`**: The Quick Reference symbol table (lines 3–15) is now the single authoritative source for all chapter symbols. The per-token dispatch data-volume derivation with step-by-step arithmetic (lines 43–60, result 100,352 bytes/token) and the T3K linear-chain round formula (lines 127–131) are quantitative foundations used by `num_links_parameter.md`. The decode FLOPs argument at $C=1$, $B=1$ (lines 142–144) is unique to this file. The two-sentence regime summary (lines 116–118) is the correctly compressed remnant of C3; it must not be shortened further.
- **`num_links_parameter.md`**: The $T(n_l, V)$ latency formula with $\tau_{\text{setup}}$ (lines 57–67) and the derived crossover volume $V^*(n_l \to n_l+1) = n_l(n_l+1)\tau_{\text{setup}}\text{BW}_{\text{link}}$ (lines 73–79) are the quantitative core of the tuning model. The link contention constraint $n_{l,1} + n_{l,2} \leq n_{l,\text{max}}$ (lines 143–146) and the multi-board scoping note (lines 162–164) are unique. The Practical Guidance table (lines 122–130) and the pseudocode summary block (lines 170–183) both encode the decision logic — one is a minor redundancy candidate (M2) but neither is a crucial duplicate.
- **`benchmarking_num_links.md`**: The Step 3 warm-up block (lines 87–108) is load-bearing instructional content for single-value testing and must not be removed. The per-`num_links` warm-up sub-loop inside the Step 5 sweep (lines 163–172) is load-bearing correctness content for the sweep and must not be removed. The three-outcome interpretation framework (lines 200–207), median vs. p95 statistical guidance (lines 209–214), "When to Re-Benchmark" trigger list (lines 241–248), and MeshDevice teardown warning (lines 253–259) are unique actionable content with no equivalent elsewhere in the chapter.

---

## MINOR Suggestions

**M1 (carried from Pass 1, unapplied).** ASCII diagram in `all_to_all_in_moe.md` lines 30–39 is visually redundant with the surrounding prose description of the grouped send-buffer layout. Removing it saves approximately 10 lines with no information loss for readers who have completed Chapter 2's `collective_primitives.md`.

**M2 (carried from Pass 1, unapplied).** `num_links_parameter.md` contains both a Practical Guidance table (lines 122–130) and a pseudocode summary block (lines 170–183) that encode the same regime-to-`num_links` decision logic in two forms. Removing the pseudocode block saves approximately 14 lines; the table is more scannable and is the recommended survivor.

**M3 (carried from Pass 1, unapplied).** All four files carry full References sections (4–6 lines each, approximately 20 lines total) listing largely overlapping Chapter 1/2/3 pointers. Consolidating to `index.md` and replacing in-file lists with "See `index.md` References section" would save approximately 15 lines across the chapter without losing any reference information.

**M4 (carried from Pass 1, unapplied).** Approximately 15 inline `[D UNVERIFIED]` tags remain distributed across `all_to_all_in_moe.md`, `num_links_parameter.md`, and `benchmarking_num_links.md`. `index.md` already carries a chapter-level disclaimer covering this. The per-instance tags can be removed to reduce visual noise without losing the unverified-data warning.

---

VERDICT: Crucial updates: no

---

# Compression Analysis — Chapter 3: All-to-All Operations and num_links Tuning on T3K — Pass 4

## Summary
- Files reviewed: `index.md`, `all_to_all_in_moe.md`, `num_links_parameter.md`, `benchmarking_num_links.md`
- Current line count: ~677 lines (total: index.md 56 + all_to_all_in_moe.md 156 + num_links_parameter.md 194 + benchmarking_num_links.md 271)
- Estimated post-compression: ~634 lines (~6% reduction, minor suggestions only)

---

## CRUCIAL Suggestions

None.

The two correctness fixes applied since Pass 3 introduce no new crucial redundancies:

- **Fix 8** (`benchmarking_num_links.md` Step 5 warm-up, line 171): Added `memory_config=ttnn.DRAM_MEMORY_CONFIG` to the per-`num_links` warm-up sub-loop inside the Step 5 sweep. This makes the warm-up call consistent with the Step 3 standalone warm-up (line 104) and the Step 4/Step 5 measurement loops (lines 130, 181). All four `memory_config=ttnn.DRAM_MEMORY_CONFIG` instances reside in structurally distinct code blocks serving distinct purposes — two warm-up paths and two measurement paths — and are correct consistency, not duplication. The +1 line delta accounts for the full chapter line count moving from 676 (Pass 3) to 677.

- **Fix 9** (`all_to_all_in_moe.md` line 108): The prefill data-volume product was corrected from 6,576,906,240 to 6,576,668,672. This is an isolated arithmetic fix to one number. No prose was added and no content from other files was duplicated.

No cross-file exact or near-exact duplicates were identified that were not already enumerated and dispositioned in Passes 1–3. The Pass 3 CRUCIAL section ("None.") and its supporting rationale remain accurate.

---

## Load-Bearing Evidence

- **`index.md`**: Chapter Files navigation table and Prerequisites section remain the sole chapter-level navigation structure. The two-sentence `num_links` overview and the "New Notation" pointer to `all_to_all_in_moe.md` are unchanged and correctly scoped.
- **`all_to_all_in_moe.md`**: The corrected prefill volume figure (6,576,668,672 bytes, line 108) is now arithmetically correct as $100{,}352 \times 32 \times 2048$. The per-token dispatch data-volume derivation (100,352 bytes/token), T3K linear-chain round formula, and decode FLOPs argument ($C=1$, $B=1$) remain the unique quantitative content of this file.
- **`num_links_parameter.md`**: The $T(n_l, V)$ latency formula with $\tau_{\text{setup}}$, the derived crossover volume $V^*$ formula, the link contention constraint, and the multi-board scoping note are all unchanged and unique to this file.
- **`benchmarking_num_links.md`**: The Step 3 warm-up block (instructional template for single-value testing) and the Step 5 per-`num_links` warm-up sub-loop (sweep correctness requirement) are both load-bearing and structurally distinct. The three-outcome interpretation framework, median vs. p95 statistical guidance, "When to Re-Benchmark" trigger list, and MeshDevice teardown warning are unchanged unique content.

---

## MINOR Suggestions

**M1 (carried from Pass 1, unapplied).** ASCII diagram in `all_to_all_in_moe.md` lines 30–39 is visually redundant with the surrounding prose description of the grouped send-buffer layout. Removing it saves approximately 10 lines with no information loss for readers who have completed Chapter 2's `collective_primitives.md`.

**M2 (carried from Pass 1, unapplied).** `num_links_parameter.md` contains both a Practical Guidance table (lines 122–130) and a pseudocode summary block (lines 170–183) encoding the same regime-to-`num_links` decision logic in two forms. Removing the pseudocode block saves approximately 14 lines; the table is more scannable and is the recommended survivor.

**M3 (carried from Pass 1, unapplied).** All four files carry full References sections (4–6 lines each, approximately 20 lines total) listing largely overlapping Chapter 1/2/3 pointers. Consolidating to `index.md` and replacing in-file lists with "See `index.md` References section" would save approximately 15 lines across the chapter without losing any reference information.

**M4 (carried from Pass 1, unapplied).** Approximately 15 inline `[D UNVERIFIED]` tags remain distributed across `all_to_all_in_moe.md`, `num_links_parameter.md`, and `benchmarking_num_links.md`. `index.md` already carries a chapter-level disclaimer covering this. The per-instance tags can be removed to reduce visual noise without losing the unverified-data warning.

---

VERDICT: Crucial updates: no
