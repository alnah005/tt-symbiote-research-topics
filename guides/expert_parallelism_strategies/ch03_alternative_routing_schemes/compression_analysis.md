# Compression Analysis — Chapter 3: Alternative Expert Routing Schemes — Pass 1

## Summary
- Files reviewed: `index.md`, `expert_sharding.md`, `pipeline_expert_parallelism.md`, `hierarchical_routing.md`, `scheme_comparison_matrix.md`
- Current line count: ~798 lines (total: 100 + 173 + 171 + 166 + 188)
- Estimated post-compression: ~700 lines (~12% reduction)

---

## CRUCIAL Suggestions

**C1 — $V_{a2a}$ baseline formula repeated verbatim across three files.**
The all-to-all dispatch volume formula $V_{a2a} = \frac{N-1}{N} \times B \times k \times H \times 2$ with the Qwen3.5-35B numerical instantiation ($100{,}352 \times B$ bytes) appears in full, with identical arithmetic, in:
- `index.md` lines 76–84 ("Relationship to Chapter 2 Baseline" section)
- `expert_sharding.md` lines 62–70 ("Communication Volume: All-to-All (Baseline Recap)" section)
- `hierarchical_routing.md` lines 57–59 (formula block under "Flat Routing (Baseline)")
- `scheme_comparison_matrix.md` lines 29–30 (Volume Arithmetic subsection)

The `index.md` instance and `expert_sharding.md` instance are the most duplicative pair: both give the full derivation with the same $B=32$ arithmetic check ($3{,}211{,}264$ bytes $\approx 3.06$ MiB). The `expert_sharding.md` section is labeled "Baseline Recap" and signals its own redundancy. Cutting `expert_sharding.md` lines 62–70 and replacing with a one-line back-reference to `index.md` would save ~9 lines with zero information loss — the formula was just derived in full in `index.md` and the reader is told to read `index.md` first.

**C2 — Pipeline "not applicable to Qwen3.5-35B" conclusion restated in full in two files.**
The conclusion that pipeline expert parallelism does not apply to Qwen3.5-35B because it uses parallel top-8 execution appears in:
- `pipeline_expert_parallelism.md` lines 42–46 and again at lines 150–158 ("When Pipeline Expert Parallelism Wins" section, last paragraph)
- `scheme_comparison_matrix.md` lines 151–156 ("Not Applicable" subsection)

The `pipeline_expert_parallelism.md` lines 42–46 (Precondition section) establish the limitation and lines 150–158 then repeat it almost word-for-word as a concluding sentence. The full re-statement in `scheme_comparison_matrix.md` lines 153–156 is warranted as a synthesis, but the internal repetition within `pipeline_expert_parallelism.md` (two separate locations, ~60 lines apart, saying the same thing) is redundant. The second occurrence at lines 150–158 can be compressed to one sentence pointing back to the Precondition section, saving ~4–5 lines.

**C3 — Hierarchical routing "not applicable to Qwen3.5-35B" conclusion triply stated.**
The determination that hierarchical routing requires retraining and is therefore not applicable appears in:
- `hierarchical_routing.md` lines 105–107 ("Inference-Time Load Skew" section, last paragraph)
- `hierarchical_routing.md` lines 110–119 ("Training Dependency" section — full paragraph)
- `hierarchical_routing.md` line 150 ("When Hierarchical Routing Wins" section, last sentence)
- `scheme_comparison_matrix.md` lines 151–156

All four are carrying the same disqualifying statement. Within `hierarchical_routing.md` itself, lines 105–107 state it, lines 110–119 re-explain it in greater detail, and line 150 repeats the verdict. The lines 105–107 instance is redundant with the Training Dependency section that follows immediately — it can be cut or collapsed into a forward-reference to that section, saving ~3 lines with no information loss.

---

## Load-Bearing Evidence

- **`index.md`:** The notation table (lines 29–43) is the canonical definition for $V_\text{gather}$, $G$, $k_c$, $k_f$, $P$, $M$, $\eta$ — these symbols are used throughout all four topic files and cannot be removed. The Cross-Chapter Dependency Note (lines 60–68) and Reading Order (lines 46–55) provide structural scaffolding essential for navigating the chapter correctly. The $V_{a2a}$ baseline derivation with arithmetic check (lines 76–84) establishes the comparison anchor used in every subsequent file.

- **`expert_sharding.md`:** The $k=N$ coincidence derivation (lines 75–97) — showing algebraically *why* $V_\text{gather} = V_{a2a}$ when $k = N$ and providing the ratio formula for $k < N$ — is irreplaceable. The compute waste analysis distinguishing router overhead under all-gather vs. all-to-all (lines 100–131) and the memory implications section (lines 134–144) are not summarized in any other file and directly inform the comparison matrix decision rule.

- **`pipeline_expert_parallelism.md`:** The bubble analysis with efficiency formula $\eta = \mu / (\mu + P - 1)$ (lines 66–83) and its numeric instantiations ($P=8$: $\eta = 12.5\%$, 53.3%, 90.1%) are the quantitative backbone for the comparison matrix's pipeline row. The per-stage communication cost derivation with $V_\text{stage} = M \times H \times 2$ and the T3K topology fit analysis (lines 137–142) are load-bearing for Chapter 6 cross-references.

- **`hierarchical_routing.md`:** The communication reduction formula derivation (lines 54–87) — including the non-obvious result that expected hierarchical volume equals flat volume under uniform group selection, and the identification that the real reduction requires $p_\text{local} > 1/N$ — is the core correctness-bearing content. The coarse router tension (load-balance vs. communication efficiency) analysis (lines 91–107) and the expected layer-time formula $\mathbb{E}[T_\text{layer}] = T_\text{FFN} + (1-p_\text{local}) \times T_\text{a2a}$ (lines 130–138) are irreplaceable.

- **`scheme_comparison_matrix.md`:** The six-dimension comparison table (lines 17–25), the ASCII decision flowchart with numbered summary rules (lines 59–115), and the crossover batch size formula for all-gather vs. all-to-all (lines 141–149) are the synthesis deliverables of Chapter 3 and are referenced by Chapters 4, 6, and 8. The "Key Results Carried Forward" table (lines 162–171) is the only place this downstream dependency map is consolidated.

---

## MINOR Suggestions

**M1 — `expert_sharding.md` lines 33–35:** The parenthetical clarifying that all-gather + reduce-scatter together are "equivalent to all-reduce in terms of buffer requirements, but not in terms of data layout" is a useful precision, but the immediately following sentence re-explains the point. The two sentences can be merged into one without losing the distinction.

**M2 — `pipeline_expert_parallelism.md` lines 50–58 ("Pipeline Structure for Qwen3.5-35B (Hypothetical)"):** This section establishes $P = k = N = 8$ for two different mappings and then concludes "Both mappings yield $P = 8$ stages for this configuration." The section adds no new analysis beyond confirming the value of $P$ used in subsequent formulas. It could be collapsed to a one-sentence parenthetical in the Bubble Analysis section.

**M3 — `hierarchical_routing.md` lines 39–49 (Configuration Examples table):** The table includes a prose paragraph immediately below it (lines 47–49) that re-explains the expected cross-device sends formula already presented in the table's rightmost column header. The prose can be trimmed to just the key sentence about the $k_c=1$ best case.

**M4 — Arithmetic check duplication for $3{,}211{,}264$ bytes ($\approx 3.06$ MiB):** This specific computed value and the $\approx 3.06$ MiB annotation appear in `index.md` line 82, `expert_sharding.md` line 55, and `scheme_comparison_matrix.md` line 29–30. Given it is the same number, the `scheme_comparison_matrix.md` arithmetic check (lines 29–30) adds no new verification beyond what was already checked twice. A single reference sentence would suffice there.

**M5 — `scheme_comparison_matrix.md` lines 36–51 ("Crossover: All-to-All vs. All-Gather"):** This section largely recaps what `expert_sharding.md` already concluded in its "Decisive Difference" and "When Expert Sharding Wins" sections, and then defers the formal threshold to Chapter 6. The decision rule formula presented (line 47) does not add precision beyond the prose in `expert_sharding.md`. This crossover section could be condensed to 3–4 lines that cite `expert_sharding.md` and point to Chapter 6.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: Removed "Baseline Recap" section from `expert_sharding.md`; replaced with cross-reference to `index.md`. New analytical content in the section retained.
- C2 applied: Removed duplicate pipeline-not-applicable conclusion from `pipeline_expert_parallelism.md` end section; replaced with back-reference to Precondition section.
- C3 applied: Removed premature retraining verdict at `hierarchical_routing.md` lines ~105-107; authoritative Training Dependency section preserved.

---

# Compression Analysis — Chapter 3: Alternative Expert Routing Schemes — Pass 2

## Summary
- Files reviewed: `index.md`, `expert_sharding.md`, `pipeline_expert_parallelism.md`, `hierarchical_routing.md`, `scheme_comparison_matrix.md`
- Current line count: ~783 lines (total: 99 + 164 + 170 + 163 + 187)
- Estimated post-compression: ~760 lines (~3% reduction)

## Pass 1 Item Verification

**C1 — ADDRESSED.** `expert_sharding.md` line 61 now reads: "For the all-to-all baseline volume formula and Qwen3.5-35B numerical instantiation, see `index.md` Section [notation/baseline section]." The full "Baseline Recap" section with its redundant derivation and arithmetic check is gone; only the single cross-reference line remains.

**C2 — ADDRESSED.** `pipeline_expert_parallelism.md` "When Pipeline Expert Parallelism Wins" section ends (line 157) with: "As established in the Precondition section, Qwen3.5-35B's parallel top-$k$ routing makes this scheme inapplicable without retraining." This is one sentence pointing back to the Precondition section, not a full re-statement of the disqualification.

**C3 — ADDRESSED.** `hierarchical_routing.md` "Inference-Time Load Skew" section (lines 103-104) ends with a general empirical observation about MoE routers and inference-time load skew. The premature disqualification verdict that previously appeared there has been removed; the authoritative "Training Dependency" section (lines 110-119) remains as the sole location of the "not applicable" determination for Qwen3.5-35B.

## CRUCIAL Suggestions

None.

## Load-Bearing Evidence

- **`index.md`:** Notation table (lines 29-43) is the canonical symbol definition for $V_\text{gather}$, $G$, $k_c$, $k_f$, $P$, $M$, $\eta$ — referenced in all four topic files and undeletable. The $V_{a2a}$ baseline derivation with arithmetic check (lines 76-84) is the single authoritative instantiation of the all-to-all formula used as the comparison anchor throughout the chapter; all other files now defer to this section.

- **`expert_sharding.md`:** The algebraic proof that $V_\text{gather} = V_{a2a}$ when $k = N$ and the $k/N$ ratio formula for $k < N$ (lines 68-88) are the core correctness-bearing content for the sharding vs. all-to-all comparison. The compute waste analysis distinguishing $N\times$ router overhead under all-gather (lines 92-122) and the memory implications section (lines 128-148) are not summarized in any other file and directly ground the comparison matrix decision rule.

- **`pipeline_expert_parallelism.md`:** Bubble efficiency formula $\eta = \mu / (\mu + P - 1)$ with three numeric instantiations at $P=8$ (lines 66-83) is the quantitative backbone for the comparison matrix pipeline row. The per-stage volume formula $V_\text{stage} = M \times H \times 2$ and T3K linear topology fit analysis (lines 88-142) are load-bearing for Chapter 6 references.

- **`hierarchical_routing.md`:** The communication reduction derivation showing that expected hierarchical volume equals flat all-to-all under uniform group selection (lines 54-87) — including the non-obvious finding that real reduction requires $p_\text{local} > 1/N$ — is irreplaceable. The expected layer-time formula $\mathbb{E}[T_\text{layer}] = T_\text{FFN} + (1-p_\text{local}) \times T_\text{a2a}$ (lines 130-138) is the only quantitative expression of hierarchical routing's latency benefit.

- **`scheme_comparison_matrix.md`:** Six-dimension comparison table (lines 17-25), ASCII decision flowchart with five numbered summary rules (lines 59-115), and crossover batch-size formula for all-gather vs. all-to-all (lines 141-149) are the synthesis deliverables cited by Chapters 4, 6, and 8. The "Key Results Carried Forward" table (lines 162-171) is the only consolidated downstream dependency map in the chapter.

## MINOR Suggestions

**M1 (carried from Pass 1) — `expert_sharding.md` lines 33-35:** The parenthetical comparing all-gather + reduce-scatter to all-reduce ("equivalent in buffer requirements, but not in data layout") and the following sentence re-explain the same point. Merge into one sentence; saves ~1 line.

**M2 (carried from Pass 1) — `pipeline_expert_parallelism.md` lines 50-58 ("Pipeline Structure for Qwen3.5-35B (Hypothetical)"):** The section presents two mappings that both yield $P = 8$ and concludes "Both mappings yield $P = 8$ stages for this configuration." The entire section contributes only the value $P = 8$, which is self-evident from $k = N = 8$. Collapse to a one-sentence parenthetical in the Bubble Analysis section; saves ~7 lines.

**M3 (carried from Pass 1) — `hierarchical_routing.md` line 47:** The prose paragraph after the Configuration Examples table re-explains the expected cross-device sends formula already encoded in the table's rightmost column. Trim to just the key sentence about the $k_c = 1$ best case; saves ~1-2 lines.

**M4 (carried from Pass 1) — Arithmetic check for $3{,}211{,}264$ bytes:** The full arithmetic check computing $3{,}211{,}264$ bytes $\approx 3.06$ MiB appears in `index.md` line 82 (authoritative), `expert_sharding.md` line 55 (for $V_\text{gather}$, serving a distinct purpose — verifying the new all-gather formula equals the baseline), and `scheme_comparison_matrix.md` line 29 (full re-derivation including a self-correction mid-step). The `scheme_comparison_matrix.md` instance (line 29) with its in-line self-correction ("$7 \times 4{,}096 = 28{,}672$; no —") adds no new verification and can be replaced by "Confirmed: equals the all-to-all figure from `index.md` Section 'Relationship to Chapter 2 Baseline'."; saves ~1-2 lines.

**M5 (carried from Pass 1) — `scheme_comparison_matrix.md` lines 34-51 ("Crossover: All-to-All vs. All-Gather"):** This section recaps `expert_sharding.md`'s "Decisive Difference" conclusion and then defers the formal threshold to Chapter 6. The decision rule formula at line 47 repeats content already present in `expert_sharding.md` without adding precision. Condense to 3-4 lines citing `expert_sharding.md` for analysis and Chapter 6 for the quantitative threshold; saves ~10-12 lines.

VERDICT: Crucial updates: no
