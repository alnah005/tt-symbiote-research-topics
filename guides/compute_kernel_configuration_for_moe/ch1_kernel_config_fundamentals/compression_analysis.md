## Agent A Change Log — B Feedback Pass 1
- fp32_dest_acc_en.md: Removed incorrect "twice the L1 space" claim; clarified fp32 precision is in FPU dest register, not L1 buffer
- wormhole_compute_kernel_config_api.md + index.md: Added note that math_approx_mode=True is inert for pure matmuls in HIFI2 config
- fp32_dest_acc_en.md: Added K_t definition (K_t = K/32, tile width = 32 elements) to make table self-contained

---

# Compression Analysis: Chapter 1 — Compute Kernel Config Fundamentals — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~570 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~25%

## CRUCIAL Suggestions

### [index.md] ~lines 36–42 (Field Summary Table)
**Issue:** The four-row field summary table in `index.md` duplicates the "Primary Fields" section of `wormhole_compute_kernel_config_api.md` (lines 76–138). Every field's type, default, and one-line effect is restated verbatim in both places. The table in `index.md` is the shorter form; the API file is the authoritative longer form.
**Suggestion:** Replace the four-row table in `index.md` with a single sentence pointing readers to `wormhole_compute_kernel_config_api.md` for field semantics, keeping only the tip callout about `packer_l1_acc` (which is a genuine emphasis worth preserving at the chapter level).

### [index.md] ~lines 51–56 (Canonical Production Configs table)
**Issue:** The two-row canonical configs table in `index.md` restates the config values that are already spelled out in full in `wormhole_compute_kernel_config_api.md` lines 182–219 (with code, inline comments, and rationale). The `index.md` table adds nothing beyond a compressed view that is superseded by the detail file.
**Suggestion:** Remove the table from `index.md`. The Chapter Structure table (lines 62–68) already tells readers that `wormhole_compute_kernel_config_api.md` contains "The two canonical production configs from DeepSeek-V3." The two-row table in `index.md` is redundant navigation overhead.

### [wormhole_compute_kernel_config_api.md] ~lines 222–249 ("Putting It Together: Two-Config Pattern")
**Issue:** This section re-instantiates both `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` in full Python code blocks that are character-for-character identical to the code blocks 10–30 lines earlier in the same file (lines 190–215). The only addition is the three-line expert forward-pass usage snippet at the bottom.
**Suggestion:** Delete the two full config instantiations from this section (lines 230–242); keep only the expert forward-pass usage snippet (lines 244–248) with a one-line comment like `# Using the configs defined above`. The repeated instantiations are pure duplication within the same file.

### [math_fidelity_overview.md] ~lines 135–144 ("Relationship to fp32_dest_acc_en")
**Issue:** This section restates the independence of `math_fidelity` and `fp32_dest_acc_en` and the down-projection HiFi2+fp32 rationale. The same two-column "source of error / controlled by" table and the identical explanation ("For the down projection, DeepSeek-V3 enables both improvements…") appear in `fp32_dest_acc_en.md` lines 107–144. The section in `math_fidelity_overview.md` adds no new information.
**Suggestion:** Reduce to a two-sentence cross-reference: "Math fidelity controls per-multiply mantissa truncation; `fp32_dest_acc_en` controls accumulation-register rounding. See `fp32_dest_acc_en.md` for the interaction analysis and the down-projection case." Delete the duplicated table and rationale paragraph.

## MINOR Suggestions

### [wormhole_compute_kernel_config_api.md] ~lines 62–71 (third code block, `program_config` orthogonality)
**Issue:** The third code block in the "Construction and Usage" section exists solely to show that `program_config` and `compute_kernel_config` can coexist on the same call. The comment "controls core mapping and tiling" / "controls FPU precision and packer behavior" restates what is already clear from the parameter names and is covered in the Overview paragraph just above.
**Suggestion:** Delete the third code block (lines 63–71). The two-sentence prose introducing it (line 61) can be trimmed to one sentence: "The `compute_kernel_config` argument is orthogonal to `program_config`; both can be passed to the same `ttnn.matmul` call."

### [math_fidelity_overview.md] ~lines 39–43 (enum values code block)
**Issue:** The four-line enum listing (`LoFi`, `HiFi2`, `HiFi3`, `HiFi4` with integer-code comments) is identical to the listing in `wormhole_compute_kernel_config_api.md` lines 85–90. The API file already shows these values; listing them again before the same table content is minor duplication.
**Suggestion:** Remove the code block; the table immediately below it (lines 47–53) carries the same information in structured form and is the more useful representation.

### [math_fidelity_overview.md] ~lines 101–113 ("Why the Default is LoFi" — reasoning points 1–3)
**Issue:** Reasoning points 1–3 are verbose elaborations. Point 1 ("Most inference workloads are bandwidth-bound…") repeats the bandwidth-bound framing already established in `index.md` Overview and in `wormhole_compute_kernel_config_api.md` default-behavior section. Point 2 ("bfloat16 already has limited precision") is a genericism that adds no actionable information. Point 3 ("LoFi is the safe failure mode") duplicates the warning in `wormhole_compute_kernel_config_api.md` lines 178.
**Suggestion:** Collapse points 1–3 into one or two sentences: "The default is LoFi because inference workloads are typically bandwidth-bound at decode batch sizes, and bfloat16's 7-bit mantissa baseline already limits precision — LoFi degradation is detectable via PCC testing before deployment."

### [fp32_dest_acc_en.md] ~lines 6–14 (Overview bullet list)
**Issue:** The "This document explains:" bullet list (six bullets) is a table-of-contents for a 174-line file. Every bullet restates a section heading that already exists in the document. This pattern appears in every file's Overview section and contributes no navigational value beyond the headings themselves.
**Suggestion:** Delete the bullet list from the `fp32_dest_acc_en.md` Overview (and apply the same trim to the identical pattern in `wormhole_compute_kernel_config_api.md` lines 7–13 and `math_fidelity_overview.md` lines 8–14). Keep the single descriptive sentence that precedes the bullets.

### [wormhole_compute_kernel_config_api.md + math_fidelity_overview.md + fp32_dest_acc_en.md] — "Next Steps" sections
**Issue:** Each file ends with a "Next Steps" section that is a single sentence pointing to the next file in reading order. These are pure navigation boilerplate and add no content.
**Suggestion:** Delete the "Next Steps" sections from all three sub-files. The `index.md` Chapter Structure table already maps reading order. If navigation is desired, a single "Read files in this order: …" line in `index.md` is sufficient.

## Load-Bearing Evidence
- `fp32_dest_acc_en.md` line ~93: `"> **Note:** K_t = K/32 (tile width = 32 elements); e.g., K=7168/32=224 tiles, K=2048/32=64 tiles."` — load-bearing because it makes the summary table self-contained; readers need this definition to interpret the K_t column without context from the body text.
- `fp32_dest_acc_en.md` line ~101: `"> **Note:** Even though down has a shallower K-loop (K_t=64 vs K_t=224), it uses fp32_dest_acc_en=True because its output is more sensitivity-relevant."` — load-bearing because it explicitly resolves the apparent paradox (shorter K-loop gets higher precision); this is a genuine insight that cannot be cut.
- `wormhole_compute_kernel_config_api.md` line ~176: `"The critical difference from the production configs is packer_l1_acc=False. For decode-mode MoE expert matmuls where the output is small (M=1 to 32) and K is large (2048 to 7168), this means every K-loop iteration pays a DRAM round-trip…"` — load-bearing because it quantifies the exact performance cost of the default config omission; this is the motivating argument for the whole chapter.
- `math_fidelity_overview.md` line ~64: `"For a matmul that is bandwidth-bound (the typical case for MoE decode with M = batch size = 1 to 32), the compute savings from LoFi may be partially masked by memory access latency. Even so, reducing compute time per tile frees the pipeline sooner, reducing stall time and contributing to overall throughput."` — load-bearing because it gives the nuanced answer for why LoFi helps even on bandwidth-bound ops; cutting this loses a key conceptual distinction.
- `fp32_dest_acc_en.md` line ~143: `"The L1 accumulation buffer itself holds bfloat16, not float32. The fp32_dest_acc_en field affects only the on-chip FPU destination register (not L1), so there is no 'twice the L1 space' overhead from enabling it."` — load-bearing because it corrects a natural misconception (the Agent A changelog confirms this was an actual error that needed fixing); removing this risks re-introducing confusion.

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1
- index.md: Replaced duplicate Field Summary Table with cross-reference to wormhole_compute_kernel_config_api.md
- index.md: Removed duplicate Canonical Production Configs table
- wormhole_compute_kernel_config_api.md: Removed duplicate config instantiations from "Putting It Together"; kept usage snippet only
- math_fidelity_overview.md: Collapsed "Relationship to fp32_dest_acc_en" to 2-sentence cross-reference

---

# Compression Analysis: Chapter 1 — Compute Kernel Config Fundamentals — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~626 lines (index.md ~52, wormhole_compute_kernel_config_api.md ~239, math_fidelity_overview.md ~161, fp32_dest_acc_en.md ~174)
- Estimated post-compression line count: ~565 lines
- Estimated reduction: ~10%

## CRUCIAL Suggestions

None — all Pass 1 CRUCIAL items resolved.

### Verification Detail

**CRUCIAL 1 — index.md Field Summary Table** (was lines ~36–42): RESOLVED. The section now reads `## Field Summary Table` followed by a single sentence: "See `wormhole_compute_kernel_config_api.md` for the complete field reference." No four-row duplicate table is present (verified: index.md lines 32–34).

**CRUCIAL 2 — index.md Canonical Production Configs table** (was lines ~51–56): RESOLVED. The file ends at line 52 with a "Next Steps" pointer. No two-row canonical configs table is present anywhere in index.md.

**CRUCIAL 3 — wormhole_compute_kernel_config_api.md "Putting It Together" duplicate instantiations** (was lines ~230–242): RESOLVED. The "Putting It Together: Two-Config Pattern" section (lines 222–233) now contains only the three-line expert forward-pass usage snippet. The full `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` instantiation blocks are not repeated here.

**CRUCIAL 4 — math_fidelity_overview.md "Relationship to fp32_dest_acc_en" section** (was lines ~135–144): RESOLVED. The section now contains exactly two sentences as a cross-reference: "For how `fp32_dest_acc_en` interacts with fidelity choice, see `fp32_dest_acc_en.md`. Down-projection typically uses HiFi2 + `fp32_dest_acc_en=True` because residual stream accumulation is sensitive to rounding." The duplicated table and rationale paragraph are gone (verified: math_fidelity_overview.md lines 135–138).

## MINOR Suggestions

All five MINOR items from Pass 1 remain unresolved. They are carried forward unchanged.

### [CARRIED] wormhole_compute_kernel_config_api.md — third code block, `program_config` orthogonality (lines 62–71)
**Issue:** The third code block in the "Construction and Usage" section exists solely to illustrate that `program_config` and `compute_kernel_config` can coexist on the same `ttnn.matmul` call. The inline comments ("controls core mapping and tiling" / "controls FPU precision and packer behavior") restate what the prose line above and the parameter names already convey. No new information is added.
**Suggestion:** Delete the third code block (lines 63–71). Trim the prose at line 61 to: "The `compute_kernel_config` argument is orthogonal to `program_config`; both can be passed to the same `ttnn.matmul` call."

### [CARRIED] math_fidelity_overview.md — enum values code block (lines 36–43)
**Issue:** The four-line enum listing (`LoFi`, `HiFi2`, `HiFi3`, `HiFi4` with integer-code comments) is identical to the listing in `wormhole_compute_kernel_config_api.md` lines 85–90. The structured table immediately following (lines 47–53) conveys the same information plus throughput and use-case columns.
**Suggestion:** Remove the code block; the table carries all the same content in a more useful form.

### [CARRIED] math_fidelity_overview.md — "Why the Default is LoFi" verbose reasoning (lines 103–113)
**Issue:** Reasoning points 1–3 are verbose elaborations that repeat context already established elsewhere. Point 1 (bandwidth-bound framing) repeats `index.md` Overview and the `wormhole_compute_kernel_config_api.md` default-behavior section. Point 2 (bfloat16 limited precision) is a genericism with no actionable content. Point 3 (LoFi as safe failure mode) duplicates the warning at `wormhole_compute_kernel_config_api.md` line 178.
**Suggestion:** Collapse points 1–3 to one or two sentences: "The default is LoFi because inference workloads are typically bandwidth-bound at decode batch sizes, and bfloat16's 7-bit mantissa baseline already limits precision — LoFi degradation is detectable via PCC testing before deployment."

### [CARRIED] fp32_dest_acc_en.md — Overview bullet list (lines 6–14)
**Issue:** The six-bullet "This document explains:" list is a table-of-contents that restates every section heading already present in the document. The same pattern appears in `wormhole_compute_kernel_config_api.md` lines 7–13 and `math_fidelity_overview.md` lines 8–14. None of these bullet lists add navigational value beyond the headings.
**Suggestion:** Delete the bullet list from `fp32_dest_acc_en.md` Overview (and apply the same trim to the matching patterns in `wormhole_compute_kernel_config_api.md` and `math_fidelity_overview.md`). Retain the single descriptive sentence that precedes each list.

### [CARRIED] wormhole_compute_kernel_config_api.md, math_fidelity_overview.md, fp32_dest_acc_en.md — "Next Steps" sections
**Issue:** Each sub-file ends with a "Next Steps" section that is a single sentence (or short paragraph) pointing to the next file in reading order. This is pure navigation boilerplate; the Chapter Structure table in `index.md` already maps reading order.
**Suggestion:** Delete the "Next Steps" sections from all three sub-files. If in-file navigation is desired, a single "Read files in this order: …" line in `index.md` is sufficient.

### [NEW] wormhole_compute_kernel_config_api.md — "Putting It Together" comment gap (line 228)
**Issue:** After the Pass 1 edit removed the repeated config instantiations from this section, the usage snippet now refers to `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` without any inline comment indicating these were defined above. A reader skimming to this section without reading the "Canonical Production Configs" section above it may not immediately know where these names come from. The Pass 1 suggestion noted to add "a one-line comment like `# Using the configs defined above`" — this comment is absent.
**Suggestion:** Add a one-line comment above the usage snippet: `# COMPUTE_KERNEL_CONFIG_LOFI and COMPUTE_KERNEL_CONFIG_HIFI2 are defined in the "Canonical Production Configs" section above`. This is a one-line fix.

## Load-Bearing Evidence
- `fp32_dest_acc_en.md` line 93: `"> **Note:** K_t = K/32 (tile width = 32 elements); e.g., K=7168/32=224 tiles, K=2048/32=64 tiles."` — load-bearing; makes the summary table self-contained without requiring readers to hold the definition from body text.
- `fp32_dest_acc_en.md` line 101: `"> **Note:** Even though down has a shallower K-loop (K_t=64 vs K_t=224), it uses fp32_dest_acc_en=True because its output is more sensitivity-relevant."` — load-bearing; explicitly resolves the apparent paradox between K depth and precision choice. Cannot be cut.
- `wormhole_compute_kernel_config_api.md` line 176: `"The critical difference from the production configs is packer_l1_acc=False. For decode-mode MoE expert matmuls where the output is small (M=1 to 32) and K is large (2048 to 7168), this means every K-loop iteration pays a DRAM round-trip…"` — load-bearing; quantifies the exact performance cost of the default config omission; is the primary motivating argument for the chapter.
- `math_fidelity_overview.md` lines 62–64: the nuanced explanation of why LoFi helps even on bandwidth-bound ops ("reducing compute time per tile frees the pipeline sooner, reducing stall time") — load-bearing; cutting this loses a key conceptual distinction between compute-bound and bandwidth-bound savings.
- `fp32_dest_acc_en.md` line 143: the clarification that `fp32_dest_acc_en` affects only the on-chip FPU destination register and not L1, with explicit statement there is no "twice the L1 space" overhead — load-bearing; corrects a natural and confirmed misconception. Removing risks re-introducing the confusion that the Agent A changelog records was an actual error.

## VERDICT
- Crucial updates: no
