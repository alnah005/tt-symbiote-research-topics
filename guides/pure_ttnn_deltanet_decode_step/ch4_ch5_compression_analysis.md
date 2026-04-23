# Compression Analysis — Chapters 4 and 5: GDN Fused Kernel and Scan Primitives Survey

## Pass 1

### Crucial issues found: 0

No pair of files across Chapters 4 and 5 contains a substantially identical block of 5 or more near-verbatim lines that serves no purpose.

The closest candidates examined and ruled out:

**CB layout tables (`gdn_full_fused_inplace_analysis.md` §3.3 vs. `wormhole_t3k_adaptation.md` §1.2)**
Both files contain a 5-row CB layout table covering CB0–CBOUT. The rows share the same CB identifiers and role descriptions, but the tables differ in column structure (the analysis table has 3 columns: index, role, size; the adaptation table has 5 columns: CB, role, elements, BF16 bytes, notes), in numeric presentation (the analysis file uses a single combined size field; the adaptation file separates element count from byte count), and in purpose (the analysis file accounts for total CB usage to establish feasibility; the adaptation file is a specification checklist for the port engineer to verify against actual source values). Each table does work its own file cannot delegate to the other. This is not a crucially redundant block.

**"Key Finding" callouts (`ch4/index.md`, `gdn_full_fused_inplace_analysis.md`, `wormhole_t3k_adaptation.md`)**
All three files contain a callout stating that the composed TTNN form is the immediate fix and the fused kernel is a latency optimization. In the index this is one sentence orienting the reader; in the analysis file it is two sentences with the dispatch-count framing; in the adaptation file it summarizes the port scope. The wording differs across all three. No block reaches 5 near-verbatim lines.

**Survey summary tables (`ch5/index.md` §Survey Summary vs. `gla_and_related_kernel_survey.md` §3)**
The ch5 index contains a 4-row summary table (Mamba, parallel prefix, GLA/RetNet, composed TTNN). The GLA survey contains a 6-row table with different columns (adds "structural similarity" and expands GLA and RetNet into separate rows). The index table is a navigation aid; the survey table is the authoritative finding. They are not near-verbatim.

**Mamba inner loop pseudocode (`mamba_ssm_kernel_review.md` §3.2)**
The DeltaNet vs. Mamba inner loop pseudocode block appears only in `mamba_ssm_kernel_review.md`. It is referenced by concept in `ch5/index.md` and `gla_and_related_kernel_survey.md` but not reproduced there.

---

### VERDICT

Crucial updates: no

Chapters 4 and 5 approved.
