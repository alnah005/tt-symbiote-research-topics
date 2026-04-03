# Compression Analysis — Chapter 7: Implementation Roadmap

**Analyst:** Agent C (Compressor)
**Date:** 2026-04-03
**Scope:** Redundancy and bloat only

---

## File 1: `index.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The High-Level Timeline Summary table (lines 32-38) and the total duration estimate (line 39) duplicate `phased_plan.md` lines 229-234 (Timeline Summary) nearly verbatim. The Decision Points section (lines 45-51) restates the Phase Gate content from `phased_plan.md` (Gates 1, 2, and 3) in slightly different words.

**MINOR suggestion:** Remove the High-Level Timeline Summary table and the Decision Points section entirely. Replace both with a single sentence: "See [`phased_plan.md`](./phased_plan.md) for the four-phase timeline (5-7 engineer-weeks total) and phase gate criteria." The index file should orient the reader, not re-present content from the files it links to. This would reduce `index.md` by approximately 20 lines (~36% of its body content).

---

## File 2: `phased_plan.md`

**Crucial updates: no**

**Load-Bearing Evidence:** The Critical Path section (lines 237-244) restates what is already said in the index.md Overview (line 41: "The critical path runs through Phase 2, where the attention module with 2D RoPE integration is both the highest-effort and highest-risk component") and also echoes the Phase 2 Phase Gate (line 117: "Particular attention to the attention module with 2D RoPE"). The same point — "attention + 2D RoPE is highest-effort, highest-risk, on the critical path" — appears three times across the two files.

**MINOR suggestion:** Consolidate the Critical Path and Parallelization Opportunities subsections (lines 237-251) into the Phase 2 section itself, since that is where the critical-path work actually lives. A single sentence after the Module Port Order table ("The attention + 2D RoPE integration is the critical-path item; assign it to the most experienced engineer") replaces the standalone section. The Timeline Summary ASCII art (lines 229-234) also duplicates the index table and can be removed. Net reduction: ~25 lines.

---

## File 3: `risk_register.md`

**Crucial updates: no**

**Load-Bearing Evidence:** Each risk entry contains a "Residual risk after mitigation" paragraph that is functionally identical across five of the six risks — all say "Low" with a brief restatement of why the mitigation works. This is structural boilerplate, not new information. Additionally, Risk 1's mitigation steps (lines 28-34) repeat the Phase 4 task list from `phased_plan.md` (lines 183-189: pre-trace five budgets, pad to nearest budget, batch by budget). The same five-budget strategy is described in full detail in both files.

**MINOR suggestion:** (1) Replace the six "Residual risk after mitigation" paragraphs with a single row in the Risk Summary Table (add a "Residual" column). This eliminates ~6 repetitive paragraphs. (2) In Risk 1's mitigation, replace the four numbered steps with a cross-reference: "See Phase 4 tasks in [`phased_plan.md`](./phased_plan.md) for the full pre-tracing and budget-padding strategy." Keep only the summary: "Pre-trace five standard budgets; pad inputs to the nearest budget at runtime." Net reduction: ~20 lines.

---

## Cross-File Redundancy

| Repeated Content | Locations | Recommendation |
|-----------------|-----------|----------------|
| 5-7 week timeline + phase durations | `index.md` table, `phased_plan.md` ASCII summary | Keep in `phased_plan.md` only |
| Go/no-go gate criteria | `index.md` Decision Points, `phased_plan.md` Phase Gates | Keep in `phased_plan.md` only |
| "Attention + 2D RoPE is critical path" | `index.md` line 41, `phased_plan.md` lines 117/243, `risk_register.md` Risk 2 | State once in `phased_plan.md` Phase 2 |
| Five-budget pre-tracing strategy | `phased_plan.md` Phase 4 tasks, `risk_register.md` Risk 1 mitigation | Full detail in `phased_plan.md`; cross-reference from risk register |
| PCC thresholds (0.999 module, 0.998 e2e) | `phased_plan.md` Gates 2-3, `risk_register.md` Monitoring table | Keep in `phased_plan.md` gates; reference from risk register |

**Estimated total reduction:** ~65 lines across the three files (~15% of combined content), with no loss of actionable information.
