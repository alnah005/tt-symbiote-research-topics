# Compression Analysis: Chapter 8 — Optimization Roadmap and Testing — Pass 1

## Summary
- Total files analyzed: 4 (index.md, optimization_roadmap.md, testing_infrastructure.md, running_tests.md)
- Estimated current line count: ~310 lines
- Estimated post-compression line count: ~275 lines
- Estimated reduction: ~11%

## CRUCIAL Suggestions

### [running_tests.md, testing_infrastructure.md] Test class threshold tables
**Issue:** running_tests.md (lines ~43–66) restates test class names, methods, and PCC thresholds in table form — identical information already documented in testing_infrastructure.md (lines ~61–105).
**Suggestion:** Replace duplicate table in running_tests.md with a single cross-reference line to testing_infrastructure.md.

## MINOR Suggestions

### [testing_infrastructure.md, running_tests.md] TestFusedKernelPCC exception
**Issue:** Both files note that TestFusedKernelPCC does not require model download, phrased differently ("does not need model download" vs "ignore HF_MODEL").
**Suggestion:** Unify to one phrasing in testing_infrastructure.md and use a cross-reference in running_tests.md.

## Load-Bearing Evidence
- `optimization_roadmap.md` lines ~25–50: Metal Trace prerequisites section — load-bearing because it enumerates specific in-place copy patterns (conv ring buffer, recurrent state, RoPE matrices) that must remain stable for Trace to work.
- `testing_infrastructure.md` lines ~107–124: PCC metric definition and code snippet — load-bearing; not stated anywhere else in the guide.
- `running_tests.md` lines ~5–17: HF_MODEL environment setup with concrete bash examples — load-bearing for operational use; the testing_infrastructure.md does not provide executable commands.

## VERDICT
- Crucial updates: no

---

# Compression Analysis: Chapter 8 — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~275 lines (after Pass 1 consolidation)
- Estimated post-compression line count: ~265 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
(none — Pass 1 CRUCIAL item resolved)

## MINOR Suggestions

### [optimization_roadmap.md] Expected latency table
**Issue:** Latency trajectory table repeats "Baseline (host recurrence + host RoPE) | 86 ms | 6.8%" which also appears in ch7 bottleneck_analysis.md efficiency ceiling table.
**Suggestion:** Add a cross-reference note "(see also ch7 bottleneck_analysis.md)" rather than removing — the table provides the roadmap context so it is useful here, but acknowledgment of the cross-chapter repeat would help.

## Load-Bearing Evidence
- `optimization_roadmap.md` lines ~65–85: per-row MoE routing implementation path (3-step: extend to_torch, per-row topk, group-by-expert dispatch) — load-bearing detail not present in ch5.
- `testing_infrastructure.md` lines ~55–60: "Tests require both a Blackhole device and (except TestFusedKernelPCC) a downloaded HF checkpoint" — load-bearing operational constraint.
- `running_tests.md` lines ~90–115: "Adding a new test" section — unique content with PCC threshold rationale (0.99 for bfp8, 0.998 for kernel output, 0.999 for state).

## VERDICT
- Crucial updates: no
