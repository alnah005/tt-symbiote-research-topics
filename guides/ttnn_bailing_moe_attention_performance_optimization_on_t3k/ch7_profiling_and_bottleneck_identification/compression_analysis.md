# Compression Analysis — Ch7 Pass 5

## Verdict: Crucial updates: no

## Load-Bearing Evidence

- **index.md prerequisite summaries** are load-bearing: each bullet restates a specific estimate (e.g., "~150–171 µs [ESTIMATE]", "74–92 µs [ESTIMATE]") and a specific op cluster name, giving the reader a quantitative preview before they enter the sub-files. These numbers are not duplicated verbatim in the sub-files' own introductions.
- **index.md tool-comparison table** is load-bearing: the two-row table is the only location that differentiates when to use op timers vs. Tracy with side-by-side granularity, setup complexity, and best-use framing.
- **ttnn_op_timers.md Section 1.1–1.3 ordering rationale** ("Use them in the order shown") is load-bearing: the three mechanisms differ in accuracy and setup cost, and the ordering instruction is the only place this priority is stated.
- **ttnn_op_timers.md Section 2.1 op-name-to-source mapping table** is the canonical cross-reference between CSV `op_name` values and `TTNNBailingMoEAttention` call sites. It appears nowhere else at this granularity.
- **ttnn_op_timers.md Section 2.2 `metadata` column explanation** is load-bearing: the `HEIGHT_SHARDED->DRAM_INTERLEAVED` example is the only place that explains how to distinguish between the many `ttnn::to_memory_config` rows from each other.
- **ttnn_op_timers.md Section 4 dispatch-bound vs. execution-bound distinction** is load-bearing: the with-sync / without-sync comparison methodology is explained only here, and the per-op characterizations (`paged_sdpa_decode` execution-bound, `ConcatMeshToTensor` blocking host-side) are specific and non-redundant.
- **tracy_profiling.md Section 1.1 `-DENABLE_TRACY_ETH=ON` note** is load-bearing: the consequence (CCL Ethernet spans silently absent without it) is stated only in this note and affects the primary use case of CCL investigation.
- **tracy_profiling.md Section 5.2 IOMMU inflation pitfall** is load-bearing: the "first-execution only, resolves after 1 call not 5" detail is distinct from the JIT warm-up instruction in Section 5.1 and is stated only here.
- **bottleneck_decision_tree.md Branch H (paged_fill_cache)** is load-bearing: the three root-cause sub-cases (DRAM allocation, block_size misconfiguration, seq_len growth) are specific and not covered in any other chapter or branch.
- **bottleneck_decision_tree.md Section 3 iteration loop** is load-bearing: the "change exactly one variable" rule and the delta-comparison script are operational instructions not stated elsewhere.
- **bottleneck_decision_tree.md Scenario 3** (post-CCL fix revealing memory transitions) is load-bearing: it demonstrates that the dominant bottleneck shifts after a fix, which is not illustrated by Scenarios 1 or 2.

## CRUCIAL Updates (only if verdict is yes)

N/A

## MINOR Suggestions

1. **ttnn_op_timers.md Section 1.2, `ttnn.tracer.profile` parameter list** repeats what the inline code comments already convey. The `output_dir`, `device`, and `enabled` bullet descriptions are each one sentence that restates the parameter name itself (e.g., "`output_dir` (str): directory path where the profiler report is written"). The code block immediately above shows all three parameters in context. The bullet list adds no new information and can be removed, saving approximately 6 lines without any content loss.

2. **tracy_profiling.md Section 3.1 capture window sizing prose** duplicates information already shown in the Section 3.2 code. The sentences "A single Ling decode step takes approximately 600–1200 µs … The Tracy capture should cover at minimum: 5 warm-up steps, a brief idle gap, 1–3 profiled decode steps" are immediately demonstrated by the Section 3.2 isolation harness. The key number ("50–100 ms capture window") can be kept as an inline note on the `tracy-capture` command itself; the preceding two prose sentences can be cut.

3. **bottleneck_decision_tree.md Section 3, Steps 1 and 4** each contain a `cp` bash snippet whose only purpose is to illustrate a filename naming convention. The two blocks consume approximately 8 lines combined. A single prose sentence ("save each CSV with a descriptive name encoding the active configuration and context length, e.g., `baseline_hifi4_num_links1_ctx4096.csv`") in Step 1, plus a back-reference in Step 4, would convey the same convention without duplicating the bash block structure.

4. **index.md "Reading Order" list** has three bullets each of which contains a long description that closely paraphrases the opening sentence of the linked file (e.g., "How to enable op-level timing, how to read the output report, and how to map op names to `TTNNBailingMoEAttention` source call sites" mirrors the first sentence of `ttnn_op_timers.md`). Shortening each bullet to the link plus a brief label (e.g., "Op-level timing: enabling, reading, and source-code mapping") removes the maintenance risk of two near-identical descriptions drifting out of sync.

5. **ttnn_op_timers.md Section 4, Step 3** opens with a definitional sentence ("The row where `cumulative_pct` first exceeds 80 defines the minimal set of ops that account for >80% of decode latency") that restates what the column name `cumulative_pct` and the cumsum code in Step 2 already make self-evident. The following sentence ("In practice, for Ling on T3K, this cutoff is typically reached within the top 3–5 ops [ESTIMATE]") is the only new information. The definitional sentence can be deleted; the estimate sentence should be retained.
