# Chapter 7 — End-to-End Optimization Summary

## Context

This chapter addresses all research questions **Q1–Q8**. It does not introduce new profiling data. Every claim here traces directly to the chapter that established it (Ch1–Ch6). The purpose is to synthesize findings into an actionable, prioritized roadmap.

---

## What This Chapter Contains

Chapters 1–6 each answered a focused research question about `TTNNMoE` performance on T3K:

- **Ch1** established the full call-stack anatomy of `TTNNMoE.forward` and `TTNNExperts.forward`, and identified the `Glm4MoeNaiveMoeHybrid` CPU fallback path.
- **Ch2** measured CCL op latency and evaluated topology and buffer settings, finding that `all_gather_async` + `reduce_scatter_minimal_async` collectively account for 55–65% of decode latency.
- **Ch3** isolated the dominant cost inside `TTNNExperts.forward`: the three `sparse_matmul` calls account for the majority of expert compute time; weight application (repeat+permute) is a secondary overhead.
- **Ch4** evaluated program config optimality and math fidelity. `in0_block_w=min(4, hidden_tiles)` is suboptimal for Bailing's hidden size; LoFi math fidelity can reduce expert matmul latency by ~10–20% with acceptable accuracy impact at decode quality levels.
- **Ch5** established the profiling toolchain (TTNN op timers and Tracy), and quantified the `TTNNMoERouterDecode` 3-pass centering overhead at ~10–15 µs per pass — active only when `n_group <= r.topk_group`.
- **Ch6** confirmed that `Glm4MoeNaiveMoeHybrid` with `ttnn = False` at `moe.py:L569` is the sole CPU fallback. The fix is a one-line change.

This chapter delivers two documents:

1. **[Optimization Priority Matrix](optimization_priority_matrix.md)** — A ranked table mapping each of the eight research questions (Q1–Q8) to its primary chapter, expected latency impact, implementation complexity, and a one-line action.

2. **[Recommended Action Plan](recommended_action_plan.md)** — A sequenced, phase-by-phase action plan with code locations, measurement gates, and model-specific notes for GLM-4-MoE and Bailing.

---

## How to Use This Chapter

Start with the Priority Matrix to determine which optimizations apply to your deployment. If `Glm4MoeNaiveMoeHybrid` is not active (i.e., you are running `TTNNMoE` or `TTNNBailingMoE` directly), skip Q6 entirely. Then follow the Action Plan in phase order: establish a baseline before making any change, and re-collect timing data after each phase before proceeding to the next.

The profiling toolchain described in Ch5 and formalized in Phase 1 of the Action Plan is the prerequisite for all quantitative decisions in Phases 3–5. Do not skip it.
