# Chapter 3 — Trace Compatibility Verdict and Requirements

This chapter synthesises the findings from Chapters 1 and 2 into direct answers to
the three research questions posed at the start of the guide. It states the concrete
requirements that `TTNNLinearIColShardedWAllReduced` must satisfy to operate correctly
under `TracedRun`, and identifies any remaining risks that downstream integration work
must address.

## Research questions and one-line answers

| # | Question | One-line answer | Detail section |
|---|----------|-----------------|----------------|
| Q1 | Is `ttnn.all_reduce` (synchronous, Ring topology) compatible with trace capture and replay? | Conditionally yes: the non-composite path (reduce-scatter + all-gather) avoids persistent semaphore issues, but its intermediate `scattered_tensor` is dynamically allocated and must be pre-allocated as a persistent buffer before trace capture. The composite path uses `composite_all_gather` (`all_broadcast` + `concat`) followed by a local reduce (`moreh_sum` for non-float32, `ttnn::sum` via transpose for float32) and is trace-incompatible without persistent buffer injection. | [`q1_trace_compatibility.md`](./q1_trace_compatibility.md) |
| Q2 | Does `ttnn.all_reduce` use any internal semaphore state that could conflict with trace replay? | No: `ttnn.all_reduce` passes `std::nullopt` for all three `GlobalSemaphore` arguments; the downstream synchronous ops use only per-program local semaphores that are re-created on every dispatch and hold no state across replays. | [`q2_semaphore_state.md`](./q2_semaphore_state.md) |
| Q3 | Are there any known limitations or requirements for using `ttnn.all_reduce` inside a traced region? | Yes: five requirements must be met — output buffer stability, intermediate buffer pre-allocation, `FABRIC_1D_RING` fabric config, even ring size, and confirmed routing to the non-composite path. | [`q3_requirements_and_limitations.md`](./q3_requirements_and_limitations.md) |

## Reading order

Work through the files in the order below. Each file ends with a navigation footer
pointing to the next.

1. **[`q1_trace_compatibility.md`](./q1_trace_compatibility.md)**
   Path selection logic, the composite vs non-composite decision, what
   `TTNNLinearIColShardedWAllReduced` inputs trigger on T3K, and the role of
   `sharded_to_interleaved` inside `all_reduce_async`.

2. **[`q2_semaphore_state.md`](./q2_semaphore_state.md)**
   The semaphore audit: why `ttnn.all_reduce` introduces no persistent global
   semaphore state, and how this contrasts with `ttnn.experimental.all_reduce_async`.

3. **[`q3_requirements_and_limitations.md`](./q3_requirements_and_limitations.md)**
   The complete, actionable requirements list and known limitations, structured for
   direct use during integration.

## Key source files referenced

| File | Purpose |
|------|---------|
| `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp` | `ExecuteAllReduce::invoke` — passes `std::nullopt` semaphores, entry point |
| `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp` | `ExecuteAllReduceAsync::invoke` (cluster-axis overload) — path selection, `sharded_to_interleaved`, composite and non-composite branches |
| `ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp` | `use_composite_all_gather`, `use_composite_reduce_scatter` — path selection guards |
| `models/experimental/tt_symbiote/modules/linear.py` | `TTNNLinearIColShardedWAllReduced.forward` — calls `ttnn.all_reduce` with `ttnn.DRAM_MEMORY_CONFIG` |
| `models/experimental/tt_symbiote/core/run_config.py` | `TracedRun`, `@trace_enabled`, `@trace_disabled` decorator logic |
| `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py` | CCL trace-mode test demonstrating `all_reduce_async` under `begin_trace_capture` |
| `models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_all_reduce.py` | End-to-end CCL trace test showing persistent output and intermediate buffer pattern |

## What's next

1. [`q1_trace_compatibility.md`](./q1_trace_compatibility.md)
2. [`q2_semaphore_state.md`](./q2_semaphore_state.md)
3. [`q3_requirements_and_limitations.md`](./q3_requirements_and_limitations.md)
