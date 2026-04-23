# Chapter 7 — Implementation Roadmap and Trace Integration

This chapter synthesizes the findings from all prior chapters into a concrete, prioritized implementation plan that a developer can follow to achieve Metal Trace compatibility for `TTNNQwen3LinearAttention`. Prior chapters established the mathematical structure of the DeltaNet recurrence (Chapter 2), identified the TTNN op gaps and fallback paths (Chapter 3), analyzed the fused kernel portability from CUDA to Wormhole (Chapter 4), surveyed the full op availability across the decode forward pass (Chapter 5), and quantified the latency cost of the current host-CPU fallback and the expected gains from on-device forms (Chapter 6). This chapter converts all of that analysis into a numbered task list with priority and complexity ratings, a step-by-step Metal Trace integration checklist, and a verification test matrix.

> **Key Finding:** There are 7 tasks to achieve a fully on-device, trace-compatible DeltaNet decode step. The critical path is Tasks 1, 2, and 5 (state tensor on-device, decay gates on-device, recurrent step wired to TTNN); these three tasks are sufficient to achieve Metal Trace compatibility for `TTNNQwen3LinearAttention`. Tasks 3 and 4 (causal conv1d and gated RMSNorm, both `[AVAILABLE — needs wiring]`) are independent of the critical path and can proceed in parallel. Tasks 6 and 7 are latency optimizations and prefill coverage respectively — neither is required for correctness or trace compatibility, but Task 6 delivers the 50–120× speedup over the host fallback.

## Files in This Chapter

Read in this order:

1. `task_list_and_priority.md` — The complete 7-task prioritized list. Each task includes priority rating, complexity rating, description, prerequisites, and chapter cross-references. The critical path (Tasks 1 → 2 → 5) and parallel tracks (Tasks 3, 4) are called out explicitly.
2. `trace_integration_checklist.md` — An 8-step checklist for integrating the on-device DeltaNet decode into a Metal Trace capture/replay loop. Covers pre-allocation, host-crossing guard, in-place state update compatibility, trace capture, multi-step correctness verification, long-run divergence testing, Tracy profiling, and latency measurement.
3. `verification_and_testing.md` — A test matrix covering unit tests for each task, an integration test on a 10-layer prefix of Qwen3.6-35B-A3B on T3K, a trace-specific correctness test, and a performance regression test with latency targets from Chapter 6.

## What's Next

After all tasks are complete and the trace integration checklist passes, this guide's goal — a pure on-device DeltaNet decode step running inside a Metal Trace on T3K — is achieved. Before a full-stack end-to-end trace capture of the entire Qwen3.6-35B-A3B decoder works, the companion topics on `trace-safe cos/sin pre-replication in TTNNQwen3FullAttention` and removing `synchronize_device` from `maybe_all_gather` must also be resolved. This guide's implementation is independent of those companion topics and can be completed first, validated in isolation, and then integrated once the full-stack trace prerequisites are met.
