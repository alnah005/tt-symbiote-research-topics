# Chapter 4 — Integration Checklist and Test Strategy

This chapter translates the requirements established in Chapter 3 into two concrete
deliverables: a sequential checklist of pre-conditions and post-capture validation steps
for integrating `TTNNLinearIColShardedWAllReduced` into a `TracedRun` execution context,
and an annotated test skeleton that exercises the module end-to-end under trace capture
and replay. Together they bridge the gap between the theoretical compatibility analysis
and production-ready validation.

## Learning objectives

After working through this chapter you will be able to:

- Enumerate every pre-condition that must hold before calling
  `ttnn.begin_trace_capture` with `TTNNLinearIColShardedWAllReduced` in scope.
- Identify which buffer stability requirements `TracedRun._capture_trace` handles
  automatically and which ones the caller must handle explicitly.
- Write or review a minimal unit test that validates trace correctness using two
  consecutive `ttnn.execute_trace` calls and a `comp_pcc` comparison against a
  non-traced reference.

## Reading order

Work through the files below in order. Each content file ends with a navigation footer.

1. **[`integration_checklist.md`](./integration_checklist.md)**
   Pre-conditions to verify before enabling trace, ordered from device setup through
   buffer allocation and `TracedRun` configuration. Includes post-capture validation
   steps for detecting aliasing and numerical divergence.

2. **[`minimal_test_pattern.md`](./minimal_test_pattern.md)**
   An annotated code skeleton for a pytest unit test covering device setup, module
   instantiation, kernel warm-up, trace capture, dual replay, and numerical comparison.
   Parametrize axes are listed for `in_features`, `dtype`, and `cluster_axis`.

## What's next

1. [`integration_checklist.md`](./integration_checklist.md)
2. [`minimal_test_pattern.md`](./minimal_test_pattern.md)
