# Chapter 7 — Validation: Confirming Correctness of the Async All-Gather Pipeline

This chapter describes the validation strategy for confirming that removing `ttnn.synchronize_device()` from `_maybe_all_gather` — and, for the Type B2 path, replacing synchronous `ttnn.all_gather` with `ttnn.experimental.all_gather_async` plus cycling semaphores — does not introduce race conditions, numerical errors, or output corruption in the hybrid attention stack. By the end of this chapter you will have a concrete test plan covering three complementary validation dimensions, with acceptance criteria for each.

---

## Prerequisites

This chapter assumes the implementation described in [Chapter 6](../ch6_implementation/index.md) is complete and the following are available:

- A working T3K setup with both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` instantiated and runnable for decode.
- A reference implementation that uses the original synchronous `ttnn.all_gather` plus `ttnn.synchronize_device()` (either a separate branch or a configuration flag that reverts to the old behavior).
- Tracy installed and accessible if the latency measurement procedure (Method 2 in Chapter 5) is used.

---

## Three Validation Dimensions

The three files in this chapter address complementary failure modes:

1. **Functional correctness** — Does the modified implementation produce numerically correct outputs? This catches races and semaphore bugs at the output level. A PCC comparison against a reference run is the primary acceptance criterion.

2. **Multi-replay stability** — Does the traced implementation produce consistent outputs across N consecutive trace replays? The first replay passing is necessary but not sufficient; intermittent failures on replay 2, 3, or later are the signature of an unresolved semaphore aliasing or missing reset-before-replay bug.

3. **Latency measurement** — Does the implementation deliver the expected throughput improvement predicted by Chapter 5? This confirms that the removal is attributable to the `synchronize_device` call and that no regression was introduced.

Run all three in the order listed. A failure in dimension 1 must be resolved before dimension 2 is meaningful, and dimension 3 is only interpretable after dimensions 1 and 2 both pass.

---

## What's Next

Read the following files in order:

1. [`functional_correctness.md`](./functional_correctness.md) — The PCC-based numerical correctness test: how to run it, what a failure indicates, how to distinguish semaphore bugs from race conditions, and how to extend it to the full hybrid decoder stack.

2. [`multi_replay_stability.md`](./multi_replay_stability.md) — The trace replay consistency test: N-replay comparison, deadlock detection with timeout, and stress-test interleaving of traced and non-traced calls.

3. [`latency_measurement.md`](./latency_measurement.md) — Before/after latency measurement for the `synchronize_device` removal benefit and, after full trace enablement, the combined improvement from trace dispatch overhead elimination.
