# Async CCL Semaphore Behavior Under Trace Replay

When a model using `ttnn.experimental.reduce_scatter_minimal_async` or `ttnn.experimental.all_gather_async` is placed inside a `ttnn.begin_trace_capture` / `ttnn.execute_trace` bracket, the tt-metal trace mechanism bakes semaphore handle L1 addresses into an immutable DRAM command buffer at capture time. The host-side double-buffer cycling counter in `TT_CCL` continues advancing independently across replays, so the handle the host believes is active diverges from the handle the trace actually uses. This mismatch produces either silent data corruption or a hang on every replay after the first.

## Who This Guide Is For

This guide is written for ML systems engineers and model framework developers working with the tt-transformers stack or `models/common/modules/tt_ccl.py` who need to enable trace capture for models that use async CCL ops. It assumes you are diagnosing an existing failure or are about to add trace support and want to understand the correct approach before writing code.

## Prerequisites

Readers are assumed to be familiar with the following before starting:

- The tt-transformers `Generator` trace API — `begin_trace_capture`, `end_trace_capture`, and `execute_trace` — and the compile-capture-replay lifecycle that those calls define.
- The `TT_CCL` class and the fact that `get_and_cycle_*` methods advance a host-side cycling counter to select the next semaphore handle from a double-buffered pair.
- The high-level purpose of async CCL: overlapping Ethernet data movement with compute so that communication latency is hidden behind kernel execution.
- Basic `ttnn` tensor and device APIs, and Python-level familiarity with `ttnn.create_global_semaphore` and `ttnn.reset_global_semaphore_value`.

## Chapter Overview

| Chapter | Title | Key Question Answered |
|---------|-------|-----------------------|
| Ch1 | GlobalSemaphore Internals and the Double-Buffer Design | What is a GlobalSemaphore and why does TT_CCL use double-buffered handles? |
| Ch2 | How Semaphore Addresses Flow into Kernel Runtime Arguments | How do semaphore L1 addresses get baked into the trace, and why can't they change per-replay? |
| Ch3 | The Host-Counter / Trace-Handle Mismatch | What exactly diverges after trace capture, and what failure modes does it produce? |
| Ch4 | Correct Synchronization Strategies for Traced Async CCL | What resets are required, and how must the capture be structured? |
| Ch5 | Implementing Trace Support: A Step-by-Step Guide | What code changes are needed and how do you verify correctness? |

## Reading Order

The chapters build on each other; read them in order 1 through 5. Ch1 establishes what a `GlobalSemaphore` is, Ch2 explains how its address enters the trace, and Ch3 uses both of those to characterize the mismatch precisely. Ch4 and Ch5 then prescribe and implement the fix. If you have a specific question, use the Chapter Overview table to identify the most relevant chapter, but be aware that later chapters define terms and reference analysis introduced in earlier ones.

## Quick Reference

The following rules summarize the guidance developed across all five chapters. Return here after reading for a condensed reminder.

- The trace bakes handle L1 addresses at capture time. Every replay uses those exact handles regardless of what `get_and_cycle_*` has returned since capture.
- Before each `execute_trace` call: reset device semaphore values to 0 for all handles that were captured, then restore host index fields to their capture-time values so the host and trace are back in sync.
- For `use_composite=True`, four handle groups must be reset: the RS semaphore handles, the barrier semaphore for RS, the AG semaphore handles, and the barrier semaphore for AG.
- In the older `models/tt_transformers/tt/ccl.py`: `cluster_axis=0` maps to `semaphore_index=2` (the same slot as `None`) because `not 0` evaluates to `True`. Verify which file your model imports before writing reset logic.
- The two failure modes are silent data corruption (wrong output values, no error raised) and hang (deadlock waiting for a semaphore count that is never reached). Both have the same root cause: a missing reset or a handle mismatch between host and trace.
