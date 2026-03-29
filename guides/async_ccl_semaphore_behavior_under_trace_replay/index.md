# Async CCL Semaphore Behavior Under Trace Replay

When a model using `ttnn.experimental.reduce_scatter_minimal_async` or `ttnn.experimental.all_gather_async` is placed inside a `ttnn.begin_trace_capture` / `ttnn.execute_trace` bracket, the tt-metal trace mechanism bakes semaphore handle L1 addresses into an immutable DRAM command buffer at capture time. The host-side double-buffer cycling counter in `TT_CCL` continues advancing independently across replays, so the handle the host believes is active diverges from the handle the trace actually uses — producing either silent data corruption or a hang on every replay after the first.

---

## Who This Guide Is For

This guide is written for ML systems engineers and model framework developers working with the tt-transformers stack or `models/common/modules/tt_ccl.py` who need to enable trace capture for models that use async CCL ops. It assumes you are diagnosing an existing failure or are about to add trace support and want to understand the correct approach before writing code.

---

## Prerequisites

Readers are assumed to be familiar with the following before starting:

- The tt-transformers `Generator` trace API — `begin_trace_capture`, `end_trace_capture`, and `execute_trace` — and the compile-capture-replay lifecycle.
- The `TT_CCL` class and the fact that `get_and_cycle_*` methods advance a host-side cycling counter to select the next semaphore handle from a double-buffered pair.
- The high-level purpose of async CCL: overlapping Ethernet data movement with compute so that communication latency is hidden behind kernel execution.
- Basic `ttnn` tensor and device APIs, and Python-level familiarity with `ttnn.create_global_semaphore` and `ttnn.reset_global_semaphore_value`.

---

## How to Use This Guide

| Goal | Recommended path |
|---|---|
| Understand why trace replay breaks with async CCL ops | Read [Ch1](ch1_global_semaphore_internals/index.md) → [Ch2](ch2_semaphore_rta_path/index.md) → [Ch3 — `what_gets_baked_in.md`](ch3_mismatch_analysis/what_gets_baked_in.md) |
| Identify the exact failure mode you are seeing | Read [Ch3 — `failure_modes.md`](ch3_mismatch_analysis/failure_modes.md) |
| Understand what resets are required and why | Read [Ch4](ch4_synchronization_strategies/index.md) |
| Implement trace support in a model right now | Go directly to [Ch5 — `code_changes_required.md`](ch5_implementation_guide/code_changes_required.md) |
| Verify an implementation is correct | Read [Ch5 — `verifying_correctness.md`](ch5_implementation_guide/verifying_correctness.md) |
| Find out whether existing tt-transformers traces need changes | Read [Ch4 — `existing_patterns_in_tt_transformers.md`](ch4_synchronization_strategies/existing_patterns_in_tt_transformers.md) |

---

## Chapter Index

| Chapter | Title | Description |
|---|---|---|
| [Ch1 — GlobalSemaphore Internals and the Double-Buffer Design](ch1_global_semaphore_internals/index.md) | GlobalSemaphore Internals and the Double-Buffer Design | What a `GlobalSemaphore` is, how `TT_CCL` allocates double-buffered handles, and how `get_and_cycle_*` selects them |
| [Ch2 — How Semaphore Addresses Flow into Kernel Runtime Arguments](ch2_semaphore_rta_path/index.md) | How Semaphore Addresses Flow into Kernel Runtime Arguments | How semaphore L1 addresses become RTAs, how `override_runtime_arguments` writes them on every cache hit, and how `assemble_dispatch_commands` freezes them into the DRAM command buffer |
| [Ch3 — The Host-Counter / Trace-Handle Mismatch](ch3_mismatch_analysis/index.md) | The Host-Counter / Trace-Handle Mismatch | Exactly what diverges after trace capture, the two failure modes (silent corruption and hang), and whether async CCL ops can be placed inside a trace bracket |
| [Ch4 — Correct Synchronization Strategies for Traced Async CCL](ch4_synchronization_strategies/index.md) | Correct Synchronization Strategies for Traced Async CCL | The required resets (device semaphore values and host index fields), how to structure the capture, and an audit of existing tt-transformers trace paths |
| [Ch5 — Implementing Trace Support: A Step-by-Step Guide](ch5_implementation_guide/index.md) | Implementing Trace Support: A Step-by-Step Guide | Concrete code changes, complete capture and replay checklists, and a verification methodology |

The chapters build on each other; read them in order 1 through 5. If you have a specific question, use the "How to Use This Guide" table above to identify the most relevant entry point, but be aware that later chapters define terms and reference analysis introduced in earlier ones.

---

## Quick Reference

| Operation | What it does | Where to learn more |
|---|---|---|
| `ttnn.create_global_semaphore(device, cores, 0)` | Allocates a sharded L1 buffer; returns a `GlobalSemaphore` whose `.address()` is stable for its lifetime | [Ch1 — `global_semaphore_api.md`](ch1_global_semaphore_internals/global_semaphore_api.md) |
| `ttnn.reset_global_semaphore_value(handle, 0)` | Enqueues a write that sets the L1 semaphore word to 0; must be enqueued **before** `execute_trace` to take effect before trace kernels run | [Ch4 — `resetting_device_semaphore_values.md`](ch4_synchronization_strategies/resetting_device_semaphore_values.md) |
| `captured_ag_idx = list(tt_ccl.ag_semaphores_idx)` | Snapshots the current host index state (use `list(...)` to copy, not reference) | [Ch5 — `code_changes_required.md`](ch5_implementation_guide/code_changes_required.md) |
| `tt_ccl.ag_semaphores_idx = list(captured_ag_idx)` | Restores host indices to capture-time values before each `execute_trace` | [Ch5 — `code_changes_required.md`](ch5_implementation_guide/code_changes_required.md) |
| `semaphore_index = 2 if not cluster_axis else cluster_axis` | Older `models/tt_transformers/tt/ccl.py` formula — `cluster_axis=0` maps to `si=2`, not 0, because `not 0` is `True` | [Ch1 — `double_buffer_design.md`](ch1_global_semaphore_internals/double_buffer_design.md) |

**Key rules:**

- The trace bakes handle L1 addresses at capture time. Every replay uses those exact handles regardless of what `get_and_cycle_*` has returned since capture.
- Before each `execute_trace`: (1) reset device semaphore values to 0 for all captured handles, (2) restore `TT_CCL` index fields to capture-time values.
- For `use_composite=True`: four handle groups must be reset — RS handles, barrier for RS, AG handles, barrier for AG — including both barrier double-buffer slots.
- The two failure modes are silent data corruption (wrong values, no error) and hang (deadlock). Both stem from a missing reset or handle mismatch.

---

## Source Code Locations

| Component | Location |
|---|---|
| `TT_CCL` class (older) | `models/tt_transformers/tt/ccl.py` |
| `TT_CCL` class (newer) | `models/common/modules/tt_ccl.py` |
| AG program factory (RTA assignments) | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp` |
| RS program factory (RTA assignments) | `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp` |
| RS reader kernel (self-reset at line 275) | `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_reader.cpp` |
| RS writer kernel (self-reset at lines 226, 479) | `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduce_scatter_minimal_async_writer.cpp` |
| AG reader kernel (self-reset at line 295) | `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp` |
| Mesh trace assembly | `tt_metal/impl/dispatch/mesh_command_queue.cpp` (`assemble_dispatch_commands`, `populate_mesh_buffer`) |
