# Plan: TTNN Ops Trace Guide

## Audience

**Target reader:** ML engineers who already write and run TTNN-based inference models on Tenstorrent hardware. They are comfortable with basic TTNN op APIs, device initialization, and tensor manipulation. They have encountered terms like "trace" or "command queue" in documentation or profiling output but do not have a systematic understanding of how the dispatch pipeline works or when tracing applies.

**What they already know:**
- TTNN Python API: creating tensors, calling ops, moving data between host and device
- Basic inference loop structure (prefill / decode separation)
- High-level awareness that Tenstorrent devices have host-managed dispatch
- Familiarity with Python async concepts at a surface level

**What they do not yet know:**
- How TTNN internally dispatches work through command queues
- What "trace capture" records and replays
- How async ops pipeline host and device work
- How to decide whether trace will help a given model and how much

---

## Chapter List

### Chapter 1 — TTNN Dispatch Fundamentals
**Description:** Establishes the mental model of how TTNN translates Python op calls into device work, covering the host-side dispatch path before any tracing or async concepts are introduced.

**Directory:** `ch1_dispatch_fundamentals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: host Python call → TTNN runtime → device execution
  - Glossary of terms introduced in this chapter (dispatch, kernel, CQ)

- `host_dispatch_path.md`
  - Walk through what happens between a `ttnn.matmul(...)` call and kernel execution on device
  - Identify the discrete phases: argument validation, kernel selection, command encoding, submission
  - Explain why host dispatch time is non-trivial and appears in profiler traces
  - Introduce the concept of dispatch overhead as a measurable cost

- `command_queues.md`
  - Define a command queue (CQ) as the ordered channel through which the host sends work to the device
  - Explain the role of CQ0 (primary compute) and CQ1 (secondary / data movement or prefetch)
  - Describe how commands are enqueued, how the device consumes them, and the FIFO ordering guarantee
  - Cover the implications of using a single CQ vs dual CQ for throughput and synchronization

---

### Chapter 2 — Asynchronous Op Execution
**Description:** Explains how TTNN ops can be dispatched asynchronously, allowing the host to continue Python execution while the device processes previously enqueued work.

**Directory:** `ch2_async_ops/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: timeline showing host thread and device execution overlapping
  - Recap of synchronous execution as the baseline

- `async_execution_model.md`
  - Define the async op model: `ttnn` calls return control to the host before device execution completes
  - Explain how the runtime tracks in-flight commands and manages dependencies between ops
  - Describe the synchronization points: `ttnn.synchronize_device()`, tensor readback, event barriers
  - Show a concrete example: two sequential matmuls dispatched asynchronously and when they actually run

- `pipelining_host_and_device.md`
  - Explain host-device pipelining: while the device executes op N, the host dispatches op N+1
  - Identify the conditions under which pipelining breaks down (synchronous readbacks, Python control flow on device output)
  - Describe how the decode loop in autoregressive inference is the primary beneficiary
  - Cover the cost of synchronization barriers and when they are unavoidable

---

### Chapter 3 — Trace Capture and Replay
**Description:** Covers the TTNN trace API in detail — what it records, how replay works, and the internal mechanics that make replay faster than live dispatch.

**Directory:** `ch3_trace_capture/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: capture phase vs replay phase side by side
  - Relationship to concepts from Chapters 1 and 2

- `trace_api.md`
  - Introduce `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, and `ttnn.execute_trace`
  - Explain what is recorded: the command sequence, kernel arguments, buffer bindings
  - Explain what is NOT recorded: Python-side logic, tensor shape changes, dynamic control flow
  - Show the minimal code pattern for capturing and replaying a decode step

- `trace_internals.md`
  - Explain how the captured trace is stored: a pre-encoded command buffer on the device
  - Describe why replay bypasses host dispatch overhead (no re-encoding, no kernel re-selection)
  - Cover buffer aliasing: how trace replay reuses the same device memory addresses recorded at capture time
  - Explain the implication for input/output tensors: they must live at the same addresses used during capture

- `trace_constraints.md`
  - List operations that cannot be traced: ops with dynamic shapes, ops that require host-side branching, data-dependent control flow
  - Explain why shape dynamism breaks trace (buffer addresses would differ)
  - Cover the prefill vs decode asymmetry: prefill typically cannot be traced, decode typically can
  - Describe how to structure a model to maximize the traceable region

---

### Chapter 4 — When to Use Trace
**Description:** Provides decision criteria for applying trace to a model, covering the workload patterns that benefit most and those where trace adds complexity without payoff.

**Directory:** `ch4_when_to_use_trace/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Decision flowchart: should I trace this loop?

- `latency_sensitive_workloads.md`
  - Explain why trace is primarily a latency optimization, not a throughput optimization
  - Describe the decode loop as the canonical use case: fixed shapes, repeated execution, host overhead dominates
  - Contrast with prefill: variable sequence length, executed once, host overhead is a small fraction
  - Discuss batch size 1 vs larger batches and how host overhead fraction changes

- `when_not_to_trace.md`
  - Enumerate disqualifying conditions: dynamic shapes per call, ops that read back to host mid-loop, models with Python control flow on device outputs
  - Explain the maintenance cost of trace: captured buffers are pinned, shape changes require re-capture
  - Describe the debugging difficulty: trace replay errors are harder to diagnose than live dispatch errors
  - Provide guidance on splitting a model into a traced inner loop and an untraced outer wrapper

---

### Chapter 5 — Estimating Improvement from Trace
**Description:** Gives engineers a concrete methodology for measuring host dispatch overhead and predicting the latency reduction that trace would provide before committing to the implementation.

**Directory:** `ch5_estimating_improvement/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Overview of the two-phase measurement approach: baseline profiling then overhead isolation

- `measuring_dispatch_overhead.md`
  - Explain how to use Tracy or TTNN's built-in profiler to separate host dispatch time from kernel execution time
  - Describe the key metrics: op dispatch latency per call, total host time per decode step, kernel occupancy
  - Walk through reading a Tracy trace to identify dispatch-dominated ops
  - Provide reference numbers: typical dispatch overhead per op on current hardware generations

- `estimating_trace_speedup.md`
  - Define the speedup model: `speedup = total_step_time / (total_step_time - dispatch_overhead)`
  - Explain what dispatch overhead trace eliminates vs what it cannot eliminate (kernel execution, data movement, host Python)
  - Walk through a worked example: a 32-op decode step with measured overhead, computing expected latency after trace
  - Discuss diminishing returns: when kernel time dominates, trace savings become negligible
  - Provide a checklist for deciding whether measured speedup justifies the implementation cost

- `profiling_workflow.md`
  - End-to-end workflow: instrument a model, run baseline, capture trace, run traced version, compare
  - Describe the exact commands and environment variables needed to enable TTNN profiling
  - Explain how to validate that trace replay is producing numerically identical outputs
  - Cover regression testing: how to detect when a code change invalidates a captured trace

---

### Chapter 6 — Putting It All Together
**Description:** Synthesizes all prior chapters into an annotated reference implementation of a traced autoregressive decode loop, and covers operational concerns like re-capture triggers and error handling.

**Directory:** `ch6_reference_implementation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Map from each prior chapter to the corresponding section of the reference implementation

- `traced_decode_loop.md`
  - Fully annotated code example: model setup, capture phase, decode loop using `execute_trace`
  - Inline comments explaining every non-obvious decision (buffer pinning, sync placement, CQ selection)
  - Highlight the exact lines where host dispatch overhead is eliminated vs retained
  - Show the before/after profiler output for the example model

- `operational_concerns.md`
  - Describe re-capture triggers: when does a trace become invalid (shape change, weight update, device reset)
  - Explain how to detect a stale trace at runtime
  - Cover error handling: what exceptions trace replay raises and how to recover
  - Provide guidance on CI integration: capturing traces as part of a test suite vs capturing at first run

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| CQ | Command Queue — the ordered dispatch channel between host and device |
| CQ0 | The primary command queue used for compute ops |
| CQ1 | The secondary command queue, used for data movement or prefetch when dual-CQ mode is enabled |
| trace | A pre-recorded, pre-encoded command buffer that can be replayed without re-invoking host dispatch |
| capture phase | The single execution of an op sequence during which the trace is recorded |
| replay | Subsequent executions of the same op sequence using the captured trace |
| dispatch overhead | The host-side time spent encoding and submitting commands, excluding kernel execution time |
| async op | An op call that returns to the host before device execution of that op is complete |
| synchronization point | Any API call or operation that forces the host to wait for the device to catch up |
| decode step | One autoregressive token generation step; the primary unit of work targeted by trace |

**Notation:**

- All TTNN API symbols are formatted as inline code: `ttnn.begin_trace_capture`, `ttnn.Tensor`, etc.
- Device memory addresses and buffer identifiers are formatted in `monospace`.
- Profiler output excerpts use fenced code blocks with the `text` language tag.
- Timing values are always expressed in microseconds (us) unless the value exceeds 1000 us, in which case milliseconds (ms) are used with a parenthetical us equivalent.
- Diagrams showing host/device timelines use horizontal swimlanes with the host on top and device on the bottom.

**Formatting rules:**

- Each `.md` file begins with an H1 title matching the file's topic, followed by a one-paragraph orientation that states what the reader will know by the end of the file.
- Code examples are complete and runnable where possible; if a snippet requires surrounding context, that context is shown as a collapsed `<details>` block.
- Every chapter's `index.md` ends with a "What's next" section listing the files in that chapter in reading order.
- Callout blocks use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Example:**`.
- No emoji in any file.

---

## Cross-Chapter Dependencies

```
Chapter 1 (Dispatch Fundamentals)
  - Introduces: CQ, host dispatch path, dispatch overhead, command encoding
  - Required by: all subsequent chapters

Chapter 2 (Async Ops)
  - Depends on: Chapter 1 (CQ model, dispatch path)
  - Introduces: async execution model, synchronization points, host-device pipelining
  - Required by: Chapters 3, 4, 5, 6

Chapter 3 (Trace Capture and Replay)
  - Depends on: Chapter 1 (CQ, command encoding), Chapter 2 (async ops, sync points)
  - Introduces: trace API, capture vs replay phases, buffer aliasing, trace constraints
  - Required by: Chapters 4, 5, 6

Chapter 4 (When to Use Trace)
  - Depends on: Chapter 3 (trace constraints, capture/replay model), Chapter 2 (pipelining, decode loop context)
  - Introduces: decision criteria, workload classification, maintenance cost
  - Required by: Chapter 6 (the reference implementation motivates its choices using Chapter 4 criteria)

Chapter 5 (Estimating Improvement)
  - Depends on: Chapter 1 (dispatch overhead definition), Chapter 3 (what trace eliminates), Chapter 2 (kernel execution time as the non-eliminable floor)
  - Introduces: speedup model, profiling workflow, overhead measurement methodology
  - Required by: Chapter 6 (the profiling workflow section references the tools introduced here)

Chapter 6 (Reference Implementation)
  - Depends on: all prior chapters
  - Synthesizes: CQ selection (Ch1), async dispatch (Ch2), trace API usage (Ch3), applicability decision (Ch4), profiling validation (Ch5)
  - Introduces no new concepts; serves as integration and operational reference
```
