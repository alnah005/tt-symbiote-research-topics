# Plan: ttnn.all_reduce Trace Compatibility

## Audience

**Target reader:** ML engineers contributing to or reviewing the `tt-symbiote` model stack who need to understand whether `ttnn.all_reduce` can safely be used inside a traced linear layer execution on T3K multi-device configurations. They are specifically concerned with `TTNNLinearIColShardedWAllReduced`, which calls `ttnn.all_reduce` with a synchronous Ring topology and no user-supplied semaphore handles.

**What they already know:**
- TTNN Python API: tensor creation, op calls, mesh device initialization, `ttnn.from_torch`, `ttnn.to_device`
- The basic trace API at a surface level: `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace`
- Multi-device concepts: MeshDevice, cluster axis, column/row sharding, CCL operations such as all_gather and reduce_scatter
- The tt-symbiote `TracedRun` pattern and how `@trace_enabled` / `@trace_disabled` decorators gate trace capture in the symbiote module framework
- That `TTNNLinearIColShardedWRowSharded` uses `ttnn.experimental.reduce_scatter_minimal_async` with explicit cycling semaphores, while `TTNNLinearIColShardedWAllReduced` uses `ttnn.all_reduce` without explicit semaphore handles

**What they do not yet know:**
- What `ttnn.all_reduce` does internally (it delegates to `ttnn.experimental.all_reduce_async` with `std::nullopt` semaphores, which in turn takes either a reduce-scatter+all-gather path or a composite all-gather+local-sum path)
- Whether that delegation path is trace-safe
- Whether the absence of caller-supplied semaphores means no semaphore state exists, or that internally managed semaphores are used instead
- What requirements a traced CCL collective must satisfy regarding persistent buffers, fixed buffer addresses, and semaphore initialization
- Whether there are existing test precedents for tracing CCL collectives (including `all_reduce_async`) on T3K or TG

---

## Chapter List

### Chapter 1 — Trace Capture Mechanics on MeshDevice

**Description:** Establishes how `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace` work at the MeshDevice level — what gets recorded, what the replay contract is, and why buffer address stability and semaphore initialization matter for any op inside a trace.

**Directory:** `ch1_trace_mechanics/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Relationship to the broader `TTNNLinearIColShardedWAllReduced` question
  - Reading order for the chapter

- `what_trace_records.md`
  - Explain that `begin_trace_capture` records the sequence of device commands (kernel dispatches, buffer reads/writes, semaphore operations) enqueued on a MeshCommandQueue, not Python-level logic
  - Describe that the trace stores concrete device buffer addresses captured at record time; replay re-issues those exact commands against those exact addresses
  - Explain the warm-up / compile run that must precede capture (as done in `TracedRun._capture_trace`) and why it is necessary for kernel compilation to complete before the trace command stream is recorded
  - Show the three-phase pattern used throughout the codebase: compile run → `begin_trace_capture` → op calls → `end_trace_capture` → `execute_trace`

- `buffer_address_stability.md`
  - Explain the central constraint of trace replay: every buffer touched during capture must live at the same device address during every subsequent replay
  - Define "persistent input buffers" and "persistent output buffers" as the mechanism to satisfy this constraint
  - Show how `TracedRun._capture_trace` allocates persistent input buffers via `ttnn.to_device` before capture, then uses `ttnn.copy` to update their contents at replay time without moving or reallocating them
  - Describe what happens if an op allocates a new output tensor dynamically during replay — the address changes, causing incorrect memory access or silent corruption
  - Explain that weight tensors satisfy this constraint naturally because they are loaded once and kept at a fixed device address for the lifetime of the module

- `semaphore_initialization_and_replay.md`
  - Explain that global semaphores are SRAM locations on device cores; their addresses are fixed at creation time by `ttnn.create_global_semaphore`
  - Describe what the trace records with respect to semaphores: the address of the semaphore and the operations (increment, wait, reset) performed on it during the captured execution
  - Explain the critical re-entry requirement: before each trace replay, any semaphore that was reset to zero during the captured execution must be back at its expected initial value — if the trace ends with a semaphore at zero, replay can proceed; if the semaphore retains a non-zero residual, the replay kernel hangs
  - Distinguish between semaphores whose full lifecycle (set, wait, reset-to-zero) occurs within one trace capture (self-contained) vs semaphores that are set before capture and consumed inside capture (caller-managed initialization required before each replay)
  - Note that `ttnn.all_reduce` passes `std::nullopt` for all semaphore arguments to `ttnn.experimental.all_reduce_async`; the downstream path either creates no semaphores (composite AG+local-sum path) or uses the synchronous `ttnn.reduce_scatter` + `ttnn.all_gather` path, both of which are examined in Chapter 2

---

### Chapter 2 — ttnn.all_reduce Internal Architecture

**Description:** Traces the full call chain from `ttnn.all_reduce` down to the specific code paths it takes on T3K (1×N or N×1 mesh), identifying which sub-operations are invoked, what semaphore state those sub-operations use, and whether that state is compatible with trace capture.

**Directory:** `ch2_all_reduce_internals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary answer to the semaphore question stated upfront, with the detail chapters as supporting evidence

- `call_chain.md`
  - Walk the complete call chain:
    1. Python `ttnn.all_reduce(input, num_links=1, topology=Ring, cluster_axis=1)` in `TTNNLinearIColShardedWAllReduced.forward`
    2. C++ `ExecuteAllReduce::invoke` in `all_reduce.cpp` — passes `std::nullopt` for all three semaphore groups (`barrier_semaphores`, `rs_global_semaphores`, `ag_global_semaphores`)
    3. Delegates to `ttnn::experimental::all_reduce_async` (the `cluster_axis` overload in `all_reduce_async.cpp`)
    4. Inside `ExecuteAllReduceAsync::invoke`: topology resolution via `get_usable_topology`, scatter dimension detection, composite path check, then either the composite all-gather + local-sum path or the reduce-scatter + all-gather path
  - Explain the path-selection predicates: `use_composite_all_gather`, `use_composite_reduce_scatter`, `is_true_2d_mesh`, `dim != composite_dim`
  - State which path T3K (1×8 or 1×4 mesh) takes for typical decode tensors of shape `[1, 1, 32, hidden_dim]` with DRAM interleaved memory config

- `composite_path.md`
  - Describe the composite all-gather + local-sum path selected when the tensor dimensions do not divide cleanly for reduce-scatter, or when a 2D fabric is active
  - Show that this path calls `ttnn.all_gather` (the non-async, no-semaphore version) followed by `ttnn.moreh_sum` (a single-device reduction)
  - Explain that `ttnn.all_gather` in the non-async form uses internal device-side synchronization without caller-visible global semaphore objects — there is no semaphore state the caller must reset between replays
  - Explain that `ttnn.moreh_sum` is a standard single-device compute op with no CCL synchronization
  - Conclude that the composite path is trace-compatible under the same constraints as any non-CCL op: fixed output buffer addresses (because intermediate tensors are allocated dynamically, this path allocates new buffers on each call, which conflicts with trace — note this as a key finding)

- `reduce_scatter_all_gather_path.md`
  - Describe the reduce-scatter + all-gather path selected when `rs_global_semaphores` and `barrier_semaphores` are `std::nullopt`: the code falls through to `ttnn::reduce_scatter` (synchronous, no global semaphore argument) and `ttnn::all_gather` (synchronous, no global semaphore argument)
  - Confirm that the synchronous `ttnn.reduce_scatter` and `ttnn.all_gather` calls used in this fallback do not accept or cycle external global semaphore handles
  - State whether the synchronous CCL ops use local (per-program) semaphores that are re-created on each dispatch and therefore have no persistent state to corrupt between trace captures

- `contrast_with_async_variants.md`
  - Contrast `ttnn.all_reduce` (synchronous, no explicit semaphore args) with `ttnn.experimental.all_reduce_async` (requires caller-supplied `GlobalSemaphore` objects)
  - Contrast with `ttnn.experimental.reduce_scatter_minimal_async` used in `TTNNLinearIColShardedWRowSharded`, which requires explicit cycling semaphore handles from `ccl_manager.get_and_cycle_rs_semaphore_handles`
  - Explain why the cycling semaphore pattern (double-buffered `GlobalSemaphore` objects, alternated across calls) is necessary for async CCL ops inside a trace: the semaphore address is baked into the trace, so the same object must be reused every replay, and double-buffering prevents a slow previous iteration from corrupting the next
  - Confirm that `ttnn.all_reduce` avoids this complexity by passing `std::nullopt`, placing the semaphore lifecycle entirely inside the dispatch layer where it is managed per-program-instance

---

### Chapter 3 — Trace Compatibility Verdict and Requirements

**Description:** Synthesizes the findings from Chapters 1 and 2 into direct answers to the three research questions, states the requirements that must be met for `TTNNLinearIColShardedWAllReduced` to operate correctly under `TracedRun`, and identifies any remaining risks.

**Directory:** `ch3_verdict/`

**Files:**

- `index.md`
  - Chapter overview
  - Table of the three research questions and one-line answers, with pointers to the supporting detail sections

- `q1_trace_compatibility.md`
  - **Question:** Is `ttnn.all_reduce` (synchronous, Ring topology) compatible with trace capture and replay?
  - State the conclusion: `ttnn.all_reduce` delegates to paths that do not hold persistent semaphore state between calls; however, the intermediate tensors allocated inside the composite path are dynamically allocated on every call, which makes the composite path trace-incompatible without additional constraints
  - Identify the conditions under which the reduce-scatter + all-gather fallback (no semaphore args) is taken vs the composite path, and assess whether typical `TTNNLinearIColShardedWAllReduced` inputs on T3K trigger the composite or non-composite path
  - State the requirement: the input tensor shape and memory config must route `ttnn.all_reduce` to the non-composite path, or the intermediate buffers must be pre-allocated as persistent tensors passed in — document which condition applies in practice based on the current code
  - Explain the role of `ttnn.sharded_to_interleaved` called inside `all_reduce_async` when the input is sharded: this allocates an additional intermediate buffer, which must also be at a stable address during replay — note whether `TTNNLinearIColShardedWAllReduced`'s DRAM interleaved output from `ttnn.linear` causes this conversion to be skipped
  - Note the evidence from the existing codebase: `test_new_all_reduce.py` demonstrates `ttnn.experimental.all_reduce_async` (with explicit semaphores) captured in trace mode (`trace_mode=True`) on TG, and `test_ccl_all_reduce.py` demonstrates a custom kernel-based all-reduce traced on a submesh — these confirm that CCL-class operations are traceable in general

- `q2_semaphore_state.md`
  - **Question:** Does `ttnn.all_reduce` use any internal semaphore state that could conflict with trace replay?
  - State that `ttnn.all_reduce` passes `std::nullopt` for all `GlobalSemaphore` arguments; the downstream synchronous `ttnn.reduce_scatter` and `ttnn.all_gather` use per-program local semaphores that are created fresh on each program dispatch
  - Explain that local semaphores (created as part of a `tt_metal::Program`) are re-initialized every time the program is enqueued; they do not survive between trace captures and therefore cannot carry stale state into a replay
  - Contrast with `ttnn.experimental.all_reduce_async` which requires explicitly created `GlobalSemaphore` objects that persist on device and must be correctly initialized before each replay
  - Conclude: `ttnn.all_reduce` introduces no persistent global semaphore state that the caller must manage between trace replays — this is the key advantage of the synchronous variant for trace usage

- `q3_requirements_and_limitations.md`
  - **Question:** Are there any known limitations or requirements for using `ttnn.all_reduce` inside a traced region?
  - List the concrete requirements in order of importance:
    1. **Output buffer stability**: the output tensor of `ttnn.all_reduce` must be allocated before the trace capture begins and reused across all replays; `TTNNLinearIColShardedWAllReduced` must not allocate a new output tensor inside the trace — verify whether `create_output_tensors` inside the device operation re-allocates or reuses the preallocated buffer
    2. **Intermediate buffer stability**: if the reduce-scatter + all-gather path is taken, the intermediate tensor produced by `ttnn.reduce_scatter` must also be at a stable address; check whether the symbiote `TracedRun` warm-up run causes these intermediates to be allocated at stable addresses before capture
    3. **Fabric configuration**: `ttnn.all_reduce` on T3K requires `FABRIC_1D` fabric config; confirm the device is initialized with this config when `TracedRun` captures the trace
    4. **Even ring size**: `AllReduceAsyncDeviceOperation::validate_on_program_cache_miss` asserts `args.ring_size % 2 == 0`; T3K has 8 devices on cluster axis 1 (ring size 8), which satisfies this
    5. **Width-sharded memory layout**: if the minimal async device operation path is reached, it requires `WIDTH_SHARDED` memory layout for input, buffer, and output; `TTNNLinearIColShardedWAllReduced` uses `DRAM_MEMORY_CONFIG` (interleaved), which routes away from this path
  - List known limitations:
    - The composite path (all-gather + local sum) allocates intermediate tensors dynamically and is not trace-compatible without explicit persistent buffer management; if tensor shapes cause composite path selection, the forward method must be restructured or `@trace_disabled` must be applied
    - `ttnn.all_reduce` does not support `float32` on all paths (the `local_sum` helper typecasts bfloat8_b and the float32 variant uses a different reduction); verify the dtype used in `TTNNLinearIColShardedWAllReduced` (bfloat16 from `ttnn.linear` output) is handled by the standard path
    - `TTNNLinearIColShardedWAllReduced` inherits from `TTNNLinearIColShardedWRowSharded` which is not decorated with `@trace_enabled` or `@trace_disabled`; the `@trace_enabled` / `@trace_disabled` decoration on the parent class `TTNNLinear` applies — confirm that `TTNNLinearIColShardedWAllReduced` falls within the `@trace_enabled` set at runtime via `is_trace_enabled` inheritance check

---

### Chapter 4 — Integration Checklist and Test Strategy

**Description:** Provides a concrete, actionable checklist for integrating `TTNNLinearIColShardedWAllReduced` into a traced execution context, along with a minimal test strategy to validate trace correctness before deployment.

**Directory:** `ch4_integration/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives

- `integration_checklist.md`
  - Pre-conditions to verify before enabling trace on `TTNNLinearIColShardedWAllReduced`:
    - [ ] Confirm the device is opened with `FABRIC_1D` fabric config and a non-zero `trace_region_size`
    - [ ] Confirm the input tensor shape routes `ttnn.all_reduce` to the non-composite path (run a debug build and check `log_debug` output for "Using reduce scatter + all gather" vs "Using composite all gather + local reduce")
    - [ ] Confirm `TTNNLinearIColShardedWAllReduced` inherits `@trace_enabled` correctly — add a runtime assertion `assert is_trace_enabled(module_instance)` in test setup
    - [ ] Confirm the warm-up run in `TracedRun._capture_trace` runs to completion and `ttnn.synchronize_device` is called before `begin_trace_capture`
    - [ ] Confirm that the output tensor of the forward pass is captured as `trace_output` in the `TraceEntry` and that subsequent replays read from this fixed buffer
    - [ ] Confirm that no intermediate tensor from inside `ttnn.all_reduce` is referenced after trace capture completes (they will be garbage-collected between replays; only the final output tensor is retained)
  - Post-capture validation steps:
    - Run two consecutive `ttnn.execute_trace` calls on the same trace and compare outputs to golden — if the outputs differ, a semaphore or buffer aliasing issue is present
    - Compare the traced output against a non-traced reference run using `comp_pcc` with threshold 0.999

- `minimal_test_pattern.md`
  - Provide an annotated code skeleton (not a runnable script, but a structured pattern) for a unit test that:
    1. Opens a T3K mesh with `FABRIC_1D` and `trace_region_size=131072`
    2. Instantiates `TTNNLinearIColShardedWAllReduced` with representative `in_features` / `out_features` for a decode step
    3. Creates a column-sharded input tensor of shape `[1, 1, 32, in_features]`
    4. Runs the forward method once outside trace to compile kernels
    5. Calls `ttnn.begin_trace_capture`, runs forward, calls `ttnn.end_trace_capture`
    6. Calls `ttnn.execute_trace` twice and compares outputs
    7. Calls `ttnn.release_trace`
  - Note which steps correspond to which `TracedRun` internals so the test author can confirm the symbiote framework performs each step correctly
  - List the parametrize axes: `in_features` in `{4096, 8192}`, `dtype` in `{bfloat16}`, `cluster_axis` in `{1}`

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| `ttnn.all_reduce` | The stable, synchronous collective operation registered at `ttnn::all_reduce` in `all_reduce.hpp`; passes `std::nullopt` for all semaphore arguments |
| `ttnn.experimental.all_reduce_async` | The experimental async collective; requires caller-supplied `GlobalSemaphore` objects and is used in `TTNNLinearIColShardedWRowSharded` via cycling semaphores |
| trace capture | The phase bounded by `ttnn.begin_trace_capture` and `ttnn.end_trace_capture` during which device commands are recorded |
| trace replay | Each subsequent call to `ttnn.execute_trace`; re-issues the recorded commands without host-side re-dispatch |
| persistent buffer | A device tensor allocated before trace capture and kept at the same address throughout all replays |
| global semaphore | A `GlobalSemaphore` object created by `ttnn.create_global_semaphore`; holds a fixed device L1 address shared across devices in a mesh |
| local semaphore | A semaphore created as part of a `tt_metal::Program`; re-initialized on every program dispatch; not visible at the `ttnn.all_reduce` caller level |
| composite path | The all-gather + `moreh_sum` code path inside `all_reduce_async` selected when dimensions do not divide evenly for reduce-scatter or a 2D fabric is active |
| non-composite path | The reduce-scatter + all-gather code path inside `all_reduce_async`; used when `rs_global_semaphores` is `std::nullopt`, falls back to synchronous `ttnn.reduce_scatter` + `ttnn.all_gather` |
| cycling semaphores | The double-buffered `GlobalSemaphore` pool in `TT_CCL` accessed via `get_and_cycle_*` methods; used by async CCL ops to avoid semaphore aliasing across iterations |
| T3K | An 8-device Wormhole mesh arranged as a 1×8 logical ring on cluster axis 1 |
| `TTNNLinearIColShardedWAllReduced` | The symbiote linear module variant that uses `ttnn.all_reduce` instead of `reduce_scatter_minimal_async` |
| `TracedRun` | The symbiote execution mode class that captures and replays traces; only applies to `@trace_enabled` modules |

**Notation:**

- All TTNN Python API symbols are formatted as inline code: `ttnn.all_reduce`, `ttnn.begin_trace_capture`, etc.
- C++ class names and namespaces are formatted as inline code: `ExecuteAllReduce`, `ttnn::experimental::prim`.
- File paths relative to the `tt-metal` repository root are formatted as inline code: `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`.
- Mathematical expressions use LaTeX: inline as $...$ and display as $$...$$. For example, the speedup from eliminating dispatch overhead is expressed as $S = T_\text{total} / (T_\text{total} - T_\text{dispatch})$.
- All buffer sizes are expressed in bytes unless stated otherwise.

**Formatting rules:**

- Each `.md` file begins with an H1 title matching the file's topic, followed by a one-paragraph orientation that states what the reader will know by the end of the file.
- Code patterns (not required to be runnable) are shown in fenced code blocks with `python` or `cpp` language tags.
- Callout blocks use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Key finding:**`.
- Every chapter's `index.md` ends with a "What's next" section listing the files in that chapter in reading order.
- No emoji in any file.

---

## Cross-Chapter Dependencies

```
Chapter 1 (Trace Capture Mechanics on MeshDevice)
  - Introduces: what trace records, buffer address stability, persistent buffers,
    global vs local semaphore lifecycle, semaphore re-entry requirement
  - Required by: Chapters 2, 3, 4

Chapter 2 (ttnn.all_reduce Internal Architecture)
  - Depends on: Chapter 1 (trace constraints, semaphore definitions)
  - Introduces: call chain from ttnn.all_reduce to sub-operations, composite vs
    non-composite path selection, local semaphore usage in synchronous CCL ops,
    contrast with async variants and cycling semaphore pattern
  - Required by: Chapter 3 (verdict uses the path analysis from Ch2),
    Chapter 4 (checklist items reference the path selection logic from Ch2)

Chapter 3 (Trace Compatibility Verdict and Requirements)
  - Depends on: Chapter 1 (replay contract, buffer stability) and Chapter 2
    (semaphore state analysis, path selection)
  - Introduces: direct answers to the three research questions, concrete
    requirements list, identification of the composite-path risk, dtype and
    memory-layout constraints
  - Required by: Chapter 4 (the checklist is derived from the requirements in Ch3)

Chapter 4 (Integration Checklist and Test Strategy)
  - Depends on: all prior chapters
  - Introduces: no new concepts; synthesizes requirements (Ch3) and internals
    (Ch2) into actionable verification steps and a test skeleton
  - Serves as the operational reference for engineers implementing or reviewing
    the integration
```
