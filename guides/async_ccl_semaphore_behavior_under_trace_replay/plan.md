# Plan: Async CCL Semaphore Behavior Under Trace Replay

## Audience

**Target reader:** ML systems engineers and model framework developers who work with the tt-transformers stack or with the common modules layer (`models/common/modules/tt_ccl.py`) and who need to enable `ttnn.begin_trace_capture` / `ttnn.execute_trace` for models that call async CCL ops (`ttnn.experimental.reduce_scatter_minimal_async`, `ttnn.experimental.all_gather_async`). They have a concrete bug or design question in hand: the host-side semaphore cycling counter advances across calls, but trace replay always reuses the semaphore handle baked in at capture time.

**What they already know:**
- The tt-transformers `Generator` trace API (`begin_trace_capture`, `end_trace_capture`, `execute_trace`) and the compile-capture-replay lifecycle
- The `TT_CCL` class (in `models/tt_transformers/tt/ccl.py` and `models/common/modules/tt_ccl.py`): that it creates double-buffered global semaphore handles, and that `get_and_cycle_*` advances a host-side modular counter and returns the next handle
- That `tt_all_reduce`, `tt_all_gather`, and related helpers in `ccl.py` call `get_and_cycle_*` on every invocation to pass distinct semaphore handles to each CCL op
- The high-level purpose of async CCL: overlapping Ethernet data movement with compute by using pre-allocated global semaphores to signal completion
- Basic ttnn tensor and device APIs; Python-level familiarity with `ttnn.create_global_semaphore` and `ttnn.reset_global_semaphore_value`

**What they do not yet know:**
- Whether semaphore handles are embedded as kernel compile-time arguments or as runtime arguments (RTAs), and the implication of that distinction for trace replay correctness
- Exactly what data is snapshotted into a `TraceNode` at `end_trace_capture` time, and what the `override_runtime_arguments` path does versus what the trace replay path does
- Whether `ttnn.experimental.reduce_scatter_minimal_async` and `ttnn.experimental.all_gather_async` can be placed inside a `begin_trace_capture` / `end_trace_capture` bracket at all, and under what conditions
- What existing patterns (if any) in tt-transformers handle the host-counter / trace-handle mismatch problem
- Whether resetting only the host-side `*_semaphores_idx` counters before each replay is sufficient, or whether device-side semaphore values also need to be reset before each replay
- The precise sequencing required to make semaphore state consistent at both the host and device level across repeated trace replays

---

## Chapter List

### Chapter 1 — GlobalSemaphore Internals and the Double-Buffer Design

**Description:** Explains what a `GlobalSemaphore` is at the tt-metal level — how it maps to an L1 buffer address, how `create_global_semaphore` allocates it, and why double-buffering with a host-side cycling counter was introduced to allow pipelining of async CCL ops.

**Directory:** `ch1_global_semaphore_internals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: relationship between `GlobalSemaphore` Python handle, L1 buffer address, and device-side semaphore word
  - Glossary of terms introduced in this chapter: `GlobalSemaphore`, handle, L1 semaphore address, double-buffer, cycling counter
  - "What's next" section listing files in reading order

- `global_semaphore_api.md`
  - Describe `ttnn.create_global_semaphore(mesh_device, core_range_set, initial_value)` at the Python layer and what it returns: a `GlobalSemaphore` object wrapping a sharded L1 buffer
  - Explain `GlobalSemaphore::address()` (C++, in `tt_metal/api/tt-metalium/global_semaphore.hpp`): returns the L1 address at which each involved core stores the semaphore word; this address is stable for the lifetime of the object
  - Explain `ttnn.reset_global_semaphore_value(semaphore, reset_value)`: issues host-side dispatch commands that write `reset_value` to the L1 address on all cores in the `CoreRangeSet`; this is the mechanism for resetting a semaphore between uses
  - Show where semaphore handles are created in `TT_CCL.__init__`: the triple outer loop over `cluster_axis` variants (0, 1, none) and the inner double-buffer loop of 2; for `ag_semaphore_handles` each double-buffer slot contains a list of 2 handles, for `rs_semaphore_handles` each slot contains 3
  - Note that the address returned by `.address()` is per-device (broadcast across the mesh via sharded buffer allocation) and is fixed at object creation time

- `double_buffer_design.md`
  - Explain the double-buffer motivation: if the same semaphore is used for two back-to-back CCL ops, the second op's kernel may read a stale value left by the first; two alternating handles allow op N+1 to use a fresh semaphore while op N's completion signal is still live
  - Describe the three host-side index arrays (`barrier_semaphore_idx`, `ag_semaphores_idx`, `rs_semaphores_idx`), each of length 3 (one entry per cluster-axis variant), each cycling modulo 2
  - Show the `get_and_cycle_*` methods in full: how `semaphore_index = 2 if cluster_axis is None else cluster_axis` maps `None`/`0`/`1` to array slots `2`/`0`/`1`, then how `current_idx = *_idx[semaphore_index]` is returned while `*_idx[semaphore_index]` is incremented modulo 2
  - Illustrate a 4-call sequence (e.g., two `tt_all_reduce` calls on `cluster_axis=0`): which handle is selected at each call and why both handles must be clean (reset to 0) before reuse
  - Emphasize the key invariant: the host counter and the device's current semaphore state must be consistent before each CCL invocation; double-buffering relaxes the timing requirement but does not eliminate the state dependency

---

### Chapter 2 — How Semaphore Addresses Flow into Kernel Runtime Arguments

**Description:** Traces the path from Python `GlobalSemaphore` handle through the C++ device operation layer into the per-core runtime argument (RTA) arrays, establishing precisely where semaphore addresses are stored and what "baking into the trace" means.

**Directory:** `ch2_semaphore_rta_path/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: Python `GlobalSemaphore` handle → `operation_attributes.semaphore` → `override_runtime_arguments` → per-core RTA slot → dispatch command stream → trace snapshot
  - Recap of Chapter 1 prerequisites (handle identity, L1 address stability)
  - "What's next" section listing files in reading order

- `rta_vs_compile_time_args.md`
  - Distinguish compile-time kernel arguments (embedded in the kernel binary at program factory creation time; never change per-invocation) from runtime arguments (RTAs, written into per-core L1 config space before each dispatch)
  - Explain that global semaphore addresses are passed as RTAs, not compile-time args: show the relevant lines in `all_gather_async_default_program_factory.cpp` where `worker_reader_sender_runtime_args[2] = out_ready_semaphore.address()` and `worker_writer_sender_runtime_args[3] = out_ready_semaphore.address()` are set; and similarly in `reduce_scatter_minimal_async_program.cpp` where `worker_reader_sender_runtime_args[2] = semaphore.at(dir).address()` and `worker_writer_sender_runtime_args[4] = semaphore.at(dir).address()` are written
  - Explain that because semaphore addresses are RTAs, they are governed by the `override_runtime_arguments` mechanism rather than requiring a program cache miss to change them
  - Note that the `barrier_semaphore.value().address()` is also an RTA (e.g., `worker_writer_sender_runtime_args[5]` for AG, `worker_writer_sender_runtime_args[7]` for RS ring); both the multi-device semaphore list and the barrier semaphore follow the same RTA path

- `override_runtime_arguments_flow.md`
  - Explain the device_operation program cache lifecycle: on a cache miss, `create_mesh_workload` builds the program and sets initial RTAs via `SetRuntimeArgs`; on a cache hit, `override_runtime_arguments` re-writes only the per-invocation RTAs (buffer addresses and semaphore addresses) into the existing program command sequence
  - Show how this works for `AllGatherAsync`: `DefaultMeshWorkloadFactory::override_runtime_arguments` in `all_gather_async_default_program_factory.cpp` calls `all_gather_async_minimal_default_helper_override_runtime_arguments`, which iterates over all cores and writes `out_ready_semaphore.address()` and `barrier_semaphore.value().address()` into the live RTA data via `GetRuntimeArgs` + direct assignment
  - Show the same pattern for `ReduceScatterMinimalAsync`: `RingReduceScatterMeshWorkloadFactory::override_runtime_arguments` and `LineReduceScatterMeshWorkloadFactory::override_runtime_arguments` both delegate to helper functions that update `semaphore.at(dir).address()` and `barrier_semaphore.value().address()` per core
  - Emphasize that `override_runtime_arguments` is called on every program cache hit, which includes calls made during a `begin_trace_capture` / `end_trace_capture` bracket

- `trace_node_rta_snapshot.md`
  - Explain the `TraceNode` struct (in `tt_metal/impl/trace/trace_node.hpp`): it stores `rta_data` as a `std::vector<std::vector<uint8_t>>` — a snapshot of every RTA block as it existed at the moment `end_trace_capture` was called
  - Trace the creation path: `create_trace_node` in `tt_metal/impl/program/dispatch.cpp` is called at `end_trace_capture` time; it iterates `cached_program_command_sequence.rta_updates`, copying each `update.src` region into `rta_data`
  - Show how `rta_updates` is populated: the `rta_data` pointer in each `Transfer` object tracks where in the command stream the RTA block lives; when writing the command stream (in the batched write path), `transfer.rta_data->rt_args_data` is updated to point directly into the command stream, so `rta_updates[i].src` is the live command-stream location
  - Explain replay: at each `execute_trace` / `replay_mesh_trace`, the dispatch path calls the trace replay handler which does `std::memcpy(rta_update.dst, trace_node.rta_data[i].data(), rta_update.size)` for each RTA block — restoring the snapshot values into the command stream before submission
  - **Key conclusion:** the semaphore address recorded at `end_trace_capture` time is the one replayed on every subsequent `execute_trace`, regardless of what `get_and_cycle_*` has returned since capture

---

### Chapter 3 — The Host-Counter / Trace-Handle Mismatch

**Description:** Precisely characterizes the inconsistency that arises when the host-side cycling counter advances while trace replay locks the semaphore handle to its capture-time value, explains under what conditions this produces silent corruption vs. deadlock vs. correct behavior, and answers whether the two async CCL ops can be placed inside a trace bracket at all.

**Directory:** `ch3_mismatch_analysis/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: timeline of host counter vs. trace handle across one capture and four replays
  - Recap of Chapters 1–2 prerequisites
  - "What's next" section listing files in reading order

- `what_gets_baked_in.md`
  - Walk through the exact sequence during a decode trace capture that contains one `tt_all_reduce` call with `use_composite=False` (i.e., `all_gather_async` followed by `fast_reduce_nc`):
    - Before capture: `ag_semaphores_idx[cluster_axis]` is at some value N, `barrier_semaphore_idx[cluster_axis]` is at some value M
    - At capture time: `get_and_cycle_ag_semaphore_handles(cluster_axis)` returns `ag_semaphore_handles[cluster_axis][N]` (and advances the idx to (N+1)%2); `get_and_cycle_barrier_semaphore_handle(cluster_axis)` returns `barrier_semaphore_handles[cluster_axis][M]` (and advances M)
    - `override_runtime_arguments` writes the L1 addresses of handle `ag_semaphore_handles[cluster_axis][N]` and `barrier_semaphore_handles[cluster_axis][M]` into the RTA slots
    - `end_trace_capture` snapshots those addresses into `TraceNode.rta_data`
  - After capture: `ag_semaphores_idx[cluster_axis]` is at `(N+1)%2` and `barrier_semaphore_idx[cluster_axis]` is at `(M+1)%2`
  - At replay 1 (`execute_trace`): the snapshot addresses (handle N and M) are restored into the command stream; the host counter remains at `(N+1)%2` and `(M+1)%2` — no Python code runs, so no cycling occurs
  - At replay 2 onward: same as replay 1; every replay uses handle N and M in the RTA slots
  - **Conclusion:** the host counter diverges immediately after the first capture, pointing to the opposite double-buffer slot from what the trace uses

- `traceability_of_async_ccl_ops.md`
  - Answer the question: can `ttnn.experimental.reduce_scatter_minimal_async` and `ttnn.experimental.all_gather_async` be placed inside `begin_trace_capture` / `end_trace_capture`?
  - Explain that at the program dispatch level, both ops use the standard device_operation cache path; the trace recorder captures the resulting command stream entries for these ops exactly as for any other op — there is no architectural barrier preventing their inclusion in a trace bracket
  - Note that the persistent output buffer arguments (passed as `None` in most tt-transformers uses) do not cause a problem: when `None`, the output tensor is allocated fresh at each invocation and its address changes, but `override_runtime_arguments` updates the output buffer address as an RTA on every cache hit, and that updated address is what gets snapshotted into the trace
  - **Critical caveat:** because semaphore addresses are snapshotted into the trace at capture time, any CCL op inside the trace is constrained to use the capture-time handles for all replays; a model is free to place these ops inside a trace bracket as long as it manages semaphore state consistently across the capture-replay boundary

- `failure_modes.md`
  - Describe what happens if the host counter is not reset before replay and no device-side semaphore reset is performed:
    - If the host counter is at `(N+1)%2` (the slot not in the trace), any non-traced CCL call after replay will use handle `(N+1)%2`, while the trace continues to use handle N — two concurrent uses of the same handle are avoided accidentally only if non-traced CCL calls don't happen concurrently with replays
    - If a non-traced `tt_all_reduce` call occurs between two trace replays (e.g., in a prefill-then-decode pattern where prefill is not traced), the counter advances further and the gap between the host index and the trace-baked index widens
  - Describe what happens if device-side semaphore values are not reset before replay:
    - Async CCL kernels on device wait for the semaphore to reach a specific value (typically the ring_size or a completion count); if a previous replay left the semaphore in a non-zero state and the reset was skipped, the waiting kernel on the next replay immediately passes through the wait and proceeds to the next phase, causing data corruption rather than a hang
    - Whether this manifests as silent corruption, a timeout/hang, or an incorrect result depends on the exact kernel logic and the timing of the skip

---

### Chapter 4 — Correct Synchronization Strategies for Traced Async CCL

**Description:** Describes the concrete patterns required to make trace replay work correctly with async CCL semaphores: resetting host-side indices before each replay, resetting device-side semaphore values, and how to structure the trace capture so that the semaphore state at replay time matches the state at capture time.

**Directory:** `ch4_synchronization_strategies/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: correct state machine — reset counters and semaphores → `execute_trace` → wait for completion → reset again
  - Recap of Chapters 1–3 prerequisites
  - "What's next" section listing files in reading order

- `resetting_host_counters.md`
  - Explain that the simplest fix for the host-counter mismatch is to reset all relevant `*_semaphores_idx` fields of the `TT_CCL` instance to the values they held at capture time, immediately before each `execute_trace` call
  - Describe how to determine the capture-time index values: before calling `begin_trace_capture`, record the current index values for every axis variant that the traced model uses; after `end_trace_capture`, those recorded values are the capture-time state
  - Show a code pattern: save indices before capture, restore them before each `execute_trace`
  - Explain why this is necessary even when device-side semaphore values are reset: if the host counter is wrong, the next non-traced CCL call (e.g., a prefill that is not traced) will use the wrong handle, potentially colliding with a trace's in-flight use of the same handle
  - Note that for a purely traced decode loop with no interleaved non-traced CCL calls, the host counter mismatch is harmless to the trace itself (the trace always uses the capture-time addresses) but is harmful to any subsequent non-traced calls

- `resetting_device_semaphore_values.md`
  - Explain that the device-side semaphore value (the uint32 in L1 at `semaphore.address()`) must be reset to 0 before each replay, because async CCL kernels leave it in a non-zero state after completion
  - Show how to call `ttnn.reset_global_semaphore_value(semaphore_handle, 0)` for each handle that the trace uses: the `ag_semaphore_handles[cluster_axis][N]` list (2 handles, each being a list of 2), the `rs_semaphore_handles[cluster_axis][N]` list (2 handles, each being a list of 3), and the `barrier_semaphore_handles[cluster_axis][N]` (2 handles, one per double-buffer slot)
  - Explain the timing: `reset_global_semaphore_value` dispatches a write command into CQ0; it must be enqueued before `execute_trace` so that the reset reaches the cores before the CCL kernels begin; since `execute_trace` with `blocking=False` dispatches to the same CQ, ordering is guaranteed by CQ FIFO semantics
  - Note that resetting only the capture-time handles (index N and M, not index (N+1)%2 and (M+1)%2) is sufficient if non-traced CCL calls are not interleaved with trace replays; if they are interleaved, all handles must be tracked and reset appropriately

- `structuring_the_capture.md`
  - Describe the recommended approach for ensuring capture-time and replay-time semaphore state are consistent:
    1. Reset all semaphore device values to 0 before the compile run (not just before capture), so that the compile run itself uses clean semaphores
    2. Record `TT_CCL` index values after the compile run completes and before `begin_trace_capture`
    3. Reset all semaphore device values to 0 again before `begin_trace_capture`, so that the capture run starts with the same device state as each future replay
    4. After `end_trace_capture`, record the final index values (these are the post-capture state to restore before each replay)
    5. Before each `execute_trace`: reset device semaphore values for all handles used in the trace to 0; restore `TT_CCL` index fields to the capture-time values recorded in step 2
  - Explain why step 3 (reset before capture, not just before replay) is important: the trace snapshot includes RTA slots but not device L1 state; if the capture run leaves device semaphores non-zero and a replay starts from a different initial device state, the kernel behavior will differ from the capture
  - Note that for the `barrier_semaphore` handle, the same reset logic applies: it is passed as an RTA, snapshotted into the trace, and must be at 0 on the device before each replay

- `existing_patterns_in_tt_transformers.md`
  - Investigate whether existing traced code in tt-transformers (`generator.py`, `_capture_decode_trace_text`, `_decode_forward_trace_text`) performs any semaphore management around `execute_trace` calls
  - Document what is found: the current `_capture_decode_trace_text` and `_decode_forward_trace_text` paths do not call `reset_global_semaphore_value` and do not reset `TT_CCL` index fields; this is because the existing decode trace in the current codebase uses non-async CCL variants (barrier-free or composite paths that do not go through `reduce_scatter_minimal_async` / `all_gather_async` with global semaphores)
  - Confirm that `tt_all_reduce` with `use_composite=False` (the TG path) and the direct `all_gather_async` path are the ones that use `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle`; trace support for these paths requires adding explicit semaphore management as described in this chapter
  - Note that the `llama_ccl.py` in `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` and `models/demos/deepseek_v3/tt/ccl.py` follow the same `TT_CCL` pattern and would need the same treatment

---

### Chapter 5 — Implementing Trace Support: A Step-by-Step Guide

**Description:** Consolidates the analysis from all prior chapters into a concrete implementation guide for adding trace support to a module that uses `reduce_scatter_minimal_async` or `all_gather_async`, covering all required code changes, the correct sequence of operations around each trace replay, and how to verify correctness.

**Directory:** `ch5_implementation_guide/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisites: all prior chapters must be read first
  - Diagram: annotated code flow from model initialization through repeated trace replays
  - "What's next" section pointing to the two implementation topic files and then to the worked example

- `code_changes_required.md`
  - Describe the `TT_CCL` changes needed: add methods `snapshot_semaphore_indices()` (returns a copy of all six index values) and `restore_semaphore_indices(snapshot)` (writes them back); add a helper `get_all_handles_for_axis(cluster_axis)` that returns all handles at both double-buffer slots for a given axis variant — useful for looping reset calls
  - Describe changes to the trace capture wrapper (e.g., in `generator.py` or a model-specific trace helper):
    - Before compile run: call `ttnn.reset_global_semaphore_value` for all handles that will be used in the trace
    - After compile run, before `begin_trace_capture`: call `snapshot_semaphore_indices()` to record capture-time state; call `reset_global_semaphore_value` again for all handles
    - After `end_trace_capture`: the snapshot from before capture is the set of indices to restore at each replay
  - Describe changes to the trace replay wrapper:
    - Before each `execute_trace`: call `reset_global_semaphore_value` for each handle used in the trace (identified by the captured index values); call `restore_semaphore_indices(snapshot)` to reset host counter
    - Call `ttnn.execute_trace` with `blocking=False` as usual
    - If a subsequent non-traced CCL call is required (e.g., sampling or embedding not covered by the trace), it will use the handles at the now-restored indices — which are the capture-time slots; note that the device values for those slots were just reset, so they are safe to reuse
  - Show how to identify which handles are actually used in the trace: it is the set of handles selected by `get_and_cycle_*` during the `begin_trace_capture` / `end_trace_capture` bracket; this can be determined by inspecting the index values before and after capture, then collecting the handles at each pre-capture index

- `verifying_correctness.md`
  - Describe a minimal test pattern: run the model with trace enabled for N decode steps; compare the output tensors to a reference run with trace disabled; any numerical difference indicates a semaphore state bug
  - Explain how to detect the deadlock case: if a CCL kernel hangs waiting for a semaphore that is never reset, `ttnn.execute_trace` with `blocking=True` will stall indefinitely; use a timeout wrapper or the Tenstorrent watchdog to detect this
  - Explain how to detect the silent corruption case: if the semaphore skip-through case occurs (stale non-zero value from a previous replay), results will be numerically wrong but the op will complete; only comparing against a reference run exposes this
  - Describe the use of `ttnn.reset_global_semaphore_value` in a debug loop: after each replay, read back the semaphore value (via `ttnn.from_device` on a single-element tensor written by a debug kernel, or by inspecting L1 via watcher) to verify it was reset to 0 before the next replay began
  - Note that the correctness of the host-counter reset can be verified independently by printing the `TT_CCL` index fields before and after each `execute_trace` call and confirming they match the expected capture-time values

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| `GlobalSemaphore` | A tt-metal object wrapping a sharded L1 buffer that acts as a 32-bit counter; its L1 address is stable for the object's lifetime |
| handle | A single `GlobalSemaphore` object, as returned by `ttnn.create_global_semaphore`; identified by its L1 address |
| double-buffer slot | One of the two alternating `GlobalSemaphore` objects at a given axis variant; indexed by the host-side cycling counter modulo 2 |
| cycling counter | The host-side integer (`barrier_semaphore_idx`, `ag_semaphores_idx`, or `rs_semaphores_idx`) that determines which double-buffer slot the next `get_and_cycle_*` call returns |
| RTA | Runtime argument: a per-core scalar written into L1 config space before each program dispatch; semaphore addresses are passed as RTAs |
| `TraceNode` | The internal struct that stores a snapshot of RTA blocks and CB configs at `end_trace_capture` time; these snapshots are restored before each replay |
| capture-time handle | The specific `GlobalSemaphore` object whose address was snapshotted into the `TraceNode`; always used by the trace on every replay |
| program cache hit | The code path in which `override_runtime_arguments` is called to update RTAs for an already-compiled program; this path executes during `begin_trace_capture` and causes the current handle addresses to be recorded into the trace |
| replay | A call to `ttnn.execute_trace`; dispatches the snapshotted command buffer without running any Python or host dispatch logic |
| device-side semaphore value | The uint32 word at `semaphore.address()` in device L1; written by async CCL kernels to signal completion and must be reset to 0 before reuse |
| `reset_global_semaphore_value` | The TTNN API that dispatches a write command setting the L1 semaphore word to a given value; must be enqueued before `execute_trace` to take effect before the CCL kernels run |

**Notation:**

- All TTNN Python API symbols are formatted as inline code: `ttnn.create_global_semaphore`, `ttnn.reset_global_semaphore_value`, `ttnn.execute_trace`, etc.
- All C++ class/struct names are formatted as inline code: `GlobalSemaphore`, `TraceNode`, `ReduceScatterMinimalAsyncDeviceOperation`, etc.
- File paths are given relative to the tt-metal repository root and formatted as inline code paths: `models/tt_transformers/tt/ccl.py`, `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`, etc.
- Field names on Python objects use dot notation in inline code: `TT_CCL.ag_semaphores_idx`, `TT_CCL.ag_semaphore_handles`, etc.
- When describing the three cluster-axis variants (0, 1, None), use the exact Python values: `cluster_axis=0`, `cluster_axis=1`, `cluster_axis=None`; and the corresponding array slot indices: `semaphore_index=0`, `semaphore_index=1`, `semaphore_index=2`
- Math for the modular counter uses the form `(N+1) % 2` with explicit N for the pre-call index value
- Diagrams use a timeline format with host state on one row and device state on a second row, separated by a vertical line at each operation boundary
- Callout blocks use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Key insight:**`
- No emoji in any file
- Each `.md` file begins with an H1 title and a one-paragraph orientation stating what the reader will know by the end of the file
- Every chapter's `index.md` ends with a "What's next" section listing files in reading order

**Formatting rules:**

- Code snippets illustrating RTA slot indices (e.g., `worker_reader_sender_runtime_args[2]`) must include a comment identifying the semantic: `// out_ready_semaphore L1 address`
- When citing source files, include the function or struct name so the reader can navigate directly: "see `ring_reduce_scatter_minimal_async_helper_override_runtime_arguments` in `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`"
- All reset-before-replay steps are presented as a numbered list so they can be used as a checklist
- LaTeX math is not needed in this guide; all arithmetic is simple modular arithmetic expressible in inline code

---

## Cross-Chapter Dependencies

```
Chapter 1 (GlobalSemaphore Internals and the Double-Buffer Design)
  - Introduces: GlobalSemaphore handle, L1 semaphore address, address() stability,
                reset_global_semaphore_value, double-buffer slots, cycling counter,
                get_and_cycle_* methods, TT_CCL index arrays and handle arrays
  - Required by: all subsequent chapters

Chapter 2 (How Semaphore Addresses Flow into Kernel Runtime Arguments)
  - Depends on: Chapter 1 (handle identity, L1 address, cycling counter)
  - Introduces: RTA vs. compile-time arg distinction, override_runtime_arguments
                mechanism, RTA slot indices in AG and RS factories,
                TraceNode rta_data snapshot, create_trace_node, replay memcpy path
  - Required by: Chapter 3 (the snapshot mechanism is why addresses are baked in),
                 Chapter 4 (reset and restore strategies depend on knowing what is baked),
                 Chapter 5 (implementation must address both RTA snapshot and device state)

Chapter 3 (The Host-Counter / Trace-Handle Mismatch)
  - Depends on: Chapter 1 (cycling counter, double-buffer design),
                Chapter 2 (RTA snapshot at end_trace_capture, override_runtime_arguments)
  - Introduces: the divergence between host counter state and trace-baked handle after capture,
                traceability of reduce_scatter_minimal_async and all_gather_async,
                persistent output buffer non-issue, failure modes (silent corruption vs. hang)
  - Required by: Chapter 4 (the mismatch characterization motivates each reset step),
                 Chapter 5 (implementation checklist is derived from the failure modes)

Chapter 4 (Correct Synchronization Strategies for Traced Async CCL)
  - Depends on: Chapter 1 (reset_global_semaphore_value API),
                Chapter 2 (which handles are baked in and which RTA slots they occupy),
                Chapter 3 (the mismatch problem being solved and the failure modes avoided)
  - Introduces: host-counter reset (snapshot/restore pattern), device-side semaphore value
                reset (reset_global_semaphore_value before execute_trace), capture
                structuring steps, existing pattern audit in tt-transformers and llama_ccl.py
  - Required by: Chapter 5 (implementation guide references all strategies from Chapter 4)

Chapter 5 (Implementing Trace Support: A Step-by-Step Guide)
  - Depends on: all prior chapters
  - Synthesizes: handle allocation and addressing (Ch1), RTA snapshot mechanism (Ch2),
                 mismatch characterization and failure modes (Ch3), reset and restore
                 strategies and capture structuring (Ch4)
  - Introduces no new concepts; provides an integrated implementation checklist and
    verification methodology
```
