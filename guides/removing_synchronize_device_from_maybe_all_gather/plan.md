# Plan: Removing synchronize_device from _maybe_all_gather in Hybrid Attention Modules

## Audience

**Target reader:** An ML systems engineer or framework developer who maintains or extends the tt-symbiote attention module stack (`TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, and related modules) and who needs to enable end-to-end Metal Trace capture across the full hybrid DeltaNet + full-attention decoder stack on T3K (1×8 Wormhole mesh).

**What they already know:**

- The tt-symbiote module hierarchy: `TTNNModule`, `LayerStack`, `TracedRun`, and the `@trace_enabled` / `@trace_disabled` decorator system for gating per-module trace participation
- The three-phase trace lifecycle (compile run → `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` → `ttnn.execute_trace`) and the requirement that no host-side Python logic, readback, or synchronization may occur between `begin_trace_capture` and `end_trace_capture`
- The `TT_CCL` class from `models/tt_transformers/tt/ccl.py`: double-buffered `GlobalSemaphore` handles, `get_and_cycle_ag_semaphore_handles`, `get_and_cycle_barrier_semaphore_handle`, and why the cycling pattern exists
- `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` at a surface level — that they accept explicit `GlobalSemaphore` handles and operate without host-blocking
- That `ttnn.synchronize_device(mesh_device)` is a host-blocking call that flushes the device command queue and waits for all in-flight ops to complete before returning to Python

**What they do not yet know:**

- Why `_maybe_all_gather` in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` calls `ttnn.synchronize_device()` — the original motivating reason (race condition workaround, ordering guarantee, or debugging artifact)
- Whether TTNN's single command-queue dispatch ordering provides the same guarantee as `synchronize_device`, eliminating the need for it
- The exact async CCL pattern used in `models/tt_transformers/tt/attention.py` and `models/tt_transformers/tt/ccl.py` for all_gather inside traced decode — and whether `_maybe_all_gather` can adopt that pattern directly
- Which other modules in the tt-symbiote codebase contain `ttnn.synchronize_device()` calls inside their `forward` method that would also block full-stack trace capture
- The measured host-blocking latency of `ttnn.synchronize_device()` at decode batch=1 on T3K, and the expected throughput improvement from removing it
- What validation methodology (numerical correctness tests, race-condition stress tests, PCC checks) is required to confirm the async all_gather pipeline is correct after removing the synchronize call

---

## Chapter List

---

### Chapter 1 — `_maybe_all_gather`: Role, Call Sites, and the synchronize_device Call

**Description:** Establishes exactly what `_maybe_all_gather` does, where it is called in both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`, what the `ttnn.synchronize_device()` inside it is intended to guarantee, and why that guarantee is incompatible with Metal Trace capture.

**Directory:** `ch1_maybe_all_gather_anatomy/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation links to each section file in reading order
  - One-paragraph summary of the core problem: `_maybe_all_gather` is a shared helper that conditionally runs a tensor-parallel all_gather, but its embedded `ttnn.synchronize_device()` drains the device command queue on every call, creating a host synchronization barrier that Metal Trace capture cannot record
  - Glossary of terms introduced in this chapter: `_maybe_all_gather`, `ttnn.synchronize_device`, host-blocking call, device command queue, Metal Trace capture boundary
  - "What's next" section listing the files in reading order

- `call_sites_and_control_flow.md`
  - Walk the complete call path in `TTNNQwen3FullAttention.forward`: where `_maybe_all_gather` is invoked (before or after QKV projection, before SDPA, after output projection), what tensor is passed in, and what the returned tensor's shape and memory config are expected to be
  - Walk the equivalent call path in `TTNNQwen3LinearAttention.forward`: how the linear attention (DeltaNet) variant invokes `_maybe_all_gather`, whether the same tensor shapes and memory configs apply, and whether the call is on the same code path or a separate branch
  - Identify all other modules in tt-symbiote that share or call `_maybe_all_gather` — specifically note if it is a method on a base class or a standalone function
  - Note whether `_maybe_all_gather` is gated on a multi-device flag (e.g., only runs when `num_devices > 1`) so the synchronize call is also conditionally executed

- `synchronize_device_semantics.md`
  - Explain what `ttnn.synchronize_device(mesh_device)` does at the TTNN/tt-metal level: it enqueues a host-wait that blocks until the device command queue is empty and all submitted kernels have completed
  - Explain why this is a host-blocking call: Python execution halts until the device drains; no new TTNN ops can be submitted during the wait; any pending host work is also blocked
  - State the two plausible reasons `synchronize_device` could be inside `_maybe_all_gather`:
    1. To ensure a preceding async op (e.g., a prior `all_gather_async` or compute kernel) has produced its output before the all_gather in `_maybe_all_gather` reads that tensor — a sequencing concern
    2. As a debugging or stability measure inserted during development to avoid non-deterministic failures, left in production code by mistake
  - Explain that in a single-command-queue (CQ0) TTNN dispatch model, op ordering is guaranteed by FIFO queue semantics: op N cannot begin executing before op N-1 has completed or its output is available; `synchronize_device` adds no sequencing guarantee beyond what the command queue already provides
  - Note that the research question of which scenario applies (race condition workaround vs. debugging artifact vs. necessary sequencing) must be answered by code archaeology and, if the code history is unclear, by running without the call under controlled conditions

- `why_this_blocks_trace_capture.md`
  - Explain the Metal Trace capture contract: between `ttnn.begin_trace_capture` and `ttnn.end_trace_capture`, the host must only enqueue TTNN ops; any call that causes host-side Python to wait for device completion violates the recording invariant because the trace buffer records only device-side commands, not the host-side wait
  - Explain what happens if `ttnn.synchronize_device()` is called during a trace capture bracket: the call blocks the host, but the resulting command stream recorded into the trace does not include the synchronization barrier — so on replay, the barrier is absent and any sequencing guarantee it provided is lost
  - Clarify whether `ttnn.synchronize_device()` is silently dropped by the trace recorder, raises an error, or causes undefined behavior — document which behavior is observed in practice when this is attempted
  - Explain the scope of the problem: if `_maybe_all_gather` is called from within a `LayerStack` iteration that is enclosed in a `TracedRun.capture`, every layer in the stack that invokes `_maybe_all_gather` would hit this barrier, making the entire layer stack non-traceable

---

### Chapter 2 — The Async CCL Pattern in tt-transformers for Traced Decode

**Description:** Documents the exact pattern used by `models/tt_transformers/tt/attention.py` and `ccl.py` to call `all_gather_async` inside traced decode regions, covering cycling semaphores, persistent output buffers, and the absence of any `synchronize_device` in that path, so the reader has a working reference model to adapt for `_maybe_all_gather`.

**Directory:** `ch2_async_ccl_pattern/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: the decode trace loop in tt-transformers — compile run, capture bracket, repeated `execute_trace` — with annotation of where `all_gather_async` appears and where no synchronize_device call appears
  - Recap of Chapter 1 prerequisites (why synchronize_device is incompatible, what it would need to be replaced with)
  - "What's next" section listing the files in reading order

- `all_gather_async_in_traced_attention.md`
  - Walk the `Attention.forward` decode path in `models/tt_transformers/tt/attention.py` for the non-TG, Linear-topology case: the call to `ttnn.experimental.all_gather_async` at line ~570 and the fused `all_gather_matmul_async` variant at line ~552
  - Show the complete argument list for `all_gather_async`: `persistent_output_buffer=None`, `multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles()`, `barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle()`, `num_links`, `topology`, `memory_config`, `chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`
  - Confirm that there is no `ttnn.synchronize_device()` call anywhere in the `Attention.forward` decode path — the async CCL op and all subsequent compute ops are enqueued to the same CQ0 without a host barrier
  - Explain why this works without a synchronize: `all_gather_async` completes asynchronously; the downstream `ttnn.linear` call is enqueued to the same command queue and will not begin executing until the all_gather has delivered its output, because the queue is FIFO and the all_gather's output buffer is the input to the linear op
  - Note that `persistent_output_buffer=None` causes the runtime to allocate a fresh output buffer on the first call (compile run) and reuse it on subsequent calls via the program cache — this is the mechanism that provides buffer address stability across trace replays

- `cycling_semaphore_mechanics.md`
  - Explain the double-buffer design in `TT_CCL.__init__`: three axis variants (cluster_axis=0, 1, None), two slots per variant for `ag_semaphore_handles`, `rs_semaphore_handles`, and `barrier_semaphore_handles`
  - Show `get_and_cycle_ag_semaphore_handles(cluster_axis)` and `get_and_cycle_barrier_semaphore_handle(cluster_axis)`: the modular index cycling that alternates between slot 0 and slot 1 on each call
  - Explain why cycling is needed inside a trace: the `GlobalSemaphore` object's L1 address is baked into the trace at capture time; if the same handle were reused for two consecutive `all_gather_async` calls in a pipelined loop, the second call could read a stale completion signal left by the first; double-buffering prevents aliasing between consecutive iterations
  - Note that for `_maybe_all_gather` to use `all_gather_async`, it must have access to a `TT_CCL` instance (or equivalent semaphore pool) — this is a key structural change required relative to the current implementation, which presumably does not hold cycling semaphore handles

- `persistent_output_buffer_contract.md`
  - Explain the "persistent output buffer" contract for trace-safe ops: the output tensor of any op inside a trace must be allocated at the same device address on every replay; if the op allocates a new buffer on each call, the trace's baked-in address becomes stale
  - Show how `ttnn.experimental.all_gather_async` with `persistent_output_buffer=None` satisfies this: on the first call (compile run), the op allocates a buffer and caches its address in the program cache entry; on subsequent calls (capture run and all replays), `override_runtime_arguments` updates the output pointer from the cached entry, ensuring address stability
  - Contrast with an op that returns a freshly allocated tensor on every call without program caching — such an op would break trace replay
  - Explain what `_maybe_all_gather` must do to satisfy this contract: either it passes a pre-allocated persistent output buffer, or it uses an op (such as `all_gather_async`) whose program-cache mechanism provides the stability guarantee automatically
  - Note that if `_maybe_all_gather` currently wraps a synchronous `ttnn.all_gather` (not `all_gather_async`), the persistent buffer question for that call must also be investigated

---

### Chapter 3 — Root Cause Analysis: Why synchronize_device Is Present

**Description:** Investigates the original reason `ttnn.synchronize_device()` was added to `_maybe_all_gather`, determines whether it provides a guarantee that cannot be obtained from command-queue ordering alone, and gives a definitive answer to whether the call is removable or must be replaced by a lighter-weight mechanism.

**Directory:** `ch3_root_cause_analysis/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Recap of Chapters 1–2 prerequisites (what synchronize_device does, what async CCL achieves without it)
  - "What's next" section listing files in reading order

- `what_all_gather_variant_is_used.md`
  - Determine whether `_maybe_all_gather` currently calls the synchronous `ttnn.all_gather`, the async `ttnn.experimental.all_gather_async`, or some other variant — the synchronize_device's necessity depends entirely on which variant is used
  - If it calls synchronous `ttnn.all_gather`: explain that the synchronous form internally uses local (per-program) semaphores that are managed by the tt-metal dispatch layer; the host does not see these semaphores and there is no persistent state that can race; the `synchronize_device` call in this case is strictly unnecessary from a correctness standpoint because the synchronous call already blocks until the operation is queued (not until it completes — clarify the distinction between "queued" and "completed" for TTNN operations)
  - If it calls `ttnn.experimental.all_gather_async`: explain that the async form requires the caller to manage semaphore lifecycle; if the semaphores were not correctly cycled or reset, the `synchronize_device` may have been added as a blunt instrument to ensure the semaphore is in a known state before the next use — but this would be a bug in semaphore management, not a requirement of the async op itself
  - Document the exact call signature of `_maybe_all_gather`'s internal all_gather call as found in the source, including memory config, num_links, and cluster_axis arguments

- `command_queue_ordering_guarantee.md`
  - Explain the TTNN single-CQ ordering model in detail: all ops submitted to CQ0 are executed in submission order; an op cannot begin until the op that produced its input tensor has delivered that output to the agreed-upon device address
  - Show that this ordering guarantee holds even for async CCL ops (`all_gather_async`, `reduce_scatter_minimal_async`): the device-side semaphore mechanism within the CCL kernel ensures that the output buffer is valid before any kernel reading from that buffer is dispatched — and since downstream kernels are submitted to the same CQ0 after the CCL op, the dispatch-level ordering ensures they will not execute until the CCL op is complete
  - Conclude: in a single-CQ deployment (the only mode compatible with trace), `ttnn.synchronize_device()` provides no ordering guarantee beyond what CQ0 ordering already provides; it adds host-blocking latency without contributing to correctness
  - Identify any exception: if the code previously used multi-CQ dispatch (CQ0 for compute, CQ1 for CCL), `synchronize_device` might have been used to synchronize across queues; confirm whether multi-CQ is ever used in the tt-symbiote attention modules

- `verdict_is_it_removable.md`
  - State the definitive answer: `ttnn.synchronize_device()` in `_maybe_all_gather` is removable if and only if (a) the underlying all_gather variant's output ordering is guaranteed by CQ0 FIFO semantics, and (b) no multi-CQ dispatch or cross-queue dependency exists in the surrounding code
  - State the alternative: if `_maybe_all_gather` currently uses synchronous `ttnn.all_gather` and the intent is to switch to `ttnn.experimental.all_gather_async` for trace compatibility, then `synchronize_device` must be removed and the cycling semaphore pattern from `TT_CCL` must be adopted in its place — this is the path that makes `_maybe_all_gather` both trace-compatible and latency-optimal
  - Identify the structural change required: `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` (or their shared base class) must hold a reference to a `TT_CCL` instance so that `_maybe_all_gather` can call `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle`

---

### Chapter 4 — Symbiote-Wide Audit: Other synchronize_device Calls That Block Trace

**Description:** Surveys the entire tt-symbiote module codebase for other `ttnn.synchronize_device()` calls that appear inside module `forward` methods, documents their location and probable purpose, and assesses whether each one also blocks full-stack trace capture.

**Directory:** `ch4_symbiote_audit/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Explanation of scope: this chapter surveys only calls inside forward-path code (`forward`, `_forward`, `__call__`, helper methods invoked during forward) — not calls in test setup, profiling harnesses, or weight loading routines
  - Summary table (to be filled by the writer): module name, file path, call location, probable purpose, trace-blocking status
  - "What's next" section listing files in reading order

- `audit_methodology.md`
  - Describe the search procedure: `grep -rn "ttnn.synchronize_device\|synchronize_devices" <symbiote_source_root>` filtered to files under `models/` and `tt_symbiote/` (excluding `tests/`, `scripts/`, `benchmarks/`)
  - Explain how to distinguish a forward-path call from a non-forward-path call: forward-path calls appear inside a class method that is reachable from `Module.forward` or `TracedRun.__call__`; calls inside `__init__`, `move_weights_to_device`, or `warmup` are not forward-path and do not block trace
  - Explain how to assess whether each forward-path call is inside a `@trace_enabled` module or a `@trace_disabled` module: only calls inside `@trace_enabled` modules matter for full-stack trace capture; calls inside `@trace_disabled` modules are a separate concern
  - Note that `ttnn.synchronize_device()` and `ttnn.synchronize_devices()` (plural) are both blocking and both must be flagged

- `audit_results.md`
  - Present the complete findings: for each `synchronize_device` / `synchronize_devices` call found in forward-path code, document:
    - Module class name and file path (relative to tt-symbiote repo root)
    - Method name and approximate line number
    - The preceding op or condition that the call appears to guard
    - The `@trace_enabled` / `@trace_disabled` status of the enclosing module
    - Whether removing the call is straightforward (CQ ordering suffices) or requires a structural change (e.g., adopting async CCL with cycling semaphores)
  - Call out `TTNNQwen3FullAttention._maybe_all_gather` and `TTNNQwen3LinearAttention._maybe_all_gather` explicitly as the primary subjects of this guide, and note that they were the motivation for the audit
  - Note any calls that are inside `@trace_disabled` modules but that would need to move into `@trace_enabled` territory as the trace-enablement project progresses

- `prioritization.md`
  - Rank the found calls by urgency for full-stack trace capture: which modules are on the critical path for the hybrid DeltaNet + full-attention decoder stack, which are in side paths or optional modules
  - Identify which calls can be addressed alongside the `_maybe_all_gather` fix (same PR) and which require separate investigation
  - Note any calls that appear to be load-time or warm-up artifacts that will be removed as a side effect of other cleanup work

---

### Chapter 5 — Latency Cost of synchronize_device and Throughput Impact

**Description:** Quantifies the host-blocking latency introduced by `ttnn.synchronize_device()` in `_maybe_all_gather` at decode batch=1 on T3K, expresses this as a fraction of total decode step time, and estimates the throughput improvement from removing it.

**Directory:** `ch5_latency_analysis/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Recap of what has been established: the call is unnecessary for correctness (Chapter 3) and is present in multiple modules (Chapter 4); this chapter answers "how much does it cost?"
  - "What's next" section listing files in reading order

- `synchronize_device_latency_model.md`
  - Explain what contributes to `ttnn.synchronize_device()` latency at decode batch=1: the host must wait for all in-flight device commands to complete; the primary component is the time for the last submitted kernel to finish plus PCIe round-trip for the completion signal
  - Estimate the latency components: PCIe round-trip from host to device and back (approximately 10–30 µs for Wormhole on T3K based on known PCIe latency); kernel completion time if any long-running kernel is in flight when `synchronize_device` is called; host scheduling jitter
  - Explain that in the context of `_maybe_all_gather`, the preceding op is an all_gather (either synchronous or async); if that all_gather is already complete before the host reaches the `synchronize_device` call, the wait time is only the PCIe round-trip; if the all_gather is still running, the wait includes the all_gather's remaining execution time
  - Note that at decode batch=1, device kernels are short (typically < 50 µs); the PCIe overhead of `synchronize_device` can dominate, making it a significant fraction of the per-step time

- `measuring_the_cost.md`
  - Describe the Tracy-based measurement procedure: run `TTNNQwen3FullAttention` decode with `TT_METAL_DEVICE_PROFILER=1` and `TT_METAL_PROFILER_TRACE_TRACKING=1`; identify the wall-clock gap between the last enqueued op and the first op after `synchronize_device` returns; this gap is the synchronize cost
  - Alternatively, describe a Python-level `time.perf_counter()` bracket around the `synchronize_device` call as a simpler first approximation
  - Explain how to use TTNN op timer infrastructure (`TT_METAL_PROFILER_SYNC=1`) to isolate the call
  - Provide an expected order-of-magnitude estimate: at decode batch=1 on T3K, `ttnn.synchronize_device()` costs approximately 0.1–0.5 ms per call, which at 2 calls per layer (one for each attention module) and N hybrid layers amounts to N × 0.2–1.0 ms of wasted time per decode step
  - Note how this figure should be read from actual profiling output rather than estimated: the guide instructs the researcher to fill in measured values

- `throughput_improvement_estimate.md`
  - Explain the decode throughput model: tokens per second = 1 / (per-step latency); removing K calls to `synchronize_device` per step saves K × T_sync ms, reducing per-step latency and increasing throughput proportionally
  - For the hybrid DeltaNet + full-attention model with H hybrid attention layers each calling `_maybe_all_gather` once: the total saving per step is H × T_sync; express this as a percentage improvement given a reference per-step latency
  - Note that the improvement compounds with the broader trace-enablement project: once the attention stack is fully captured under Metal Trace, the additional savings from eliminating host dispatch overhead across all ops (not just `synchronize_device`) dominate; removing `synchronize_device` is a prerequisite, not the final performance goal
  - Provide a worked example: assume H = 16 hybrid attention layers, T_sync = 0.3 ms per call, total decode step = 30 ms; the synchronize overhead is 16 × 0.3 = 4.8 ms, a 16% reduction in latency upon removal

---

### Chapter 6 — Implementation Plan: Removing synchronize_device and Adopting async CCL

**Description:** Provides the concrete, step-by-step code changes needed to remove `ttnn.synchronize_device()` from `_maybe_all_gather` and replace the underlying all_gather with `ttnn.experimental.all_gather_async` using cycling semaphores, making the method trace-compatible.

**Directory:** `ch6_implementation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisites: all prior chapters; the reader must understand the cycling semaphore pattern (Chapter 2), the root cause verdict (Chapter 3), and the audit results (Chapter 4) before implementing changes
  - Diagram: before and after — `_maybe_all_gather` without cycling semaphores and with `synchronize_device` vs. `_maybe_all_gather` with `TT_CCL` and `all_gather_async` and no synchronize
  - "What's next" section listing the two topic files in reading order

- `structural_changes.md`
  - Describe the `TT_CCL` wiring change: `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` (or their shared base class) must accept a `tt_ccl: TT_CCL` parameter and store it as `self.tt_ccl`; or alternatively, a lightweight per-module semaphore pool (two `GlobalSemaphore` handles for all_gather, two for the barrier) can be created in `__init__` if a shared `TT_CCL` instance is not available in the tt-symbiote construction path
  - Explain the trade-off: a shared `TT_CCL` instance ensures semaphore handles are not over-allocated (important for L1 space), but requires the parent `LayerStack` or model to thread the instance through the constructor chain; per-module pools are self-contained but multiply the semaphore count by the number of layers
  - Describe the `_maybe_all_gather` signature change: add `cluster_axis` parameter (if not already present) to select the correct semaphore pool slot; ensure the method signature remains compatible with both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` call sites
  - Show the `all_gather_async` argument list for `_maybe_all_gather`, matching the pattern from `Attention.forward` in `models/tt_transformers/tt/attention.py`: `persistent_output_buffer=None`, `multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis)`, `barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)`, with `num_links`, `topology`, and `memory_config` matching the existing configuration
  - Show the removal of `ttnn.synchronize_device()` — a one-line deletion after the structural changes are in place
  - Note that the trace capture wrapper (`TracedRun._capture_trace` or equivalent in tt-symbiote) must perform semaphore index snapshot-and-restore around each `execute_trace` call, following the pattern described in the `async_ccl_semaphore_behavior_under_trace_replay` guide — reference that guide explicitly rather than repeating its content here

- `trace_capture_wrapper_changes.md`
  - Describe the changes needed in the `TracedRun` capture and replay logic to account for the cycling semaphore indices now used inside `_maybe_all_gather`:
    - Before `begin_trace_capture`: record the current `ag_semaphores_idx` and `barrier_semaphore_idx` values for the `cluster_axis` variant used by `_maybe_all_gather`; call `ttnn.reset_global_semaphore_value(handle, 0)` for the capture-time handles
    - After `end_trace_capture`: store the recorded pre-capture index values as the "restore point" for each subsequent replay
    - Before each `execute_trace`: call `ttnn.reset_global_semaphore_value` for each handle used in the trace; restore the `TT_CCL` index fields to the pre-capture values
  - Explain why this is necessary: the cycling counter advances during capture, so each `execute_trace` replay must present the same handle addresses that were baked into the trace at capture time
  - Provide the exact sequence as a numbered checklist so it can be used as a code review reference
  - Note that this wrapper logic is analogous to what the `async_ccl_semaphore_behavior_under_trace_replay` guide covers in detail for `Attention` in tt-transformers; the only new element is that `_maybe_all_gather`'s cluster_axis may differ from the axis used by the model's outer reduce_scatter — the correct axis must be identified per call site

---

### Chapter 7 — Validation: Confirming Correctness of the Async All-Gather Pipeline

**Description:** Describes the validation strategy for confirming that removing `synchronize_device` and adopting `all_gather_async` does not introduce race conditions, numerical errors, or output corruption in the hybrid attention stack.

**Directory:** `ch7_validation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Recap of Chapters 1–6 prerequisites
  - Summary: three complementary validation approaches — functional correctness (PCC against reference), multi-replay stability (trace replay consistency across N steps), and latency measurement (confirming the expected throughput gain)
  - "What's next" section listing files in reading order

- `functional_correctness.md`
  - Describe the numerical correctness test: run `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` with the modified `_maybe_all_gather` (async, no synchronize) in non-traced mode; compare per-token output tensors against a reference run that uses the original synchronous implementation; compute PCC (Pearson Correlation Coefficient) across the full hidden dimension; require PCC > 0.999
  - Explain what a PCC failure indicates: either the async all_gather is producing incorrect data (a semaphore initialization bug causing a false-completion signal) or the downstream op is reading stale input (a race condition where the consumer starts before the all_gather completes)
  - Explain how to distinguish these two failure modes: if PCC is low and the error is spatially correlated with specific devices or ranks, it is likely a semaphore bug; if PCC is low and the error is random across positions, it is likely a race condition with partial all_gather completion
  - Describe the hybrid stack test: run a full decoder layer containing one `TTNNQwen3LinearAttention` and one `TTNNQwen3FullAttention` with the same approach; confirm that the outputs of both attention modules are numerically correct in sequence

- `multi_replay_stability.md`
  - Describe the trace replay consistency test: capture a trace of the modified attention module's decode forward pass; execute the trace N times (N >= 10); compare the output tensors across all replays against the first replay's output; all outputs must be bit-identical or differ only by rounding noise (PCC > 0.9999)
  - Explain why stability matters beyond first-replay correctness: race conditions in async pipelines can be intermittent; the semaphore reset-before-replay protocol must be verified to eliminate stale-semaphore skip-through on replay 2, 3, etc.
  - Describe the deadlock detection procedure: if the semaphore is not correctly reset before replay, the async CCL kernel will wait indefinitely for a semaphore that never reaches its target value; wrap each `execute_trace` call with a timeout (e.g., 10 seconds); if the timeout fires, the test fails with a semaphore-init diagnostic
  - Describe the stress test variant: alternate between traced and non-traced calls (simulating a prefill that is not traced followed by a traced decode) and verify that the cycling counters remain consistent and no aliasing occurs between traced and non-traced semaphore slots

- `latency_measurement.md`
  - Describe the before/after latency measurement: run decode step timing (Tracy or Python wall-clock) with the original implementation (synchronous all_gather + `synchronize_device`) vs. the modified implementation (async all_gather, no synchronize) in non-traced mode; the difference is the `synchronize_device` overhead; this should match the estimate from Chapter 5
  - Describe the full-trace latency measurement: after the implementation is complete and the trace capture is working, measure per-step latency with `enable_trace=True`; the total improvement includes both the `synchronize_device` removal and the trace dispatch overhead elimination; present the two contributions separately
  - Explain how to confirm that the measured improvement is attributable to the `synchronize_device` removal and not to other changes: run with the async CCL changes but with `ttnn.synchronize_device()` temporarily re-inserted (no trace); the latency should match the original; then remove the synchronize call and re-measure
  - Describe the expected measurement procedure using Tracy: `python3 -m tracy -r -- pytest <test_file>` with `TT_METAL_DEVICE_PROFILER=1` and `TT_METAL_PROFILER_TRACE_TRACKING=1`; identify the per-decode-step trace replay duration in the `METAL TRACE REPLAY SESSION ID >= 2` rows of the ops CSV

---

## Conventions

**Terminology:**

| Term | Meaning in this guide |
|---|---|
| `_maybe_all_gather` | The helper method in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` that conditionally performs a tensor-parallel all_gather when the module is running on multiple devices; the primary subject of this guide |
| `ttnn.synchronize_device` | A host-blocking TTNN call that drains the device command queue and waits for all submitted kernels to complete before returning; abbreviated as "synchronize_device" throughout |
| host-blocking call | Any Python call that halts host execution until a device-side operation completes; includes `ttnn.synchronize_device`, `ttnn.to_torch`, `ttnn.from_device`, and similar readback operations |
| Metal Trace / trace | The pre-encoded device command buffer captured by `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` and replayed by `ttnn.execute_trace`; requires a fully async device command stream with no host-blocking calls during capture |
| CQ0 | Command Queue 0, the single device command queue used in trace-compatible deployment; all ops submitted to CQ0 execute in FIFO order |
| CQ0 ordering guarantee | The guarantee that op N+1 submitted to CQ0 will not begin reading its inputs until op N has written its outputs; eliminates the need for `synchronize_device` as a sequencing mechanism in single-CQ mode |
| `all_gather_async` | `ttnn.experimental.all_gather_async`; the async, non-blocking variant of all_gather that uses explicit `GlobalSemaphore` handles for completion signaling; trace-compatible when used with the cycling semaphore pattern |
| `TT_CCL` | The class in `models/tt_transformers/tt/ccl.py` that manages double-buffered `GlobalSemaphore` pools for async CCL ops; provides `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle` |
| cycling semaphore | The double-buffered `GlobalSemaphore` pattern where two handles alternate on consecutive calls via a modular index counter; prevents aliasing between back-to-back async CCL invocations |
| capture-time handle | The specific `GlobalSemaphore` object whose L1 address was snapshotted into the trace at `end_trace_capture` time; must be the handle used on every subsequent `execute_trace` replay |
| semaphore index snapshot | A copy of the `ag_semaphores_idx` and `barrier_semaphore_idx` fields from a `TT_CCL` instance taken before `begin_trace_capture`; used to restore the indices before each `execute_trace` so the correct capture-time handles are selected |
| PCC | Pearson Correlation Coefficient; the numerical correctness metric used throughout; a value > 0.999 is the minimum acceptable threshold for attention output |
| T3K | The 8-device Wormhole mesh used in tt-symbiote deployment, arranged as a 1×8 logical ring on cluster axis 1 |
| hybrid decoder stack | The alternating DeltaNet (`TTNNQwen3LinearAttention`) and standard attention (`TTNNQwen3FullAttention`) layer stack in the Qwen3.6-35B-A3B model; the end-to-end trace capture of this stack is the overarching goal |

**Notation:**

- All TTNN Python API symbols are formatted as inline code: `ttnn.synchronize_device`, `ttnn.experimental.all_gather_async`, `ttnn.begin_trace_capture`, `ttnn.execute_trace`, `ttnn.reset_global_semaphore_value`, etc.
- Class names and method names are formatted as inline code: `TT_CCL`, `_maybe_all_gather`, `get_and_cycle_ag_semaphore_handles`, `TTNNQwen3FullAttention`, `LayerStack`, `TracedRun`, etc.
- File paths are given relative to the relevant repository root (tt-symbiote or tt-metal as appropriate) and formatted as inline code: `models/tt_transformers/tt/ccl.py`, `models/tt_transformers/tt/attention.py`
- When specifying cluster axis values, use the exact Python values: `cluster_axis=0`, `cluster_axis=1`, `cluster_axis=None`; and the corresponding semaphore array slot indices: `semaphore_index=0`, `semaphore_index=1`, `semaphore_index=2`
- Latency estimates are expressed in microseconds (µs) for values < 1000 µs and milliseconds (ms) for larger values, with the other unit in parentheses where clarity is needed
- Numbered lists are used for sequential procedures (pre-capture steps, pre-replay steps) so they can serve as checklists
- Callout blocks use blockquote syntax with a bold label: `> **Note:**`, `> **Warning:**`, `> **Key finding:**`
- No emoji in any file
- Each `.md` file begins with an H1 title and a one-paragraph orientation stating what the reader will know by the end of the file
- Every chapter's `index.md` ends with a "What's next" section listing the files in that chapter in reading order
- The distinction between `ttnn.synchronize_device(mesh_device)` (single-mesh form) and `ttnn.synchronize_devices(mesh_device)` (plural form, if it exists) must be noted wherever both could appear — do not conflate them

**Formatting rules:**

- Code snippets showing the before/after of `_maybe_all_gather` must include inline comments that identify the semantic of each argument: `# cycling semaphore handle slot 0 or 1`, `# barrier semaphore for completion signal`, etc.
- When citing source files, include the function name and approximate line number: "see `Attention.forward` at line ~570 in `models/tt_transformers/tt/attention.py`"
- All pre-replay reset steps are presented as a numbered checklist in `ch6_implementation/trace_capture_wrapper_changes.md` so they can be used as a code-review reference
- Latency estimates stated without a source measurement must be labeled as estimates and include the formula used to derive them

---

## Cross-Chapter Dependencies

```
Chapter 1 (_maybe_all_gather: Role, Call Sites, and the synchronize_device Call)
  - Introduces: _maybe_all_gather call sites in TTNNQwen3FullAttention and
                TTNNQwen3LinearAttention, ttnn.synchronize_device semantics,
                why synchronize_device is a host-blocking call, why it is
                incompatible with Metal Trace capture, the single-CQ ordering model
  - Required by: all subsequent chapters

Chapter 2 (The Async CCL Pattern in tt-transformers for Traced Decode)
  - Depends on: Chapter 1 (why synchronize_device must go, what trace capture needs)
  - Introduces: all_gather_async call signature from Attention.forward,
                cycling semaphore mechanics from TT_CCL, persistent output
                buffer contract, absence of synchronize_device in the tt-transformers
                async CCL path, how CQ0 ordering eliminates the need for
                host-side barriers
  - Required by: Chapter 3 (the async CCL pattern is the replacement model that
                 informs the root cause verdict), Chapter 6 (the implementation
                 directly adopts this pattern), Chapter 7 (validation reuses the
                 same replay stability protocol)

Chapter 3 (Root Cause Analysis: Why synchronize_device Is Present)
  - Depends on: Chapter 1 (synchronize_device semantics and call context),
                Chapter 2 (CQ0 ordering guarantee, async CCL without synchronize)
  - Introduces: whether the underlying all_gather is synchronous or async,
                the CQ0 ordering guarantee as a sufficient replacement,
                the structural change requirement (TT_CCL wiring),
                the definitive verdict on removability
  - Required by: Chapter 4 (the audit applies the same CQ0 ordering reasoning to
                 other modules), Chapter 6 (the implementation follows the verdict),
                 Chapter 5 (the latency model depends on knowing the call is
                 unnecessary, not just inconvenient)

Chapter 4 (Symbiote-Wide Audit: Other synchronize_device Calls That Block Trace)
  - Depends on: Chapter 1 (what constitutes a forward-path synchronize_device call
                and why it blocks trace), Chapter 3 (the CQ0 ordering reasoning
                used to assess removability of each found call)
  - Introduces: complete audit results for all forward-path synchronize_device calls
                in tt-symbiote, prioritization by criticality to the hybrid stack
  - Required by: Chapter 6 (the implementation scope is informed by the audit;
                 the PR scope is defined by the prioritization)

Chapter 5 (Latency Cost of synchronize_device and Throughput Impact)
  - Depends on: Chapter 1 (where the calls are), Chapter 3 (verdict that they are
                unnecessary), Chapter 4 (how many calls are on the critical path)
  - Introduces: latency model for synchronize_device, measurement procedure using
                Tracy, throughput improvement estimate, worked example
  - Required by: Chapter 7 (the latency measurement in validation follows the
                 procedure described here and verifies the Chapter 5 estimate)

Chapter 6 (Implementation Plan: Removing synchronize_device and Adopting async CCL)
  - Depends on: Chapter 1 (call sites to modify), Chapter 2 (the async CCL pattern
                to adopt), Chapter 3 (structural changes required), Chapter 4 (audit
                scope defines which modules need changes in the same PR)
  - Introduces: concrete code changes (TT_CCL wiring, _maybe_all_gather signature,
                all_gather_async arguments, synchronize_device removal),
                trace capture wrapper changes (semaphore index snapshot/restore,
                reset_global_semaphore_value protocol)
  - Required by: Chapter 7 (validation tests are run against the implementation
                 described here)

Chapter 7 (Validation: Confirming Correctness of the Async All-Gather Pipeline)
  - Depends on: all prior chapters
  - Synthesizes: the async CCL pattern (Ch2), the removability verdict (Ch3),
                 the audit scope (Ch4), the latency model (Ch5), the implementation
                 spec (Ch6)
  - Introduces: PCC correctness test methodology, multi-replay stability test,
                deadlock detection via timeout, stress test with traced/non-traced
                interleaving, before/after latency measurement procedure
  - Introduces no new architectural concepts; provides an integrated validation
    checklist and acceptance criteria
```
