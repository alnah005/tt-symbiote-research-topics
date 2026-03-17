# Plan: Tracy Profiling and MoE Forward Pass Analysis

---

## 1. Audience

**Primary audience:** ML engineers who need to profile Mixture of Experts (MoE) models on Tenstorrent hardware (Wormhole B0, T3K mesh) in order to identify performance bottlenecks — in particular, an observed ~16 ms latency gap in the MoE forward pass.

**What they already know:**
- TTNN op usage: `ttnn.matmul`, `ttnn.to_device`, memory configs, sharding
- Basic transformer and MoE architecture: gating network, top-K routing, dispatch-combine pattern
- Python development and familiarity with running tt-metal workloads on device
- General profiling concepts (wallclock timing, Python `time` module)

**What they do NOT need to know in advance:**
- The Tracy profiler tool, its UI, or how to build it
- How tt-metal instruments ops with Tracy markers internally
- How to read or interpret a Tracy `.tracy` trace file
- The `ttnn.device_profiler` Python API or its output CSV format
- How to design a controlled scaling experiment to attribute latency gaps

This guide fills those gaps progressively, starting from Tracy fundamentals and building to an actionable bottleneck identification and scaling analysis workflow tailored to MoE workloads.

---

## 2. Chapter List

---

### Chapter 1: Tracy Profiler Overview

**Description:** Introduces the Tracy profiler — what it is, how it integrates with tt-metal and TTNN, and what categories of performance data it captures for device workloads.

**Directory:** `ch1_tracy_overview/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: what Tracy captures vs. what the TTNN device profiler captures
  - Navigation to sub-topics and prerequisites checklist

- `what_is_tracy.md`
  - Tracy's origins: a real-time, low-overhead C++ profiler originally designed for game engines, adopted in systems-level performance tooling
  - What Tracy records: CPU zones (function-level spans), GPU zones, memory allocations, frame markers, and free-form messages
  - The Tracy data model: every recorded event is a named zone with a start timestamp, end timestamp, thread ID, and optional color and value payload
  - The two-process model: the profiled application emits events over a socket; the Tracy server (GUI or `capture` CLI) records them to a `.tracy` file
  - How Tracy integrates with tt-metal: the `TRACY_ENABLE` compile flag activates macros in `tt_metal/tools/profiler/tt_metal_tracy.hpp`; when disabled all macros are no-ops with zero runtime cost
  - What tt-metal annotates by default: op dispatch calls, program enqueue events, mesh trace lifecycle events (`TracyTTMetalBeginMeshTrace`, `TracyTTMetalEndMeshTrace`, `TracyTTMetalReplayMeshTrace`, `TracyTTMetalReleaseMeshTrace`, `TracyTTMetalEnqueueMeshWorkloadTrace`)

- `tracy_vs_device_profiler.md`
  - Distinction between two complementary tools: Tracy (CPU-side op dispatch instrumentation) and the TTNN device profiler (on-device kernel execution timing via hardware cycle counters)
  - What each tool answers: Tracy answers "when did the host enqueue this op?" and "how long did host-side dispatch take?"; the device profiler answers "how long did the kernel actually run on Tensix cores?"
  - When to use Tracy alone, device profiler alone, and both together
  - The combined workflow: use Tracy to identify which host-side phase has unexpected latency, then use the device profiler to confirm whether the gap is in kernel execution or in host-device synchronization
  - Known blind spots: Tracy does not show on-device kernel execution time unless explicitly correlated; device profiler output requires post-processing via Python scripts

---

### Chapter 2: Setting Up Tracy Profiling

**Description:** Covers everything needed to capture a Tracy trace from a tt-metal workload: build flags, environment variables, the Tracy capture binary, and the output file format.

**Directory:** `ch2_tracy_setup/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Setup checklist: five steps from clean checkout to first `.tracy` file
  - Pointer to Tracy upstream release page for matching server/client version requirements

- `build_flags.md`
  - The `ENABLE_PROFILER` CMake flag: what it enables beyond Tracy (`kernel_profiler.hpp`, cycle counter instrumentation in Tensix kernels)
  - The `TRACY_ENABLE` preprocessor define: how it is controlled via CMake and how to verify it is active in a build
  - Example CMake invocation: `cmake -DENABLE_PROFILER=ON -DTRACY_ENABLE=ON ...`
  - The `tracy` submodule location in the tt-metal tree and version pinning
  - Build-time warning: enabling `ENABLE_PROFILER` incurs a small but nonzero runtime overhead; do not use for SLA-critical benchmarks
  - Verifying the build: `ldd` check for Tracy shared library, or checking that `TT_METAL_TRACY` appears in the binary's symbol table

- `capture_workflow.md`
  - The three components that must run together: (1) the profiled tt-metal process, (2) the Tracy capture server (`tracy-capture`), (3) optionally the Tracy GUI (`tracy-profiler`)
  - Environment variable `TT_METAL_DEVICE_PROFILER=1`: enables the on-device CSV profiler in addition to Tracy CPU zones
  - Environment variable `TRACY_NO_EXIT=1`: keeps the profiled process alive after main() returns so Tracy can finish flushing all events
  - Launching the capture server: `./tracy-capture -o output.tracy -f` (force overwrite) before launching the workload
  - Connecting the GUI to a live capture vs. opening a saved `.tracy` file after the fact
  - Output artifacts: `output.tracy` (binary Tracy database), `profile_log_device.csv` (device profiler CSV if `TT_METAL_DEVICE_PROFILER=1`)
  - Common failure mode: Tracy client and server version mismatch — symptoms and fix

- `output_format.md`
  - Anatomy of a `.tracy` file: binary format, not human-readable; must be opened in Tracy GUI or processed with `tracy-csvexport`
  - Using `tracy-csvexport` to produce a human-readable CSV of all zones with columns: zone name, thread, start (ns), end (ns), duration (ns)
  - The `profile_log_device.csv` format from `TT_METAL_DEVICE_PROFILER=1`: columns include op name, device ID, core coordinates, start cycle, end cycle, duration cycles, and derived duration in microseconds
  - How to align Tracy CPU timestamps with device profiler cycle counts: the `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` environment variable and its role in anchoring device time to host time

---

### Chapter 3: TTNN Op-Level Profiling API

**Description:** Explains the Python-level `ttnn` device profiler API, how to annotate your own code with profiling markers, and how to extract per-op timing from the output CSV.

**Directory:** `ch3_ttnn_profiling_api/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: three ways to get per-op timing from TTNN (Tracy zones, device profiler CSV, `ttnn.tracer`)
  - Navigation to sub-topics

- `device_profiler_api.md`
  - The `ttnn.device_profiler_state` context manager: how to wrap a block of TTNN ops to capture device-side timing
  - What the profiler records per op: op name (derived from the TTNN op class), program ID, device ID, core grid used, start/end cycle, duration
  - Enabling via environment variable vs. Python API: `TT_METAL_DEVICE_PROFILER=1` at process start vs. `ttnn.enable_program_cache()` interaction
  - How TTNN op names appear in the CSV: the naming convention (`tt::operations::primary::matmul`, `ttnn::experimental::moe_dispatch`, etc.) and how to match CSV rows to source code calls
  - Post-processing script: `tt_metal/tools/profiler/process_ops_logs.py` — how to run it, what it produces, and how to read the output timeline

- `annotating_your_code.md`
  - Using Tracy's Python bindings (`tracy-client` pip package or the bundled Python shim) to add custom zone markers around user-defined code sections
  - The `ZoneScoped` equivalent in Python: `tracy.zone(name)` context manager
  - Naming conventions for custom zones that will appear clearly in the Tracy GUI: use `MoE/dispatch`, `MoE/expert_matmul`, `MoE/combine` as zone names to get a readable hierarchy
  - How to add a Tracy message (free-form annotation) to mark sequence length or batch size at a specific point in the trace: `tracy.message(f"seq_len={seq_len}")`
  - Interaction with `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`: Tracy zones recorded during trace capture will appear in every replay; annotate the replay call site, not the capture site, for accurate timing

- `reading_op_timing_output.md`
  - Step-by-step: run the model, collect `profile_log_device.csv`, run `process_ops_logs.py`, open the resulting ODS/HTML report
  - Interpreting the per-op table: what "dispatch time" vs. "kernel time" means in the CSV columns
  - Finding the MoE-related ops by name in the CSV: filtering for `moe`, `matmul`, `all_to_all`, `topk`, `gather` in the op name column
  - Summing op durations to reconstruct the MoE forward pass total and comparing against wallclock measurement to find unaccounted gaps
  - Known gap sources: host-side Python overhead between ops, device synchronization barriers, CCL collective latency not attributed to a named op

---

### Chapter 4: MoE Forward Pass Op Breakdown

**Description:** Documents the sequence of TTNN ops that execute during a MoE forward pass — from expert dispatch through expert matmuls to combine — and establishes the expected timeline and latency budget.

**Directory:** `ch4_moe_op_breakdown/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Reference timeline diagram: linear sequence of ops from router input to MoE output
  - Prerequisites: Chapters 1–3 (Tracy and profiler setup)

- `dispatch_phase.md`
  - What happens during expert dispatch: the router computes token-to-expert assignments (`ttnn.topk` on router logits)
  - Ops in order: `ttnn.linear` (router projection) → `ttnn.softmax` → `ttnn.topk` → index tensor construction → `ttnn.gather` (token reordering by expert)
  - Typical latency budget for the dispatch phase at seq_len=1024: expected range and what factors drive it (seq_len, num_experts, d_model)
  - How the dispatch phase output shapes depend on routing: the `[num_active_tokens_per_expert, d_model]` gather output shape and why it varies
  - On T3K: the all-to-all communication op (`ttnn.experimental.all_to_all` or CCL equivalent) that redistributes tokens to the chip holding each expert shard — its expected latency and how to identify it in the trace

- `expert_matmul_phase.md`
  - What expert computation entails: two matmuls per expert (gate projection and up projection, fused or sequential) followed by an activation function and a down projection
  - TTNN ops in order: `ttnn.matmul` (gate proj) → `ttnn.matmul` (up proj) → `ttnn.silu` or `ttnn.gelu` → element-wise multiply → `ttnn.matmul` (down proj)
  - How batched expert matmul is structured in TTNN: the `[num_experts, expert_capacity, d_model]` activation tensor and `[num_experts, d_model, d_ff]` weight tensor
  - Expected latency per matmul given common shapes (Qwen 235B MoE expert dims: d_model=7168, d_ff=2048, top_k=8, num_experts=128) and how to calculate theoretical vs. observed FLOPs
  - Why the expert matmul phase dominates total MoE latency in the prefill regime and is less dominant in decode

- `combine_phase.md`
  - What happens during expert combine: gathering per-expert outputs back into token order and applying routing weights
  - Ops in order: `ttnn.scatter` or inverse-gather → element-wise multiply by router scores → `ttnn.sum` across top-K experts per token
  - On T3K: the reduce-scatter or all-reduce op that aggregates partial expert outputs from each chip
  - Expected latency for the combine phase and its relationship to seq_len and top_k
  - The combine phase as a source of load imbalance: unequal token counts per expert cause some cores to idle during scatter

- `full_op_sequence_reference.md`
  - Consolidated table: every named TTNN op in the MoE forward pass, its input shapes, output shapes, expected duration in microseconds (prefill and decode regimes), and the Tracy zone name it appears under
  - How to use this table as a "ground truth" checklist when reading a Tracy trace: which ops should be present, which are optional (e.g., the CCL all-to-all is only present in the multi-chip case), and what their absence or unusual duration indicates
  - Known variations across MoE model families: DeepSeek-V3 shared expert path, Mixtral's simpler two-expert routing, Qwen MoE's 128-expert high-sparsity configuration

---

### Chapter 5: Identifying the 16ms Gap

**Description:** Teaches how to read a Tracy trace to find latency gaps between ops, correlate unaccounted time with specific synchronization points, and determine whether the 16ms gap is in host dispatch, device execution, or communication.

**Directory:** `ch5_identifying_gap/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The three gap hypotheses: (1) host-side Python overhead, (2) device synchronization barrier, (3) CCL collective latency
  - Quick-reference: decision tree for attributing a latency gap

- `reading_tracy_traces.md`
  - Tracy GUI orientation: the timeline view, the zone statistics view, the frame graph, and the statistics panel
  - Zooming in on the MoE forward pass: how to identify the start and end of one MoE layer in the timeline using the custom Tracy zones added in Chapter 3
  - Reading zone nesting: understanding that `MoE/dispatch` and `MoE/combine` are children of the enclosing `MoE/forward` zone
  - Finding gaps: any horizontal whitespace between consecutive zones on the same thread represents untracked time — what to look for and how to measure it
  - Using `tracy-csvexport` to compute gap durations programmatically: sort CSV by start time, compute `start[i+1] - end[i]` for consecutive zones, flag gaps > 1ms

- `gap_attribution.md`
  - Method 1: Compare Tracy CPU zone end times with device profiler kernel start times — a gap between CPU zone end and next kernel start is host dispatch latency
  - Method 2: Look for explicit synchronization calls in the Tracy timeline — `ttnn.synchronize_device` or `ttnn.wait_for_event` zones that block the host thread
  - Method 3: Check if the gap aligns with a CCL collective: the all-to-all or reduce-scatter duration should appear as a Tracy zone if annotated, or as a gap if not
  - The 16ms hypothesis table: for each hypothesis, what the Tracy evidence looks like (zone pattern, gap size, repeatability across runs)
  - How to rule out measurement noise: run 10 iterations and report mean ± stddev; a consistent 16ms gap is real, a variable gap may be OS scheduling jitter

- `common_gap_patterns.md`
  - Pattern A: gap immediately after `ttnn.topk` — indicates the index construction step (CPU-side tensor manipulation) is not inside a Tracy zone; fix by annotating the index construction loop
  - Pattern B: gap between last expert matmul and first combine op — indicates a device synchronization barrier; likely `ttnn.synchronize_device` called between phases
  - Pattern C: gap between dispatch and expert compute that scales with num_active_tokens — indicates the all-to-all CCL latency is in this gap
  - Pattern D: gap at the very beginning of the MoE layer — indicates TTNN program cache miss (recompilation); check if tensor shapes changed since last call
  - How to distinguish Pattern C from Pattern D: CCL latency scales with message size (tokens × d_model); recompilation latency is fixed per unique shape

---

### Chapter 6: Sequence Length Scaling Analysis

**Description:** Explains how to design and execute a controlled scaling experiment to determine whether and how the 16ms gap scales with sequence length, and what different scaling behaviors imply about the root cause.

**Directory:** `ch6_sequence_length_scaling/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The four expected scaling behaviors and their implications (see `scaling_theory.md`)
  - Prerequisites: Chapters 4–5 (op breakdown and gap identification)

- `scaling_theory.md`
  - Why sequence length is the primary independent variable: it sets the total number of tokens processed by the MoE layer, which affects all three phases (dispatch, compute, combine)
  - Expected scaling behavior by op class:
    - Matrix multiplications: latency scales as O(seq_len) when seq_len < num_cores × tile_size (memory-bound regime) and is flat when the matmul is compute-bound
    - Softmax and TopK: O(seq_len) always (element-wise)
    - All-to-all CCL: O(seq_len × d_model / num_chips), linear in message size
    - Host-side Python index construction: O(seq_len) if using Python loops, O(1) if fully tensor-ized
    - Device synchronization barriers: O(1), independent of seq_len
  - A gap that is O(1) in seq_len is almost certainly a synchronization barrier or a fixed-cost host operation (e.g., dispatching a program with program cache miss)
  - A gap that is O(seq_len) is either compute, bandwidth, or a linear host-side operation

- `experiment_design.md`
  - Choosing seq_len sweep points: recommended set is {64, 128, 256, 512, 1024, 2048, 4096} to span two decades and reveal the scaling exponent
  - Controlling confounders: fix batch size = 1, fix num_experts and top_k, fix d_model and d_ff, vary only seq_len
  - Warm-up runs: always run 3 warm-up iterations before measuring to ensure program cache is warm and DRAM is paged in
  - What to measure per seq_len point: (1) total MoE layer wallclock (Python `time.perf_counter`), (2) per-phase latency from device profiler CSV, (3) gap duration from Tracy CSV export
  - Sample size and statistical rigor: 20 timed iterations, report median and p95 (not mean, which is sensitive to outliers from OS scheduling)
  - How to automate the sweep: parameterized pytest fixture or a standalone Python script that loops over seq_len values and writes results to a structured CSV

- `interpreting_scaling_results.md`
  - How to plot the results: gap duration (y-axis, ms) vs. seq_len (x-axis, log scale) with a best-fit line in log-log space
  - Reading the scaling exponent: slope ≈ 1.0 in log-log → linear; slope ≈ 0 → constant; slope between 0 and 1 → sublinear (e.g., bandwidth-limited matmul that transitions from memory-bound to compute-bound)
  - Decision table: scaling exponent → most likely gap source → recommended investigation path
  - How to decompose a mixed gap: if total gap = constant_term + linear_term × seq_len, fit a linear regression to separate the two components and attribute each to a different root cause
  - What to do when results are non-monotonic: indicates program cache invalidation at specific seq_len boundaries (tile alignment thresholds); look for tile-count discontinuities at seq_len multiples of 32

---

### Chapter 7: Interpretation and Next Steps

**Description:** Translates profiler findings — Tracy traces, device profiler CSVs, and scaling experiment results — into concrete optimization actions, and documents the investigation process for handoff to the broader team.

**Directory:** `ch7_interpretation_and_next_steps/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The output of this chapter: a written gap analysis document and a prioritized optimization backlog
  - Prerequisites: all previous chapters; this chapter synthesizes the full investigation

- `gap_to_action_mapping.md`
  - Systematic mapping from each gap pattern identified in Chapter 5 to a recommended optimization action:
    - Synchronization barrier gap → investigate whether the barrier is required for correctness or can be deferred/eliminated using `ttnn.event` for fine-grained synchronization
    - Host-side Python overhead gap → tensor-ize the offending operation (replace Python loops with TTNN ops), or move the operation inside a `ttnn.begin_trace_capture` region
    - CCL all-to-all gap → evaluate whether expert parallelism degree can be reduced; investigate overlap of CCL communication with local compute using `ttnn.experimental.ccl.all_to_all_async`
    - Program cache miss gap → audit tensor shape changes across MoE layers; pad or canonicalize shapes to ensure cache hits
  - How to prioritize: use the scaling analysis to identify which gap has the largest total contribution across a typical inference workload (prefill + N decode steps)

- `writing_a_gap_analysis.md`
  - Template for a gap analysis document: (1) observed latency, (2) profiling methodology, (3) per-op breakdown table, (4) gap attribution findings, (5) scaling behavior, (6) prioritized root causes, (7) recommended actions
  - What evidence to include: annotated Tracy trace screenshots, device profiler CSV excerpts, scaling plots with fit lines
  - How to express uncertainty: distinguish between "confirmed" gaps (consistent across all runs, attributable to a named op or host call) and "suspected" gaps (present in most runs, not yet attributed)
  - Audience for the gap analysis: optimization engineers who will implement fixes; must include enough detail for someone who was not present during profiling to reproduce the findings

- `optimization_action_reference.md`
  - Concise reference table of optimization levers available in TTNN for MoE performance:
    - `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`: eliminates per-op host dispatch overhead for the captured op sequence; critical for decode where the same op sequence repeats
    - `ttnn.enable_program_cache()`: ensures compiled kernels are reused across calls with the same shapes; must be called before the first inference
    - Sharded memory configs for expert weight tensors: reduces DRAM bandwidth pressure by keeping weights in L1 between matmul calls
    - `ttnn.experimental.ccl.all_to_all_async` with compute overlap: hides CCL latency behind expert matmul compute
    - Expert capacity padding strategy: padding token counts to multiples of 32 ensures tile alignment and avoids partial-tile performance penalties
  - For each lever: expected latency reduction, conditions under which it applies, and any correctness caveats
  - How to validate that an optimization did not degrade output correctness: PCC check against a CPU reference (threshold > 0.999 for bfloat16)

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **Tracy** | A real-time C++/Python profiler that records named CPU zones with nanosecond timestamps; see [Tracy GitHub](https://github.com/wolfpld/tracy) |
| **Tracy zone** | A named, time-bounded profiling scope recorded by Tracy; analogous to a "span" in distributed tracing |
| **`TRACY_ENABLE`** | The preprocessor define that activates Tracy instrumentation in tt-metal; when undefined, all Tracy macros are no-ops |
| **`ENABLE_PROFILER`** | The CMake flag that enables both Tracy and the on-device cycle-counter profiler in tt-metal |
| **device profiler** | The tt-metal on-device profiler that records kernel execution times using Tensix hardware cycle counters; output is `profile_log_device.csv` |
| **`TT_METAL_DEVICE_PROFILER`** | Environment variable (set to `1`) that enables the device profiler at runtime |
| **`.tracy` file** | The binary database file produced by `tracy-capture`; must be opened in the Tracy GUI or processed with `tracy-csvexport` |
| **op sequence** | The ordered list of TTNN ops that execute during one forward pass of the MoE layer |
| **gap** | Unaccounted elapsed time between the end of one profiled zone and the start of the next |
| **16ms gap** | The specific latency gap under investigation, observed between expert dispatch and expert combine in the MoE forward pass |
| **dispatch phase** | The portion of the MoE forward pass that routes tokens to experts: router linear → softmax → topk → gather |
| **expert matmul phase** | The portion that executes expert FFN computation: gate proj → up proj → activation → down proj matmuls |
| **combine phase** | The portion that aggregates expert outputs back to token order: scatter → weighted sum |
| **CCL** | Collective Communication Library — the TTNN library providing multi-chip ops like all-to-all and reduce-scatter |
| **program cache** | The tt-metal kernel compilation cache; a cache hit means the compiled kernel is reused without recompilation |
| **tile** | The 32×32 element atomic compute unit on Tenstorrent Wormhole hardware |
| **T3K** | The 8-chip Tenstorrent Wormhole mesh; chips are connected via high-speed ethernet links |
| **PCC** | Pearson Cross-Correlation — the primary correctness metric for TTNN output validation |
| **ep_degree** | Expert Parallelism degree — the number of chips across which expert weights are distributed |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions where helpful, e.g., `[seq_len, d_model]`.
- Tile counts are written as `M_t`, `K_t`, `N_t` where `M_t = seq_len / 32`.
- Latency values are quoted in milliseconds (ms) for human-scale discussion and microseconds (µs) for per-op device profiler values.
- Tracy zone names follow a hierarchical slash notation: `MoE/dispatch/topk`, `MoE/expert_matmul`, `MoE/combine`.
- Environment variables are written in `SCREAMING_SNAKE_CASE` and prefixed with their package: `TT_METAL_DEVICE_PROFILER`, `TRACY_NO_EXIT`.
- Code blocks use Python syntax and assume `import ttnn`, `import torch`, and `import time` are in scope.
- Performance numbers are indicative and based on Wormhole B0 silicon at firmware version current as of the guide's writing; always re-profile on the target system.
- All chapter files end with a "Next Steps" section pointing forward to the next logical file or chapter.

### Formatting Rules

- Every chapter directory has an `index.md` providing an overview, learning objectives, and navigation links to all files in the chapter.
- Code examples are fenced with ` ```python ` and include inline comments explaining non-obvious lines.
- Tables are used for comparisons, op sequences, and decision matrices; prose is used for conceptual explanation.
- Warnings about correctness pitfalls use `> **Warning:** ...` blockquote formatting.
- Performance recommendations use `> **Tip:** ...` blockquote formatting.
- Investigative hypotheses that have not been confirmed are marked with `> **Hypothesis (unconfirmed):** ...`.

---

## 4. Cross-Chapter Dependencies

The guide is designed to be read front-to-back. The table below clarifies which chapters depend on concepts introduced earlier, and flags specific forward references that must be consistent.

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: Tracy Profiler Overview | None (foundational) |
| Ch 2: Setting Up Tracy Profiling | Ch 1 (Tracy zones, `TRACY_ENABLE`, the two-process model) |
| Ch 3: TTNN Op-Level Profiling API | Ch 1 (Tracy vs. device profiler distinction), Ch 2 (build flags, `TT_METAL_DEVICE_PROFILER`, output files) |
| Ch 4: MoE Forward Pass Op Breakdown | Ch 3 (op naming in device profiler CSV, how to read op timing output) — does not require Chapters 1–2 if the reader only wants the op sequence reference table |
| Ch 5: Identifying the 16ms Gap | Ch 2 (Tracy trace files and `tracy-csvexport`), Ch 3 (device profiler CSV and per-op timing), Ch 4 (the expected op sequence to compare against) |
| Ch 6: Sequence Length Scaling Analysis | Ch 4 (op-level latency budget for each phase), Ch 5 (gap duration measurement methodology) |
| Ch 7: Interpretation and Next Steps | All previous chapters; Ch 5 (gap patterns and attribution), Ch 6 (scaling results), Ch 3 (op names and device profiler CSV format for the gap analysis template) |

**Specific forward references to flag:**

- Ch 1 (`tracy_vs_device_profiler.md`) introduces the distinction between Tracy CPU zones and device profiler cycle counts. Ch 5 (`gap_attribution.md`) relies on this distinction to explain Method 1 (comparing Tracy CPU zone end times with device kernel start times). These two files must use consistent terminology.
- Ch 3 (`annotating_your_code.md`) defines the Tracy zone naming convention (`MoE/dispatch`, `MoE/expert_matmul`, `MoE/combine`). Ch 5 (`reading_tracy_traces.md`) instructs the reader to use these zone names to orient themselves in the Tracy GUI. The zone names must match exactly between these two files.
- Ch 4 (`full_op_sequence_reference.md`) establishes the canonical per-op latency table for prefill and decode regimes. Ch 6 (`scaling_theory.md`) references the per-op scaling behavior, and Ch 7 (`gap_to_action_mapping.md`) references the total latency budget. All three must be consistent with the same model configuration (e.g., Qwen 235B MoE: d_model=7168, d_ff=2048, num_experts=128, top_k=8).
- Ch 5 (`common_gap_patterns.md`) names four gap patterns (A–D). Ch 7 (`gap_to_action_mapping.md`) maps each of these patterns to an optimization action. The pattern names and descriptions must match across these two files.
- Ch 6 (`experiment_design.md`) defines the standard seq_len sweep: `{64, 128, 256, 512, 1024, 2048, 4096}`. Ch 6 (`interpreting_scaling_results.md`) and Ch 7 (`writing_a_gap_analysis.md`) reference this sweep. All three files must use the same sweep definition.
- Ch 7 (`optimization_action_reference.md`) references `ttnn.begin_trace_capture` and `ttnn.enable_program_cache()`. These APIs are introduced in Ch 3 (`device_profiler_api.md`) and `annotating_your_code.md`. Ch 7 must not redefine them, only reference them.
