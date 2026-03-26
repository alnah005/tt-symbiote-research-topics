# Plan: TTNNMoE Performance Optimization on T3K

## Audience

This guide is written for ML systems engineers and hardware-aware model developers who are working on or evaluating Mixture-of-Experts (MoE) inference on Tenstorrent T3K hardware using the `tt-symbiote` / `tt-transformers` stack. Readers are expected to:

- Understand MoE routing fundamentals (topk gating, expert dispatch, token combining).
- Have working knowledge of TTNN ops, tensors, and program configs.
- Be familiar with T3K's 1×8 mesh topology and its collective communication primitives (all-gather, reduce-scatter, all-to-all).
- Know how to run a model forward pass in the existing codebase and read op-level timing output.

Readers do **not** need prior experience profiling with Tracy or tuning `MatmulMultiCoreReuseMultiCast1DProgramConfig` — those are covered from first principles.

---

## Chapter List

### Chapter 1 — `ch1_moe_forward_pass_anatomy/`

**Description:** Walk through the full `TTNNMoE.forward` and `TTNNExperts.forward` call stacks so readers have a precise mental model of every op before any measurement begins.

**Files:**

- `index.md`
  - Overview of the chapter; explains why a code-first reading is a prerequisite for all later profiling and tuning chapters.
  - Diagram of the end-to-end data flow: input tensor → all-gather → gate → `TTNNExperts` → reduce-scatter → shared experts → residual add.

- `ttnn_moe_forward.md`
  - Annotated walkthrough of `TTNNMoE.forward` (lines 1346–1496 in `moe.py`).
  - Describes the all-gather call: `all_gather_async` with `Linear` topology, `num_links=1`; notes that the result feeds both the gate linear and the expert path.
  - Describes the reduce-scatter call: `reduce_scatter_minimal_async` with `Ring` topology, `chunks_per_sync=10`, `num_workers_per_link=2`, `num_links=1`.
  - Documents the gate linear's `HiFi4` math fidelity and `fp32_dest_acc_en=True`, `packer_l1_acc=True` settings and why they differ from expert matmuls.
  - Notes `TTNNBailingMoE` (lines 1499+) as a subclass that inherits this path with Bailing-specific config.

- `ttnn_experts_forward.md`
  - Step-by-step walkthrough of `TTNNExperts.forward` (lines 1027–1343).
  - Covers every stage in order: token padding to `SPARSITY_BLOCK_SIZE=32` (line 52), `all_to_all_dispatch`, `moe_expert_token_remap`, three `sparse_matmul` calls, `silu`+`mul` activation, `all_to_all_combine`, and the weight application via `ttnn.repeat` + permute.
  - Explains the role of `TOPK_MIN_WIDTH=64` (line 51) in determining the minimum dispatch width.
  - Identifies which tensors are on-device versus host at each stage, flagging any implicit host-device transfers.

- `cpu_fallback_paths.md`
  - Documents `Glm4MoeNaiveMoeHybrid` (lines 559–613): the `ttnn = False` flag that hardcodes CPU execution for the older GLM-4 path.
  - Explains why this class bypasses `TTNNExperts` entirely and runs expert matmuls in PyTorch on CPU.
  - Lists every conditional branch in `moe.py` that could silently fall back to CPU during inference (search criteria: `ttnn = False`, `if not ttnn`, device placement guards).
  - Provides a checklist for verifying that a given model config is exercising the TTNN path end-to-end.

---

### Chapter 2 — `ch2_ccl_latency_and_topology/`

**Description:** Measure and understand the collective communication costs (all-gather and reduce-scatter) that bookend the expert computation on T3K's 1×8 mesh.

**Files:**

- `index.md`
  - Motivates why CCL ops are a primary suspect for decode-regime bottlenecks: at batch=1, the compute/comms ratio is low and small-message latency dominates.
  - Defines T3K's physical topology (1×8 mesh, Ethernet links) and how it constrains topology choices.

- `all_gather_linear_topology.md`
  - Focuses on the `all_gather_async` call in `TTNNMoE.forward` (lines 1346–1496): `Linear` topology, `num_links=1`.
  - Explains why `Linear` topology is used on a 1×8 mesh and its latency model versus `Ring`.
  - Describes how to isolate and measure just the all-gather cost using Tracy or TTNN op timers (cross-reference Chapter 5).
  - Discusses whether increasing `num_links` (if hardware supports it) or switching to `Ring` topology would reduce latency for the all-gather message size at batch=1.

- `reduce_scatter_ring_topology.md`
  - Focuses on the `reduce_scatter_minimal_async` call: `Ring` topology, `chunks_per_sync=10`, `num_workers_per_link=2`, `num_links=1`.
  - Explains the pipelining effect of `chunks_per_sync` and `num_workers_per_link` — how these parameters affect overlap between compute and communication.
  - Provides a methodology for sweeping `chunks_per_sync` values (e.g., 1, 5, 10, 20) and `num_workers_per_link` (1 vs 2) to find the optimal setting for T3K's 1×8 mesh at batch=1 decode.
  - Discusses the tradeoff between synchronization granularity and end-to-end latency.

- `ccl_sensitivity_analysis.md`
  - Summarizes measurement results across both CCL ops: which dominates, and by how much.
  - Provides guidance on whether re-ordering ops (e.g., overlapping reduce-scatter with shared-expert compute) is feasible given the current code structure in `TTNNMoE.forward`.
  - Addresses research question: are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?

---

### Chapter 3 — `ch3_expert_dispatch_pipeline_profiling/`

**Description:** Profile each stage of the `TTNNExperts.forward` pipeline at op-level granularity to identify which step dominates latency at batch=1 decode.

**Files:**

- `index.md`
  - States the key question: among `all_to_all_dispatch`, `moe_expert_token_remap`, the three `sparse_matmul` calls, `silu+mul`, and `all_to_all_combine`, which step is the bottleneck?
  - Describes the experimental setup: batch=1 decode, GLM-4-MoE and Bailing configs, T3K 1×8 mesh.

- `token_padding_and_dispatch.md`
  - Covers token padding to `SPARSITY_BLOCK_SIZE=32` and the `all_to_all_dispatch` op.
  - Explains how padding affects the effective batch size seen by expert matmuls and why 32 was chosen.
  - Describes how to measure dispatch latency in isolation and whether reducing `SPARSITY_BLOCK_SIZE` is safe.

- `sparse_matmul_profiling.md`
  - Deep-dives the three `sparse_matmul` calls in `TTNNExperts.forward` (lines 1027–1343).
  - Documents `_make_sparse_matmul_program_config` (lines 62–91): `MatmulMultiCoreReuseMultiCast1DProgramConfig`, `in0_block_w=min(4, hidden_tiles)`, `per_core_M=1`.
  - Explains what `in0_block_w` and `per_core_M` control in the multi-core reuse schedule and when the current defaults are suboptimal.
  - Provides a sweep methodology for tuning these parameters for the hidden/intermediate sizes specific to GLM-4-MoE and Bailing.
  - Discusses `HiFi2` math fidelity for expert matmuls versus `HiFi4` for the gate: what precision is lost, and how to measure it numerically (e.g., cosine similarity of expert outputs vs a reference).

- `weight_application_overhead.md`
  - Examines the post-`all_to_all_combine` weight application step: `ttnn.repeat` broadcasting `topk_experts_weights` to `(hidden_size, 1, 1, 1)` followed by a permute.
  - Profiles the cost of `ttnn.repeat` + permute versus the alternative of an elementwise multiply after a reshape (avoiding the broadcast dimension expansion).
  - Provides a code sketch of the alternative and criteria for deciding whether to replace it.

- `bottleneck_summary.md`
  - Aggregates per-stage latency measurements into a ranked table.
  - Identifies the single biggest bottleneck in `TTNNExperts.forward` at batch=1 decode.
  - Maps findings back to the source lines in `moe.py` for actionable follow-up.

---

### Chapter 4 — `ch4_matmul_config_and_math_fidelity/`

**Description:** Tune `_make_sparse_matmul_program_config` and evaluate math fidelity settings (HiFi2 vs LoFi) for expert matmuls across GLM-4-MoE and Bailing model configs.

**Files:**

- `index.md`
  - Explains the relationship between program config parameters (`in0_block_w`, `per_core_M`), hardware tile dimensions, and achieved throughput.
  - Scopes the chapter to expert matmuls only (not the gate linear, which uses HiFi4 and is addressed separately).

- `program_config_tuning.md`
  - Detailed guide to understanding `MatmulMultiCoreReuseMultiCast1DProgramConfig` fields as used in `_make_sparse_matmul_program_config` (lines 62–91).
  - Derives the valid range of `in0_block_w` for GLM-4-MoE hidden/intermediate sizes (tiles = hidden_dim / 32) and for Bailing's sizes.
  - Derives the valid range of `per_core_M` given the padded token count at batch=1 decode (effective M after `SPARSITY_BLOCK_SIZE=32` padding).
  - Provides a tuning grid and evaluation harness: how to benchmark each config combination and select the Pareto-optimal setting.
  - Notes the `min(4, hidden_tiles)` cap and when it is a binding constraint versus a no-op.

- `math_fidelity_evaluation.md`
  - Compares `HiFi2` (current expert matmuls) vs `LoFi` on expert computation.
  - Defines the accuracy metric: per-token cosine similarity and max absolute error of expert outputs relative to FP32 reference.
  - Describes the measurement protocol: run both fidelity settings on the same batch, compare outputs, and record latency delta.
  - Documents the gate linear's `HiFi4` setting and explains why it is held fixed (precision-critical routing decision).
  - Provides a go/no-go criterion for adopting `LoFi` based on accuracy thresholds.

---

### Chapter 5 — `ch5_profiling_methodology/`

**Description:** Establish a repeatable, op-level profiling workflow for `TTNNMoE.forward` on T3K using Tracy, TTNN op timers, and related tooling.

**Files:**

- `index.md`
  - Explains why op-level granularity is necessary (model-level timing cannot distinguish CCL from matmul from routing overhead).
  - Lists the three available profiling tools and when each is appropriate: Tracy (visual, timeline), TTNN op timers (programmatic, CSV-exportable), and device-side cycle counters.

- `tracy_profiling_setup.md`
  - Step-by-step guide to enabling Tracy instrumentation for a `TTNNMoE.forward` call on T3K.
  - Explains how to annotate the key regions in `moe.py` (all-gather, expert dispatch, sparse matmuls, reduce-scatter) with Tracy zones.
  - Describes how to read the resulting timeline to identify the critical path.

- `ttnn_op_timer_profiling.md`
  - Guide to using TTNN's built-in op timer infrastructure to collect per-op latencies programmatically.
  - Shows how to extract timing for each call in `TTNNExperts.forward` (lines 1027–1343) and `TTNNMoE.forward` (lines 1346–1496) without modifying the model code.
  - Describes output format and how to aggregate results across multiple forward passes to reduce noise.

- `router_latency_profiling.md`
  - Isolates and measures the cost of `TTNNMoERouterDecode.forward` (lines 855–1024): the 3-pass BF16 centering trick for topk selection.
  - Explains what the 3-pass centering does and why it was introduced (numerical stability in BF16 topk).
  - Compares its latency against a single-pass topk baseline and documents how to implement the baseline for benchmarking.
  - Describes how to measure whether the precision difference is visible in output quality (e.g., routing decision agreement rate, downstream perplexity).

---

### Chapter 6 — `ch6_cpu_fallback_elimination/`

**Description:** Audit and eliminate any remaining CPU execution paths in the MoE stack, with focus on `Glm4MoeNaiveMoeHybrid` and other silent fallback conditions.

**Files:**

- `index.md`
  - Motivates the chapter: a single CPU fallback can cause orders-of-magnitude slowdown and invalidate all other profiling results.
  - Defines "silent fallback" as any code path that runs on CPU without raising an error or warning.

- `glm4_cpu_path_audit.md`
  - Full analysis of `Glm4MoeNaiveMoeHybrid` (lines 559–613): the hardcoded `ttnn = False` flag, the Torch-based expert loop it invokes, and the exact latency cost versus `TTNNMoE`/`TTNNExperts`.
  - Documents how to measure the latency of the CPU path (wall-clock timing of the full `Glm4MoeNaiveMoeHybrid.forward`) and compare it to `TTNNMoE.forward` on the same batch.
  - Provides a migration checklist for moving the GLM-4 path to `TTNNExperts`.

- `fallback_detection_and_testing.md`
  - Systematic approach to detecting all silent CPU fallbacks across `moe.py`: patterns to grep for (`ttnn = False`, `if not ttnn`, `.cpu()`, `.to("cpu")`), and how to add runtime assertions.
  - Describes a test harness that wraps `TTNNMoE.forward` and asserts that no host-device transfers occur during a decode step.
  - Addresses whether any fallback paths are exercised during typical GLM-4-MoE or Bailing inference and how to confirm they are not.

---

### Chapter 7 — `ch7_end_to_end_optimization_summary/`

**Description:** Synthesize findings from all earlier chapters into a prioritized optimization roadmap for `TTNNMoE` on T3K.

**Files:**

- `index.md`
  - States the goal: given profiling data from Chapters 2–5 and the audit from Chapter 6, rank optimizations by expected impact-per-effort and provide a sequenced action plan.

- `optimization_priority_matrix.md`
  - Table mapping each of the 8 research questions to: the chapter that answers it, the expected latency impact (high/medium/low), and implementation complexity (low/medium/high).
  - Covers: CCL topology/link settings (Ch2), expert pipeline bottleneck (Ch3), `sparse_matmul` program config (Ch4), math fidelity (Ch4), weight application overhead (Ch3), CPU fallback elimination (Ch6), router precision cost (Ch5), and profiling toolchain (Ch5).

- `recommended_action_plan.md`
  - Step-by-step action plan ordered by priority: which optimizations to implement first, what to measure after each change, and what regression tests to run.
  - Includes specific code locations in `moe.py` for each recommended change (class names, line ranges, constant names).
  - Notes which optimizations are model-agnostic (apply to both GLM-4-MoE and Bailing) versus model-specific (require separate tuning per model).

---

## Conventions

**Terminology:**

- **CCL** — Collective Communication Library; refers to `all_gather_async` and `reduce_scatter_minimal_async` as used in `TTNNMoE.forward`.
- **sparse matmul** — the `ttnn.linear` calls using `_make_sparse_matmul_program_config`; "sparse" here refers to the token-sparse dispatch pattern, not a sparse weight matrix.
- **decode regime** — single-token autoregressive generation, batch=1 unless otherwise stated.
- **T3K mesh** — the 1×8 Tenstorrent Wormhole device mesh; "1×8" always means 1 row, 8 columns.
- **CPU fallback** — any computation that executes in PyTorch on the host CPU instead of on-device via TTNN.
- **program config** — a `MatmulMultiCoreReuseMultiCast1DProgramConfig` instance created by `_make_sparse_matmul_program_config`.

**Notation:**

- Line references use the format `moe.py:L<N>` (e.g., `moe.py:L62`) for pinpoint citations and `moe.py:L<start>–L<end>` for ranges.
- Latency values are reported in microseconds (µs) unless otherwise noted; throughput in tokens/second.
- Parameter sweeps use the notation `param ∈ {v1, v2, …}`.
- "HiFi2", "HiFi4", "LoFi" refer to `ttnn.MathFidelity` enum values exactly as they appear in the codebase.
- Constants defined in `moe.py` are always written in `SCREAMING_SNAKE_CASE` matching their source definitions (e.g., `SPARSITY_BLOCK_SIZE`, `TOPK_MIN_WIDTH`).

**Formatting rules:**

- All code snippets use fenced code blocks with explicit language tags (`python`, `shell`, etc.).
- Each file begins with a `## Context` section stating which research question(s) it addresses (by number, 1–8 as listed in this plan).
- Tables for measurement data use Markdown pipe tables with aligned columns.
- Class and function names are always rendered in inline code (e.g., `TTNNMoE`, `TTNNExperts`, `_make_sparse_matmul_program_config`).
- File paths within the repo are written as absolute paths from the repo root.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch2 (CCL Latency) | Ch1 | Readers must understand where `all_gather_async` and `reduce_scatter_minimal_async` appear in `TTNNMoE.forward` before measuring them. |
| Ch3 (Expert Pipeline Profiling) | Ch1 | Profiling `TTNNExperts.forward` requires knowing the full op sequence documented in Ch1. |
| Ch4 (Matmul Config and Math Fidelity) | Ch1, Ch3 | Program config tuning is only meaningful after identifying which `sparse_matmul` calls dominate (Ch3); parameters derive from shapes documented in Ch1. |
| Ch5 (Profiling Methodology) | Ch1 | Tracy and op-timer annotations reference specific functions and call sites established in Ch1. Ch5's router profiling section directly references `TTNNMoERouterDecode.forward` introduced in Ch1. |
| Ch6 (CPU Fallback Elimination) | Ch1 | The `Glm4MoeNaiveMoeHybrid` context and its relationship to `TTNNMoE` is established in Ch1's `cpu_fallback_paths.md`. |
| Ch7 (Optimization Summary) | Ch2–Ch6 | Synthesizes all measurement and audit results; cannot be written until Chapters 2–6 are complete. |

**Research question to chapter/file mapping:**

| Research Question | Primary File(s) |
|-------------------|----------------|
| Q1: CCL op latency costs and topology/link/buffer optimality | `ch2_ccl_latency_and_topology/all_gather_linear_topology.md`, `ch2_ccl_latency_and_topology/reduce_scatter_ring_topology.md`, `ch2_ccl_latency_and_topology/ccl_sensitivity_analysis.md` |
| Q2: Which step in `TTNNExperts.forward` dominates at batch=1 decode | `ch3_expert_dispatch_pipeline_profiling/bottleneck_summary.md`, `ch3_expert_dispatch_pipeline_profiling/token_padding_and_dispatch.md`, `ch3_expert_dispatch_pipeline_profiling/sparse_matmul_profiling.md` |
| Q3: Are `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1` optimal for GLM-4-MoE and Bailing | `ch4_matmul_config_and_math_fidelity/program_config_tuning.md` |
| Q4: Is HiFi2 sufficient for expert matmuls; would LoFi improve throughput without accuracy loss | `ch4_matmul_config_and_math_fidelity/math_fidelity_evaluation.md` |
| Q5: Is the `ttnn.repeat` + permute weight application a meaningful overhead; is there a cheaper alternative | `ch3_expert_dispatch_pipeline_profiling/weight_application_overhead.md` |
| Q6: `TTNNGlm4MoeMoE`/`Glm4MoeNaiveMoeHybrid` CPU path latency vs TTNNMoE; silent fallbacks | `ch6_cpu_fallback_elimination/glm4_cpu_path_audit.md`, `ch6_cpu_fallback_elimination/fallback_detection_and_testing.md`, `ch1_moe_forward_pass_anatomy/cpu_fallback_paths.md` |
| Q7: Cost of `TTNNMoERouterDecode` 3-pass BF16 centering vs single-pass topk | `ch5_profiling_methodology/router_latency_profiling.md` |
| Q8: Best way to profile `TTNNMoE.forward` at op-level granularity on T3K | `ch5_profiling_methodology/tracy_profiling_setup.md`, `ch5_profiling_methodology/ttnn_op_timer_profiling.md` |
