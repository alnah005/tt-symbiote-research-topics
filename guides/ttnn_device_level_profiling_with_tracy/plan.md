# Plan: TTNN Device-Level Profiling with Tracy

---

## 1. Audience

**Primary audience:** ML engineers and kernel developers who need to profile individual TTNN operations on Tenstorrent Wormhole hardware to diagnose performance problems — specifically determining whether an op is compute-bound or bandwidth-bound and understanding what device-level timing data means.

**What they already know:**
- TTNN op usage: `ttnn.matmul`, `ttnn.conv2d`, `ttnn.linear`, memory configs, sharding, `ttnn.enable_program_cache()`
- Basic Tenstorrent hardware concepts: Tensix cores, DRAM/L1 memory hierarchy, Wormhole B0 architecture
- Python development and running tt-metal workloads via pytest
- General benchmarking concepts: wallclock timing, warmup runs, avoiding OS jitter

**What they do NOT need to know in advance:**
- How to invoke the Tracy profiler for a pytest run (env vars, CLI flags)
- What each column in `ops_perf_results.csv` means (DEVICE KERNEL DURATION, BRISC/NCRISC/TRISC durations, PM IDEAL, FPU UTIL, etc.)
- How to determine compute-bound vs. bandwidth-bound status from profiler output
- Common causes of low FPU utilization and remediation strategies
- How host dispatch overhead compares to device kernel time for small vs. large ops

This guide fills those gaps directly, starting from profiler invocation mechanics and building to a principled compute-vs-bandwidth analysis workflow.

---

## 2. Chapter List

---

### Chapter 1: Tracy Profiler Fundamentals for TTNN

**Description:** Introduces Tracy's role in the TTNN profiling stack, explains how it integrates with tt-metal, and distinguishes it from the on-device CSV profiler so readers know which tool answers which question.

**Directory:** `ch1_tracy_fundamentals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - One-paragraph orientation: what Tracy captures, what the device CSV profiler captures, and why both are needed
  - Navigation links to all files in the chapter with one-line descriptions
  - Prerequisites checklist: hardware access, tt-metal checkout with profiler build support

- `what_is_tracy.md`
  - Tracy's origins: a real-time, low-overhead C++ profiler designed for game engines, adopted in systems-level performance tooling
  - What Tracy records: CPU zones (named time-bounded spans), GPU zones, memory allocations, frame markers, and free-form text messages
  - The Tracy data model: every event is a named zone with a start timestamp (ns), end timestamp (ns), thread ID, and optional payload
  - The two-process model: the profiled application emits events over a local socket; the Tracy capture server (`tracy-capture`) records them to a binary `.tracy` file
  - How Tracy integrates with tt-metal: the `TRACY_ENABLE` compile-time define activates instrumentation macros in `tt_metal/tools/profiler/tt_metal_tracy.hpp`; when absent, all macros are zero-cost no-ops
  - What tt-metal annotates by default: op dispatch calls, program enqueue events, and trace lifecycle events

- `two_profilers_compared.md`
  - The two complementary profiling tools: Tracy (CPU-side, host timing) and the TTNN device profiler (on-device, kernel timing via Tensix hardware cycle counters)
  - What each tool answers:
    - Tracy: "When did the host enqueue this op, and how long did host-side dispatch take?"
    - Device profiler: "How long did the kernel actually run on Tensix cores, broken down by RISC processor?"
  - Why both are needed to understand total op latency: `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead`
  - When to use Tracy alone (diagnosing host-side bottlenecks, tracing op ordering), device profiler alone (kernel microarchitecture analysis), or both together (attributing gaps between expected and observed throughput)
  - Known blind spots of each tool: Tracy does not see on-device kernel internals; the device CSV profiler does not show host-side dispatch time or inter-op gaps

---

### Chapter 2: Invoking the Profiler for a TTNN Pytest

**Description:** Covers the exact environment variables, build flags, and CLI invocation needed to capture a Tracy trace and a device profiler CSV from a TTNN pytest run, with a working end-to-end example.

**Directory:** `ch2_invocation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Five-step setup checklist: from clean checkout to first `.tracy` file and first `ops_perf_results.csv`
  - Navigation links to all files in the chapter

- `build_requirements.md`
  - The `ENABLE_PROFILER` CMake flag: enables both Tracy CPU-side instrumentation and the on-device cycle-counter profiler (`kernel_profiler.hpp`)
  - The `TRACY_ENABLE` preprocessor define: controlled via CMake; how to verify it is active in the build (symbol table check, `ldd`)
  - Example CMake invocation: `cmake -DENABLE_PROFILER=ON ...` and how `TRACY_ENABLE` is implied
  - The Tracy submodule location in the tt-metal source tree and why client/server version must match
  - Build-time warning: `ENABLE_PROFILER=ON` adds nonzero runtime overhead from hardware cycle-counter reads; do not use for SLA-critical production benchmarks
  - Verifying the build artifact: checking that profiler symbols are present before attempting a capture

- `env_vars_and_flags.md`
  - Complete reference table of all environment variables relevant to TTNN profiling:

    | Variable | Value | Effect |
    |---|---|---|
    | `TT_METAL_DEVICE_PROFILER` | `1` | Enables on-device CSV profiler; writes `ops_perf_results.csv` |
    | `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES` | `0` or `1` | Includes dispatch core timing in device CSV; anchors device time to host time |
    | `TRACY_NO_EXIT` | `1` | Prevents the profiled process from exiting before Tracy finishes flushing all events |
    | `TT_METAL_CLEAR_L1` | `1` | Clears L1 between runs; ensures deterministic starting state for profiling |
    | `TT_METAL_PROFILER_SYNC` | `1` | Adds a host–device sync point after each op; increases timing accuracy at the cost of throughput |

  - How these variables interact: `TT_METAL_DEVICE_PROFILER=1` is required for the CSV; `TRACY_NO_EXIT=1` is required whenever using `tracy-capture` in non-interactive mode
  - When to use `TT_METAL_PROFILER_SYNC=1` and when to leave it off (latency measurement vs. throughput measurement)
  - pytest-specific considerations: setting env vars via `os.environ` in a conftest fixture vs. setting them in the shell before running pytest

- `capture_workflow.md`
  - The three components that run together: (1) the profiled pytest process, (2) `tracy-capture` server, (3) optionally the Tracy GUI
  - Step-by-step: start `tracy-capture -o profile.tracy -f`, then run `pytest tests/ttnn/my_test.py -s` with the required env vars set
  - Output artifacts: `profile.tracy` (Tracy binary database), `ops_perf_results.csv` (device profiler CSV), per-core cycle logs in `tt_metal/tools/profiler/logs/`
  - Common failure mode: Tracy client/server version mismatch — symptom (silent capture of zero events) and fix (rebuild both from same tag)
  - Common failure mode: `TT_METAL_DEVICE_PROFILER=1` set but `ENABLE_PROFILER` not in the build — symptom (CSV produced but all kernel durations are zero) and fix
  - Minimal working example: a 10-line pytest that runs `ttnn.matmul` with profiling enabled and checks that `ops_perf_results.csv` is non-empty

---

### Chapter 3: Reading the ops_perf_results CSV

**Description:** Documents every column in the `ops_perf_results.csv` output, explains the units and derivation of each field, and teaches how to filter and interpret the data for a specific op.

**Directory:** `ch3_csv_reference/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference column cheat sheet (condensed one-liner per column)
  - Navigation links to files in the chapter

- `csv_column_definitions.md`
  - How `ops_perf_results.csv` is produced: the `process_ops_logs.py` post-processing script aggregates per-core cycle-counter logs from `tt_metal/tools/profiler/logs/` into a single summarized CSV
  - How to run post-processing: `python tt_metal/tools/profiler/process_ops_logs.py` — required options, output location, and what happens if it is skipped
  - Complete column reference table:

    | Column | Units | Description |
    |---|---|---|
    | `OP TYPE` | string | TTNN op class name, e.g., `ttnn::operations::matmul::Matmul` |
    | `OP CODE` | string | Internal kernel identifier |
    | `DEVICE ID` | int | Zero-indexed Tenstorrent device |
    | `CORE COUNT` | int | Number of Tensix cores participating in this op |
    | `DEVICE KERNEL DURATION [ns]` | nanoseconds | Total elapsed time from first core start to last core end (wall-clock on device) |
    | `DEVICE KERNEL DURATION [cycle]` | cycles | Same, in raw Tensix core clock cycles |
    | `BRISC KERNEL DURATION [ns]` | nanoseconds | Time the Base RISC (BRISC) processor spent in kernel code |
    | `NCRISC KERNEL DURATION [ns]` | nanoseconds | Time the NoC RISC (NCRISC) processor spent in kernel code |
    | `TRISC0 KERNEL DURATION [ns]` | nanoseconds | Time the first Tensix RISC (math unpacker) spent in kernel code |
    | `TRISC1 KERNEL DURATION [ns]` | nanoseconds | Time the second Tensix RISC (math engine) spent in kernel code |
    | `TRISC2 KERNEL DURATION [ns]` | nanoseconds | Time the third Tensix RISC (math packer) spent in kernel code |
    | `PM IDEAL [ns]` | nanoseconds | Performance model ideal duration — theoretical minimum kernel time assuming perfect memory and compute |
    | `PM IDEAL [cycle]` | cycles | Same, in cycles |
    | `FPU UTIL` | fraction (0.0–1.0) | Ratio of actual FPU throughput to peak FPU throughput (derived from TRISC1 duration and op FLOPs) |
    | `NOC BW UTIL` | fraction (0.0–1.0) | Ratio of actual NoC bandwidth used to peak NoC bandwidth |
    | `INPUT_0_W`, `INPUT_0_Z`, etc. | int | Shape dimensions of the op's first input tensor |
    | `OUTPUT_0_W`, `OUTPUT_0_Z`, etc. | int | Shape dimensions of the op's first output tensor |
    | `MATH FIDELITY` | string | Precision mode used by the math engine, e.g., `HiFi2`, `LoFi` |
    | `DATA FORMAT` | string | Data type of operands, e.g., `BFLOAT16`, `BFLOAT8_B` |

  - How cycle counts are converted to nanoseconds: `duration_ns = cycles / core_clock_Hz × 1e9`; the core clock frequency for Wormhole B0 and how to find it in device configuration
  - How `DEVICE KERNEL DURATION` relates to individual RISC durations: the device kernel duration is the wall-clock span from the earliest start across all cores to the latest end; individual RISC durations reflect what each processor did within that span

- `pm_ideal_and_fpu_util.md`
  - What PM IDEAL represents: the performance model's theoretical minimum duration, computed as `max(compute_cycles, memory_cycles)` — the roofline model in a single number
  - How PM IDEAL is computed for a matmul: `compute_cycles = (M_t × K_t × N_t × 2 × 1024) / FPU_peak_ops_per_cycle`; `memory_cycles = (input_bytes + weight_bytes + output_bytes) / (NoC_BW_bytes_per_cycle × num_active_cores)`
  - How FPU UTIL is derived: `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles` — a value close to 1.0 means the math engine ran near theoretical peak; a value near 0 means the math engine was stalled
  - Why FPU UTIL can exceed 1.0 and how to interpret that (PM IDEAL rounding or measurement noise)
  - The relationship between PM IDEAL and actual duration: `actual_duration ≥ PM_IDEAL` always; the ratio `actual / PM_IDEAL` is the roofline efficiency

---

### Chapter 4: Compute-Bound vs. Bandwidth-Bound Analysis

**Description:** Teaches the roofline model as applied to TTNN ops, explains how to use the CSV columns to classify an op, and provides decision criteria and worked examples for matmul and elementwise ops.

**Directory:** `ch4_compute_vs_bandwidth/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The central question: is the kernel limited by FPU throughput or by NoC/DRAM memory bandwidth?
  - Quick-reference decision table (condensed; full details in `classification_method.md`)
  - Navigation links to files in the chapter
  - Prerequisites: Chapter 3 (understanding PM IDEAL, FPU UTIL, TRISC durations)

- `roofline_model_primer.md`
  - The roofline model: for a given kernel, the achievable throughput is `min(peak_FLOPs/s, peak_BW × arithmetic_intensity)`
  - Arithmetic intensity definition: `AI = total_FLOPs / total_bytes_transferred`; units are FLOPs/byte
  - The roofline ridge point: the arithmetic intensity at which compute throughput equals memory bandwidth throughput; ops with AI above the ridge are compute-bound, below are bandwidth-bound
  - Wormhole B0 hardware ceilings:
    - Peak FPU throughput: 128 FP16 FMA ops per cycle per core (512-bit SIMD on 32-element tiles)
    - Peak NoC bandwidth: 32 bytes/cycle per link (read and write links are independent)
    - Peak DRAM bandwidth: ~300 GB/s aggregate across all channels (system-level)
    - Ridge point for a single matmul core: derived from the ratio of peak FPU to peak NoC BW
  - Why tile size matters: computations are always on 32×32 element tiles; sub-tile shapes incur overhead that lowers effective FPU utilization
  - L1 vs. DRAM access: ops whose operands fit in L1 (192 KB per core) can achieve higher effective bandwidth than DRAM-bound ops

- `classification_method.md`
  - Step-by-step classification procedure using only the CSV columns:
    1. Compute the op's theoretical arithmetic intensity from input/output shapes and the data format byte width
    2. Compare `FPU_UTIL` to a threshold (guideline: `FPU_UTIL > 0.7` → likely compute-bound; `FPU_UTIL < 0.3` → likely bandwidth-bound)
    3. Check `NOC_BW_UTIL`: if high (`> 0.7`) and `FPU_UTIL` is low, the bottleneck is NoC bandwidth; if both are low, suspect other overhead (dispatch, L1 bank conflicts, pipeline stalls)
    4. Cross-check with `DEVICE KERNEL DURATION` vs. `PM IDEAL`: a ratio close to 1.0 means the op is running at the modeled bound; a ratio > 2× means there is unexplained overhead
    5. Examine RISC breakdown: if `TRISC1_DURATION >> TRISC0_DURATION` or `TRISC1_DURATION >> TRISC2_DURATION`, the math engine is pipelined and both unpacker and packer can keep up — compute-bound; if `TRISC0_DURATION ≈ TRISC1_DURATION ≈ TRISC2_DURATION` and all are long, suspect memory stalls feeding all three stages
  - Decision flowchart (in text form) from CSV columns to classification
  - Classification result table:

    | FPU UTIL | NOC BW UTIL | Classification | Primary Bottleneck |
    |---|---|---|---|
    | High (>0.7) | Low (<0.4) | Compute-bound | FPU throughput |
    | Low (<0.3) | High (>0.7) | Bandwidth-bound (NoC) | NoC data movement |
    | Low (<0.3) | Low (<0.3) | Overhead-bound | Dispatch, stalls, or pipeline inefficiency |
    | Medium | Medium | Balanced | Both limits active |

- `worked_examples.md`
  - Worked example 1: large matmul (M=1024, K=4096, N=4096, BF16)
    - Computing theoretical AI and locating on the roofline
    - Reading CSV values: expected FPU UTIL ~0.8+, NOC BW UTIL ~0.3, DEVICE KERNEL DURATION / PM IDEAL ~1.1–1.3
    - Conclusion: compute-bound; FPU is the limiting resource
  - Worked example 2: small matmul (M=32, K=256, N=256, BF16)
    - Computing theoretical AI: much lower due to small tile count
    - Reading CSV values: expected FPU UTIL ~0.1–0.2 due to underutilization of cores with a single tile
    - Conclusion: overhead-bound; core count too large for the problem size; recommendation: reduce core grid
  - Worked example 3: elementwise op (`ttnn.silu` on a [1024, 4096] tensor, BF16)
    - AI is very low (read 2 bytes, write 1 byte, do ~2 FLOPs): always bandwidth-bound
    - Reading CSV values: FPU UTIL ~0.05, NOC BW UTIL ~0.8
    - Conclusion: bandwidth-bound on NoC; cannot improve without fusing with adjacent ops or reducing data movement

---

### Chapter 5: Low FPU Utilization — Causes and Remediation

**Description:** Enumerates the common root causes of low FPU utilization as seen in the device profiler, explains how to diagnose each from the CSV, and describes the TTNN-level levers to address them.

**Directory:** `ch5_low_fpu_util/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: cause, CSV signature, and fix (condensed; full details in body files)
  - Navigation links to files in the chapter
  - Prerequisites: Chapter 3 (CSV columns), Chapter 4 (roofline model, classification)

- `causes_of_low_fpu_util.md`
  - Cause 1 — Insufficient tile count (small M, K, or N): with fewer tiles than cores, some cores receive no work; `CORE_COUNT` is high but effective utilization per core is low; fix: use fewer cores (`compute_with_storage_grid_size`) or pack multiple ops into a single kernel
  - Cause 2 — Sub-optimal data format (FP32 where BF16 suffices): FP32 halves the number of FMA operations per cycle on Wormhole B0; fix: convert to BF16 or BF8 where the model allows it
  - Cause 3 — Math fidelity mismatch: using `HiFi4` fidelity for a workload that tolerates `LoFi`; `HiFi4` runs ~4× fewer FMA iterations per tile than `LoFi`; fix: specify `math_fidelity=MathFidelity.LoFi` in the op config when precision allows
  - Cause 4 — TRISC0/TRISC2 pipeline stalls: the unpacker (TRISC0) cannot keep up with the math engine (TRISC1) because DRAM reads are slow; observable as `TRISC0_DURATION > TRISC1_DURATION`; fix: increase L1 allocation for double-buffering, or use sharded memory configs to keep operands in L1
  - Cause 5 — NoC contention: multiple active cores issuing reads simultaneously saturate the NoC; observable as high `NOC_BW_UTIL` with low `FPU_UTIL`; fix: reduce the number of active cores or interleave read patterns
  - Cause 6 — Program cache miss (recompilation): the first call with a new shape re-compiles the kernel; TRISC durations are inflated by compile time rather than execution time; observable as anomalously long first-call durations that drop to steady-state on subsequent calls; fix: `ttnn.enable_program_cache()`
  - Cause 7 — Incorrect loop count in kernel: the kernel performs fewer FMA iterations than expected because the loop bounds are computed from padded shapes rather than actual shapes; requires kernel-level investigation

- `csv_signatures.md`
  - For each cause from `causes_of_low_fpu_util.md`, the specific CSV column pattern that identifies it:
    - Cause 1: `FPU_UTIL < 0.2`, `CORE_COUNT > 4 × output_tiles / 32`
    - Cause 2: `DATA_FORMAT == "FLOAT32"`, `FPU_UTIL ≈ 0.5 × expected_BF16_util`
    - Cause 3: `MATH_FIDELITY == "HiFi4"`, `FPU_UTIL ≈ 0.25 × LoFi_util`
    - Cause 4: `TRISC0_DURATION > 1.2 × TRISC1_DURATION`
    - Cause 5: `NOC_BW_UTIL > 0.8`, `FPU_UTIL < 0.3`
    - Cause 6: first-call `DEVICE_KERNEL_DURATION >> 10 × steady_state_duration`
    - Cause 7: `FPU_UTIL` stable across calls but consistently below 0.4 with no other cause present
  - How to use these signatures as a diagnostic checklist: start with Cause 6 (easiest to rule out), then Cause 1, then Cause 3/2, then Cause 4, then Cause 5, then Cause 7

- `remediation_levers.md`
  - `compute_with_storage_grid_size` in `ttnn.matmul` config: controls how many Tensix cores are assigned to the matmul; how to choose it based on tile count (guideline: `M_t × N_t / core_count ≥ 4` for good utilization)
  - `math_fidelity` parameter: `MathFidelity.LoFi`, `MathFidelity.HiFi2`, `MathFidelity.HiFi4` — performance vs. numerical accuracy tradeoffs for each
  - Data format selection via `ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.float32`: how to specify per-tensor and how to verify the format in the CSV
  - Sharded memory configs (`ttnn.MemoryConfig` with `TensorMemoryLayout.HEIGHT_SHARDED`, `BLOCK_SHARDED`, etc.): keeps weight tiles in L1 across calls, eliminating DRAM reads for stationary weights; how to verify L1 usage in profiler output
  - `ttnn.enable_program_cache()`: must be called once before inference; eliminates kernel recompilation on repeated calls with the same shapes
  - Double-buffering via `ttnn.MatmulMultiCoreReuseProgramConfig` `in0_block_w`: controls how many tiles the unpacker prefetches ahead of the math engine; setting this too small causes TRISC0 stalls

---

### Chapter 6: Host Dispatch Overhead vs. Device Kernel Time

**Description:** Quantifies the host-side dispatch cost that adds to every TTNN op's observed latency, explains how it scales with op size, and shows how to use mesh trace capture to eliminate it for latency-sensitive workloads.

**Directory:** `ch6_host_dispatch_overhead/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The key insight: for small ops, host dispatch latency dominates device kernel time; for large ops, device kernel time dominates
  - Navigation links to files in the chapter
  - Prerequisites: Chapter 2 (Tracy capture), Chapter 3 (CSV columns and device kernel duration)

- `what_is_dispatch_overhead.md`
  - The TTNN dispatch pipeline: Python call → TTNN C++ op dispatch → program creation (or cache hit) → command queue enqueue → device MMIO write → device firmware decode → kernel launch
  - Where time is spent on the host: op argument validation, memory config resolution, tile layout computation, program object creation, host-to-device command buffer write
  - Where time is spent on the device before kernel execution begins: firmware command decode, core grid configuration, NoC route setup, DRAM descriptor writes
  - Total dispatch overhead for a typical TTNN op: ~5–50 µs for a cache-hit call, ~100–500 µs for a first-call (cache miss with recompilation)
  - Why this matters for small ops: a `ttnn.add` on a [32, 32] tensor may have a device kernel time of ~1 µs but a host dispatch overhead of ~10 µs — the op is 10× slower than its kernel would suggest

- `measuring_dispatch_vs_kernel.md`
  - Using Tracy to measure host-side dispatch: the Tracy zone for `ttnn.matmul` begins when the Python call is made and ends when the enqueue completes; this is the host dispatch overhead
  - Using the device CSV to measure kernel time: `DEVICE_KERNEL_DURATION` in `ops_perf_results.csv` is the pure on-device execution time
  - The gap between them: `observed_latency = dispatch_overhead + device_kernel_duration + host_device_sync_time`
  - Worked data: expected dispatch overhead and kernel time for representative ops at small and large sizes:

    | Op | Shape | Device Kernel (µs) | Host Dispatch (µs) | Ratio (dispatch/kernel) |
    |---|---|---|---|---|
    | `ttnn.matmul` | [32, 32] × [32, 32] | ~0.5 | ~8 | ~16× |
    | `ttnn.matmul` | [1024, 4096] × [4096, 1024] | ~200 | ~12 | ~0.06× |
    | `ttnn.add` | [32, 4096] | ~0.3 | ~6 | ~20× |
    | `ttnn.softmax` | [1, 1, 32, 4096] | ~2 | ~7 | ~3.5× |

  - How to measure the host dispatch overhead directly: use `TT_METAL_PROFILER_SYNC=1` with Tracy to ensure the Tracy zone end aligns with op completion, then subtract the CSV `DEVICE_KERNEL_DURATION`
  - When dispatch overhead is negligible: once device kernel time > ~10× dispatch overhead, focus optimization effort on the kernel rather than dispatch

- `eliminating_dispatch_overhead.md`
  - The mesh trace mechanism: `ttnn.begin_trace_capture`, `ttnn.end_trace_capture`, `ttnn.execute_trace` — records the entire device command sequence once and replays it without any host involvement
  - How trace capture eliminates dispatch overhead: the replay path sends a single "execute trace" MMIO write to the device, bypassing all per-op dispatch logic; device firmware executes the pre-recorded command sequence directly
  - Expected speedup from trace capture for decode-regime workloads (many small ops repeated N times): typically 3–10× reduction in total latency for op sequences dominated by dispatch overhead
  - Constraints on trace capture: tensor shapes must not change between capture and replay; ops that modify host state (Python callbacks, tensor reads back to host) cannot be inside a trace; `ttnn.allocate_tensor_on_device` must be used for outputs inside the trace
  - How to verify trace execution using the device profiler: device kernel durations should be identical between the captured run and traced replay runs; any difference indicates trace capture failed silently
  - When NOT to use trace capture: during model development (shapes change frequently), for ops that require shape-dependent branching, or when correctness validation requires reading intermediate tensors to host

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **Tracy** | A real-time C++/Python profiler that records named CPU zones with nanosecond timestamps; see [Tracy GitHub](https://github.com/wolfpld/tracy) |
| **Tracy zone** | A named, time-bounded profiling scope recorded by Tracy; analogous to a "span" in distributed tracing |
| **`TRACY_ENABLE`** | The preprocessor define that activates Tracy instrumentation in tt-metal; when absent, all Tracy macros are zero-cost no-ops |
| **`ENABLE_PROFILER`** | The CMake flag that enables both Tracy CPU-side zones and the on-device cycle-counter profiler in tt-metal |
| **device profiler** | The tt-metal on-device profiler that records kernel execution times using Tensix hardware cycle counters; its output is `ops_perf_results.csv` |
| **`TT_METAL_DEVICE_PROFILER`** | Environment variable (set to `1`) that enables the device profiler at runtime |
| **`ops_perf_results.csv`** | The post-processed device profiler output produced by `process_ops_logs.py`; one row per op per call |
| **DEVICE KERNEL DURATION** | The wall-clock elapsed time on device from the first core's kernel start to the last core's kernel end for a single op invocation |
| **BRISC** | Base RISC — the Tensix processor responsible for kernel dispatch and control flow |
| **NCRISC** | NoC RISC — the Tensix processor responsible for NoC data movement (DRAM reads/writes) |
| **TRISC0** | First Tensix RISC — the math unpacker; prepares tile data for the FPU |
| **TRISC1** | Second Tensix RISC — the math engine (FPU); executes FMA operations on tiles |
| **TRISC2** | Third Tensix RISC — the math packer; writes FPU output tiles back to L1 |
| **PM IDEAL** | Performance Model Ideal — the roofline-model theoretical minimum kernel duration for a given op shape and data format |
| **FPU UTIL** | FPU Utilization — the ratio of PM IDEAL cycles to actual TRISC1 kernel duration cycles; indicates how efficiently the math engine is used |
| **NOC BW UTIL** | NoC Bandwidth Utilization — the ratio of actual NoC bytes transferred to peak possible NoC bytes for the kernel duration |
| **tile** | The 32×32 element atomic compute unit on Tenstorrent Wormhole hardware |
| **M_t, K_t, N_t** | Tile counts for matmul dimensions: `M_t = M / 32`, `K_t = K / 32`, `N_t = N / 32` |
| **arithmetic intensity (AI)** | FLOPs per byte of data transferred; determines whether a kernel is compute-bound or bandwidth-bound |
| **ridge point** | The arithmetic intensity at which FPU throughput equals memory bandwidth throughput; the boundary between compute-bound and bandwidth-bound regimes |
| **dispatch overhead** | The host-side time spent preparing and enqueuing an op before device execution begins |
| **program cache** | The tt-metal kernel compilation cache; a cache hit reuses the previously compiled kernel binary without recompilation |
| **mesh trace** | A pre-recorded device command sequence played back via a single MMIO write, eliminating per-op host dispatch |
| **`ttnn.execute_trace`** | The TTNN function that replays a previously captured mesh trace on the device |
| **math fidelity** | The precision mode of the Tensix FPU: `LoFi` (fewest iterations, lowest precision), `HiFi2` (balanced), `HiFi4` (highest precision, slowest) |
| **Wormhole B0** | The Tenstorrent ASIC generation targeted by this guide; all hardware ceilings and timing examples are for Wormhole B0 |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions where helpful, e.g., `[seq_len, d_model]`.
- Tile counts use the notation `M_t`, `K_t`, `N_t` where `M_t = M / 32`.
- Latency values for human-scale discussion are quoted in milliseconds (ms); per-op device profiler values are quoted in microseconds (µs); raw cycle counts are quoted in cycles without unit suffix.
- Formulas use `×` for multiplication and `/` for division; no ambiguous `*` syntax.
- Environment variables are written in `SCREAMING_SNAKE_CASE`, e.g., `TT_METAL_DEVICE_PROFILER`.
- CSV column names are written in `ALL CAPS WITH SPACES` matching the actual column headers, e.g., `DEVICE KERNEL DURATION [ns]`; when used as prose references they are rendered in backtick code style.
- Code blocks use Python syntax and assume `import ttnn`, `import torch`, and `import time` are in scope unless otherwise noted.
- Performance numbers are indicative and based on Wormhole B0 silicon; always re-profile on the target system.
- Blockquote formatting conventions:
  - `> **Warning:** ...` — correctness pitfalls and measurement traps
  - `> **Tip:** ...` — performance recommendations
  - `> **Note:** ...` — contextual clarifications that are not warnings

### Formatting Rules

- Every chapter directory has an `index.md` providing an overview, learning objectives, a quick-reference element (table or checklist), and navigation links to all files in the chapter.
- Code examples are fenced with ` ```python ` and include inline comments explaining non-obvious lines.
- Shell commands use ` ```bash ` fencing; environment variable assignments are shown as `VAR=value command` on a single line for clarity.
- Tables are used for comparisons, column references, decision matrices, and worked data; prose is used for conceptual explanation and sequential procedures.
- Every content file (any `.md` that is not `index.md`, `b_review.md`, `compression_analysis.md`, or `plan.md`) ends with a navigation footer:
  - If not the last file in the chapter: `---\n\n**Next:** [\`next_file.md\`](./next_file.md)`
  - If the last file in the chapter but not the last chapter: `---\n\n**Next:** [Chapter N+1 — Title](../chN+1_title/index.md)`
  - If the last file of the entire guide: `---\n\n**End of guide.** Return to [Guide Index](../index.md)`

---

## 4. Cross-Chapter Dependencies

The guide is designed to be read front-to-back. The table below clarifies which chapters depend on concepts introduced in earlier chapters and which specific terms must remain consistent across files.

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: Tracy Profiler Fundamentals | None (foundational) |
| Ch 2: Invoking the Profiler for a TTNN Pytest | Ch 1 (Tracy zone model, two-process architecture, `TRACY_ENABLE` define, device profiler vs. Tracy distinction) |
| Ch 3: Reading the ops_perf_results CSV | Ch 2 (how `TT_METAL_DEVICE_PROFILER=1` and `process_ops_logs.py` produce the CSV; output artifact names) |
| Ch 4: Compute-Bound vs. Bandwidth-Bound Analysis | Ch 3 (PM IDEAL, FPU UTIL, NOC BW UTIL, TRISC duration columns, DEVICE KERNEL DURATION) |
| Ch 5: Low FPU Utilization — Causes and Remediation | Ch 3 (CSV column signatures), Ch 4 (roofline model, classification method, the concept of the ridge point) |
| Ch 6: Host Dispatch Overhead vs. Device Kernel Time | Ch 1 (Tracy zones for host-side measurement), Ch 2 (Tracy capture workflow), Ch 3 (DEVICE KERNEL DURATION as pure kernel time) |

**Specific cross-file consistency requirements:**

- Ch 1 (`two_profilers_compared.md`) introduces the equation `total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead`. Ch 6 (`measuring_dispatch_vs_kernel.md`) expands on this equation. Both files must use identical variable names and the same sign convention.

- Ch 2 (`env_vars_and_flags.md`) defines the complete set of environment variables. Ch 6 (`measuring_dispatch_vs_kernel.md`) references `TT_METAL_PROFILER_SYNC=1` as a measurement aid. The variable name and its described effect must match exactly between these files.

- Ch 3 (`csv_column_definitions.md`) defines `DEVICE KERNEL DURATION [ns]` as the wall-clock span from first core start to last core end. Ch 4 (`roofline_model_primer.md`) and Ch 6 (`measuring_dispatch_vs_kernel.md`) both reference this definition. All three must use consistent phrasing.

- Ch 3 (`pm_ideal_and_fpu_util.md`) defines `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles`. Ch 4 (`classification_method.md`) and Ch 5 (`csv_signatures.md`) use numerical thresholds on FPU UTIL. These threshold values must be consistent with the definition in Ch 3.

- Ch 4 (`roofline_model_primer.md`) states the Wormhole B0 hardware ceilings (peak FPU ops/cycle, peak NoC BW, peak DRAM BW). Ch 5 (`causes_of_low_fpu_util.md`) references the effect of math fidelity on effective FPU ops/cycle. Both must cite the same base hardware numbers.

- Ch 4 (`worked_examples.md`) establishes the worked example shapes and expected FPU UTIL values for representative ops. Ch 5 (`csv_signatures.md`) uses similar shapes when describing CSV signatures. Both files must use the same representative shapes to avoid contradicting each other.

- Ch 5 (`remediation_levers.md`) introduces `ttnn.begin_trace_capture` / `ttnn.execute_trace` as a lever to eliminate dispatch overhead. Ch 6 (`eliminating_dispatch_overhead.md`) covers this mechanism in full. Ch 5 must not redefine the mechanism — it should cross-reference Ch 6 for details.

- Ch 6 (`measuring_dispatch_vs_kernel.md`) contains the worked data table of dispatch overhead vs. kernel time for representative op shapes. Ch 1 (`two_profilers_compared.md`) claims small ops have dispatch-dominated latency. The worked data in Ch 6 must corroborate (not contradict) the qualitative claim in Ch 1.
