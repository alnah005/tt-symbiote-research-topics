# Research Guide Plan: T3K Mesh Device Optimizations

**Topic:** T3K Mesh Device Optimizations for TTNNQwen3MoE Expert Parallelism
**Guide Version:** 1.0 (initial plan)
**Date:** 2026-03-16

---

## 1. Audience

**Primary audience:** ML systems engineers and kernel developers who are integrating or optimizing large language model inference (specifically Mixture-of-Experts models such as Qwen3MoE) on Tenstorrent T3K hardware.

**What they already know:**
- General LLM inference concepts: prefill vs. decode phases, KV cache, token generation
- Basic Tenstorrent TTNN API: tensor creation, operations, and device placement
- Familiarity with distributed inference concepts (tensor parallelism, pipeline parallelism)
- Python and C++ proficiency at an intermediate level
- General understanding of hardware memory hierarchies (cache, DRAM)

**What they do NOT need to know in advance:**
- T3K-specific topology or Ethernet link details (covered in Chapter 1)
- TTNN MeshDevice multi-chip API internals (covered in Chapter 2)
- MoE expert parallelism strategies specific to T3K (covered in Chapters 3–5)
- Memory configuration best practices for Wormhole (covered in Chapter 4)

---

## 2. Chapter List

### Chapter 1 — T3K Hardware Topology and Interconnect Fundamentals
**Description:** Establishes the physical layout of the T3K 1x8 mesh, chip-to-chip Ethernet link topology, and raw bandwidth characteristics that underpin all subsequent optimization decisions.

**Directory:** `ch01_t3k_topology/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Brief summary of what the chapter covers
  - Prerequisites and pointers to external Tenstorrent documentation
  - Navigation links to each sub-topic file

- `t3k_physical_layout.md` — Physical and logical organization of the T3K system
  - Description of the 1x8 Wormhole device mesh (8 chips on a single board)
  - Logical device IDs and coordinate system used by TTNN (row, col indexing)
  - Distinction between intra-board and inter-board links (relevant when T3K is part of a larger multi-board setup)
  - Diagram description: mesh grid, device IDs 0–7, neighbor adjacency

- `ethernet_link_bandwidth.md` — Chip-to-chip Ethernet link characteristics
  - Number of Ethernet links per Wormhole device and how they are wired on T3K
  - Unidirectional vs. bidirectional bandwidth figures (GB/s per link, aggregate)
  - Link latency characteristics and how they affect collective operation latency
  - Comparison of direct-neighbor vs. non-neighbor hops (multi-hop routing)
  - How bandwidth scales with `num_links` parameter: theoretical vs. achievable throughput
  - Known saturation thresholds and contention effects when multiple collectives run concurrently

- `topology_implications_for_collectives.md` — How T3K topology shapes collective operation design
  - Why ring-based all-reduce and all-to-all patterns map naturally to a linear 1x8 mesh
  - Hop count analysis for different collective patterns (ring vs. tree vs. direct)
  - Implications for expert parallelism: which devices hold which experts and the resulting communication pattern
  - Introduction to the concept of `num_links` as a tunable parameter (detailed in Chapter 3)

---

### Chapter 2 — TTNN MeshDevice API for Multi-Chip Operations
**Description:** Covers the TTNN abstractions for managing a multi-device T3K system, including MeshDevice creation, tensor sharding across devices, and the collective communication primitives available.

**Directory:** `ch02_ttnn_mesh_api/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Summary of TTNN multi-device concepts introduced in this chapter
  - Relationship to Chapter 1 topology knowledge

- `mesh_device_setup.md` — Creating and configuring a MeshDevice for T3K
  - `MeshDevice` constructor parameters: mesh shape `(1, 8)`, device IDs, dispatch configuration
  - Device ordering conventions and how logical coordinates map to physical chip IDs
  - Initialization sequence: opening devices, programming mesh firmware, worker threads
  - Teardown and resource cleanup patterns
  - Common pitfalls: incorrect mesh shape, mismatched device IDs, stale device state

- `tensor_distribution.md` — Distributing tensors across the 8-device mesh
  - `TensorSpec` and `ShardSpec` concepts for describing how a tensor is split across devices
  - Row-wise vs. column-wise sharding and when each is appropriate
  - Replicated vs. sharded tensors: memory and bandwidth trade-offs
  - How weight tensors for MoE experts are placed across devices
  - API patterns: `ttnn.from_torch` with multi-device placement, `ttnn.to_device` for mesh targets

- `collective_primitives.md` — All-to-all, all-reduce, and other collective operations
  - `ttnn.all_to_all`: purpose, signature, key parameters (`num_links`, `memory_config`, `cluster_axis`)
  - `ttnn.all_reduce`: use cases distinct from all-to-all (weight gradient sync vs. activation routing)
  - `ttnn.reduce_scatter` and `ttnn.all_gather`: when to use each in MoE context
  - Synchronization semantics: blocking vs. async dispatch, event-based synchronization
  - Error handling: shape mismatches, device unreachable, link errors

---

### Chapter 3 — All-to-All Operations and `num_links` Tuning on T3K
**Description:** Deep-dives into the `all_to_all` collective as the critical communication primitive for MoE expert dispatch and combine, and systematically addresses how to select optimal `num_links` settings on T3K.

**Directory:** `ch03_all_to_all_num_links/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Why all-to-all is the dominant collective in MoE inference
  - What `num_links` controls and why it is the primary T3K tuning knob

- `all_to_all_in_moe.md` — Role of all-to-all in MoE expert dispatch and combine
  - Token routing phase: sending activated-token embeddings to the device holding each selected expert
  - Expert compute phase: local matmul on each device for tokens routed to it
  - Combine phase: returning expert outputs back to the originating device via all-to-all
  - Data volume analysis: how sequence length, hidden dimension, and top-K affect all-to-all payload size
  - Comparison of prefill all-to-all (large batch, high volume) vs. decode all-to-all (small batch, latency-sensitive)

- `num_links_parameter.md` — Understanding and tuning `num_links`
  - Definition: `num_links` controls how many of the available Ethernet links are used per collective operation
  - Valid range on T3K Wormhole devices and default value behavior
  - Bandwidth vs. link-count relationship: near-linear scaling up to contention point
  - Impact on latency: fewer links may reduce per-message overhead for tiny payloads
  - Impact on throughput: more links increase peak bandwidth for large payloads
  - Interaction with concurrent operations: link sharing and saturation when multiple collectives are in-flight
  - Recommended values:
    - Prefill (large tensors, throughput-bound): use maximum available `num_links`
    - Decode (small tensors, latency-bound): benchmark 1 vs. 2 vs. max; often 1–2 links optimal
    - Single-operation vs. pipelined multi-operation scenarios

- `benchmarking_num_links.md` — Methodology for empirically finding optimal `num_links`
  - Setting up a minimal benchmark harness using TTNN profiler and device-side performance counters
  - Sweeping `num_links` from 1 to max and measuring latency and throughput
  - Interpreting results: identifying the knee in the latency-vs-links curve
  - Controlling for variability: warm-up iterations, outlier filtering
  - Reference benchmark results table for representative tensor sizes on T3K (to be filled during research)
  - How to re-run benchmarks after firmware or driver updates

---

### Chapter 4 — Memory Configuration: L1 vs. DRAM for Decode and Prefill
**Description:** Explains the Wormhole memory hierarchy (L1 SRAM per core vs. DRAM), provides decision criteria for choosing memory placement of activations and intermediate buffers, and gives concrete recommendations for decode vs. prefill phases.

**Directory:** `ch04_memory_config/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Memory hierarchy overview at a glance
  - When to read this chapter vs. accepting defaults

- `wormhole_memory_hierarchy.md` — L1 and DRAM on Wormhole devices
  - L1 SRAM: per-core capacity, aggregate capacity across all Tensix cores, bandwidth to compute units
  - DRAM: per-device capacity, number of DRAM channels, peak bandwidth (GB/s)
  - Comparison table: L1 vs. DRAM — latency, bandwidth, capacity, persistence across operations
  - L1 banking and allocation granularity: minimum allocation size, alignment constraints
  - DRAM interleaving: how TTNN distributes data across DRAM banks for bandwidth maximization
  - Circular buffers (CBs) in L1: their role in the TTNN kernel execution model

- `memory_config_api.md` — TTNN memory configuration API
  - `ttnn.MemoryConfig` dataclass: `buffer_type` (L1, DRAM), `shard_spec`, `memory_layout`
  - `TensorMemoryLayout` variants: `INTERLEAVED`, `WIDTH_SHARDED`, `HEIGHT_SHARDED`, `BLOCK_SHARDED`
  - How to pass `memory_config` to operations (all_to_all, matmul, softmax, etc.)
  - Querying current memory placement of a tensor
  - Explicit tensor movement: `ttnn.to_memory_config` for migrating between L1 and DRAM
  - Memory pressure monitoring: how to detect L1 overflow and fallback behavior

- `decode_memory_strategy.md` — Memory placement recommendations for decode phase
  - Characteristics of decode: batch size 1–32, single new token per step, latency-critical
  - KV cache placement: DRAM preferred (large, persistent, sequential access pattern)
  - Activation buffers for the current token: L1 preferred (small, reused every step, latency-sensitive)
  - Expert routing scores and top-K indices: L1 preferred (tiny tensors, hot path)
  - All-to-all input/output buffers during decode: L1 when tensor fits; DRAM fallback for large top-K
  - Practical L1 budget estimation for a single decode step in Qwen3MoE configuration
  - Trade-off table: L1 placement benefits vs. risk of allocation failure under peak load

- `prefill_memory_strategy.md` — Memory placement recommendations for prefill phase
  - Characteristics of prefill: large sequence lengths (512–32K tokens), throughput-critical
  - Activation tensors: too large for L1 at long sequences; DRAM interleaved recommended
  - Attention intermediate buffers (Q, K, V projections): L1 sharded for short sequences, DRAM for long
  - All-to-all input/output buffers during prefill: DRAM interleaved due to volume
  - KV cache generation during prefill: written directly to DRAM
  - Chunked prefill strategy: breaking long sequences into chunks that fit L1 activations
  - Interaction between prefill memory layout and subsequent decode memory layout

---

### Chapter 5 — Expert Parallelism on T3K: Mapping Experts to Devices
**Description:** Covers strategies for distributing MoE expert weights across the 8 T3K devices to minimize communication volume, balance load, and maximize utilization of both compute and interconnect.

**Directory:** `ch05_expert_parallelism/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Recap of MoE architecture and expert parallelism definition
  - How this chapter ties together topology (Ch 1), API (Ch 2), all-to-all (Ch 3), and memory (Ch 4)

- `expert_placement_strategies.md` — Strategies for assigning experts to devices
  - Naive uniform placement: N_experts / 8 experts per device
  - Load-balanced placement considering routing frequency distributions
  - Locality-aware placement: grouping experts that are co-activated to reduce inter-device traffic
  - Expert replication for hot experts: trade-off between memory cost and communication reduction
  - Qwen3MoE-specific expert counts and how they partition across 8 devices
  - Impact of expert placement on all-to-all payload structure and `num_links` choice

- `token_routing_and_dispatch.md` — Efficient token dispatch for expert parallelism
  - Computing the expert assignment mask on device 0 vs. distributed computation
  - Constructing the all-to-all send buffer: packing tokens by destination device
  - Variable token counts per device: handling uneven routing with padding vs. dynamic shapes
  - Fusing router softmax and top-K selection with all-to-all dispatch preparation
  - Latency breakdown: routing compute vs. all-to-all communication time

- `combine_and_accumulation.md` — Aggregating expert outputs back to originating devices
  - Structure of the reverse all-to-all (combine phase)
  - Weighted combination: applying router scores to expert outputs
  - In-place accumulation on L1 vs. DRAM accumulation buffer strategies
  - Handling the combine for top-K > 1: multi-expert output summation
  - Overlap opportunities: pipelining combine all-to-all with next-layer operations

---

### Chapter 6 — Profiling and Performance Analysis on T3K
**Description:** Teaches how to use TTNN's profiling infrastructure, device-side performance counters, and host-side timing to identify bottlenecks in MoE inference on T3K.

**Directory:** `ch06_profiling/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - Why profiling is non-negotiable before tuning
  - Overview of available profiling tools

- `ttnn_profiler.md` — Using the TTNN performance profiler
  - Enabling profiler capture: environment variables and API calls
  - Op-level timing: per-operation latency breakdown in the dispatch timeline
  - Reading profiler output: CSV format, key columns (op name, device time, host overhead)
  - Identifying all-to-all as a bottleneck vs. compute bottleneck
  - Comparing prefill vs. decode profiles side-by-side

- `device_perf_counters.md` — Device-side performance counters on Wormhole
  - Available counters: Ethernet link utilization, DRAM bandwidth, NOC traffic
  - How to enable and read hardware performance counters via TTNN debug APIs
  - Mapping counter data to operation phases (dispatch, all-to-all, matmul)
  - Detecting link saturation: what counter values indicate `num_links` should be increased
  - Detecting L1 pressure: spill events and their counter signatures

- `bottleneck_diagnosis_guide.md` — Systematic guide to diagnosing performance bottlenecks
  - Decision tree: is the bottleneck compute, memory bandwidth, or communication?
  - Compute-bound indicators and remediation (kernel tile size, data format choices)
  - Memory-bandwidth-bound indicators and remediation (DRAM interleaving, L1 promotion)
  - Communication-bound indicators and remediation (`num_links` increase, expert placement change)
  - Common anti-patterns observed in MoE workloads on T3K

---

### Chapter 7 — End-to-End Integration: TTNNQwen3MoE on T3K
**Description:** Walks through the complete integration of all optimizations into a working TTNNQwen3MoE inference pipeline on T3K, providing a reference configuration and validation methodology.

**Directory:** `ch07_end_to_end_integration/`

**Files:**
- `index.md` — Chapter overview and reading guide
  - What this chapter synthesizes from all prior chapters
  - Target: a fully optimized single-batch decode and chunked prefill pipeline

- `reference_configuration.md` — Recommended T3K configuration for Qwen3MoE
  - MeshDevice initialization parameters for T3K
  - Expert placement mapping (which experts on which device IDs)
  - `num_links` settings per operation type (all-to-all dispatch, combine, any all-reduce)
  - Memory config choices for each tensor class (weights, KV cache, activations, routing buffers)
  - Data format recommendations (bfloat16 weights, float32 accumulation, etc.)
  - Annotated code sketch showing how all parameters come together

- `validation_and_correctness.md` — Validating correctness of the optimized pipeline
  - Numerical correctness checks: comparing T3K output against single-device reference
  - Tolerances for bfloat16 operations and acceptable deviation ranges
  - Testing with varied sequence lengths, batch sizes, and routing distributions
  - Regression test structure: fast smoke tests vs. full correctness suite

- `known_limitations_and_future_work.md` — Current limitations and open questions
  - `num_links` behavior under firmware versions (version-specific notes)
  - L1 allocation failures under edge-case batch/sequence configurations
  - Open questions requiring further benchmarking (to be updated as research progresses)
  - Pointers to upstream TTNN issues and relevant GitHub discussions

---

## 3. Conventions

### Terminology
- **T3K**: The Tenstorrent 1x8 Wormhole mesh board. Always written as "T3K", not "t3k" or "TT-3K".
- **Wormhole**: The Tenstorrent ASIC generation used in T3K. Capitalized as a proper noun.
- **MeshDevice**: The TTNN Python class for multi-device management. Always written in `code font` when referring to the API object.
- **num_links**: The TTNN parameter controlling Ethernet link count for collectives. Written in `code font` as `num_links` (snake_case, no spaces).
- **all_to_all**: The collective operation. Written in `code font` as `ttnn.all_to_all` when referring to the API call, and as "all-to-all" (hyphenated) in prose.
- **L1**: The per-core SRAM on Wormhole Tensix cores. Always "L1" (not "SRAM" or "scratchpad") unless explicitly contrasting with a non-TTNN context.
- **DRAM**: Off-chip memory. Always all-caps "DRAM" in prose.
- **prefill**: The prompt-processing phase of LLM inference. Lowercase, one word.
- **decode**: The token-generation phase. Lowercase, one word.
- **MoE** / **Mixture-of-Experts**: Spelled out on first use in each chapter, then abbreviated as "MoE".
- **expert parallelism**: Lowercase in prose; the strategy of placing different experts on different devices.
- **top-K**: Hyphenated when used as a modifier; K is always capitalized (e.g., "top-2 routing").

### Notation
- Tensor shapes are written in `(batch, seq_len, hidden_dim)` order using parentheses and lowercase names.
- Bandwidth figures are expressed in GB/s (gigabytes per second, base-10).
- Latency figures are expressed in microseconds (µs) for operation-level measurements.
- Device IDs are written as integers 0–7 in prose and as `device_id=N` in code contexts.
- `num_links` values are always written as plain integers (e.g., `num_links=2`), never as fractions or percentages.
- Memory sizes use MiB/GiB (binary) for hardware capacity and MB/GB (decimal) for bandwidth-product calculations; the distinction is noted explicitly where it matters.

### Formatting Rules
- Code blocks use Python syntax highlighting unless the snippet is shell/CLI, in which case `bash` is specified.
- Every chapter's `index.md` must contain a "Prerequisites" section listing which prior chapters must be read first.
- Every file that introduces a new TTNN API symbol must include a "Quick Reference" box at the top with the symbol name, module path, and a one-line description.
- Tables are used for comparison data (L1 vs. DRAM, num_links sweep results); prose paragraphs are used for explanations and rationale.
- Benchmark result tables include a "Status" column marked as `[placeholder — to be filled]` until empirical data is collected.
- Avoid passive voice in headings and section titles.
- Use second-person ("you") for instructional sections and third-person for descriptive sections.

---

## 4. Cross-Chapter Dependencies

The following map shows which later chapters depend on concepts first introduced in earlier chapters. A chapter listed as a dependency must be read before the dependent chapter.

| Chapter | Depends On | Reason |
|---|---|---|
| Ch 2 (TTNN MeshDevice API) | Ch 1 (T3K Topology) | MeshDevice coordinate system and `cluster_axis` parameters require understanding of the physical 1x8 layout and Ethernet link topology. |
| Ch 3 (All-to-All / `num_links`) | Ch 1 (T3K Topology), Ch 2 (TTNN API) | `num_links` tuning requires knowledge of how many Ethernet links exist (Ch 1) and the `ttnn.all_to_all` API signature (Ch 2). |
| Ch 4 (Memory Config) | Ch 2 (TTNN API) | `MemoryConfig` and `TensorMemoryLayout` are TTNN API constructs introduced in Ch 2; Wormhole hardware details from Ch 1 provide context but are not strictly required. |
| Ch 5 (Expert Parallelism) | Ch 1, Ch 2, Ch 3, Ch 4 | Expert placement depends on topology (Ch 1); dispatch uses the MeshDevice API (Ch 2) and all-to-all (Ch 3); memory strategies for expert weights and activations rely on Ch 4. |
| Ch 6 (Profiling) | Ch 2, Ch 3, Ch 4, Ch 5 | Profiling is meaningful only after the operations being measured (Ch 2–5) are understood; bottleneck diagnosis references `num_links` (Ch 3) and memory config (Ch 4) as remediation levers. |
| Ch 7 (End-to-End Integration) | Ch 1 through Ch 6 | The reference configuration synthesizes all hardware, API, communication, memory, expert parallelism, and profiling knowledge from all prior chapters. |

### Specific Concept Forward-References
- **`num_links`**: First introduced conceptually in `ch01_t3k_topology/ethernet_link_bandwidth.md`; further discussed for collective algorithm design in `ch01_t3k_topology/topology_implications_for_collectives.md`; formally defined and tuned in `ch03_all_to_all_num_links/num_links_parameter.md`; referenced in `ch05_expert_parallelism/expert_placement_strategies.md` and `ch07_end_to_end_integration/reference_configuration.md`.
- **L1 vs. DRAM trade-off**: Hardware basis in `ch04_memory_config/wormhole_memory_hierarchy.md`; applied to decode in `ch04_memory_config/decode_memory_strategy.md` and to prefill in `ch04_memory_config/prefill_memory_strategy.md`; referenced in `ch05_expert_parallelism/combine_and_accumulation.md` and `ch07_end_to_end_integration/reference_configuration.md`.
- **Bandwidth characteristics**: Established in `ch01_t3k_topology/ethernet_link_bandwidth.md`; referenced in `ch03_all_to_all_num_links/num_links_parameter.md` and `ch06_profiling/device_perf_counters.md`.
- **Decode vs. prefill distinction**: Defined in `ch04_memory_config/index.md` (brief); fully treated in `ch04_memory_config/decode_memory_strategy.md` and `ch04_memory_config/prefill_memory_strategy.md`; applied to all-to-all in `ch03_all_to_all_num_links/all_to_all_in_moe.md`; integrated in `ch07_end_to_end_integration/reference_configuration.md`.
- **Expert routing and top-K**: Introduced in `ch05_expert_parallelism/token_routing_and_dispatch.md`; assumed known in `ch06_profiling/bottleneck_diagnosis_guide.md` and `ch07_end_to_end_integration/validation_and_correctness.md`.
