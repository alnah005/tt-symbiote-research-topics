# Plan: MoE Optimization Techniques for TTNN

---

## 1. Audience

**Primary audience:** ML engineers and performance engineers who are deploying or optimizing Mixture of Experts (MoE) transformer models on Tenstorrent hardware (Wormhole B0, T3K mesh). They are comfortable with:

- PyTorch and tensor operations
- Transformer model architecture at a conceptual level
- Basic TTNN op usage (ttnn.matmul, ttnn.to_device, memory configs)
- Python profiling and performance measurement

**What they do NOT need to know in advance:**

- The internals of TTNN's dispatch pipeline or Metalium kernel scheduling
- The details of how sparse_matmul differs from a dense matmul at the silicon level
- How program configs map to hardware subblock constraints

This guide fills those gaps progressively, starting from MoE architecture fundamentals and building toward expert-level TTNN kernel configuration.

---

## 2. Chapter List

---

### Chapter 1: MoE Architecture Fundamentals

**Description:** Establishes the vocabulary and structural patterns of Mixture of Experts models as a foundation for all subsequent optimization discussion.

**Directory:** `ch01_moe_architecture_fundamentals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to sub-topics
  - Prerequisites checklist

- `moe_overview.md`
  - What MoE layers are: the gating network, the expert pool, and the dispatch-combine pattern
  - Sparse vs. dense MoE activation: top-K routing and why only a fraction of experts fire per token
  - Common MoE model families (Switch Transformer, Mixtral, DeepSeek-MoE) and their routing strategies
  - How MoE changes the compute graph compared to a dense FFN layer

- `routing_and_sparsity.md`
  - Token-to-expert assignment: how the router produces per-token expert indices and scores
  - Expert capacity, load balancing losses, and dropped tokens
  - The relationship between routing decisions and the shape of downstream compute: why sparse activation creates irregular tensor shapes
  - Definition of the sparsity pattern and sparsity ratio as used throughout this guide

- `moe_on_hardware.md`
  - High-level view of why MoE is challenging on accelerators: irregular memory access, expert imbalance, and synchronization costs
  - Why naively looping over experts in PyTorch is slow and what batched approaches aim to solve
  - Brief preview of the two TTNN approaches covered in this guide: batched matmul and sparse_matmul

---

### Chapter 2: TTNN and Wormhole Hardware Primer

**Description:** Provides the minimal necessary background on Tenstorrent Wormhole architecture and TTNN's programming model so readers can reason about kernel performance.

**Directory:** `ch02_ttnn_wormhole_primer/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Pointer to official TTNN documentation for deeper reference

- `wormhole_architecture.md`
  - Tensix core structure: RISC-V cores, Math engines (FPU/SFPU), and the NoC
  - L1 SRAM per core vs. DRAM bandwidth: the memory hierarchy and its impact on matmul tile scheduling
  - Grid layouts: how a logical 2D core grid maps to physical Tensix cores on Wormhole B0
  - T3K multi-chip mesh: how 8 Wormhole chips are connected via ethernet links and how tensor parallelism extends across the mesh

- `ttnn_programming_model.md`
  - Tensors in TTNN: shapes, dtypes (bfloat16, bfloat8_b), tile layout vs. row-major layout
  - Memory configs: DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG, and interleaved vs. sharded placement
  - The op dispatch model: how a Python ttnn call becomes a Metalium kernel on device
  - Tracing and program caching: why static shapes matter for performance

- `matmul_fundamentals_in_ttnn.md`
  - How ttnn.matmul maps M, K, N dimensions onto the Tensix grid
  - Tile size (32x32) and its role as the atomic compute unit
  - Output subblock constraints: what (out_subblock_h, out_subblock_w) mean and why violating constraints causes fallback or errors
  - Introduction to MatmulMultiCoreReuseMultiCastProgramConfig vs. MatmulMultiCoreProgramConfig — when each is appropriate

---

### Chapter 3: Batched Matmul for MoE — Approach and Performance

**Description:** Covers the batched matmul strategy for MoE expert computation: how to formulate it, what TTNN ops are used, and its performance profile across batch and sequence size configurations.

**Directory:** `ch03_batched_matmul_for_moe/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: batched matmul pros, cons, and recommended use cases

- `formulating_batched_matmul.md`
  - How to gather tokens by expert assignment and stack them into a batched tensor of shape [num_experts, expert_capacity, d_model]
  - Padding to expert capacity: why padding is needed and its effect on FLOP efficiency
  - Using ttnn.matmul in batch mode: the role of the batch dimension in kernel dispatch
  - Expert weight tensor layout: [num_experts, d_model, d_ff] and how it is stored on device

- `program_configs_batched.md`
  - How to select MatmulMultiCoreReuseMultiCastProgramConfig for the batched case
  - Key parameters: in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N
  - How per_core_M and per_core_N scale with expert_capacity and d_ff
  - Example configurations for common shapes: (batch=1, seq=128), (batch=1, seq=2048), (batch=32, seq=128)
  - How to validate a config without running: tile divisibility checks and L1 footprint estimation

- `performance_profile_batched.md`
  - Latency breakdown: gather/scatter overhead vs. pure matmul time
  - Throughput characteristics: how utilization changes as expert_capacity increases
  - When batched matmul is preferred: high batch size, high sequence length, balanced routing
  - Known bottlenecks: padding waste at low load, gather cost on DRAM, recompilation on shape change

---

### Chapter 4: sparse_matmul for MoE — Approach and Performance

**Description:** Introduces TTNN's sparse_matmul operation, explains how it differs from batched matmul at the kernel level, and characterizes its performance for MoE workloads.

**Directory:** `ch04_sparse_matmul_for_moe/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: sparse_matmul pros, cons, and recommended use cases

- `sparse_matmul_internals.md`
  - What sparse_matmul does differently from a dense matmul: skipping zero tiles in the activation tensor
  - The role of the sparsity mask: a per-tile boolean or integer tensor that encodes which tiles are non-zero
  - How the kernel uses the sparsity mask to gate tile reads and avoid redundant FMAs
  - Interaction with program caching: why the sparsity mask shape must be static even if values change

- `when_sparse_matmul_wins.md`
  - Sparsity ratio thresholds: at what fraction of active tiles does sparse_matmul outperform batched matmul
  - Impact of d_model and d_ff on the crossover point
  - Sequence length effects: why sparse_matmul is more advantageous at shorter sequences (lower expert_capacity)
  - Decode vs. prefill regimes: why decode (batch=1 or small batch, seq=1) is the sweet spot for sparse_matmul

- `program_configs_sparse.md`
  - How program configs differ for sparse_matmul vs. standard matmul
  - Setting out_subblock_h and out_subblock_w under sparsity constraints
  - The effect of sparsity ratio on recommended per_core_M: fewer active tiles means different grid utilization
  - Example configurations for decode-regime MoE: (batch=1, seq=1), (batch=8, seq=1), (batch=32, seq=1)

---

### Chapter 5: Sparsity Tensor Construction

**Description:** Provides a complete, practical guide to constructing correct and performant sparsity tensors for use with sparse_matmul in TTNN.

**Directory:** `ch05_sparsity_tensor_construction/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: sparsity tensor construction checklist

- `sparsity_tensor_format.md`
  - Required shape, dtype, and layout for the sparsity tensor accepted by ttnn.sparse_matmul
  - Tile-level vs. element-level sparsity: why TTNN operates at tile granularity (32x32)
  - How the sparsity tensor dimensions map to the activation tensor dimensions (M-tiles, K-tiles)
  - Valid value encodings: 0 for skip, 1 for compute — and what happens with partial tiles

- `constructing_from_router_output.md`
  - Step-by-step: converting router logits → top-K expert indices → token-to-expert mask → sparsity tensor
  - Handling the case where multiple tokens map to the same expert (capacity overflow)
  - Ensuring tile alignment: padding token counts to multiples of 32 before constructing the mask
  - Code pattern: building the sparsity tensor on CPU and transferring to device with the correct memory config

- `sparsity_tensor_placement.md`
  - Where to place the sparsity tensor in memory: L1 vs. DRAM tradeoffs
  - Sharding the sparsity tensor for multi-expert parallelism on T3K
  - Reuse across decode steps: when the sparsity tensor can be cached vs. must be recomputed
  - Interaction with ttnn tracing: how to include sparsity tensor updates inside a trace region

- `common_pitfalls.md`
  - Shape mismatches between activation tensor and sparsity tensor
  - Non-tile-aligned token counts causing silent correctness errors
  - Forgetting to update the sparsity tensor between decode steps in a KV-cache loop
  - Performance regression from placing the sparsity tensor in DRAM instead of L1

---

### Chapter 6: Comparative Analysis — Choosing the Right Approach

**Description:** Synthesizes the findings from Chapters 3–5 into a decision framework for choosing between batched matmul and sparse_matmul based on model configuration and runtime characteristics.

**Directory:** `ch06_comparative_analysis/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Decision flowchart: batched matmul vs. sparse_matmul

- `performance_comparison_matrix.md`
  - Side-by-side latency and throughput comparison across four canonical scenarios:
    1. Prefill, large batch (batch=32, seq=2048)
    2. Prefill, small batch (batch=1, seq=2048)
    3. Decode, large batch (batch=32, seq=1)
    4. Decode, small batch (batch=1, seq=1)
  - How d_model, d_ff, num_experts, and top_k interact with the choice
  - Explanation of why sparse_matmul latency is non-monotonic in sparsity ratio (metadata overhead at low sparsity)

- `memory_and_bandwidth_tradeoffs.md`
  - DRAM bandwidth pressure: batched matmul requires gathering tokens (memory-bound at low utilization) vs. sparse_matmul avoids gather but requires sparsity tensor reads
  - L1 footprint comparison: how each approach uses on-chip memory
  - Multi-chip (T3K) considerations: how expert parallelism interacts with both approaches

- `decision_guide.md`
  - Structured decision rules: sparsity ratio > threshold → sparse_matmul; high expert_capacity → batched matmul
  - How to measure sparsity ratio at runtime and feed it into config selection
  - Hybrid strategy: using batched matmul for prefill and sparse_matmul for decode within the same model
  - When to profile rather than guess: indicators that the default choice may be wrong

---

### Chapter 7: T3K Multi-Chip MoE Optimization

**Description:** Extends the single-chip optimization techniques to the T3K 8-chip mesh, covering expert parallelism, tensor distribution, and the additional tuning knobs available at multi-chip scale.

**Directory:** `ch07_t3k_multi_chip_moe/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisites: Chapters 1–6 must be understood before this chapter

- `expert_parallelism_on_t3k.md`
  - How to partition num_experts across 8 chips: expert parallelism (EP) degree and its interaction with top-K routing
  - All-to-all communication pattern for MoE on T3K: token dispatch and expert result reduction
  - TTNN ops used for cross-chip communication: ccl.all_to_all and ccl.reduce_scatter
  - Latency model: compute time vs. ethernet link latency for different expert distributions

- `sharding_strategies.md`
  - Sharding the activation tensor across the T3K mesh for MoE: row-wise vs. column-wise sharding
  - How to shard expert weight tensors: each chip holds a subset of experts
  - Sparsity tensor sharding for sparse_matmul in the multi-chip case
  - Code pattern: setting up DistributedTensorConfig for MoE weight shards

- `program_configs_t3k.md`
  - How per-chip program configs change when expert parallelism is active
  - Grid utilization per chip when num_local_experts < num_experts
  - Adjusting per_core_M for the reduced local expert capacity
  - Worked example: Mixtral 8x7B on T3K — per-chip config derivation

---

### Chapter 8: End-to-End Optimization Workflow and Troubleshooting

**Description:** Walks through a complete optimization workflow from baseline measurement to tuned deployment, and provides a troubleshooting reference for common failure modes.

**Directory:** `ch08_e2e_workflow_and_troubleshooting/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary: the five-step optimization loop

- `optimization_workflow.md`
  - Step 1: Establish a correct baseline — verifying PCC (peak cross-correlation) against a CPU reference
  - Step 2: Profile to identify the bottleneck — using ttnn.device.profiler and reading Tracy traces
  - Step 3: Select the matmul strategy — applying the decision guide from Chapter 6
  - Step 4: Tune program configs — iterating on per_core_M, out_subblock_h, out_subblock_w
  - Step 5: Validate correctness after tuning — PCC threshold guidelines (typically > 0.999 for bfloat16)

- `correctness_validation.md`
  - What PCC measures and why it is the standard correctness metric in TTNN development
  - How to compute PCC between TTNN output and a PyTorch reference
  - PCC thresholds for different dtypes: bfloat16, bfloat8_b, and mixed-precision MoE
  - Common sources of PCC degradation in MoE: incorrect sparsity mask, wrong gather indices, dtype mismatch

- `troubleshooting_reference.md`
  - Error: "Matmul subblock size does not divide output block" — cause and fix
  - Error: "L1 allocation failed" — how to reduce per_core_M or switch to DRAM memory config
  - Silent correctness error: sparse_matmul output zeros where non-zeros expected — sparsity mask construction bug checklist
  - Performance regression after shape change — program cache invalidation and how to re-tune
  - T3K-specific: "CCL op timeout" during all-to-all in MoE dispatch — expert imbalance and capacity fixes

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **expert** | A single FFN sub-network in the MoE layer |
| **top-K routing** | Router assigns each token to exactly K experts |
| **expert_capacity** | Maximum number of tokens routed to a single expert per forward pass |
| **sparsity ratio** | Fraction of expert slots that receive zero tokens (0.0 = fully dense, 1.0 = fully sparse) |
| **sparsity tensor** | The tile-level boolean/integer mask passed to sparse_matmul |
| **tile** | The 32x32 element atomic compute unit on Tenstorrent hardware |
| **grid** | The 2D array of Tensix cores assigned to an op |
| **program config** | A TTNN data class specifying kernel parameters (e.g., MatmulMultiCoreReuseMultiCastProgramConfig) |
| **PCC** | Pearson Cross-Correlation — the primary correctness metric |
| **EP** | Expert Parallelism — distributing experts across chips |
| **T3K** | The 8-chip Tenstorrent Wormhole mesh product |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions where helpful, e.g., `[batch, seq, d_model]`.
- Tile counts are written as `M_t`, `K_t`, `N_t` (e.g., `M_t = seq / 32`).
- Program config parameters use their exact TTNN Python attribute names (e.g., `per_core_M`, `out_subblock_h`).
- Hardware-specific quantities (number of cores, L1 size) refer to Wormhole B0 unless explicitly stated otherwise.
- Code blocks use Python syntax and assume `import ttnn` and `import torch` are in scope.
- Performance numbers are indicative; always re-profile on the target firmware and model checkpoint.

### Formatting Rules

- Every chapter directory has an `index.md` that provides an overview, learning objectives, and navigation links.
- Code examples are fenced with ` ```python ` and include comments explaining non-obvious lines.
- Tables are used for comparisons (pros/cons, config matrices); prose is used for explanations.
- Warnings about correctness pitfalls are formatted as `> **Warning:** ...` blockquotes.
- Performance-sensitive recommendations are formatted as `> **Tip:** ...` blockquotes.
- All chapter files end with a "Next Steps" section pointing to the next logical file or chapter.

---

## 4. Cross-Chapter Dependencies

The guide is designed to be read front-to-back, but the dependencies below clarify which later chapters rely on concepts introduced earlier.

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: MoE Architecture Fundamentals | None (foundational) |
| Ch 2: TTNN and Wormhole Primer | None (foundational) |
| Ch 3: Batched Matmul for MoE | Ch 1 (routing, expert_capacity, sparsity ratio), Ch 2 (matmul program configs, tile layout) |
| Ch 4: sparse_matmul for MoE | Ch 1 (sparsity ratio, routing), Ch 2 (tile layout, kernel dispatch), Ch 3 (program config vocabulary) |
| Ch 5: Sparsity Tensor Construction | Ch 1 (router output format, top-K indices), Ch 2 (tile granularity, memory configs), Ch 4 (sparsity tensor format accepted by sparse_matmul) |
| Ch 6: Comparative Analysis | Ch 3 (batched matmul performance profile), Ch 4 (sparse_matmul performance profile), Ch 5 (construction cost of sparsity tensor) |
| Ch 7: T3K Multi-Chip MoE | Ch 3 and Ch 4 (single-chip strategies), Ch 5 (sparsity tensor sharding), Ch 6 (decision guide applied per-chip) |
| Ch 8: E2E Workflow and Troubleshooting | All previous chapters; serves as a synthesis and reference chapter |

**Specific forward references to flag:**

- Ch 2 (`matmul_fundamentals_in_ttnn.md`) introduces `out_subblock_h` and `out_subblock_w` — these are used without re-explanation in Ch 3, Ch 4, and Ch 7.
- Ch 5 (`sparsity_tensor_format.md`) defines the canonical sparsity tensor layout — Ch 4 references this definition and must not redefine it independently.
- Ch 6 (`decision_guide.md`) references the performance crossover point first quantified in Ch 4 (`when_sparse_matmul_wins.md`) — Ch 4 must include the sparsity ratio threshold table before Ch 6 is written.
- Ch 8 (`correctness_validation.md`) references PCC thresholds — these must be consistent with any thresholds mentioned in Ch 3 and Ch 4.
