# Plan: Expert Weight Memory Layout Optimization

---

## 1. Audience

**Primary audience:** ML engineers and performance engineers who are deploying or optimizing Mixture of Experts (MoE) transformer models on Tenstorrent hardware (Wormhole B0, T3K mesh). They are comfortable with:

- PyTorch tensor operations and model deployment
- Basic TTNN op usage: `ttnn.matmul`, `ttnn.to_device`, `ttnn.DRAM_MEMORY_CONFIG`, `ttnn.L1_MEMORY_CONFIG`
- High-level understanding of how MoE layers work (routing, expert FFNs)
- Python profiling and performance measurement at the op level

**What they do NOT need to know in advance:**

- The internals of TTNN's sharding subsystem or how `ShardSpec` maps to physical DRAM controllers
- How NoC routing affects memory access latency and bandwidth under different shard layouts
- The tile-level (32×32) alignment constraints that govern valid shard shapes
- The specific DRAM controller topology of Wormhole B0 and how it interacts with weight prefetch

This guide fills those gaps progressively, starting from the Wormhole memory hierarchy and building toward a fully tuned DRAM-sharded expert weight configuration with validated performance gains.

---

## 2. Chapter List

---

### Chapter 1: TTNN Memory Architecture

**Description:** Establishes the foundational concepts of Wormhole's memory hierarchy, TTNN's `MemoryConfig` API, and the distinction between interleaved and sharded placement as a prerequisite for all subsequent sharding discussion.

**Directory:** `ch01_ttnn_memory_architecture/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation map to sub-topics
  - Prerequisites checklist (what readers must already know)
  - Summary table: DRAM_MEMORY_CONFIG vs L1_MEMORY_CONFIG vs sharded configs at a glance

- `wormhole_memory_hierarchy.md`
  - Wormhole B0 physical memory topology: 6 DRAM controllers, 12 GDDR6 banks (2 per controller, 1 GB per bank), 12 GB total
  - L1 SRAM per Tensix core: 1.5 MB shared across all 5 RISC-V sub-cores on a tile
  - Total on-chip L1: ~120 MB across the full 8×8 Tensix grid (minus harvested rows)
  - DRAM bandwidth vs L1 bandwidth: approximate figures for Wormhole B0 and why the gap matters for weight-heavy inference
  - The NoC (Network-on-Chip): how Tensix cores request data from DRAM tiles and the latency/bandwidth model for NoC hops

- `memory_config_api.md`
  - The `ttnn.MemoryConfig` class: constructor arguments `memory_layout`, `buffer_type`, and optional `shard_spec`
  - `ttnn.BufferType` enum: `DRAM` vs `L1` — what each controls in buffer allocation
  - `ttnn.TensorMemoryLayout` enum: `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `SINGLE_BANK` — definitions and intended use cases
  - Predefined configs `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG` as shorthand for the default interleaved cases
  - Code pattern: creating a custom `MemoryConfig` with `buffer_type=DRAM` and a non-default layout

- `interleaved_vs_sharded.md`
  - How interleaved allocation works: round-robin page distribution across all DRAM controllers at page-size granularity
  - What "interleaved" means in practice for weight tensors: each tile of a weight matrix lands on a different controller in rotation
  - How sharded allocation works: the tensor is partitioned into contiguous chunks (shards), each assigned to a specific set of cores or DRAM banks
  - Why interleaved can cause NoC contention under heavy parallel access: multiple Tensix cores competing for the same NoC links to reach different DRAM controllers
  - The reshard pattern: allocating a tensor as DRAM-sharded and then resharding into L1-sharded for compute — the canonical flow for large weight tensors

---

### Chapter 2: DRAM-Sharded Memory Layout

**Description:** Provides a complete, precise treatment of DRAM-sharded memory configuration in TTNN: the `ShardSpec` struct, the three `TensorMemoryLayout` sharding strategies, shard grid specification, and how to construct a valid DRAM-sharded `MemoryConfig` for weight tensors.

**Directory:** `ch02_dram_sharded_memory_layout/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: `ShardSpec` field summary
  - Forward pointer: this chapter defines the config primitives used in Ch3–Ch7

- `shard_spec_deep_dive.md`
  - `ttnn.ShardSpec` struct fields: `grid` (`CoreRangeSet`), `shape` (shard shape in elements `[H, W]`), `orientation` (`ShardOrientation.ROW_MAJOR` or `COL_MAJOR`)
  - `ttnn.CoreRange` and `ttnn.CoreRangeSet`: how to express a rectangular or non-contiguous grid of cores as a shard target
  - How `shard_shape` relates to total tensor shape and number of shards: the relationship `shard_shape[0] * num_cores_height = tensor_height` for HEIGHT_SHARDED
  - The `orientation` field: row-major means shards fill cores left-to-right then top-to-bottom; col-major fills top-to-bottom then left-to-right — and how this affects sequential prefetch
  - Worked example: constructing a `ShardSpec` for a weight tensor of shape `[4096, 14336]` distributed across a 1×8 core grid

- `sharding_strategies.md`
  - `HEIGHT_SHARDED` (1D along height): each shard is a contiguous row-block of the tensor; all shards have the same width as the full tensor; natural fit when the computation iterates over rows (e.g., token dispatch into expert)
  - `WIDTH_SHARDED` (1D along width): each shard is a contiguous column-block; all shards have the same height as the full tensor; natural fit when the weight matrix is partitioned column-wise (output dimension sharding)
  - `BLOCK_SHARDED` (2D): each shard is a rectangular sub-block; requires a 2D core grid; reduces both height and width simultaneously; preferred when both M and N dimensions are large enough to benefit from 2D distribution
  - Comparison table: strategy vs tensor dimension partitioned vs typical expert weight use case vs NoC access pattern
  - When DRAM-sharded differs from L1-sharded: for DRAM-sharded buffers the shard grid references DRAM controllers rather than Tensix L1 cores — the shards are stored in DRAM banks, not in on-chip L1

- `constructing_dram_sharded_config.md`
  - Step-by-step: building a `MemoryConfig` with `buffer_type=BufferType.DRAM`, `memory_layout=TensorMemoryLayout.WIDTH_SHARDED`, and a fully specified `ShardSpec`
  - How to call `ttnn.create_sharded_memory_config` with `strategy=ShardStrategy.WIDTH` for DRAM placement (noting the L1-only limitation of this helper and the direct `MemoryConfig` constructor path for DRAM)
  - Using `ttnn.to_memory_config` to move an existing tensor from interleaved DRAM into a sharded DRAM layout
  - Verifying the resulting config: inspecting `tensor.memory_config()`, `tensor.shard_spec()`, and checking that `buffer_type` is `DRAM`
  - Code pattern: full end-to-end snippet for converting an expert weight tensor from `DRAM_MEMORY_CONFIG` to a DRAM-sharded layout

---

### Chapter 3: Expert Weight Tensor Structure

**Description:** Analyzes the shape, dtype, and tiling properties of MoE expert weight tensors (gate, up, and down projections) and maps them to valid shard grids on Wormhole B0, establishing the concrete inputs to sharding configuration decisions.

**Directory:** `ch03_expert_weight_tensor_structure/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Reference table: canonical expert weight shapes for common MoE models (Mixtral 8x7B, DeepSeek-MoE-16B, Qwen MoE)

- `projection_shapes.md`
  - Gate projection: weight shape `[d_model, d_ff]` — maps input tokens to expert hidden dimension
  - Up projection: weight shape `[d_model, d_ff]` — the SwiGLU parallel path to gate projection
  - Down projection: weight shape `[d_ff, d_model]` — maps back from expert hidden to model dimension
  - How `num_experts` multiplies these: stacked weight tensors `[num_experts, d_model, d_ff]` vs individually stored tensors, and the TTNN convention for expert weight storage
  - Concrete shapes for Mixtral 8x7B: `d_model=4096`, `d_ff=14336`, `num_experts=8`, `top_k=2`
  - Concrete shapes for a 64-expert model: `d_model=2048`, `d_ff=8192`, `num_experts=64`, `top_k=6`

- `tensor_to_shard_grid_mapping.md`
  - How the weight tensor shape constrains valid shard grids: the shard height or width must divide the tensor height or width evenly
  - Grid size selection heuristic: prefer grids that align with DRAM controller count (6 or 12 on Wormhole) to maximize bank-level parallelism
  - Column-wise sharding of the down projection (`[d_ff, d_model]`) vs row-wise sharding of gate/up projections (`[d_model, d_ff]`) — which dimension to shard and why
  - How `num_experts` interacts with the grid: whether to shard within-expert (across the weight matrix dimensions) or across-expert (each shard holds one expert's weights)
  - Worked example: `[8, 4096, 14336]` stacked weight tensor — computing valid WIDTH_SHARDED and HEIGHT_SHARDED grid options

- `dtype_and_tile_layout.md`
  - Standard dtypes for expert weights: `bfloat16` (2 bytes/element), `bfloat8_b` (1 byte/element)
  - TILE_LAYOUT vs ROW_MAJOR_LAYOUT: why expert weights must be in TILE_LAYOUT for `ttnn.matmul` and how this affects shard shape calculations
  - In TILE_LAYOUT, the minimum addressable unit is a 32×32 tile (1024 elements × bytes_per_element) — how this establishes a floor on shard dimensions
  - Total per-expert weight memory: example calculation for bfloat16 gate+up+down for Mixtral-scale experts

---

### Chapter 4: Prefetch Patterns and Bandwidth

**Description:** Explains how the choice of shard layout and orientation affects the data access sequence from Tensix cores to DRAM, how NoC contention arises under interleaved access, and what prefetch efficiency gains DRAM-sharded layouts provide.

**Directory:** `ch04_prefetch_patterns_and_bandwidth/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Conceptual diagram: interleaved vs sharded DRAM access pattern under a matmul kernel

- `noc_and_dram_access.md`
  - How a Tensix core issues a read request to DRAM: the NoC packet model (destination X/Y coordinate + address + size)
  - DRAM tile addresses on Wormhole B0: each DRAM controller occupies a fixed column in the NoC grid; there are 6 DRAM columns, each with 2 banks
  - Interleaved access pattern: core (0,0) fetching a weight tile that round-robins across all 6 DRAM columns — each request traverses a different number of NoC hops
  - Why multiple cores fetching interleaved weight tiles simultaneously creates NoC hotspots on the links leading to popular DRAM columns
  - Measured effect: NoC contention under interleaved access can reduce effective DRAM bandwidth by 20–40% under heavy parallel load (indicative range)

- `sharded_access_pattern.md`
  - How DRAM-sharded layout assigns a contiguous shard of the weight tensor to a specific DRAM bank range
  - Locality benefit: a Tensix core assigned to compute with shard N always fetches from the same DRAM bank — the NoC path is deterministic and short
  - How `ShardOrientation.ROW_MAJOR` determines the sequential prefetch order: shards are assigned to cores in row-major order, so adjacent cores hold adjacent weight rows — spatial prefetching is effective
  - How `ShardOrientation.COL_MAJOR` affects access: useful when the matmul K-dimension is partitioned, as column-major orientation aligns shard boundaries with K-tile boundaries
  - Double-buffering interaction: for L1-sharded compute, the DMA engine can prefetch the next shard from DRAM while the current shard is being consumed — shard shape sizing to enable this

- `bandwidth_estimation.md`
  - Simple bandwidth model: `effective_bandwidth = total_bytes_read / kernel_time_s`
  - How to measure with `ttnn.device.EnableMemoryReports()` or Tracy profiling: what metrics to read
  - Estimated bandwidth for interleaved DRAM weight access under full-grid matmul: how to compute the expected vs measured gap
  - Estimated bandwidth improvement from DRAM-sharded layout: reducing NoC fan-out and contention, targeting closer to peak DRAM bandwidth
  - The "roofline" framing: at what arithmetic intensity does the expert matmul become compute-bound vs memory-bound, and where DRAM sharding matters most (decode regime, small batch)

---

### Chapter 5: Tile Size Constraints

**Description:** Covers all tile-level alignment requirements that govern valid shard shapes in TTNN, explains why violations produce errors or silent misalignment, and provides a complete checklist for deriving tile-valid shard configurations for expert weight tensors.

**Directory:** `ch05_tile_size_constraints/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: tile alignment checklist for a given weight tensor shape and shard grid

- `tile_fundamentals.md`
  - The 32×32 tile as the atomic compute and memory transfer unit on Wormhole B0: why all tensor shapes must be multiples of 32 in both dimensions when in TILE_LAYOUT
  - How TTNN automatically zero-pads tensors to the next 32-multiple: the effect on effective tensor shapes and on shard boundary calculations
  - Tile memory footprint by dtype: `bfloat16` tile = 2048 bytes, `bfloat8_b` tile = 1024 bytes — and why shard shapes in bytes must be page-aligned
  - The concept of "tile columns" and "tile rows": `N_t = width / 32`, `M_t = height / 32` — how program configs refer to these rather than element counts

- `shard_shape_alignment_rules.md`
  - Rule 1: shard height must be a multiple of 32 (for TILE_LAYOUT tensors) — what happens when it is not (error at `to_memory_config` or incorrect shard mapping)
  - Rule 2: shard width must be a multiple of 32 — same consequences
  - Rule 3: for BLOCK_SHARDED, both shard height and shard width must independently satisfy the 32-multiple rule
  - Rule 4: `tensor_height / shard_height` must equal the number of rows in the core grid (for HEIGHT_SHARDED), and `tensor_width / shard_width` must equal the number of columns (for WIDTH_SHARDED)
  - Rule 5: for DRAM-sharded configurations, the shard size in bytes should be page-aligned (typically 32 bytes on Wormhole) to avoid partial-page reads
  - Worked example: deriving the shard shape for a `[4096, 14336]` bfloat16 gate-projection weight using WIDTH_SHARDED across 8 DRAM banks — step by step

- `common_pitfalls.md`
  - Pitfall 1: using ROW_MAJOR_LAYOUT for weights that will be sharded — ROW_MAJOR tensors have a different internal memory layout than TILE_LAYOUT tensors, and sharding constraints differ; always convert to TILE_LAYOUT before sharding
  - Pitfall 2: specifying shard shape in tiles rather than elements (or vice versa) — the `ShardSpec.shape` field takes element counts, not tile counts
  - Pitfall 3: non-power-of-2 shard widths that are multiples of 32 but not multiples of the matmul `in0_block_w` — this does not cause an error but causes suboptimal tiling in the downstream matmul kernel
  - Pitfall 4: shard grid that exceeds the number of available DRAM banks — oversubscription leads to multiple shards per bank with no bandwidth benefit
  - Pitfall 5: forgetting that `num_experts` in the batch dimension does not get sharded by the width/height sharding strategy — only the 2D weight matrix dimensions are sharded; the expert batch dimension requires a separate partitioning strategy

---

### Chapter 6: Performance Analysis and Trade-offs

**Description:** Quantifies the performance gain from DRAM-sharded expert weight storage, characterizes the shard setup overhead, and establishes the conditions under which sharding pays off versus when interleaved DRAM is preferable.

**Directory:** `ch06_performance_analysis_and_tradeoffs/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary: when DRAM sharding pays off vs when to stay interleaved (decision table)

- `bandwidth_gain_analysis.md`
  - Theoretical peak DRAM bandwidth on Wormhole B0: approximately 288 GB/s aggregate across 6 controllers
  - Measured effective bandwidth under standard interleaved config for expert weight matmul: typical range 60–80% of peak due to NoC contention and access irregularity
  - Measured effective bandwidth under DRAM-sharded config: expected improvement toward 85–95% of peak when shard-to-core assignment eliminates inter-controller competition
  - The decode regime bottleneck: in single-token decode, each expert FFN is memory-bound (low arithmetic intensity), making weight bandwidth the dominant latency component — DRAM sharding's highest-impact scenario
  - The prefill regime: larger token batches increase arithmetic intensity; compute becomes the bottleneck before peak DRAM bandwidth is saturated — sharding provides diminishing returns as batch size grows
  - Rule of thumb: DRAM sharding for expert weights is most valuable when `batch_size × top_k ≤ 16` (decode-like regime)

- `shard_setup_overhead.md`
  - What "shard setup overhead" means: the one-time cost of calling `ttnn.to_memory_config` to convert a weight tensor from interleaved to sharded DRAM
  - When this cost is paid: at model load time (weights are resharded once before inference starts) vs at inference time (dynamic resharding per forward pass)
  - The recommended pattern: reshard all expert weights to DRAM-sharded at load time and keep them sharded for the lifetime of the model — no per-inference overhead
  - Reshard latency estimate: for a `[4096, 14336]` bfloat16 tensor, `to_memory_config` from interleaved to sharded takes on the order of single-digit milliseconds — negligible for deployment
  - Program cache interaction: sharded weight tensors must have a stable `ShardSpec` for `ttnn.matmul` program caching to work; changing shard config between calls forces recompilation

- `tradeoff_matrix.md`
  - Comparison table across four inference regimes: (a) decode small batch, (b) decode large batch, (c) prefill small batch, (d) prefill large batch
  - For each regime: recommended weight layout (interleaved vs sharded), primary bottleneck (memory vs compute), expected latency delta from switching to sharded
  - When sharding hurts: very large batch prefill where compute is saturated — sharding adds metadata overhead with no bandwidth benefit
  - Interaction with L1 memory pressure: DRAM-sharded weights reduce L1 working set if the matmul kernel can stream directly from DRAM shards without staging in L1 — when this is feasible vs when a DRAM→L1 reshard is still needed
  - Multi-expert parallelism interaction: if experts are already sharded across chips (T3K), per-chip weight sharding compounds with inter-chip sharding — combined configuration guidance

---

### Chapter 7: Implementation and Validation

**Description:** Provides complete, runnable code patterns for converting expert weight tensors to DRAM-sharded layout, integrating sharded weights with `ttnn.matmul`, verifying correctness via PCC, and benchmarking the memory bandwidth gain.

**Directory:** `ch07_implementation_and_validation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - End-to-end implementation checklist: steps from weight loading to validated sharded inference

- `code_patterns.md`
  - Loading expert weights from a checkpoint: constructing the weight tensors on CPU and transferring to device with initial `DRAM_MEMORY_CONFIG`
  - Converting to DRAM-sharded layout: the `ttnn.to_memory_config` call with a fully constructed `MemoryConfig(buffer_type=DRAM, memory_layout=WIDTH_SHARDED, shard_spec=ShardSpec(...))`
  - Constructing the `ShardSpec` programmatically from model config: deriving `shard_shape` from `(d_ff // num_dram_banks, d_model)` and building the `CoreRangeSet` for the DRAM grid
  - Integrating sharded weights with `ttnn.matmul`: how the matmul kernel consumes a DRAM-sharded weight tensor — whether a DRAM→L1 reshard is needed before the matmul call or whether the kernel supports direct DRAM-sharded weight input
  - Handling the three expert projections (gate, up, down) with different shard orientations: gate/up use WIDTH_SHARDED along `d_ff`, down uses HEIGHT_SHARDED along `d_ff`
  - Context manager for device and program cache: ensuring static shapes for program caching with `ttnn.enable_program_cache(device)`

- `correctness_verification.md`
  - PCC (Pearson Cross-Correlation) as the correctness metric: definition and how to compute it between TTNN output and a PyTorch CPU reference
  - Acceptable PCC thresholds: `> 0.9999` for bfloat16, `> 0.999` for bfloat8_b — and how dtype choice affects the threshold
  - How weight memory layout changes should not affect numerical output: verifying that resharding from interleaved to DRAM-sharded produces bit-identical weight reads (since only addressing changes, not values)
  - Step-by-step verification workflow: (1) run expert FFN with interleaved DRAM weights, capture output; (2) reshard weights to DRAM-sharded; (3) run again, verify PCC ≈ 1.0 between the two TTNN outputs
  - Common correctness failure: shard shape misalignment causing partial-tile reads — how to diagnose from the PCC value (low PCC at specific positions vs uniform degradation)

- `benchmark_methodology.md`
  - Setting up a minimal benchmark harness: device init, weight loading, warmup iterations (to fill program cache), timed iterations
  - What to measure: per-call latency (wall time), DRAM read bytes (from device profiler), derived effective bandwidth (bytes / latency)
  - How to read the Tracy profiler output for matmul ops: identifying the DRAM read phase vs compute phase in the trace
  - Comparing interleaved vs DRAM-sharded configs: the two-config benchmark loop — toggle `weight_memory_config` between `DRAM_MEMORY_CONFIG` and the sharded config, keep all other parameters fixed
  - Reporting results: table format showing latency (ms), bandwidth (GB/s), and efficiency (% of peak) for each config across decode and prefill regimes
  - Reproducing the benchmark: hardware setup (single Wormhole B0 vs T3K), firmware version, and TTNN version pinning for result reproducibility

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **expert weight** | The weight matrix (gate, up, or down projection) belonging to a single FFN expert in a MoE layer |
| **shard** | A contiguous sub-tensor assigned to a specific memory bank or core in a sharded memory layout |
| **ShardSpec** | The TTNN struct specifying the shard grid, shard shape in elements, and shard orientation |
| **TensorMemoryLayout** | The enum controlling how shards are distributed: `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `SINGLE_BANK` |
| **BufferType** | The enum selecting the physical memory: `DRAM` or `L1` |
| **MemoryConfig** | The TTNN class combining `TensorMemoryLayout`, `BufferType`, and an optional `ShardSpec` into a complete memory placement specification |
| **interleaved** | Memory layout where tensor pages are distributed round-robin across all banks of the selected `BufferType` |
| **DRAM-sharded** | Memory layout where tensor data is partitioned into contiguous shards stored in specific DRAM banks, enabling deterministic NoC access patterns |
| **reshard** | The operation of converting a tensor from one memory layout/config to another, typically using `ttnn.to_memory_config` |
| **tile** | The 32×32 element atomic unit of compute and memory transfer on Wormhole B0 |
| **M_t, K_t, N_t** | Tile-count dimensions: `M_t = height / 32`, `K_t = inner_dim / 32`, `N_t = width / 32` |
| **NoC** | Network-on-Chip — the on-die interconnect on Wormhole used by Tensix cores and DRAM controllers to exchange data |
| **DRAM controller** | One of the 6 memory controllers on Wormhole B0, each managing 2 GDDR6 banks (2 GB total per controller) |
| **program cache** | TTNN's kernel compilation cache, keyed on op type + input tensor shapes + memory configs; invalidated when any key changes |
| **PCC** | Pearson Cross-Correlation — the standard numerical correctness metric in TTNN development |
| **decode regime** | Inference mode with small batch sizes (batch 1–32, seq_len 1), typically memory-bandwidth-bound |
| **prefill regime** | Inference mode with larger sequence lengths (seq_len 128–8192), typically compute-bound at moderate batch |
| **bfloat16** | Brain float 16-bit format: 1 sign, 8 exponent, 7 mantissa bits — 2 bytes per element |
| **bfloat8_b** | Tenstorrent 8-bit block float format — 1 byte per element; lower precision but halves weight memory |
| **T3K** | The 8-chip Tenstorrent Wormhole mesh product |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions where helpful, e.g., `[num_experts, d_model, d_ff]`.
- Tile counts use subscript-t notation: `M_t`, `K_t`, `N_t` (e.g., `M_t = d_model / 32 = 128` for `d_model=4096`).
- Shard shapes are written in elements as `[shard_H, shard_W]` unless tile counts are specifically discussed (then `[shard_H_t, shard_W_t]`).
- TTNN API names appear in `code font` exactly as they appear in Python: `ttnn.MemoryConfig`, `ttnn.ShardSpec`, `ttnn.TensorMemoryLayout.WIDTH_SHARDED`.
- Performance numbers (bandwidth, latency) are indicative estimates based on Wormhole B0 architecture; always re-profile on the target firmware and TTNN version.
- Code blocks use Python syntax and assume `import ttnn` is in scope; device initialization is shown only in Ch7 and omitted in shorter examples.
- Hardware-specific quantities (core counts, DRAM sizes, bandwidth figures) refer to Wormhole B0 unless explicitly stated otherwise.

### Formatting Rules

- Every chapter directory has an `index.md` providing an overview, learning objectives, and navigation links to chapter files.
- Code examples are fenced with ` ```python ` and include inline comments explaining non-obvious lines.
- Tables are used for comparisons (strategy choices, config matrices, trade-off summaries); prose is used for explanations.
- Alignment constraints and rules are formatted as numbered lists (Rule 1, Rule 2, …) for easy cross-reference.
- Warnings about pitfalls or incorrect usage are formatted as `> **Warning:** ...` blockquotes.
- Performance recommendations are formatted as `> **Tip:** ...` blockquotes.
- All chapter files end with a "Next Steps" section pointing to the next logical file or chapter.
- API field names in prose are always written in `code font` to distinguish them from English words.

---

## 4. Cross-Chapter Dependencies

The guide is designed to be read front-to-back, but the table below clarifies which later chapters rely on specific concepts introduced earlier.

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: TTNN Memory Architecture | None (foundational) |
| Ch 2: DRAM-Sharded Memory Layout | Ch 1 (`MemoryConfig` API, `BufferType`, interleaved concept, NoC overview) |
| Ch 3: Expert Weight Tensor Structure | Ch 1 (TILE_LAYOUT, dtype fundamentals), Ch 2 (`ShardSpec` field definitions, sharding strategies) |
| Ch 4: Prefetch Patterns and Bandwidth | Ch 1 (NoC topology, DRAM controller layout), Ch 2 (shard orientation, shard-to-bank assignment), Ch 3 (concrete weight shapes for examples) |
| Ch 5: Tile Size Constraints | Ch 1 (tile fundamentals, 32×32 tile unit), Ch 2 (`ShardSpec.shape` in elements, strategy-specific grid rules), Ch 3 (expert weight shapes used in worked examples) |
| Ch 6: Performance Analysis and Trade-offs | Ch 4 (bandwidth model, NoC contention quantification), Ch 5 (valid shard configurations for measurement), Ch 3 (decode vs prefill regime definitions) |
| Ch 7: Implementation and Validation | All previous chapters; Ch 2 (config construction), Ch 5 (tile alignment rules for code correctness), Ch 6 (what metrics to measure and what numbers to expect) |

**Specific forward references to flag:**

- Ch 1 (`interleaved_vs_sharded.md`) introduces the "reshard pattern" (DRAM-sharded → L1-sharded → compute) as a forward pointer to Ch 2 and Ch 7 — the full code for this pattern must not appear until Ch 7.
- Ch 2 (`shard_spec_deep_dive.md`) defines the canonical `ShardSpec` field vocabulary — Ch 3, Ch 4, Ch 5, and Ch 7 all use these field names without re-defining them; Ch 2 must be finalized before any downstream chapter is written.
- Ch 3 (`projection_shapes.md`) establishes the canonical example shapes (`d_model=4096`, `d_ff=14336`, Mixtral 8x7B) — all worked examples in Ch 4, Ch 5, and Ch 7 must use these same shapes for consistency.
- Ch 5 (`shard_shape_alignment_rules.md`) enumerates the five tile-alignment rules — Ch 7 (`code_patterns.md`) must implement code that satisfies all five rules and may reference them by number.
- Ch 6 (`bandwidth_gain_analysis.md`) establishes the bandwidth improvement claim (indicative 60–80% → 85–95% of peak) — Ch 7 (`benchmark_methodology.md`) provides the measurement methodology to validate or refine this claim; the numbers in Ch 6 must be marked as estimates pending the Ch 7 benchmark.
- Ch 6 (`tradeoff_matrix.md`) references the decode-regime rule of thumb (`batch_size × top_k ≤ 16`) — this threshold is derived from the arithmetic intensity crossover discussed in Ch 4 (`bandwidth_estimation.md`) and must be consistent between the two chapters.
