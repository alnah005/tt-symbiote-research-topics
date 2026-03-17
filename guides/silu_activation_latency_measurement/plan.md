# Plan: SiLU Activation Latency Measurement

---

## 1. Audience

**Primary audience:** ML engineers optimizing Mixture of Experts (MoE) inference on Tenstorrent hardware (Wormhole B0, T3K mesh). They are comfortable with:

- PyTorch, tensor operations, and standard transformer model architecture
- Basic TTNN op usage: `ttnn.matmul`, `ttnn.linear`, `ttnn.to_device`, memory configs
- Python-level profiling (e.g., `time.perf_counter`, `torch.cuda`-style warm-up loops)
- Conceptual understanding of MoE FFN layers and top-K routing

**What they do NOT need to know in advance:**

- The internal structure of Tenstorrent's Tensix compute engine or SFPU pipeline
- How activation functions are scheduled relative to matmul tiles at the hardware level
- How TTNN's `fused_activation` parameter maps to kernel-level instruction scheduling
- The difference between compute-bound and memory-bound regimes on Wormhole B0

This guide fills those gaps progressively, starting from SiLU's architectural role in MoE, moving through Wormhole hardware execution, and building toward actionable latency measurement and fusion recommendations.

---

## 2. Chapter List

---

### Chapter 1: SiLU in MoE Architecture

**Description:** Establishes where SiLU (and its SwiGLU variant) appears in the MoE FFN computation graph, what role it plays, and why its latency matters relative to the dominant matmul operations.

**Directory:** `ch01_silu_in_moe_architecture/`

**Files:**

- `index.md`
  - Chapter overview, learning objectives, and prerequisites checklist
  - Navigation to sub-topics within the chapter
  - Glossary of terms introduced: SiLU, SwiGLU, gate projection, up projection, FFN block

- `ffn_compute_graph.md`
  - Structure of a standard dense FFN: two matmuls with a single non-linearity (`silu(x) * linear(x)` for SwiGLU)
  - How the MoE FFN differs from a dense FFN: expert weight slicing, per-expert dispatch, and the role of activation in each expert's computation
  - Annotated diagram of the SwiGLU variant used in Mixtral, Qwen-MoE, and DeepSeek-MoE: `gate_proj`, `up_proj`, `down_proj`, and the SiLU gate applied to `gate_proj` output before element-wise multiply with `up_proj`
  - Why SiLU is applied per-token per-expert: tensor shapes entering the activation op and why shape variability from expert dispatch makes latency non-trivial

- `swiglu_variant.md`
  - Mathematical definition: `SwiGLU(x, W, V, b, c) = SiLU(xW + b) * (xV + c)`
  - How SiLU is defined: `SiLU(x) = x * sigmoid(x)` and its derivative properties
  - Difference between a standalone `ttnn.silu` call and a fused matmul+activation pattern
  - Which production MoE models use SwiGLU vs. plain SiLU, with concrete examples (Llama 3, Mixtral 8x7B, Qwen2-MoE)

- `compute_role_and_cost_hypothesis.md`
  - Why activation functions are traditionally considered "free" in transformer benchmarks but are potentially non-trivial on SFPU-based architectures
  - Hypothesis: SiLU requires non-linear SFPU instructions (sigmoid requires polynomial approximation), unlike ReLU which maps to a single clamp; this may make it measurably more expensive on Wormhole
  - Preliminary cost framing: what fraction of total MoE forward-pass time could realistically belong to activation ops given typical hidden sizes (2048–8192) and expert counts (8–64)

---

### Chapter 2: SiLU on Wormhole Hardware

**Description:** Explains how SiLU executes inside the Tensix SFPU unit on Wormhole B0, contrasting its cycle cost with FPU-based matrix multiply and establishing the theoretical performance model.

**Directory:** `ch02_silu_on_wormhole_hardware/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Pointer to official TT-Metalium documentation on compute engines and SFPU LLK APIs

- `tensix_compute_engine.md`
  - Overview of the three execution units inside each Tensix core: RISC-V unpacker/packer, FPU (matrix engine), and SFPU (vector engine)
  - FPU role: tile-level matrix multiply-accumulate operations; throughput in terms of tiles/cycle
  - SFPU role: element-wise scalar operations applied to 32-element lanes; SFPU LReg is 32 elements wide at 32 bits each
  - Why FPU and SFPU are distinct pipelines: they cannot both operate simultaneously on the same output tile; post-matmul activation is a sequential SFPU pass
  - Data type considerations: BF16 matmul output must be converted for SFPU processing; BFP8 path differences

- `silu_sfpu_execution.md`
  - How `ttnn.silu` maps to SFPU Low-Level Kernel (LLK) instructions on Wormhole
  - SiLU decomposition at the instruction level: `sigmoid(x)` requires iterative polynomial approximation (not a single instruction); multiplication by `x` is separate
  - Comparison with simpler activations: ReLU = one conditional clamp, GELU = more expensive approximation
  - The SFPU LReg bottleneck: 32-element register width means a BF16 tile (32x32 = 1024 elements) requires 32 sequential SFPU passes through the register file
  - Expected relative cost ordering: ReLU < SiLU < GELU in SFPU cycles per tile

- `cycles_vs_matmul.md`
  - FPU throughput for a matmul tile: peak multiply-accumulate rate and how it scales with subblock configuration
  - Estimated SFPU cycle budget for SiLU per output tile: sigmoid polynomial depth and multiply step
  - Arithmetic intensity comparison: matmul is compute-bound (FLOPs/byte >> 1); SiLU on SFPU is memory-bound (reads output tiles from L1, transforms, writes back)
  - Roofline model introduction: how to place matmul and SiLU activation on the same roofline plot for Wormhole B0
  - Quantitative estimate: for a typical MoE expert with hidden_dim=4096 and sequence tokens=32, what is the expected ratio of SiLU latency to matmul latency

---

### Chapter 3: Measuring SiLU Latency in TTNN

**Description:** Provides a concrete methodology for isolating and measuring `ttnn.silu` latency using TTNN's built-in profiling infrastructure, including warm-up, dispatch overhead subtraction, and result interpretation.

**Directory:** `ch03_measuring_silu_latency/`

**Files:**

- `index.md`
  - Chapter overview, learning objectives, and required environment (tt-metal installed, METAL_TRACE_MODE or Tracy profiler available)
  - Links to official profiling documentation: `docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html`

- `profiling_setup.md`
  - TTNN profiling infrastructure overview: Tracy-based `profile_this.py` script, `ops_perf_results_*.csv` output format
  - Environment variables for enabling op-level profiling: `TT_METAL_DEVICE_PROFILER`, device profiler read cadence
  - How to structure a benchmark script: device initialization, tensor allocation to L1 vs. DRAM, warm-up iterations (minimum two to populate program cache), timed measurement loop
  - Reading `ReadDeviceProfiler` correctly: when to call it explicitly to avoid dropping ops beyond the 1000-op automatic collection limit
  - Toolchain: `tt-metal/tools/profiler/`, TTNN Visualizer for interactive trace inspection

- `isolating_silu_from_matmul.md`
  - The measurement challenge: in a real MoE forward pass, SiLU is always preceded and followed by matmuls; naive end-to-end timing conflates all three
  - Isolation strategy 1: standalone benchmark — allocate a pre-filled output tensor matching the matmul output shape, call `ttnn.silu` alone, measure device op time in the CSV
  - Isolation strategy 2: difference measurement — benchmark `matmul` alone vs. `matmul + ttnn.silu` with identical inputs, report the delta as SiLU cost
  - Handling async dispatch: TTNN operations are dispatched asynchronously; use `ttnn.synchronize_device(device)` before stopping the clock for host-side timing
  - Controlling tensor layout: ensure both the reference matmul output and the standalone SiLU input use `TILE_LAYOUT` and the same dtype (BF16 recommended for Wormhole)

- `measurement_methodology.md`
  - Recommended input tensor shapes for MoE expert FFN benchmarking: `[1, 1, num_tokens, hidden_dim]` where `num_tokens` in {1, 8, 32, 128} and `hidden_dim` in {2048, 4096, 8192}
  - Statistical rigor: minimum 20 timed iterations after warm-up, report median and p95 latency (not mean, to avoid outliers from device dispatch jitter)
  - CSV column guide: `DEVICE KERNEL DURATION [ns]` is the on-device time; `OP TO OP LATENCY [ns]` includes dispatch overhead — use the former for hardware comparison
  - Interpreting multi-core vs. single-core results: `ttnn.silu` launches on the full core grid by default; note the core count used and normalize if comparing across hidden dims
  - Pitfalls: first-run cache miss latency inflation, dtype mismatches causing unexpected format conversions, DRAM-backed tensors vs. L1-sharded tensors giving different results

---

### Chapter 4: SiLU vs. Matmul Latency Comparison

**Description:** Uses the measurement methodology from Chapter 3 to produce a quantitative comparison of SiLU latency against the surrounding matmul operations in a realistic MoE expert compute sequence.

**Directory:** `ch04_silu_vs_matmul_comparison/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Dependency: requires profiling methodology from Chapter 3

- `roofline_analysis.md`
  - Wormhole B0 roofline parameters: 131 TFLOP/s BF16 peak FPU throughput, DRAM bandwidth (~300 GB/s per chip), L1 bandwidth (substantially higher, per-core ~hundreds of GB/s aggregate)
  - Positioning matmul on the roofline: arithmetic intensity for a square matmul vs. a skinny token-batched matmul (e.g., 32 tokens × 4096 → 4096 × 8192 down-projection)
  - Positioning SiLU on the roofline: arithmetic intensity = 2 FLOPs per element (multiply + sigmoid) / 2 bytes per element (BF16 read + write) ≈ 1 FLOP/byte — deeply memory-bound
  - Expected regime: matmul becomes compute-bound at large batch sizes; SiLU remains memory-bound at all practical MoE token counts
  - Annotated roofline diagram placeholder with both operations marked

- `latency_ratio_by_shape.md`
  - Table: for each `(num_tokens, hidden_dim)` combination, expected and measured latency of `gate_proj matmul`, `ttnn.silu`, `up_proj matmul`, and element-wise multiply
  - Expected finding: at small token counts (1–8 tokens, decode phase), SiLU latency may represent 15–40% of the gate_proj matmul time because both are memory-bandwidth-limited at small batch
  - Expected finding: at larger token counts (128+, prefill phase), SiLU shrinks to < 5% of total FFN time as matmul becomes compute-bound and scales faster
  - How to read the table and draw conclusions about where to focus optimization effort
  - Discussion: why SiLU latency matters more in decode (single-token) than in prefill

- `compute_vs_memory_bound_regimes.md`
  - Detailed walk-through of the compute-bound / memory-bound transition for matmul as batch size grows
  - How SiLU's memory-bound nature means its latency scales linearly with tensor size (unlike matmul which can saturate compute)
  - Practical threshold: the token count above which fusing SiLU into matmul stops providing meaningful speedup
  - Summary decision table: `num_tokens < 16` → fusion likely beneficial; `num_tokens >= 64` → SiLU latency negligible, no need to fuse

---

### Chapter 5: Fused Activation Strategies in TTNN

**Description:** Covers the mechanisms available in TTNN to fuse SiLU (and SwiGLU) with the preceding matmul, how to configure them, and the conditions under which fusion is architecturally valid.

**Directory:** `ch05_fused_activation_strategies/`

**Files:**

- `index.md`
  - Chapter overview, learning objectives, and dependency on Chapters 2 and 4
  - Summary of available fusion mechanisms in TTNN as of tt-metal v0.59+

- `ttnn_fused_activation_api.md`
  - `ttnn.matmul` and `ttnn.linear` `activation` parameter: how to pass `"silu"`, `"relu"`, `"gelu"` as a post-op fusion string
  - `fused_activation` parameter in program configs (`MatmulMultiCoreReuseMultiCastProgramConfig`, `MatmulMultiCoreReuseMultiCast1DProgramConfig`): when to use the program_config-level parameter vs. the top-level parameter (required when using sharded tensors)
  - What fusion means at the kernel level: the SFPU activation pass is folded into the same kernel dispatch as the FPU matmul tiles, eliminating a separate op launch and L1 round-trip
  - Data type constraints: `activation_dtype` in `MatmulMultiCoreReuseMultiCastProgramConfig` controls the intermediate accumulation type; using BFP8_B for activation output can reduce L1 pressure
  - Code example: `ttnn.matmul(gate_proj_weights, x, activation="silu")` vs. the two-op alternative

- `swiglu_fusion_pattern.md`
  - SwiGLU fusion challenge: SwiGLU requires `silu(gate_proj(x)) * up_proj(x)` — two separate matmuls and one element-wise multiply; only the SiLU into `gate_proj` is fusible in a single `ttnn.matmul` call
  - Pattern A: fuse SiLU into `gate_proj` matmul, then issue a separate element-wise multiply (`ttnn.mul`)
  - Pattern B: separate SiLU + mul ops with no fusion — baseline for comparison
  - Pattern C: custom tt-metal kernel that computes gate_proj, applies SiLU, multiplies by up_proj output in a single kernel — requires Metalium kernel authoring, not covered in TTNN high-level API
  - Practical recommendation: Pattern A (fuse SiLU into gate_proj) is the correct default; it eliminates the L1 round-trip for the activation tensor
  - Tensor shapes and memory config requirements for Pattern A to be valid

- `activation_dtype_and_precision.md`
  - How `activation_dtype=ttnn.bfloat8_b` affects accuracy vs. throughput for SiLU output
  - BFP8_B vs. BF16 for activation: BFP8_B halves the L1 footprint of the activation tensor, important for large hidden dims with sharded layouts
  - When to use BF16 activation dtype: when downstream element-wise multiply precision is sensitive (e.g., fine-tuning, low-precision accumulation errors)
  - Accuracy validation approach: compare model output logits with and without BFP8_B activation dtype across representative MoE benchmark inputs

---

### Chapter 6: Performance Impact and Recommendations

**Description:** Synthesizes the findings from all previous chapters into concrete configuration recommendations for SiLU/SwiGLU in production MoE inference on Wormhole B0 and T3K, with guidance on when to invest in fusion and when latency is negligible.

**Directory:** `ch06_performance_impact_and_recommendations/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary of key findings from Chapters 3–5 in bullet form
  - Quick-reference recommendation table for common MoE configurations

- `when_fusion_helps.md`
  - Decision framework: map `(num_tokens, hidden_dim, dtype)` to expected SiLU-to-matmul latency ratio; identify the regime where fusion provides >5% end-to-end speedup
  - Decode phase (1–16 tokens): memory-bound regime; both matmul and SiLU are limited by L1/DRAM bandwidth; fusion eliminates extra kernel dispatch overhead and L1 read/write round-trip — most beneficial
  - Prefill phase (64–2048 tokens): compute-bound regime; matmul dominates; SiLU overhead is a small fraction; fusion still eliminates dispatch overhead but yield is smaller (1–3% range)
  - T3K multi-chip MoE: expert parallelism distributes experts across chips; each chip handles fewer experts with the same token batch; per-chip token count may remain small, preserving the decode-phase analysis
  - Anti-pattern: applying `activation="silu"` when the matmul output is consumed by multiple subsequent ops — fusion forces recomputation or requires saving the unfused value separately

- `configuration_recommendations.md`
  - Recommended configuration table for three scenarios:
    1. Decode (1–8 tokens), BF16, standard MoE FFN: use `ttnn.matmul(gate_proj, x, activation="silu")` + `ttnn.mul`; set `activation_dtype=ttnn.bfloat16`
    2. Decode (1–8 tokens), BFP8 inference: use fused activation with `activation_dtype=ttnn.bfloat8_b`; validate accuracy delta
    3. Prefill (128+ tokens): fusion is optional; standalone `ttnn.silu` is acceptable; prioritize matmul program config tuning over activation fusion
  - How to set `fused_activation` in sharded matmul program configs for large hidden dim experts
  - Verification checklist: confirm fusion is active by checking Tracy profiler CSV — a fused op will show a single `MatmulMultiCoreReuse*` kernel entry with no separate `silu` kernel entry

- `measurement_summary_and_next_steps.md`
  - Summary of expected latency numbers from Chapter 4 measurements: absolute SiLU latency range (expected ~2–20 µs on Wormhole depending on tensor size), matmul latency range for the same shapes
  - Interpretation: SiLU is not negligible at decode-phase token counts; it is a real contributor to per-expert FFN latency
  - Open questions and future work: (1) does a custom Metalium kernel fusing the full SwiGLU pattern (gate_proj + silu + up_proj + mul) provide additional benefit? (2) how does SiLU latency scale on T3K with expert parallelism reducing per-chip token count? (3) does BFP8 activation dtype hurt MoE perplexity at INT8-quantized weight scales?
  - Pointers to related guides: `moe_optimization_techniques_for_ttnn` for batched matmul strategies; `t3k_mesh_device_optimizations` for multi-chip expert dispatch

---

## 3. Conventions

### Terminology Table

| Term | Definition |
|---|---|
| SiLU | Sigmoid Linear Unit: `SiLU(x) = x * sigmoid(x)`; used as the gate activation in SwiGLU FFN blocks |
| SwiGLU | Swish-Gated Linear Unit: `silu(gate_proj(x)) * up_proj(x)`; the FFN non-linearity used in Llama 3, Mixtral, Qwen-MoE |
| SFPU | Special Function Processing Unit; the vector engine inside each Tensix core on Wormhole; handles element-wise ops including activations |
| FPU | Floating-Point Unit; the matrix engine inside each Tensix core; handles tile-level matrix multiply-accumulate (FMA) |
| LLK | Low-Level Kernel; hardware-level instruction sequence compiled for SFPU or FPU operations |
| TILE_LAYOUT | TTNN tensor layout where data is stored in 32×32 tiles; required for FPU/SFPU execution |
| BF16 | bfloat16; 16-bit floating point format (1 sign, 8 exponent, 7 mantissa); default dtype on Wormhole |
| BFP8_B | bfloat8_b; block floating-point 8-bit format used in TTNN for reduced-precision activations |
| fused_activation | TTNN parameter on `matmul`/`linear` and program configs that merges the activation SFPU pass into the matmul kernel |
| activation_dtype | TTNN program config field controlling the precision of the fused activation output tensor |
| gate_proj | The weight matrix in SwiGLU whose output is passed through SiLU before element-wise multiply |
| up_proj | The weight matrix in SwiGLU whose output is multiplied with the SiLU-gated gate_proj output |
| down_proj | The final projection weight matrix in the FFN block, applied after SwiGLU |
| MoE | Mixture of Experts; transformer architecture variant where each token is routed to a sparse subset of FFN experts |
| per-expert latency | Time to execute one expert's FFN computation (gate_proj + silu + up_proj + mul + down_proj) for its assigned tokens |
| arithmetic intensity | Ratio of floating-point operations to bytes transferred; determines whether an op is compute-bound or memory-bound |
| roofline model | Performance model plotting achievable throughput against arithmetic intensity, bounded by compute and memory bandwidth |
| T3K | Tenstorrent 3000 series multi-chip board with 8 Wormhole B0 chips connected via ethernet |
| Wormhole B0 | Tenstorrent's current-generation AI accelerator chip; 80 Tensix cores, 131 TFLOP/s BF16 peak |

### Notation Conventions

- Tensor shapes are written as `[batch, seq, rows, cols]` using TTNN's 4D convention; 2D tensors are written as `[1, 1, M, K]`.
- Latency values are reported in microseconds (µs) at the device kernel level (from Tracy CSV `DEVICE KERNEL DURATION [ns]` divided by 1000) unless stated otherwise.
- TTNN API names are written in `monospace` (e.g., `ttnn.silu`, `ttnn.matmul`).
- Parameter names and config fields are written in `monospace` (e.g., `fused_activation`, `activation_dtype`).
- Hardware unit names (FPU, SFPU, Tensix) are capitalized throughout.
- All code examples are Python using the TTNN Python API.

### Formatting Rules

- Each chapter `index.md` must begin with a `## Prerequisites` section listing chapters that must be read first.
- Tables use GFM pipe table syntax.
- Code blocks specify language: ` ```python ` for Python examples, ` ```bash ` for shell commands.
- Forward references use the pattern: *See Chapter N, `filename.md` for details.*
- Measurement result placeholders are marked with `[MEASURED: <description>]` to indicate where actual benchmark data should be inserted when the guide is executed.

---

## 4. Cross-Chapter Dependencies

### Dependency Table

| Chapter | Depends On | Reason |
|---|---|---|
| Ch1: SiLU in MoE Architecture | None | Foundational; no prerequisites |
| Ch2: SiLU on Wormhole Hardware | Ch1 | Requires understanding of where SiLU appears in the compute graph before explaining how it executes |
| Ch3: Measuring SiLU Latency | Ch1, Ch2 | Measurement setup requires knowing which tensor shapes to use (Ch1) and what hardware counters to interpret (Ch2) |
| Ch4: SiLU vs. Matmul Comparison | Ch2, Ch3 | Roofline analysis requires hardware model from Ch2; latency numbers require measurement methodology from Ch3 |
| Ch5: Fused Activation Strategies | Ch2, Ch4 | Fusion mechanisms require SFPU execution model from Ch2; fusion decision requires latency ratio knowledge from Ch4 |
| Ch6: Performance Impact and Recommendations | Ch3, Ch4, Ch5 | Synthesis chapter; all measurement, comparison, and fusion results must be established first |

### Forward References to Flag

The following forward references exist in earlier chapters and should be clearly marked with cross-reference links in the final content:

1. **Ch1 → Ch5** (`compute_role_and_cost_hypothesis.md`): The cost hypothesis section mentions that fusing SiLU with matmul may eliminate the SFPU dispatch overhead. This is a forward reference to the `ttnn_fused_activation_api.md` content in Ch5.

2. **Ch2 → Ch3** (`silu_sfpu_execution.md`): The SFPU pass count per tile (32 passes for a BF16 tile through a 32-element SFPU LReg) is referenced when explaining the isolation methodology in Ch3's `isolating_silu_from_matmul.md`.

3. **Ch2 → Ch5** (`cycles_vs_matmul.md`): The roofline model introduced in Ch2 is used again in Ch4 and referenced in Ch5's `when_fusion_helps.md` to justify the decode-phase recommendation.

4. **Ch3 → Ch6** (`measurement_methodology.md`): The statistical rigor rules (median over 20 iterations, `DEVICE KERNEL DURATION` column) defined in Ch3 are assumed throughout Ch6's `measurement_summary_and_next_steps.md`.

5. **Ch4 → Ch6** (`latency_ratio_by_shape.md`): The `num_tokens < 16` / `num_tokens >= 64` threshold identified in Ch4 is directly cited in Ch6's `when_fusion_helps.md` decode vs. prefill decision framework.

6. **Ch5 → Ch6** (`swiglu_fusion_pattern.md`): Pattern A (fuse SiLU into gate_proj matmul) is defined in Ch5 and referenced by name in Ch6's `configuration_recommendations.md` without re-explaining.
