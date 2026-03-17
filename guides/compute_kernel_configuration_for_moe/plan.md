# Plan: Compute Kernel Configuration for MoE

---

## 1. Audience

**Primary audience:** ML engineers and performance engineers who are optimizing Mixture of Experts (MoE) models on Tenstorrent Wormhole B0 hardware using TTNN. They are comfortable with:

- Writing and running TTNN programs (ttnn.matmul, memory configs, program configs)
- Basic transformer and MoE architecture concepts (gate/up/down projections, routing)
- Python profiling and iterative performance tuning
- Reading TTNN op documentation and source-level configs

**What they do NOT need to know in advance:**

- The internal Tensix FPU pipeline stages or how math fidelity is implemented in microcode
- How the Wormhole packer/unpacker pipeline differs from GPU tensor core pipelines
- The formal numerical analysis of floating-point rounding in reduced-precision multiply-accumulate

**Motivation and gap this guide fills:**

DeepSeek-V3 uses `COMPUTE_KERNEL_CONFIG_LOFI` (with `packer_l1_acc=True`) for gate and up projections and `COMPUTE_KERNEL_CONFIG_HIFI2` for down projections. Qwen MoE implementations in tt-transformers do not specify any `compute_kernel_config`, defaulting to TTNN-chosen behavior. This guide explains what each parameter controls, why the DeepSeek choices were made, how to characterize the accuracy and throughput trade-off experimentally, and how to apply this to Qwen MoE expert matmuls for a measurable latency improvement.

---

## 2. Chapter List

---

### Chapter 1: Compute Kernel Config Fundamentals

**Directory:** `ch1_kernel_config_fundamentals/`

**Description:** Introduces `WormholeComputeKernelConfig` as the primary handle for controlling per-op compute behavior on Wormhole B0, and explains the role and default value of each field.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table of all four key fields with their types, defaults, and one-line effects
  - Prerequisites: basic TTNN matmul usage, familiarity with bfloat16 dtype

- `wormhole_compute_kernel_config_api.md`
  - How `ttnn.WormholeComputeKernelConfig` is constructed in Python and passed to `ttnn.matmul` via the `compute_kernel_config` argument
  - The four primary fields: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`
  - Two secondary fields: `dst_full_sync_en` and `throttle_level` (brief description, not covered in depth)
  - What happens when `compute_kernel_config` is omitted: TTNN selects defaults, which are hardware-conservative but not performance-optimal
  - Concrete Python snippet showing the two canonical configs used in production (DeepSeek-V3 `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2`) with exact parameter values

- `fp32_dest_acc_en.md`
  - What the destination accumulator register is in the Tensix FPU pipeline: the register where partial sums land before being packed and written out
  - Default behavior (fp32_dest_acc_en=False): partial sums are truncated to bfloat16 in the destination register after each accumulation step
  - With fp32_dest_acc_en=True: the destination register holds full float32 precision throughout the accumulation, preventing rounding error from accumulating across K tiles
  - When it matters: K-dimension depth (large hidden sizes) amplifies the rounding difference; for MoE expert matmuls with K=2048 (d_model), the effect is small but measurable
  - Interaction with `packer_l1_acc`: both control accumulation behavior but at different pipeline stages (compute vs. pack)
  - Cost: using fp32 dest cuts the number of simultaneously live accumulation registers, which can slightly reduce parallelism within a tile

- `math_fidelity_overview.md`
  - What math fidelity controls at the silicon level: how many mantissa bits from each bfloat16 operand are actually presented to the multiplier hardware during a dot-product accumulation
  - Enum values and their numeric codes: `LoFi = 0`, `HiFi2 = 2`, `HiFi3 = 3`, `HiFi4 = 4`
  - The key intuition: higher fidelity = more mantissa bits used per multiply = more cycles per tile = higher accuracy; lower fidelity = fewer bits = fewer cycles = lower accuracy
  - Why the default for `WormholeComputeKernelConfig` is `LoFi`: throughput-first default optimized for bandwidth-bound workloads
  - Forward pointer to Chapter 2 for the full LoFi/HiFi2/HiFi4 comparison

---

### Chapter 2: Math Fidelity Levels — LoFi vs HiFi2 vs HiFi4

**Directory:** `ch2_math_fidelity_levels/`

**Description:** Characterizes the four MathFidelity levels with respect to compute throughput (cycles per tile), numerical precision (mantissa bits effectively used), and PCC impact on MoE expert matmul outputs.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Decision table: which fidelity level to use for each MoE projection type (gate, up, down)
  - Prerequisites: Chapter 1

- `fidelity_precision_model.md`
  - How bfloat16 operands are processed in the Wormhole FPU: the multiply-accumulate pipeline and the role of the mantissa width in each multiplication step
  - LoFi: effectively uses the top mantissa bits of each operand; approximates the product but dramatically reduces the number of cycles needed per tile (reported as the fastest mode: ~2x throughput advantage over HiFi4 for matmul-intensive workloads)
  - HiFi2: uses more mantissa bits; bridges the gap between LoFi accuracy and HiFi4 speed; the typical choice for operations where PCC > 0.999 is required
  - HiFi3: intermediate level, rarely used directly; included for completeness
  - HiFi4: full-precision BF16 multiplication; matches PyTorch reference output most closely; recommended for operations feeding into softmax or layer norm where small errors amplify
  - Table: Fidelity level vs. approximate relative throughput multiplier vs. expected PCC range for a 4096x2048 bfloat16 matmul

- `fidelity_and_moe_accuracy.md`
  - Why MoE gate projections can tolerate LoFi: the gating weights produce logits that are softmax'd; small mantissa errors in the logits produce negligible changes in the routing decisions for well-separated expert scores
  - Why MoE up projections tolerate LoFi: the SiLU activation following the gate/up product provides a soft saturation that absorbs small numerical errors
  - Why MoE down projections prefer HiFi2: the down projection accumulates expert outputs into the residual stream; accumulated rounding errors from K tiles (d_ff = 2048–18432 depending on model) can noticeably shift hidden state values and degrade end-to-end PCC
  - Empirical PCC data pattern (from DeepSeek-V3 approach): gate/up at LoFi yields PCC > 0.999 vs PyTorch; down at LoFi yields PCC ~0.99; down at HiFi2 restores PCC > 0.999
  - How to measure PCC in TTNN: `torch.corrcoef` between flattened tensors, the standard threshold used in tt-metal CI (typically 0.9995 for bfloat16 MoE)

- `fidelity_selection_workflow.md`
  - Step-by-step process for validating a fidelity choice for a new MoE model: (1) run with HiFi4 to establish baseline PCC, (2) step down to HiFi2 and measure PCC delta, (3) step down to LoFi and measure; accept the lowest fidelity that stays above the PCC threshold
  - How to write a parameterized test in Python that sweeps `math_fidelity` and logs PCC and latency for each MoE projection
  - Warning about PCC testing with small batch sizes: PCC on a single decode token is less reliable than prefill with seq >= 128 due to statistical variance

---

### Chapter 3: packer_l1_acc — Throughput Effect

**Directory:** `ch3_packer_l1_acc/`

**Description:** Explains the packer pipeline stage in the Tensix compute flow and how enabling `packer_l1_acc=True` eliminates DRAM write-backs between outer loop iterations, reducing bandwidth pressure and improving matmul throughput.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference: when to enable `packer_l1_acc` (almost always for matmul-dominant workloads)
  - Prerequisites: Chapter 1 (compute pipeline overview), basic understanding of L1 SRAM vs. DRAM hierarchy

- `tensix_packer_pipeline.md`
  - The Tensix compute pipeline: FPU produces output tiles → packer serializes tiles from the destination register → tiles are written to memory
  - With `packer_l1_acc=False` (default): after each outer-loop iteration the packer writes partial sums to DRAM; the next iteration re-reads them from DRAM to accumulate; this creates unnecessary DRAM round-trips for each K-loop step
  - With `packer_l1_acc=True`: the packer accumulates partial sums into an L1 buffer; DRAM is written only once at the end of the full accumulation; eliminates the per-iteration DRAM read-modify-write
  - Bandwidth model: for a matmul of shape [M, K] x [K, N] with `in0_block_w = b` tiles, disabling `packer_l1_acc` causes `(K/b - 1)` extra DRAM reads of the output tile per core; enabling it eliminates these

- `throughput_impact.md`
  - Quantitative framing: for bandwidth-bound MoE expert matmuls (small M, large K, large N), enabling `packer_l1_acc` can reduce effective memory traffic by up to `(K/b - 1)/(K/b)` — roughly 50-90% reduction in output DRAM reads depending on `in0_block_w`
  - Regime where it matters most: decode-mode MoE (M = batch size, typically 1-32) where the matmul is heavily bandwidth-bound rather than compute-bound; reducing DRAM accesses directly translates to lower latency
  - Regime where it matters less: prefill with large M (seq >= 512); the matmul is more compute-bound and packer traffic is a smaller fraction of total time
  - Why both `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` in DeepSeek-V3 set `packer_l1_acc=True`: the improvement applies regardless of fidelity level

- `packer_l1_acc_constraints.md`
  - L1 footprint requirement: the packer buffer for accumulation must fit in per-core L1 alongside input and output tile buffers; for large `per_core_N`, enabling `packer_l1_acc` can push L1 over capacity
  - How to detect L1 overflow: TTNN raises an allocation error at op dispatch time; reducing `per_core_N` or `out_subblock_w` resolves it
  - Interaction with `fp32_dest_acc_en`: when both are enabled, the intermediate buffer is fp32; when only `packer_l1_acc` is enabled, the intermediate buffer is bfloat16; the former requires 2x the L1 space for the accumulation buffer
  - Safe default: `packer_l1_acc=True`, `fp32_dest_acc_en=False` for LoFi configs (used in DeepSeek gate/up); `packer_l1_acc=True`, `fp32_dest_acc_en=True` for HiFi2 configs (used in DeepSeek down) when L1 budget allows

---

### Chapter 4: math_approx_mode — Accuracy Trade-offs

**Directory:** `ch4_math_approx_mode/`

**Description:** Covers the `math_approx_mode` flag, which enables hardware-approximated implementations of SFPU transcendental functions (exp, reciprocal, sqrt, sigmoid), and characterizes the accuracy risk for MoE operations.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Table: which TTNN ops are affected by `math_approx_mode`, which are not (matmul itself is not; activations are)
  - Prerequisites: Chapter 1

- `sfpu_approx_operations.md`
  - What the SFPU (Special Function Processing Unit) is: the scalar unit on each Tensix core that handles non-linear functions, as opposed to the FPU which handles matrix multiply
  - Operations routed through SFPU that use the approximation when `math_approx_mode=True`: exp (for softmax), reciprocal (for layer norm and softmax denominator), sqrt, sigmoid, gelu, silu
  - How the approximation works conceptually: piecewise polynomial lookup rather than iterative Newton-Raphson refinement; faster but introduces ~0.1-0.3% relative error per evaluation
  - Operations NOT affected by `math_approx_mode`: pure matmul/dot-product (the FPU path); `math_approx_mode` has no effect on the multiply-accumulate precision
  - Why `COMPUTE_KERNEL_CONFIG_LOFI` sets `math_approx_mode=False`: gate and up projections in MoE do not apply transcendental ops; the flag is irrelevant, so it is conservatively set to False
  - Why `COMPUTE_KERNEL_CONFIG_HIFI2` sets `math_approx_mode=True`: the HiFi2 config is also used for attention softmax and layer norm in the broader model; approximation is acceptable there for the throughput gain

- `approx_mode_accuracy_risks.md`
  - When approximation mode causes problems: softmax over very long sequences (K >= 16K tokens) where chunk-accumulated exp approximation errors can compound across chunks; `math_approx_mode=False` is recommended for flash attention with seq > 16K
  - When approximation mode is safe: SiLU activation in MoE FFN blocks (applied element-wise to the gate projection output before multiplying with up projection); error is bounded and does not accumulate across sequence length
  - When approximation mode is irrelevant: pure matmul without a fused activation; setting `math_approx_mode=True` in this case does not change outputs
  - PCC characterization: for typical MoE expert layer (gate -> silu -> elem_mul -> down), using `math_approx_mode=True` vs `False` with HiFi2 fidelity changes PCC by < 0.0001 on random inputs; difference becomes larger at extreme input magnitudes

- `approx_mode_for_moe.md`
  - Recommended setting for each MoE projection:
    - Gate (w1): `math_approx_mode=False` — no transcendental ops in the gate matmul itself; use LoFi for throughput
    - Up (w3): `math_approx_mode=False` — same reasoning; SiLU fusion does not require high-accuracy exp approximation for well-scaled inputs
    - Down (w2): `math_approx_mode=True` (when paired with HiFi2) — if the down projection uses HiFi2 and is fused with any activation, approx mode is acceptable; for pure matmul, the setting is irrelevant
  - How Qwen MoE FFN structure differs: Qwen uses SwiGLU (silu(gate) * up) which is structurally the same as DeepSeek; the same kernel config recommendations apply
  - Practical note: when in doubt, set `math_approx_mode=False`; the throughput difference for MoE expert matmuls is negligible compared to the fidelity and `packer_l1_acc` settings

---

### Chapter 5: MoE Expert Matmul Configuration

**Directory:** `ch5_moe_expert_matmul_config/`

**Description:** Applies the parameter knowledge from Chapters 1–4 to the specific gate, up, and down expert projections in both DeepSeek-V3 and Qwen MoE, showing how to configure each projection and why the configurations differ.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Reference table: recommended `WormholeComputeKernelConfig` for each MoE projection type, with rationale
  - Prerequisites: Chapters 1–4

- `deepseek_v3_config_analysis.md`
  - How `COMPUTE_KERNEL_CONFIG_LOFI` is defined in `models/demos/deepseek_v3/utils/config_helpers.py`: exact parameter values
  - How `COMPUTE_KERNEL_CONFIG_HIFI2` is defined alongside it: exact parameter values
  - In `models/demos/deepseek_v3/tt/experts.py`: w1_experts (gate_proj) and w3_experts (up_proj) use LOFI; w2_experts (down_proj) uses HIFI2
  - Rationale for the asymmetry: gate and up outputs feed into SiLU and element-wise multiply — operations tolerant of LoFi rounding; down projection accumulates the full d_ff dimension into the residual stream — requires HiFi2 for PCC above threshold
  - DeepSeek-V3 expert dimensions: d_model = 7168, d_ff (intermediate) = 2048 per expert; the deep K dimension in the down projection (K=2048 tiles) makes fidelity matter more there
  - What `packer_l1_acc=True` buys for DeepSeek expert decode: bandwidth analysis for small batch (batch=1) decode where M=1 and DRAM bandwidth is the bottleneck

- `qwen_moe_current_state.md`
  - Current state of Qwen MoE in tt-transformers: no explicit `compute_kernel_config` is passed to expert matmul ops, so TTNN uses its internal default
  - What the TTNN default is when `compute_kernel_config` is omitted: falls back to device-level defaults, typically equivalent to LoFi with `packer_l1_acc=False` — suboptimal because it pays DRAM round-trip costs during accumulation
  - Qwen 2.5 MoE expert dimensions: d_model = 2048, d_ff (intermediate) = 768 per expert (Qwen2-57B-A14B has 64 experts, top-8 routing); larger models vary
  - Expected performance gap: with `packer_l1_acc=False` vs `True` for Qwen decode-mode expert matmuls, the bandwidth-bound matmul should see measurable improvement
  - The missing optimization: adding `compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)` to Qwen gate and up projections

- `applying_configs_to_qwen.md`
  - Step-by-step code change: where in the Qwen MoE expert forward pass to add `compute_kernel_config` arguments
  - The two-config pattern: define `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` (mirroring DeepSeek), assign LOFI to gate and up, HIFI2 to down
  - Correctness validation: running PCC comparison between the new config and unoptimized baseline before and after the change; expected PCC > 0.999 for all projections
  - Latency measurement: how to isolate expert matmul time using `ttnn.device.profiler` and compare pre/post config change
  - Interaction with existing Qwen program configs (MatmulMultiCoreReuseMultiCastProgramConfig): `compute_kernel_config` is orthogonal to program config; both can be specified simultaneously
  - Edge case: if the Qwen model runs in prefill mode with large sequences (M >= 512), the LoFi vs HiFi2 latency difference is smaller but the PCC difference may also narrow; the same config is still correct

---

### Chapter 6: Performance Benchmarking and Config Selection

**Directory:** `ch6_benchmarking_and_selection/`

**Description:** Provides a systematic decision framework and benchmarking methodology for selecting compute kernel configs in production MoE deployments, including when to deviate from the DeepSeek defaults.

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Decision flowchart: given a new MoE model and projection type, how to arrive at the right `WormholeComputeKernelConfig`
  - Prerequisites: Chapters 1–5

- `benchmarking_methodology.md`
  - How to construct a standalone benchmark for a single MoE expert matmul: creating synthetic input tensors, avoiding program cache warm-up effects, running multiple iterations for stable timing
  - What to measure: per-op latency in microseconds (from Tracy or `ttnn.device.profiler`), not wall-clock Python time
  - Sweep dimensions: `math_fidelity` (LoFi / HiFi2 / HiFi4), `packer_l1_acc` (True / False), `fp32_dest_acc_en` (True / False); hold `math_approx_mode` constant at False for matmul-only benchmarks
  - How to isolate the `packer_l1_acc` effect: run two configs that differ only in `packer_l1_acc` and compare latency; the difference is a pure bandwidth benefit
  - How to quantify PCC impact: compare output tensors against a PyTorch reference (float32 matmul) for each config combination; use token-level and layer-level PCC, not just per-element error

- `config_decision_matrix.md`
  - Structured decision rules organized by projection type and regime:
    - Gate projection (any model, any sequence length): LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    - Up projection (SwiGLU/SiLU gating, any model): same as gate
    - Down projection (decode, K <= 4096): HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
    - Down projection (prefill, seq >= 512): HiFi2 or LoFi depending on model's PCC tolerance; benchmark both
    - Any projection for models requiring > 0.9995 PCC (e.g., fine-tuned generation at strict quality gates): HiFi4 with fp32_dest_acc_en=True
  - How d_ff / d_model ratio affects the decision: deeper K in the down projection (d_ff large relative to d_model) makes higher fidelity more important
  - How top-K routing depth (top-1 vs top-8) interacts: higher top-K means more expert activations per token per layer, amplifying any systematic rounding bias across the sum of expert outputs

- `production_config_checklist.md`
  - Pre-deployment checklist for MoE compute kernel configs: PCC threshold verified, latency measured at both prefill and decode batch sizes, L1 budget confirmed (no allocation errors at max batch size), configs stored in a shared `config_helpers.py` for reuse
  - How to handle models with heterogeneous expert sizes (e.g., DeepSeek shared experts vs. routed experts): shared experts use dense FFN and may warrant different configs than routed expert FFNs
  - Version control note: `WormholeComputeKernelConfig` parameters may change behavior across tt-metal firmware releases; pin the firmware version alongside the config in production deployments
  - Common mistake: setting `packer_l1_acc=True` and `fp32_dest_acc_en=True` for a matmul with very large `per_core_N` without checking L1 budget; always verify with a dry-run before profiling

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **WormholeComputeKernelConfig** | The TTNN Python class `ttnn.WormholeComputeKernelConfig` that controls per-op compute behavior on Wormhole B0 hardware |
| **math_fidelity** | Field controlling how many mantissa bits are used per multiply-accumulate step; enum values: LoFi, HiFi2, HiFi3, HiFi4 |
| **LoFi** | `ttnn.MathFidelity.LoFi` — lowest fidelity, highest throughput; trades mantissa precision for compute speed |
| **HiFi2** | `ttnn.MathFidelity.HiFi2` — medium-high fidelity; the standard choice for accuracy-sensitive MoE projections |
| **HiFi4** | `ttnn.MathFidelity.HiFi4` — highest fidelity; use when PCC must be maximized regardless of throughput cost |
| **packer_l1_acc** | Boolean field enabling in-L1 accumulation of partial output sums during matmul, eliminating DRAM round-trips per K-loop iteration |
| **fp32_dest_acc_en** | Boolean field enabling float32 precision in the FPU destination register; prevents bfloat16 rounding during K-loop accumulation |
| **math_approx_mode** | Boolean field enabling hardware-approximate SFPU ops (exp, reciprocal, sqrt, sigmoid); does not affect pure matmul FPU path |
| **SFPU** | Special Function Processing Unit — the scalar unit on each Tensix core that handles transcendental functions |
| **FPU** | Floating-Point Unit — the matrix multiply unit on each Tensix core, handles dot-product accumulation |
| **packer** | The Tensix pipeline stage that serializes FPU output tiles from the destination register into L1 or DRAM |
| **expert projection** | One of gate_proj (w1), up_proj (w3), or down_proj (w2) in a SwiGLU/SiLU MoE FFN expert |
| **PCC** | Pearson Cross-Correlation — standard numerical correctness metric in tt-metal; target is typically > 0.999 for bfloat16 MoE outputs |
| **d_model** | Hidden state dimension (input/output width of the expert projection) |
| **d_ff** | Expert intermediate dimension (inner width of the expert FFN; the K dimension of the down projection) |
| **K-loop** | The loop over K-dimension tiles in the matmul kernel; each iteration produces a partial sum that must be accumulated |
| **in0_block_w** | TTNN matmul program config parameter controlling how many K-tiles are processed per inner loop iteration |

### Notation

- TTNN config fields are always written in their exact Python attribute names: `math_fidelity`, `packer_l1_acc`, `fp32_dest_acc_en`, `math_approx_mode`.
- Tensor shapes are written as `[M, K]` x `[K, N]` with named dimensions where relevant (e.g., `[batch * seq, d_model]` x `[d_model, d_ff]`).
- Tile-space quantities use subscript notation: M_t = M/32, K_t = K/32, N_t = N/32.
- Configuration constants are written in `SCREAMING_SNAKE_CASE` as they appear in the DeepSeek codebase: `COMPUTE_KERNEL_CONFIG_LOFI`, `COMPUTE_KERNEL_CONFIG_HIFI2`.
- Code blocks use Python syntax with `import ttnn` and `import torch` assumed in scope.
- All performance numbers in this guide refer to Wormhole B0 hardware; Blackhole and Grayskull differ and are not covered.
- PCC values are cited as scalars between 0 and 1; a value of 0.9995 means 99.95% correlation with the reference output.

### Formatting Rules

- Every chapter directory has an `index.md` providing overview, learning objectives, and a summary table or flowchart.
- Code examples are fenced with ` ```python ` and include inline comments on non-obvious lines.
- Configuration parameter tables appear before prose explanations so readers can anchor on concrete values first.
- Accuracy warnings use `> **Warning:** ...` blockquotes; throughput tips use `> **Tip:** ...` blockquotes.
- Every file ends with a "Next Steps" section listing the next file in the reading order.
- When citing DeepSeek-V3 source paths, use repo-relative paths prefixed with the repository root, e.g., `models/demos/deepseek_v3/utils/config_helpers.py`.

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: Compute Kernel Config Fundamentals | None (foundational); introduces all four key fields |
| Ch 2: Math Fidelity Levels | Ch 1 (math_fidelity field overview, FPU pipeline concept) |
| Ch 3: packer_l1_acc Throughput Effect | Ch 1 (packer pipeline introduced, fp32_dest_acc_en interaction), Ch 2 (bandwidth-bound vs compute-bound framing) |
| Ch 4: math_approx_mode Accuracy Trade-offs | Ch 1 (math_approx_mode field introduced), Ch 2 (understanding what ops benefit from fidelity vs approx mode distinction) |
| Ch 5: MoE Expert Matmul Configuration | Ch 1–4 (all parameter semantics); Ch 2 (LoFi vs HiFi2 PCC characterization for gate/down); Ch 3 (packer_l1_acc bandwidth model for decode) |
| Ch 6: Performance Benchmarking and Config Selection | Ch 2 (fidelity sweep methodology), Ch 3 (packer_l1_acc isolation test), Ch 5 (DeepSeek and Qwen configs as concrete baselines) |

**Specific forward references to flag:**

- Ch 1 (`math_fidelity_overview.md`) introduces LoFi/HiFi2/HiFi4 names but deliberately defers the throughput multiplier and PCC data to Ch 2; Ch 2 must contain the quantitative table before Ch 5 references specific PCC thresholds.
- Ch 1 (`fp32_dest_acc_en.md`) notes the interaction with `packer_l1_acc` and L1 space; Ch 3 (`packer_l1_acc_constraints.md`) must give the full L1 footprint analysis — avoid duplicating the constraint math in Ch 1.
- Ch 3 (`throughput_impact.md`) establishes the bandwidth saving formula; Ch 5 (`deepseek_v3_config_analysis.md`) and Ch 6 (`config_decision_matrix.md`) both reference this formula without re-deriving it.
- Ch 4 (`approx_mode_for_moe.md`) gives the per-projection recommendation table for `math_approx_mode`; Ch 5 (`applying_configs_to_qwen.md`) references these recommendations by name rather than restating them — Ch 4 must be written first.
- Ch 6 (`benchmarking_methodology.md`) references the sweep dimensions introduced in Chapters 1–4; any change to the parameter names in Ch 1 must be propagated to the Ch 6 sweep table.
- Ch 5 (`qwen_moe_current_state.md`) asserts that Qwen's current missing `compute_kernel_config` defaults to LoFi + `packer_l1_acc=False`; this claim must be validated against tt-metal source at time of writing and updated if TTNN defaults change.
