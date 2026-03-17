# Plan: Weight Quantization for MoE Experts

---

## 1. Audience

**Primary audience:** ML engineers and performance engineers deploying or optimizing Mixture of Experts (MoE) models on Tenstorrent hardware (Wormhole B0, T3K mesh). Readers are comfortable with:

- PyTorch and standard tensor operations (matmul, linear, dtype casting)
- Transformer model architecture: attention blocks, FFN layers, residual connections
- Basic TTNN op usage: `ttnn.linear`, `ttnn.matmul`, `ttnn.to_device`, memory configs, program configs
- Python profiling and quantitative reasoning about model accuracy (e.g., perplexity, PCC)

**What they do NOT need to know in advance:**

- Tenstorrent-specific numeric formats (bfloat8_b, bfloat4_b) and how they differ from standard INT8 or FP8
- How Wormhole's MathFidelity levels (LoFi, HiFi2, HiFi4) affect compute throughput
- Why the down-projection in a SwiGLU FFN may require higher precision than the gate or up projections
- How TTNN stores quantized weights on-chip vs. in DRAM and what that means for bandwidth utilization

This guide fills those gaps progressively, starting from quantization format fundamentals and building toward a concrete mixed-precision strategy for MoE expert projections.

---

## 2. Chapter List

---

### Chapter 1: Quantization Formats on Wormhole — bfloat16, bfloat8_b, and bfloat4_b

**Description:** Explains the three key floating-point dtypes available in TTNN for weight storage on Wormhole B0, covering their binary representations, numeric ranges, hardware throughput multipliers, and memory footprint implications.

**Directory:** `ch01_quantization_formats/`

**Files:**

- `index.md`
  - Chapter overview, learning objectives, and navigation links to sub-files
  - Prerequisites checklist: readers should understand IEEE 754 floating point at a conceptual level
  - Summary table: bfloat16 vs. bfloat8_b vs. bfloat4_b across key dimensions (bits, mantissa, range, TTNN dtype constant)

- `bfloat16_format.md`
  - bfloat16 binary layout: 1 sign bit, 8 exponent bits, 7 mantissa bits — same exponent range as float32
  - Why bfloat16 is the standard training and baseline inference dtype for Tenstorrent models
  - TTNN representation: `ttnn.bfloat16`, tile layout requirement (32x32 tiles)
  - Memory footprint: 2 bytes per element; a 7168×2048 expert weight matrix occupies ~29 MB
  - Wormhole compute throughput at bfloat16: 74 TFLOPS (n150) / 131 TFLOPS (n300)

- `bfloat8_b_format.md`
  - bfloat8_b binary layout: block floating-point format, 8 bits per element, shared exponent across a tile (32×32 block)
  - The role of the shared exponent: how it extends the dynamic range compared to a naive 8-bit float, and why the "_b" suffix denotes block format
  - TTNN representation: `ttnn.bfloat8_b`, requires `ttnn.TILE_LAYOUT` — row-major layout is not supported
  - Memory footprint: 1 byte per element; 2× reduction vs. bfloat16
  - Wormhole BlockFP8 compute throughput: 148 TFLOPS (n150) / 262 TFLOPS (n300) — 2× throughput vs. bfloat16
  - Wormhole FP8 compute throughput: 262 TFLOPS (n150) — distinguishing FP8 (standard) from BlockFP8
  - Packing behavior: how 8-bit elements are packed into 32×32 tiles and transferred between DRAM and L1

- `bfloat4_b_format.md`
  - bfloat4_b binary layout: block floating-point format, 4 bits per element, shared exponent across a 32×32 tile
  - 4× memory reduction vs. bfloat16; two elements packed per byte in DRAM
  - TTNN representation: `ttnn.bfloat4_b`, tile layout only
  - Numeric precision limitation: with only 3 mantissa bits (plus shared exponent), the representable range is narrower and the quantization grid is coarser
  - Expected signal-to-noise ratio: rough analogy to 4-bit integer quantization, but with the dynamic range benefit of the shared block exponent
  - Wormhole compute throughput at 4-bit: up to 4× throughput over bfloat16 due to 4× tile compute density
  - When bfloat4_b is viable: weight-stationary operations with large weight matrices and low activation precision requirements

- `hardware_dtype_support.md`
  - TTNN DataType enum: `ttnn.bfloat16`, `ttnn.bfloat8_b`, `ttnn.bfloat4_b`, `ttnn.float32`, `ttnn.uint8`, `ttnn.int32` — the full list and which are supported for weight tensors in matmul/linear
  - Tile layout constraint: bfloat8_b and bfloat4_b require `ttnn.TILE_LAYOUT`; TTNN auto-converts when either dtype is detected
  - MathFidelity levels: LoFi, HiFi2, HiFi4 as `WormholeComputeKernelConfig` — how fidelity affects accumulation precision independent of weight dtype
  - `fp32_dest_acc_en` flag: enabling FP32 accumulation in the output register for HiFi2/HiFi4; its effect on accuracy vs. compute throughput
  - DRAM bandwidth impact: how storing weights in bfloat8_b vs. bfloat4_b reduces DRAM read volume during matmul, particularly relevant for decode (memory-bound) workloads

---

### Chapter 2: TTNN Quantization API — Loading and Using Quantized Weights

**Description:** Covers the practical TTNN API for converting, loading, and computing with bfloat8_b and bfloat4_b weight tensors, including dtype configs in `ttnn.linear`, `ttnn.matmul`, and associated `WormholeComputeKernelConfig` objects.

**Directory:** `ch02_ttnn_quantization_api/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Quick-reference code snippet: converting a PyTorch bfloat16 weight tensor to bfloat4_b for TTNN
  - Prerequisites: Chapter 1 (format definitions), basic TTNN tensor and memory config usage

- `weight_conversion.md`
  - Converting weights at checkpoint load time: `ttnn.as_tensor(weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)`
  - Difference between load-time quantization (one-time cost at startup) vs. on-the-fly quantization (per-forward-pass overhead)
  - Transposing before conversion: why expert weight matrices are often transposed to `[d_ff, d_model]` for column-major matmul, and how that interacts with tile packing for bfloat4_b
  - The `shard_dims` argument when distributing expert weights across T3K devices: `shard_dims=(1, 1)` for sharding the expert and the inner dimension
  - Dequantization path: TTNN automatically dequantizes bfloat8_b/bfloat4_b weights to bfloat16 (or the compute accumulation type) during the matmul kernel; no explicit user-side dequantize call is needed at inference time

- `compute_kernel_config.md`
  - `WormholeComputeKernelConfig` and its four key fields: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, `packer_l1_acc`
  - LoFi (`MathFidelity.LoFi`, `fp32_dest_acc_en=False`): designed for low-precision weight operations; paired with bfloat4_b in DeepSeek-V3 gate and up projections
  - HiFi2 (`MathFidelity.HiFi2`, `fp32_dest_acc_en=True`): balanced accuracy; paired with bfloat8_b in DeepSeek-V3 down projections and dense MLP layers
  - HiFi4 (`MathFidelity.HiFi4`, `fp32_dest_acc_en=True`): highest accuracy, used for attention SDPA — not typical for expert weight matmuls
  - Rule of thumb: lower precision weights benefit from higher MathFidelity to recover accumulation accuracy; but for bfloat4_b the gate/up projections in SwiGLU can tolerate LoFi because the SILU nonlinearity and subsequent multiplication are the dominant noise source
  - Code pattern: constructing `WormholeComputeKernelConfig` and passing it as `compute_kernel_config=` to `ttnn.linear`

- `dtype_in_linear_and_matmul.md`
  - How dtype is specified at conversion time (not at call time in `ttnn.linear`): the weight tensor's stored dtype drives the kernel selection
  - Example: creating a `ttnn.linear` call with a pre-converted bfloat8_b weight tensor vs. a bfloat4_b weight tensor — the only difference is which dtype was used in `ttnn.as_tensor`
  - Activation dtype: inputs to the matmul are typically bfloat16; the hardware unpack path handles the asymmetry between bfloat16 activations and bfloat8_b/bfloat4_b weights
  - Output dtype: by default the output accumulates to bfloat16 (or float32 if `fp32_dest_acc_en=True`), then the packer writes bfloat16 to L1/DRAM
  - Program config interaction: `MatmulMultiCoreReuseMultiCastProgramConfig` parameters (`in0_block_w`, `per_core_M`, `per_core_N`, `out_subblock_h`, `out_subblock_w`) are unchanged by weight dtype — tile count logic is the same; only kernel dispatch path differs

- `validation_patterns.md`
  - Verifying weight conversion accuracy: computing PCC between the dequantized bfloat8_b/bfloat4_b weight and the original bfloat16 weight; expected PCC > 0.99 for bfloat8_b, lower (~0.97–0.98) for bfloat4_b depending on weight distribution
  - Forward pass PCC thresholds observed in practice: MLP with bfloat8_b weights achieves PCC ~0.975; full MoE layer with mixed bfloat4_b gate/up and bfloat8_b down achieves PCC ~0.97 against bfloat16 reference
  - `comp_pcc` helper: TTNN convention for asserting output accuracy against a PyTorch reference
  - Test scaffolding pattern: load real checkpoint weights → run forward pass → compare against `torch.bfloat16` CPU reference → assert `pcc >= threshold`

---

### Chapter 3: Accuracy Analysis for MoE Expert Quantization

**Description:** Characterizes the accuracy loss from quantizing MoE expert weights at each precision level, identifying which metrics are most affected, which projections are most sensitive, and what the practical tolerance limits are.

**Directory:** `ch03_accuracy_analysis/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: quantization level vs. expected PCC range, perplexity delta, and recommended use case
  - Prerequisites: Chapters 1 and 2

- `accuracy_metrics_for_moe.md`
  - PCC (Pearson Cross-Correlation): definition, why it is the standard TTNN correctness metric, and how to compute it via `comp_pcc`
  - PCC thresholds used in practice: >0.999 for bfloat16 baselines, ~0.97–0.98 for bfloat8_b, potentially lower for bfloat4_b with aggressive compression
  - Perplexity delta: how quantization-induced weight error propagates to token prediction probability and perplexity; expected perplexity increase of <1 PPL for bfloat8_b, potentially 1–3 PPL for bfloat4_b on language modeling benchmarks
  - Task-specific accuracy: code generation and reasoning tasks (MBPP, HumanEval, MATH) are more sensitive to quantization noise than open-ended generation; benchmark implications for model validation
  - Why PCC alone is insufficient: a high PCC at a single layer can mask cumulative error across all MoE layers in a 60-layer transformer

- `projection_sensitivity.md`
  - SwiGLU FFN structure: three projections — gate (w1), up (w3), down (w2) — and how their roles differ
  - Gate projection (w1): produces pre-activation logits for SILU; its output is element-wise multiplied with the up projection output; moderate sensitivity because SILU compresses dynamic range
  - Up projection (w3): produces the linear path in SwiGLU; its output is scaled by the SILU gate output; moderate sensitivity, similar to gate
  - Down projection (w2): maps from intermediate dimension back to hidden dimension; directly contributes to the residual stream; highest sensitivity because errors are not compressed by a nonlinearity before accumulation
  - Empirical evidence from DeepSeek-V3 TTNN implementation: w1/w3 use bfloat4_b + LoFi compute, w2 uses bfloat8_b + HiFi2 compute — this mixed-precision choice directly reflects the sensitivity ordering
  - Why down projection errors accumulate: each MoE layer's down-projection output is added to the residual stream and feeds directly into the next layer's attention; no nonlinearity intervenes to clip quantization noise

- `bfloat16_vs_bfloat8_vs_bfloat4_accuracy.md`
  - Layer-level PCC comparison: bfloat16 (reference) vs. bfloat8_b weights vs. bfloat4_b weights for a single expert FFN, measuring output PCC against bfloat16 CPU reference
  - Expected PCC ranges: bfloat8_b gate/up projections ~0.98–0.99, bfloat8_b down projection ~0.975–0.985; bfloat4_b gate/up projections ~0.96–0.98, bfloat4_b down projection ~0.94–0.97
  - Cumulative error in deep models: with 28 MoE layers (as in Qwen3.5-35B), per-layer PCC of 0.97 compounds to meaningful end-to-end degradation; why bfloat4_b for the down projection risks falling below quality thresholds
  - Weight distribution effects: expert weights trained with auxiliary load-balancing losses tend to have more uniform weight magnitudes, which is favorable for block floating-point quantization (shared exponent covers the distribution well); experts with high kurtosis weight distributions (outliers) suffer more from bfloat4_b
  - Outlier sensitivity in the down projection: the down-projection weight matrix in large MoE models often has a heavier tail than the gate/up projections, making it more susceptible to bfloat4_b quantization error

- `qwen_vs_deepseek_accuracy_comparison.md`
  - DeepSeek-V3 quantization strategy: gate and up projections in bfloat4_b + LoFi, down projection in bfloat8_b + HiFi2; validated at PCC ~0.97 for full MoE layer
  - Qwen3.5-35B baseline: uses bfloat16 for all expert weights; higher accuracy baseline but 2–4× more memory per expert
  - The trade-off question: can Qwen expert weights tolerate bfloat4_b for gate/up and bfloat8_b for down without unacceptable accuracy loss?
  - Model-specific factors: DeepSeek-V3 was trained with FP8 mixed-precision quantization-aware training, meaning its weights are already optimized for low-precision storage; Qwen3.5-35B was trained in bfloat16 and may have larger outliers in expert weights
  - Recommended evaluation procedure: run calibration perplexity on WikiText-2 or C4 under bfloat8_b-only, then bfloat4_b gate/up + bfloat8_b down, then full bfloat4_b — compare delta vs. bfloat16 baseline

---

### Chapter 4: Throughput and Memory Bandwidth Impact on Wormhole

**Description:** Quantifies how weight quantization affects compute throughput, DRAM bandwidth consumption, and tile compute efficiency on Wormhole B0, distinguishing prefill (compute-bound) from decode (memory-bound) regimes.

**Directory:** `ch04_throughput_impact/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Roofline model sketch: Wormhole DRAM bandwidth (288 GB/s n150, 576 GB/s n300) vs. compute throughput at bfloat16/bfloat8_b/bfloat4_b
  - Prerequisites: Chapter 1 (throughput numbers), Chapter 2 (compute kernel configs)

- `prefill_compute_throughput.md`
  - Prefill regime: large sequence lengths (seq=2048+) mean the matmul is compute-bound, not memory-bound
  - Throughput multipliers: bfloat16 → bfloat8_b/BlockFP8 achieves ~2× throughput (148 vs. 74 TFLOPS on n150); bfloat4_b achieves up to ~4× if fully compute-bound
  - MathFidelity overhead: LoFi vs. HiFi2 affects the number of FMA accumulation passes; HiFi2 with `fp32_dest_acc_en=True` adds ~20–30% latency versus LoFi for the same tile size
  - Effective throughput for expert FFNs: expert weight matrices in Qwen3.5-35B are [d_ff=2048 or 4096, d_model=4096]; tile count analysis showing how bfloat4_b doubles effective tile density
  - Prefill bottleneck: in practice, prefill is often limited by all-to-all dispatch/combine overhead, not by expert FFN compute; quantization gains are most visible when expert FFN is the measured bottleneck

- `decode_memory_bandwidth.md`
  - Decode regime: small batch sizes (batch=1 to 32, seq=1) make expert FFN compute memory-bound — each weight element is read once per forward pass
  - Bandwidth reduction from quantization: bfloat8_b halves DRAM read volume for weights vs. bfloat16; bfloat4_b quarters it
  - Decode throughput improvement: for a memory-bound matmul, bandwidth reduction directly translates to latency reduction — bfloat4_b gate/up projections approach ~4× throughput improvement in ideal decode conditions
  - L1 vs. DRAM placement: decode mode uses L1 memory config for expert activation tensors while weight tensors remain in DRAM; quantized weights in DRAM are loaded in fewer cache lines, reducing NoC traffic
  - Arithmetic intensity crossover: formula for the batch size above which the matmul transitions from memory-bound to compute-bound for each dtype; relevant for choosing the quantization strategy based on deployment batch size

- `tile_compute_efficiency.md`
  - Tile size is fixed at 32×32 elements regardless of dtype; but effective compute per tile is higher for denser dtypes
  - bfloat4_b packing: two 4-bit elements per byte means a 32×32 tile is 512 bytes (vs. 2048 bytes for bfloat16); 4× more tiles can be streamed through L1 in the same memory bandwidth budget
  - Wormhole Math engine: the FPU/SFPU handles block-floating-point unpack as part of the tile read path; no software-side dequantization loop is needed — unpack happens in hardware before FMA
  - MathFidelity and tile throughput: LoFi uses a single accumulation pass per tile (highest throughput); HiFi2 uses two passes; HiFi4 uses four passes — relevant for understanding why bfloat4_b + LoFi is preferred for gate/up projections to maximize decode throughput
  - Grid utilization: on a 72-core Wormhole B0 chip, expert parallelism assigns a core subset to each local expert; quantized weights allow more experts' weight tiles to fit in L1 simultaneously, improving grid utilization

- `bandwidth_vs_accuracy_tradeoff.md`
  - Joint analysis: for each projection type (gate/up/down), plot accuracy (PCC) vs. DRAM bandwidth consumption and compute throughput under bfloat16, bfloat8_b, and bfloat4_b
  - The "efficiency frontier": combinations of {dtype, MathFidelity} that are Pareto-optimal — not dominated on both accuracy and throughput
  - Pareto-optimal configurations identified:
    - Gate/up: bfloat4_b + LoFi (highest throughput, acceptable accuracy for SwiGLU path)
    - Down: bfloat8_b + HiFi2 (best accuracy-throughput balance for residual-stream contribution)
    - Dense MLP layers: bfloat8_b + HiFi2 (conservative choice for non-expert FFNs)
  - Why full bfloat4_b (all projections) is not on the Pareto frontier: down-projection PCC drops below 0.94, causing unacceptable perplexity degradation for most use cases

---

### Chapter 5: Per-Projection Quantization Strategy — Gate, Up, and Down

**Description:** Provides a principled per-projection mixed-precision strategy for MoE expert FFNs, explaining why each projection gets a different dtype, how to configure TTNN accordingly, and how to validate the choices.

**Directory:** `ch05_per_projection_strategy/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Reference table: projection → recommended dtype → compute kernel config → rationale
  - Prerequisites: Chapters 1–4

- `gate_and_up_projection_strategy.md`
  - Why gate and up projections tolerate bfloat4_b: in a SwiGLU layer, the gate projection (w1) feeds into SILU activation, which clips the output to a narrow range [0, ~1]; quantization noise in the pre-activation is compressed by the SILU saturation curve
  - The multiplication path: gate_out (post-SILU) × up_out (linear); even if both contain quantization noise, their product is bounded and errors partially cancel when they are uncorrelated
  - Recommended configuration: `dtype=ttnn.bfloat4_b`, `compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI` (LoFi, `fp32_dest_acc_en=False`)
  - Validation criterion: PCC of the element-wise product (gate_out × up_out) vs. bfloat16 reference should be ≥0.96
  - Code pattern: creating bfloat4_b gate and up weight tensors and computing `ttnn.linear(x, w1, compute_kernel_config=lofi_cfg)` then `ttnn.linear(x, w3, compute_kernel_config=lofi_cfg)`

- `down_projection_strategy.md`
  - Why down projection requires bfloat8_b: the down projection (w2) maps from intermediate to hidden dimension and its output is added directly to the residual stream; errors are not compressed before the next layer's attention operation
  - Residual stream sensitivity: the token representation at each layer is the sum of attention output and MoE output; if the MoE down projection introduces significant quantization noise, that noise accumulates in the residual stream through all remaining layers
  - Recommended configuration: `dtype=ttnn.bfloat8_b`, `compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2` (HiFi2, `fp32_dest_acc_en=True`)
  - Why HiFi2 is needed for bfloat8_b down projection: with 8-bit weights and fp32 accumulation, the output achieves accuracy close to bfloat16; without `fp32_dest_acc_en`, accumulation in bfloat16 introduces additional rounding error on top of weight quantization
  - Validation criterion: PCC of w2_out vs. bfloat16 reference should be ≥0.975; if below, consider increasing to bfloat8_b + HiFi4 or falling back to bfloat16 for the down projection

- `mixed_precision_memory_layout.md`
  - How to store gate, up, and down projections with different dtypes in a single MoE module:
    - `w1_experts`: shape `[num_experts_per_device, d_model, d_ff]`, dtype bfloat4_b, DRAM_MEMORY_CONFIG
    - `w3_experts`: shape `[num_experts_per_device, d_model, d_ff]`, dtype bfloat4_b, DRAM_MEMORY_CONFIG
    - `w2_experts`: shape `[num_experts_per_device, d_ff, d_model]`, dtype bfloat8_b, DRAM_MEMORY_CONFIG
  - Total DRAM footprint per expert: `d_model × d_ff × (0.5 + 0.5 + 1.0) bytes` (bfloat4_b for w1/w3, bfloat8_b for w2) vs. `d_model × d_ff × 6 bytes` for full bfloat16 — approximately 2.67× memory reduction
  - On a T3K system with 8 chips and 32 experts per chip (256-expert model): total expert weight memory at mixed precision vs. bfloat16, and whether the model fits in DRAM with the reduced footprint
  - Transposition convention: weights are stored transposed for efficient matmul tile access; bfloat4_b requires careful tile alignment when transposing

- `qwen_adaptation_guide.md`
  - Starting from Qwen3.5-35B bfloat16 checkpoints: step-by-step weight conversion procedure for the mixed-precision strategy
  - Step 1: identify all MoE layer expert weight parameters (e.g., `model.layers.*.mlp.experts.*.gate_proj.weight`)
  - Step 2: convert gate and up projection weights to bfloat4_b via `ttnn.as_tensor(..., dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)`
  - Step 3: convert down projection weights to bfloat8_b via `ttnn.as_tensor(..., dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)`
  - Step 4: assign compute kernel configs per projection in the forward pass function
  - Step 5: run validation suite (PCC per layer, end-to-end perplexity on calibration set)
  - Checkpoint caching: save the converted TTNN weights to disk to avoid repeated conversion at each startup; using TTNN's weight serialization utilities

---

### Chapter 6: Comparative Study — DeepSeek-V3 vs. Qwen Quantization Approach

**Description:** Analyzes and contrasts the quantization strategies used in the DeepSeek-V3 TTNN implementation versus Qwen's full-precision baseline, deriving practical recommendations for teams choosing between them.

**Directory:** `ch06_comparative_study/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Side-by-side summary: DeepSeek-V3 strategy vs. Qwen bfloat16 baseline vs. recommended mixed approach for Qwen
  - Prerequisites: Chapters 1–5

- `deepseek_v3_quantization_design.md`
  - DeepSeek-V3 TTNN implementation: gate/up projections use bfloat4_b + LoFi; down projection uses bfloat8_b + HiFi2; dense MLP (non-expert) layers use bfloat8_b + HiFi2
  - Training context: DeepSeek-V3 was trained with FP8 mixed-precision quantization-aware training (bfloat4_b-aware training in the original paper); weights are pre-adapted to low-precision storage and have smaller outlier magnitudes
  - Performance outcome: MoE forward pass PCC ~0.97 vs. bfloat16 reference; full model perplexity within acceptable range for production deployment
  - Memory and throughput gains: significant DRAM footprint reduction enabling larger effective batch sizes on T3K; decode throughput improvement from reduced DRAM bandwidth
  - Key insight: the DeepSeek design works well because the model was trained to be quantization-friendly; naively applying the same strategy to a bfloat16-trained model like Qwen may not achieve the same accuracy

- `qwen_bfloat16_baseline.md`
  - Qwen3.5-35B expert weights stored as bfloat16 in the current TTNN implementation: all three projections (gate, up, down) use `ttnn.bfloat16`
  - Memory cost: 256 experts × 3 projections × d_model × d_ff × 2 bytes; with d_model=4096 and d_ff=4096, total expert weight memory ≈ 6.4 GB — compared to ~2.4 GB under the mixed-precision strategy
  - Throughput cost: bfloat16 achieve 74 TFLOPS vs. up to 148–262 TFLOPS for quantized variants; the gap is most significant during decode
  - Why Qwen uses bfloat16: conservative choice to avoid any accuracy risk on a model not trained with quantization-awareness; suitable for initial deployment but leaves significant performance headroom
  - The 16 ms gap motivation: the 16 ms per-layer latency gap mentioned in the Qwen MoE optimization analysis is partially attributable to higher DRAM bandwidth consumption from bfloat16 expert weights during decode

- `recommendations_and_decision_framework.md`
  - Decision criterion 1 — accuracy budget: if target perplexity delta ≤ 0.5 PPL from bfloat16 baseline, use bfloat8_b for all projections; if ≤ 1.5 PPL is acceptable, apply DeepSeek-style mixed precision (bfloat4_b gate/up, bfloat8_b down)
  - Decision criterion 2 — deployment regime: decode-heavy workloads (batch ≤ 8, interactive inference) benefit most from bfloat4_b gate/up due to memory-bandwidth relief; prefill-heavy workloads (batch > 32) benefit most from bfloat8_b for the compute throughput gain
  - Decision criterion 3 — training history: models trained with FP8/bfloat4 quantization-aware training (e.g., DeepSeek-V3) tolerate aggressive quantization better than purely bfloat16-trained models (e.g., Qwen)
  - Recommended starting point for Qwen: bfloat8_b for all projections + HiFi2 — this achieves ~2× memory reduction and ~2× throughput improvement with low accuracy risk; migrate to bfloat4_b gate/up only after validating perplexity on a representative benchmark
  - When to fall back to bfloat16: if any projection's output PCC falls below 0.97 after quantization, or if end-to-end perplexity delta exceeds the accepted threshold

---

### Chapter 7: Implementation and Validation Workflow

**Description:** Provides an end-to-end guide for implementing mixed-precision expert weight quantization in a TTNN MoE model: from weight conversion through correctness validation, profiling, and iterative tuning.

**Directory:** `ch07_implementation_workflow/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - The five-step workflow: (1) establish bfloat16 baseline, (2) convert weights to target dtypes, (3) validate per-layer PCC, (4) profile throughput, (5) tune compute kernel configs
  - Prerequisites: all previous chapters

- `baseline_and_weight_conversion.md`
  - Step 1: establish a correct bfloat16 baseline — run the MoE layer with bfloat16 weights and record output tensor, compute PCC against CPU reference (target: PCC > 0.999)
  - Step 2: weight conversion script template — iterating over all expert weight parameters, applying dtype conversion with tile layout, storing to DRAM; handling the gate/up vs. down distinction
  - Weight conversion correctness check: computing PCC between dequantized TTNN weight and original PyTorch weight; expected PCC > 0.99 for bfloat8_b, > 0.97 for bfloat4_b (PCC will be lower for bfloat4_b due to coarser quantization grid)
  - Handling non-tile-aligned weight shapes: padding to nearest 32×32 tile boundary before conversion; removing padding from output if necessary
  - TTNN weight serialization: saving converted weights to avoid re-conversion at every startup

- `per_layer_pcc_validation.md`
  - Setting up a per-layer PCC test harness: for each MoE layer, run bfloat16 and quantized variants with identical random input, compute PCC of outputs
  - PCC thresholds by projection:
    - w1 (gate, bfloat4_b): PCC ≥ 0.96 for layer output
    - w3 (up, bfloat4_b): PCC ≥ 0.96 for layer output
    - w2 (down, bfloat8_b): PCC ≥ 0.975 for layer output
    - Full MoE layer (mixed precision): PCC ≥ 0.97 end-to-end
  - When a layer fails the PCC threshold: diagnostic steps — check weight conversion PCC, check compute kernel config, check for shape/layout issues
  - Batch dimension effects: PCC can vary by batch size and sequence position; test with both decode (batch=1, seq=1) and prefill (batch=1, seq=2048) configurations
  - Multi-layer cumulative error test: run N consecutive MoE layers with quantized weights and measure drift from bfloat16 reference at layer N; flag if per-layer PCC is below 0.975 since cumulative error compounds

- `throughput_profiling.md`
  - Using TTNN's device profiler (`ttnn.device.enable_program_cache`, Tracy traces) to measure per-op latency
  - Measuring expert FFN latency breakdown: gate matmul, up matmul, elementwise mul + SILU, down matmul — identifying which projection is the bottleneck
  - Expected latency profile under mixed precision: gate + up at bfloat4_b should run faster than bfloat8_b equivalents; down at bfloat8_b is the controlled bottleneck
  - Comparing decode vs. prefill throughput under bfloat16 vs. bfloat8_b vs. bfloat4_b gate/up: quantifying the decode memory-bandwidth improvement
  - T3K multi-chip considerations: cross-chip all-to-all latency often dominates over expert FFN compute in small batch decode; measure whether quantization's throughput gains are visible above the communication floor

- `iterative_tuning_guide.md`
  - Decision tree for tuning after initial validation:
    - If PCC < threshold for down projection: upgrade bfloat8_b → bfloat16, or HiFi2 → HiFi4
    - If PCC < threshold for gate/up: upgrade bfloat4_b → bfloat8_b, or LoFi → HiFi2
    - If throughput improvement is insufficient: check if matmul is truly memory-bound; consider bfloat4_b for down projection only if PCC budget allows
  - Calibration perplexity procedure: compute perplexity on 512-token WikiText-2 or C4 samples with quantized weights; compare delta from bfloat16 baseline; accept if delta ≤ 1.0 PPL for bfloat8_b or ≤ 2.0 PPL for mixed bfloat4_b+bfloat8_b
  - Locking in a final configuration: document the chosen {dtype, MathFidelity} tuple for each projection, rationale, and the PCC/perplexity evidence; store in the model config dataclass alongside memory and program configs
  - Regression testing: add quantization-specific tests to the model test suite asserting PCC ≥ threshold for both decode and prefill modes; run after each checkpoint update or TTNN version bump

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **bfloat16** | Standard 16-bit brain floating-point format: 1 sign, 8 exponent, 7 mantissa bits; `ttnn.bfloat16` |
| **bfloat8_b** | Block floating-point 8-bit format: 8 bits per element with shared exponent across a 32×32 tile; `ttnn.bfloat8_b` |
| **bfloat4_b** | Block floating-point 4-bit format: 4 bits per element with shared exponent across a 32×32 tile; `ttnn.bfloat4_b` |
| **tile** | The 32×32 element atomic compute unit on Tenstorrent hardware; all quantized dtypes require tile layout |
| **MathFidelity** | A `WormholeComputeKernelConfig` setting controlling accumulation precision: LoFi < HiFi2 < HiFi4 |
| **LoFi** | `MathFidelity.LoFi`; single-pass accumulation, no FP32 dest accumulation; highest throughput, lowest precision |
| **HiFi2** | `MathFidelity.HiFi2`; two-pass accumulation with optional FP32 dest; balanced accuracy and throughput |
| **HiFi4** | `MathFidelity.HiFi4`; four-pass accumulation with FP32 dest; highest accuracy, lowest throughput |
| **fp32_dest_acc_en** | `WormholeComputeKernelConfig` flag; when True, accumulates matmul output in FP32 before packing to output dtype |
| **gate projection (w1)** | The first linear layer in a SwiGLU FFN; its output feeds into SILU activation |
| **up projection (w3)** | The second linear layer in a SwiGLU FFN; its output is multiplied with the SILU-gated w1 output |
| **down projection (w2)** | The third linear layer in a SwiGLU FFN; maps from intermediate dimension back to hidden dimension; output enters the residual stream |
| **PCC** | Pearson Cross-Correlation; the primary output correctness metric in TTNN development; computed via `comp_pcc` |
| **expert capacity** | Maximum number of tokens routed to a single expert per forward pass |
| **MoE** | Mixture of Experts; a transformer FFN variant replacing a single FFN with a router plus a pool of expert FFNs |
| **T3K** | Tenstorrent's 8-chip Wormhole mesh product; 8 Wormhole B0 ASICs connected via Ethernet links |
| **Wormhole B0** | The Tenstorrent ASIC used in n150 and n300 boards; 72 Tensix cores per chip |
| **DRAM_MEMORY_CONFIG** | `ttnn.DRAM_MEMORY_CONFIG`; stores tensors in off-chip DRAM; default for large weight tensors |
| **L1_MEMORY_CONFIG** | `ttnn.L1_MEMORY_CONFIG`; stores tensors in on-chip SRAM; default for activations in decode mode |
| **program config** | A TTNN dataclass (e.g., `MatmulMultiCoreReuseMultiCastProgramConfig`) specifying kernel tile and grid parameters |
| **compute kernel config** | `WormholeComputeKernelConfig`; specifies MathFidelity, fp32_dest_acc_en, packer_l1_acc for a matmul/linear op |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions on first use, e.g., `[num_experts_per_device, d_model, d_ff]`.
- Tile counts are written as subscripted T: `M_t = M / 32`, `K_t = K / 32`, `N_t = N / 32`.
- Memory sizes are given in bytes (B), kilobytes (KB), megabytes (MB), gigabytes (GB) — do not mix SI and IEC prefixes.
- Throughput is in TFLOPS (teraFLOPS = 10^12 floating-point operations per second).
- Bandwidth is in GB/s (gigabytes per second).
- Code blocks use Python syntax and assume `import ttnn` and `import torch` are in scope; device and memory config objects are named consistently across all chapters.
- Performance numbers refer to Wormhole B0 (n150 single chip) unless explicitly stated as n300 or T3K.
- PCC values are reported as 4-decimal floats (e.g., 0.9750) to distinguish meaningful thresholds.

### Formatting Rules

- Every chapter directory contains an `index.md` providing chapter overview, learning objectives, prerequisites, and a navigation list of files.
- Code examples use fenced blocks with ` ```python ` and include inline comments on non-obvious lines.
- Warnings about correctness pitfalls are formatted as `> **Warning:** ...` blockquotes.
- Performance-sensitive recommendations are formatted as `> **Tip:** ...` blockquotes.
- Tables are used for comparisons (dtype properties, PCC thresholds, decision matrices); prose is used for causal reasoning.
- Cross-chapter references use the form: "see Chapter N, `filename.md`" with exact chapter number and filename.
- Every content file ends with a `## Next Steps` section pointing to the next file or chapter.
- Abbreviations are spelled out on first use in each file, even if defined elsewhere.

---

## 4. Cross-Chapter Dependencies

The guide is designed to be read front-to-back, with each chapter building on the previous. The dependencies below clarify which later chapters rely on specific concepts from earlier ones.

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: Quantization Formats | None (foundational) |
| Ch 2: TTNN Quantization API | Ch 1 (format definitions, tile layout constraint, throughput numbers) |
| Ch 3: Accuracy Analysis | Ch 1 (numeric precision of each dtype), Ch 2 (PCC validation patterns, compute kernel configs) |
| Ch 4: Throughput Impact | Ch 1 (TFLOPS numbers, bfloat8_b/bfloat4_b bandwidth reduction), Ch 2 (compute kernel config and MathFidelity definitions) |
| Ch 5: Per-Projection Strategy | Ch 1–4 (all); directly synthesizes format knowledge, API patterns, accuracy analysis, and throughput trade-offs |
| Ch 6: Comparative Study | Ch 3 (accuracy comparison methodology), Ch 4 (throughput analysis), Ch 5 (mixed-precision configuration patterns) |
| Ch 7: Implementation Workflow | All previous chapters; serves as the practical execution guide referencing concepts from Ch 1–6 |

**Specific forward references to flag:**

- Ch 1 (`bfloat8_b_format.md` and `bfloat4_b_format.md`) introduces throughput numbers that are analyzed in depth in Ch 4 (`prefill_compute_throughput.md` and `decode_memory_bandwidth.md`). Ch 1 should note that the throughput numbers will be contextualized in Ch 4.
- Ch 2 (`compute_kernel_config.md`) introduces LoFi and HiFi2 configuration objects. Ch 5 (`gate_and_up_projection_strategy.md` and `down_projection_strategy.md`) assigns specific configs to specific projections. Ch 2 must not make projection-specific recommendations — that is reserved for Ch 5.
- Ch 3 (`projection_sensitivity.md`) describes the empirical DeepSeek-V3 mixed-precision choices (bfloat4_b for w1/w3, bfloat8_b for w2) as evidence. Ch 5 provides the mechanistic explanation for why that choice is correct. Ch 3 should forward-reference Ch 5 for the full rationale.
- Ch 4 (`bandwidth_vs_accuracy_tradeoff.md`) identifies the Pareto-optimal configurations. Ch 5 and Ch 6 both cite this Pareto frontier. Ch 4 must be written before Ch 5 and Ch 6 to ensure consistent enumeration of the frontier points.
- Ch 7 (`per_layer_pcc_validation.md`) specifies PCC thresholds per projection. These thresholds must be consistent with those established in Ch 3 (`accuracy_metrics_for_moe.md`) and Ch 5 (`gate_and_up_projection_strategy.md`, `down_projection_strategy.md`). Any threshold changes must be propagated to all three locations.
- Ch 6 (`recommendations_and_decision_framework.md`) refers to the "16 ms gap" between Qwen and optimized MoE decode performance. This is an external reference to the Qwen MoE optimization research; Ch 6 should cite it explicitly rather than re-derive it.
