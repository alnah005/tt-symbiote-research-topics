# Plan: Gemma 4 Vision Encoder TTNN Porting Strategy

---

## 1. Audience

**Primary audience:** ML engineers and model-bringup engineers at Tenstorrent who are responsible for porting the Gemma 4 multimodal model to TTNN on Wormhole hardware. They are comfortable with:

- PyTorch and HuggingFace Transformers model code
- Vision Transformer (ViT) architecture: patch embeddings, multi-head self-attention, FFN blocks
- Basic TTNN op usage (`ttnn.matmul`, `ttnn.linear`, `ttnn.to_device`, memory configs)
- The existing Gemma 3 TTNN codebase at `models/demos/multimodal/gemma3/tt/`
- Rotary Position Embeddings (RoPE) in the context of language models

**What they do NOT need to know in advance:**

- How Gemma 4's vision encoder differs from Gemma 3's SigLIP encoder at the config and module level
- How 2D factored (multidimensional) RoPE works and how it maps to existing TTNN RoPE utilities
- The latency tradeoffs between running the vision encoder on CPU vs. porting it to TTNN
- How Gemma 4's variable-aspect-ratio image processing and adaptive pooling affect the porting strategy

This guide fills those gaps, starting from a side-by-side architectural comparison and culminating in a concrete porting recommendation with implementation steps.

---

## 2. Chapter List

---

### Chapter 1: Gemma 4 Vision Encoder Architecture Overview

**Description:** Describes the Gemma 4 vision encoder end-to-end, establishing the module hierarchy and data flow that all subsequent chapters reference.

**Directory:** `ch01_gemma4_vision_architecture/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to sub-topics
  - Prerequisites: familiarity with ViT and HuggingFace Transformers

- `module_hierarchy.md`
  - The top-level `Gemma4VisionModel` and its three sub-components: `Gemma4VisionPatchEmbedder`, `Gemma4VisionEncoder`, and `Gemma4VisionPooler`
  - `Gemma4VisionEncoder` internals: a stack of 27 `Gemma4VisionEncoderLayer` blocks, each containing `Gemma4VisionAttention` and an MLP with GeLU activation
  - The `Gemma4MultimodalEmbedder` projection layer that maps vision hidden states (dim 1152) to language model hidden states via RMSNorm + linear
  - Complete data flow: raw pixels -> patch embedding -> 27 encoder layers with 2D RoPE -> adaptive pooling -> RMSNorm + projection -> soft tokens for the language model

- `config_parameters.md`
  - Full `Gemma4VisionConfig` parameter table for the 31B model: `hidden_size=1152`, `num_hidden_layers=27`, `num_attention_heads=16`, `num_key_value_heads=16`, `head_dim=72`, `intermediate_size=4304`, `patch_size=16`, `pooling_kernel_size=3`, `position_embedding_size=10240`
  - RoPE parameters: `rope_theta=100.0`, `rope_type="default"`
  - Parameter count: approximately 550M parameters for the vision encoder
  - Explanation of `default_output_length=280` and the configurable token budgets (70, 140, 280, 560, 1120)

- `variable_resolution_processing.md`
  - How Gemma 4 preserves original aspect ratios instead of squashing to a fixed square
  - The constraint that both height and width must be divisible by 48 (derived from patch_size=16 times pooling_kernel_size=3)
  - How token budget maps to total pixel count: 280 tokens corresponds to approximately 645K pixels
  - No ImageNet mean/std normalization; the patch embedding layer handles value scaling internally
  - Implications for TTNN: variable input shapes across images in a batch

---

### Chapter 2: Gemma 3 SigLIP vs. Gemma 4 Vision Encoder Comparison

**Description:** Provides a detailed side-by-side comparison of the Gemma 3 SigLIP encoder and the Gemma 4 vision encoder to quantify reuse potential.

**Directory:** `ch02_siglip_vs_gemma4_comparison/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: what is shared, what differs, what is new in Gemma 4

- `config_diff.md`
  - Side-by-side config comparison table:
    - Shared: `hidden_size=1152`, `num_hidden_layers=27`, `num_attention_heads=16`, `intermediate_size=4304`, `hidden_activation=gelu_pytorch_tanh`
    - Changed: `patch_size` (Gemma 3: 14, Gemma 4: 16), image input handling (Gemma 3: fixed 896x896, Gemma 4: variable aspect ratio)
    - New in Gemma 4: `num_key_value_heads=16`, `head_dim=72`, `pooling_kernel_size=3`, `position_embedding_size=10240`, `rope_parameters` with `rope_theta=100.0`
    - Removed from Gemma 4: SigLIP absolute position embeddings, fixed-square input assumption
  - Impact of `patch_size` change from 14 to 16 on the patch embedding convolution weights and output sequence length

- `module_mapping.md`
  - Module-by-module mapping between existing Gemma 3 TTNN files and their Gemma 4 equivalents:
    - `siglip_vision_embedding.py` -> needs rewrite (patch_size change, 2D learned position embeddings replace absolute embeddings)
    - `gemma_vision_block.py` / `gemma_image_block.py` -> largely reusable (same hidden_size, same layer count, same activation)
    - `gemma_image_attention.py` -> needs modification (add 2D RoPE, change from absolute position bias to rotary embeddings)
    - `gemma_image_mlp.py` -> directly reusable (same hidden_size=1152, same intermediate_size=4304, same GeLU activation)
    - `gemma_vision_rmsnorm.py` -> directly reusable (same eps=1e-6, same hidden_size)
    - `multi_modal_projector.py` -> needs modification (adaptive pooling with kernel_size=3 replaces fixed pooling, standardization option added)
    - `gemma_conv2d_patch.py` -> needs update (patch_size 14->16, kernel/stride change)
  - Reuse percentage estimate: approximately 40-50% of existing code can be reused directly, 30% needs modification, 20% needs new implementation

- `positional_encoding_shift.md`
  - Gemma 3 SigLIP: learned absolute position embeddings added after patch embedding, supports only fixed 896x896 input
  - Gemma 4: dual positional encoding system:
    1. Learned 2D position embeddings (shape `[2, 10240, 1152]`) added to patch embeddings, indexed by (x, y) grid coordinates
    2. 2D factored RoPE applied within each attention layer, with `rope_theta=100.0`
  - Why this change matters: variable aspect ratios require position encodings that generalize across different grid shapes
  - Implications for TTNN: need to implement or adapt RoPE for 2D spatial coordinates in the vision encoder attention

---

### Chapter 3: 2D Factored RoPE for Vision — Theory and TTNN Mapping

**Description:** Explains the 2D multidimensional RoPE used in Gemma 4's vision encoder and analyzes whether existing TTNN RoPE implementations can be reused or extended.

**Directory:** `ch03_2d_factored_rope/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisites: basic understanding of standard 1D RoPE from language models

- `multidimensional_rope_theory.md`
  - Standard 1D RoPE recap: frequency computation with `theta`, rotation of query/key pairs, how `rope_theta` controls the wavelength spectrum
  - Extension to 2D: each spatial dimension (x, y) gets independent frequency ranges
  - The head dimension is split by the number of spatial dimensions (2 for images): first half encodes x-position, second half encodes y-position
  - Concatenation of cos/sin embeddings from both dimensions
  - Why `rope_theta=100.0` (vs. typical 10000.0 for text): the vision encoder operates over a much smaller position range (image grid coordinates, not token positions in a 128K context), so a smaller theta produces higher-frequency rotations appropriate for the spatial scale

- `reference_implementation.md`
  - Walkthrough of the HuggingFace `Gemma4VisionRotaryEmbedding` class
  - The `apply_multidimensional_rope()` function: splits input along head dimension, applies RoPE piece-wise to x and y position components, concatenates results
  - How position IDs are computed: 2D grid coordinates derived from the image's patch grid dimensions (height_patches, width_patches)
  - Key numerical properties: frequency range, position range, and the resulting rotation angles for typical image sizes

- `ttnn_rope_gap_analysis.md`
  - Current TTNN RoPE capabilities: optimized for 1D sequence positions in language model decoders
  - Gap 1: TTNN RoPE kernels assume a 1D position index; Gemma 4 vision needs a 2D (x, y) coordinate pair
  - Gap 2: the head dimension split (first half for x, second half for y) is not natively supported
  - Gap 3: `rope_theta=100.0` is non-standard but should not cause numerical issues — it only changes the frequency table values
  - Three implementation strategies ranked by effort:
    1. **Precompute on CPU, apply on device** (lowest effort): compute cos/sin tables on CPU for each image resolution, transfer to device, use element-wise multiply
    2. **Compose from existing TTNN ops** (medium effort): use `ttnn.split`, apply standard RoPE to each half, then `ttnn.concat`
    3. **Custom TTNN kernel** (highest effort): implement a fused 2D RoPE kernel for maximum performance
  - Recommendation for initial bringup vs. performance-optimized path

---

### Chapter 4: Patch Embedding and Adaptive Pooling in TTNN

**Description:** Covers the two vision-specific operations that differ most from Gemma 3 and require new or significantly modified TTNN implementations.

**Directory:** `ch04_patch_embedding_and_pooling/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Why these two modules are critical path items for the port

- `patch_embedding_port.md`
  - Gemma 4 `Gemma4VisionPatchEmbedder`: flattens 16x16x3 patches and projects to hidden_size=1152 via a linear layer (not Conv2d)
  - Comparison with Gemma 3: SigLIP uses Conv2d with kernel_size=14, stride=14; Gemma 4 uses flatten + linear with patch_size=16
  - Existing TTNN module `gemma_conv2d_patch.py`: can be replaced with a reshape + `ttnn.linear` operation
  - 2D learned position embeddings: shape `[2, 10240, 1152]`, indexed separately for x and y coordinates, summed and added to patch embeddings
  - TTNN implementation plan: `ttnn.embedding` lookup for x and y positions, element-wise add, then add to patch embeddings
  - Variable input shapes: the patch grid dimensions change per image — implications for program caching and tracing

- `adaptive_pooling_port.md`
  - Gemma 4 `Gemma4VisionPooler`: 2D average pooling that reduces the patch token count to the target `output_length`
  - How pooling_kernel_size=3 interacts with the grid: for a default 280-token output from a ~840-patch input, the pooler averages 3x3 neighborhoods
  - Optional standardization: learned bias and scale parameters applied after pooling
  - Comparison with Gemma 3 `multi_modal_projector.py`: Gemma 3 uses a fixed average pooling; Gemma 4's pooling adapts to the input grid shape
  - TTNN implementation options:
    1. Use `ttnn.avg_pool2d` if it supports the required kernel/stride/padding configuration
    2. Reshape to 2D grid, apply manual mean reduction using `ttnn.reshape` and `ttnn.mean`
    3. Implement as a custom op if neither built-in option handles variable spatial dimensions
  - The RMSNorm + linear projection in `Gemma4MultimodalEmbedder` is straightforward and maps directly to existing TTNN ops

---

### Chapter 5: CPU vs. TTNN Latency Analysis

**Description:** Estimates the latency of running the Gemma 4 vision encoder on CPU versus TTNN to determine whether porting is justified.

**Directory:** `ch05_cpu_vs_ttnn_latency/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary of the decision framework: when CPU is acceptable vs. when TTNN is required

- `cpu_baseline_profiling.md`
  - Methodology for profiling the vision encoder on CPU: PyTorch eager mode, `torch.profiler`, wall-clock timing
  - Expected latency breakdown by module for the 550M-parameter encoder at default 280-token budget:
    - Patch embedding: small fraction of total time
    - 27 encoder layers (attention + MLP): dominant cost (~85-90% of total)
    - Pooling + projection: small fraction
  - CPU latency estimates at different token budgets (70, 280, 560, 1120 tokens) and batch sizes (1, 4, 8)
  - Hardware assumptions: Intel Xeon or AMD EPYC server CPUs typically paired with Wormhole cards

- `ttnn_latency_projection.md`
  - Estimating TTNN latency from first principles: matmul FLOP counts, Wormhole B0 peak throughput, memory bandwidth
  - Reference point: the existing Gemma 3 SigLIP TTNN encoder performance (the tt-metal repo tracks vision perf targets for T3K)
  - Key factors that affect TTNN speedup over CPU:
    1. Attention matmuls at hidden_size=1152, num_heads=16, head_dim=72 — these are moderately sized and map well to Wormhole's 8x8 grid
    2. MLP matmuls at 1152->4304->1152 — standard sizes for TTNN
    3. RoPE overhead: precomputed tables are cheap, but the 2D split adds a small constant
    4. Variable input shapes: may prevent tracing and force recompilation for different image resolutions
  - Projected speedup factor: vision encoder is compute-bound at batch >= 1, so TTNN should provide significant speedup
  - The "break-even" analysis: accounting for host-to-device transfer time for pixel data, the minimum batch size or token budget where TTNN wins

- `decision_matrix.md`
  - Decision matrix with recommendations based on deployment scenario:
    - **Single-image inference, low token budget (70-140)**: CPU may be acceptable; vision encoder latency is small relative to LLM decode
    - **Batch inference or high token budget (560-1120)**: TTNN porting strongly recommended
    - **Continuous batching with mixed modalities**: TTNN porting required to avoid CPU bottleneck stalling the LLM pipeline
    - **Prefill-dominated workloads (long image descriptions)**: vision encoder runs once per image, so even CPU latency may be amortized over many decode steps
  - How Gemma 3's existing TTNN vision perf targets inform expectations for Gemma 4

---

### Chapter 6: Reuse Strategy for Existing Gemma 3 TTNN Modules

**Description:** Provides a concrete file-by-file reuse plan for the existing Gemma 3 TTNN vision encoder code, identifying what can be copied, what needs modification, and what must be written from scratch.

**Directory:** `ch06_reuse_strategy/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary: reuse scorecard for each existing module

- `direct_reuse_modules.md`
  - Modules that can be reused with no or minimal changes:
    - `gemma_image_mlp.py`: hidden_size=1152 and intermediate_size=4304 are identical; GeLU activation is the same; only verify weight loading paths
    - `gemma_vision_rmsnorm.py`: same eps=1e-6, same hidden_size=1152; direct reuse
    - `gemma_image_block.py` / `gemma_vision_block.py`: the layer structure (attention -> residual -> MLP -> residual) is architecturally identical; verify normalization placement (pre-norm in both)
    - `model_config.py`: can be extended with Gemma 4 config parameters; sharding and memory config strategies likely transfer
    - `load_checkpoints.py`: weight loading utilities need updated key mappings but the infrastructure is reusable
  - Estimated effort: 1-2 days of validation and minor adjustments

- `modification_required_modules.md`
  - Modules that need targeted modifications:
    - `gemma_image_attention.py`: must add 2D RoPE application (currently uses no rotary embeddings or uses absolute position bias); the Q/K/V projection and output projection shapes are the same (hidden_size=1152, num_heads=16) but head_dim changes if Gemma 3 used 1152/16=72 (same) — verify and add RoPE call
    - `multi_modal_projector.py`: must change from fixed pooling to adaptive 2D average pooling with kernel_size=3; add optional standardization (learned bias + scale); the final RMSNorm + linear projection is similar
    - `gemma_conv2d_patch.py` / `siglip_vision_embedding.py`: patch_size changes from 14 to 16; Gemma 4 uses flatten+linear instead of Conv2d; position embedding changes from absolute to 2D learned
  - Estimated effort per module and key implementation notes

- `new_implementation_modules.md`
  - Modules that must be written from scratch:
    - **2D RoPE module**: `gemma4_vision_rope.py` — precompute 2D frequency tables, apply multidimensional rotation; see Chapter 3 for implementation strategies
    - **2D learned position embedding**: `gemma4_vision_position_embedding.py` — dual embedding lookup (x, y) with `position_embedding_size=10240`, sum and add to patches
    - **Variable-resolution image preprocessor**: logic to compute patch grid dimensions, enforce divisibility-by-48, compute 2D position indices per image
  - Estimated effort: 3-5 days for initial implementation, additional time for performance optimization

---

### Chapter 7: Implementation Roadmap and Risk Assessment

**Description:** Lays out a phased implementation plan with milestones, estimated timelines, and a risk register for the porting effort.

**Directory:** `ch07_implementation_roadmap/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - High-level timeline summary

- `phased_plan.md`
  - **Phase 1 — CPU reference and correctness baseline (1 week)**
    - Run HuggingFace Gemma 4 vision encoder on CPU with reference inputs
    - Capture intermediate activations at every layer for PCC validation
    - Establish the golden reference outputs for multiple image sizes and token budgets
  - **Phase 2 — Module-level TTNN port (2-3 weeks)**
    - Port modules in dependency order: RMSNorm -> MLP -> Attention (without RoPE) -> Attention (with 2D RoPE) -> Patch Embedding -> Position Embedding -> Pooler -> Projector
    - Validate each module independently against CPU reference (PCC > 0.999 for bfloat16)
    - Use CPU-precomputed RoPE tables initially to unblock attention layer validation
  - **Phase 3 — End-to-end integration and optimization (1-2 weeks)**
    - Assemble full vision encoder on TTNN
    - Profile end-to-end latency and compare against CPU baseline
    - Optimize critical path: attention matmuls, MLP matmuls, memory configs
    - Investigate tracing feasibility for fixed token budgets (280 is the default — trace this first)
  - **Phase 4 — Variable resolution support (1 week)**
    - Handle multiple token budgets without recompilation where possible
    - If tracing is required per resolution, pre-trace the five supported budgets (70, 140, 280, 560, 1120)
    - Validate correctness across all supported resolutions

- `risk_register.md`
  - **Risk 1: Variable input shapes prevent tracing** — Mitigation: pre-trace all five token budgets; pad to nearest supported budget at runtime
  - **Risk 2: 2D RoPE implementation has numerical divergence** — Mitigation: validate against HuggingFace reference with tight PCC thresholds; use float32 for frequency table computation
  - **Risk 3: Adaptive pooling has no direct TTNN op** — Mitigation: implement as reshape + manual mean; fallback to CPU for pooling if TTNN implementation is blocked
  - **Risk 4: patch_size=16 Conv2d or linear does not map efficiently to tile size 32** — Mitigation: pad as needed; profile to confirm overhead is acceptable
  - **Risk 5: Vision encoder latency on TTNN does not justify porting effort** — Mitigation: run Phase 1 CPU profiling first; abort TTNN port if CPU latency is < 5% of total inference time for the target deployment scenario
  - **Risk 6: Weight format differences between Gemma 3 and Gemma 4 checkpoints** — Mitigation: update `load_checkpoints.py` key mappings early in Phase 2

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| **vision encoder** | The `Gemma4VisionModel` module that converts image pixels to soft tokens; excludes the language model |
| **SigLIP encoder** | The vision encoder used in Gemma 3, based on the SigLIP ViT architecture with absolute position embeddings |
| **patch grid** | The 2D grid of non-overlapping patches extracted from an image; dimensions are `(H/patch_size, W/patch_size)` |
| **token budget** | The target number of vision tokens output by the pooler; one of {70, 140, 280, 560, 1120} |
| **2D factored RoPE** | Multidimensional Rotary Position Embedding where the head dimension is split in half, with each half encoding one spatial axis (x or y) using independent frequency tables |
| **rope_theta** | The base frequency parameter for RoPE; Gemma 4 vision uses `theta=100.0` |
| **adaptive pooling** | The `Gemma4VisionPooler` operation that averages patch tokens in a 2D grid to reduce count to the target token budget |
| **multimodal embedder** | The `Gemma4MultimodalEmbedder` projection layer (RMSNorm + linear) that maps vision hidden states to language model dimension |
| **PCC** | Pearson Cross-Correlation — the primary correctness metric for validating TTNN outputs against CPU reference |
| **tile** | The 32x32 element atomic compute unit on Tenstorrent Wormhole hardware |
| **program cache** | TTNN's mechanism for caching compiled kernels; cache keys include tensor shapes, so variable shapes cause recompilation |
| **tracing** | TTNN's mechanism for recording and replaying a sequence of ops with fixed shapes for maximum throughput |

### Notation

- Tensor shapes are written as `[dim0, dim1, ...]` with named dimensions where helpful, e.g., `[batch, seq, hidden]`.
- Gemma 4 vision config parameters use their exact HuggingFace attribute names (e.g., `hidden_size`, `num_attention_heads`, `pooling_kernel_size`).
- TTNN op names use the `ttnn.` prefix (e.g., `ttnn.matmul`, `ttnn.linear`, `ttnn.avg_pool2d`).
- File paths in the Gemma 3 TTNN codebase are relative to `models/demos/multimodal/gemma3/tt/` unless otherwise noted.
- Performance numbers are estimates unless marked as measured; always re-profile on the target hardware and firmware.
- Code blocks use Python syntax and assume `import ttnn`, `import torch`, and `from transformers import Gemma4VisionConfig` are in scope.

### Formatting Rules

- Every chapter directory has an `index.md` that provides an overview, learning objectives, and navigation links.
- Side-by-side comparisons between Gemma 3 and Gemma 4 use tables with columns labeled **Gemma 3 (SigLIP)** and **Gemma 4**.
- Code examples are fenced with ` ```python ` and include comments explaining non-obvious lines.
- Warnings about correctness pitfalls are formatted as `> **Warning:** ...` blockquotes.
- Performance-sensitive recommendations are formatted as `> **Tip:** ...` blockquotes.
- Risk items are formatted as `> **Risk:** ...` blockquotes with severity (High/Medium/Low) and mitigation.
- All chapter files end with a "Next Steps" section pointing to the next logical file or chapter.

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends on concepts from |
|---|---|
| Ch 1: Gemma 4 Vision Architecture | None (foundational) |
| Ch 2: SigLIP vs. Gemma 4 Comparison | Ch 1 (Gemma 4 module hierarchy and config) |
| Ch 3: 2D Factored RoPE | Ch 1 (attention module structure), Ch 2 (positional encoding shift from absolute to rotary) |
| Ch 4: Patch Embedding and Pooling | Ch 1 (variable resolution processing, patch grid), Ch 2 (patch_size and pooling differences) |
| Ch 5: CPU vs. TTNN Latency | Ch 1 (parameter counts and compute profile), Ch 2 (reuse potential affects porting effort estimate) |
| Ch 6: Reuse Strategy | Ch 2 (module mapping), Ch 3 (RoPE gap analysis), Ch 4 (patch embedding and pooling implementation) |
| Ch 7: Implementation Roadmap | All previous chapters; synthesizes findings into an actionable plan |

**Specific forward references to flag:**

- Ch 2 (`module_mapping.md`) references the RoPE implementation strategies from Ch 3 (`ttnn_rope_gap_analysis.md`) — Ch 3 must define the three strategies before Ch 6 references them.
- Ch 2 (`config_diff.md`) establishes the shared/changed/new parameter classification — this classification is used without re-derivation in Ch 4, Ch 5, and Ch 6.
- Ch 3 (`ttnn_rope_gap_analysis.md`) produces a ranked list of implementation strategies — Ch 6 (`new_implementation_modules.md`) references the recommended strategy.
- Ch 5 (`decision_matrix.md`) produces the port-or-not recommendation — Ch 7 (`phased_plan.md`) assumes the decision is to port and references Ch 5 for justification.
- Ch 6 (`direct_reuse_modules.md` and `modification_required_modules.md`) produces effort estimates — Ch 7 (`phased_plan.md`) aggregates these into the timeline.
