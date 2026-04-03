# Phased Implementation Plan

This file describes the four-phase plan for porting the Gemma 4 vision encoder to TTNN on Wormhole hardware. Each phase has defined entry criteria, tasks, deliverables, and a phase gate that must be passed before proceeding.

The effort estimates from [Chapter 6 — Reuse Strategy](../ch06_reuse_strategy/index.md) are aggregated here into the timeline. The latency justification from [Chapter 5 — CPU vs. TTNN Latency Analysis](../ch05_cpu_vs_ttnn_latency/index.md) is assumed to support proceeding with the port.

## Phase 1 — CPU Reference and Correctness Baseline (1 Week)

### Objective

Establish a golden reference implementation by running the HuggingFace Gemma 4 vision encoder on CPU. Capture intermediate activations at every module boundary for use as PCC validation targets throughout the remaining phases.

### Entry Criteria

- Access to the Gemma 4 31B model weights (HuggingFace checkpoint)
- Working HuggingFace Transformers installation with Gemma 4 support
- Test images covering multiple aspect ratios and resolutions

### Tasks

| Task | Description | Effort |
|------|-------------|--------|
| 1.1 | Load `Gemma4VisionModel` from the HuggingFace checkpoint and run inference on a single test image at the default 280-token budget | 0.5 day |
| 1.2 | Instrument the model to capture intermediate activations: patch embedding output, each of the 27 encoder layer outputs, pooler output, and final projection output | 1 day |
| 1.3 | Run the instrumented model on a test suite covering all five token budgets (70, 140, 280, 560, 1120) with at least two different aspect ratios per budget | 1 day |
| 1.4 | Save all intermediate activations as `.pt` files with a standardized naming convention (e.g., `ref_layer_12_budget_280_landscape.pt`) | 0.5 day |
| 1.5 | Profile the CPU model using `torch.profiler` to measure per-module latency breakdown | 1 day |
| 1.6 | Document the CPU baseline latency at each token budget and batch size (1, 4, 8) | 1 day |

### Deliverables

- Golden reference activation snapshots for all 27 encoder layers at five token budgets
- CPU latency profile with per-module breakdown
- Go/no-go recommendation based on the [decision matrix](../ch05_cpu_vs_ttnn_latency/decision_matrix.md)

### Phase Gate

> **Gate 1:** Review the CPU profiling results. If the vision encoder accounts for less than 5% of end-to-end inference time at the target deployment configuration, escalate the decision to the technical lead. The TTNN port may not be the highest-priority optimization.

If Gate 1 passes (vision encoder latency is a meaningful fraction of total inference time), proceed to Phase 2.

## Phase 2 — Module-Level TTNN Port (2-3 Weeks)

### Objective

Port each vision encoder module to TTNN individually, validating each against the CPU reference activations from Phase 1. By the end of this phase, every module has a standalone TTNN implementation that passes PCC > 0.999 against the golden reference.

### Entry Criteria

- Phase 1 deliverables complete (golden reference activations available)
- TTNN development environment configured for Wormhole B0

### Module Port Order

Modules are ported in dependency order as established in [Chapter 6](../ch06_reuse_strategy/index.md). The following table adds timeline targets:

```
Week 1:
  [Day 1-2]  RMSNorm (direct reuse) + MLP (direct reuse)
  [Day 2-3]  2D RoPE module (new) + 2D position embedding (new)
  [Day 3-5]  Attention without RoPE (modify from Gemma 3)

Week 2:
  [Day 1-2]  Attention with 2D RoPE integration
  [Day 2-3]  Single encoder layer (assemble: attention + MLP + RMSNorm)
  [Day 3-5]  Patch embedding (modify from Gemma 3)

Week 3:
  [Day 1-2]  Position embedding integration into patch embedder
  [Day 2-3]  Adaptive pooler (modify from Gemma 3 projector)
  [Day 3-5]  Multimodal embedder (pooler + RMSNorm + projection)
```

> **Tip:** The RMSNorm, MLP, 2D RoPE, and 2D position embedding modules have no mutual dependencies. Assign them to two engineers working in parallel during Week 1 to compress the schedule.

### Validation Protocol

Each module is validated using the following protocol:

1. **Load weights** from the HuggingFace checkpoint into the TTNN module.
2. **Feed the reference input** (the output of the preceding module in the reference pipeline) through the TTNN module.
3. **Compare the TTNN output** against the corresponding reference activation using Pearson Cross-Correlation.
4. **Pass criterion:** PCC > 0.999 for BF16. If a module fails, investigate whether the issue is in weight loading, numerical precision, or implementation logic.

```python
import torch
from models.utility_functions import comp_pcc

def validate_module(ttnn_output, reference_output, threshold=0.999):
    """Validate a TTNN module output against the CPU reference."""
    ttnn_torch = ttnn.to_torch(ttnn_output)
    passing, pcc_value = comp_pcc(reference_output, ttnn_torch, threshold)
    assert passing, f"PCC {pcc_value:.6f} below threshold {threshold}"
    return pcc_value
```

> **Warning:** The 2D RoPE module is the most likely source of PCC failures. The cos/sin table computation involves transcendental functions (sin, cos) that are sensitive to precision. Always compute the frequency tables in float32 on the CPU, then convert to BF16 only when transferring to the device. See [Chapter 3 — TTNN RoPE Gap Analysis](../ch03_2d_factored_rope/ttnn_rope_gap_analysis.md) for details.

### Handling PCC Failures

If a module fails PCC validation:

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| PCC < 0.999 but > 0.99 | BF16 precision loss in intermediate computations | Identify the operation causing precision loss; try float32 accumulation for that specific op |
| PCC < 0.99 | Implementation bug or incorrect weight mapping | Dump intermediate tensors within the module and bisect to find the diverging operation |
| PCC near 0 or negative | Completely wrong output | Check weight loading key mappings; verify tensor shapes and layouts |

### Deliverables

- Standalone TTNN implementation for each of the 11 modules listed in the [Chapter 6 reuse scorecard](../ch06_reuse_strategy/index.md)
- PCC validation report: per-module PCC values at the 280-token budget
- Updated weight key mapping in `gemma4_load_checkpoints.py`

### Phase Gate

> **Gate 2:** All 11 modules pass PCC > 0.999 individually at the 280-token budget. If any module consistently fails below 0.99, investigate root cause before proceeding. Particular attention to the attention module with 2D RoPE — if PCC is marginal (0.99-0.999), run an end-to-end sanity check to assess error accumulation across 27 layers.

## Phase 3 — End-to-End Integration and Optimization (1-2 Weeks)

### Objective

Assemble the individually validated modules into a complete vision encoder running end-to-end on TTNN. Profile latency, optimize the critical path, and investigate tracing feasibility.

### Entry Criteria

- Phase 2 deliverables complete (all modules pass PCC individually)

### Tasks

| Task | Description | Effort |
|------|-------------|--------|
| 3.1 | Assemble `TtGemma4VisionModel`: stack 27 encoder layers, wire in patch embedding, pooler, and projector | 1-2 days |
| 3.2 | End-to-end PCC validation at 280-token budget: compare final output against CPU reference. Target PCC > 0.998 (slightly relaxed due to error accumulation across 27 layers) | 1 day |
| 3.3 | Profile end-to-end latency using `ttnn.profiler` and identify top-5 ops by execution time | 1 day |
| 3.4 | Optimize memory configs: tune sharding strategies for attention and MLP matmuls based on profiler output | 2-3 days |
| 3.5 | Investigate tracing: attempt to trace the full encoder at the 280-token budget. If successful, measure traced vs. non-traced latency | 1-2 days |
| 3.6 | Compare end-to-end TTNN latency against the CPU baseline from Phase 1 | 0.5 day |

### Optimization Priorities

Based on the architecture analysis from [Chapter 1](../ch01_gemma4_vision_architecture/index.md) and the latency model from [Chapter 5](../ch05_cpu_vs_ttnn_latency/index.md), the optimization priority order is:

1. **Attention matmuls** (Q*K^T and attn*V): 27 layers x 16 heads, largest compute contribution. Tune sharding to maximize utilization of the 8x8 core grid.
2. **MLP matmuls** (gate-projection 1152->4304, up-projection 1152->4304, down-projection 4304->1152): 27 layers, second-largest compute contribution.
3. **2D RoPE application**: six element-wise ops per layer (mul, mul, add for each spatial half). Consider fusing if profiling shows significant overhead.
4. **Patch embedding linear**: single matmul, runs once per image. Low priority unless it is unexpectedly slow.
5. **Adaptive pooling**: runs once per image after all 27 layers. Optimize only if it shows up in the profile.

### Tracing Strategy

TTNN tracing records a fixed sequence of ops and replays them without re-dispatching. For the vision encoder:

- **Fixed token budget (280):** The patch count is fixed at approximately 2520 patches (depending on the exact aspect ratio quantization). If all images in a batch use the same budget, tracing should work directly.
- **Mixed token budgets:** Tracing requires fixed shapes. If different images in a batch have different patch counts, tracing is not directly applicable. See Phase 4 for the mitigation strategy.

> **Tip:** Start tracing attempts with the 280-token budget and batch size 1. This is the simplest case and the most common deployment scenario. If tracing succeeds here, extend to batch > 1 and other budgets in Phase 4.

### Deliverables

- Complete `TtGemma4VisionModel` running end-to-end on TTNN
- End-to-end PCC validation report (target > 0.998)
- Latency comparison: TTNN vs. CPU at 280-token budget, batch size 1 and 4
- Tracing feasibility report for the 280-token budget
- Optimized memory configs for attention and MLP matmuls

### Phase Gate

> **Gate 3:** End-to-end PCC > 0.998 and TTNN latency is at least 2x faster than CPU at batch size 1 for the 280-token budget. If the speedup is less than 2x, evaluate whether further optimization can close the gap or whether a hybrid approach (CPU for vision, TTNN for language) is more practical.

## Phase 4 — Variable Resolution Support (1 Week)

### Objective

Extend the TTNN vision encoder to handle all five standard token budgets (70, 140, 280, 560, 1120) without manual reconfiguration. Pre-trace or pre-compile programs for each budget to eliminate runtime recompilation.

### Entry Criteria

- Phase 3 deliverables complete (end-to-end working at 280-token budget)

### Tasks

| Task | Description | Effort |
|------|-------------|--------|
| 4.1 | Validate end-to-end PCC at all five token budgets (70, 140, 280, 560, 1120) | 1 day |
| 4.2 | Pre-trace or pre-compile TTNN programs for each of the five budgets. Store in the program cache for runtime reuse | 1-2 days |
| 4.3 | Implement budget selection logic: given an input image, select the nearest supported budget and pad/truncate the patch sequence accordingly | 1 day |
| 4.4 | Validate batch inference with mixed budgets (all images in a batch padded to the same budget) | 1 day |
| 4.5 | Final performance characterization: latency at each budget, memory utilization, throughput at batch sizes 1-8 | 1 day |

### Budget-to-Shape Mapping

Each token budget maps to a fixed patch count after pooling. The pre-pooling patch count depends on the aspect ratio but is constrained:

| Token Budget | Post-Pooling Tokens | Pre-Pooling Patches (approx.) | TTNN Program Key |
|-------------|--------------------|-----------------------------|-----------------|
| 70 | 70 | ~630 | `program_budget_70` |
| 140 | 140 | ~1260 | `program_budget_140` |
| 280 | 280 | ~2520 | `program_budget_280` |
| 560 | 560 | ~5040 | `program_budget_560` |
| 1120 | 1120 | ~10080 | `program_budget_1120` |

> **Warning:** The pre-pooling patch count varies slightly across aspect ratios even within the same token budget (e.g., 280 tokens from a 42x60 grid = 2520 patches vs. a 45x56 grid = 2520 patches). Ensure that the program cache key accounts for the exact patch grid dimensions, not just the budget number.

### Runtime Flow

```
1. Receive image(s)
2. Compute target dimensions and token budget (host-side)
3. Pad all images in the batch to the same budget's patch count
4. Select the pre-traced/pre-compiled TTNN program for that budget
5. Execute the vision encoder on device
6. Return soft tokens to the language model
```

### Deliverables

- TTNN vision encoder validated at all five token budgets
- Pre-traced programs for each budget (if tracing is feasible) or pre-compiled program cache entries
- Budget selection and padding utility in `gemma4_variable_resolution.py`
- Final performance characterization report

### Phase Gate

> **Final gate:** The TTNN vision encoder passes PCC > 0.998 at all five token budgets and handles batch inference with budget-padded inputs. Performance meets the target latency established in Phase 3.

## Timeline Summary

```
Week 1:     Phase 1 — CPU reference and profiling
Weeks 2-4:  Phase 2 — Module-level TTNN port
Weeks 5-6:  Phase 3 — End-to-end integration and optimization
Week 7:     Phase 4 — Variable resolution support
```

### Critical Path

The critical path through the plan is:

```
Phase 1 (reference) -> Phase 2 (attention + 2D RoPE) -> Phase 3 (integration) -> Phase 4
```

The attention module with 2D RoPE is on the critical path because it is the highest-effort, highest-risk component and blocks the encoder layer assembly in Phase 3. Prioritize this module and allocate the most experienced engineer to it.

### Parallelization Opportunities

- **Within Phase 2:** RMSNorm + MLP can be developed in parallel with 2D RoPE + position embedding (no mutual dependencies).
- **Phase 2 and Phase 4 prep:** The variable-resolution preprocessor (`gemma4_variable_resolution.py`) is host-only and can be developed during Phase 2 by a separate engineer.
- **Phase 3 and documentation:** Performance characterization documentation can be started during Phase 3 optimization work.

---

**Next:** [`risk_register.md`](./risk_register.md) — Risk register with severity ratings and mitigation strategies for the six identified risks.
