# Task List and Priority

This file defines the complete 7-task implementation plan for the `pure_ttnn_deltanet_decode_step` guide. Tasks are ordered by the critical path first (Tasks 1, 2, 5), then by parallel tracks (Tasks 3, 4), then by latency optimization (Task 6) and prefill coverage (Task 7). Each task includes a priority rating, complexity rating, description, prerequisites, and cross-references to prior chapters.

**Priority scale:** Critical (blocks trace compatibility) → High (blocks correctness or important parallel ops) → Medium (latency optimization) → Low (prefill, not required for decode trace)

**Complexity scale:** Low (< 1 day, mostly wiring existing TTNN primitives) → Medium (1–3 days, requires new code or porting) → High (3–7 days, requires kernel work and tuning)

---

## Task 1 — Refactor state tensor storage to on-device TTNN tensors

**Priority:** Critical
**Complexity:** Low

### Description

Change `TTNNQwenPagedAttentionKVCache.recurrent_states` and `.conv_states` from dicts of PyTorch tensors (currently on CPU) to dicts of `ttnn.Tensor` objects pre-allocated on the Wormhole mesh device. Allocation must happen during model setup (the warm-up phase, before any trace capture bracket).

Allocation pattern:

```python
# During model setup — outside trace bracket
s_init = ttnn.zeros(
    shape=[1, H, d_k, d_v],   # [1, 4, 128, 128] per device
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# Or use ttnn.allocate_tensor_on_device for pre-allocated buffer handles
recurrent_states[layer_idx] = s_init
```

Memory configuration: `(DRAM, TILE)`. Tensor shapes: `[1, H, d_k, d_v]` = `[1, 4, 128, 128]` per device per layer, where `H = 4` (heads per T3K device after tensor-parallel sharding), `d_k = d_v = 128`. Total DRAM footprint: 128 KB × 30 layers = 3.75 MB per device — well within the 12 GB DRAM capacity of a single Wormhole device.

### What This Eliminates

Eliminates `ttnn.to_torch(S_prev)` (100–300 µs per layer) and `ttnn.from_torch(S_new)` (100–300 µs per layer). This alone removes approximately 6–18 ms of the 9–21 ms per-step DeltaNet latency.

### Why Critical

Every subsequent task that touches the state tensor (Tasks 2, 5, 6) requires the state to already be an on-device `ttnn.Tensor`. This is the foundational prerequisite for the entire critical path.

### Prerequisites

None. This task can be started immediately.

### Chapter References

- Ch1 `device_state_persistence.md` — explains the current PyTorch-tensor storage pattern and why it requires host crossings
- Ch2 `state_tensor_memory_config.md` — defines the exact shape, layout, and memory config for `S`

---

## Task 2 — Wire decay gate and update rate ops to TTNN

**Priority:** Critical
**Complexity:** Low

### Description

Replace Python-level `torch.exp`, `torch.softplus`, and `torch.sigmoid` calls (used to compute the decay gate `g_t` and update rate `β_t`) with their TTNN equivalents: `ttnn.exp`, `ttnn.softplus`, `ttnn.sigmoid`. These primitives are `[AVAILABLE]` with no gaps.

The input tensors `in_proj_a` and `in_proj_b` (output of the linear projection for gates) must already be on-device `ttnn.Tensor` objects. Verify that these projections remain on-device after Task 1 changes; if they are inadvertently converted to PyTorch tensors anywhere in the gate computation path, trace compatibility will be broken.

Specific ops to replace:

| Python/torch call | TTNN replacement | Status |
|---|---|---|
| `torch.sigmoid(in_proj_a)` | `ttnn.sigmoid(in_proj_a)` | `[AVAILABLE]` |
| `torch.exp(-F.softplus(in_proj_b))` | `ttnn.exp(ttnn.neg(ttnn.softplus(in_proj_b)))` | `[AVAILABLE]` |

After this task, `g_t` and `β_t` are on-device `ttnn.Tensor` objects that can be passed directly into the TTNN recurrence ops in Task 5.

### Prerequisites

Task 1 (state on-device). The code for this task can be written concurrently with Task 1 since it operates on projection outputs rather than the state tensor itself, but validation must be sequential: run Task 2 correctness checks only after Task 1 is confirmed passing, to ensure no host crossings remain in the combined gate + state path.

### Chapter References

- Ch1 `forward_pass_walkthrough.md` (Step 3) — shows the current `torch.exp` / `torch.sigmoid` calls for gate computation
- Ch2 `ttnn_ops_per_step.md` (ops 3–4) — defines the TTNN equivalents and their availability status

---

## Task 3 — Wire causal conv1d update to TTNN

**Priority:** High
**Complexity:** Medium

### Description

Replace the `causal_conv1d_update` C extension (a CUDA kernel not available on Wormhole) with a sequence of TTNN primitives: `ttnn.slice`, `ttnn.concat`, `ttnn.mul`, and `ttnn.sum`. All primitives are `[AVAILABLE]`.

The causal conv1d update at decode time (B=1, one new token) is a shift-and-convolve operation on the conv state buffer. In TTNN form:

```python
# conv_state: [1, H, d_inner, conv_width] — on-device after Task 1
# x_new:      [1, 1, H, d_inner]           — current token embedding

# Shift: drop oldest entry, append new
conv_state_shifted = ttnn.concat([
    ttnn.slice(conv_state, dim=-1, start=1, end=conv_width),  # drop oldest
    ttnn.reshape(x_new, [1, H, d_inner, 1]),                  # append new
], dim=-1)

# Apply conv weights (pointwise multiply + sum over conv_width)
o_conv = ttnn.sum(ttnn.mul(conv_state_shifted, conv_weight), dim=-1)
```

The exact op sequence and shapes are specified in Ch3 `causal_conv1d_update_ttnn.md`. The key complexity is verifying that `ttnn.slice` with a non-zero start index produces correct results on TILE_LAYOUT tensors (may require a reshape to ROW_MAJOR for the slice and back to TILE for the multiply).

### Prerequisites

Task 1 (conv_state must be an on-device `ttnn.Tensor`; currently `.conv_states` holds PyTorch tensors).

### Chapter References

- Ch3 `causal_conv1d_update_ttnn.md` — full op sequence, shapes, and known edge cases for tile layout and slicing

---

## Task 4 — Wire gated RMSNorm to TTNN

**Priority:** High
**Complexity:** Low

### Description

Replace `FusedRMSNormSwishGate` (a fused CUDA kernel not available on Wormhole) with the three-op TTNN sequence: `ttnn.rms_norm`, `ttnn.silu`, `ttnn.mul`. All three primitives are `[AVAILABLE]`.

```python
# x: [1, 1, H, d_inner] — input to gated RMSNorm
# z: [1, 1, H, d_inner] — gate input (parallel branch)
x_normed = ttnn.rms_norm(x, weight=rms_weight, epsilon=1e-5)
gate     = ttnn.silu(z)
output   = ttnn.mul(x_normed, gate)
```

This is a straightforward wiring task. The gated RMSNorm `[PARTIAL_REUSE]` classification from Chapter 5 indicates the individual primitives exist but the fused form has not been ported; the three-op sequence is the correct approach.

### Prerequisites

None beyond general on-device tensor management. Can proceed in parallel with the Task 1 → 2 → 5 critical path.

### Chapter References

- Ch3 `gated_rmsnorm_ttnn.md` — defines the three-op TTNN sequence and its PCC expectation vs. the fused reference

---

## Task 5 — Wire recurrent delta rule step to TTNN

**Priority:** Critical
**Complexity:** Medium

### Description

Replace `recurrent_gated_delta_rule` (the PyTorch CPU kernel at the heart of the host fallback) with the 6-op TTNN sequence from Chapter 2. Use the composed TTNN form (12 dispatches per layer) first. The fused kernel (Task 6) is a subsequent latency optimization.

The 6 mathematical ops and their TTNN forms:

| Step | Math | TTNN ops |
|---|---|---|
| 1. Decay state | `S ← g_t · S` | `ttnn.mul(g_t, S)` |
| 2. Retrieve | `o_t = S k̃_t` | `ttnn.matmul(S, k̃_t)` |
| 3. Error | `e = o_t - ṽ_t` | `ttnn.sub(o_t, v_tilde)` |
| 4. Outer product write | `∆S = β_t · e ⊗ k̃_t` | `ttnn.mul(beta_t, ttnn.outer(e, k_tilde))` |
| 5. Add | `S ← S - ∆S` | `ttnn.sub(S, delta_S)` |
| 6. Output | `o_t = W_o o_t` | `ttnn.linear(o_t, W_o)` |

The in-place state update must write `S_new` back into the persistent pre-allocated DRAM buffer from Task 1. Use `ttnn.copy` or `ttnn.assign` for this:

```python
ttnn.copy(src=S_new, dst=recurrent_states[layer_idx])
# or equivalently:
# ttnn.assign(recurrent_states[layer_idx], S_new)
```

This write is Metal Trace compatible (see `trace_integration_checklist.md` Step 3).

### Prerequisites

Task 1 (state on-device) and Task 2 (decay gates as on-device tensors). Both must be complete before Task 5 can be validated.

### Chapter References

- Ch2 `recurrence_math_and_tensor_ops.md` — full mathematical derivation and TTNN op mapping
- Ch2 `ttnn_ops_per_step.md` — availability status for each of the 12 ops (`[AVAILABLE — needs wiring]` for all)

---

## Task 6 — Port or implement fused `gdn_full_fused_inplace` kernel for Wormhole

**Priority:** Medium
**Complexity:** High

### Description

Port the reference CUDA implementation of `gdn_full_fused_inplace` to a TT-Metalium kernel for Wormhole B0. Per Chapter 4 analysis, this kernel is classified `[REUSABLE_WITH_TUNING]`: the mathematical structure ports cleanly, but Wormhole-specific adaptations are required.

Required tuning items:
1. **Circular buffer (CB) constants** — Wormhole tile dimensions and CB slot sizes differ from CUDA shared memory; recalculate for `d_k = d_v = 128` (4 tiles × 4 tiles in BF16).
2. **FPU tile dimensions** — Wormhole FPU operates on 32×32 tiles; the 128×128 state matrix is 4×4 = 16 tiles; verify that the tile loop order in the kernel body matches the Wormhole FPU pipeline.
3. **`FP32_DEST_ACC` flag** — Enable FP32 destination accumulation for the matmul steps to prevent rounding error in intermediate accumulations before the final BF16 write.
4. **NOC routing** — Replace CUDA shared memory access patterns with NOC tile reads; verify that the state tile read sequence is compatible with the DRAM-to-L1 prefetch pattern.

Latency target: 1 dispatch × 30 layers = 30 dispatches; ~5 µs each = 150 µs dispatch + 27 µs DRAM ≈ **177 µs total** for all 30 DeltaNet layers (Chapter 6 estimate). Measure with Tracy after implementation and update estimates.

### Prerequisites

Task 5 (composed TTNN form) must be complete and passing correctness tests. The fused kernel must produce PCC > 0.999 vs. the PyTorch reference (same test as Task 5). Use the composed TTNN form as a secondary reference during fused kernel validation.

### Chapter References

- Ch4 `wormhole_t3k_adaptation.md` — full portability analysis, required Wormhole adaptations, and `FP32_DEST_ACC` recommendation
- Ch6 `on_device_latency_estimate.md` — latency targets for the fused kernel form

---

## Task 7 — Python chunk loop for prefill (`chunk_gated_delta_rule` in TTNN)

**Priority:** Low
**Complexity:** Medium

### Description

Implement the prefill path for DeltaNet as a Python loop over T/64 chunks, each chunk calling TTNN matmuls for the DeltaNet recurrence in parallel across the chunk dimension. This replaces the `chunk_gated_delta_rule` reference at prefill time.

The prefill path is not required for decode trace compatibility (Metal Traces are decode-only). Expected prefill latency at T=8192: approximately 150 ms for 30 DeltaNet layers — within the expected prefill budget for Qwen3.6-35B-A3B and not a current bottleneck.

Tackle this task after Tasks 1–5 are complete and decode trace compatibility is verified.

### Prerequisites

Task 5 (the per-step TTNN recurrence logic is reused inside the chunk loop).

### Chapter References

- Ch6 `on_device_latency_estimate.md` (Section 3, Prefill Note) — latency estimate for the Python chunk loop

---

## Critical Path Summary

```
Task 1 (state on-device)
    └─> Task 2 (gates on-device)
            └─> Task 5 (recurrent step TTNN)  ← Trace compatibility achieved here
                    └─> Task 6 (fused kernel)  ← Latency optimization

Task 3 (conv1d TTNN)    ─ parallel with Task 1 → 2 → 5
Task 4 (gated RMSNorm)  ─ parallel with Task 1 → 2 → 5

Task 7 (prefill)  ─ deferred until after decode trace is stable
```

Tasks 1 → 2 → 5 are the minimum path to decode trace compatibility. Tasks 3 and 4 must be completed before the entire `TTNNQwen3LinearAttention` forward pass is host-crossing-free, but they do not block the DeltaNet-specific trace integration.
