# Metal Trace and Program Cache Considerations

## Overview

This file covers the operational constraints that arise when deploying M-RoPE inside a Metal Trace capture/replay loop and the TTNN program cache behavior at decode and prefill time. The text-only path is unaffected; all considerations below apply only when `use_mrope=True`.

---

## 1. Metal Trace Compatibility

### How Metal Trace works

Metal Trace captures a fixed execution graph during a "compilation" pass (the first call) and replays it for all subsequent calls. The key constraint is that **tensor shapes must be fixed at trace time** — ops that change shape at runtime cannot be inside a trace. Tensor *values* can change freely across replays as long as shapes are constant.

Tensors that vary across decode steps must be passed as runtime device tensor inputs that are updated before each trace replay. Tensors that are fixed throughout generation (e.g., model weights, frequency tables) can be baked into the trace or left as persistent device tensors — either is compatible.

### M-RoPE tensors and their trace compatibility

| Tensor | Shape | Changes across steps? | Trace strategy |
|--------|-------|-----------------------|----------------|
| `position_ids_3d` (decode) | `[3, batch, 1]` | Yes — values advance each step | Pass as device tensor input; update before replay |
| `position_ids_3d` (prefill) | `[3, batch, seq_len]` | Yes — seq_len varies per input | Outside trace (prefill uses its own trace or no trace) |
| `cos_table` | `[max_seq_len, rotary_dim]` | No | Bake into trace or persistent device tensor |
| `sin_table` | `[max_seq_len, rotary_dim]` | No | Bake into trace or persistent device tensor |

### Position ID tensor management at decode time

At decode time, the 3D position ID tensor has shape `[3, batch, 1]`. This is the direct M-RoPE analogue of the existing `cur_pos_tensor` pattern used for 1D position IDs.

```python
# Initialization (before trace capture)
cur_pos_tensor_mrope = ttnn.from_torch(
    torch.zeros(3, batch_size, 1, dtype=torch.int32),
    device=device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
)

# Before each decode step (outside the trace)
next_positions = compute_next_positions_3d(step_idx, ...)  # [3, batch, 1] on CPU
ttnn.copy_host_to_device_tensor(
    ttnn.from_torch(next_positions, dtype=ttnn.int32),
    cur_pos_tensor_mrope,
)

# Inside trace capture / replay
q_rot, k_rot = rope_module.forward(q, k, cur_pos_tensor_mrope)
```

The shape `[3, batch, 1]` is constant across all decode steps, so the trace sees a fixed-shape input. Only the values change.

> **[SILENT FAILURE]** If the position ID tensor is constructed fresh on the host and re-uploaded each step (rather than updating the existing device tensor in-place), the device tensor handle changes and the trace reference becomes stale. Always update the pre-allocated device tensor in-place using `ttnn.copy_host_to_device_tensor` or the equivalent Metal buffer update API.

### Branch selection outside the trace

If the model supports both text-only and vision batches in the same deployment, the `has_vision_tokens` gate (Step 3 in `integration_steps.md`) must be resolved *outside* the trace. Metal Trace captures a single fixed path; it cannot contain dynamic Python-level branches.

```text
Option A (two separate traces):
  trace_text  = compile_trace(use_mrope=False)
  trace_vision = compile_trace(use_mrope=True)
  # At runtime:
  if has_vision_tokens:
      run_trace(trace_vision, cur_pos_tensor_mrope_3d)
  else:
      run_trace(trace_text, cur_pos_tensor_1d)

Option B (always-on M-RoPE, text uses equal position IDs):
  trace = compile_trace(use_mrope=True)
  # At runtime, always pass 3D position IDs
  # For text-only, all three axes are identical (t == h == w == sequential)
  # Ch3 proves this is numerically identical to the text-only path
```

Option B is simpler operationally (one trace instead of two) and is valid because Ch3's mathematical equivalence proof guarantees no output difference for text-only batches. The slight overhead is the 5 additional M-RoPE kernel dispatches at every decode step — which Ch5 quantified as ~25–50 µs/step (< 0.02% of a 250 ms decode step on P150).

---

## 2. Program Cache Behavior at Decode Time

### Cache key composition

The TTNN program cache key includes: op type, input tensor shapes, output tensor shape, data types, and device configuration. It does NOT include tensor values. For decode with a fixed `batch` size and `seq_len=1`, all M-RoPE op shapes are constant across every decode step.

### Ops added by M-RoPE at decode time

| Op | Input shapes | Output shape | Cache key changes across steps? |
|----|-------------|-------------|--------------------------------|
| `ttnn.embedding` (temporal) | `[batch*1]`, `[max_seq_len, s_t]` | `[batch, 1, s_t]` | No |
| `ttnn.embedding` (height) | `[batch*1]`, `[max_seq_len, s_h]` | `[batch, 1, s_h]` | No |
| `ttnn.embedding` (width) | `[batch*1]`, `[max_seq_len, s_w]` | `[batch, 1, s_w]` | No |
| `ttnn.concat` (cos assembly) | `[batch,1,s_t]`, `[batch,1,s_h]`, `[batch,1,s_w]` | `[batch, 1, rotary_dim]` | No |
| `ttnn.concat` (sin assembly) | same | `[batch, 1, rotary_dim]` | No |

**Program cache hit rate at decode: 100%.** Shapes are fixed; only tensor values change, which are not part of the cache key. There is no cache warmup penalty after the first decode step.

### Implication for the ~25–50 µs overhead figure (Ch5)

The Ch5 figure of ~25–50 µs is the **fixed per-step dispatch overhead** for 5 additional kernel dispatches. Because the program cache hits 100% at decode time, this cost is fully amortized — there is no per-step compilation cost, only the constant overhead of looking up the cached program and scheduling it. This confirms that the Ch5 assessment of M-RoPE overhead as negligible is accurate for steady-state decode.

---

## 3. Program Cache Behavior at Prefill

### Cache miss on new sequence lengths

At prefill, sequence length varies per input. Each unique `seq_len` value produces a program cache miss because the input tensor shapes change — the TTNN embedding lookup inputs grow from `[batch*1]` to `[batch*seq_len]`.

This behavior is expected and is identical to what already occurs for attention and FFN kernels at prefill. The M-RoPE embedding lookups add **3 additional cache-miss events** per new `seq_len`, each requiring one kernel compilation.

### Mitigation: sequence length bucketing

Bucket input sequence lengths to a fixed set of values and pad to the nearest bucket. The same strategy is already applied to attention kernels to limit the number of distinct compiled programs.

```python
PREFILL_SEQ_LEN_BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]

def get_padded_seq_len(actual_seq_len: int) -> int:
    """Round up to the nearest prefill bucket."""
    for bucket in PREFILL_SEQ_LEN_BUCKETS:
        if actual_seq_len <= bucket:
            return bucket
    return actual_seq_len  # Beyond largest bucket; compile a new program

def pad_position_ids_to_bucket(position_ids_3d, padded_len):
    """Pad [3, batch, seq_len] → [3, batch, padded_len] with trailing zeros."""
    pad_len = padded_len - position_ids_3d.shape[2]
    if pad_len == 0:
        return position_ids_3d
    padding = torch.zeros(3, position_ids_3d.shape[1], pad_len, dtype=torch.int32)
    return torch.cat([position_ids_3d, padding], dim=2)
```

Padding position IDs with zeros is safe: the M-RoPE gather for a position ID of 0 returns the first row of the cos/sin table, which is a valid (non-NaN, non-Inf) value. The padded output positions are masked out by the attention mask before they affect the model output.

### Expected program cache size at steady state

With 7 prefill buckets and `{cos, sin}` × 3 ops = 6 embedding kernels per seq_len, steady-state prefill adds at most 42 compiled programs to the cache. This is well within the TTNN program cache capacity.

---

## 4. Position ID Tensor Shape Contract

Document the shape contract explicitly to prevent integration errors across the model, attention, and RoPE layers.

| Scenario | `position_ids` shape | dtype | Notes |
|----------|---------------------|-------|-------|
| Decode step (any batch content) | `[3, batch, 1]` | int32 | Updated in-place before each trace replay |
| Prefill — text-only | `[3, batch, seq_len]` | int32 | All three axes identical and equal to `[0, 1, ..., seq_len-1]` |
| Prefill — mixed text+image | `[3, batch, seq_len]` | int32 | Axes differ for vision token positions (see Ch2) |
| Prefill — video | `[3, batch, seq_len]` | int32 | Temporal axis increments per frame; height/width repeat |

> **[SILENT FAILURE]** Passing `[batch, seq_len]` (2D) instead of `[3, batch, seq_len]` (3D) to the M-RoPE forward will not raise a shape error if `batch == 3` (e.g., a 3-item decode batch). The gather will silently use the wrong axis as position IDs. Always validate `position_ids.shape[0] == 3` at the top of `_mrope_forward`.

```python
def _mrope_forward(self, q, k, position_ids_3d):
    assert position_ids_3d.shape[0] == 3, (
        f"M-RoPE position_ids must have shape [3, batch, seq_len]; "
        f"got {list(position_ids_3d.shape)}"
    )
    # ... rest of implementation
```

---

## 5. Backward Compatibility with the Text-Only Path

When `use_mrope=False` (existing text-only path), the `_standard_forward` method is called with a `[batch, seq_len]` position tensor. There are no changes to the existing shape contract, no added branches in `_standard_forward`, and no change to the cos/sin table computation.

The M-RoPE path (`_mrope_forward`) is only invoked when:
1. `use_mrope=True` (set at construction time, from the model config), AND
2. Vision tokens are detected in the batch (the `has_vision_tokens` gate in the attention module)

For a text-only deployment of Qwen3.6-35B-A3B, `use_mrope` is left `False` and none of the M-RoPE code paths execute. The entire implementation is additive.

> **Key Finding:** M-RoPE is fully trace-compatible when position IDs are passed as device tensors. The program cache achieves 100% hit rate at decode time (fixed seq_len=1 shape). The only performance consideration is prefill cache misses for new sequence lengths — the same challenge already present for all variable-length ops in the model.

---

## Summary Table

| Consideration | Text-only path | M-RoPE path (vision) |
|---------------|----------------|----------------------|
| Trace compatibility | Unchanged | Compatible when pos IDs passed as device tensors |
| Decode program cache | 100% hit rate (existing) | 100% hit rate (fixed shapes) |
| Prefill program cache | Miss on new seq_len (existing) | +3 additional misses per new seq_len |
| Decode overhead | Baseline | +~25–50 µs/step (< 0.02% of step) |
| Position tensor shape | `[batch, seq_len]` | `[3, batch, seq_len]` |
| Code changes required | None | Additive only (no existing path modified) |

---

## References

- `../ch3_text_only_reduction/mathematical_equivalence_proof.md`
- `../ch3_text_only_reduction/practical_implications_for_text_inference.md`
- `../ch4_ttnn_implementation/gather_operation_on_ttnn.md`
- `../ch4_ttnn_implementation/extension_approach.md`
- `../ch5_performance_analysis/kernel_launch_overhead.md`
- `../ch5_performance_analysis/prefill_vs_decode_comparison.md`
- `../ch5_performance_analysis/operation_cost_breakdown.md`
- `integration_steps.md` (Steps 2–3, class extension and attention module changes)
