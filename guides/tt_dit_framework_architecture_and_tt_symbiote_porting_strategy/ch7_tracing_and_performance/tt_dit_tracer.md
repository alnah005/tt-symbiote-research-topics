# TT-DiT's Tracer Class and Pipeline Tracing

## Prerequisites

- [Chapter 7 Index](./index.md): TTNN trace primitive semantics and constraints.
- [Chapter 5 -- Pipelines and Serving](../ch5_pipelines_and_serving/index.md): pipeline class structure, denoising loop flow, submesh management.

## Overview

TT-DiT provides two complementary tracing mechanisms:

1. **`Tracer`** (in `tt_dit/utils/tracing.py`) -- a general-purpose wrapper that captures any callable as a TTNN trace. Used in model-level tests to trace an entire transformer forward pass.
2. **`PipelineTrace`** (a `@dataclass` defined per-pipeline) -- a lightweight struct that holds the trace ID alongside all input/output tensor buffer references for a traced denoising step. Used in production pipelines.

Both approaches operate at the **pipeline level**: the trace captures the entire forward pass through the DiT transformer, including all attention blocks, feed-forward layers, normalization, and CCL collectives.

## The `Tracer` Class

**Source:** `models/tt_dit/utils/tracing.py`

### Class Structure

```python
class Tracer:
    def __init__(self, function, /, *, device):
        self._function = function      # Callable to trace
        self._device = device          # MeshDevice
        self._args = ()                # Captured positional args
        self._kwargs = {}              # Captured keyword args
        self._outputs = None           # Trace output buffers
        self._trace_id = None          # MeshTraceId (None until captured)
```

The `Tracer` wraps a single callable (typically `model.forward`) and a target device. It uses `self._trace_id` as a state flag: `None` means no trace has been captured yet.

### Two-Phase First Call

When `__call__` is invoked for the first time (`self._trace_id is None`), two sequential executions occur:

```
First Call
  |
  v
Phase 1: Compile
  self._function(*self._args, **self._kwargs)
  |
  v
Phase 2: Capture
  trace_id = ttnn.begin_trace_capture(device, cq_id)
  outputs = self._function(*self._args, **self._kwargs)
  ttnn.end_trace_capture(device, trace_id, cq_id)
```

**Phase 1 (Compile)** runs the function normally. This triggers TTNN kernel compilation and JIT warmup. The output is discarded -- this call exists solely to ensure that all kernels are compiled before trace capture begins, since compilation involves host-side memory allocation that would violate trace capture constraints.

**Phase 2 (Capture)** runs the function again inside a `begin_trace_capture`/`end_trace_capture` bracket. The device records every operation. The output tensors from this execution become the persistent output buffers that `execute_trace` will overwrite on every subsequent call.

Error handling is careful: if the capture-phase forward raises an exception, the trace is released before re-raising:

```python
try:
    try:
        outputs = self._function(*self._args, **self._kwargs)
    finally:
        ttnn.end_trace_capture(self._device, trace_id, cq_id=tracer_cq_id)
    outputs = _tree_map(_verify_value, outputs, path_label="outputs")
except Exception:
    ttnn.release_trace(self._device, trace_id)
    raise
```

### Subsequent Calls: Input Update and Replay

On every call after the first, the `Tracer` updates input buffers and replays:

```python
# Update positional args
_tree_map(self._update_input, self._args, args, path_label="args")

# Update keyword args
for name, new in kwargs.items():
    prev = self._kwargs[name]
    _tree_map(self._update_input, prev, new, path_label=f'kwargs["{name}"]')

# Replay
ttnn.execute_trace(self._device, self._trace_id, cq_id, blocking)
return self._outputs  # Same buffer objects, new data
```

The return value is always `self._outputs` -- the same Python objects created during capture. The device has overwritten their underlying DRAM buffers with the new computation results.

### The `_update_input` Method

This is where input tensors are copied into the pre-allocated trace buffers:

```python
def _update_input(self, prev, new, *, path_label):
    if isinstance(new, ttnn.Tensor):
        if new.device() is None:
            ttnn.copy_host_to_device_tensor(new, prev)  # Host -> Device
        else:
            ttnn.copy(new, prev)  # Device -> Device
    elif new != prev:
        raise ValueError(...)  # Non-tensor values must be identical
```

Key constraints enforced:
- Tensor shape, dtype, and layout must match between `new` and `prev`.
- If `new` is a host tensor, `copy_host_to_device_tensor` is used.
- If `new` is already on-device, `ttnn.copy` performs a device-to-device copy.
- Non-tensor values (int, float, str, bool, None) must be **identical** to their initial values -- they cannot change between trace replays.

This last constraint is critical for DiT inference: it means that any scalar parameter that changes between denoising steps (like the timestep value) **must be passed as a tensor**, not as a Python scalar.

### The `_tree_map` Utility

Both `Tracer` and `_update_input` rely on `_tree_map`, a recursive structure traversal function:

```python
def _tree_map(f, x, /, *xs, path_label):
    # Handles: tuple, list, dict -> recurse
    # Leaf values -> apply f(x, *xs, path_label=...)
```

`_tree_map` traverses nested combinations of tuples, lists, and dicts, applying the mapping function `f` to every leaf value. It enforces structural consistency: all input structures must have matching types, lengths, and (for dicts) keys at every level. The `path_label` parameter threads a human-readable path string through the recursion for error messages (e.g., `args[0]["hidden_states"]`).

Supported leaf types (enforced by `_verify_value`):
- `ttnn.Tensor`
- `int`, `float`, `str`, `bool`, `NoneType`

Any other type raises `TypeError`. This strict type checking prevents accidental inclusion of PyTorch tensors or other objects that the trace system cannot handle.

### The `release` Method

```python
def release(self):
    if self._trace_id is not None:
        self._trace_id = None
        self._args = ()
        self._kwargs = {}
        self._outputs = None
        ttnn.release_trace(self._device, trace_id)
```

Releases the device-side trace buffer and clears all Python references. After `release()`, calling the `Tracer` again will trigger a fresh two-phase capture.

## `PipelineTrace` in Production Pipelines

While `Tracer` is a general-purpose wrapper, the production pipelines use a more manual but more efficient approach via `PipelineTrace` dataclasses.

### The Dataclass Pattern

Each pipeline defines its own `PipelineTrace` with fields matching the specific model's inputs and outputs. For example, Flux1:

```python
@dataclass
class PipelineTrace:
    tid: int                          # Trace ID
    spatial_input: ttnn.Tensor        # Noisy latents
    prompt_input: ttnn.Tensor         # Text embeddings
    pooled_input: ttnn.Tensor         # Pooled text embeddings
    timestep_input: ttnn.Tensor       # Timestep scalar (as tensor)
    guidance_input: ttnn.Tensor       # Guidance scale (as tensor)
    spatial_rope_cos: ttnn.Tensor     # Positional encoding
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor  # Scheduler step size
    latents_output: ttnn.Tensor       # Denoised output
```

Motif and SD3.5 have similar but slightly different field sets (Motif omits `guidance_input` and the ROPE fields). Each pipeline defines exactly the tensors it needs -- no generic tree traversal is required.

### Capture Flow in Pipelines

The pipeline `_step` method handles trace capture inline rather than delegating to a `Tracer` wrapper. The Flux1 pattern is representative:

```python
def _step(self, ..., traced: bool):
    if traced and self._traces is None:
        # FIRST CALL: Capture
        self._traces = []
        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
            pred = self._step_inner(...)  # Full transformer forward
            ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

            # Synchronize all submeshes
            for device in self._submesh_devices:
                ttnn.synchronize_device(device)

            self._traces.append(PipelineTrace(
                spatial_input=latents[submesh_id],
                ...,
                latents_output=pred,
                tid=trace_id,
            ))
```

**Key differences from `Tracer`:**

1. **No separate compile phase.** The pipeline calls `run_single_prompt(traced=False)` during `__init__` for warmup, so by the time `traced=True` is used, kernels are already compiled. The capture call is the first traced invocation.

2. **Per-submesh traces.** When running with sequence or tensor parallelism, each submesh gets its own trace. The pipeline stores a list `self._traces` indexed by submesh ID.

3. **Explicit synchronization.** After capturing each submesh's trace, all submeshes are synchronized. This is essential because CCL collectives from the capture run may still be in-flight on other submeshes.

### Replay Flow

```python
if traced:
    for submesh_id, submesh_device in enumerate(self._submesh_devices):
        # Update changing inputs
        ttnn.copy_host_to_device_tensor(timestep[submesh_id],
                                         self._traces[submesh_id].timestep_input)
        ttnn.copy_host_to_device_tensor(sigma_difference[submesh_id],
                                         self._traces[submesh_id].sigma_difference_input)

        # Replay
        ttnn.execute_trace(submesh_device,
                          self._traces[submesh_id].tid,
                          cq_id=0, blocking=False)

        noise_pred_list.append(self._traces[submesh_id].latents_output)
```

Only the timestep and sigma difference tensors are copied on each step -- all other inputs (latents, prompt embeddings, ROPE, guidance) were either pre-copied at the start of generation or are updated by the pipeline's outer loop via `copy_host_to_device_tensor` calls before entering the denoising loop.

The `blocking=False` parameter enables asynchronous execution: the host issues the trace replay and immediately proceeds to queue the next submesh's replay, overlapping host-side tensor copies with device-side computation.

### Input Management: Traced vs. Untraced Paths

The pipeline code reveals a subtle pattern in how tensors are created differently based on the `traced` flag:

```python
tt_timestep = ttnn.full([1, 1], fill_value=t, ...,
                         device=submesh_device if not traced else None)
```

When `traced=False`, tensors are created directly on-device. When `traced=True`, tensors are created on the **host** (device=None) and later copied into the pre-allocated trace buffers via `copy_host_to_device_tensor`. This is because the trace buffers were allocated during capture and must be reused -- creating new on-device tensors would not write into the correct buffer locations.

## `Tracer` vs. `PipelineTrace`: When to Use Which

| Aspect | `Tracer` | `PipelineTrace` |
|--------|----------|-----------------|
| **Scope** | Any callable | Pipeline denoising step |
| **Usage** | Tests, prototyping | Production pipelines |
| **Compile phase** | Automatic (built-in) | Separate warmup call |
| **Input update** | Generic `_tree_map` traversal | Explicit per-field `copy_host_to_device_tensor` |
| **Submesh handling** | Single device | Per-submesh trace list |
| **Type safety** | Runtime `_verify_value` checks | Dataclass field types |
| **Overhead** | Slightly higher (tree traversal) | Minimal (direct field access) |

The `Tracer` class is well-suited for testing individual models (as in `test_transformer_qwenimage.py` where `Tracer(tt_model.forward, device=mesh_device)` traces the transformer) and for quick experiments. The `PipelineTrace` pattern is preferred in production because it avoids generic traversal overhead and makes the buffer lifecycle explicit.

## Key Takeaways

1. **`Tracer` implements automatic two-phase capture** -- compile then capture -- shielding callers from the requirement that kernels must be compiled before trace capture begins.

2. **`_tree_map` enables generic nested-structure handling** for inputs and outputs, supporting arbitrary combinations of tuples, lists, and dicts, with strict structural and type validation.

3. **`PipelineTrace` dataclasses provide pipeline-specific, zero-overhead trace management** by naming every input/output buffer explicitly and avoiding generic traversal.

4. **Non-tensor inputs are immutable** in the `Tracer` -- only `ttnn.Tensor` values can change between trace replays. Scalars that vary per step (timestep, guidance) must be tensorized.

5. **Per-submesh traces with explicit synchronization** are required for multi-device execution, ensuring CCL collectives complete before trace capture begins on any submesh.

---

**Next:** [`symbiote_traced_run.md`](./symbiote_traced_run.md)
