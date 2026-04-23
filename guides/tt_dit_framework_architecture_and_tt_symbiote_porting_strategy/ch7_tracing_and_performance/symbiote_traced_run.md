# TT-Symbiote's TracedRun: Module-Level Tracing

## Prerequisites

- [Chapter 7 Index](./index.md): TTNN trace primitive semantics and memory constraints.
- [`tt_dit_tracer.md`](./tt_dit_tracer.md): understanding of TT-DiT's pipeline-level tracing for contrast.
- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): TT-Symbiote's dispatch interception model and `TTNNModule` base class.

## Overview

TT-Symbiote's `TracedRun` class (in `core/run_config.py`) provides **automatic, module-level tracing** integrated into the Symbiote run-mode system. Unlike TT-DiT's `Tracer` which wraps an entire pipeline callable, `TracedRun` operates at the granularity of individual `TTNNModule` subclasses, selectively tracing only modules marked with the `@trace_enabled` decorator.

This design reflects Symbiote's architecture: the system manages a heterogeneous graph of modules where some run native TTNN, some fall back to PyTorch, and some may not be traceable. Module-level tracing lets the framework trace what it can while leaving the rest untraced.

## The Run-Mode System Context

`TracedRun` is one of eight run modes in TT-Symbiote's `_RUN_MODE_REGISTRY`:

| Mode | Class | Purpose |
|------|-------|---------|
| `NORMAL` | `NormalRun` | Standard dispatch: try TTNN, fall back to PyTorch |
| `NORMAL_WITH_FALLBACK` | `NormalRunWithFallback` | Like NORMAL but catches TTNN exceptions |
| `SEL` | `SELRun` | Side-by-side execution and comparison |
| `DPL` | `DPLRun` | Dual-path with error propagation |
| `DPL_NO_ERROR_PROP` | `DPLRunNoErrorProp` | Dual-path without error propagation |
| `LIGHTWEIGHT` | `LightweightRun` | CPU-only dispatch (no TTNN) |
| `CPU` | `CPU` | Explicit CPU execution |
| **`TRACED`** | **`TracedRun`** | **Module-level trace capture and replay** |

`TracedRun` extends `LightweightRun` (which itself extends `NormalRun`), inheriting the lightweight `torch_dispatch` that routes all tensor-level ops through PyTorch. The tracing logic lives entirely in the `module_run` static method, which is called when a `TTNNModule.__call__` dispatches through the run mode.

Activation is via environment variable or API:

```python
# Environment variable
export TT_SYMBIOTE_RUN_MODE=TRACED

# Programmatic
from models.experimental.tt_symbiote.core.run_config import set_run_mode, TracedRun
set_run_mode("TRACED")
TracedRun.configure(device=mesh_device, cq_id=0)
```

## Three-Phase Lifecycle

`TracedRun` implements a three-phase lifecycle for each `(module_name, input_signature)` pair. This is more conservative than TT-DiT's two-phase approach because Symbiote modules may have complex initialization behavior that benefits from a dedicated warm-up pass.

### Phase 1: Warm-Up (First Encounter)

```
cache_key = (module_name, input_signature)
cache_key NOT in _warmup_keys AND NOT in _trace_cache

  --> Normal forward execution
  --> _warmup_keys.add(cache_key)
  --> _TRACE_RUNNING = True (prevents nested tracing)
```

The warm-up phase runs the module's `forward()` method normally, without any trace capture. This serves multiple purposes:

- **JIT compilation**: TTNN kernels are compiled on first execution.
- **CCL priming**: Collective operations establish their communication patterns.
- **Memory allocator warm-up**: The device memory allocator settles into a steady state.
- **Module-internal caching**: Some modules cache intermediate results on first call.

The `_TRACE_RUNNING` global flag is set to `True` during warm-up to prevent nested modules from attempting their own trace capture. This is important because `module_run` is called recursively for nested `TTNNModule` hierarchies.

### Phase 2: Capture (Second Encounter)

```
cache_key in _warmup_keys AND NOT in _trace_cache

  --> _capture_trace(module, func_args, func_kwargs, cache_key)
  --> Returns TraceEntry stored in _trace_cache
```

On the second encounter of the same `(module, signature)`, `TracedRun` captures the trace. The `_capture_trace` static method performs:

1. **Input buffer allocation**: For each tensor argument, a persistent DRAM buffer is allocated:

```python
for arg in func_args:
    if isinstance(arg, ttnn.Tensor):
        host_tensor = arg.cpu() if arg.storage_type() != ttnn.StorageType.HOST else arg
        trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
        trace_inputs.append(trace_input)
```

2. **Keyword argument buffer allocation**: Handles both scalar tensors and composite structures like `position_embeddings = [cos, sin]`:

```python
for key, val in func_kwargs.items():
    if isinstance(val, ttnn.Tensor):
        trace_kwargs_map[key] = _alloc_kwarg_tensor(val)
    elif isinstance(val, (list, tuple)):
        # Element-wise allocation for tensor lists
        bufs = [_alloc_kwarg_tensor(elem) if is_tensor(elem) else None
                for elem in val]
```

3. **Internal warm-up forward**: The module's `forward()` is called once more with the original inputs (not the trace buffers) to ensure caches are fully populated:

```python
module.forward(*func_args, **func_kwargs)
ttnn.synchronize_device(device)  # Critical: flush CCL ops
```

4. **Trace capture**: The forward is executed again, this time with the pre-allocated trace buffer inputs:

```python
trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
trace_output = module.forward(*trace_func_args, **trace_func_kwargs)
ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
ttnn.synchronize_device(device)
```

5. **Cache storage**: A `TraceEntry` is stored with all buffer references:

```python
@dataclass(slots=True)
class TraceEntry:
    trace_id: int
    trace_inputs: List[Any]        # Positional arg buffers
    trace_kwargs: Dict[str, Any]   # Keyword arg buffers
    trace_output: Any              # Output buffer
    device: Any                    # Device reference
```

### Phase 3: Replay (Third Encounter Onward)

```
cache_key in _trace_cache

  --> Copy new inputs to trace buffers
  --> Call pre_trace_execute hook (if overridden)
  --> ttnn.execute_trace(blocking=False)
  --> Call post_trace_execute hook (if overridden)
  --> Return trace_output
```

The replay path is the steady-state hot path. Its steps are:

1. **Input copy** (`_copy_inputs_to_trace_buffer`): Iterates over positional arguments, copying each tensor into its corresponding pre-allocated buffer:

```python
for arg in new_args:
    if isinstance(arg, ttnn.Tensor):
        if arg is not trace_input:
            ttnn.copy(arg, trace_input)
```

The `is not` identity check avoids redundant self-copies when the caller happens to pass the same buffer object.

2. **Kwargs copy** (`_copy_kwargs_to_trace_buffer`): Handles keyword arguments including list/tuple structures:

```python
for key, trace_buf in trace_kwargs.items():
    new_val = new_kwargs.get(key)
    if isinstance(trace_buf, (list, tuple)):
        for tb, nv in zip(trace_buf, new_val):
            TracedRun._copy_one_to_trace_buffer(nv, tb)
    else:
        TracedRun._copy_one_to_trace_buffer(new_val, trace_buf)
```

3. **Pre-trace hook**: If the module overrides `pre_trace_execute`, it is called before replay. This hook is only invoked if the method on the module's class is different from the base `TTNNModule.pre_trace_execute`:

```python
if type(self).pre_trace_execute is not TracedRun._base_pre_trace_execute:
    self.pre_trace_execute(func_args, func_kwargs)
```

4. **Trace replay**: Non-blocking execution for maximum throughput:

```python
ttnn.execute_trace(entry.device, entry.trace_id,
                   cq_id=TracedRun._cq_id, blocking=False)
```

5. **Post-trace hook**: Similarly, `post_trace_execute` is called if overridden by the module class.

## The `@trace_enabled` and `@trace_disabled` Decorators

Module-level tracing is opt-in via class decorators:

```python
_TRACE_ENABLED_CLASSES: Set[Type] = set()
_TRACE_DISABLED_CLASSES: Set[Type] = set()

def trace_enabled(cls):
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls

def trace_disabled(cls):
    _TRACE_DISABLED_CLASSES.add(cls)
    return cls
```

A module is traceable if and only if:

```python
def is_trace_enabled(module):
    return (isinstance(module, tuple(_TRACE_ENABLED_CLASSES))
            and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES)))
```

The `isinstance` check means that subclasses of a `@trace_enabled` class are also trace-enabled unless explicitly marked `@trace_disabled`. This provides fine-grained control:

```python
@trace_enabled
class TransformerBlock(TTNNModule):
    ...  # All transformer blocks are traced

@trace_disabled
class SpecialBlock(TransformerBlock):
    ...  # This specific subclass is not traced
```

When `TracedRun.module_run` encounters a module that is not trace-enabled, or when `_TRACE_RUNNING` is already `True` (indicating a parent module's trace is being captured), it falls back to normal forward execution.

## The `TTNNLayerStack` Class

**Source:** `core/module.py`

`TTNNLayerStack` is a `@trace_enabled` container that wraps a sequence of `TTNNModule` layers into a single traceable unit:

```python
@trace_enabled
class TTNNLayerStack(TTNNModule):
    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, **kwargs)
        return hidden_states
```

**Critical design choice:** `TTNNLayerStack` calls `layer.forward()` directly, not `layer()` (which would go through `__call__` -> `module_run`). This bypass is intentional:

- It avoids per-layer trace management overhead -- the entire stack is traced as one unit.
- It avoids per-layer input transformation (the `compose_transforms` chain in `module_run`).
- It avoids per-layer weight preprocessing and device movement checks (weights must be pre-loaded).

The result is that a model with $N$ transformer layers gets **one trace** for the entire layer stack rather than $N$ individual traces. For a 24-layer DiT model, this means:

- **1 trace capture** instead of 24
- **1 `execute_trace` call** per denoising step instead of 24
- **1 input copy** (the initial hidden states + shared kwargs) instead of 24 sets of copies

This is the closest Symbiote comes to TT-DiT's pipeline-level tracing approach.

## Cache Key and Signature System

`TracedRun` uses a composite cache key based on the module name and input tensor signatures:

```python
@staticmethod
def _make_cache_key(module_name, args):
    return (module_name, _compute_args_signature(args))
```

The signature captures the structural properties of tensor inputs:

```python
def _compute_tensor_signature(tensor):
    if isinstance(tensor, ttnn.Tensor):
        return (tuple(tensor.shape), tensor.dtype, tensor.layout)
    if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
        t = tensor.ttnn_tensor
        return (tuple(t.shape), t.dtype, t.layout)
    if isinstance(tensor, torch.Tensor):
        return (tuple(tensor.shape), tensor.dtype)
    return ()
```

This means the same module will have **different traces** for different input shapes. For DiT models with variable spatial resolution, this allows automatic re-capture when the resolution changes, though it also means each resolution variant consumes device memory for its own trace buffers.

## The `_TRACE_RUNNING` Guard

A global flag prevents trace nesting:

```python
_TRACE_RUNNING = False
```

During warm-up and capture phases, `_TRACE_RUNNING` is set to `True`. When `module_run` encounters this flag, it falls back to normal execution regardless of the module's trace-enabled status:

```python
if not is_trace_enabled(self) or _TRACE_RUNNING:
    # Normal forward execution
    result = self.forward(*func_args, **func_kwargs)
```

This guard is essential for `TTNNLayerStack`: when the stack's trace is being captured, the individual layers inside must execute normally (via `layer.forward()`) rather than attempting their own trace captures.

The `disable_trace` function decorator provides an explicit way to suppress tracing within a function scope:

```python
def disable_trace(fn):
    def new_fn(*args, **kwargs):
        global _TRACE_RUNNING
        was_tracing = _TRACE_RUNNING
        _TRACE_RUNNING = True
        try:
            return fn(*args, **kwargs)
        finally:
            _TRACE_RUNNING = was_tracing
    return new_fn
```

## Pre/Post Trace Execute Hooks

The `TTNNModule` base class defines two hooks for trace replay customization:

```python
class TTNNModule:
    def pre_trace_execute(self, func_args, func_kwargs):
        """Called before ttnn.execute_trace during replay."""

    def post_trace_execute(self, func_args, func_kwargs, result):
        """Called after ttnn.execute_trace during replay."""
```

These hooks enable modules to perform custom buffer management around trace replay. Use cases include:

- **`pre_trace_execute`**: Copying trace-sensitive inputs to module-owned persistent buffers that are not part of the standard arg/kwarg flow (e.g., cached key-value buffers for attention).
- **`post_trace_execute`**: Updating module state based on replay results, performing post-processing that cannot be included in the trace (e.g., host-side metric collection).

`TracedRun` checks at replay time whether the module's class has actually overridden these methods by comparing against the saved base implementations, avoiding the overhead of calling no-op base methods:

```python
if type(self).pre_trace_execute is not TracedRun._base_pre_trace_execute:
    self.pre_trace_execute(func_args, func_kwargs)
```

## Weight Management During Tracing

`TracedRun.module_run` calls `preprocess_weights()` and `move_weights_to_device()` before entering the trace logic. Both methods check `_TRACE_RUNNING` and assert that weights are already processed/on-device:

```python
def preprocess_weights(self):
    if _TRACE_RUNNING:
        assert self._preprocessed_weight, \
            "Weights must be preprocessed before traced execution."
        return
```

This ensures that:
- On the first (warm-up) call, weights are preprocessed and moved to device normally.
- On subsequent calls (capture and replay), the weight processing is skipped (already done) but the assertions verify the invariant holds.

## Cache Management API

`TracedRun` provides class methods for managing the trace cache:

```python
TracedRun.cache_size()           # Number of cached traces
TracedRun.cached_keys()          # List of cache keys
TracedRun.release_all()          # Release all traces
TracedRun.release(module_name)   # Release traces for a specific module
```

`release_all()` iterates through all cached `TraceEntry` objects and calls `ttnn.release_trace` for each, then clears the cache dictionary. `release(module_name)` selectively removes entries where the cache key starts with the given module name.

## Timing and Profiling Integration

`TracedRun` records timing information through `DispatchManager.record_timing` at each phase:

- `_preprocess_weights` -- weight preprocessing duration
- `_move_weights_to_device` -- weight transfer duration
- `_pre_trace_copy` -- input buffer copy duration (replay path)
- `_pre_trace_execute` -- pre-trace hook duration
- `_capture_trace` -- full capture duration (capture path)
- `_forward` -- total forward duration (all paths)

These entries flow into `DispatchManager`'s timing infrastructure, which can export per-operation CSV reports via `save_stats_to_file`.

## Key Takeaways

1. **Three-phase lifecycle (warm-up, capture, replay)** provides an extra warm-up pass compared to TT-DiT's two-phase approach, ensuring JIT and CCL are fully settled before trace capture.

2. **`@trace_enabled`/`@trace_disabled` decorators** give fine-grained, class-level control over which modules participate in tracing, with inheritance-aware semantics.

3. **`TTNNLayerStack` consolidates $N$ layers into a single trace**, eliminating per-layer dispatch overhead and achieving performance characteristics closer to TT-DiT's pipeline-level tracing.

4. **Cache keys based on `(module_name, input_signature)`** enable automatic re-capture when input shapes change, at the cost of increased device memory for multi-resolution workloads.

5. **`pre_trace_execute`/`post_trace_execute` hooks** provide extensibility points for modules with custom buffer management needs (e.g., KV-cache updates, CCL synchronization) without modifying the core trace logic.

---

**Next:** [`integration_strategy.md`](./integration_strategy.md)
