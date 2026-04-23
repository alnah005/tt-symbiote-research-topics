# Mapping TT-DiT Pipelines to TT-Symbiote's Serving Infrastructure

## Prerequisites

- [Chapter 5 index](./index.md): pipeline overview and lifecycle
- [`pipeline_anatomy.md`](./pipeline_anatomy.md): SD3.5 pipeline internals, tracing, and memory management
- Familiarity with TT-Symbiote's module replacement pattern (`utils/module_replacement.py`) and run modes (`core/run_config.py`)

This document examines TT-Symbiote's module replacement and dispatch architecture, identifies where a DiT pipeline does and does not fit that architecture, and proposes concrete integration strategies with trade-offs.

---

## 1. TT-Symbiote's Module Replacement Pattern

TT-Symbiote's fundamental approach to accelerating PyTorch models is **recursive module replacement**: walk the `nn.Module` tree, swap each recognized PyTorch module with a TTNN-backed equivalent, and let PyTorch's own `__call__` mechanism route data through the replaced subgraph.

### `register_module_replacement_dict()`

The entry point is in `utils/module_replacement.py`:

```python
def register_module_replacement_dict(
    model,
    old_class_to_new_class_dict,
    model_config=None,
    exclude_replacement=None,
) -> Dict[str, TTNNModule]:
    module_names = {module: name for name, module in model.named_modules()}
    result = {}
    register_module_replacement_dict_with_module_names(
        model, old_class_to_new_class_dict, model_config, module_names, exclude_replacement, result
    )
    return result
```

The caller provides a mapping from PyTorch classes to TTNN classes:

```python
old_class_to_new_class_dict = {
    torch.nn.Linear: TTNNLinear,
    torch.nn.LayerNorm: TTNNLayerNorm,
    SomeCustomAttention: TTNNSomeCustomAttention,
}
```

The replacement logic walks the module tree recursively and, for each matching module:

1. Calls `TTNNClass.from_torch(old_module)` to create the replacement.
2. Sets `_unique_name` and calls `override_children_module_names()` for hierarchical naming.
3. Calls `set_model_config(model_config)` to propagate configuration.
4. Replaces the child in the parent's `_modules` dict.

The recursion also traverses dict and list attributes of modules (not just `_modules`), ensuring that non-standard module containers (e.g., a dict of attention layers stored as a plain attribute) are also caught.

### Key Design Assumption

The module replacement pattern assumes:

- **The model is a standard `nn.Module` tree.** Each component is a child of the root model, reachable via `model.named_modules()`.
- **Replacement is leaf-level or subtree-level.** You swap a `Linear` or an `AttentionBlock`, and the parent's `forward()` calls it normally.
- **The replaced module has the same I/O contract** as the original: same signature, same tensor shapes (possibly with dtype changes handled by the `TorchTTNNTensor` wrapper).

---

## 2. TT-Symbiote's Run Modes and Dispatch Infrastructure

TT-Symbiote provides multiple execution strategies via the `run_config.py` module, selectable at runtime:

| Run Mode | Description |
|---|---|
| `NormalRun` | Dispatches `aten` ops to TTNN when possible, falls back to PyTorch |
| `NormalRunWithFallback` | Same as Normal but catches exceptions and falls back gracefully |
| `LightweightRun` | All ops dispatched to PyTorch (CPU-only, no TTNN) |
| `SELRun` | Side-by-side execution: runs both PyTorch and TTNN, compares outputs |
| `DPLRun` | Dual-path logging: like SEL but propagates TTNN results with torch reference |
| `DPLRunNoErrorProp` | DPL without error propagation between paths |
| `CPU` | Forces all execution to CPU |
| `TracedRun` | Three-phase lifecycle: warmup -> capture -> replay |

### The `TorchTTNNTensor` Dual-Representation

Central to TT-Symbiote's dispatch is the `TorchTTNNTensor` class -- a `torch.Tensor` subclass that maintains both a PyTorch tensor (`elem`) and a TTNN tensor (`ttnn_tensor`). When an `aten` operation is dispatched:

1. `__torch_dispatch__` intercepts the call.
2. `can_dispatch_to_ttnn()` checks if there is a TTNN implementation.
3. If yes, the TTNN path executes; otherwise, the PyTorch path runs.
4. The result is wrapped back into a `TorchTTNNTensor`.

This dual representation enables gradual migration: you can start with all ops on CPU and progressively move them to TTNN.

### `TracedRun`: TT-Symbiote's Tracing Infrastructure

The `TracedRun` mode provides automatic tracing for modules decorated with `@trace_enabled`:

```python
@trace_enabled
class TTNNMyTransformerBlock(TTNNModule):
    ...
```

The lifecycle per (module, input-signature) pair:

1. **Run 1 (Warmup):** Normal forward execution, no trace capture. Primes JIT and allocators.
2. **Run 2 (Capture):** `_capture_trace()` records the op sequence into device memory. Pre-allocates persistent input buffers.
3. **Run 3+ (Replay):** `execute_trace()` replays the captured trace. Inputs are copied to pre-allocated buffers before replay.

```python
if cache_key in TracedRun._trace_cache:
    # REPLAY: copy inputs, execute trace
    entry = TracedRun._trace_cache[cache_key]
    TracedRun._copy_inputs_to_trace_buffer(func_args, entry.trace_inputs)
    TracedRun._copy_kwargs_to_trace_buffer(func_kwargs, entry.trace_kwargs)
    ttnn.execute_trace(entry.device, entry.trace_id, ...)
    result = entry.trace_output
elif cache_key in TracedRun._warmup_keys:
    # CAPTURE
    entry = TracedRun._capture_trace(self, func_args, func_kwargs, cache_key)
    result = entry.trace_output
else:
    # WARMUP
    TracedRun._warmup_keys.add(cache_key)
    result = self.forward(*func_args, **func_kwargs)
```

The cache key is composed of `(module_name, input_tensor_signatures)`, where signatures include shapes, dtypes, and layouts. This means shape-changing inputs (e.g., variable-length prompts) would invalidate the trace cache.

---

## 3. Where TT-DiT Pipelines Diverge from TT-Symbiote's Assumptions

### 3.1. Multi-Component Orchestration vs. Single-Model Replacement

TT-Symbiote's module replacement pattern works best when there is a single `nn.Module` tree whose leaf operations can be individually swapped. A DiT pipeline is fundamentally different:

```
TT-Symbiote model:     root_model.layer1.attn -> swap with TTNNAttn
                        root_model.layer1.ff   -> swap with TTNNFF
                        (PyTorch __call__ routes data through replaced modules)

TT-DiT pipeline:       pipeline.__call__()
                        |-- self._encode_prompts()     [CLIP, T5, tokenizers]
                        |-- torch.randn() + patchify   [host CPU]
                        |-- self._step() * N            [DiT transformer on device]
                        |-- self._vae_decode()          [VAE decoder, possibly different submesh]
                        |-- self._image_processor       [host CPU post-processing]
```

The pipeline's control flow (choosing which encoder to run, managing mesh reshapes, computing scheduler steps) is not captured in any `nn.Module` tree. It is imperative Python code.

### 3.2. Submesh Management

TT-DiT pipelines actively manage submesh topology -- creating submeshes, reshaping them for different components, and assigning CCL managers per submesh. TT-Symbiote's `DistributedConfig` provides a single `mesh_device` with a fixed `tensor_config` (mapper + composer). It does not model the concept of submeshes being created, reshaped, and destroyed within a single inference call.

### 3.3. Manual vs. Automatic Tracing

TT-DiT pipelines use manual `PipelineTrace` management with fine-grained control over exactly which operations are traced. TT-Symbiote's `TracedRun` provides automatic per-module tracing. These two tracing strategies can conflict:

- A TT-DiT pipeline traces the *entire denoising step* (transformer forward) as a single trace.
- TT-Symbiote's `TracedRun` traces *individual modules* (e.g., a single attention block).
- Nesting these (a module-level trace inside a step-level trace) would fail because `_TRACE_RUNNING` prevents nested capture.

### 3.4. Weight Loading and Caching

TT-DiT uses `cache.load_model()` with a filesystem cache keyed by parallelism config. TT-Symbiote loads weights via `TTNNModule.from_torch()` + `preprocess_weights()` + `move_weights_to_device()`, with lazy on-demand preprocessing. The two systems are not interoperable without adaptation.

---

## 4. Integration Strategies

### Strategy A: Pipeline-as-Opaque-Service

**Approach:** Treat the TT-DiT pipeline as a black box. TT-Symbiote wraps the entire pipeline behind a service interface without attempting to replace individual modules.

```python
class DiTPipelineService:
    def __init__(self, pipeline: StableDiffusion3Pipeline):
        self.pipeline = pipeline

    def generate(self, prompt: str, **kwargs) -> List[Image]:
        return self.pipeline.run_single_prompt(prompt, **kwargs)
```

**Advantages:**
- Zero modifications to TT-DiT code.
- All TT-DiT optimizations (manual tracing, submesh management, cache) work as-is.
- Fastest path to deployment.

**Disadvantages:**
- No visibility into per-op dispatch (cannot use SEL/DPL modes for debugging).
- Cannot selectively run parts of the pipeline on different backends.
- No integration with TT-Symbiote's timing infrastructure (`DispatchManager`).

**When to choose:** Production deployment where the TT-DiT pipeline is already validated and performance-tuned.

### Strategy B: Encoder/VAE Replacement, DiT Native

**Approach:** Use TT-Symbiote's module replacement for the text encoders and VAE (which are standard HuggingFace `nn.Module` trees), but keep the DiT transformer as a native TT-DiT module.

```python
# Replace CLIP and T5 with TT-Symbiote modules
replacement_dict = {
    CLIPTextModelWithProjection: TTNNCLIPTextModel,
    T5EncoderModel: TTNNT5Encoder,
}
register_module_replacement_dict(hf_pipeline, replacement_dict)

# Keep DiT transformer as native TT-DiT
tt_transformer = SD35Transformer2DModel(...)
cache.load_model(tt_transformer, ...)

# Custom orchestration
def generate(prompt):
    embeds = hf_pipeline.encode_prompt(prompt)  # TT-Symbiote dispatch
    latents = denoise_loop(tt_transformer, embeds)  # TT-DiT native
    images = hf_pipeline.vae_decode(latents)  # TT-Symbiote dispatch
    return images
```

**Advantages:**
- Leverages TT-Symbiote's dispatch/debugging for encoders and VAE.
- DiT transformer retains TT-DiT's manual tracing and parallel optimizations.
- Encoders benefit from TT-Symbiote's `TracedRun` automatically.

**Disadvantages:**
- Requires a custom orchestration layer to bridge between the two systems.
- Tensor format conversion between `TorchTTNNTensor` and raw `ttnn.Tensor` at boundaries.
- Two different weight-loading systems to manage.

**When to choose:** When you want TT-Symbiote's debugging tools (SEL, DPL) for encoder/VAE development while keeping proven DiT transformer code.

### Strategy C: Full Module Replacement (Deep Integration)

**Approach:** Wrap every TT-DiT component as a `TTNNModule` and integrate it into TT-Symbiote's module tree.

```python
class TTNNDiTTransformer(TTNNModule):
    """Wraps a TT-DiT SD35Transformer2DModel as a TTNNModule."""

    @classmethod
    def from_torch(cls, torch_module):
        instance = cls()
        instance._tt_dit_model = SD35Transformer2DModel(...)
        return instance

    def forward(self, spatial, prompt_embed, pooled_projections, timestep, N, L):
        return self._tt_dit_model(spatial, prompt_embed, pooled_projections, timestep, N, L)

    def preprocess_weights(self):
        if not self._tt_dit_model.is_loaded():
            cache.load_model(self._tt_dit_model, ...)

    def move_weights_to_device(self):
        pass  # handled by TT-DiT's own loading
```

**Advantages:**
- Full integration with TT-Symbiote dispatch infrastructure.
- Can use `TracedRun`, `SELRun`, `DPLRun` for the transformer too.
- Unified timing statistics via `DispatchManager`.

**Disadvantages:**
- Significant adaptation work for submesh management (TT-Symbiote has no submesh concept).
- TT-Symbiote's `TracedRun` and TT-DiT's manual `PipelineTrace` would need to be reconciled.
- The `TorchTTNNTensor` wrapper adds overhead for tensor conversions at every boundary.
- Mesh reshape hacks in SD3.5 would need to be surfaced through TT-Symbiote's `DistributedConfig`.

**When to choose:** Long-term unification goal when you want a single framework for all model types.

---

## 5. Bridging the Tracing Gap

The biggest technical challenge in integration is reconciling TT-DiT's pipeline-level tracing with TT-Symbiote's module-level tracing.

### Option 1: Disable TT-Symbiote Tracing, Use TT-DiT Tracing

When using Strategy A or B, simply do not enable `TracedRun` for DiT pipeline components. The TT-DiT pipeline's own `PipelineTrace` mechanism handles the denoising loop, and encoders/VAE run in `NormalRun` mode.

### Option 2: Unify on TT-Symbiote's `TracedRun`

Decorate the TT-DiT transformer wrapper with `@trace_enabled` and let `TracedRun` handle the three-phase lifecycle. This works if:

- The transformer's input shapes are consistent across denoising steps (they are for fixed-resolution generation).
- The `_TRACE_RUNNING` guard prevents nested traces (handled automatically).
- The manual `PipelineTrace` is removed in favor of `TracedRun`'s `TraceEntry`.

### Option 3: Adopt the `Tracer` Utility

TT-DiT's own `utils/tracing.py` provides the `Tracer` class, which is a clean, tree-map-based abstraction over `begin_trace_capture` / `end_trace_capture`. This could serve as a shared primitive:

```python
# In a unified pipeline:
denoising_tracer = Tracer(transformer.forward, device=submesh_device)

for t in timesteps:
    output = denoising_tracer(latents, prompt_embeds, ..., timestep=t)
    # Tracer handles first-call capture and subsequent replay
```

This avoids the `TorchTTNNTensor` dispatch layer entirely and works with raw `ttnn.Tensor` objects.

---

## 6. Tensor Representation at Boundaries

When bridging the two systems, tensor format conversion is required:

| TT-DiT representation | TT-Symbiote representation | Conversion |
|---|---|---|
| `ttnn.Tensor` on device | `TorchTTNNTensor` wrapping `ttnn.Tensor` | `TorchTTNNTensor(tt_tensor)` |
| `torch.Tensor` on CPU | `TorchTTNNTensor` wrapping `torch.Tensor` | `TorchTTNNTensor(torch_tensor)` |
| `ttnn.Tensor` on host | `TorchTTNNTensor` with `.to_ttnn` | Convert to torch first, then wrap |

The `run_config.py` module provides several helper transforms for this:

- `wrap_to_torch_ttnn_tensor(e)` -- wraps a bare `torch.Tensor` or `ttnn.Tensor` into `TorchTTNNTensor`
- `fast_unwrap_to_device(device)` -- extracts `ttnn.Tensor` from `TorchTTNNTensor` and ensures device placement
- `compose_transforms(...)` -- chains multiple transforms into a single pass

For Strategy B, the boundary between encoder output (TT-Symbiote) and denoising input (TT-DiT) would need:

```python
# After TT-Symbiote encoder produces TorchTTNNTensor:
prompt_embeds_ttnn = prompt_embeds.to_ttnn  # Extract raw ttnn.Tensor
# Feed to TT-DiT transformer which expects raw ttnn.Tensor
```

---

## 7. Memory Management Reconciliation

TT-DiT's `unload_set` and TT-Symbiote's weight management are separate systems:

| Feature | TT-DiT | TT-Symbiote |
|---|---|---|
| Weight loading | `cache.load_model()` + filesystem cache | `from_torch()` + `preprocess_weights()` + `move_weights_to_device()` |
| Lazy loading | Yes, via `is_loaded()` check | Yes, via `preprocess_weights()` idempotence |
| Memory reclaim | `unload_set` + `deallocate_weights()` | Not built-in (modules stay loaded) |
| Cache persistence | Disk-based, keyed by parallel config | Not built-in |

For memory-constrained scenarios (e.g., Wan's two-transformer setup on a T3K), the integration must preserve TT-DiT's dynamic loading behavior. This means Strategy A is the safest for video pipelines. Strategies B and C would need to either:

- Expose `unload_set` semantics through `TTNNModule` (add a `deallocate_weights()` method), or
- Implement a shared memory manager that both systems consult before loading weights.

---

## 8. Recommended Integration Path

For teams porting DiT models from TT-DiT to a TT-Symbiote-based serving infrastructure, the recommended progression is:

1. **Start with Strategy A** (pipeline-as-opaque-service) to get end-to-end serving working immediately.
2. **Move to Strategy B** when you need per-component debugging or want to use TT-Symbiote's dispatch for new encoder implementations.
3. **Consider Strategy C** only when there is a concrete need for unified dispatch statistics across all components, or when TT-Symbiote's `TracedRun` gains submesh-aware tracing capabilities.

The key principle: **do not attempt to force a single-model-tree abstraction onto a multi-component pipeline.** The DiT pipeline's orchestration logic (scheduler steps, mesh reshaping, CFG combination) is inherently imperative and does not benefit from module-tree recursion.

---

## Key Takeaways

1. **TT-Symbiote's module replacement pattern is designed for single-model-tree architectures** (e.g., a ResNet, a transformer decoder). DiT pipelines are multi-component orchestrators where the control flow lives outside any `nn.Module` tree, making leaf-level replacement insufficient.

2. **The tracing systems are complementary but incompatible in their current forms.** TT-DiT traces the entire denoising step; TT-Symbiote traces individual modules. The `Tracer` utility class in `utils/tracing.py` is the best candidate for a shared primitive.

3. **Three integration strategies span the effort/integration-depth spectrum.** Pipeline-as-opaque-service (Strategy A) is immediate; encoder/VAE replacement (Strategy B) enables incremental debugging; full module wrapping (Strategy C) provides unified metrics at the cost of significant adaptation.

4. **Memory management is the hardest reconciliation problem.** TT-DiT's `unload_set` pattern for dynamic model swapping has no TT-Symbiote equivalent and is critical for video pipeline deployments on memory-constrained devices.

5. **Tensor format bridging is straightforward** using TT-Symbiote's existing `wrap_to_torch_ttnn_tensor` and `fast_unwrap_to_device` utilities. The performance cost is minimal for pipeline boundaries (which are infrequent compared to per-op dispatch).

---

**Next:** [Chapter 6 -- Weight Loading and Preprocessing](../ch6_weight_loading/index.md)
