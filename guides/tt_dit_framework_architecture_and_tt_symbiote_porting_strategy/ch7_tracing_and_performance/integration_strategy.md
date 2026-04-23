# Integration Strategy: Tracing for DiT in TT-Symbiote

## Prerequisites

- [`tt_dit_tracer.md`](./tt_dit_tracer.md): TT-DiT's `Tracer` class and `PipelineTrace` dataclass patterns.
- [`symbiote_traced_run.md`](./symbiote_traced_run.md): TT-Symbiote's `TracedRun` three-phase lifecycle, `TTNNLayerStack`, and hook system.
- [Chapter 2 -- Parallelism and CCL](../ch2_parallelism_and_ccl/index.md): CCL operations, submesh management, and synchronization requirements.
- [Chapter 5 -- Pipelines and Serving](../ch5_pipelines_and_serving/index.md): pipeline structure, denoising loop, submesh-per-prompt architecture.

## Comparative Analysis

### Pipeline-Level vs. Module-Level Tracing

The two frameworks make fundamentally different tracing decisions, each with distinct tradeoffs:

| Dimension | TT-DiT (`Tracer` / `PipelineTrace`) | TT-Symbiote (`TracedRun`) |
|-----------|--------------------------------------|---------------------------|
| **Trace granularity** | Entire denoising step | Individual `TTNNModule` (or `TTNNLayerStack`) |
| **Number of traces per step** | 1 per submesh | 1 per trace-enabled module per submesh |
| **Host overhead per step** | 1 `execute_trace` + input copies | $M$ `execute_trace` calls + $M$ sets of input copies |
| **Capture phases** | 2 (compile + capture) | 3 (warm-up + internal warm-up + capture) |
| **Input update mechanism** | Manual per-field `copy_host_to_device_tensor` | Automatic via `_copy_inputs_to_trace_buffer` |
| **Non-traced operations** | Not supported (all ops must be in trace) | Transparent fallback for non-traced modules |
| **CCL handling** | CCL inside trace (full pipeline captured) | CCL inside module traces; inter-module CCL needs care |
| **Multi-resolution** | Requires re-capture (manual) | Automatic via signature-based cache keys |
| **Memory per trace** | Minimal (one trace buffer per submesh) | Proportional to number of traced modules |

### Performance Implications

For a DiT model with $L$ transformer layers, $S$ denoising steps, and $D$ submeshes:

**TT-DiT pipeline trace:**
- Capture cost: $D$ traces, each covering $L$ layers + scheduling arithmetic
- Per-step replay cost: $D \times (k \text{ input copies} + 1 \text{ execute\_trace})$ where $k$ is the number of changing inputs (typically 2--3: timestep, sigma, sometimes latents)
- Total host ops for generation: $D \times (k + 1) \times S$

**TT-Symbiote with `TTNNLayerStack`:**
- Capture cost: 1 trace covering the $L$-layer stack + additional traces for embeddings, normalization, etc.
- Per-step replay cost: $D \times (M \text{ execute\_trace calls} + M \text{ input copy sets})$ where $M$ is the number of traced modules (layer stack + a few others)
- Total host ops: $D \times (M \times (\bar{k}_m + 1)) \times S$ where $\bar{k}_m$ is the average number of input copies per module

With `TTNNLayerStack`, $M$ is small (typically 3--5 for a DiT: patch embedding, layer stack, final norm, output projection). Without it, $M = L + \text{overhead modules}$, which can be 28+ for SD3.5.

### What TT-DiT Gets for Free

Several properties of TT-DiT's architecture simplify its tracing story:

1. **No dispatch interception layer.** Every operation is an explicit TTNN call, so there is no question of whether an operation can be traced -- they all can.

2. **No `TorchTTNNTensor` wrapper.** TT-DiT passes raw `ttnn.Tensor` objects, avoiding the wrapping/unwrapping overhead that Symbiote's `module_run` performs via `compose_transforms`.

3. **Static computation graph.** DiT models have no dynamic control flow in the denoising step -- the same operations execute in the same order every time. This is a prerequisite for tracing.

4. **Pipeline owns the scheduler.** The scheduler's Euler step arithmetic (sigma scaling, latent update) is included in the trace, eliminating an entire class of per-step host computation.

### What TT-Symbiote Provides That TT-DiT Does Not

1. **Automatic fallback.** Modules that cannot be traced (complex control flow, unsupported ops) are handled transparently.

2. **Heterogeneous module support.** The same model can mix traced and untraced modules, traced TTNN modules and CPU-fallback modules.

3. **Shape-adaptive re-capture.** The signature-based cache automatically captures new traces when input shapes change, supporting variable-resolution generation without manual intervention.

4. **Hook extensibility.** The `pre_trace_execute`/`post_trace_execute` hooks allow modules to customize trace replay behavior without modifying the core tracing logic.

## Recommended Approach for DiT Models in TT-Symbiote

Based on the analysis of both systems, the recommended strategy for porting TT-DiT's tracing to TT-Symbiote involves three tiers of increasing integration depth.

### Tier 1: TTNNLayerStack Wrapping (Minimal Change)

**Approach:** Wrap the DiT transformer's layer sequence in a `TTNNLayerStack`, leaving the pipeline shell, embeddings, and scheduler in untraced Symbiote code.

```python
@trace_enabled
class DiTLayerStack(TTNNLayerStack):
    """Stack of DiT transformer blocks for single-trace capture."""

    def __init__(self, blocks):
        super().__init__(blocks)

    def forward(self, hidden_states, **kwargs):
        for block in self.layers:
            hidden_states = block.forward(hidden_states, **kwargs)
        return hidden_states
```

**Pros:**
- Captures the most computationally expensive part (the $L$-layer transformer stack) in a single trace.
- Embeddings, normalization, and scheduler arithmetic remain as regular Symbiote modules, preserving flexibility.
- Compatible with Symbiote's existing run-mode infrastructure.

**Cons:**
- The scheduler Euler step is not traced (runs on host each step).
- Embedding computation (timestep, patch, positional) executes untraced.
- $M > 1$ `execute_trace` calls per step instead of TT-DiT's single call.

**Expected performance:** Within 10--20% of TT-DiT's pipeline-level trace, since the transformer stack dominates the compute. Embedding and scheduler overhead is typically <5% of step time.

### Tier 2: Expanded TTNNLayerStack with Embeddings

**Approach:** Create a broader `TracedDiTForward` module that encompasses the embedding layers, transformer stack, and final output projection as a single traced unit.

```python
@trace_enabled
class TracedDiTForward(TTNNModule):
    def __init__(self, patch_embed, timestep_embed, transformer_blocks,
                 final_norm, output_proj):
        super().__init__()
        self.patch_embed = patch_embed
        self.timestep_embed = timestep_embed
        self.blocks = list(transformer_blocks)
        self.final_norm = final_norm
        self.output_proj = output_proj

    def forward(self, latents, prompt_embeds, timestep, **kwargs):
        hidden = self.patch_embed.forward(latents)
        t_emb = self.timestep_embed.forward(timestep)
        for block in self.blocks:
            hidden = block.forward(hidden, encoder_hidden_states=prompt_embeds,
                                   temb=t_emb, **kwargs)
        hidden = self.final_norm.forward(hidden, t_emb)
        return self.output_proj.forward(hidden)
```

**Pros:**
- Captures everything except the scheduler in a single trace.
- Approaches TT-DiT's pipeline-level trace coverage.
- Embedding recomputation is eliminated on replay.

**Cons:**
- More coupling between modules -- changes to embedding or normalization APIs require updating the wrapper.
- The scheduler Euler step still runs on host.

### Tier 3: Full Pipeline Trace (Maximum Performance)

**Approach:** Implement a Symbiote-native equivalent of TT-DiT's `PipelineTrace` pattern, capturing the entire denoising step including scheduler arithmetic as a single trace.

This requires the scheduler's tensor operations (sigma scaling, latent update) to be implemented as TTNN operations rather than PyTorch/host operations:

```python
@trace_enabled
class TracedDenoisingStep(TTNNModule):
    def forward(self, latents, prompt_embeds, timestep, sigma_diff, **kwargs):
        # Full transformer forward
        noise_pred = self.transformer.forward(latents, prompt_embeds,
                                               timestep, **kwargs)
        # Euler step (in TTNN, not PyTorch)
        # latents_new = latents + sigma_diff * noise_pred
        latents_new = ttnn.add(latents,
                               ttnn.mul(sigma_diff, noise_pred))
        return latents_new
```

**Pros:**
- Matches TT-DiT's single-trace-per-step performance.
- Eliminates all per-step host computation except input tensor copies.

**Cons:**
- Requires porting scheduler arithmetic to TTNN.
- Less modular -- the scheduler is baked into the traced unit.
- Harder to support multiple scheduler types without per-scheduler trace variants.

### Recommendation

**Start with Tier 1** for initial porting. It provides the majority of the tracing benefit with minimal architectural disruption and is compatible with Symbiote's module-level testing and validation infrastructure. **Progress to Tier 2** once correctness is established, to close the remaining performance gap. **Reserve Tier 3** for production deployments where the last 5--10% of throughput matters.

## CCL-Aware Extensions Needed

DiT models on multi-device configurations (T3K, TG) use CCL operations extensively within transformer blocks (see [Chapter 2](../ch2_parallelism_and_ccl/index.md)). Integrating traced DiT execution with TT-Symbiote requires several CCL-related extensions:

### 1. Synchronization Before Capture

Both TT-DiT and TT-Symbiote already synchronize devices before trace capture, but the patterns differ:

- **TT-DiT:** Synchronizes all submeshes in a loop after each submesh's capture.
- **TT-Symbiote:** Synchronizes the single device associated with the module.

For DiT models with sequence parallelism across a submesh, the synchronization must cover all devices in the submesh:

```python
# In _capture_trace, after warm-up forward:
ttnn.synchronize_device(device)  # Current Symbiote behavior

# Needed for DiT with SP:
for d in submesh_device.get_devices():
    ttnn.synchronize_device(d)
```

Alternatively, Symbiote's `TracedRun.configure` could accept a submesh device and handle multi-device synchronization internally.

### 2. CCL Manager Integration

TT-DiT pipelines initialize `CCLManager` per-submesh (see [Chapter 2](../ch2_parallelism_and_ccl/index.md)). TT-Symbiote's `DistributedConfig` initializes `TT_CCL` similarly. When porting, the CCL manager must be:

- Initialized before any trace capture begins.
- Available to all modules within the traced `TTNNLayerStack`.
- Persistent across trace replays (CCL state must not be re-initialized).

TT-Symbiote's existing `DistributedConfig.ccl_manager` field on `TTNNModule.device_state` provides the right hook for this. The `TTNNLayerStack` should propagate the parent's `device_state` to all child layers before capture.

### 3. Per-Submesh Trace Management

TT-DiT's `PipelineTrace` pattern stores one trace per submesh. TT-Symbiote's `TracedRun` currently stores traces in a flat cache keyed by `(module_name, signature)`. For multi-submesh DiT execution, the cache key should incorporate the submesh identity:

```python
# Current
cache_key = (module_name, _compute_args_signature(args))

# Extended for multi-submesh
cache_key = (module_name, submesh_id, _compute_args_signature(args))
```

This ensures that each submesh gets its own trace, since different submeshes may have different device-local tensor layouts even for the same logical operation.

### 4. Pre-Trace Hook for CCL Buffers

Some CCL operations (all-gather, reduce-scatter) use internal buffers that must be pre-allocated and persistent across trace replays. The `pre_trace_execute` hook is the natural place for modules to copy CCL-related inputs into persistent buffers:

```python
class DiTAttentionBlock(TTNNModule):
    def pre_trace_execute(self, func_args, func_kwargs):
        # Copy sequence-parallel gather buffers
        if self.ccl_gather_buffer is not None:
            ttnn.copy(func_args[0], self.ccl_gather_buffer)
```

### 5. Topology-Aware Trace Capture

TT-DiT's `CCLManager` is initialized with a specific topology (Linear or Ring) and link count. When capturing traces that include CCL operations, the topology must be consistent between capture and replay. TT-Symbiote's `TracedRun` should validate that the CCL topology has not changed between capture and replay, or include topology parameters in the cache key.

## Migration Checklist

For teams porting a TT-DiT model to TT-Symbiote with tracing:

1. **Identify all modules in the model's forward pass.** Map each to either a `TTNNModule` subclass or a fallback PyTorch module.

2. **Mark trace-eligible modules with `@trace_enabled`.** Typically: attention blocks, feed-forward blocks, normalization layers. Mark modules with dynamic control flow or unsupported ops with `@trace_disabled`.

3. **Create a `TTNNLayerStack`** wrapping the transformer block sequence. Verify that all blocks have identical `forward()` signatures.

4. **Configure `TracedRun`** with the correct device, command queue, and input memory config:
   ```python
   TracedRun.configure(device=mesh_device, cq_id=0,
                       input_memory_config=ttnn.DRAM_MEMORY_CONFIG)
   ```

5. **Ensure weights are preprocessed and on-device** before the first `TracedRun.module_run` call. The three-phase lifecycle's warm-up phase will assert this.

6. **Tensorize all per-step scalars.** Timestep, sigma, guidance scale must be `ttnn.Tensor`, not Python scalars.

7. **Test with `set_run_mode("NORMAL")` first** to validate correctness, then switch to `set_run_mode("TRACED")`.

8. **Profile with `DispatchManager.save_stats_to_file`** to verify that traced modules show reduced forward time compared to normal execution.

9. **Verify CCL synchronization** by running multi-device generation and checking for trace capture errors related to in-flight CCL operations.

10. **Measure trace memory overhead** using device memory profiling. Each trace consumes DRAM proportional to the number of operations recorded.

## Key Takeaways

1. **`TTNNLayerStack` (Tier 1) is the recommended starting point** for DiT tracing in Symbiote -- it captures the dominant compute (transformer layers) in a single trace while preserving module-level flexibility for embeddings and scheduling.

2. **CCL synchronization is the critical correctness concern** when porting from TT-DiT's per-submesh traces to Symbiote's module-level traces -- in-flight collective operations must be flushed before any trace capture begins.

3. **Cache key extensions for submesh identity** are needed to support multi-device DiT execution, ensuring each submesh's trace is captured and replayed independently.

4. **The `pre_trace_execute` hook** provides the right extensibility point for CCL buffer management, allowing modules to prepare persistent communication buffers before trace replay.

5. **Progressive integration (Tier 1 to Tier 3)** allows teams to validate correctness at each stage before committing to deeper performance optimization, with Tier 1 already achieving approximately 80--90% of TT-DiT's traced throughput.

---

**Next:** [Chapter 8 -- Porting Strategy and Model Prioritization](../ch8_porting_strategy/index.md)
