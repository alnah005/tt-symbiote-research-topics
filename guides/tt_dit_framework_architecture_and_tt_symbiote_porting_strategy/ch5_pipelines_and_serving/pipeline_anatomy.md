# Pipeline Anatomy: StableDiffusion3Pipeline as Canonical Example

## Prerequisites

- [Chapter 1 -- Architecture Overview](../ch1_architecture_overview/index.md): framework structure and component taxonomy
- [Chapter 2 -- Parallelism and CCL](../ch2_parallelism_and_ccl/index.md): `DiTParallelConfig`, `ParallelFactor`, `CCLManager`, submesh creation
- [Chapter 3 -- Custom Layers and Ops](../ch3_custom_layers_and_ops/index.md): `Module` base class, `Parameter`, `load_torch_state_dict()`
- [Chapter 4 -- Attention and Transformer Blocks](../ch4_attention_and_transformer_blocks/index.md): `SD35Transformer2DModel` internals
- [Chapter 5 index](./index.md): overview of the six pipeline classes

This document dissects `StableDiffusion3Pipeline` (located in `models/tt_dit/pipelines/stable_diffusion_35_large/pipeline_stable_diffusion_35_large.py`) line by line, then highlights how the Flux1 and Wan pipelines diverge. The goal is to make the pipeline architecture legible enough that porting decisions for TT-Symbiote become concrete.

---

## 1. The `create_pipeline()` Static Factory

Every pipeline exposes a `create_pipeline()` class method that insulates callers from parallelism details:

```python
@staticmethod
def create_pipeline(
    mesh_device,
    batch_size=1,
    image_w=1024, image_h=1024,
    guidance_scale=3.5,
    ...
    cfg_config=None, sp_config=None, tp_config=None,
    num_links=None,
    checkpoint_name="stabilityai/stable-diffusion-3.5-large",
):
```

The factory consults a default-config lookup table keyed by mesh shape:

```python
default_config = {
    (2, 4): {"cfg_config": (2, 1), "sp_config": (2, 0), "tp_config": (2, 1), "num_links": 1},
    (4, 8): {"cfg_config": (2, 1), "sp_config": (4, 0), "tp_config": (4, 1), "num_links": 4},
}
```

For a `(2, 4)` mesh (one TG, e.g., an N150 pair or half of a Galaxy), this means:

- **CFG parallel factor 2 on axis 1**: the 4-wide column dimension is split into two submeshes of width 2, one for unconditional prediction and one for conditional prediction.
- **Sequence parallel factor 2 on axis 0**: the 2-row dimension shards spatial sequences across rows.
- **Tensor parallel factor 2 on axis 1**: within each submesh, weights are sharded across 2 devices on the column axis.

The factory builds a `DiTParallelConfig` from these values, then calls the constructor and `prepare()`:

```python
parallel_config = DiTParallelConfig(
    cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
    tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
    sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
)
pipeline = StableDiffusion3Pipeline(...)
pipeline.prepare(batch_size=batch_size, ...)
return pipeline
```

### Flux1 Divergence

Flux1's factory is similar but adds separate `encoder_tp` and `vae_tp` configs. This is because Flux1 does not use CFG (its model was trained with `cfg_parallel.factor=1`), so the entire mesh acts as a single submesh and encoders/VAE can use the full device complement.

### Wan Divergence

WanPipeline also avoids CFG parallel but introduces `VaeHWParallelConfig` with separate height and width parallel factors, reflecting the 3D nature of video VAE decoding.

---

## 2. Constructor: Component Initialization

The SD3.5 constructor performs four major phases:

### Phase 1: Submesh Creation

```python
submesh_shape = list(mesh_device.shape)
submesh_shape[parallel_config.cfg_parallel.mesh_axis] //= parallel_config.cfg_parallel.factor
self.submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))
```

For a `(2, 4)` mesh with CFG factor 2 on axis 1, this creates two `(2, 2)` submeshes. Each gets its own `CCLManager`:

```python
self.ccl_managers = [
    CCLManager(submesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    for submesh_device in self.submesh_devices
]
```

The encoder and VAE are assigned to specific submesh indices. When the submesh shape does not match the expected encoder layout (e.g., a `(2, 2)` submesh when encoders expect `(1, 4)`), the pipeline performs a **mesh reshape hack**:

```python
if encoder_device.shape[1] != 4:
    self.desired_encoder_submesh_shape = (1, 4)
    # ...
    self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))
```

This reshape is toggled on and off during the `__call__()` method, which is a source of complexity documented by `# HACK` comments throughout the code.

### Phase 2: PyTorch Model Loading

The constructor loads HuggingFace models into CPU memory:

```python
self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(checkpoint_name, subfolder="text_encoder")
# ... similarly for tokenizer_2, text_encoder_2, tokenizer_3
torch_transformer = TorchSD3Transformer2DModel.from_pretrained(
    checkpoint_name, subfolder="transformer", torch_dtype=torch.bfloat16,
)
```

These PyTorch models serve as reference implementations and weight sources. The torch transformer is only used to extract the state dict and config; it is not used during inference.

### Phase 3: TT-NN Model Creation and Weight Loading

For each submesh, a TT-NN transformer is created with the SD3.5-specific architecture parameters:

```python
for i, submesh_device in enumerate(self.submesh_devices):
    tt_transformer = SD35Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16,
        num_layers=38, attention_head_dim=64, num_attention_heads=38,
        joint_attention_dim=4096, ...
        mesh_device=submesh_device,
        ccl_manager=self.ccl_managers[i],
        parallel_config=self.dit_parallel_config,
        padding_config=padding_config,
    )
```

Weights are loaded through the centralized caching system:

```python
cache.load_model(
    tt_model=tt_transformer,
    get_torch_state_dict=torch_transformer.state_dict,
    model_name="stable-diffusion-3.5-large",
    subfolder="transformer",
    parallel_config=self.dit_parallel_config,
    mesh_shape=tuple(submesh_device.shape),
)
```

The CLIP encoders follow the same pattern but use the new `CLIPEncoder` TT-NN implementation (see [Chapter 3](../ch3_custom_layers_and_ops/index.md)), while the T5 encoder uses `T5Encoder`. The encoder weights are loaded via `load_torch_state_dict()` directly rather than through `cache.load_model()`.

### Phase 4: VAE Decoder Creation

The VAE decoder is created from the PyTorch reference:

```python
self._vae_decoder = VAEDecoder.from_torch(
    torch_ref=self._torch_vae.decoder,
    mesh_device=self.vae_device,
    parallel_config=self.vae_parallel_config,
    ccl_manager=self.ccl_managers[vae_submesh_idx],
)
```

---

## 3. Weight Caching via `cache.load_model()`

The `cache` module (`models/tt_dit/utils/cache.py`) implements a three-tier loading strategy:

```
1. Is the model already loaded on device?  -> Return immediately
       |  (tt_model.is_loaded() == True)
       v  No
2. Does a cache directory exist on disk?   -> Load from cache (tt_model.load(cache_dir))
       |  (TT_DIT_CACHE_DIR is set and dir exists)
       v  No
3. Is a PyTorch state dict available?      -> Load from torch, optionally create cache
       |  (get_torch_state_dict is not None)
       v  No
4. Raise MissingCacheError
```

### Cache Directory Structure

The cache path is computed deterministically from the parallelism configuration:

```python
def model_cache_dir(*, model_name, subfolder, parallel_config, mesh_shape, dtype="bf16", is_fsdp=False):
    parallel_key = config_id(parallel_config)  # e.g., "CP2_1_TP2_1_SP2_0_"
    mesh_key = "x".join(str(x) for x in mesh_shape)  # e.g., "2x2"
    key = f"{parallel_key}mesh{mesh_key}_{dtype}"
    return Path(cache_dir) / model_name / subfolder / key
```

This means the same model cached for a `(2, 4)` mesh with TP=2 on axis 1 will not be reused for a `(4, 8)` mesh with TP=4 on axis 1 -- the sharding is baked into the cache. This is because TT-NN tensors on disk are stored in their already-sharded form (one file per device in the mesh).

### Unload-Before-Load: `set_unload_set()`

When device memory is too small to hold all models simultaneously, the `Module.unload_set` mechanism allows one model's loading to trigger the deallocation of another:

```python
# In cache.load_model():
for module in tt_model.unload_set or []:
    module.deallocate_weights()
```

WanPipeline uses this extensively for its two-transformer architecture:

```python
self.tt_umt5_encoder.set_unload_set(self.transformer_2)
self.transformer.set_unload_set(self.transformer_2)
self.transformer_2.set_unload_set(self.transformer, self.tt_umt5_encoder)
```

This creates a circular dependency chain: loading `transformer` automatically unloads `transformer_2`, and vice versa. MochiPipeline takes a more drastic approach with `reload_dit_model`, which sets `self.transformer = None` before VAE decode and recreates the entire model from cache on the next call.

---

## 4. The `__call__()` Inference Flow

### Stage 1: Text Encoding

```python
prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
    prompt_1=prompt_1, prompt_2=prompt_2, prompt_3=prompt_3,
    negative_prompt_1=negative_prompt_1, ...
)
```

The encoding pipeline for SD3.5:

1. Tokenize with `CLIPTokenizer` (max length 77 for CLIP, 256 for T5).
2. Convert token IDs to `ttnn.Tensor` with `ttnn.TILE_LAYOUT` and `ttnn.uint32` dtype.
3. Run through the TT-NN `CLIPEncoder` on the encoder submesh. Returns `(hidden_states, projected_output)`.
4. Extract the second-to-last hidden state for sequence embeddings (respecting `clip_skip`).
5. Convert back to PyTorch for concatenation with T5 embeddings.
6. Pad CLIP embeddings to match T5 dimension and concatenate along the sequence dimension.
7. If CFG is enabled, repeat the process for negative prompts and concatenate `[negative, positive]` along the batch dimension.

The result is two tensors:
- `prompt_embeds`: shape `[2*B, L_clip + L_t5, D_joint]` (or `[B, ...]` without CFG)
- `pooled_prompt_embeds`: shape `[2*B, D_pool]`

### Stage 2: Latent Preparation

```python
latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)
latents = self.transformers[0].patchify(latents)
```

The raw noise is generated on CPU and then patchified (converted from spatial layout to sequence-of-patches layout). For SD3.5 with `patch_size=2` and a 1024x1024 image:

$$\text{latents\_shape} = (B, 128, 128, 16) \xrightarrow{\text{patchify}} (B, 4096, 64)$$

where $4096 = (128/2) \times (128/2)$ patches and $64 = 16 \times 2 \times 2$ channels per patch.

The latents are then converted to `ttnn.Tensor` with 2D mesh sharding:

```python
shard_latents_dims = [None, None]
shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
tt_initial_latents = ttnn.from_torch(latents, ..., mesh_mapper=ttnn.ShardTensor2dMesh(...))
```

This shards the sequence dimension (dim 2) across the SP axis of the mesh.

### Stage 3: Denoising Loop

The denoising loop iterates over scheduler timesteps:

```python
for i, t in enumerate(tqdm.tqdm(timesteps)):
    sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]
    tt_latents_step_list = self._step(
        timestep=tt_timestep_list,
        latents=tt_latents_step_list,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=tt_prompt_embeds_list,
        pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
        guidance_scale=guidance_scale,
        sigma_difference=tt_sigma_difference_list,
        ...
        traced=traced,
    )
```

The `_step()` method is the performance-critical inner loop, and it is where tracing comes into play. It is described in detail in the next section.

### Stage 4: VAE Decode

After the denoising loop completes:

```python
tt_latents = self.ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
    tt_latents, dim=2, mesh_axis=self.dit_parallel_config.sequence_parallel.mesh_axis
)
```

The sequence-parallel shards are gathered back to the full sequence on the VAE submesh. The latents are then unpatchified back to spatial layout, denormalized, and sent through the TT-NN VAE decoder:

```python
torch_latents = (torch_latents / self._torch_vae_scaling_factor) + self._torch_vae_shift_factor
torch_latents = self.transformers[0].unpatchify(torch_latents, width=..., height=...)
# ... convert to ttnn and decode
decoded_output = self._vae_decoder(self._vae_input_latents)
```

### Stage 5: Post-processing

The decoded output is converted to PIL images:

```python
image = self._image_processor.postprocess(decoded_output, output_type="pt")
output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
```

---

## 5. PipelineTrace and the Tracing Mechanism

### The PipelineTrace Dataclass

Each pipeline that supports tracing defines a `PipelineTrace` dataclass that records handles to the device-resident tensors used as inputs and outputs of the traced region:

**SD3.5:**
```python
@dataclass
class PipelineTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    tid: int
```

**Flux1** (more inputs due to RoPE and guidance embedding):
```python
@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    guidance_input: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor
```

### Trace Capture in `_step()`

SD3.5's trace capture follows this pattern:

```python
if traced and self._trace is None:
    self._trace = [None for _ in self.submesh_devices]
    for submesh_id, submesh_device in enumerate(self.submesh_devices):
        # 1. Compile run (first forward, populates JIT caches)
        pred = inner(latent_device, prompt_device, pooled_projection_device,
                     timestep_device, submesh_id)

        # 2. If this submesh runs the VAE, warm up VAE buffers too
        if submesh_id == self.vae_submesh_idx:
            self._vae_decode(latent_device, ...)

        ttnn.synchronize_device(submesh_device)

        # 3. Begin trace capture
        trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
        pred = inner(latent_device, prompt_device, pooled_projection_device,
                     timestep_device, submesh_id)
        ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

        # 4. Store trace handle
        self._trace[submesh_id] = PipelineTrace(
            spatial_input=latent_device,
            prompt_input=prompt_device,
            ...
            latents_output=pred,
            tid=trace_id,
        )
```

On subsequent steps, the trace is replayed by copying only the changing inputs:

```python
ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._trace[submesh_id].timestep_input)
ttnn.execute_trace(submesh_device, self._trace[submesh_id].tid, cq_id=0, blocking=False)
noise_pred_list.append(self._trace[submesh_id].latents_output)
```

Notice that `blocking=False` is used -- the trace executes asynchronously, and results are consumed only when needed (e.g., during the CFG combination step or at VAE decode time).

### Flux1's Trace Capture Difference

Flux1 skips the compile run and goes directly to trace capture. It also incorporates the `sigma_difference` multiplication into the trace itself via a pre-allocated buffer:

```python
sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)
trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
pred = self._step_inner(...)
ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
```

Then during replay, both the timestep and sigma difference are copied to the trace buffers before execution:

```python
ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
ttnn.copy_host_to_device_tensor(sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input)
ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
```

### The `Tracer` Utility Class

The `utils/tracing.py` module provides a higher-level `Tracer` class that automates the compile-then-capture pattern:

```python
class Tracer:
    def __init__(self, function, /, *, device):
        self._function = function
        self._device = device

    def __call__(self, *args, tracer_cq_id=0, tracer_blocking_execution=True, **kwargs):
        if self._trace_id is None:
            # First call: move inputs to device, compile, capture trace
            self._args = _tree_map(self._move_to_device_if_tensor, args)
            self._function(*self._args, **self._kwargs)           # compile
            trace_id = ttnn.begin_trace_capture(self._device, cq_id=tracer_cq_id)
            outputs = self._function(*self._args, **self._kwargs)  # capture
            ttnn.end_trace_capture(self._device, trace_id, cq_id=tracer_cq_id)
            self._trace_id = trace_id
            self._outputs = outputs
        else:
            # Subsequent calls: update inputs, replay trace
            _tree_map(self._update_input, self._args, args)
            ttnn.execute_trace(self._device, self._trace_id, ...)
        return self._outputs
```

Key features of `Tracer`:
- **Automatic input management**: Uses `_tree_map` to recursively traverse nested data structures (tuples, lists, dicts) and copy updated tensor inputs to the trace buffer.
- **Type safety**: `_verify_value` ensures only supported types (tensors, scalars, None) appear in inputs/outputs.
- **Resource cleanup**: `release()` frees the captured trace and clears all buffers.

Currently, the existing pipelines (SD3.5, Flux1, Motif) use manual trace management rather than the `Tracer` utility. The `Tracer` class represents a newer, cleaner API that future pipelines may adopt.

---

## 6. Scheduler Integration

All pipelines use HuggingFace `diffusers` schedulers. The interaction pattern is:

```python
self._scheduler.set_timesteps(num_inference_steps)
timesteps = self._scheduler.timesteps

for i, t in enumerate(timesteps):
    sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]
    # ... run transformer ...
    # Euler step: latents += sigma_difference * noise_pred
```

The Euler step for flow-matching models reduces to:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + (\sigma_{t+1} - \sigma_t) \cdot \hat{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$$

This is computed in-place using TTNN operations:

```python
ttnn.multiply_(self._sigma_difference_list[submesh_id], self._intermediate_noise_list[submesh_id])
ttnn.add_(latents[submesh_id], self._sigma_difference_list[submesh_id])
```

MochiPipeline delegates the step to the scheduler directly (`self.scheduler.step(noise_pred, t, latents)`), while other pipelines implement it manually for better control over device placement. The Wan pipeline uses `UniPCMultistepScheduler`, which implements a higher-order ODE solver.

---

## 7. Classifier-Free Guidance Strategies

### CFG-Parallel (SD3.5, Motif)

When `cfg_parallel.factor == 2`, the mesh is split into two submeshes. One runs the conditional prediction and the other runs the unconditional prediction simultaneously:

```python
uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True))
cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True))
torch_noise_pred = uncond + guidance_scale * (cond - uncond)
```

The CFG combination currently happens on CPU, which is a performance bottleneck. The result is then re-distributed to both submeshes.

### No CFG (Flux1)

Flux1 was trained without CFG, using a guidance embedding instead. The guidance scale is injected as a scalar tensor input to the transformer:

```python
guidance = torch.full([batch_size], fill_value=guidance_scale) if self._with_guidance_embeds else None
```

### Sequential CFG (Mochi)

Mochi runs unconditional and conditional predictions as two separate forward passes on the same device:

```python
noise_pred_uncond = self.transformer(spatial=latent_model_input[:1], prompt=prompt_embeds[:1], ...)
noise_pred_text = self.transformer(spatial=latent_model_input[1:], prompt=prompt_embeds[1:], ...)
noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
```

This is simpler but uses twice the time per step compared to CFG-parallel.

### Two-Stage Denoising (Wan)

Wan introduces a novel approach where two different transformer models handle different noise levels, controlled by a `boundary_ratio`. The first transformer handles high-noise timesteps and the second handles low-noise timesteps. This uses dynamic loading to swap models when the boundary is crossed.

---

## 8. Warmup and Buffer Pre-allocation

Flux1's constructor includes an explicit warmup call:

```python
# warmup for safe tracing.
self.run_single_prompt(prompt="", num_inference_steps=1, seed=0, traced=False)
self.synchronize_devices()
```

This ensures all lazy initializations and JIT compilations happen before the first real inference call. SD3.5 instead allocates intermediate buffers lazily during the first `_step()` call:

```python
if len(self._intermediate_noise_list) <= i:
    self._intermediate_noise_list.append(
        ttnn.from_torch(latents, ..., device=submesh_device, ...)
    )
    self._sigma_difference_list.append(ttnn.clone(self._intermediate_noise_list[-1]))
```

These buffers are persistent across calls and are reused for in-place operations during traced execution.

---

## Key Takeaways

1. **Pipelines are multi-component orchestrators, not single-model wrappers.** A single pipeline manages tokenizers, multiple text encoders, a DiT transformer (possibly per-submesh), a VAE decoder, and a scheduler. This is architecturally different from LLM serving where the model is a single forward-pass entity.

2. **Tracing is pipeline-specific and manual.** Each pipeline defines its own `PipelineTrace` dataclass, manages trace capture timing, and handles input copying. The newer `Tracer` utility in `utils/tracing.py` provides a cleaner abstraction but is not yet adopted by existing pipelines.

3. **Memory management is a first-class concern.** The `unload_set` mechanism, `reload_dit_model` pattern, and cache directory structure are all designed to handle the reality that DiT models (often 10+ GB for transformers alone) may not all fit in device memory simultaneously.

4. **CFG parallelism is the primary motivation for submesh splitting.** The submesh architecture exists to run conditional and unconditional predictions concurrently. When CFG is not used (Flux1), the submesh abstraction degenerates to a single-submesh case but remains in the code path.

5. **The denoising loop follows the Euler flow-matching step pattern** $\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta\sigma \cdot \hat{\epsilon}$, with variations in how schedulers, guidance, and the step computation are distributed across devices.

---

**Next:** [`mapping_to_symbiote_serving.md`](./mapping_to_symbiote_serving.md)
