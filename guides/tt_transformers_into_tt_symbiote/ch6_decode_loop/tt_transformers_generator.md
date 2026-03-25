# TT Transformers Generator

Source file: `models/tt_transformers/tt/generator.py`

---

## 1. `Generator.__init__`

```python
class Generator(WarmupForwardMixin):
    def __init__(self, model, model_args, mesh_device, processor=None, tokenizer=None):
```

### Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `model` | `list` | List of model instances, one per data-parallel shard. `len(model)` sets `self.data_parallel`. |
| `model_args` | `list` | List of `ModelArgs` objects aligned with `model`. Each carries `mesh_device`, `max_batch_size`, `max_prefill_chunk_size`, and the prefill seq-len lists used for warmup. |
| `mesh_device` | `ttnn.MeshDevice` | The top-level mesh. Stored but routing is per-shard via `model_args[i].mesh_device`. |
| `processor` / `tokenizer` | optional | Used by `generate()` (vision path) for post-processing tokens into text. |

### State stored at construction

```python
self.data_parallel = len(self.model)
self.prev_page_table = None

# Prefill trace state — keyed by f"{prefill_seq_len}_{model_id}"
self.trace_id_prefill   = defaultdict(lambda: None)
self.trace_inputs_prefill  = defaultdict(lambda: None)
self.trace_output_prefill  = defaultdict(lambda: None)

# Decode trace state — keyed by sampling_on_device bool
self.trace_ids_decode   = defaultdict(lambda: None)   # {bool: {device_id: trace_id}}
self.trace_inputs_decode = defaultdict(lambda: None)
self.trace_output_decode = defaultdict(lambda: None)

self.prefill_traces_warmup = False
self.already_warmed_up_prefill = False
self.enable_split_sampling = True   # split decode trace from sampling step
self.mode = None
```

The three `trace_*` dictionaries are the trace cache. The `defaultdict(lambda: None)` pattern
means a missing key returns `None`, which the calling code uses as a "not yet captured" signal.

### `SamplingParams` and `Sampling`

`SamplingParams` is imported from `models.common.sampling`. It carries per-request sampling
configuration (temperature, top-p, etc.) and is passed through `prefill_forward_text` and
`decode_forward` to control whether sampling runs on-device. `broadcast_sampling_params` and
`chunk_sampling_params` distribute a single `SamplingParams` across batch slots or
data-parallel shards. The model's `sampling` attribute (accessed via
`getattr(model_instance, "sampling", None)`) is the on-device sampling module; its presence
is checked with `_supports_on_device_sampling`.

### Class-level `model_capabilities`

```python
model_capabilities = {
    "supports_prefix_caching": True,
}
```

Subclasses (vLLM adapter) may override this dict to advertise scheduling capabilities to the
serving framework.

---

## 2. `warmup_model_prefill`

```python
def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device,
                         non_greedy_decoding_on_device):
```

This method is guarded by `self.already_warmed_up_prefill`; it runs at most once.

### What it sweeps

1. Calls `self.model_args[0].get_warmup_prefill_supported_seq_lens()` to get the list of
   sequence lengths that need compilation.
2. Iterates over data-parallel shards (`model_id`) and supported sequence lengths.
3. For `model_id != 0` it skips any sequence length that is not in
   `model_args[0].trace_prefill_supported_seq_lens` (lengths that can be traced), because
   compilation already happened on shard 0.
4. For each `(model_id, seq_len)` pair it calls `prefill_forward_text` with zero-valued
   dummy inputs to force TTNN JIT compilation.

Batch size is iterated over `[1, 32]`, but the 32-user batched-prefill path is currently
skipped with a `continue` (marked `# TODO: Remove continue when batched prefill is
supported`). A separate `logger.warning("Batched prefill in TTT is not supported")` is
issued unconditionally at the top of `warmup_model_prefill`, not inside the batch-size loop.

### Sampling parameters sweep

`_create_sampling_params` is called **once total** across the entire warmup sweep (not once
per shard). A `sampling_parameters_sweeped` boolean flag is set to `True` after the first
successful call; all subsequent `(model_id, seq_len, batch_size)` iterations receive
`sampling_params = [None]`. The `_supports_on_device_sampling` attribute on the model
instance gates whether `sampling_params` is non-None.

### Why this exists

TTNN uses lazy JIT compilation: the first execution of any op compiles a program. Warmup
ensures that all program compilations happen before the first live request, so that decode
latency is deterministic from step 1.

---

## 3. Trace capture — prefill

### `_capture_trace_prefill`

```python
def _capture_trace_prefill(self, prefill_ids, page_table=None, kv_cache=None,
                            model_id=-1, global_user_id=None):
```

The capture sequence:

```
1. prepare_prefill_inputs_trace(prefill_ids, page_table=page_table)
        -> host_inputs = (tokens_tt, rot_mats_global, rot_mats_local, page_table_tt,
                          chunk_page_table_tt)
   Rotation matrices (indices 1, 2) are pointers into the global cos/sin matrix
   already on device; they are NOT included in the host input tuple.

2. copy_host_to_device(host_inputs)        -> device_inputs

3. transform_and_embed_prefill_inputs_device(*device_inputs)
        -> transformed_inputs (embedded tokens, page table, chunk page table)

4. ttnn_prefill_forward(x, rot_mats_global, rot_mats_local, page_table,
                         chunk_page_table, kv_cache)
   -- compile run: TTNN JIT compiles all ops --

5. copy_host_to_device(host_inputs)        -> fresh device_inputs

6. trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

7. transform_and_embed_prefill_inputs_device(*device_inputs)
8. ttnn_prefill_forward(...)               -> tt_out_trace

9. ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

returns (trace_id, tt_out_trace, *device_inputs)
```

The step 4 / step 8 split is intentional: step 4 forces compilation so that step 8 records
only device commands, not compilation work. The `device_inputs` returned are the
**persistent input buffers** bound to the trace; callers must update them in-place on
subsequent executions.

### `_easy_trace_prefill`

```python
def _easy_trace_prefill(self, prefill_ids, page_table=None, user_id=0,
                         last_token_idx=None, kv_cache=None, model_id=-1,
                         prefill_seq_len=None, **kwargs):
```

This is the lazy-capture wrapper. The trace cache key is `f"{prefill_seq_len}_{model_id}"`.
On first call it invokes `_capture_trace_prefill` and stores
`(trace_id, device_inputs, tt_out_trace)`. On subsequent calls it invokes
`_prefill_forward_trace` to copy new host inputs into the stored device buffers and call
`ttnn.execute_trace` (non-blocking, `blocking=False`).

---

## 4. Trace capture — decode

### `_capture_decode_trace_text`

```python
def _capture_decode_trace_text(self, tokens, current_pos, page_table=None,
                                 kv_cache=None, sampling_on_device=False):
```

The capture sequence:

```
1. _decode_forward_no_trace_text(tokens, current_pos, ...)
   -- compile run --

2. For each data-parallel shard i:
     prepare_decode_inputs_host(tokens[i], current_pos[i], page_table=...)
       -> host_inputs_i
     copy_host_to_device(host_inputs_i) -> device_inputs[i]

3. For each shard i:
     split_enabled = (
         sampling_on_device
         and sampling_module is not None
         and sampling_module.enable_internal_trace
     )
     trace_id = ttnn.begin_trace_capture(mesh_device_i, cq_id=0)
     tt_out_trace[i] = ttnn_decode_forward(
         *device_inputs[i], kv_cache=...,
         sampling_on_device=sampling_on_device,
         capture_sampling_trace=split_enabled,
     )
     ttnn.end_trace_capture(mesh_device_i, trace_id, cq_id=0)

     if split_enabled:
         sampling_module.capture_trace(logits=tt_out_trace[i],
                                        tt_out_tok=device_inputs[i][0])

returns (trace_ids, tt_out_trace, *device_inputs)
```

### Split-sampling structure

`self.enable_split_sampling` defaults to `True`. When `sampling_on_device=True` and
`enable_split_sampling=True`, `_set_sampling_trace_mode(True)` sets
`sampling_module.enable_internal_trace = True` on each shard. This tells
`ttnn_decode_forward` to stop the main decode trace at logits (set
`capture_sampling_trace=True`) and then `sampling_module.capture_trace` records the sampling
step as a separate trace. During execution the main decode trace runs, then
`sampling_module.sample(logits=..., tt_out_tok=...)` runs the sampling trace. This split
allows the sampling step to be re-captured independently without re-capturing the full
transformer forward.

### `_decode_forward_trace_text`

```python
def _decode_forward_trace_text(self, tokens, current_pos, page_table=None,
                                 kv_cache=None, sampling_on_device=False,
                                 reset_batch=False):
```

The key is `sampling_on_device` (bool): the decode trace is keyed by whether sampling is
included. On first call the trace is captured. On subsequent calls:

- `reset_inputs` is set to `True` when `reset_batch` is set (mode switched from prefill to
  decode), when `sampling_on_device=False`, or when the page table has changed since the
  last step.
- If `reset_inputs`, `prepare_decode_inputs_host` is called and the result is copied into
  the stored device buffers.
- `ttnn.execute_trace` is called for each shard with `blocking=False`.
- If `sampling_on_device`, the split-sample path calls `sampling_module.sample(logits=...,
  tt_out_tok=...)` after the trace.

---

## 5. `decode_forward` — the main decode loop entry point

```python
def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None,
                   enable_trace=True, read_from_device=True,
                   sampling_params: SamplingParams = None,
                   reset_batch=False, prompt_tokens=None, output_tokens=None):
```

Annotated as "called by vLLM."

### Step-by-step

1. **Mode switch**: if `self.mode != Mode.DECODE`, sets it and calls
   `model[i].switch_mode(Mode.DECODE)` on each shard. Sets `mode_switched = True` which
   forces `reset_batch=True` on the first decode call after a prefill.

2. **Sampling setup**: if `sampling_params` is not None, `sampling_on_device = True`.
   `split_sampling_enabled` combines `enable_split_sampling` with `sampling_on_device`.
   `_set_sampling_trace_mode(split_sampling_enabled)` is called. `chunk_sampling_params`
   distributes `sampling_params` across `sampling_dp` shards.
   `sampling_module.apply_decode_state(...)` and `sampling_module.seed_manager.get_new_values()`
   are called per shard.

3. **Chunking inputs**: `torch.chunk` splits `tokens`, `start_pos`, and `page_table` across
   `self.data_parallel` shards.

4. **Forward call**:
   ```python
   if enable_trace:
       tt_decode_output = self._decode_forward_trace_text(**decode_kwargs,
                                                           reset_batch=mode_switched)
   else:
       tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)
   ```

5. **Output collection**:
   ```python
   if read_from_device:
       to_host = self.read_decode_output(tt_decode_output)
       return self.process_decode_output_host(to_host, is_tokens=(sampling_params is not None))
   ```
   `read_decode_output` is called without `async_read=True`, so it follows the default
   synchronous path: it calls blocking `.cpu()` per shard (no `blocking=False`, no
   `ttnn.record_event`). An async path (`.cpu(blocking=False)` + `ttnn.record_event`) exists
   in `read_decode_output` but is only used when the caller passes `async_read=True`.
   `process_decode_output_host` calls `model[i].process_output_decode` to reshape the raw
   tensor and returns `(torch.cat(logits, 0), torch.cat(log_probs, 0))`.

### `_decode_forward_no_trace_text`

The non-trace path prepares inputs with `model[i].prepare_inputs_decode(tokens[i],
current_pos[i], user_page_table)` and calls `model[i].ttnn_decode_forward` directly.
Used both for the warmup compile run inside `_capture_decode_trace_text` and when
`enable_trace=False` is passed to `decode_forward`.

---

## 6. Paged attention integration

### How `page_table` is passed in

`page_table` throughout `Generator` is a `torch.Tensor` of shape `[batch, num_blocks]` with
dtype `torch.int32`. Each row is the block index list for that user's KV cache pages. It is
chunked with `torch.chunk(page_table, self.data_parallel, 0)` so each model shard sees its
slice.

In `prefill_forward_text`, the per-user slice is computed as `page_table[idx : idx + 1]`
and passed to `_get_prefill_user_page_table` which pads it to match the required block count.
For chunked prefill, `chunk_page_table` is a sub-slice of the padded page table covering
only the blocks for the current chunk.

### `TTNNPagedAttentionKVCache` expectation

`_capture_trace_prefill` and `_capture_decode_trace_text` pass `kv_cache` directly to
`ttnn_prefill_forward` / `ttnn_decode_forward`. The model layers expect a
`TTNNPagedAttentionKVCache` object (or equivalent) that accepts a `page_table` tensor to
perform paged scatter/gather during attention. Without a `page_table`, the model falls back
to non-paged KV cache and prefill tracing is disabled (`enable_trace = False` is forced in
`prefill_forward_text` when `page_table is None`).

The block size is obtained via `get_block_size(kv_cache[model_id])` from
`models.tt_transformers.tt.common` and used in `warmup_model_prefill` to size the dummy
page table for warmup.

---

## 7. vLLM and SGLang extensions

### `generator_vllm.py` — exists

This module provides two standalone factory functions and a collection of `Generator`
subclasses — one per supported model family — that adapt the `Generator` interface for the
vLLM serving framework:

- `allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, dp_model, tt_cache_path)` —
  allocates paged KV cache tensors layer by layer on each submesh device, reading per-layer
  dtype from `dp_model[mesh_idx].args.optimizations.get_tensor_dtype(decoder_id, tensor)`.
  Falls back to `ttnn.bfloat8_b` when no optimizations are configured.
- `initialize_vllm_text_transformer(hf_config, tt_data_parallel, mesh_device, ...)` —
  creates submeshes via `create_submeshes`, builds `ModelArgs` per submesh, and returns a
  `(tt_model, model_args)` tuple (the raw model list and args list, not a `Generator`).

The `Generator` subclasses defined in this module are:

| Class | Bases | Notes |
|-------|-------|-------|
| `LlamaForCausalLM` | `Generator` | Text-only; `supports_prefix_caching: True` |
| `QwenForCausalLM` | `Generator` | Text-only; `supports_prefix_caching: True` |
| `MistralForCausalLM` | `Generator` | Text-only; `supports_prefix_caching: True` |
| `GptOssForCausalLM` | `Generator` | Sliding-window; `supports_prefix_caching: False` |
| `MllamaForConditionalGeneration` | `Generator, SupportsMultiModal` | Multimodal vision-language; `supports_prefix_caching: False` (V0 only) |
| `Gemma3ForConditionalGeneration` | `Generator, SupportsMultiModal` | Multimodal; registered with `MULTIMODAL_REGISTRY` |

Each subclass implements `initialize_vllm_model`, `prefill_forward`, `decode_forward`, and
`allocate_kv_cache`, delegating the heavy lifting to the `Generator` base-class methods.

`generator.py` marks `prefill_forward_text`, `decode_forward`, `prefill_forward`,
`read_decode_output`, and `process_decode_output_host` with `# Note: This function is called
by vLLM` comments, identifying the stable interface surface.

### `generator_sglang.py` — exists

Mirrors the vLLM module in structure, providing `allocate_sglang_kv_cache` and
`initialize_sglang_text_transformer` plus the same set of text-only `Generator` subclasses
(`LlamaForCausalLM`, `QwenForCausalLM`, `MistralForCausalLM`, `GptOssForCausalLM`). Unlike
the vLLM module, the SGLang module does not include multimodal subclasses.

- `allocate_sglang_kv_cache(...)` — identical logic to `allocate_vllm_kv_cache`; uses the
  same `empty_{kv}cache_paged_attention{shape}` cache-file template.
- `initialize_sglang_text_transformer(...)` — same pattern as the vLLM equivalent; returns
  a `(tt_model, model_args)` tuple.

Neither extension module adds scheduling or batch management logic; those responsibilities
remain in the respective serving frameworks. The extensions are thin factories that convert
framework-level KV cache allocations and config objects into the `Generator` / `ModelArgs`
types that `generator.py` consumes.

---

**Next:** [`symbiote_inference_path.md`](./symbiote_inference_path.md)
