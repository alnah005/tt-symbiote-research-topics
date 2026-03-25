# TT Symbiote Inference Path

Source files:
- `models/experimental/tt_symbiote/tests/test_llama.py`
- `models/experimental/tt_symbiote/core/module.py`
- `models/experimental/tt_symbiote/core/run_config.py`

---

## 1. Current Symbiote inference — `test_llama.py`

### How the test drives the model

```python
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
```

`test_llama.py` calls HuggingFace `model.generate()` unmodified. The LLaMA model loaded
from `AutoModelForCausalLM.from_pretrained` is a standard HF `LlamaForCausalLM`; the TTNN
acceleration is applied by replacing individual `nn.Module` instances via
`register_module_replacement_dict`. HF's `generate()` then calls the model's `forward`
repeatedly, and each replaced module intercepts its tensor arguments through the
`TorchTTNNTensor` dispatch mechanism.

### Module replacement

Two replacement passes are performed:

```python
nn_to_nn = {
    model.model.layers[0].mlp.__class__: LlamaMLP,   # rewire MLP without SiLU
}
modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)

nn_to_ttnn = {
    nn.Linear:    TTNNLinear,
    nn.SiLU:      TTNNSilu,
    nn.LayerNorm: TTNNLayerNorm,
    model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    model.model.layers[0].self_attn.__class__:       LlamaAttention,
}
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

After replacement, `set_device(model, device)` propagates the TTNN device to every replaced
module, and `preprocess_weights()` + `move_weights_to_device()` are called for each.

### What differs from TT Transformers

TT Transformers owns the entire generation loop: it controls prefill separately from decode,
manages the KV cache as an explicit `TTNNPagedAttentionKVCache`, warms up all sequence
lengths, and traces the decode step. Symbiote's current path delegates all loop control to
HF `generate()`, which is a Python loop that calls `model.forward` once per token. There is
no prefill/decode phase distinction, no trace capture at the generation-loop level, no paged
attention, and no warmup sweep.

The `test_llama_intelligent` variant replaces `TTNNLinear` with `SmartTTNNLinear` and
excludes `lm_head` from replacement (`exclude_replacement={"lm_head"}`), but is otherwise
structurally identical.

---

## 2. `@trace_enabled` and `@trace_disabled`

### What the decorators do

Both decorators are defined in `models/experimental/tt_symbiote/core/run_config.py`:

```python
def trace_enabled(cls: Type) -> Type:
    global _TRACE_ENABLED_TUPLE
    _TRACE_ENABLED_CLASSES.add(cls)
    _TRACE_ENABLED_TUPLE = None
    return cls

def trace_disabled(cls: Type) -> Type:
    global _TRACE_DISABLED_TUPLE
    _TRACE_DISABLED_CLASSES.add(cls)
    _TRACE_DISABLED_TUPLE = None
    return cls
```

They mutate module-level sets. `is_trace_enabled(module)` tests membership:

```python
def is_trace_enabled(module) -> bool:
    return (isinstance(module, _TRACE_ENABLED_TUPLE)
            and not isinstance(module, _TRACE_DISABLED_TUPLE))
```

A class decorated `@trace_disabled` that inherits from a `@trace_enabled` parent is
excluded — the disabled set takes priority.

### Which modules carry which decorator

| Module | Decorator | Location |
|--------|-----------|----------|
| `TTNNLinear` | `@trace_enabled` | `modules/linear.py:15` |
| `TTNNLinearLLama` | `@trace_disabled` | `modules/linear.py:179` |
| `TTNNLinearLLamaIColShardedWRowSharded` | `@trace_disabled` | `modules/linear.py:196` |
| `TTNNLinearLLamaBFloat16` | `@trace_disabled` | `modules/linear.py:279` |
| `TTNNLinearActivation` | `@trace_enabled` | `modules/linear.py:302` |
| `TTNNDistributedRMSNorm` | `@trace_enabled` | `modules/normalization.py:99` |
| `TTNNConv2dNHWC` | `@trace_enabled` | `modules/conv.py:137` |

### Why LLaMA linears are `@trace_disabled`

The LLaMA-specific linear variants (`TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`,
`TTNNLinearLLamaIColShardedWRowSharded`) all call `@deallocate_weights_after` in their
`forward` method: weights are moved to device at the start of forward and freed immediately
after. This behavior is incompatible with trace capture, because a captured trace holds
references to the exact device memory addresses recorded at capture time. If weights are
freed after the compile run and re-allocated at different addresses before the trace
execution, the trace will read stale or invalid memory. Marking these classes
`@trace_disabled` ensures they fall through to normal (non-traced) execution in
`TracedRun.module_run`.

---

## 3. `TT_SYMBIOTE_RUN_MODE=TRACED`

The `TRACED` run mode **does exist** in `run_config.py`. The registry entry is:

```python
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL":      NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL":         SELRun,
    "DPL":         DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU":         CPU,
    "TRACED":      TracedRun,
}
```

`get_tensor_run_implementation()` reads `os.environ.get("TT_SYMBIOTE_RUN_MODE",
_current_run_mode)` and returns the corresponding class, which becomes the global
`TENSOR_RUN_IMPLEMENTATION` used in `TTNNModule.__call__`. Setting
`TT_SYMBIOTE_RUN_MODE=TRACED` before import will route all `TTNNModule.__call__` through
`TracedRun.module_run`.

### How `TracedRun` works

`TracedRun` (a subclass of `LightweightRun`) adds a per-module trace cache keyed by
`(module_name, input_signatures, kwargs_signatures)`. On the first call to any
`@trace_enabled` module:

1. Persistent input buffers are allocated on device.
2. `module.forward` is called once with those buffers (warm-up / compile run).
3. `ttnn.begin_trace_capture` is called.
4. `module.forward` is called again.
5. `ttnn.end_trace_capture` is called and a `TraceEntry` is stored in `_trace_cache`.

On subsequent calls: inputs are copied into the persistent buffers with `ttnn.copy`, then
`ttnn.execute_trace` replays the captured command stream.

Modules that are not `@trace_enabled`, or any module called while another traced module
execution is already in progress (`_TRACE_RUNNING = True`), fall back to `self.forward(...)`
without tracing. The `_TRACE_RUNNING` flag is set at the start of `TracedRun.module_run`
for any trace-enabled module and held for the duration of that module's run — whether it is
capturing a new trace or replaying a cached one. It is not limited to the capture phase.

### Current limitation of the `TRACED` mode

`TracedRun` operates at **individual module granularity** — one trace per `TTNNModule`
instance per input shape. The TT Transformers `Generator` captures traces that span the
**entire transformer forward pass** (all layers combined), which is a much larger and more
effective unit. Symbiote's per-module traces reduce Python overhead within each module call
but still pay the HF `generate()` Python loop overhead between modules.

---

## 4. KV cache handling

`test_llama.py` passes `use_cache=True` to `model.generate()`. This activates HuggingFace's
standard dynamic KV cache (`DynamicCache` in recent HF versions), which stores past
key/value states as regular `torch.Tensor` lists that grow each step. No `StaticCache` is
pre-allocated.

The `LlamaAttention` replacement module in
`models/experimental/tt_symbiote/modules/attention.py` receives `past_key_value` as a
keyword argument (the HF `Cache` object). `run_config.py`'s `NormalRun.module_run` has
special handling for this: `past_key_value` kwargs are excluded from the
`TorchTTNNTensor` transformation and passed through raw.

This means the KV cache tensors do not go through TTNN on each step; they are read from CPU
memory by the attention module. There is no paged attention, no block allocation, and no
`page_table`.

---

## 5. Python dispatch overhead

Each call to a replaced `TTNNModule` through `TorchTTNNTensor` dispatch involves:

1. `__torch_dispatch__` intercepts every `aten` op on the wrapped tensor.
2. `can_dispatch_to_ttnn` queries whether the op has a TTNN implementation.
3. For ops dispatched to TTNN, inputs are unwrapped from `TorchTTNNTensor`, moved to device
   if not already there, the TTNN op is called, and the output is wrapped back into a
   `TorchTTNNTensor`.
4. `DispatchManager.record_timing` records a timing entry for every op.

In `NormalRun.module_run` (the default mode), `preprocess_weights()` and
`move_weights_to_device()` are called on every `TTNNModule.__call__`. These are guarded by
`_preprocessed_weight` and `_weights_on_device` flags, so they are no-ops after the first
call — but the flag checks and function call overhead still occur per decode step.

The `TT_SYMBIOTE_RUN_MODE=TRACED` path reduces the number of TTNN dispatches per module
call after the first, but the HF `generate()` Python loop still calls each `TTNNModule`
once per generated token, re-entering `TracedRun.module_run` for every module on every
step. The overhead adds up across the full stack (embedding, attention, MLP, normalization,
LM head) for each of hundreds or thousands of decode steps.

---

## 6. Gaps relative to TT Transformers

| Capability | TT Transformers `Generator` | Current Symbiote |
|------------|----------------------------|-----------------|
| Prefill / decode phase distinction | Explicit `Mode.PREFILL` / `Mode.DECODE`; `switch_mode` called per step | None; HF `generate()` calls `forward` uniformly |
| Full-model trace capture | Single trace spans entire transformer forward per decode step | Per-module traces in `TracedRun`; decode loop remains in Python |
| Paged KV cache | `TTNNPagedAttentionKVCache` with `page_table` tensor; paged scatter/gather in attention | HF `DynamicCache`; KV stored as CPU tensors |
| Warmup sweep | `warmup_model_prefill` sweeps all supported sequence lengths before first request | No warmup; first request pays compilation latency |
| Sampling on device | `SamplingParams`; optional split-sampling trace for temperature/top-p | CPU sampling via `torch.argmax` / `torch.softmax` inside HF `generate()` |
| Data-parallel sharding | `self.data_parallel`; `torch.chunk` splits batch across shards | No data-parallel decode; single model instance |
| Decode trace reset | `prev_page_table` comparison; `reset_batch` on mode switch | Not applicable |

---

**Next:** [`integration_roadmap.md`](./integration_roadmap.md)
