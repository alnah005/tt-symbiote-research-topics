# Common Integration Errors

This file catalogs the ten most frequent integration failures you will encounter when bringing a TT Symbiote model up inside `tt-inference-server`. Each entry follows the same structure: the exact error message or observable symptom, the root cause in terms of the Symbiote/vLLM call path, and a concrete fix.

Read these as a reference. If you hit an error not listed here, the best starting point is always to trace backward from the failure site through `TTPlatform.get_model_cls()` → `ModelRegistry` → `register_tt_models()` → your model class definition.

---

## 1. `AttributeError: 'TTMyModel' object has no attribute 'initialize_vllm_model'`

**Symptom**

The worker raises `AttributeError: 'TTMyModel' object has no attribute 'initialize_vllm_model'` during model loading, before any tokens are processed.

**Root Cause**

`initialize_vllm_model` is entirely absent from the class — either the method was never defined, or its name contains a typo that does not match the exact string `initialize_vllm_model`. The worker resolves the model class from `ModelRegistry` and calls `cls.initialize_vllm_model(...)` by name. If that name does not exist on the class, Python raises `AttributeError`.

A second, less obvious variant: the model class was registered under a name that differs from the architecture key constructed by `TTPlatform` (e.g., registered as `"TTMymodel"` but looked up as `"TTMyModel"`). In that case a different class is resolved and that class may not define `initialize_vllm_model` at all.

**Fix**

Ensure the method is defined and spelled exactly `initialize_vllm_model` in your model class. Check for any typo (e.g., `initialise_vllm_model`, `initialize_vLLM_model`). Also confirm the registration name matches the architecture key exactly:

```python
# Wrong — method name is misspelled; worker cannot find it
def initialise_vllm_model(cls, hf_config, ...):
    ...

# Correct — exact name required by the Symbiote contract
@classmethod
def initialize_vllm_model(cls, hf_config, ...):
    ...
```

---

## 2. `TypeError: initialize_vllm_model() missing required positional argument: 'hf_config'`

**Symptom**

`TypeError: initialize_vllm_model() missing required positional argument: 'hf_config'` during worker startup.

**Root Cause**

The method IS present and found by Python — but it is defined as a regular instance method (`def initialize_vllm_model(self, ...)`) rather than a `@classmethod`. When the worker calls `TTMyModel.initialize_vllm_model(hf_config, mesh_device, ...)` directly on the class (not on an instance), Python passes `hf_config` into the `self` parameter. Every subsequent positional argument then shifts one position to the right, so the parameter that the caller expects to fill `hf_config` has nothing to bind to, producing the `missing required positional argument: 'hf_config'` error.

This is distinct from Error 1: the method exists and is reachable, but its calling convention is wrong because the `@classmethod` decorator is absent.

**Fix**

Add `@classmethod` and rename `self` → `cls`. After fixing, verify the full signature matches the Symbiote contract:

```python
@classmethod
def initialize_vllm_model(
    cls,
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    optimizations,
) -> "TTMyModel":
    ...
```

---

## 3. `KeyError: 'TTMyModel'` in `ModelRegistry`

**Symptom**

During model loading, a `KeyError` is raised inside `ModelRegistry` with your model's class name as the missing key.

**Root Cause**

`TTPlatform.get_model_cls()` constructs a lookup key by prepending `"TT"` to the architecture name read from `config.json`'s `"architectures"` field. For example, if `config.json` contains `"architectures": ["MyModelForCausalLM"]`, the platform looks up `"TTMyModelForCausalLM"` in the registry.

If the name you passed to `register_tt_models()` does not match this constructed key exactly — including capitalization — the lookup fails. Common mismatches:

- `"architectures"` in `config.json` has a different casing than expected (e.g., `"myModelForCausalLM"` vs `"MyModelForCausalLM"`)
- The class name in `register_tt_models()` was typed differently from the constructed key
- The model's HF `config.json` was copied from a different architecture and the `"architectures"` field was not updated

**Fix**

1. Open the model's `config.json` and note the exact string in the `"architectures"` list.
2. Confirm that your registration call uses `"TT" + that_string` as the key:
   ```python
   def register_tt_models():
       ModelRegistry.register("TTMyModelForCausalLM", TTMyModel)
   ```
3. If the HF config came from another model, update `"architectures"` in `config.json` to match your model's actual architecture name.

---

## 4. Worker Hangs After Prefill — Decode Never Starts

**Symptom**

The server accepts a request, the prefill step completes (you can see it in logs), but the process then hangs indefinitely. No decode steps are logged. `CTRL+C` shows the main thread blocked in the sampler.

**Root Cause**

`prefill_forward` (or the unified `forward` used for prefill) returned a TTNN tensor instead of a CPU PyTorch tensor. The sampler expects a CPU tensor of logits. When it receives a TTNN tensor, it attempts to iterate over it as a Python sequence, which either blocks waiting for device completion in an unexpected context or silently produces wrong results that cause the scheduler to stall.

**Fix**

Ensure that the last operation in your prefill path converts the output tensor to CPU before returning:

```python
logits = ttnn.to_torch(logits_tt)          # move to CPU
logits = logits.squeeze(0).squeeze(0)      # remove batch/mesh dims if present
return logits
```

Also verify the returned shape, which differs between the two forward modes:

- **`prefill_forward`**: must return logits for **all input token positions** so that vLLM can serve `logprobs` and `prompt_logprobs` callers. The expected shape is `(batch, seq_len, vocab_size)`.
- **`decode_forward`**: returns only the **last-position logit per sequence**. The expected shape is `(batch, vocab_size)` — one logit vector per active sequence.

Returning a decode-shaped tensor from `prefill_forward` (i.e., only the last-token slice) will silently drop logprob data and can cause `prompt_logprobs` requests to produce incorrect results.

---

## 5. `RuntimeError: TTNN device not open`

**Symptom**

`RuntimeError: TTNN device not open` raised inside your model's `initialize_vllm_model` or `forward` call.

**Root Cause**

Your model code called `ttnn.close_mesh_device()` on the mesh it received, or attempted to open a second device via `ttnn.open_mesh_device()`. The `mesh_device` passed into `initialize_vllm_model` is owned and managed entirely by the `tt-inference-server` worker process. Closing it invalidates all allocated TTNN buffers. Attempting to open a new device creates a conflict with the worker's existing device handle.

**Fix**

Never call `ttnn.close_mesh_device()` or `ttnn.open_mesh_device()` inside your model class. Treat the `mesh_device` argument as a borrowed reference: use it for all TTNN operations, but do not manage its lifecycle. Device teardown is handled by the worker after all inference is complete.

If you need to test `initialize_vllm_model` in isolation outside the worker, open and close the device in your test harness, not inside the model class.

---

## 6. `AssertionError: block_size must be 64`

**Symptom**

An assertion fires inside your KV cache or `PagedAttention` code: `AssertionError: block_size must be 64`.

**Root Cause**

The TT hardware paged attention kernel requires KV cache blocks of exactly 64 tokens. Your model code hardcoded a different block size — either directly (e.g., `BLOCK_SIZE = 32`) or via a config value that does not match the hardware requirement. When `allocate_kv_cache` is called by the worker, it passes `block_size=64`, but your internal assertion or tensor shape calculation uses a different value.

**Fix**

Do not hardcode `block_size`. The `allocate_kv_cache` method receives `block_size` as an argument from the worker. Use that argument directly:

```python
def allocate_kv_cache(self, max_batch_size, max_seq_len, block_size, ...):
    assert block_size == 64, f"TT hardware requires block_size=64, got {block_size}"
    self.kv_cache = allocate_kv_blocks(
        num_blocks=max_seq_len // block_size,
        block_size=block_size,
        ...
    )
```

Do not pass an alternative `block_size` through `vllm_args`; the value is fixed at the platform level and cannot be overridden.

---

## 7. vLLM Raises `ValueError: chunked prefill is not supported`

**Symptom**

On startup or during the first request, vLLM raises `ValueError: chunked prefill is not supported` or a similar message about chunked prefill being enabled when the platform does not support it.

**Root Cause**

`TTPlatform` sets `enable_chunked_prefill = False` in its platform capabilities, which prevents vLLM from enabling chunked prefill for standard deployments. If you see this error, it means a custom platform subclass — either your own or from an intermediate integration layer — overrode `enable_chunked_prefill` back to `True` or failed to propagate the parent class's capability flags correctly.

**Fix**

Check any platform subclass in your integration for capability overrides. Ensure that if you subclass `TTPlatform`, you either call `super().__init__()` or explicitly carry forward `self.enable_chunked_prefill = False`. Do not pass `--enable-chunked-prefill` in `vllm_args`.

---

## 8. Low Throughput — Decode Step Takes > 100ms

**Symptom**

Decode steps are slow (> 100ms per step). Throughput is well below expected tokens/second for the model size.

> **Note — `optimizations="accuracy"` is expected to be slow.** If your deployment is configured with `optimizations="accuracy"`, slow decode (>100ms) is **intentional**: that mode explicitly disables TTNN trace capture, so every decode step pays full Python dispatch overhead. Error 8 applies only when `optimizations="performance"` is set but trace capture did not initialize correctly. If you are in accuracy mode and decode is slow, that is not a misconfiguration — see [performance_tuning.md](./performance_tuning.md) for an explanation of what accuracy mode trades off. Switch to `optimizations="performance"` only after validating output correctness.

**Root Cause**

TTNN trace is not active for the decode path. TTNN trace capture (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture`) records the full sequence of TTNN operations for a given input shape and replays them with `ttnn.execute_trace()` in subsequent steps, bypassing Python dispatch overhead and enabling the device to run ahead of the host. If trace is not captured during warmup, every decode step dispatches TTNN operations one by one from Python, incurring significant host-side overhead that dominates per-step latency.

Trace capture is gated on two conditions:

1. `has_builtin_warmup=True` must be set in your model's `ImplSpec` or equivalent configuration, signaling to the worker that `initialize_vllm_model` will perform warmup internally.
2. Your `initialize_vllm_model` must actually call `ttnn.begin_trace_capture`, run a dummy decode-shaped forward pass, and call `ttnn.end_trace_capture` — storing the trace handle for use in `forward`.

**Fix**

1. Confirm `has_builtin_warmup=True` is set.
2. Inside `initialize_vllm_model`, after model weights are loaded, run a warmup loop that captures the decode trace:
   ```python
   ttnn.begin_trace_capture(mesh_device, trace_id=0)
   _ = model.decode_forward(dummy_tokens, dummy_kv_cache, ...)
   ttnn.end_trace_capture(mesh_device, trace_id=0)
   model.decode_trace_id = 0
   ```
3. In the decode path of `forward`, call `ttnn.execute_trace(mesh_device, model.decode_trace_id)` instead of re-invoking the Python forward function.

---

## 9. `OOM During KV Cache Allocation`

**Symptom**

The worker crashes with an out-of-memory error during KV cache allocation at startup, before any requests are served.

**Root Cause**

The total DRAM required for the KV cache exceeds available device memory. The KV cache size is determined by `max_context`, `max_concurrency`, the number of KV heads, head dimension, and the number of layers. Two common causes:

- `max_context` or `max_concurrency` in `DeviceModelSpec` are set higher than the device can support for this model's KV head configuration.
- `trace_region_size` in `override_tt_config` is set too large, reserving a significant fraction of DRAM for trace buffers before the KV cache is allocated, leaving insufficient space.

**Fix**

1. Reduce `max_context` or `max_concurrency` in `DeviceModelSpec` incrementally until allocation succeeds.
2. Check `override_tt_config["trace_region_size"]`. A typical starting value for a 7B model on T3K is around 200–400 MB. If it is set to several GB, reduce it.
3. As a diagnostic, set `max_concurrency=1` and `max_context=2048` to establish a baseline minimum allocation, then scale up from there.

---

## 10. `ModuleNotFoundError: No module named 'models.my_symbiote_model'`

**Symptom**

`ModuleNotFoundError: No module named 'models.my_symbiote_model'` when the worker attempts to import your model module.

**Root Cause**

The `module_path` field in `ImplSpec` specifies the Python import path used to load your model class. If this path does not match the actual importable path from the repository root — due to a missing `__init__.py`, a wrong directory name, or a typo — Python cannot find the module.

**Fix**

Verify the import path from the repo root before registering:

```bash
python -c "import models.my_symbiote_model.my_symbiote_model"
```

If this fails, diagnose:

- Check that every directory in the path contains an `__init__.py`.
- Confirm the file name and directory name exactly match the import path (case-sensitive on Linux).
- Ensure `PYTHONPATH` includes the repo root if the model lives outside the default search path.

Once the import succeeds from the command line, update `ImplSpec.module_path` to match.

---

**Next:** [performance_tuning.md](./performance_tuning.md)
