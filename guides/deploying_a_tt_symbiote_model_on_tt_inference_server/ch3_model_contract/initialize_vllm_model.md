# `initialize_vllm_model()`

## Method Signature

```python
from typing import Self  # Python 3.11+; use typing_extensions.Self on earlier versions

@classmethod
def initialize_vllm_model(
    cls,
    hf_config,              # transformers.PretrainedConfig loaded from the HF model repo
    mesh_device,            # ttnn.MeshDevice already opened by tt_worker.py
    max_batch_size,         # int; derived from DeviceModelSpec.max_concurrency × tt_data_parallel
    max_seq_len,            # int; derived from DeviceModelSpec.max_context
    n_layers=None,          # int or None; allows partial-layer loading for testing
    tt_data_parallel=1,     # int; data parallel degree
    optimizations: str = "performance",  # "performance" or "accuracy"
) -> "Self":
    # Return type is Self — i.e. the same class that cls refers to.
    # In concrete implementations, replace "Self" with the actual class name,
    # e.g. -> "TTMistralForCausalLM".  "-> cls" is not valid Python.
```

`TTModelLoader` calls this classmethod after resolving the model class from `ModelRegistry`. The return value becomes the model instance that the worker holds for the lifetime of the server process. Every subsequent `prefill_forward()` and `decode_forward()` call is dispatched to that instance.

## What the Method Must Do

`initialize_vllm_model()` is responsible for the entire model construction and weight loading sequence. By the time it returns, the instance must be fully ready to receive inference requests — no lazy initialization is permitted after this point. Concretely, the method must:

1. **Instantiate the model.** Create an instance of `cls` using `hf_config` to determine architecture hyperparameters (number of layers, hidden dimension, number of attention heads, vocabulary size, etc.). If `n_layers` is not `None`, limit layer construction to the first `n_layers` transformer blocks; this is used in testing and CI to reduce load time.

2. **Preprocess weights.** Call `model.preprocess_weights()` (or the equivalent on your class) to perform any offline transformations required before the weights can be moved to device — transposing specific weight matrices, fusing QKV projections, converting dtypes, etc.

3. **Move weights to device.** Call `model.move_weights_to_device(mesh_device)` to allocate and populate TTNN tensors on the provided `mesh_device`. This is the step that places weight data into device DRAM.

4. **Run warmup sweeps.** If your model uses TTNN program caching or trace capture, run the necessary warmup calls here so that subsequent inference calls hit the cache from the very first request.

5. **Return the instance.** Return the fully initialized `cls` instance. Do not return `None` or a partially initialized object.

## The Mesh Device Is Already Open

`tt_worker.py` calls `ttnn.open_mesh_device()` once during worker startup and manages the device lifetime for the entire process. By the time `initialize_vllm_model()` is called, `mesh_device` is already open and ready for use.

Your implementation must **not** call `ttnn.open_mesh_device()` or `ttnn.close_mesh_device()`. Doing so will either raise a runtime error (double-open) or tear down the device that the worker is managing, corrupting the entire worker process. Treat `mesh_device` as a borrowed reference: use it freely, but do not take ownership of its lifecycle.

## The `optimizations` Parameter

The `optimizations` parameter is passed through to `override_tt_config["optimizations"]` and controls the fidelity/performance trade-off for TTNN operations. It accepts exactly two values:

| Value | Behavior |
|---|---|
| `"performance"` | `bfloat8` weight precision, `HiFi2` math fidelity, trace capture enabled. Maximizes throughput at the cost of some numeric precision. |
| `"accuracy"` | `bfloat16` weight precision, `HiFi4` math fidelity, trace capture disabled. Maximizes numeric fidelity; useful for evaluation and debugging. |

Read this parameter early in `initialize_vllm_model()` and use it to select the appropriate TTNN dtype constants and math fidelity settings before any tensors are allocated. Switching precision mid-initialization — for example, allocating some weights in `bfloat16` and others in `bfloat8` — produces undefined behavior in the paged attention kernel.

## `max_batch_size` and `max_seq_len`

`max_batch_size` is computed by `TTModelLoader` as `DeviceModelSpec.max_concurrency × tt_data_parallel`. Use it to pre-allocate any fixed-size TTNN buffers that depend on batch dimension — for example, KV cache scratch buffers or rotary embedding tables sized to the maximum sequence position.

`max_seq_len` is taken directly from `DeviceModelSpec.max_context`. Use it to bound position embedding tables and any decode-time position ID buffers.

Both values are integers. Your implementation should treat them as hard upper bounds: the runtime guarantees that no batch or sequence will exceed these sizes.

## Minimal Implementation Pattern for TT Symbiote Models

```python
@classmethod
def initialize_vllm_model(
    cls,
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    tt_data_parallel=1,
    optimizations: str = "performance",
) -> "TTMistralForCausalLM":
    # 1. Select precision and fidelity from the optimizations flag.
    if optimizations == "performance":
        weight_dtype = ttnn.bfloat8_b
        math_fidelity = ttnn.MathFidelity.HiFi2
        use_trace = True
    else:
        weight_dtype = ttnn.bfloat16
        math_fidelity = ttnn.MathFidelity.HiFi4
        use_trace = False

    # 2. Build the model configuration object from hf_config.
    tt_config = TTMistralConfig.from_hf_config(
        hf_config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        weight_dtype=weight_dtype,
        math_fidelity=math_fidelity,
    )

    # 3. Instantiate the model (weights are not yet on device).
    model = cls(tt_config, mesh_device)

    # 4. Preprocess weights (CPU-side transformations).
    model.preprocess_weights()

    # 5. Move weights to the provided mesh_device.
    model.move_weights_to_device(mesh_device)

    # 6. Run warmup sweeps for trace/program cache if applicable.
    if use_trace:
        model.capture_trace(max_batch_size=max_batch_size)

    return model
```

---

**Next:** [forward_interface.md](./forward_interface.md)
