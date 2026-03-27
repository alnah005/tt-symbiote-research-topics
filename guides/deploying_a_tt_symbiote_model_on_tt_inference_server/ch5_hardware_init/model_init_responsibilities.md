# Model Initialization Responsibilities

This file defines exactly what a TT Symbiote model implementation must do inside `initialize_vllm_model`, what it must not do, and how to handle multi-chip mesh devices and KV cache allocation correctly.

## What `initialize_vllm_model` Must Do

`initialize_vllm_model` receives a fully-opened `mesh_device` handle (a `ttnn.MeshDevice` object) and is responsible for preparing the model to serve inference requests. The required steps are:

1. **Load weights onto `mesh_device`.** Convert host-side weight tensors to `ttnn` device tensors and upload them. All persistent weight buffers must live on the device before `initialize_vllm_model` returns.
2. **Initialize TTNN tensor constants.** Any pre-computed tensors that are fixed for the lifetime of the model — cosine and sine matrices for RoPE positional encodings, attention mask templates, lookup tables — must be allocated and transferred to `mesh_device` here.
3. **Run compilation warmup passes.** The first forward pass through a TTNN kernel incurs JIT compilation overhead. Run one or more warmup sweeps at representative sequence lengths before returning so that all kernels are compiled and cached. This prevents the first real inference request from timing out.
4. **Return the initialized model instance.** The worker stores the returned object and passes it to `forward()` on every decode step.

A minimal sketch of a compliant `initialize_vllm_model`:

```python
class MyTTNNModel:
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        n_layers=None,
        tt_data_parallel: int = 1,
        optimizations: str = "performance",
    ) -> "MyTTNNModel":
        # 1. Load weights from disk (host side)
        state_dict = load_state_dict(hf_config)

        # 2. Build model, preprocess weights on host
        model = cls(hf_config, mesh_device)
        model.preprocess_weights(state_dict)

        # 3. Upload weights to device
        model.move_weights_to_device(mesh_device)

        # 4. Initialize constant tensors (e.g., RoPE matrices)
        model.initialize_constants(mesh_device)

        # 5. Warmup compilation
        model.warmup(mesh_device)

        return model
```

## What `initialize_vllm_model` Must Not Do

The following actions are prohibited inside `initialize_vllm_model`:

- **Do not call `ttnn.open_mesh_device()`.** The device is already open. Calling this again will attempt to allocate a second set of device resources for the same physical hardware, which is either a no-op with a stale handle or a hard fault depending on the `ttnn` version.
- **Do not call `ttnn.close_mesh_device()`.** Closing the device mid-initialization destroys all device memory including the worker's own allocations. The worker will crash when it next accesses the device.
- **Do not call `ttnn.synchronize_device()` gratuitously.** `synchronize_device` is a blocking host–device barrier. During initialization it is sometimes needed to ensure a weight upload completes before a dependent constant is computed. If correctness requires it, it is permitted — but do not use it as a defensive "flush everything" at the end of init. It adds latency and can mask ordering bugs.
- **Do not modify `TT_VISIBLE_DEVICES`.** This environment variable is owned by the server workflow. The model is never in a position to know which physical device IDs are correct for the current deployment.
- **Do not import and call `set_fabric()` directly.** Fabric topology is configured globally by the worker before the device is opened. Calling `set_fabric()` from inside the model after the device is already open is a no-op at best and corrupts the fabric state at worst.
- **Do not call `deallocate_weights`** (or any equivalent host-side weight deallocation) inside `initialize_vllm_model` — not before warmup, not after warmup, and not before returning. The server may need to re-read weights after initialization completes (e.g., for weight reloading or multi-worker synchronization), and premature deallocation will corrupt that path. Weight deallocation, when used at all, is a per-forward-call optimization managed by the decorator pattern in the model's forward methods, not an init-time step.

## Multi-Chip TT Symbiote Models

When `MESH_DEVICE` is set to `"T3K"` or another multi-chip configuration, the `mesh_device` argument is already a multi-chip `ttnn.MeshDevice` covering all allocated chips. The model does not need to do anything special to construct a multi-chip mesh — it receives one.

Before calling `move_weights_to_device`, multi-chip TT Symbiote models must pass the `mesh_device` to their distributed configuration setup so that tensor sharding and collective operations know the topology:

```python
# Pattern A: models using TTNNModule
model.set_device_state(mesh_device)
model.move_weights_to_device(mesh_device)

# Pattern B: models using an explicit DeviceState / DistributedConfig
device_state = DeviceState(mesh_device)
distributed_config = DistributedConfig.from_device_state(device_state)
model = MyTTNNModel(model_config, device_state, distributed_config)
model.preprocess_weights(state_dict)
model.move_weights_to_device(mesh_device)
```

The specific API depends on the TT Symbiote generation. Consult the model's own `README` or `model_config.py` for the correct pattern. The invariant that always holds is: the `mesh_device` passed to `initialize_vllm_model` is the authoritative device handle and must be used everywhere the model interacts with device memory.

### Verifying the Mesh Shape

As a defensive check, it is good practice to assert the expected mesh shape at the start of `initialize_vllm_model` so that configuration mismatches are caught early with a clear error message:

```python
expected_rows, expected_cols = model_config["mesh_shape"]  # e.g., (1, 8)
actual_shape = mesh_device.shape
assert actual_shape == ttnn.MeshShape(expected_rows, expected_cols), (
    f"Expected mesh shape ({expected_rows}, {expected_cols}), "
    f"got {actual_shape}. Check MESH_DEVICE."
)
```

## KV Cache Allocation

KV cache tensors are **not** allocated inside `initialize_vllm_model`. This is a common source of confusion.

The vLLM block manager negotiates the exact number of KV cache blocks based on available device memory, the configured `gpu_memory_utilization` fraction, and the block size. That negotiation happens after `initialize_vllm_model` returns. Once the block count is known, the worker calls `allocate_kv_cache(num_blocks, ...)` on the model.

The correct implementation pattern is:

```python
class MyTTNNModel:
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1, optimizations="performance"):
        # ... weight loading, constants, warmup ...
        # Do NOT call allocate_kv_cache() here.
        return cls(...)

    def allocate_kv_cache(self, num_blocks: int, block_size: int, dtype):
        # Called by the worker after vLLM block negotiation.
        # ttnn.allocate_tensor_on_device does not exist in the ttnn API.
        # Use ttnn.zeros() with device placement and an explicit memory config.
        # Each layer requires two tensors: one for K and one for V.
        self.kv_cache = {
            layer_idx: (
                ttnn.zeros(
                    shape=[num_blocks, block_size, self.n_heads, self.head_dim],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.zeros(
                    shape=[num_blocks, block_size, self.n_heads, self.head_dim],
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
            for layer_idx in range(self.n_layers)
        }
```

If you pre-allocate KV cache inside `initialize_vllm_model`, the block manager's memory profiling step will see less free DRAM than actually exists (because you already consumed it), and will allocate fewer blocks than optimal. In the worst case it will determine that there is insufficient memory to run at all and refuse to start.

---

**Next:** [`environment_variables_reference.md`](./environment_variables_reference.md)
