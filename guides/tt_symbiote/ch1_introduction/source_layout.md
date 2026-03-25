# source_layout.md

TT-Symbiote lives inside the `tt-metal` repository and is organised into four top-level subdirectories whose boundaries map cleanly onto distinct concerns: runtime mechanics, hardware-accelerated layer implementations, shared helpers, and validation.

All paths below are relative to the `tt-metal` repository root. The full prefix for every import is `models.experimental.tt_symbiote`.

## Repository tree

```
models/experimental/tt_symbiote/
├── core/
│   ├── module.py
│   ├── tensor.py
│   ├── dispatcher.py
│   ├── torch_dispatcher.py
│   ├── run_config.py
│   ├── utils.py
│   └── dispatchers/
│       ├── __init__.py
│       ├── dispatcher_config.py
│       ├── default_dispatcher.py
│       ├── debug_dispatcher.py
│       ├── cpu_dispatcher.py
│       └── tensor_operations_dispatcher.py
├── modules/
│   ├── linear.py
│   ├── attention.py
│   ├── normalization.py
│   ├── activation.py
│   ├── conv.py
│   └── tensor.py
├── utils/
│   ├── module_replacement.py
│   └── device_management.py
└── tests/
    ├── test_vit.py
    ├── test_llama.py
    ├── test_resnet50.py
    └── ... (one file per supported model)
```

## Top-level subdirectories

### `core/`

The runtime machinery. Nothing in this directory is specific to any model or operation type.

| File | Purpose |
|------|---------|
| `module.py` | `TTNNModule` base class — three-phase weight lifecycle (preprocess → move to device → deallocate). Also defines the `module_name` property, the `deallocate_weights_after` decorator, the `DeviceArch` enum, and the `run_on_devices` decorator for restricting a `forward` method to specific hardware. |
| `tensor.py` | `TorchTTNNTensor`. The dual-representation tensor wrapper that implements `__torch_dispatch__` and exposes `.to_torch` and `.to_ttnn` accessors. |
| `dispatcher.py` | Entry point for TTNN operation dispatch. Calls `get_active_dispatcher()` and routes `aten` operations to their TTNN equivalents. |
| `torch_dispatcher.py` | Fallback dispatch path. Routes operations to PyTorch when TTNN dispatch is unavailable or when operating in CPU mode. |
| `run_config.py` | Run mode classes (`NormalRun`, `NormalRunWithFallback`, `SELRun`, `DPLRun`, `DPLRunNoErrorProp`, `CPU`, `LightweightRun`, `TracedRun`) and the `_RUN_MODE_REGISTRY` that maps environment variable strings to those classes. Also defines `DistributedConfig`, `DistributedTensorConfig`, and `CCLManagerConfig` for multi-device setups. |
| `utils.py` | Dtype conversion utilities (`torch_dtype_to_ttnn_dtype`, `ttnn_dtype_to_torch_dtype`, `TORCH_TO_TTNN` map) and the `tree_map` helper used throughout `run_config.py`. |
| `dispatchers/` | Pluggable dispatcher implementations. `dispatcher_config.py` owns the `_DISPATCHER_REGISTRY` and the `get_active_dispatcher()` function. Each sibling module (`default_dispatcher.py`, `debug_dispatcher.py`, `cpu_dispatcher.py`) must export `can_dispatch_to_ttnn` and `dispatch_to_ttnn`. |

### `modules/`

Ready-to-use TTNN replacements for common PyTorch layer types. Every class in this directory inherits from `TTNNModule` and provides a `from_torch` classmethod so `register_module_replacement_dict` can instantiate it from an existing PyTorch layer.

| File | Key classes |
|------|------------|
| `linear.py` | `TTNNLinear`, `TTNNLinearLLama` (bfloat8_b + auto-deallocation via `deallocate_weights_after`), `TTNNLinearLLamaBFloat16`, `TTNNLinearGelu` |
| `attention.py` | `TTNNSelfAttention`, `TTNNSDPAAttention`, `TTNNFusedQKVSelfAttention`, `TTNNWhisperAttention`, `TTNNViTSelfAttention` (deprecated) |
| `normalization.py` | `TTNNLayerNorm` |
| `activation.py` | `TTNNSilu`, `TTNNReLU`, `TTNNGelu` |
| `conv.py` | `TTNNConv2dNHWC`, `TTNNConv2dBNNHWC`, `TTNNConv2dBNActivationNHWC`, `TTNNBottleneck`, `TTNNMaxPool2dNHWC`, `TTNNUpsampleNHWC`, `TTNNPatchEmbedding`, `TTNNViTEmbeddings` |
| `tensor.py` | `TTNNPermute`, `TTNNReshape`, `TTNNAdd` |

### `utils/`

Two thin utilities that sit above `core/` and are called at model setup time.

| File | Purpose |
|------|---------|
| `module_replacement.py` | `register_module_replacement_dict` — recursively walks the `nn.Module` tree and swaps each mapped type for its TTNN equivalent. Accepts an `exclude_replacement` set of module name strings for selective bypass. |
| `device_management.py` | `set_device` — walks the module tree after replacement and calls `TTNNModule.to_device` on every `TTNNModule` instance, propagating the TTNN device handle downward. |

### `tests/`

One pytest file per validated model. Each file exercises a full inference pipeline end-to-end, relying on pytest fixtures (provided by the tt-metal test infrastructure) for device creation. The test suite covers CNNs (`test_resnet50.py`, `test_conv.py`), transformers (`test_vit.py`, `test_llama.py`, `test_whisper3.py`), multimodal models (`test_openvla.py`, `test_hunyuan_video.py`), and debugging modes (`test_dpl.py`).

## Key environment variables

Three environment variables control runtime behaviour without requiring any code changes.

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `TT_SYMBIOTE_RUN_MODE` | See table below | `NORMAL` | Selects the run mode class from `_RUN_MODE_REGISTRY` in `core/run_config.py`. |
| `TT_SYMBIOTE_DISPATCHER` | `DEFAULT`, `DEBUG`, `CPU` | `CPU` | Selects the dispatcher module from `_DISPATCHER_REGISTRY` in `core/dispatchers/dispatcher_config.py`. When unset, `get_active_dispatcher()` falls back to the `CPU` dispatcher. |
| `MESH_DEVICE` | `N150`, `N300`, `T3K`, `TG`, `P150`, `P300`, `P150x4`, `P150x8`, `BHGLX` | — | Declares the target hardware topology. Read by the `run_on_devices` decorator in `core/module.py` via `MeshShapeToDeviceArch` to enforce architecture restrictions on individual `forward` methods. |

> **Warning:** `TT_SYMBIOTE_RUN_MODE` is read once at module import time by `get_tensor_run_implementation()`. If you change it after `core/module.py` has been imported, the in-process `TENSOR_RUN_IMPLEMENTATION` binding will not update. Always set environment variables before importing any TT-Symbiote module.

> **Note:** When `TT_SYMBIOTE_DISPATCHER` is not set, the framework operates in CPU-only mode by default. You must explicitly export `TT_SYMBIOTE_DISPATCHER=DEFAULT` to route operations to the TTNN hardware backend. Certain run modes (e.g., `LIGHTWEIGHT`) additionally require `TT_SYMBIOTE_DISPATCHER=CPU` and will assert otherwise.

### Run modes

| `TT_SYMBIOTE_RUN_MODE` value | Class in `run_config.py` | Description |
|------------------------------|--------------------------|-------------|
| `NORMAL` | `NormalRun` | Default TTNN execution. Operations are dispatched to TTNN; failures raise exceptions. |
| `NORMAL_WITH_FALLBACK` | `NormalRunWithFallback` | TTNN with automatic per-operation fallback to PyTorch on errors. |
| `SEL` | `SELRun` | Segment Each Layer. PyTorch receives copied torch tensors as input; TTNN receives the same inputs converted to TTNN tensors. Their outputs are compared with PCC. |
| `DPL` | `DPLRun` | Debug Per Layer. Runs both TTNN and PyTorch in parallel for every layer and compares outputs with PCC. TTNN receives the accumulated prior-layer outputs as inputs (numerical errors propagate across layers). Use `DPL_NO_ERROR_PROP` if you need to isolate a single layer's error without compounding from prior layers. |
| `DPL_NO_ERROR_PROP` | `DPLRunNoErrorProp` | Like DPL, but both PyTorch and TTNN receive independent fresh copies of the original inputs, preventing accumulated TTNN error from propagating across layers. |
| `CPU` | `CPU` | CPU-only execution. No TTNN operations are performed. Useful for validating model logic without hardware. |
| `LIGHTWEIGHT` | `LightweightRun` | Lightweight execution path; requires `TT_SYMBIOTE_DISPATCHER=CPU`. |
| `TRACED` | `TracedRun` | Trace-based execution for reduced dispatch overhead on repeated calls. Requires `TT_SYMBIOTE_DISPATCHER=CPU`. |

## Supported device architectures

The `DeviceArch` enum in `models/experimental/tt_symbiote/core/module.py` lists every hardware target the framework can address. The string value in each member is the canonical identifier used by `MeshShapeToDeviceArch` when resolving the `MESH_DEVICE` environment variable.

| `DeviceArch` member | String value | Hardware |
|---------------------|-------------|---------|
| `DeviceArch.N150` | `n150` | Wormhole N150 single-chip board |
| `DeviceArch.N300` | `n300` | Wormhole N300 dual-chip board |
| `DeviceArch.T3K` | `t3k_wh` | T3000 Galaxy (8-chip Wormhole mesh) |
| `DeviceArch.TG` | `gx_wh` | TG Galaxy (32-chip Wormhole mesh) |
| `DeviceArch.P150` | `p150` | Blackhole P150 single-chip board |
| `DeviceArch.P300` | `p300` | Blackhole P300 dual-chip board |
| `DeviceArch.P150x4` | `p150x4` | Four-chip Blackhole P150 mesh |
| `DeviceArch.P150x8` | `p150x8` | Eight-chip Blackhole P150 mesh |
| `DeviceArch.BHGLX` | `bhglx` | Blackhole Galaxy mesh |

The `run_on_devices` decorator uses this enum to gate `forward` methods at call time:

```python
from models.experimental.tt_symbiote.core.module import run_on_devices, DeviceArch

class TTNNCustomLayer(TTNNModule):
    @run_on_devices(DeviceArch.N300, DeviceArch.T3K)
    def forward(self, x):
        ...
```

If `MESH_DEVICE` does not match one of the allowed architectures, a `RuntimeError` is raised before the forward pass executes.

---

**Next:** [Chapter 2 — Core Abstractions](../ch2_core_abstractions/index.md)
