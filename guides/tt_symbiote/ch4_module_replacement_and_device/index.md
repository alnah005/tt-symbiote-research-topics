# Chapter 4 — Module Replacement and Device Setup

This chapter covers the two-phase initialization sequence that converts a PyTorch model into a TT Symbiote–accelerated model: replacing PyTorch submodules with `TTNNModule` instances, and then binding those instances to a TTNN device so that weights can be moved and inference can begin.

---

## Overview

Before any TTNN kernel can execute, two things must happen:

1. **Module replacement** — every `nn.Module` subclass that has a TTNN counterpart is swapped out in-place inside the model graph. This is driven by `register_module_replacement_dict` from `utils/module_replacement.py`.
2. **Device setup** — the device handle is propagated to every `TTNNModule` in the graph, forward-timing hooks are installed on every `nn.Module`, and (for multi-device meshes) distributed state is initialized. This is driven by `set_device` from `utils/device_management.py`.

Weight preprocessing and the actual weight-to-device transfer happen between or after these two steps, using the module dict returned by `register_module_replacement_dict`.

---

## Typical Setup Sequence

The table below shows the canonical order of operations for initializing a model:

| Step | Call | Purpose |
|------|------|---------|
| 1 | `register_module_replacement_dict(model, nn_to_ttnn_dict, model_config, exclude_replacement)` | Replace PyTorch submodules with `TTNNModule` instances; obtain the module dict |
| 2 | `module.preprocess_weights()` for each module in the returned dict | Run any per-module weight transformation (layout changes, quantization, etc.) while weights are still on CPU |
| 3 | `set_device(model, ttnn_device, device_init, **kwargs)` | Bind each `TTNNModule` to the device; initialize `DistributedConfig` for multi-device meshes; install forward hooks |
| 4 | `module.move_weights_to_device()` for each module in the returned dict | Upload preprocessed weights to the TTNN device |
| 5 | `model(inputs)` | Run inference |

> **Note:** Steps 2 and 4 (weight preprocessing and weight transfer) are the caller's responsibility; they are not called automatically by either `register_module_replacement_dict` or `set_device`. The module dict returned in step 1 is intended as the driver for these loops.

---

## Minimal Example

```python
import ttnn
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device

# 1. Build or load a standard PyTorch model
model = MyPyTorchModel()

# 2. Define the replacement mapping
nn_to_ttnn_dict = {
    MyAttentionLayer: TTAttentionLayer,
    MyMLPLayer:       TTMLPLayer,
}

# 3. Replace modules and get the module dict
module_dict = register_module_replacement_dict(
    model,
    nn_to_ttnn_dict,
    model_config={"dtype": ttnn.bfloat16},
    exclude_replacement={"model.layers.0.attention"},  # keep first attention layer as PyTorch
)

# 4. Preprocess weights (CPU-side)
for module in module_dict.values():
    module.preprocess_weights()

# 5. Bind device (installs hooks; initializes distributed state if multi-device)
device = ttnn.open_mesh_device(...)
set_device(model, device)

# 6. Upload weights
for module in module_dict.values():
    module.move_weights_to_device()

# 7. Run inference
output = model(inputs)
```

---

## File Reference

| File | Primary contents |
|------|----------------|
| `utils/module_replacement.py` | `register_module_replacement_dict`, `register_module_replacement_dict_with_module_names`, `initialize_module` |
| `utils/device_management.py` | `set_device`, `DeviceInit`, `_initialize_module_on_device` |
| `core/module.py` | `TTNNModule` base class (`from_torch`, `set_model_config`, `preprocess_weights`, `move_weights_to_device`, `to_device`, `set_device_state`) |
| `core/run_config.py` | `DistributedConfig`, `DistributedTensorConfig`, `DispatchManager` |
| `utils/graph_visualization.py` | `draw_model_graph` |

---

## Chapter Contents

- [Module Replacement](module_replacement.md)
- [Device Setup](device_setup.md)

---

**Navigation:** [Previous — Chapter 3: Run Modes and Timing](../ch3_run_modes_and_timing/index.md)
