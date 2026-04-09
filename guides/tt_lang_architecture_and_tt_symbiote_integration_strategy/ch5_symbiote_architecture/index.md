# Chapter 5 --- TT-Symbiote Architecture and Pain Points

TT-Symbiote is Tenstorrent's framework for **transparent PyTorch-to-TTNN acceleration**. Its central promise is simple: wrap a standard PyTorch model, and TT-Symbiote intercepts tensor operations at the ATen dispatch level, routing them to TTNN device kernels whenever possible. No model rewrite required.

In practice, achieving that transparency demands three interlocking subsystems --- a module lifecycle that manages weight preprocessing and device placement, a tensor subclass that hijacks PyTorch's dispatch machinery, and a hand-written registry of ATen-to-TTNN handler functions. Each subsystem carries specific pain points that TT-Lang custom kernels could address.

This chapter dissects all three.

## Architecture Overview

```
                          PyTorch Model (nn.Module)
                                 |
                    .from_torch() conversion
                                 |
                    +------------------------+
                    |   TTNNModule subclass   |   <-- Module lifecycle
                    |  preprocess -> move ->  |       (ch5: ttnn_module_lifecycle.md)
                    |  forward -> deallocate  |
                    +------------------------+
                                 |
                         forward() calls
                    ttnn.linear, ttnn.matmul, ...
                                 |
              +------------------+------------------+
              |                                     |
    Direct TTNN calls                   PyTorch ATen ops on
    (inside TTNNModule.forward)         TorchTTNNTensor inputs
              |                                     |
              v                          +----------v-----------+
        TTNN Device                      | __torch_dispatch__   |
                                         | TorchTTNNTensor      |
                                         +----------+-----------+
                                                    |
                                         +----------v-----------+
                                         | Dispatcher Registry   |
                                         | _get_func_to_ttnn_   |
                                         | compatible()          |
                                         | ~80 ATen op handlers  |
                                         +----------+-----------+
                                                    |
                                          +---------v----------+
                                          | handle_mul         |
                                          | handle_add         |
                                          | handle_softmax     |
                                          | handle_sdpa  ...   |
                                          +--------------------+
                                                    |
                                              TTNN Device
```

The architecture splits into two acceleration paths:

1. **Module path** --- `TTNNModule` subclasses (e.g., `TTNNLinear`, `TTNNRMSNorm`) replace entire PyTorch layers with hand-optimized TTNN implementations. The module author controls weight layout, memory config, and compute kernel selection.

2. **Dispatch path** --- For operations that occur *between* modules (residual adds, activation functions invoked by PyTorch code, reshapes), `TorchTTNNTensor.__torch_dispatch__` intercepts ATen ops and routes them through the dispatcher registry to TTNN handlers.

Both paths eventually execute on Tenstorrent hardware via TTNN, but they differ sharply in how much control the developer has and how much boilerplate is required.

## Chapter Contents

| File | Description |
|------|-------------|
| [`ttnn_module_lifecycle.md`](./ttnn_module_lifecycle.md) | TTNNModule base class: 3-phase lifecycle, boilerplate burden, device architecture constraints, distributed config |
| [`dispatch_system.md`](./dispatch_system.md) | TorchTTNNTensor and the dispatcher registry: how ATen ops are intercepted and mapped to TTNN |
| [`module_catalog.md`](./module_catalog.md) | Complete inventory of all TTNNModule subclasses across activation, attention, linear, MoE, normalization, RoPE, conv, embedding, tensor, and decoder layer categories |

## Key Takeaways

- **TT-Symbiote has two acceleration paths** (module-level and dispatch-level), each with distinct trade-offs between control and maintenance burden.

- **The 3-phase module lifecycle** (`preprocess_weights` / `move_weights_to_device` / `forward` + `deallocate_weights`) is powerful but imposes 3--4 method overrides per module, creating significant boilerplate that scales with the number of module variants.

- **The dispatch registry is hand-written**: ~80 ATen op handlers, each 10--60 lines of Python, mapping one ATen op to its TTNN equivalent. Every new ATen op that PyTorch models use requires a new handler. This is the single largest scaling bottleneck in TT-Symbiote.

- **Boilerplate patterns repeat across handlers**: tensor type checking, `_prepare_binary_inputs`, `ensure_tile_layout`, `_cleanup_tensors`, `TorchTTNNTensor` wrapping. These patterns are candidates for TT-Lang abstraction.

- **Device architecture constraints** (`@run_on_devices`, `DeviceArch`) and **distributed tensor config** (`DistributedConfig`, `DistributedTensorConfig`, `CCLManagerConfig`) add further per-module complexity that a higher-level language could encapsulate.

- **These pain points directly motivate TT-Lang integration**: custom kernels could auto-generate dispatch handlers, reduce module boilerplate through declarative weight specs, and provide compile-time validation of device constraints.
