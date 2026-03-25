# Chapter 2 — Core Abstractions

Chapter 2 covers the three foundational building blocks of TT Symbiote: the module base class, the hybrid tensor type, and the dispatcher system that connects them.

## Overview

TT Symbiote sits between PyTorch and TTNN. It lets a developer write a model in familiar PyTorch idioms while transparently routing compute to Tenstorrent hardware. Three abstractions make this possible:

1. **`TTNNModule`** — the base class every TTNN-accelerated layer inherits from. It replaces `torch.nn.Module` and owns the lifecycle of weights: preprocessing, placement on device, and deallocation after use.

2. **`TorchTTNNTensor`** — a `torch.Tensor` subclass that carries either a plain PyTorch tensor (`elem`) or a live TTNN tensor (`ttnn_tensor`) as its backing store. Because it is a real `torch.Tensor`, it passes through any framework code that inspects types, but its `__torch_dispatch__` hook intercepts every ATen operation and routes it through the dispatcher.

3. **The Dispatcher System** — a pluggable registry that decides, for each intercepted ATen operation, whether to execute it on TTNN hardware or fall back to PyTorch. The active dispatcher is selected by the `TT_SYMBIOTE_DISPATCHER` environment variable or by an explicit call to `set_dispatcher`.

## How the three abstractions fit together

```
User code calls module(input)
        │
        ▼
TTNNModule.__call__
  └─► TENSOR_RUN_IMPLEMENTATION.module_run(self, *args, **kwargs)
        │
        ▼
module.forward(TorchTTNNTensor, ...)
        │
        ▼
Any torch op on a TorchTTNNTensor
  └─► TorchTTNNTensor.__torch_dispatch__
        │
        ├─► can_dispatch_to_ttnn(func_name, args, kwargs)  ──True──► dispatch_to_ttnn(...)
        │                                                                      │
        │                                                               TTNN hardware
        │
        └─► False ──► PyTorch fallback (CPU/eager)
```

A module's weights are stored as `TorchTTNNTensor` instances (or plain TTNN tensors). Before a forward pass the weights have already been preprocessed and placed on device. After the pass, `deallocate_weights` can reclaim device SRAM. The dispatcher is consulted at every operation boundary, so a single forward pass may mix TTNN and PyTorch execution transparently.

## Files in this chapter

| File | Topic |
|---|---|
| [`ttnn_module.md`](./ttnn_module.md) | `TTNNModule` base class, weight lifecycle, and decorators |
| [`torch_ttnn_tensor.md`](./torch_ttnn_tensor.md) | `TorchTTNNTensor` dual-backing tensor and dispatch hook |
| [`dispatcher_system.md`](./dispatcher_system.md) | Dispatcher registry, selection, and custom dispatchers |
