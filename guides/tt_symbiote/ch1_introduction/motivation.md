# motivation.md

TT-Symbiote exists because running a PyTorch model on Tenstorrent hardware is not a matter of changing a device string — it requires bridging two fundamentally different execution models, and TT-Symbiote does that bridging with minimal changes to existing model code.

## The mismatch between PyTorch dispatch and TTNN kernels

PyTorch uses a layered dispatch mechanism. When you call `torch.nn.Linear.forward`, PyTorch resolves that call through a chain of dispatch keys (autograd, backend, etc.) until it reaches a kernel registered for the target device (CPU or CUDA). The kernel is chosen at runtime by inspecting the tensor's device tag.

TTNN, Tenstorrent's neural network library, does not participate in that dispatch chain. Its operations — `ttnn.linear`, `ttnn.layer_norm`, `ttnn.matmul`, and so on — are standalone functions that accept `ttnn.Tensor` objects, not `torch.Tensor` objects. TTNN tensors carry their own layout constraints (e.g., `ttnn.TILE_LAYOUT`), dtype semantics (e.g., `ttnn.bfloat16`, `ttnn.bfloat8_b`), and device placement that are entirely separate from PyTorch's type system.

This creates a fundamental tension:

- **Model code** is written in terms of `nn.Module` subclasses and `torch.Tensor` arithmetic.
- **Hardware kernels** expect `ttnn.Tensor` objects with specific memory layouts already set up on the device.
- **Weights** must be converted from PyTorch format to TTNN format, transferred to the device, and eventually deallocated — a lifecycle that PyTorch has no concept of.

Bridging these two worlds naively would require rewriting every model from scratch in TTNN terms. TT-Symbiote avoids that.

## How TT-Symbiote provides transparency

TT-Symbiote's approach is _module replacement_ combined with _tensor wrapping_.

**Module replacement** walks an existing `nn.Module` tree and swaps standard PyTorch modules for TTNN-accelerated equivalents. A single utility call handles the entire tree:

```python
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.LayerNorm: TTNNLayerNorm,
}
register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

After this call, the model's `nn.Linear` instances have been replaced with `TTNNLinear` instances — but the rest of the model code is unchanged. The parent module still calls `self.linear(x)` the same way; TT-Symbiote intercepts that call inside the replacement module's `__call__` method.

**Tensor wrapping** is handled by `TorchTTNNTensor` (`core/tensor.py`). This wrapper holds both a `torch.Tensor` and a `ttnn.Tensor` representation simultaneously and implements `__torch_dispatch__` so that PyTorch operations dispatched on the wrapper are transparently redirected to TTNN equivalents. Model code that does `tensor * 2.0 + 3.0` never needs to know whether it is operating on a CPU tensor or a TTNN tensor on a Wormhole device.

**Weight lifecycle** is managed by `TTNNModule` (`core/module.py`). The base class enforces a three-phase weight lifecycle:

1. `preprocess_weights()` — convert PyTorch weights to TTNN format once, storing on host.
2. `move_weights_to_device()` — transfer the converted weights to the Tenstorrent device.
3. `deallocate_weights()` — free device memory when the weights are no longer needed.

From the user's perspective, only two setup calls are needed after loading a pretrained model:

```python
from models.experimental.tt_symbiote.utils.device_management import set_device

register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
set_device(model, ttnn_device)

model.eval()
result = model(torch.randn(1, 3, 224, 224))
```

No per-layer rewrites. No manual tensor conversion. The model's `forward` method is called exactly as before.

> **Note:** The TTNN device handle (`ttnn_device`) must be obtained separately, either through a pytest fixture provided by the tt-metal test infrastructure or by calling `ttnn.CreateDevice` directly. TT-Symbiote manages what happens _with_ the device but does not own device creation.

## Automatic fallback

When a TTNN operation fails (unsupported input shape, unimplemented kernel, etc.), `NormalRunWithFallback` mode (`TT_SYMBIOTE_RUN_MODE=NORMAL_WITH_FALLBACK`) catches the exception and re-executes the same operation using the original PyTorch layer stored in `TTNNModule._fallback_torch_layer`. This means a partially-supported model can still produce correct results while individual layers are being brought up.

## Positioning relative to tt-transformers

TT-Symbiote is **model-agnostic**. It provides the replacement infrastructure and the TTNN-accelerated building blocks (`TTNNLinear`, `TTNNLayerNorm`, `TTNNConv2dNHWC`, etc.), but it makes no assumptions about model architecture. The same framework has been used with ViT, ResNet-50, LLaMA, Whisper, SpeechT5, HunyuanVideo, and OpenVLA without changing the core.

`tt-transformers` (in `models/tt_transformers/`) takes a different stance. It provides **pre-written, Tenstorrent-native implementations** of specific transformer architectures. Those implementations are written directly in TTNN from the ground up rather than adapting existing PyTorch models. The two approaches are complementary:

| | TT-Symbiote | tt-transformers |
|---|---|---|
| Starting point | Existing PyTorch model | Clean-room TTNN implementation |
| Model coverage | Any architecture | Specific transformer families |
| Code changes required | A handful of setup calls | Full model rewrite in TTNN |
| Primary use case | Rapid bring-up of new models | Production-optimized transformer inference |

---

**Next:** [`source_layout.md`](./source_layout.md)
