# Vision and Multimodal Models

Sources:
- `models/experimental/tt_symbiote/tests/test_vit.py`
- `models/experimental/tt_symbiote/tests/test_resnet50.py`
- `models/experimental/tt_symbiote/modules/conv.py`
- `models/experimental/tt_symbiote/modules/attention.py`
- `models/experimental/tt_symbiote/modules/linear.py`

The tests directory contains two vision test files with working end-to-end integration. `test_vit.py` covers ViT-base-patch16-224 for image classification. `test_resnet50.py` covers ResNet-50 from torchvision. No OWL-ViT or other object-detection test is present in the test suite.

## Available built-in vision modules

The following TTNN-accelerated modules are available in `modules/conv.py` and `modules/attention.py`. They are the building blocks used in both test files.

### Convolution modules (`modules/conv.py`)

| Class | Description |
|---|---|
| `TTNNConv2dNHWC` | Single conv2d layer; expects NHWC input |
| `TTNNConv2dBNNHWC` | Conv2d with batch norm folded into the kernel weights |
| `TTNNConv2dBNActivationNHWC` | Conv2d with batch norm and an activation function folded in |
| `TTNNBottleneck` | Full ResNet Bottleneck block wrapping three fused conv-BN-activation stacks |
| `TTNNViTEmbeddings` | Patch embedding plus CLS token concatenation plus position embedding addition |
| `TTNNPatchEmbedding` | The inner patch projection used by `TTNNViTEmbeddings` |
| `TTNNMaxPool2dNHWC` | Max pooling with NHWC input |

All `NHWC` modules expect the spatial tensor in `[N, H, W, C]` order. They each keep a `NHWCConvPytorch` (or equivalent) as their `_fallback_torch_layer` so that the normal dispatch fallback path works correctly.

Batch norm parameters are folded into the convolution weights at `preprocess_weights_impl` time using `fold_batch_norm2d_into_conv2d`, which computes a single fused weight and bias. The original `nn.BatchNorm2d` is discarded after folding.

### Attention modules (`modules/attention.py`)

| Class | Description |
|---|---|
| `TTNNViTSelfAttention` | ViT multi-head self-attention; replaces `ViTSelfAttention` from HuggingFace |
| `TTNNWhisperAttention` | Whisper encoder/decoder attention (see `speech_and_debugging.md`) |

`TTNNViTSelfAttention` extends the internal `TTNNSelfAttention` base class, which handles Q/K/V projection using `TTNNLinear` and the attention score computation using TTNN ops.

### Linear modules with built-in activations (`modules/linear.py`)

| Class | Description |
|---|---|
| `TTNNViTIntermediate` | Replaces `ViTIntermediate`; wraps the dense layer with GELU using `TTNNLinearGelu` |

`TTNNViTIntermediate.from_torch` accepts a `ViTIntermediate` instance and asserts that its activation is `GELUActivation` before constructing the TTNN replacement.

## NHWC layout implications

Standard PyTorch models store spatial tensors in NCHW order. TTNN's convolution implementation operates most efficiently on NHWC. The `TTNNConv2dNHWC` fallback torch layer `NHWCConvPytorch` wraps an `nn.Conv2d` and applies `permute(0, 3, 1, 2)` before calling the conv and `permute(0, 2, 3, 1)` after, keeping the PyTorch path consistent with the TTNN path.

When replacing a `Bottleneck` block, `TTNNBottleneck.forward` explicitly permutes the input from NCHW to NHWC at the start and permutes the output back to NCHW at the end:

```python
x = self.permute(x, perm=[0, 2, 3, 1])   # NCHW -> NHWC
...
out = self.permute(out, perm=[0, 3, 1, 2])  # NHWC -> NCHW
```

This means surrounding `nn.Module` layers that have not been replaced still receive their expected NCHW tensors.

## ViT-base walkthrough (`test_vit.py`)

### Model loading

```python
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
model = model.to(dtype=torch.bfloat16)
```

### Structural pre-pass (`nn_to_nn`)

Three structural rewrites run before the TTNN replacement pass:

```python
nn_to_nn = {
    ViTEmbeddings: RewrittenViTEmbeddings,
    ViTLayer: RewrittenViTLayer,
    ViTOutput: RewrittenViTOutput,
}
modules1 = register_module_replacement_dict(model, nn_to_nn, model_config={"program_config_ffn": {}})
```

`RewrittenViTEmbeddings` replaces the HuggingFace `ViTEmbeddings` with a version that delegates patch embedding to `TTNNViTEmbeddings.from_torch` and exposes `self.embeddings` as a `TTNNModule`. This makes the patch-embedding weights accessible to the replacement machinery.

`RewrittenViTLayer` replaces `ViTLayer` to add a `TTNNAdd()` module for the residual connection. The residual addition in the original `ViTLayer` is a plain Python `+` operator on tensors, which would not be intercepted. Replacing it with `TTNNAdd()` (an `nn.Module`) allows it to be dispatched to TTNN.

`RewrittenViTOutput` replaces `ViTOutput` for the same reason — it adds a `TTNNAdd()` for the second residual connection.

### TTNN replacement pass (`nn_to_ttnn`)

```python
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.LayerNorm: TTNNLayerNorm,
    ViTSelfAttention: TTNNViTSelfAttention,
    ViTIntermediate: TTNNViTIntermediate,
}
modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
```

Both passes are combined before the setup loop:

```python
modules = {**modules1, **modules2}
```

### Setup sequence

```python
set_device(model, device)
for k, v in modules.items():
    v.preprocess_weights()
    v.move_weights_to_device()
model.eval()
torch.set_grad_enabled(False)
```

### Running inference and saving timings

```python
input_tensor = torch.randn(1, 224, 224, 4)
model(input_tensor)           # warm-up pass
DispatchManager.clear_timings()
result = model(input_tensor)  # timed pass
DispatchManager.save_stats_to_file("vit_timing_stats.csv")
print(result.logits)
```

The warm-up pass before `clear_timings()` ensures that any TTNN program compilation or one-time device setup is excluded from the recorded statistics. The timed pass then reflects steady-state per-layer throughput.

Note: the test passes a tensor of shape `(1, 224, 224, 4)` — four channels instead of the expected three. This is a limitation of the test as written; in production, use `(1, 224, 224, 3)`.

## ResNet-50 walkthrough (`test_resnet50.py`)

### Model loading

```python
model = resnet50(pretrained=True).to(torch.bfloat16)
```

`resnet50` is imported from `torchvision.models`. The test uses `pretrained=True`, which loads ImageNet weights.

This test requires a custom `device_params` fixture to increase L1 memory:

```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
```

The larger L1 allocation is needed because `TTNNConv2dNHWC` uses L1 for intermediate activations inside the convolution kernel.

### Replacement mapping

```python
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    Bottleneck: TTNNBottleneck,
}
register_module_replacement_dict(model, nn_to_ttnn, model_config={"program_config_ffn": {}})
```

`Bottleneck` is imported from `torchvision.models.resnet`. `TTNNBottleneck.from_torch` creates the TTNN replacement and immediately calls `initilize_submodules`, which builds three internal TTNN conv layers:

- `self.conv1 = TTNNConv2dBNActivationNHWC.from_torch(conv1, bn1, relu)`
- `self.conv2 = TTNNConv2dBNActivationNHWC.from_torch(conv2, bn2, relu)`
- `self.conv3 = TTNNConv2dBNNHWC.from_torch(conv3, bn3)`

`conv3` is not followed by a ReLU inside the bottleneck block (ReLU is applied after the residual addition), so `TTNNConv2dBNNHWC` — without activation — is used there.

This test does not call `DispatchManager.save_stats_to_file`. Timing output is not produced by the ResNet-50 test.

### Setup and inference

```python
set_device(model, device)
model.eval()
torch.set_grad_enabled(False)
result = model(torch.randn(1, 3, 224, 224, dtype=torch.bfloat16))
print(result)
```

The ResNet test calls `set_device` but does not iterate over `modules` to call `preprocess_weights` and `move_weights_to_device` explicitly. `TTNNBottleneck` builds its sub-modules in `initilize_submodules` with weights already attached, and `NormalRun.module_run` (the default dispatch path) calls `preprocess_weights` and `move_weights_to_device` lazily on the first forward call.

## Conceptual replacement pattern summary

For any vision model not yet covered by the test suite, the general pattern is:

1. Replace structural modules that hide `nn.Linear` or `nn.Conv2d` children using an `nn_to_nn` pre-pass, exactly as `RewrittenViTLayer` exposes residual additions.
2. Map remaining standard PyTorch types to their TTNN equivalents in `nn_to_ttnn`.
3. For convolutional backbones, use `TTNNBottleneck` if the architecture uses the standard ResNet bottleneck design, or build a custom `TTNNModule` subclass that chains `TTNNConv2dBNActivationNHWC` instances.
4. For transformer vision encoders (ViT, DeiT, Swin variants), the `TTNNViTSelfAttention` + `TTNNViTIntermediate` pair covers the encoder block internals; apply `RewrittenViTLayer`-style wrappers to expose residual additions.
5. Patch embedding for vision transformers is handled by `TTNNViTEmbeddings.from_torch`, which accepts a `ViTPatchEmbeddings` instance, a CLS token tensor, and a position embeddings tensor.
