# Direct Reuse Modules

This file covers the two Gemma 3 TTNN modules that can be reused with no or minimal changes for the Gemma 4 vision encoder port. These modules account for approximately 15% of the total codebase and require less than 1 day of validation and minor adjustments.

> **Note:** The encoder block (`gemma_image_block.py`), model config (`model_config.py`), and checkpoint loading (`load_checkpoints.py`) were previously listed here but have been reclassified as **Modification required** due to the scope of changes needed. See [modification_required_modules.md](./modification_required_modules.md) for their updated analysis.

All file paths are relative to `models/demos/multimodal/gemma3/tt/` for Gemma 3 and `models/demos/multimodal/gemma4/tt/` for the proposed Gemma 4 directory.

## Why These Modules Transfer Directly

The dominant compute dimensions are identical between the two encoders:

| Dimension | Gemma 3 (SigLIP) | Gemma 4 |
|-----------|-------------------|---------|
| `hidden_size` | 1152 | 1152 |
| `intermediate_size` | 4304 | 4304 |
| `num_attention_heads` | 16 | 16 |
| `head_dim` | 72 | 72 |
| `num_hidden_layers` | 27 | 27 |
| `hidden_activation` | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` |

Because the weight matrix shapes, activation functions, and layer counts are the same, any TTNN module whose behavior depends only on these parameters can be copied and used without structural changes. The only work is verifying weight key names for checkpoint loading and confirming numerical equivalence.

## `gemma_image_mlp.py`

**Gemma 4 target:** `gemma4_vision_mlp.py`
**Reuse class:** Direct reuse
**Effort:** < 1 day

### Architecture Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Gate projection | `[1152, 4304]` | `[1152, 4304]` |
| Up projection | `[1152, 4304]` | `[1152, 4304]` |
| Down projection | `[4304, 1152]` | `[4304, 1152]` |
| Activation | `gelu_pytorch_tanh` | `gelu_pytorch_tanh` |
| Bias | None | None |
| Structure | `down(gelu(gate(x)) * up(x))` | `down(gelu(gate(x)) * up(x))` |

The MLP is architecturally identical. The gated GeLU structure, all three projection shapes, and the absence of bias are all the same.

### Validation Steps

1. **Copy the module** from `gemma_image_mlp.py` to `gemma4_vision_mlp.py`.
2. **Update weight key names.** Verify the HuggingFace checkpoint key mapping. Gemma 4 keys follow the pattern `vision_tower.encoder.layers.{N}.mlp.{gate_proj,up_proj,down_proj}.weight`. Confirm these match the key expectations in the load path.
3. **Verify `Gemma4ClippableLinear`** wrapper. Gemma 4 introduces a `Gemma4ClippableLinear` wrapper around standard linear layers. When `use_clipped_linears=False` (the default for the vision encoder), this wrapper is a transparent passthrough. Confirm that the TTNN `ttnn.linear` call is unaffected.
4. **Run PCC validation.** Feed a random input tensor of shape `[1, 840, 1152]` through both the PyTorch reference and the TTNN module. Confirm PCC > 0.999 in BF16.

> **Tip:** Since the MLP accounts for approximately 70% of per-layer parameters and 60% of per-layer FLOPs, confirming its direct reusability early de-risks a large fraction of the port. Make this the first module you validate.

### Sharding and Memory Config

The existing Gemma 3 sharding strategy for the MLP weight matrices should transfer directly since the shapes are identical. The `[1152, 4304]` gate and up projections and the `[4304, 1152]` down projection have the same tiling and sharding requirements.

> **Tip:** If Gemma 3 uses width-sharded matmuls for the MLP projections on Wormhole's 8x8 grid, the same configuration applies to Gemma 4. No re-tuning should be necessary.

## `gemma_vision_rmsnorm.py`

**Gemma 4 target:** `gemma4_vision_rmsnorm.py`
**Reuse class:** Direct reuse
**Effort:** < 1 day

### Architecture Comparison

| Property | Gemma 3 (SigLIP) | Gemma 4 |
|----------|-------------------|---------|
| Dimension | 1152 | 1152 |
| Epsilon | 1e-6 | 1e-6 |
| Learnable scale | Yes | Yes (most norms); No (V-norm, pre-projection norm) |
| Norm type | RMSNorm | RMSNorm |

### Changes Required

The only change is adding support for a "no learnable scale" variant. Gemma 4 uses RMSNorm without a learnable scale parameter in two places:

1. **V-norm** in attention: the per-head RMSNorm applied to the value projection has no learnable scale.
2. **Pre-projection norm** in the multimodal embedder: the RMSNorm before the final linear projection has no learnable scale.

This can be implemented with a simple boolean flag:

```python
class TtGemma4VisionRMSNorm:
    def __init__(self, device, state_dict, layer_name, eps=1e-6, has_weight=True):
        self.eps = eps
        if has_weight:
            self.weight = ttnn.as_tensor(
                state_dict[f"{layer_name}.weight"],
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            self.weight = None

    def __call__(self, x):
        # ttnn.rms_norm handles the no-weight case when weight=None
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)
```

### Validation Steps

1. **Copy the module** and add the `has_weight` parameter.
2. **Validate with learnable scale** against PyTorch `Gemma4RMSNorm` with default parameters. PCC > 0.999.
3. **Validate without learnable scale** against the same reference with `elementwise_affine=False` (or equivalent). PCC > 0.999.

> **Warning:** Gemma 3 SigLIP uses `nn.LayerNorm` in the vision encoder, while Gemma 4 uses `RMSNorm`. If the existing `gemma_vision_rmsnorm.py` was written for the Gemma 3 language model side (which does use RMSNorm), it can be reused directly. If it was written for the SigLIP vision encoder (which uses LayerNorm), you need the language model's RMSNorm implementation instead.

## Summary

These two modules form the validated compute backbone of the Gemma 4 TTNN port. Because they require only copying and minor parameter additions — no algorithmic changes — they can be completed in less than a day of effort. Completing them early provides:

1. **A validated compute backbone.** The MLP and RMSNorm handle the majority of per-layer FLOPs and parameters.
2. **Confidence in the reuse strategy.** If PCC validation passes for these modules, the shared dimensions are confirmed and the remaining work focuses on the modules that need modification or new implementation.

---

**Next:** [`modification_required_modules.md`](./modification_required_modules.md) — Modules that need targeted modifications for the Gemma 4 port.
