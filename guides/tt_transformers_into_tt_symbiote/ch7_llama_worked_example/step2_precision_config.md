# Step 2: Precision Configuration — Selecting Dtypes for Each Component

## 1. Current Symbiote Precision in `test_llama.py`

`TTNNLinear.preprocess_weights_impl` stores weights in `ttnn.bfloat16`:

```python
def preprocess_weights_impl(self):
    self.tt_weight_host = preprocess_linear_weight(
        self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    self.tt_bias_host = None
    if self.bias is not None:
        self.tt_bias_host = preprocess_linear_bias(
            self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
```

`TTNNRMSNorm.preprocess_weights_impl` also uses `ttnn.bfloat16` for the weight tensor, after expanding it to tile-aligned shape `(32, hidden_size)`.

Summary of `test_llama` (basic) precision:

| Component | Symbiote class | Weight dtype | Activation dtype |
|---|---|---|---|
| MLP gate / up / down proj | `TTNNLinear` | `ttnn.bfloat16` | Inherited from input tensor |
| Attention Q / K / V / O proj | `TTNNLinear` (via `LlamaAttention`) | `ttnn.bfloat16` | Inherited from input tensor |
| RMSNorm (all instances) | `TTNNRMSNorm` | `ttnn.bfloat16` | `ttnn.rms_norm` output dtype |
| SiLU activation | `TTNNSilu` | N/A | Same as input |
| Embedding | `nn.Embedding` (PyTorch) | `torch.bfloat16` (model loaded with `.to(dtype=torch.bfloat16)`) | N/A |
| LM head | `TTNNLinear` | `ttnn.bfloat16` | Inherited |

`test_llama_intelligent` substitutes `SmartTTNNLinear` for `TTNNLinear` for the `nn.Linear` entries in `nn_to_ttnn`. `SmartTTNNLinear` inherits `preprocess_weights_impl` from `TTNNLinear`, so the weight dtype remains `ttnn.bfloat16`. The difference is in the forward pass: `SmartTTNNLinear` dispatches to `prefill_forward` or `decode_forward` based on the sequence length threshold of 32 tokens, and `prefill_forward` applies a `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` computed by `get_prefill_pc`.

## 2. TT Transformers Precision

TT Transformers resolves per-layer dtypes at construction time via `decoders_optimizations.get_tensor_dtype(decoder_id, tensor)`. The `TensorGroup` enum identifies distinct weight groups.

### MLP (`models/tt_transformers/tt/mlp.py`)

```python
ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
)
ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
)

self.w1 = as_sharded_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims)   # gate_proj
self.w3 = as_sharded_tensor("w3_sharded", ff1_3_dtype, dims=w1_dims)   # up_proj
self.w2 = as_sharded_tensor("w2_sharded", ff2_dtype,   dims=w2_dims)   # down_proj
```

The comment in `mlp.py` for `w1` notes: "bfp4 normally ok here but sub .99 pcc for llama 3.1 weights", indicating the default for `FF1_FF3` in practice is `ttnn.bfloat8_b`. The `FF2` group (`down_proj`, `w2`) uses its own potentially distinct dtype via `ff2_dtype`.

The SiLU activation in TT Transformers is fused into the `ttnn.mul` call using `input_tensor_a_activations=[self.activation_type]` rather than a separate module:

```python
w2_in = ttnn.mul(
    w1_out,
    w3_out,
    input_tensor_a_activations=[self.activation_type],
    dtype=activation_dtype or ttnn.bfloat8_b,
    memory_config=w1_out.memory_config(),
)
```

The `activation_dtype` also comes from `decoders_optimizations.get_tensor_dtype(decoder_id, TensorGroup.ACTIVATION)`.

### Attention (`models/tt_transformers/tt/attention.py`)

```python
self.wqkv_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.WQKV, prefetcher=use_prefetcher
)
self.wo_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.WO, prefetcher=use_prefetcher
)
self.kv_cache_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE, prefetcher=use_prefetcher
)
```

The fused QKV weight is stored at `self.wqkv_dtype`, the output projection at `self.wo_dtype`, and the KV cache at `self.kv_cache_dtype`. The attention module also resolves separate `compute_kernel_config` objects for decode QKV, decode SDPA, decode output, prefill SDPA, prefill QKV, and prefill output via `OpGroup` lookups.

## 3. Recommended Mapping: Symbiote Class per Component for TT Transformers-Equivalent Precision

The Symbiote module library provides several `TTNNLinear` subclasses distinguished primarily by weight dtype and the `@deallocate_weights_after` decorator. The following table gives the recommended class for each LLaMA component when targeting TT Transformers-equivalent precision.

| LLaMA component | TT Transformers dtype group | Recommended Symbiote class | Notes |
|---|---|---|---|
| MLP `gate_proj` (w1) | `TensorGroup.FF1_FF3` — typically `ttnn.bfloat8_b` | `TTNNLinearLLama` or `SmartTTNNLinearLLama` | `TTNNLinearLLama` stores weights at `ttnn.bfloat8_b`; `SmartTTNNLinearLLama` adds prefill/decode dispatch |
| MLP `up_proj` (w3) | `TensorGroup.FF1_FF3` — typically `ttnn.bfloat8_b` | `TTNNLinearLLama` or `SmartTTNNLinearLLama` | Same class as gate_proj |
| MLP `down_proj` (w2) | `TensorGroup.FF2` — typically `ttnn.bfloat8_b` | `TTNNLinearLLama` or `SmartTTNNLinearLLama` | If FF2 dtype differs from FF1_FF3, use `TTNNLinearLLamaBFloat16` for `ttnn.bfloat16` or a custom subclass |
| Attention QKV projections | `TensorGroup.WQKV` | `TTNNLinear` (created inside `LlamaAttention.from_torch`) | `LlamaAttention` constructs its own linears; changing the outer `nn_to_ttnn` entry does not affect them. To use bfloat8 here, `LlamaAttention.init_fused_parameters` / `init_parameters` must be updated to use `TTNNLinearLLama` |
| Attention output projection (`o_proj`) | `TensorGroup.WO` | `TTNNLinear` (created inside `LlamaAttention.from_torch`) | Same note as QKV |
| RMSNorm | No TT Transformers `TensorGroup` for norm weight | `TTNNRMSNorm` | Norm weights are kept in `ttnn.bfloat16`; TT Transformers `RMSNorm` also uses bfloat16 |
| LM head (when included) | Not a `TensorGroup` in standard LLaMA config | `TTNNLinear` (bfloat16) | See section 5 |

## 4. Verifying Output Quality Using DPL Mode

DPL (Dual-Path Logging) mode runs both the PyTorch reference path and the TTNN path on every module and op, compares their outputs using PCC (Pearson Correlation Coefficient), and propagates the PyTorch result downstream so accuracy errors do not compound between layers.

### Enabling DPL Mode

Set the environment variable before running the test:

```bash
TT_SYMBIOTE_RUN_MODE=DPL pytest models/experimental/tt_symbiote/tests/test_llama.py::test_llama_intelligent -s
```

### What DPL Does at Each Module

When a `TTNNModule` is called under `DPLRun`, the `module_run` static method:
1. Copies the input tensors to pure-Torch form.
2. Runs `self.torch_layer(...)` to obtain the reference output.
3. Runs `self.forward(...)` on TTNN to obtain the accelerated output.
4. Calls `compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)` which prints PCC for each matched tensor pair.
5. Returns the Torch output so downstream layers receive the reference value — errors in this layer do not affect the next layer's comparison.

The same dual-path logic applies at the op level: `DPLRun.torch_dispatch` runs the Torch op and, if TTNN can handle it, also runs the TTNN op and calls `compare_fn_outputs`.

### What to Look For in the Output

A healthy run produces lines like:

```
TTNNLinear: model.model.layers.0.mlp.gate_proj on device ...
PCC: 0.9998  (TTNNLinear_forward vs torch_layer)
TTNNRMSNorm: model.model.layers.0.input_layernorm on device ...
PCC: 0.9999  (TTNNRMSNorm_forward vs torch_layer)
LlamaAttention: model.model.layers.0.self_attn on device ...
PCC: 0.9994  (LlamaAttention_forward vs torch_layer)
```

Thresholds used in practice by TT Transformers integration tests:
- PCC >= 0.99 is generally acceptable for intermediate activations.
- PCC >= 0.999 is expected for normalization layers.
- PCC < 0.95 at any layer warrants investigation before moving to production.

If the PCC printed for `TTNNLinearLLama` drops below 0.99, consider switching that layer back to `TTNNLinear` (bfloat16 weights) or increasing math fidelity via a custom `compute_kernel_config`. `SmartTTNNLinear.prefill_forward` uses `ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, ...)` from `compute_kernel_config()` in `linear_intelligent.py`.

### Requiring `torch_layer` for DPL

`DPLRun.module_run` asserts `self.torch_layer is not None`. Every `TTNNModule.from_torch` call stores the original PyTorch layer in `self._fallback_torch_layer`. Modules created via `from_parameters` (without a backing PyTorch layer) cannot be used under DPL mode.

## 5. The `exclude_replacement={"lm_head"}` Pattern and Why Precision Matters Differently

The LM head maps hidden states of shape `[batch, seq_len, hidden_size]` to logit vectors of shape `[batch, seq_len, vocab_size]`. For LLaMA 3.2-1B, `vocab_size=128256` and `hidden_size=2048`, so the weight matrix is `128256 x 2048` — the largest single tensor in the model.

Precision concerns for the LM head differ from the MLP and attention projections in two ways:

1. **Argmax sensitivity.** Token selection in greedy decoding (`model.generate` with default settings) is determined by `argmax(logits)`. Even a small absolute error in a small number of logit entries can flip the selected token. A bfloat8 LM head may preserve PCC > 0.99 in aggregate while still selecting different tokens, because the margin between the top-1 and top-2 logits can be very narrow.

2. **Memory vs. correctness trade-off.** Keeping `lm_head` as a PyTorch `nn.Linear` avoids allocating the 128256 x 2048 weight on device. With `@deallocate_weights_after` this memory would be freed after each token generation step, but the per-step allocation and deallocation cost may be significant.

The recommended approach for initial integration is to keep `exclude_replacement={"lm_head"}` (as `test_llama_intelligent` does) and validate generation quality on a held-out prompt set before considering whether device-resident LM head execution is needed for throughput.
