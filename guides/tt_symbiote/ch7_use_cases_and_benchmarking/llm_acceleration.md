# LLM Acceleration: LLaMA-3.2-1B-Instruct

Source: `models/experimental/tt_symbiote/tests/test_llama.py`

The test file defines two test functions. `test_llama` uses `TTNNLinear` for all `nn.Linear` layers. `test_llama_intelligent` replaces `nn.Linear` with `SmartTTNNLinear`, which selects between a decode-optimized and a prefill-optimized matmul path at runtime, and additionally excludes the language-model head from replacement.

## HuggingFace model loading

Both tests load the same checkpoint:

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(dtype=torch.bfloat16)
```

`AutoModelForCausalLM` returns a `LlamaForCausalLM` instance. The transformer layers are stored in `model.model.layers`. Each layer contains:

- `input_layernorm` and `post_attention_layernorm` — instances of `LlamaRMSNorm`
- `self_attn` — an instance of `LlamaSdpaAttention` (or a similar attention variant depending on the HF version)
- `mlp` — an instance of `LlamaMLP` (the HuggingFace class, not the wrapper defined in the test)

## The `LlamaMLP` wrapper

The HuggingFace `LlamaMLP` stores its three projection matrices (`gate_proj`, `up_proj`, `down_proj`) directly as attributes rather than as entries in `_modules`. Because `register_module_replacement_dict` traverses `_modules` to find children to replace, the three `nn.Linear` instances inside the original `LlamaMLP` are invisible to the replacement pass.

The test solves this by introducing a local `LlamaMLP` wrapper that re-exposes the three projections as regular `nn.Module` children:

```python
class LlamaMLP(nn.Module):
    def __init__(self, old_layer):
        super().__init__()
        self.config = old_layer.config
        self.hidden_size = old_layer.hidden_size
        self.intermediate_size = old_layer.intermediate_size
        self.gate_proj = old_layer.gate_proj
        self.up_proj = old_layer.up_proj
        self.down_proj = old_layer.down_proj
        assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in this test."
        self.act_fn = nn.SiLU()

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

Assigning `gate_proj`, `up_proj`, and `down_proj` inside `__init__` causes PyTorch to register them in `_modules`, making them visible to subsequent replacement passes. The `act_fn = nn.SiLU()` assignment gives the `TTNNSilu` replacement a target to match.

The wrapper uses `from_torch` as its construction classmethod, which is the interface `register_module_replacement_dict` expects when it replaces a module. The test asserts that `hidden_act == "silu"` before wrapping to guard against silent precision errors on checkpoints with a different activation.

## Replacement mapping: `test_llama`

The test runs two consecutive calls to `register_module_replacement_dict`. The first call handles the structural pre-pass (nn-to-nn), the second handles the TTNN replacement (nn-to-ttnn).

### First pass: `nn_to_nn`

```python
nn_to_nn = {
    model.model.layers[0].mlp.__class__: LlamaMLP,
}
modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)
```

`model.model.layers[0].mlp.__class__` resolves to the HuggingFace `LlamaMLP` class at runtime. Every `LlamaMLP` instance in the model is replaced with the wrapper defined in the test. After this pass `gate_proj`, `up_proj`, and `down_proj` are in `_modules`.

### Second pass: `nn_to_ttnn`

```python
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.SiLU: TTNNSilu,
    nn.LayerNorm: TTNNLayerNorm,
    model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    model.model.layers[0].self_attn.__class__: LlamaAttention,
}
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

`model.model.layers[0].input_layernorm.__class__` resolves to `LlamaRMSNorm` at runtime. `model.model.layers[0].self_attn.__class__` resolves to the specific attention class present in the loaded checkpoint (`LlamaSdpaAttention` or similar). Using `.__class__` instead of importing the HF class by name avoids version-specific import paths.

The full replacement effect:

| PyTorch class | Replaced by |
|---|---|
| `nn.Linear` | `TTNNLinear` |
| `nn.SiLU` | `TTNNSilu` |
| `nn.LayerNorm` | `TTNNLayerNorm` |
| `LlamaRMSNorm` | `TTNNRMSNorm` |
| `LlamaSdpaAttention` (or equivalent) | `LlamaAttention` |

`LlamaAttention` is imported from `models/experimental/tt_symbiote/modules/attention.py`. It is a `TTNNModule` subclass that implements the LLaMA attention computation using TTNN operations.

## Replacement mapping: `test_llama_intelligent`

`test_llama_intelligent` uses the same `nn_to_nn` pre-pass and the same attention and normalization replacements, but substitutes `SmartTTNNLinear` for `TTNNLinear`:

```python
nn_to_ttnn = {
    nn.Linear: SmartTTNNLinear,
    nn.SiLU: TTNNSilu,
    nn.LayerNorm: TTNNLayerNorm,
    model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    model.model.layers[0].self_attn.__class__: LlamaAttention,
}
```

It also passes `exclude_replacement={"lm_head"}` to the second registration call:

```python
modules = register_module_replacement_dict(
    model, nn_to_ttnn, model_config=None, exclude_replacement={"lm_head"}
)
```

`lm_head` is the final projection that maps hidden states to vocabulary logits. It is an `nn.Linear` layer with `out_features` equal to the vocabulary size (32 000 for this checkpoint). `SmartTTNNLinear._get_prefill_pc` hard-codes a `None` program config for any module whose `module_name` is `"lm_head"`, which means the layer would fall back to a less-optimized path. Excluding it entirely keeps the layer as a plain `nn.Linear` that runs on CPU, avoiding that path.

## `SmartTTNNLinear`: prefill vs decode dispatch

`SmartTTNNLinear` (defined in `modules/linear_intelligent.py`) extends `TTNNLinear` and adds two-path dispatch:

```python
def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    ...
    seq_len = int(input_shape[-2])
    mode = "decode" if seq_len <= 32 else "prefill"

    if mode == "decode":
        tt_output = self.decode_forward(input_tensor)
    else:
        tt_output = self.prefill_forward(input_tensor, seq_len)
```

`seq_len <= 32` is treated as decode; longer sequences go through `prefill_forward`, which computes a `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` keyed on `seq_len` and caches it in `_prefill_pc_cache`. The decode path calls `ttnn.linear` with `memory_config=ttnn.DRAM_MEMORY_CONFIG` and no additional program config. The prefill path adds `compute_kernel_config` using `ttnn.WormholeComputeKernelConfig` with `math_fidelity=ttnn.MathFidelity.HiFi2` and `packer_l1_acc=True`.

Use `SmartTTNNLinear` when the model will be used for both prefill (long prompt) and decode (single-token generation) phases. Use `TTNNLinear` when you want the simplest possible replacement or are profiling a single phase.

Two further subclasses are available in `linear_intelligent.py`:

- `SmartTTNNLinearLLama` — overrides `preprocess_weights_impl` to use `ttnn.bfloat8_b` precision and wraps `forward` with `@deallocate_weights_after`.
- `SmartTTNNLinearLLamaBFloat16` — uses the default `bfloat16` preprocessing from `TTNNLinear` but still wraps `forward` with `@deallocate_weights_after`.

## Setup sequence

Both test functions follow the same five-step sequence after building `nn_to_ttnn`, differing only in the output filename passed to `save_stats_to_file`:

```python
# 1. Run the structural pre-pass (nn_to_nn)
modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)

# 2. Run the TTNN replacement pass (nn_to_ttnn)
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

# 3. Assign device to every TTNNModule found in the model
set_device(model, device)

# 4. Preprocess weights and move them to device
for k, v in tqdm(modules.items()):
    v.preprocess_weights()
    v.move_weights_to_device()

# 5. Disable training-mode features and run inference
model.eval()
torch.set_grad_enabled(False)
DispatchManager.clear_timings()
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
# test_llama saves to:
DispatchManager.save_stats_to_file("llama_timing_stats.csv")
# test_llama_intelligent saves to:
# DispatchManager.save_stats_to_file("llama_intelligent_timing_stats.csv")
```

Both `preprocess_weights` and `move_weights_to_device` are idempotent; they no-op on subsequent calls.

## Output files

`DispatchManager.save_stats_to_file("llama_timing_stats.csv")` produces two files in the current working directory:

- `llama_timing_stats.csv` — flat event log with one row per recorded event
- `llama_timing_stats_pivot.csv` — pivot table aggregated by `(func_name, module_name)` and `backend`

### Flat log columns (`llama_timing_stats.csv`)

| Column | Description |
|---|---|
| (index) | Sequential integer row index |
| `attrs` | Dictionary of extra attributes (empty `{}` for most entries) |
| `module_name` | Dotted path of the module as named by PyTorch, e.g. `model.layers.0.self_attn` |
| `func_name` | Operation or class name that was timed, e.g. `LlamaAttention_forward` |
| `duration` | Wall-clock seconds for this single invocation |
| `backend` | One of `TTNN`, `Torch`, or `TorchModules`; `TTNN` = a TTNNModule forward/weight call dispatched to device; `Torch` = an individual ATen op through the Python dispatch key; `TorchModules` = the outer `nn.Module.forward` timer injected by `set_device`'s `timed_call` wrapper |

### Pivot table columns (`llama_timing_stats_pivot.csv`)

| Column | Description |
|---|---|
| `func_name` | Operation or class name (pivot index) |
| `module_name` | Dotted path (pivot index) |
| `TTNN` | Total seconds across all invocations under the TTNN backend |
| `Torch` | Total seconds under the Torch backend |
| `TorchModules` | Total seconds under the TorchModules backend |
| `Total_Duration` | Sum of all backend columns |
| `Min_Duration` | Minimum single-invocation time across all backends |
| `Max_Duration` | Maximum single-invocation time across all backends |
| `Row_Count` | Number of raw events merged into this row |

### Reading a per-layer breakdown

To find the slowest attention layer, sort the pivot table on `Total_Duration` and filter `func_name` to `LlamaAttention_forward`. Each row in the actual `llama_timing_stats_pivot.csv` output corresponds to one transformer layer position. For example:

```
func_name               module_name               TTNN    Torch  TorchModules  Total_Duration  Row_Count
LlamaAttention_forward  model.layers.0.self_attn  3.006   0.0    0.0           3.006           128
LlamaAttention_forward  model.layers.1.self_attn  0.611   0.0    0.0           0.611           128
```

`Row_Count` of 128 means the attention forward was called 128 times during the 128-token generation run. Layer 0 shows roughly 5x higher total time than later layers because the first invocation includes any one-time setup cost (e.g., program compilation).

### Identifying fallback layers

A non-zero `TorchModules` value with a zero `TTNN` value on a row where you expected TTNN dispatch indicates that the module is running as a plain `nn.Module` — either because it was excluded from replacement, because no replacement mapping covered its class, or because the TTNN forward raised an exception and the fallback torch layer ran instead. Compare `TorchModules` totals in the pivot table against `TTNN` totals to quantify what fraction of compute is still on CPU.

`save_stats_to_file` also prints the top 30 modules by `TorchModules` duration to stdout when that column is present, providing a quick view of the largest remaining CPU workloads.

### Timing files produced by the test suite

The tests directory contains pre-generated output from actual runs:

- `tests/llama_timing_stats.csv` and `tests/llama_timing_stats_pivot.csv` — from `test_llama`
