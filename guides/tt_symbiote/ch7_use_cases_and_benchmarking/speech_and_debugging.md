# Speech Models and the Debugging Workflow

Sources:
- `models/experimental/tt_symbiote/tests/test_dpl.py`
- `models/experimental/tt_symbiote/tests/test_whisper3.py`
- `models/experimental/tt_symbiote/core/run_config.py`
- `models/experimental/tt_symbiote/core/dispatchers/debug_dispatcher.py`

## What `test_dpl.py` actually tests

`test_dpl.py` does not test a speech model. It tests the DPL (Dual Path Logger) run mode itself using a minimal two-operation custom module:

```python
class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 2.0
        x = x + 3.0
        return x
```

The test asserts that `TT_SYMBIOTE_RUN_MODE` is set to `"DPL"` before it proceeds:

```python
assert (
    os.environ.get("TT_SYMBIOTE_RUN_MODE") == "DPL"
), f"Expected TT_SYMBIOTE_RUN_MODE environment variable to be 'DPL', got {os.environ.get('TT_SYMBIOTE_RUN_MODE')}"
```

The test wraps a plain `torch.Tensor` in a `TorchTTNNTensor`, moves it to device, runs the model, and then verifies that the TTNN result matches the CPU result:

```python
inputs = TorchTTNNTensor(torch.tensor([1.0, 2.0, 3.0]))
inputs.ttnn_tensor = ttnn.to_device(inputs.to_ttnn, device)
outputs = model(inputs)
assert (outputs.elem == outputs.to_torch).all()
result = outputs.elem.clone()
outputs.elem = None  # Force using TTNN tensor only
ttnn_result = outputs.to_torch
assert (result == ttnn_result).all()
```

This test demonstrates that when `TT_SYMBIOTE_RUN_MODE=DPL`, the framework keeps both a PyTorch copy and a TTNN copy of each tensor in flight simultaneously. The second assertion clears the CPU copy (`outputs.elem = None`) and confirms that `to_torch` can reconstruct the result from the TTNN tensor alone.

## Whisper integration (`test_whisper3.py`)

Source: `models/experimental/tt_symbiote/tests/test_whisper3.py`

The test accelerates `distil-whisper/distil-large-v3` (`AutoModelForSpeechSeq2Seq`) and uses the existing TTNN-optimized Whisper encoder from `models/demos/whisper/tt/ttnn_optimized_functional_whisper.py`.

### Model loading

```python
model_id = "distil-whisper/distil-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(dtype=torch_dtype)
```

This test requires a custom L1 size:

```python
@pytest.mark.parametrize("device_params", [{"l1_small_size": 1024}], indirect=True)
```

### `TTNNWhisperAttention`

`TTNNWhisperAttention` is defined in `modules/attention.py` and imported directly in the test. It is a `TTNNModule` subclass that handles Whisper's cross- and self-attention patterns. It appears in the `nn_to_ttnn` dict alongside the full encoder-layer replacement:

```python
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.SiLU: TTNNSilu,
    OriginalWhisperEncoderLayer: WhisperEncoderLayer,
    OriginalWhisperAttention: TTNNWhisperAttention,
    nn.LayerNorm: TTNNLayerNorm,
    nn.GELU: TTNNGelu,
}
```

`OriginalWhisperEncoderLayer` is replaced by a local `WhisperEncoderLayer` (defined in the test file itself), which is a `TTNNModule` subclass that delegates to the pre-built `encoder_layer` function from the TTNN optimized Whisper demo. This replacement pre-processes parameters using `preprocess_model_parameters` and a custom mesh preprocessor, rather than the standard `preprocess_weights_impl` flow.

### Setup sequence

```python
all_modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
set_device(model, device)
for k, v in tqdm(all_modules.items()):
    v.preprocess_weights()
    v.move_weights_to_device()
model.eval()
torch.set_grad_enabled(False)
```

### Running inference

The test uses the HuggingFace `pipeline` API with the modified model:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
)
dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]
DispatchManager.clear_timings()
result = pipe(sample, return_timestamps=True)
print(result["text"])
DispatchManager.save_stats_to_file("whisper_timing_stats.csv")
```

`DispatchManager.clear_timings()` is called before inference to exclude the weight-loading phase. The pipeline handles audio feature extraction internally; TT Symbiote sees the model only at the `nn.Module.forward` boundary.

## The DPL debugging workflow

The run-mode system is controlled by the `TT_SYMBIOTE_RUN_MODE` environment variable, read at import time in `core/run_config.py`. The full set of modes is:

| Mode | Description |
|---|---|
| `NORMAL` | Default: TTNN forward with PyTorch fallback on error |
| `DPL` | Dual path: runs both PyTorch and TTNN, propagates a combined tensor (PyTorch elem + TTNN tensor), compares outputs; TTNN errors in one layer affect downstream inputs |
| `DPL_NO_ERROR_PROP` | Dual path: runs both with independently copied tensors for the TTNN path; TTNN errors do not corrupt the PyTorch-path tensors that downstream layers receive, compares outputs |
| `SEL` | Single evaluation log: runs both, compares outputs, does not propagate TTNN errors |
| `CPU` | CPU-only: TTNN dispatch is disabled |
| `TRACED` | Traced execution for performance profiling |

### `DPL` mode

Because the original tensor buffers are shared across layers, TTNN errors in one layer propagate forward and will affect all downstream layers.

To enable it:

```bash
export TT_SYMBIOTE_RUN_MODE=DPL
```

### `DPL_NO_ERROR_PROP` mode

This means TTNN numerical errors in one layer do not corrupt the PyTorch-path tensors that are fed to subsequent layers. Use this mode to isolate a single layer's numerical accuracy without needing to first fix all upstream layers.

```bash
export TT_SYMBIOTE_RUN_MODE=DPL_NO_ERROR_PROP
```

When `DPL_NO_ERROR_PROP.module_run` finishes executing a module's TTNN path, it prints a completion message to stdout:

```
DPLNoErrorPropRun: Done Executing {class_name} from {module_name} on device {device}
```

This message appears for every `TTNNModule.forward` call, making it easy to see which modules reached the TTNN path and which fell back.

### Reading `compare_fn_outputs` output

`compare_fn_outputs` is called on both inputs and outputs at every dispatch boundary (both individual ATen ops and full `TTNNModule.forward` calls). It computes the Pearson Correlation Coefficient (PCC) between the PyTorch and TTNN tensors and prints the result when a mismatch is detected. A PCC close to 1.0 indicates good numerical agreement; values below approximately 0.99 for `bfloat16` typically indicate a problem in that op or module.

To find which module first introduces a degradation:

1. Run with `TT_SYMBIOTE_RUN_MODE=DPL_NO_ERROR_PROP` so errors do not propagate.
2. Watch stdout for PCC values printed by `compare_fn_outputs`.
3. The first module where PCC drops significantly is the likely source of the regression.
4. Add that module's path to `exclude_replacement` to confirm: if accuracy recovers, the replacement for that module is the cause.

### Isolating a module with `exclude_replacement`

```python
modules = register_module_replacement_dict(
    model,
    nn_to_ttnn,
    model_config=None,
    exclude_replacement={"model.layers.0.self_attn"},
)
```

Any module path listed in `exclude_replacement` is skipped during the replacement pass and remains as the original `nn.Module`. Run the model in `DPL_NO_ERROR_PROP` mode, check whether the PCC for surrounding layers improves, then narrow down to the exact module.

## `TT_SYMBIOTE_DISPATCHER=DEBUG` for verbose logging

The dispatcher layer is separate from the run mode. It is selected by `TT_SYMBIOTE_DISPATCHER` at import time:

```bash
export TT_SYMBIOTE_DISPATCHER=DEBUG
```

The `DEBUG` dispatcher (defined in `core/dispatchers/debug_dispatcher.py`) wraps the default dispatcher and logs at `logging.DEBUG` level for every ATen operation that is evaluated for TTNN dispatch:

- When an operation cannot be dispatched to TTNN, it logs: `Cannot dispatch {func_name} to TTNN`
- When an operation is dispatched, the default `dispatch_to_ttnn` is called and the result is returned

The logger name is the module's `__name__` (`models.experimental.tt_symbiote.core.dispatchers.debug_dispatcher`). To capture the output, configure the Python logging system before importing TT Symbiote:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This is useful when a layer you expect to run on TTNN is silently falling back to Torch. The debug log will show the exact `aten::` operation name that failed the `can_dispatch_to_ttnn` check.

## Interpreting timing pivot tables for speech models

The pivot table produced by `DispatchManager.save_stats_to_file("whisper_timing_stats.csv")` has the same structure as the LLaMA pivot table. The fallback-layer identification and TTNN vs CPU column guidance in `llm_acceleration.md` ("Identifying fallback layers") applies equally here.

### First-token vs steady-state latency

The `Min_Duration` and `Max_Duration` columns in the pivot show the range across all `Row_Count` invocations. For speech models that process variable-length audio, a large `Max_Duration` relative to `Min_Duration` on encoder attention layers typically reflects the first-call compilation overhead for a new sequence length. The `SmartTTNNLinear` prefill path caches its `MatmulMultiCoreReuseMultiCastProgramConfig` per `seq_len`, so after the first call for a given length the overhead disappears.
