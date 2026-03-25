# Step 3: Validation and Benchmarking

## 1. Running the Existing Tests as a Baseline

Both test functions in `test_llama.py` require:
- A `device` fixture that opens a Wormhole device (N150 or N300).
- The `meta-llama/Llama-3.2-1B-Instruct` weights to be accessible via the HuggingFace cache or a local path.
- The `tt-metal` Python environment to be active with `ttnn` importable.

### Basic test (`test_llama`)

```bash
pytest models/experimental/tt_symbiote/tests/test_llama.py::test_llama -s
```

This test:
1. Loads the model in `torch.bfloat16`.
2. Applies the two-pass `LlamaMLP` wrapper and `TTNNLinear` replacement.
3. Preprocesses and moves all replaced weights to device.
4. Calls `model.generate(**inputs, max_new_tokens=128, use_cache=True)`.
5. Writes timing data to `llama_timing_stats.csv` and `llama_timing_stats_pivot.csv` in the working directory.
6. Prints the decoded output string.

### Intelligent test (`test_llama_intelligent`)

```bash
pytest models/experimental/tt_symbiote/tests/test_llama.py::test_llama_intelligent -s
```

This test uses `SmartTTNNLinear` instead of `TTNNLinear` for the `nn.Linear` entries, excludes `lm_head` from replacement, and writes its timing data to `llama_intelligent_timing_stats.csv` and `llama_intelligent_timing_stats_pivot.csv`.

### Running with DPL mode for accuracy validation

```bash
TT_SYMBIOTE_RUN_MODE=DPL pytest models/experimental/tt_symbiote/tests/test_llama.py::test_llama_intelligent -s
```

DPL mode significantly increases wall-clock time because every module runs both the Torch reference and the TTNN path and copies tensors between them. It is intended for validation, not benchmarking.

## 2. Reading the Timing CSV Files

`DispatchManager.save_stats_to_file` produces two CSV files from each test run.

### `llama_timing_stats.csv` (raw entries)

Each row is one timing event. Columns:

| Column | Description |
|---|---|
| `attrs` | Dict of extra metadata (currently empty `{}` for all entries) |
| `module_name` | Dot-separated path of the module that was active when the event was recorded, e.g. `model.model.layers.3.mlp.gate_proj` |
| `func_name` | The specific function timed. For module-level events this is the class name (e.g. `TTNNLinear`); for op-level events it is the TTNN or Torch op name (e.g. `TTNN::aten::mm`, `aten::mm`) |
| `duration` | Wall-clock seconds for this event |
| `backend` | One of `TTNN`, `Torch`, or `TorchModules` |

The `TorchModules` backend entry is recorded once per module invocation and covers the entire module call from input transformation through weight preprocessing and forward execution. It is the "outer" timer. The `TTNN` and `Torch` entries inside it are the "inner" per-op timers.

### `llama_timing_stats_pivot.csv` (aggregated)

The pivot table is indexed by `(func_name, module_name)` and has one column per backend, with `fill_value=0`. Additional columns:

| Column | Description |
|---|---|
| `TTNN` | Total seconds spent in TTNN dispatch for this `(func, module)` pair |
| `Torch` | Total seconds spent in Torch dispatch |
| `TorchModules` | Total seconds for full module-level invocations |
| `Total_Duration` | Sum across all backend columns |
| `Min_Duration` | Minimum single-invocation duration across backends |
| `Max_Duration` | Maximum single-invocation duration across backends |
| `Row_Count` | Number of raw timing rows merged into this pivot row |

### Identifying Fallback Layers

A layer is running on Torch (not TTNN) if:
- The `TorchModules` column has a nonzero value for a given `module_name`, **and**
- The `TTNN` column is zero (or absent) for the same `module_name`.

This indicates the module was called but no TTNN ops were dispatched — the module fell through to its `_fallback_torch_layer`. Common causes:

1. The module was not in the replacement dictionary (e.g., `lm_head` under `exclude_replacement`).
2. A `@trace_disabled` module triggered re-weight upload on every call without recording a trace, which is normal behavior but shows up with non-zero `TorchModules` time.
3. `TTNNSDPAAttention` fell back to `_matmul_attention` because `ttnn.transformer.scaled_dot_product_attention` raised a `RuntimeError` (indicated by the printed warning and `_sdpa_available = False`).

`save_stats_to_file` also prints the top-30 entries by `TorchModules` duration to stdout, aggregated by `func_name` (i.e., by class name such as `TTNNLinear`, `LlamaAttention`). Function classes that appear in this list with large values and zero `TTNN` time are the highest-priority candidates for replacement.

## 3. Confirming Per-Layer Accuracy in DPL Mode

Under `TT_SYMBIOTE_RUN_MODE=DPL`, each module's `DPLRun.module_run` calls `compare_fn_outputs` twice:
1. Comparing the input tensors (Torch copy vs. TTNN-path tensors) — this confirms the inputs reaching the TTNN path match the reference.
2. Comparing the output tensors (Torch reference output vs. TTNN output) — this is the PCC measurement of interest.

Checklist for a passing per-layer accuracy run:

- [ ] Each `TTNNRMSNorm` instance reports PCC >= 0.999. Norm layers are sensitive to tile layout and epsilon; lower PCC here usually indicates a dtype mismatch in the weight expansion (`unsqueeze(0).expand(32, -1)`).
- [ ] Each `TTNNLinear` / `SmartTTNNLinear` instance inside `LlamaMLP` reports PCC >= 0.99. If using `TTNNLinearLLama` (bfloat8 weights), expect PCC in the 0.99–0.999 range.
- [ ] `LlamaAttention` reports PCC >= 0.99. Lower values often trace to the query padding path (the zero-pad + slice logic for decode steps where `original_q_len < kv_len`).
- [ ] No `compare_fn_outputs` call reports `nan` or `inf` in either the Torch or TTNN output. These indicate a numerical failure, not just reduced precision.
- [ ] The final decoded text is identical (or nearly identical for stochastic sampling) between a DPL run and a standard run. If the text diverges significantly, use the timing CSV to find the layers with the lowest PCC and investigate those first.

## 4. Known Limitations and Open Issues

### Trace Capture: `@trace_disabled` Modules

The `trace_enabled` and `trace_disabled` decorators in `run_config.py` control which module classes are eligible for TTNN trace capture (where a sequence of TTNN ops is recorded once and replayed from device memory on subsequent calls, skipping host dispatch overhead).

`TTNNLinear` is marked `@trace_enabled`. Its subclasses `TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`, and `TTNNLinearLLamaIColShardedWRowSharded` are marked `@trace_disabled`. This means:

- `TTNNLinear` (used in `test_llama`) is eligible for trace capture.
- `TTNNLinearLLama` and `TTNNLinearLLamaBFloat16` (the bfloat8 / deallocate variants) are explicitly excluded from tracing. The reason is that `@deallocate_weights_after` deallocates the device weight tensor after each forward call. If a trace were captured, the weight deallocation would be replayed on subsequent calls, which would deallocate the weight before it could be used — making trace capture incompatible with per-call weight deallocation.

Practical consequence: if you substitute `TTNNLinearLLama` for `TTNNLinear` in the MLP projections to get bfloat8 weights, you lose trace capture for those layers. This increases host dispatch overhead on decode steps. The trade-off is reduced peak device memory (weights are freed between token generation steps) versus higher per-step dispatch latency.

`SmartTTNNLinear` does not carry either decorator and therefore inherits the `@trace_enabled` status of `TTNNLinear`. `SmartTTNNLinearLLama` adds `@deallocate_weights_after` but does not carry `@trace_disabled`; this is a potential issue if trace capture is enabled for it — the deallocation in the forward method would be recorded into the trace.

### LM Head Exclusion

If `lm_head` is not excluded and is replaced by `TTNNLinear` or `SmartTTNNLinear`, the following happens:

- `SmartTTNNLinear._get_prefill_pc` checks `if self.module_name == "lm_head"` and returns `None` for the program config. The `prefill_forward` call then passes `program_config=None` to `ttnn.linear`, which falls back to automatic program selection.
- The 128256-entry logit vector must be transferred back to the host for `torch.argmax` token selection. This round-trip is an overhead source that does not exist when the LM head runs on CPU.
- Memory: the `128256 x 2048` weight tensor is approximately 512 MB at bfloat16. With `@deallocate_weights_after` it is freed after each token, but allocating and freeing 512 MB per decode step adds fragmentation pressure.

### Memory: `@deallocate_weights_after` Effect on Peak Device Memory

The `@deallocate_weights_after` decorator, defined in `models/experimental/tt_symbiote/core/module.py`, calls `ttnn.deallocate` on `self.tt_weight` and `self.tt_bias` after `forward` returns. This keeps peak device memory proportional to the number of concurrently-live weight tensors rather than the total weight count.

The effect is most significant for large models where all layer weights cannot fit simultaneously on device. For LLaMA 3.2-1B on N150 (12 GB device DRAM), the full bfloat16 weight set is approximately 2 GB, well within device capacity. The deallocate pattern matters more for larger models or when device memory is shared with KV cache.

When `@deallocate_weights_after` is active, `move_weights_to_device` is called on every forward pass (it is a no-op if `tt_weight` is already on device, but `deallocate_weights_after` clears `tt_weight` after each call, so re-upload occurs on every token). This adds host-to-device transfer overhead per token. `TTNNLinear` without the decorator uploads weights once and reuses them across all decode steps.

### Attention Mask Handling

`LlamaAttention.forward` prints a warning and ignores `attention_mask` when it is not `None`:

```python
if attention_mask is not None:
    print("Warning: attention_mask is not None, but TTNN LlamaAttention does not support it yet.")
```

HuggingFace's `model.generate` passes an attention mask for padding in batch generation. With batch size 1 (as in `test_llama.py`) this is a non-issue, but batch sizes greater than 1 will silently drop the mask and produce incorrect outputs for padded positions.

## 5. What the N300 Distributed Path Would Add

N300 exposes two Wormhole chips connected by an Ethernet link, forming a `MeshDevice` with shape `(1, 2)`. The Symbiote infrastructure supports this through the `TTNNLinearIColShardedWRowSharded` and `TTNNLinearLLamaIColShardedWRowSharded` classes, both annotated `@run_on_devices(DeviceArch.T3K)`.

On N300 the MLP projections would change as follows:

| MLP projection | N150 class | N300 class |
|---|---|---|
| `gate_proj`, `up_proj` | `TTNNLinear` (or `TTNNLinearLLama`) | `TTNNLinearIColShardedWRowSharded` (or `TTNNLinearLLamaIColShardedWRowSharded`) |
| `down_proj` | `TTNNLinear` (or `TTNNLinearLLama`) | Column-sharded input, row-sharded weight with reduce-scatter |

`TTNNLinearIColShardedWRowSharded.forward` issues `ttnn.experimental.reduce_scatter_minimal_async` after the local matmul to sum the partial results across the two chips. This is the Symbiote analog of the `tt_all_reduce` call in `MLP.forward` in TT Transformers.

The `TTNNDistributedRMSNorm` class in `normalization.py` handles the distributed norm case: it calls `ttnn.rms_norm_pre_all_gather`, issues `ttnn.experimental.all_gather_async`, then calls `ttnn.rms_norm_post_all_gather`. This mirrors the split-norm pattern used in TT Transformers for multi-chip RMSNorm.

For attention on N300, `TTNNLinearIReplicatedWColSharded` handles the QKV projection (replicated input, column-sharded weight — each chip computes a subset of the Q/K/V heads), and `TTNNLinearIColShardedWRowSharded` handles the output projection with a reduce-scatter.

None of the N300 variants are exercised by `test_llama.py` or `test_llama_intelligent`. Enabling the distributed path requires:
1. Opening a `MeshDevice` with `num_devices=2`.
2. Replacing the linear replacement entries in `nn_to_ttnn` with the sharded classes.
3. Replacing `TTNNRMSNorm` with `TTNNDistributedRMSNorm` in `nn_to_ttnn`.
4. Updating `LlamaAttention.init_fused_parameters` and `init_parameters` to use the sharded linear variants for the attention projections.

---

**End of guide.** Return to [Guide Index](../index.md)
