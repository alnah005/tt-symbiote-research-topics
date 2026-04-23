# Chapter 7 -- Critic Review (Pass 1)

## Issue 1: Incorrect code snippet for `_compute_tensor_signature` TorchTTNNTensor branch

**File:** `symbiote_traced_run.md`, Cache Key and Signature System section

**Chapter shows:**
```python
if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
    return (tuple(tensor.shape), tensor.dtype, tensor.layout)
```

**Actual source (`run_config.py`, lines 879-881):**
```python
if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
    t = tensor.ttnn_tensor
    return (tuple(t.shape), t.dtype, t.layout)
```

The real code extracts the underlying `ttnn.Tensor` into a local variable `t` and reads `t.shape`, `t.dtype`, `t.layout`. The chapter's snippet reads `tensor.shape`, `tensor.dtype`, `tensor.layout` -- i.e., from the `TorchTTNNTensor` wrapper, not the inner TTNN tensor. This is a material difference because `tensor.dtype` on a `TorchTTNNTensor` returns a `torch.dtype` (e.g., `torch.bfloat16`), whereas `t.dtype` on the underlying `ttnn.Tensor` returns a `ttnn.DataType` (e.g., `ttnn.bfloat16`). The cache key would hash differently. Fix by adding the intermediate variable to match the source.

---

No other factual, numerical, or implementation errors found. All other claims verified against source:
- `Tracer` class structure, two-phase capture, `_update_input`, `_tree_map`, and `release` are accurate.
- `PipelineTrace` dataclass fields for Flux1 match exactly.
- Per-submesh synchronization loop after capture matches.
- `TracedRun` three-phase lifecycle, `_TRACE_RUNNING` guard, class hierarchy, and hook dispatch logic are correct.
- `TTNNLayerStack` calling `layer.forward()` directly (not `layer()`) is correct.
- Run-mode registry has 8 entries as stated.
- `disable_trace` decorator implementation is accurate.
