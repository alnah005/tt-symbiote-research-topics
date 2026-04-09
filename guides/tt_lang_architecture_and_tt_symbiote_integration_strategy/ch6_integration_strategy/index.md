# Chapter 6 --- Integration Strategy: TT-Lang Kernels in TT-Symbiote

This chapter defines the concrete interface contract and code changes required to use TT-Lang compiled kernels as **drop-in replacements for TTNN ops** inside TT-Symbiote modules. Where earlier chapters examined the two codebases independently --- TT-Lang's compilation pipeline ([Chapter 2](../ch2_compilation_pipeline/index.md)) and TT-Symbiote's module architecture ([Chapter 5](../ch5_symbiote_architecture/index.md)) --- this chapter is where they meet.

## Integration Philosophy

The integration rests on a single architectural insight: **TT-Lang kernels compile to `CompiledTTNNKernel` objects that accept `ttnn.Tensor` inputs and produce `ttnn.Tensor` outputs.** This means they slot into any place where a TTNN op call currently sits in a TT-Symbiote `forward()` method.

The key constraints that make this work:

1. **TILE_LAYOUT required.** All tensor arguments must use `ttnn.TILE_LAYOUT`. TT-Symbiote modules already enforce this (every `forward()` starts with a layout check). TT-Lang validates at compilation time via `_compile_ttnn_kernel` (see `ttl_api.py` line 636).

2. **L1 or DRAM memory space.** Tensors must reside in device L1 or DRAM --- no host tensors. TT-Symbiote's `move_weights_to_device_impl` pipeline guarantees device placement before `forward()` runs.

3. **No mixed tensor types.** All arguments must be `ttnn.Tensor` (not a mix of torch and ttnn). This is enforced by `_compile_ttnn_kernel`'s validation check.

4. **Compilation caching is built in.** The `pykernel_gen` decorator maintains a per-kernel `cache: Dict[tuple, CompiledTTNNKernel]` keyed on tensor shapes, dtypes, memory spaces, and compiler options. Re-invocations with the same tensor profile skip recompilation entirely.

## What Changes, What Stays the Same

| Aspect | Current (TTNN ops) | Proposed (TT-Lang kernels) |
|--------|-------------------|---------------------------|
| Tensor type in `forward()` | `ttnn.Tensor` | `ttnn.Tensor` (unchanged) |
| Layout requirement | `TILE_LAYOUT` | `TILE_LAYOUT` (unchanged) |
| Weight lifecycle | `preprocess` -> `move_to_device` -> `forward` -> `deallocate` | Same lifecycle, weights passed as kernel args |
| Op invocation | `ttnn.linear(x, w)` | `self._compiled_kernel(x, w, out)` |
| Compilation | Pre-compiled in TTNN library | JIT on first call, cached thereafter |
| Device placement | `ttnn.to_device()` | `ttnn.to_device()` (unchanged) |
| `@deallocate_weights_after` | Compatible | Compatible (runs after `forward` returns) |

## Chapter Contents

- [`interface_contract.md`](./interface_contract.md) --- The `CompiledTTNNKernel.__call__` API: tensor requirements, grid resolution via `_resolve_grid`, and compilation caching via `_make_cache_key`.

- [`weight_pipeline_interaction.md`](./weight_pipeline_interaction.md) --- How TT-Lang JIT interacts with TT-Symbiote's `preprocess`/`move` pipeline. Weight tensors as kernel arguments. Lazy compilation. Mesh tensor handling.

- [`forward_method_changes.md`](./forward_method_changes.md) --- Before/after code examples showing the current TTNN op pattern vs. the proposed TT-Lang kernel pattern. Module-level kernel caching, `@deallocate_weights_after` compatibility, and device placement.

## Key Takeaways

1. **The integration boundary is `CompiledTTNNKernel.__call__`** --- a callable that takes N `ttnn.Tensor` arguments and dispatches them through `ttnn.generic_op`. TT-Symbiote modules need only swap a TTNN op call for this callable.

2. **TT-Symbiote's weight lifecycle is fully compatible** --- `preprocess_weights_impl` and `move_weights_to_device_impl` run before `forward()`, so by the time a TT-Lang kernel executes, all weight tensors are already on-device in `TILE_LAYOUT`.

3. **JIT compilation adds a one-time cost on the first forward pass.** The `_make_cache_key` mechanism ensures recompilation only happens when tensor shapes, dtypes, memory spaces, or compiler options change. For inference with fixed shapes, this means compile-once-run-forever.

4. **No changes to module registration, dispatch, or the `TorchTTNNTensor` subclass** are required. The integration is entirely within `forward()` method bodies.

5. **Output tensor allocation is the caller's responsibility.** Unlike TTNN ops that allocate outputs internally, `CompiledTTNNKernel.__call__` expects the output tensor to be pre-allocated and passed as the last argument. This is the most significant API difference.
