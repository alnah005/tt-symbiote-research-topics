# Dispatch System

**Source:**
- `core/tensor.py` --- `TorchTTNNTensor`
- `core/dispatcher.py` --- public dispatch API
- `core/dispatchers/dispatcher_config.py` --- dispatcher registry
- `core/dispatchers/default_dispatcher.py` --- ATen-to-TTNN handler implementations

The dispatch system is TT-Symbiote's mechanism for intercepting PyTorch operations that happen *outside* of explicit `TTNNModule.forward()` calls --- residual additions, activation functions invoked by PyTorch code, tensor reshapes, and similar operations.

## TorchTTNNTensor: The PyTorch Tensor Subclass

`TorchTTNNTensor` subclasses `torch.Tensor` and acts as a bridge between PyTorch's tensor API and TTNN device tensors:

```python
class TorchTTNNTensor(torch.Tensor):
    elem: torch.Tensor
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        return TENSOR_RUN_IMPLEMENTATION.new_instance(cls, elem, *args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            cls, func, types, args, kwargs
        )
```

Key properties:

- **Dual representation**: A `TorchTTNNTensor` can hold a PyTorch tensor in `elem` (for shape/dtype metadata on the meta device), a TTNN device tensor in `ttnn_tensor`, or both. The `to_ttnn` property lazily converts `elem` to a TTNN tensor; `to_torch` converts back.
- **Operator overloads**: `__mul__`, `__add__`, `__sub__`, `__matmul__`, etc. delegate to `torch.mul`, `torch.add`, etc., which then hit `__torch_dispatch__`.
- **Distributed tensor config**: `_distributed_tensor_config` (a `DistributedTensorConfig`) tracks how the tensor is sharded across a mesh, and overrides the `shape` property to report the logical (unsharded) shape.

### The `__torch_dispatch__` Flow

When PyTorch executes an ATen op on a `TorchTTNNTensor`, the dispatch flow is:

```
1. PyTorch calls aten::mul.Tensor(TorchTTNNTensor, TorchTTNNTensor)
2. __torch_dispatch__ is invoked (via TENSOR_RUN_IMPLEMENTATION)
3. The implementation calls can_dispatch_to_ttnn("aten::mul.Tensor", args)
4. If True: dispatch_to_ttnn("aten::mul.Tensor", args, kwargs)
5. If False: fall back to PyTorch CPU execution (unwrap -> compute -> rewrap)
```

The `TENSOR_RUN_IMPLEMENTATION` is selected at import time via `get_tensor_run_implementation()` in `run_config.py`, allowing different backends (device, CPU, debug).

## The Dispatcher Registry

The dispatcher system is pluggable. `dispatcher_config.py` maintains a global registry:

```python
_DISPATCHER_REGISTRY: Dict[str, Any] = {}
```

Four dispatchers are auto-registered on import:

| Name | Module | Purpose |
|------|--------|---------|
| `DEFAULT` | `default_dispatcher` | Full ATen-to-TTNN mapping (~80 ops) |
| `DEBUG` | `debug_dispatcher` | Verbose logging for debugging |
| `CPU` | `cpu_dispatcher` | CPU fallback (default when no env var set) |
| `TENSOR_OPS` | `tensor_operations_dispatcher` | Tensor-specific operations |

Selection is controlled by the `TT_SYMBIOTE_DISPATCHER` environment variable or `set_dispatcher()`:

```python
def get_active_dispatcher():
    env_dispatcher = os.environ.get("TT_SYMBIOTE_DISPATCHER", None)
    if env_dispatcher is not None and env_dispatcher in _DISPATCHER_REGISTRY:
        return _DISPATCHER_REGISTRY[env_dispatcher]
    elif env_dispatcher is None and _current_dispatcher is None:
        return _DISPATCHER_REGISTRY["CPU"]  # default fallback
    ...
```

A custom dispatcher must implement two functions:
- `can_dispatch_to_ttnn(func_name, args, kwargs) -> bool`
- `dispatch_to_ttnn(func_name, args, kwargs) -> result`

## The Default Dispatcher: Handler-by-Handler Mapping

The `default_dispatcher.py` is the heart of TT-Symbiote's op coverage. It contains:

1. **A dispatch table** (`_get_func_to_ttnn_compatible()`) mapping ATen op names to handler functions
2. **~80 individual handler functions**, each translating one ATen op to TTNN calls
3. **A `can_dispatch_to_ttnn()` function** with op-specific validation logic
4. **Shared helper functions** for tensor preparation and cleanup

### The Dispatch Table

The `_get_func_to_ttnn_compatible()` function returns a dictionary mapping ATen op strings to handler functions:

```python
def _get_func_to_ttnn_compatible():
    return {
        "aten::view": handle_view,
        "aten::transpose.int": handle_transpose,
        "aten::mul.Tensor": handle_mul,
        "aten::sub.Tensor": handle_sub,
        "aten::div.Tensor": handle_div,
        "aten::slice.Tensor": handle_slice,
        "aten::add.Tensor": handle_add,
        "aten::bmm": handle_bmm,
        "aten::_softmax": handle_softmax,
        "aten::gelu": handle_gelu,
        "aten::relu": handle_relu,
        "aten::sigmoid": handle_sigmoid,
        "aten::embedding": handle_embedding,
        "aten::_scaled_dot_product_attention": handle_sdpa,
        # ... ~80 entries total
    }
```

The full table covers: arithmetic (add, sub, mul, div), activations (gelu, relu, sigmoid, tanh, silu), shape ops (view, reshape, permute, squeeze, unsqueeze, expand, slice, split, chunk, cat, stack, unbind, flatten, narrow), comparison (ge, gt, lt, eq), reduction (sum, mean, max, topk), memory (clone, contiguous, copy\_, zeros\_like, ones\_like, full\_like, empty\_like, new\_zeros), advanced (embedding, addmm, bmm/mm, sdpa, layer\_norm, mse\_loss, scatter, gather, index, masked\_fill, where, broadcast\_tensors, bernoulli, im2col, pixel\_unshuffle, constant\_pad\_nd), and type conversion (\_to\_copy, to.dtype).

### Anatomy of a Handler

Every handler follows the same structural pattern. Here is `handle_mul`:

```python
def handle_mul(func, args, kwargs):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    input_tensor1, input_tensor2, deallocate_a, deallocate_b, device = \
        _prepare_binary_inputs(args[0], args[1])

    ttnn_tensor1 = ensure_tile_layout(input_tensor1.to_ttnn)
    ttnn_tensor2 = ensure_tile_layout(input_tensor2.to_ttnn)

    res = TorchTTNNTensor(ttnn.multiply(ttnn_tensor1, ttnn_tensor2))
    _cleanup_tensors(
        (input_tensor1, deallocate_a), (input_tensor2, deallocate_b)
    )
    return res
```

The pattern is consistent:

1. **Import** `TorchTTNNTensor` (deferred to avoid circular imports)
2. **Prepare inputs**: Ensure both operands are `TorchTTNNTensor` with TTNN device tensors, track which temporaries to deallocate
3. **Ensure layout**: Convert to `TILE_LAYOUT` if needed
4. **Call TTNN op**: The single line that actually does work
5. **Wrap result**: Return as `TorchTTNNTensor`
6. **Cleanup**: Deallocate temporary tensors

### Shared Helpers

The helpers that appear in nearly every handler:

**`_prepare_tensor_input(tensor, device, ref_dtype)`** --- Wraps a plain value or torch tensor into `TorchTTNNTensor`, returning `(tensor, should_deallocate, device)`.

**`_prepare_binary_inputs(tensor1, tensor2, device)`** --- Prepares two operands for a binary op, handling device mismatches by moving both to the same device.

**`ensure_tile_layout(ttnn_tensor)`** --- Converts to `TILE_LAYOUT` if not already (from `core/utils.py`).

**`_cleanup_tensors(*pairs)`** --- Deallocates temporary TTNN tensors marked for cleanup.

### The `can_dispatch_to_ttnn()` Validation

Beyond checking whether a handler exists, `can_dispatch_to_ttnn()` performs op-specific validation:

- Verifies at least one arg is a `TorchTTNNTensor` with an allocated device buffer
- Checks dtype compatibility (e.g., `aten::sum` only supports float32, bfloat16, bfloat8_b, uint32)
- Validates argument shapes/types (e.g., `aten::slice.Tensor` requires int args)
- Applies environment-variable-controlled fallback thresholds (e.g., `TT_SYMBIOTE_UNBIND_FALLBACK_THRESHOLD`, `TT_SYMBIOTE_SPLIT_FALLBACK_THRESHOLD`, `TT_SYMBIOTE_IM2COL_MAX_DEVICE_NUMEL`)

This validation logic is ~200 lines of `if/elif` chains, one block per op that needs special handling.

## Pain Points for TT-Lang Integration

### Pain Point 1: Manual Handler Per New ATen Op

Every time a PyTorch model uses an ATen op not in the dispatch table, TT-Symbiote falls back to CPU. The fallback is logged:

```python
def _log_fallback_op(func_name):
    print(
        f"Found Operation {func_name} that if written in ttnn would be "
        "more efficient. Please map this function to an appropriate "
        "ttnn function."
    )
```

Adding a new handler means writing 10--60 lines of Python, following the same prepare/call/wrap/cleanup pattern. This is the **single largest scaling bottleneck** in TT-Symbiote.

**TT-Lang opportunity:** A code-generation system could produce handlers from declarative mappings like `aten::mul.Tensor -> ttnn.multiply [binary, tile_layout]`. The shared boilerplate (input preparation, layout enforcement, cleanup) could be generated automatically.

### Pain Point 2: Repeated Boilerplate Across Handlers

The deferred import of `TorchTTNNTensor`, the `_prepare_binary_inputs` / `_prepare_tensor_input` calls, `ensure_tile_layout`, and `_cleanup_tensors` pattern appear in almost every handler. Compare `handle_mul`, `handle_sub`, and `handle_div` --- they are structurally identical except for the TTNN function name.

**TT-Lang opportunity:** A higher-order function or decorator could encapsulate the binary-op pattern. More ambitiously, TT-Lang could define op categories (unary, binary, reduction) with automatic boilerplate.

### Pain Point 3: Complex Handlers Are Error-Prone

Some handlers are substantially more complex than the binary-op pattern. `handle_add` is ~40 lines because it must handle mixed `TorchTTNNTensor` / plain `torch.Tensor` inputs with special fallback logic for deallocated buffers. `handle_sdpa` must manage attention masks, scaling, and multiple input tensors. `handle_im2col` has environment-variable-controlled numel thresholds.

These complex handlers are where bugs concentrate. The `_ttnn_from_torchttnn_safe` helper exists specifically to work around buffer deallocation issues that surface in `handle_add`.

**TT-Lang opportunity:** Complex ops that combine multiple TTNN primitives (e.g., `all_reduce` decomposed into `reduce_scatter` + `all_gather` for trace compatibility) are prime candidates for TT-Lang fused kernels.

### Pain Point 4: Validation Logic Is Scattered

The `can_dispatch_to_ttnn()` function is a monolithic 200-line function with op-specific validation blocks. There is no schema or type system enforcing which ops accept which argument types. Validation failures produce print-based warnings rather than structured errors.

**TT-Lang opportunity:** Op signatures could be declared with typed argument specs, and validation could be generated automatically. Device capability checks could move from runtime to compile time.

### Pain Point 5: No Op Composition

The dispatch table is flat --- each ATen op maps to exactly one handler. There is no mechanism for composing ops (e.g., "fused add + relu") or expressing op patterns. When PyTorch models use op sequences that could be fused on device, TT-Symbiote executes them as separate dispatches.

**TT-Lang opportunity:** TT-Lang's kernel fusion capabilities could recognize dispatch patterns and replace multi-op sequences with fused device kernels, reducing dispatch overhead and memory traffic.

---

**Next:** [`module_catalog.md`](./module_catalog.md)
